"""Row-level security and tenant context, against the real database as the real restricted role.

Setup data is written as the table owner (RLS does not apply to it); every assertion runs as
`rag_app`, the role the API uses, connected to the same temporary schema.
"""

import psycopg
import pytest

from app.config import Settings
from app.db.tenant import check_role_is_restricted, set_session_tenant, tenant_connection
from tests.test_search import World, vec


@pytest.fixture
def world(db_schema):  # type: ignore[no-untyped-def]
    w = World(db_schema)
    schema = db_schema.execute("SELECT current_schema()").fetchone()[0]
    db_schema.execute(f"GRANT USAGE ON SCHEMA {schema} TO rag_app")
    for table in (
        "documents", "chunks", "ingestion_jobs", "conversations", "messages", "memories",
        "widget_sessions", "widget_messages", "leads", "feedback", "site_usage",
    ):  # fmt: skip
        db_schema.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {schema}.{table} TO rag_app")
    db_schema.execute(f"GRANT SELECT ON {schema}.orgs, {schema}.users TO rag_app")
    w.schema = schema  # type: ignore[attr-defined]
    w.ann = w.person("acme", "ann")  # type: ignore[attr-defined]
    w.amy = w.person("acme", "amy")  # type: ignore[attr-defined]
    w.bob = w.person("globex", "bob")  # type: ignore[attr-defined]
    w.chunk(w.ann, "acme", "acme shared handbook", vec(1), visibility="org", uri="shared.md")
    w.chunk(w.ann, "acme", "ann private notes", vec(2), visibility="private", uri="priv.md")
    w.chunk(w.bob, "globex", "globex secret", vec(3), visibility="org", uri="g.md")
    return w


def app_connection(world) -> psycopg.Connection:  # type: ignore[no-untyped-def]
    url = Settings().database_url("app")
    return psycopg.connect(url, options=f"-c search_path={world.schema},public", autocommit=True)


def texts(conn) -> set[str]:  # type: ignore[no-untyped-def]
    return {r[0] for r in conn.execute("SELECT text FROM chunks").fetchall()}


def test_a_tenant_sees_only_its_own_rows_even_with_no_filter(world) -> None:  # type: ignore[no-untyped-def]
    conn = app_connection(world)
    set_session_tenant(conn, world.bob)
    assert texts(conn) == {"globex secret"}
    set_session_tenant(conn, world.amy)
    # amy is in acme: she sees the shared doc, but NOT ann's private one, and nothing of globex
    assert texts(conn) == {"acme shared handbook"}
    set_session_tenant(conn, world.ann)
    assert texts(conn) == {"acme shared handbook", "ann private notes"}


def test_no_tenant_set_means_no_rows_not_all_rows(world) -> None:  # type: ignore[no-untyped-def]
    conn = app_connection(world)
    assert texts(conn) == set()
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0


def test_writing_into_another_org_is_refused(world) -> None:  # type: ignore[no-untyped-def]
    conn = app_connection(world)
    set_session_tenant(conn, world.bob)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):  # WITH CHECK violation
        conn.execute(
            "INSERT INTO documents (org_id, owner_id, visibility, title, source_type, source_uri, "
            "content_hash) VALUES (%s, %s, 'org', 'T', 'md', 'x.md', 'h')",
            (world.ann.org_id, world.ann.user_id),
        )


def test_app_and_worker_roles_cannot_change_users_or_read_password_hashes(db_schema) -> None:  # type: ignore[no-untyped-def]
    """Checks the grants migration 0004 applied to the real tables (the test schema has none)."""
    if db_schema.execute("SELECT to_regclass('public.users')").fetchone()[0] is None:
        pytest.skip("migrations have not been applied to the dev database")
    for role in ("rag_app", "rag_worker"):

        def can(priv: str, obj: str = "public.users", role: str = role) -> bool:
            return db_schema.execute(
                "SELECT has_table_privilege(%s, %s, %s)", (role, obj, priv)
            ).fetchone()[0]

        assert not can("INSERT") and not can("UPDATE") and not can("DELETE")
        assert not can("INSERT", "public.orgs") and not can("UPDATE", "public.orgs")
        assert not db_schema.execute(
            "SELECT has_column_privilege(%s, 'public.users', 'password_hash', 'SELECT')", (role,)
        ).fetchone()[0]


# ---- 3.14: pooled connections never carry a tenant over ------------------------------------------
class PooledConnection:
    """One real connection handed out again and again, like a connection pool does."""

    def __init__(self, conn) -> None:  # type: ignore[no-untyped-def]
        self._conn = conn

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._conn, name)

    def __setattr__(self, name: str, value) -> None:  # type: ignore[no-untyped-def]
        if name == "_conn":
            object.__setattr__(self, name, value)
        else:
            setattr(self._conn, name, value)

    def close(self) -> None:  # returned to the pool, not closed
        pass


def test_a_reused_connection_never_keeps_the_previous_tenant(world) -> None:  # type: ignore[no-untyped-def]
    real = app_connection(world)
    pool = lambda: PooledConnection(real)  # noqa: E731

    with tenant_connection(pool, world.ann) as conn:
        assert texts(conn) == {"acme shared handbook", "ann private notes"}
    # back in the pool: nothing may be left behind
    assert real.execute("SELECT current_setting('app.org_id', true)").fetchone()[0] in (None, "")
    assert texts(real) == set()  # a request that forgot to set a tenant sees nothing

    with tenant_connection(pool, world.bob) as conn:
        assert texts(conn) == {"globex secret"}  # bob never sees ann's rows


def test_the_tenant_is_dropped_even_when_the_request_fails(world) -> None:  # type: ignore[no-untyped-def]
    real = app_connection(world)
    with pytest.raises(RuntimeError), tenant_connection(lambda: PooledConnection(real), world.ann):
        raise RuntimeError("request blew up")
    assert real.execute("SELECT current_setting('app.org_id', true)").fetchone()[0] in (None, "")


def test_the_startup_check_rejects_the_owner_and_accepts_the_app_role(world, db_schema) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(RuntimeError, match="bypasses row-level security"):
        check_role_is_restricted(db_schema)  # rag_owner owns the tables
    check_role_is_restricted(app_connection(world))


# ---- 3.5: SQL-level checks (the database alone, no Python filtering) ------------------------------
def test_updating_or_deleting_another_tenants_rows_touches_nothing(world) -> None:  # type: ignore[no-untyped-def]
    conn = app_connection(world)
    set_session_tenant(conn, world.bob)
    assert conn.execute("UPDATE documents SET title = 'hacked'").rowcount == 1  # only bob's own
    assert conn.execute("DELETE FROM chunks WHERE text LIKE 'acme%'").rowcount == 0
    set_session_tenant(conn, world.ann)
    assert "hacked" not in {r[0] for r in conn.execute("SELECT title FROM documents").fetchall()}
    assert len(texts(conn)) == 2  # ann's chunks are all still there


def test_a_forged_scope_cannot_beat_the_database(world) -> None:  # type: ignore[no-untyped-def]
    """Python code passes bob's scope to the search, but the connection says ann: RLS wins."""
    from app.retrieval.search import dense_search, keyword_search

    conn = app_connection(world)
    set_session_tenant(conn, world.ann)  # the verified login
    assert dense_search(conn, vec(3), world.bob, k=5) == []  # a bug asking for globex's rows
    assert keyword_search(conn, "globex secret", world.bob, k=5) == []


def test_sharing_a_document_flows_to_its_chunks_and_back(world) -> None:  # type: ignore[no-untyped-def]
    from app.documents import set_visibility

    conn = app_connection(world)
    private_doc = world.conn.execute(
        "SELECT id FROM documents WHERE source_uri LIKE 'priv.md%'"
    ).fetchone()[0]
    set_session_tenant(conn, world.ann)
    set_visibility(conn, world.ann, private_doc, "org")
    set_session_tenant(conn, world.amy)
    assert "ann private notes" in texts(conn)  # shared now
    set_session_tenant(conn, world.ann)
    set_visibility(conn, world.ann, private_doc, "private")
    set_session_tenant(conn, world.amy)
    assert "ann private notes" not in texts(conn)  # private again: the chunks followed at once


def test_a_colleague_cannot_change_someone_elses_document(world) -> None:  # type: ignore[no-untyped-def]
    from app.documents import DocumentError, set_visibility

    shared_doc = world.conn.execute(
        "SELECT id FROM documents WHERE source_uri LIKE 'shared.md%'"
    ).fetchone()[0]
    conn = app_connection(world)
    set_session_tenant(conn, world.amy)  # sees it (shared) but does not own it
    with pytest.raises(DocumentError):
        set_visibility(conn, world.amy, shared_doc, "private")
    set_session_tenant(conn, world.bob)  # another org: cannot even see it; same message
    with pytest.raises(DocumentError, match="not found"):
        set_visibility(conn, world.bob, shared_doc, "private")


def test_jobs_are_private_to_their_owner(world) -> None:  # type: ignore[no-untyped-def]
    world.conn.execute(
        "INSERT INTO ingestion_jobs (org_id, owner_id, source_uri, source_type, visibility, "
        "content_hash, raw_path, idempotency_key) VALUES (%s,%s,'a.md','md','org','h','/x','k1')",
        (world.ann.org_id, world.ann.user_id),
    )
    conn = app_connection(world)
    count = lambda: conn.execute("SELECT count(*) FROM ingestion_jobs").fetchone()[0]  # noqa: E731
    set_session_tenant(conn, world.ann)
    assert count() == 1
    set_session_tenant(conn, world.amy)  # same org, different user
    assert count() == 0
    set_session_tenant(conn, world.bob)
    assert count() == 0


# ---- chat history and memory are private to the USER (stricter than the org) --------------------
def test_conversations_messages_and_memories_are_invisible_to_colleagues_and_other_orgs(
    world,
) -> None:  # type: ignore[no-untyped-def]
    from app.chat_history import add_message, create_conversation
    from app.memory import list_memories, store_facts
    from tests.test_memory import SETTINGS, AxisEmbedder

    conn = app_connection(world)
    set_session_tenant(conn, world.ann)
    conv = create_conversation(conn, world.ann, "private")
    add_message(conn, world.ann, conv, "user", "my secret plan")
    store_facts(conn, world.ann, AxisEmbedder(), ["User likes green tea"], settings=SETTINGS)
    count = lambda t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]  # noqa: E731
    assert (count("conversations"), count("messages"), count("memories")) == (1, 1, 1)
    for other in (world.amy, world.bob):  # same org colleague, and another org
        set_session_tenant(conn, other)
        assert (count("conversations"), count("messages"), count("memories")) == (0, 0, 0)
        assert list_memories(conn, other) == []
    set_session_tenant(conn, world.amy)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):  # WITH CHECK: cannot write as ann
        conn.execute(
            "INSERT INTO memories (org_id, user_id, fact, embedding, embedding_model) "
            "VALUES (%s, %s, 'User x y z abc', %s::vector, 'm')",
            (world.ann.org_id, world.ann.user_id, "[" + ",".join(["0"] * 384) + "]"),
        )
