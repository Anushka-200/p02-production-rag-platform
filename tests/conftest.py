import pytest

from app import tracing
from app.config import Settings


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on the developer's environment.

    Some libraries (litellm) load .env into os.environ when imported, which would otherwise leak
    real passwords and ports into unrelated tests.
    """
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


@pytest.fixture(autouse=True)
def memory_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory extraction calls a model after every chat turn; tests opt in explicitly."""
    monkeypatch.setenv("MEMORY_ENABLED", "false")


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never send traces to Langfuse, even if the developer's .env has real keys."""
    monkeypatch.setattr(tracing, "get_client", lambda: None)


@pytest.fixture(scope="session")
def _test_schema():  # type: ignore[no-untyped-def]
    """ONE throwaway schema per test run, with every migration applied once.

    Creating a schema and applying all migrations took several seconds per test (the suite grew
    to 37 minutes). Now it is done once; each test just empties the tables (see `db_schema`).
    Tests never touch real tables: the schema is dropped at the end of the run.
    """
    import uuid

    import psycopg

    from app.db.migrate import apply_migrations

    try:
        url = Settings().database_url("owner")
        conn = psycopg.connect(url, connect_timeout=3)
    except Exception as exc:  # no .env, or database not running (e.g. CI)
        pytest.skip(f"database not available: {type(exc).__name__}")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    try:
        apply_migrations(conn, schema=schema)
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = %s "
                "AND tablename <> 'schema_migrations'",
                (schema,),
            ).fetchall()
        ]
        # children before parents, so DELETE never trips a foreign key
        edges = conn.execute(
            "SELECT c.conrelid::regclass::text, c.confrelid::regclass::text FROM pg_constraint c "
            "JOIN pg_namespace n ON n.oid = c.connamespace WHERE c.contype = 'f' AND n.nspname = %s",
            (schema,),
        ).fetchall()
        children: dict[str, set[str]] = {t: set() for t in tables}
        for child, parent in edges:
            child, parent = child.split(".")[-1], parent.split(".")[-1]
            if child != parent and child in children and parent in children:
                children[parent].add(child)
        order: list[str] = []

        def visit(table: str) -> None:
            if table in order:
                return
            for child in children[table]:
                visit(child)
            order.append(table)  # a table is appended after all tables that reference it

        for table in tables:
            visit(table)
        yield url, schema, order
    finally:
        conn.autocommit = True
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.close()


@pytest.fixture
def db_schema(_test_schema):  # type: ignore[no-untyped-def]
    """A connection to the (empty) test schema. Each test starts with no rows in any table."""
    import psycopg

    url, schema, order = _test_schema
    conn = psycopg.connect(url, autocommit=True)
    try:
        conn.execute(f"SET search_path TO {schema}, public")
        # DELETE, not TRUNCATE: on this machine TRUNCATE costs ~6 s per test (file syncs), DELETE ~ms
        for table in order:
            conn.execute(f"DELETE FROM {schema}.{table}")
        yield conn
    finally:
        conn.close()
