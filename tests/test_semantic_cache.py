"""The semantic cache: a hit must never leak, never be stale, and never replace a fresh answer wrongly."""

import uuid
from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.retrieval.context import Source
from app.retrieval.search import Scope
from app.semantic_cache import lookup, store
from tests.test_api import FakeStream, SharedConnection, chat
from tests.test_rls import app_connection, set_session_tenant, world  # noqa: F401
from tests.test_search import World, vec

ANSWER = {"text": "Twenty days [S1].", "citations": [], "sources": [], "model": "m"}
KW = {"threshold": 0.9, "ttl_hours": 24.0}


def source(doc_id: uuid.UUID) -> Source:
    return Source("S1", uuid.uuid4(), doc_id, "a.md", "A", None, [], "t", 1)


@pytest.fixture
def w(db_schema):  # type: ignore[no-untyped-def]
    return World(db_schema)


def doc_of(w: World, who: Scope, visibility: str, uri: str = "a.md") -> uuid.UUID:
    w.chunk(who, "acme", "text", vec(1), visibility=visibility, uri=uri)
    return w.conn.execute(
        "SELECT id FROM documents WHERE owner_id = %s AND visibility = %s AND source_uri LIKE %s",
        (who.user_id, visibility, f"{uri}-{visibility}%"),
    ).fetchone()[0]


def save(w: World, who: Scope, doc: uuid.UUID, v: list[float] | None = None, **kw) -> bool:  # type: ignore[no-untyped-def]
    args = {"ttl_hours": 24.0, "max_per_org": 100, **kw}
    return store(w.conn, who, "How many vacation days?", v or vec(5), ANSWER, [source(doc)], **args)


# ---- hit and miss -----------------------------------------------------------------------------
def test_the_same_meaning_hits_and_a_different_meaning_misses(w: World) -> None:
    ann = w.person("acme", "ann")
    assert save(w, ann, doc_of(w, ann, "org"))
    assert lookup(w.conn, ann, vec(5), **KW).answer["text"] == "Twenty days [S1]."  # type: ignore[union-attr]
    assert lookup(w.conn, ann, vec(6), **KW) is None  # orthogonal: another question
    close, far = vec(5, 0.2, 6), vec(5, 0.8, 6)  # cosine 0.98 and 0.78
    assert lookup(w.conn, ann, close, **KW) is not None
    assert lookup(w.conn, ann, far, **KW) is None  # below the 0.9 threshold


def test_the_threshold_is_respected_exactly(w: World) -> None:
    ann = w.person("acme", "ann")
    save(w, ann, doc_of(w, ann, "org"))
    probe = vec(5, 0.5, 6)  # cosine about 0.894
    assert lookup(w.conn, ann, probe, threshold=0.95, ttl_hours=24) is None
    assert lookup(w.conn, ann, probe, threshold=0.85, ttl_hours=24) is not None


# ---- who may read an entry -------------------------------------------------------------------
def test_an_answer_built_from_a_private_document_is_never_shown_to_a_colleague(w: World) -> None:
    ann, amy = w.person("acme", "ann"), w.person("acme", "amy")
    save(w, ann, doc_of(w, ann, "private"))
    assert lookup(w.conn, ann, vec(5), **KW) is not None
    assert lookup(w.conn, amy, vec(5), **KW) is None


def test_an_answer_from_shared_documents_is_reusable_by_colleagues(w: World) -> None:
    ann, amy = w.person("acme", "ann"), w.person("acme", "amy")
    save(w, ann, doc_of(w, ann, "org"))
    assert lookup(w.conn, amy, vec(5), **KW) is not None


def test_one_private_source_among_shared_ones_makes_the_whole_entry_private(w: World) -> None:
    ann, amy = w.person("acme", "ann"), w.person("acme", "amy")
    shared, secret = doc_of(w, ann, "org"), doc_of(w, ann, "private", uri="n.md")
    store(
        w.conn,
        ann,
        "q",
        vec(5),
        ANSWER,
        [source(shared), source(secret)],
        ttl_hours=24,
        max_per_org=9,
    )
    assert lookup(w.conn, amy, vec(5), **KW) is None


def test_another_tenant_never_gets_a_hit_even_for_the_identical_question(w: World) -> None:
    ann, bob = w.person("acme", "ann"), w.person("globex", "bob")
    save(w, ann, doc_of(w, ann, "org"))
    assert lookup(w.conn, bob, vec(5), **KW) is None


def test_anonymous_visitors_neither_read_nor_write_the_cache(w: World) -> None:
    ann = w.person("acme", "ann")
    anon = Scope(ann.org_id, None)
    doc = doc_of(w, ann, "org")
    assert save(w, anon, doc) is False
    save(w, ann, doc)
    assert lookup(w.conn, anon, vec(5), **KW) is None


# ---- staleness ----------------------------------------------------------------------------------
def test_a_re_indexed_document_makes_its_cached_answers_stale(w: World) -> None:
    ann = w.person("acme", "ann")
    doc = doc_of(w, ann, "org")
    save(w, ann, doc)
    w.conn.execute("UPDATE documents SET active_version = 2 WHERE id = %s", (doc,))
    assert lookup(w.conn, ann, vec(5), **KW) is None


def test_a_deleted_document_makes_its_cached_answers_stale(w: World) -> None:
    ann = w.person("acme", "ann")
    doc = doc_of(w, ann, "org")
    save(w, ann, doc)
    w.conn.execute("DELETE FROM documents WHERE id = %s", (doc,))
    assert lookup(w.conn, ann, vec(5), **KW) is None


def test_a_document_that_became_private_is_no_longer_served_to_colleagues(w: World) -> None:
    ann, amy = w.person("acme", "ann"), w.person("acme", "amy")
    doc = doc_of(w, ann, "org")
    save(w, ann, doc)
    assert lookup(w.conn, amy, vec(5), **KW) is not None
    w.conn.execute("UPDATE documents SET visibility = 'private' WHERE id = %s", (doc,))
    assert (
        lookup(w.conn, amy, vec(5), **KW) is None
    )  # revalidated at lookup, whatever `shareable` says
    assert lookup(w.conn, ann, vec(5), **KW) is not None  # the owner may still read it


def test_an_expired_entry_is_ignored(w: World) -> None:
    ann = w.person("acme", "ann")
    save(w, ann, doc_of(w, ann, "org"))
    w.conn.execute("UPDATE semantic_cache SET created_at = now() - interval '2 days'")
    assert lookup(w.conn, ann, vec(5), **KW) is None


# ---- storing ------------------------------------------------------------------------------------
def test_the_cache_is_capped_per_org_keeping_the_newest(w: World) -> None:
    ann = w.person("acme", "ann")
    doc = doc_of(w, ann, "org")
    for i in range(4):
        store(w.conn, ann, f"q{i}", vec(10 + i), ANSWER, [source(doc)], ttl_hours=24, max_per_org=2)
    left = {r[0] for r in w.conn.execute("SELECT question FROM semantic_cache").fetchall()}
    assert left == {"q2", "q3"}


def test_nothing_is_stored_without_sources(w: World) -> None:
    ann = w.person("acme", "ann")
    assert store(w.conn, ann, "q", vec(5), ANSWER, [], ttl_hours=24, max_per_org=9) is False


# ---- row-level security is a second lock -----------------------------------------------------
def test_the_policies_hide_entries(world) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
    doc = world.conn.execute(
        "SELECT id FROM documents WHERE source_uri LIKE 'shared.md%'"
    ).fetchone()[0]
    priv = world.conn.execute(
        "SELECT id FROM documents WHERE source_uri LIKE 'priv.md%'"
    ).fetchone()[0]
    store(
        world.conn,
        world.ann,
        "shared q",
        vec(5),
        ANSWER,
        [source(doc)],
        ttl_hours=24,
        max_per_org=9,
    )
    store(
        world.conn,
        world.ann,
        "private q",
        vec(6),
        ANSWER,
        [source(priv)],
        ttl_hours=24,
        max_per_org=9,
    )
    conn = app_connection(world)

    def questions(who: Scope) -> set[str]:
        set_session_tenant(conn, who)
        return {r[0] for r in conn.execute("SELECT question FROM semantic_cache").fetchall()}

    assert questions(world.ann) == {"shared q", "private q"}
    assert questions(world.amy) == {"shared q"}  # same org, not the owner: only the shareable one
    assert questions(world.bob) == set()  # another org: nothing
    set_session_tenant(conn, world.amy)
    with pytest.raises(
        psycopg.errors.InsufficientPrivilege
    ):  # cannot plant an entry for someone else
        conn.execute(
            "INSERT INTO semantic_cache (org_id, user_id, question, embedding, answer, doc_refs) "
            "VALUES (%s, %s, 'x', %s::vector, '{}', '[]')",
            (world.ann.org_id, world.ann.user_id, "[" + ",".join(["0"] * 384) + "]"),
        )


# ---- through the HTTP chat endpoint -------------------------------------------------------------
class TwoTopicEmbedder:
    model, dimension = "fake", 384

    def embed_query(self, text: str) -> list[float]:
        return vec(30) if "vacation" in text.lower() else vec(40)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec(30) for _ in texts]


@pytest.fixture
def cached_api(db_schema):  # type: ignore[no-untyped-def]
    w = World(db_schema)
    ann = w.person("acme", "ann")
    w.person("acme", "amy")
    w.chunk(ann, "acme", "Employees receive twenty vacation days", vec(30), uri="acme/leave.md")
    calls: list[object] = []

    def stream_fn(messages, **kw):  # type: ignore[no-untyped-def]
        calls.append(messages)
        return FakeStream(["Twenty days [S1]."])

    s = Settings(
        _env_file=None,
        environment="dev",
        reranker_provider="none",
        retrieval_candidates=10,
        semantic_cache=True,
        semantic_cache_threshold=0.9,
    )  # type: ignore[call-arg]
    app = create_app(
        s, connect=lambda: SharedConnection(w.conn), embedder=TwoTopicEmbedder(), reranker=None,
        stream_fn=stream_fn, complete_fn=lambda *a, **k: pytest.fail("no JSON fallback expected"),
    )  # fmt: skip
    return SimpleNamespace(client=TestClient(app), calls=calls, w=w)


def test_a_repeated_question_is_served_from_the_cache_without_calling_the_model(cached_api) -> None:  # type: ignore[no-untyped-def]
    q = "How many vacation days do I get?"
    _, _, first = chat(cached_api.client, question=q)
    assert len(cached_api.calls) == 1 and dict(first)["done"]["model"] != "cache"
    _, _, second = chat(cached_api.client, question=q)
    done = dict(second)["done"]
    assert len(cached_api.calls) == 1  # no second model call
    assert (
        done["model"] == "cache" and done["cost_usd"] == 0 and done["text"] == "Twenty days [S1]."
    )
    assert [n for n, _ in second][0] == "sources" and done["citations"][0]["document"]


def test_a_different_question_is_not_served_from_the_cache(cached_api) -> None:  # type: ignore[no-untyped-def]
    chat(cached_api.client, question="How many vacation days do I get?")
    chat(cached_api.client, question="What is the parking policy?")
    assert len(cached_api.calls) == 2


def test_a_colleague_gets_the_shared_answer_but_a_turn_inside_a_conversation_never_uses_the_cache(
    cached_api,
) -> None:  # type: ignore[no-untyped-def]
    q = "How many vacation days do I get?"
    chat(cached_api.client, question=q)
    _, _, amy = chat(cached_api.client, who="amy@x.com", question=q)
    assert dict(amy)["done"]["model"] == "cache"
    cached_api.calls.clear()
    # a conversation turn with history must search afresh: the cache ignores follow-ups entirely
    _, _, a = chat(cached_api.client, question=q)
    cid = dict(a)["done"]["conversation_id"]
    with cached_api.client.stream(
        "POST",
        "/chat",
        json={"question": q, "conversation_id": cid},
        headers={"X-Dev-User": "ann@x.com"},
    ) as r:
        lines = list(r.iter_lines())
        assert r.status_code == 200 and "event: error" not in lines, lines
    assert len(cached_api.calls) == 1


def test_a_refusal_is_never_cached(db_schema) -> None:  # type: ignore[no-untyped-def]
    from app.answer import Answer
    from app.rag import _save_to_cache
    from app.retrieval.context import Context

    saved: list[object] = []
    ctx = Context("", [source(uuid.uuid4())], 1)
    scope = Scope(uuid.uuid4(), uuid.uuid4())
    refused = Answer("no", True, "unsupported")
    _save_to_cache(lambda *a: saved.append(a), vec(5), scope, "q", refused, ctx)
    assert saved == []
