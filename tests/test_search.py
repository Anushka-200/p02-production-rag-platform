"""Dense and keyword search against the real database (temporary schema)."""

import uuid

import pytest

from app.embeddings import vector_literal
from app.retrieval.search import Scope, dense_search, keyword_query, keyword_search

DIM = 384


def vec(i: int, noise: float = 0.0, j: int = 0) -> list[float]:
    """A unit-ish vector along axis i, optionally leaning towards axis j (a bit farther away)."""
    v = [0.0] * DIM
    v[i] = 1.0
    if noise:
        v[j] = noise
    return v


class World:
    def __init__(self, conn) -> None:  # type: ignore[no-untyped-def]
        self.conn = conn
        self.orgs: dict[str, uuid.UUID] = {}
        self.users: dict[str, uuid.UUID] = {}
        self.docs: dict[tuple[str, str], uuid.UUID] = {}

    def person(self, org: str, name: str) -> Scope:
        if org not in self.orgs:
            self.orgs[org] = self.conn.execute(
                "INSERT INTO orgs (name) VALUES (%s) RETURNING id", (org,)
            ).fetchone()[0]
        key = f"{org}/{name}"
        if key not in self.users:
            self.users[key] = self.conn.execute(
                "INSERT INTO users (org_id, email) VALUES (%s, %s) RETURNING id",
                (self.orgs[org], f"{name}@x.com"),
            ).fetchone()[0]
        return Scope(self.orgs[org], self.users[key])

    def chunk(
        self,
        who: Scope,
        org: str,
        text: str,
        embedding=None,
        visibility="org",
        level=0,
        uri="a.md",
        blurb="",
        status="ready",
        name="u",
    ):  # type: ignore[no-untyped-def]
        key = (f"{org}/{who.user_id}", f"{uri}/{visibility}/{status}")
        if key not in self.docs:
            self.docs[key] = self.conn.execute(
                "INSERT INTO documents (org_id, owner_id, visibility, title, source_type, source_uri, content_hash, "
                "status, active_version) VALUES (%s,%s,%s,'T','md',%s,%s,%s,1) RETURNING id",
                (
                    who.org_id,
                    who.user_id,
                    visibility,
                    f"{uri}-{visibility}-{status}-{uuid.uuid4().hex[:6]}",
                    uuid.uuid4().hex,
                    status,
                ),
            ).fetchone()[0]
        doc = self.docs[key]
        n = self.conn.execute(
            "SELECT count(*) FROM chunks WHERE document_id = %s", (doc,)
        ).fetchone()[0]
        self.conn.execute(
            "INSERT INTO chunks (org_id, owner_id, visibility, document_id, version, level, chunk_index, section_path, "
            "text, context_blurb, content_hash, anchors, embedding, embedding_model) "
            "VALUES (%s,%s,%s,%s,1,%s,%s,'Doc > Sec',%s,%s,'h','{sec}',%s::vector,%s)",
            (who.org_id, who.user_id, visibility, doc, level, n, text, blurb,
             vector_literal(embedding) if embedding else None, "test" if embedding else None),
        )  # fmt: skip


@pytest.fixture
def w(db_schema):  # type: ignore[no-untyped-def]
    return World(db_schema)


# ---- dense -------------------------------------------------------------------------------------
def test_dense_returns_the_closest_chunk_first_with_its_distance(w) -> None:  # type: ignore[no-untyped-def]
    me = w.person("acme", "ann")
    w.chunk(me, "acme", "vacation policy", vec(1))
    w.chunk(me, "acme", "sick policy", vec(2))
    w.chunk(me, "acme", "travel policy", vec(3))
    got = dense_search(w.conn, vec(2), me, k=3)
    assert (
        [c.text for c in got][0] == "sick policy"
        and got[0].rank == 1
        and got[0].score == pytest.approx(0.0, abs=1e-6)
    )
    assert [c.rank for c in got] == [1, 2, 3] and got[1].score > got[0].score


def test_dense_never_returns_another_tenants_chunks_even_if_they_are_closer(w) -> None:  # type: ignore[no-untyped-def]
    ann, bob = w.person("acme", "ann"), w.person("globex", "bob")
    w.chunk(bob, "globex", "SECRET globex salary table", vec(5))  # exactly the query vector
    w.chunk(ann, "acme", "acme policy", vec(6))  # far from the query
    got = dense_search(w.conn, vec(5), ann, k=10)
    assert [c.text for c in got] == ["acme policy"]


def test_dense_respects_private_documents(w) -> None:  # type: ignore[no-untyped-def]
    ann, ben = w.person("acme", "ann"), w.person("acme", "ben")
    w.chunk(ann, "acme", "ann private note", vec(7), visibility="private")
    w.chunk(ann, "acme", "shared handbook", vec(8), visibility="org")
    assert {c.text for c in dense_search(w.conn, vec(7), ben, k=10)} == {
        "shared handbook"
    }  # ben: not private
    assert {c.text for c in dense_search(w.conn, vec(7), ann, k=10)} == {
        "ann private note",
        "shared handbook",
    }


def test_dense_skips_parents_chunks_without_vectors_and_unready_documents(w) -> None:  # type: ignore[no-untyped-def]
    me = w.person("acme", "ann")
    w.chunk(me, "acme", "a parent", None, level=1)
    w.chunk(me, "acme", "no vector yet", None)
    w.chunk(me, "acme", "half-ingested", vec(9), status="processing")
    w.chunk(me, "acme", "good chunk", vec(9))
    assert [c.text for c in dense_search(w.conn, vec(9), me, k=10)] == ["good chunk"]


def test_a_small_tenant_still_gets_k_results_when_a_big_tenant_is_closer(w) -> None:  # type: ignore[no-untyped-def]
    """Regression test for the filtered-HNSW problem (benchmark B-007): without the iterative scan
    the index returns its nearest neighbours across ALL tenants and the filter leaves almost nothing."""
    small, big = w.person("small", "sue"), w.person("big", "bo")
    for i in range(300):
        w.chunk(
            big, "big", f"big {i}", vec(10, 0.001 * (i + 1), 11)
        )  # 300 chunks nearer to the query
    for i in range(60):
        w.chunk(
            small, "small", f"small {i}", vec(10, 0.3 + 0.001 * i, 12)
        )  # 60 chunks, a bit farther
    # On 360 rows Postgres prefers "organisation index + sort" (an exact search) and the problem cannot
    # occur. Forbid sorting: then only the HNSW index can return rows ordered by distance.
    w.conn.execute("SET enable_seqscan = off")
    w.conn.execute("SET enable_sort = off")
    got = dense_search(w.conn, vec(10), small, k=50)
    assert len(got) == 50 and all(c.text.startswith("small") for c in got)


# ---- keyword -----------------------------------------------------------------------------------
def test_keyword_query_is_or_of_safe_words() -> None:
    assert (
        keyword_query("How many vacation days do I get?")
        == "how OR many OR vacation OR days OR do OR get"
    )
    assert keyword_query("") == "" and keyword_query("?! -- ,") == ""
    nasty = keyword_query('vacation OR -secret "quoted" & (x) | y; DROP TABLE')
    assert all(ch.isalnum() or ch in " _" for ch in nasty) and "-" not in nasty and "(" not in nasty


def test_keyword_search_matches_any_word_and_ranks_more_matches_higher(w) -> None:  # type: ignore[no-untyped-def]
    me = w.person("acme", "ann")
    w.chunk(me, "acme", "Employees receive twenty vacation days per year")
    w.chunk(me, "acme", "Vacation requests need approval")
    w.chunk(me, "acme", "The cafeteria serves lunch")
    got = keyword_search(w.conn, "How many vacation days per year?", me)
    assert [c.text for c in got][0].startswith("Employees receive") and len(
        got
    ) == 2  # lunch: no match
    assert got[0].score > got[1].score > 0


def test_keyword_search_finds_words_that_only_the_blurb_contains(w) -> None:  # type: ignore[no-untyped-def]
    me = w.person("acme", "ann")
    w.chunk(me, "acme", "The fee is five percent", blurb="From the Acme late payment terms")
    assert [c.text for c in keyword_search(w.conn, "acme payment terms", me)] == [
        "The fee is five percent"
    ]


def test_keyword_search_handles_stop_word_only_questions(w) -> None:  # type: ignore[no-untyped-def]
    me = w.person("acme", "ann")
    w.chunk(me, "acme", "something")
    assert keyword_search(w.conn, "what is the", me) == []  # nothing left after stop words
    assert keyword_search(w.conn, "", me) == []


def test_keyword_search_is_scoped_like_dense(w) -> None:  # type: ignore[no-untyped-def]
    ann, bob, ben = w.person("acme", "ann"), w.person("globex", "bob"), w.person("acme", "ben")
    w.chunk(bob, "globex", "globex vacation secret")
    w.chunk(ann, "acme", "ann private vacation", visibility="private")
    w.chunk(ann, "acme", "shared vacation policy")
    w.chunk(ann, "acme", "parent vacation text", level=1)
    assert [c.text for c in keyword_search(w.conn, "vacation", ben)] == ["shared vacation policy"]
