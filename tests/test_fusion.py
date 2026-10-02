import uuid

import pytest

from app.config import Settings
from app.retrieval.fusion import fuse
from app.retrieval.retrieve import retrieve
from app.retrieval.search import Candidate, Scope
from tests.test_search import World, vec


def cand(name: str, rank: int, score: float = 0.0) -> Candidate:
    return Candidate(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, name), document_id=uuid.uuid4(), source_uri=f"{name}.md",
        section_path=name, page_start=None, page_end=None, anchors=[name], text=name, context_blurb="",
        parent_id=None, rank=rank, score=score,
    )  # fmt: skip


def names(fused) -> list[str]:  # type: ignore[no-untyped-def]
    return [f.candidate.text for f in fused]


# ---- the worked example from the learning step ---------------------------------------------------
def test_chunks_both_methods_like_beat_a_chunk_only_one_method_loves() -> None:
    dense = [cand("A", 1), cand("B", 2), cand("C", 3)]
    keyword = [cand("B", 1), cand("C", 2), cand("D", 3)]
    out = fuse(dense, keyword, k=60)
    assert names(out) == ["B", "C", "A", "D"]
    scores = {f.candidate.text: f.rrf_score for f in out}
    assert scores["A"] == pytest.approx(1 / 61)  # dense only, rank 1
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)  # dense 2 + keyword 1 = 0.03252
    assert scores["C"] == pytest.approx(1 / 63 + 1 / 62)
    assert scores["D"] == pytest.approx(1 / 63)  # keyword only, rank 3
    assert [f.rank for f in out] == [1, 2, 3, 4]


def test_the_quiz_question_chunk_in_both_lists_wins() -> None:
    # P: rank 4 dense and rank 3 keyword. Q: rank 1 in dense only.
    out = fuse(
        [cand("Q", 1), cand("x", 2), cand("y", 3), cand("P", 4)],
        [cand("a", 1), cand("b", 2), cand("P", 3)],
    )
    assert names(out)[0] == "P"
    assert next(f for f in out if f.candidate.text == "P").rrf_score == pytest.approx(
        1 / 64 + 1 / 63
    )


def test_each_result_keeps_what_each_method_said_about_it() -> None:
    out = fuse([cand("A", 1, 0.12), cand("B", 2, 0.30)], [cand("B", 1, 0.41)])
    b = next(f for f in out if f.candidate.text == "B")
    assert (b.dense_rank, b.dense_distance, b.keyword_rank, b.keyword_score) == (2, 0.30, 1, 0.41)
    a = next(f for f in out if f.candidate.text == "A")
    assert (a.keyword_rank, a.keyword_score) == (None, None)  # not found by keyword search


def test_weights_can_favour_one_method() -> None:
    dense, keyword = [cand("D1", 1)], [cand("K1", 1)]
    assert names(fuse(dense, keyword, dense_weight=1.0, keyword_weight=2.0))[0] == "K1"
    assert names(fuse(dense, keyword, dense_weight=2.0, keyword_weight=1.0))[0] == "D1"


def test_k_controls_how_much_the_top_rank_matters() -> None:
    dense = [cand("top", 1), cand("mid", 2)]
    keyword = [cand("x", 2), cand("mid", 3)]
    assert names(fuse(dense, keyword, k=0))[0] == "top"  # k=0: 1/1 beats 1/2 + 1/3
    assert names(fuse(dense, keyword, k=60))[0] == "mid"  # k=60: 1/62 + 1/63 beats 1/61


def test_empty_lists_and_ties_are_handled_and_deterministic() -> None:
    assert fuse([], []) == []
    only = fuse([cand("A", 1), cand("B", 2)], [])
    assert names(only) == ["A", "B"]
    tie_a = fuse([cand("A", 1)], [cand("B", 1)])
    tie_b = fuse([cand("A", 1)], [cand("B", 1)])
    assert names(tie_a) == names(tie_b)  # equal scores always come out in the same order


# ---- retrieve(): the modes, against the real database ------------------------------------------
class FakeEmbedder:
    model, dimension = "fake", 384

    def __init__(self, mapping: dict[str, int]) -> None:
        self.mapping = mapping

    def embed_query(self, text: str) -> list[float]:
        return vec(self.mapping.get(text, 100))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


@pytest.fixture
def world(db_schema):  # type: ignore[no-untyped-def]
    w = World(db_schema)
    me = w.person("acme", "ann")
    w.chunk(me, "acme", "Staff may take paid time off", vec(20))  # meaning-match: no shared words
    w.chunk(
        me, "acme", "Vacation requests need manager approval", vec(21)
    )  # word-match, other topic
    w.chunk(me, "acme", "The cafeteria serves lunch", vec(22))
    return w, me


S = Settings(_env_file=None, retrieval_candidates=10)  # type: ignore[call-arg]
QUESTION = "How many vacation days do I get?"


def test_dense_mode_finds_the_meaning_match_keyword_mode_the_word_match(world) -> None:  # type: ignore[no-untyped-def]
    w, me = world
    embedder = FakeEmbedder({QUESTION: 20})
    dense = retrieve(w.conn, embedder, QUESTION, me, settings=S, mode="dense")
    keyword = retrieve(w.conn, embedder, QUESTION, me, settings=S, mode="keyword")
    assert names(dense.items)[0] == "Staff may take paid time off"
    assert names(keyword.items) == ["Vacation requests need manager approval"]
    assert all(f.keyword_rank is None for f in dense.items)  # single modes keep the same shape


def test_hybrid_combines_both_and_reports_stage_timings(world) -> None:  # type: ignore[no-untyped-def]
    w, me = world
    got = retrieve(w.conn, FakeEmbedder({QUESTION: 20}), QUESTION, me, settings=S, mode="hybrid")
    assert {"Staff may take paid time off", "Vacation requests need manager approval"} <= set(
        names(got.items)
    )
    assert set(got.timings_ms) == {"embed_ms", "dense_ms", "keyword_ms", "fuse_ms"}
    assert got.keywords == "vacation"  # the informative words only (stop and unknown words dropped)
    both = [f for f in got.items if f.dense_rank and f.keyword_rank]
    assert both and all(f.rank >= 1 for f in got.items)


def test_mode_comes_from_settings_unless_overridden_and_top_truncates(world) -> None:  # type: ignore[no-untyped-def]
    w, me = world
    embedder = FakeEmbedder({QUESTION: 20})
    from_settings = Settings(_env_file=None, retrieval_mode="keyword", retrieval_candidates=10)  # type: ignore[call-arg]
    assert retrieve(w.conn, embedder, QUESTION, me, settings=from_settings).mode == "keyword"
    assert len(retrieve(w.conn, embedder, QUESTION, me, settings=S, mode="dense", top=2).items) == 2


def test_retrieval_is_scoped_to_the_asker(world) -> None:  # type: ignore[no-untyped-def]
    w, me = world
    other = w.person("globex", "bob")
    w.chunk(other, "globex", "Globex vacation secret", vec(20))
    got = retrieve(w.conn, FakeEmbedder({QUESTION: 20}), QUESTION, me, settings=S, mode="hybrid")
    assert "Globex vacation secret" not in names(got.items)
    assert Scope(me.org_id, me.user_id) == me
