import os
import uuid

import pytest
from litellm import exceptions as exc

from app.config import Settings
from app.retrieval import rerank as rr
from app.retrieval.fusion import Fused
from app.retrieval.rerank import CohereReranker, RerankError, get_reranker, rerank
from app.retrieval.search import Candidate


def fused(text: str, rank: int, blurb: str = "") -> Fused:
    c = Candidate(
        uuid.uuid4(),
        uuid.uuid4(),
        "d.md",
        "Doc > S",
        None,
        None,
        ["s"],
        text,
        blurb,
        None,
        rank,
        0.0,
    )
    return Fused(candidate=c, rrf_score=1 / (60 + rank), dense_rank=rank, rank=rank)


class KeywordReranker:
    """A fake cross-encoder: scores a passage by how many query words it contains."""

    name = "fake"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.seen = passages
        words = set(query.lower().split())
        return [float(len(words & set(p.lower().split()))) for p in passages]


def texts(items) -> list[str]:  # type: ignore[no-untyped-def]
    return [r.fused.candidate.text for r in items]


def test_the_reranker_can_promote_a_chunk_the_first_stage_ranked_low() -> None:
    pool = [
        fused("lunch menu today", 1),
        fused("sabbatical leave after five years", 2),
        fused("parking rules", 3),
    ]
    out = rerank(pool, "sabbatical leave", KeywordReranker())
    assert texts(out)[0] == "sabbatical leave after five years" and [r.rank for r in out] == [
        1,
        2,
        3,
    ]
    assert out[0].rerank_score == 2.0


def test_only_the_top_candidates_are_scored_and_only_top_n_are_kept() -> None:
    pool = [fused(f"chunk {i}", i) for i in range(1, 31)]
    fake = KeywordReranker()
    out = rerank(pool, "chunk 25", fake, candidates=10, top=3)
    assert len(fake.seen) == 10 and len(out) == 3  # chunk 25 was never a candidate: latency saved


def test_ties_keep_the_earlier_order_and_no_reranker_keeps_the_fused_order() -> None:
    pool = [fused("alpha", 1), fused("beta", 2), fused("gamma", 3)]
    assert texts(rerank(pool, "zzz", KeywordReranker())) == [
        "alpha",
        "beta",
        "gamma",
    ]  # all score 0
    plain = rerank(pool, "anything", None, top=2)
    assert texts(plain) == ["alpha", "beta"] and all(r.rerank_score is None for r in plain)
    assert rerank([], "q", KeywordReranker()) == []


def test_passages_include_the_blurb_and_are_cut_to_the_word_limit() -> None:
    fake = KeywordReranker()
    rerank(
        [fused("one two three four five six", 1, blurb="From the policy")], "q", fake, max_words=5
    )
    assert fake.seen == ["From the policy one two"]  # blurb first, then cut at 5 words


# ---- hosted adapter (fake provider, no network) -----------------------------------------------
class FakeCohere:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes, self.calls = list(outcomes), 0

    def __call__(self, **kwargs: object) -> object:
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_cohere_results_are_mapped_back_to_the_input_order() -> None:
    resp = type(
        "R",
        (),
        {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.4},
                {"index": 1, "relevance_score": 0.1},
            ]
        },
    )()
    scores = CohereReranker(rerank_fn=FakeCohere(resp)).score("q", ["a", "b", "c"])
    assert scores == [0.4, 0.1, 0.9]


def test_cohere_retries_temporary_errors_then_gives_up() -> None:
    limited = exc.RateLimitError("slow down", "cohere", "m")
    ok = type("R", (), {"results": [{"index": 0, "relevance_score": 1.0}]})()
    fake = FakeCohere(limited, ok)
    assert (
        CohereReranker(rerank_fn=fake, sleep_fn=lambda _: None).score("q", ["a"]) == [1.0]
        and fake.calls == 2
    )
    always = FakeCohere(*[limited] * 4)
    with pytest.raises(RerankError, match="failed after 4 attempts"):
        CohereReranker(rerank_fn=always, sleep_fn=lambda _: None).score("q", ["a"])
    with pytest.raises(RerankError, match="AuthenticationError"):
        CohereReranker(rerank_fn=FakeCohere(exc.AuthenticationError("bad", "c", "m"))).score(
            "q", ["a"]
        )


def test_the_provider_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_reranker(Settings(_env_file=None, reranker_provider="none")) is None  # type: ignore[call-arg]
    monkeypatch.setattr(rr, "LocalReranker", lambda model: ("local", model))
    got = get_reranker(Settings(_env_file=None, reranker_provider="local", reranker_model="m1"))  # type: ignore[call-arg]
    assert got == ("local", "m1")
    cohere = get_reranker(Settings(_env_file=None, reranker_provider="cohere"))  # type: ignore[call-arg]
    assert isinstance(cohere, CohereReranker)


# ---- the real local model (downloads ~80 MB once; skipped on CI) -------------------------------
def test_the_real_cross_encoder_prefers_the_passage_that_answers_the_question() -> None:
    if os.environ.get("CI"):
        pytest.skip("real model tests are skipped on CI")
    try:
        model = rr.LocalReranker()
    except Exception as e:
        pytest.skip(f"model unavailable: {type(e).__name__}")
    good = "Employees receive twenty vacation days per year, accrued monthly."
    bad = "The cafeteria on the third floor serves lunch from noon until two."
    scores = model.score("How many vacation days do employees get?", [bad, good])
    assert scores[1] > scores[0] + 1.0
