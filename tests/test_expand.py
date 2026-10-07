"""Retrieval with transforms against the real database (fake embedder and fake model)."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.retrieval.expand import retrieve_expanded
from app.retrieval.fusion import Fused, merge_fused
from app.retrieval.retrieve import retrieve
from app.retrieval.search import Candidate
from tests.test_search import World, vec

HISTORY = [
    {"role": "user", "content": "What training does the company offer?"},
    {"role": "assistant", "content": "It offers Team Workshops."},
]
FOLLOW_UP = "How many of them are there altogether?"
STANDALONE = "How many topics do team workshops cover?"


class TextEmbedder:
    """A word-driven fake: the vector depends on which topic words the text contains."""

    model, dimension = "fake", 384
    AXES = {"workshop": 1, "refund": 2, "money back": 2, "holiday": 3, "vacation": 3}

    def embed_query(self, text: str) -> list[float]:
        low = text.lower()
        for word, axis in self.AXES.items():
            if word in low:
                return vec(axis)
        return vec(200, 0.3, 3)  # unknown words: lean towards the (irrelevant) vacation chunk


def model(replies: dict[str, Any], calls: list | None = None):  # type: ignore[no-untyped-def,type-arg]
    """A fake LLM: picks the reply by which task its system prompt describes."""

    def complete_fn(messages: list[dict[str, Any]], **kw: Any) -> Any:
        if calls is not None:
            calls.append(kw.get("name"))
        system = messages[0]["content"]
        key = (
            "rewrite"
            if "rewrite the user" in system.lower()
            else "hyde"
            if "passage" in system.lower()
            else "expand"
        )
        reply = replies[key]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            text=reply if isinstance(reply, str) else json.dumps(reply), cost_usd=0.001
        )

    return complete_fn


@pytest.fixture
def world(db_schema) -> World:  # type: ignore[no-untyped-def]
    w = World(db_schema)
    ann = w.person("acme", "ann")
    w.chunk(ann, "acme", "Team workshops cover twelve topics.", vec(1), uri="workshops.md")
    w.chunk(ann, "acme", "Refund requests are accepted within 30 days.", vec(2), uri="refunds.md")
    w.chunk(ann, "acme", "Vacation is twenty-five days.", vec(3), uri="vacation.md")
    bob = w.person("globex", "bob")
    w.chunk(bob, "globex", "GLOBEX workshop topics are secret.", vec(1), uri="globex.md")
    w.ann = ann  # type: ignore[attr-defined]
    return w


def settings(**kw: Any) -> Settings:
    kw.setdefault("retrieval_candidates", 10)
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def top_text(r) -> str:  # type: ignore[no-untyped-def]
    return r.items[0].candidate.text if r.items else ""


def test_with_every_switch_off_it_is_exactly_plain_retrieval_and_calls_no_model(
    world: World,
) -> None:
    calls: list = []  # type: ignore[type-arg]
    plain = retrieve(world.conn, TextEmbedder(), FOLLOW_UP, world.ann, settings=settings())
    out = retrieve_expanded(
        world.conn,
        TextEmbedder(),
        FOLLOW_UP,
        world.ann,
        history=HISTORY,
        settings=settings(),
        complete_fn=model({}, calls),
    )
    assert [f.candidate.chunk_id for f in out.items] == [f.candidate.chunk_id for f in plain.items]
    assert calls == [] and out.question == FOLLOW_UP and out.transform_cost_usd == 0.0


def test_rewriting_turns_a_hopeless_follow_up_into_a_search_that_finds_the_answer(
    world: World,
) -> None:
    s = settings(query_rewrite=True)
    before = retrieve(world.conn, TextEmbedder(), FOLLOW_UP, world.ann, settings=settings())
    assert "Team workshops" not in top_text(before)  # "these sessions" matches nothing
    out = retrieve_expanded(
        world.conn,
        TextEmbedder(),
        FOLLOW_UP,
        world.ann,
        history=HISTORY,
        settings=s,
        complete_fn=model({"rewrite": STANDALONE}),
    )
    assert top_text(out).startswith("Team workshops") and out.question == STANDALONE
    assert out.transform_cost_usd == 0.001 and out.queries == [STANDALONE]


def test_rewriting_without_history_does_nothing(world: World) -> None:
    calls: list = []  # type: ignore[type-arg]
    out = retrieve_expanded(
        world.conn,
        TextEmbedder(),
        "refund rules?",
        world.ann,
        history=[],
        settings=settings(query_rewrite=True),
        complete_fn=model({"rewrite": "x"}, calls),
    )
    assert calls == [] and out.question == "refund rules?"


def test_a_failing_model_never_breaks_retrieval(world: World) -> None:
    from app.llm import LLMError

    out = retrieve_expanded(
        world.conn, TextEmbedder(), "refund rules?", world.ann, history=HISTORY,
        settings=settings(query_rewrite=True, multi_query=2, hyde=True),
        complete_fn=model({"rewrite": LLMError("x"), "expand": LLMError("x"), "hyde": LLMError("x")}),
    )  # fmt: skip
    assert top_text(out).startswith("Refund")  # the original question still works


def test_multi_query_finds_a_place_the_original_wording_missed(world: World) -> None:
    question = "can I get my cash returned?"  # no topic word the fake embedder knows
    alone = retrieve(
        world.conn, TextEmbedder(), question, world.ann, settings=settings(retrieval_mode="dense")
    )
    assert top_text(alone).startswith("Vacation")  # the original wording lands on the wrong place
    out = retrieve_expanded(
        world.conn, TextEmbedder(), question, world.ann, history=[],
        settings=settings(multi_query=2, retrieval_mode="dense", retrieval_candidates=1),
        complete_fn=model({"expand": {"queries": ["refund conditions", "money back rules"]}}),
    )  # fmt: skip
    assert [f.candidate.text for f in out.items if "Refund" in f.candidate.text]
    assert out.queries == [question, "refund conditions", "money back rules"]
    # found by two of the three phrasings -> it must be among the first two (ties with the original
    # wording's top hit are broken by random ids, so "exactly first" would be a flaky claim)
    assert any("Refund" in f.candidate.text for f in out.items[:2])


def test_hyde_adds_a_dense_search_by_the_made_up_passage(world: World) -> None:
    question = "what is the policy on getting cash back?"
    out = retrieve_expanded(
        world.conn, TextEmbedder(), question, world.ann, history=[],
        settings=settings(hyde=True, retrieval_mode="dense", retrieval_candidates=1),
        complete_fn=model({"hyde": "Refund requests are accepted within thirty days of purchase."}),
    )  # fmt: skip
    assert out.hyde and any("Refund" in f.candidate.text for f in out.items[:2])


def test_transforms_never_cross_tenants(world: World) -> None:
    out = retrieve_expanded(
        world.conn, TextEmbedder(), "workshop topics?", world.ann, history=HISTORY,
        settings=settings(query_rewrite=True, multi_query=2, hyde=True),
        complete_fn=model({"rewrite": "workshop topics", "expand": {"queries": ["workshop list", "workshop agenda"]}, "hyde": "The workshop covers many topics in detail for teams."}),
    )  # fmt: skip
    assert all("GLOBEX" not in f.candidate.text for f in out.items)


def cand(i: int, text: str = "t") -> Candidate:
    import uuid

    return Candidate(
        uuid.UUID(int=i),
        uuid.UUID(int=0),
        "d.md",
        "S",
        None,
        None,
        [],
        f"{text}{i}",
        "",
        None,
        i,
        0.1,
    )


def test_merge_fused_rewards_agreement_and_keeps_details() -> None:
    a = [Fused(cand(1), 0.0, dense_rank=1, rank=1), Fused(cand(2), 0.0, dense_rank=2, rank=2)]
    b = [
        Fused(cand(3), 0.0, keyword_rank=1, keyword_score=9.0, rank=1),
        Fused(cand(2), 0.0, keyword_rank=2, rank=2),
    ]
    merged = merge_fused([a, b], k=60)
    assert [f.candidate.chunk_id.int for f in merged] == [2, 1, 3]  # 2 is in both lists: first
    assert merged[0].rrf_score == pytest.approx(1 / 62 + 1 / 62) and [f.rank for f in merged] == [
        1,
        2,
        3,
    ]
    assert merged[2].keyword_score == 9.0  # details of the list it came from are kept
