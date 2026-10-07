"""Query transforms: rewriting, multi-query, HyDE, and the cache. No network: the model is a fake."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.llm import LLMError
from app.retrieval.transform import (
    TransformCache,
    clean_question,
    expand_queries,
    format_history,
    hyde_passage,
    rewrite_query,
)

HISTORY = [
    {"role": "user", "content": "What kinds of training programs does GitLab offer?"},
    {"role": "assistant", "content": "GitLab offers Team Workshops, customized group sessions."},
]
QUESTION = "How many different topics are available for these sessions?"


class FakeModel:
    def __init__(self, reply: Any) -> None:
        self.reply, self.calls = reply, []  # type: ignore[var-annotated]

    def __call__(self, messages: list[dict[str, Any]], **kw: Any) -> Any:
        self.calls.append((messages, kw))
        if isinstance(self.reply, Exception):
            raise self.reply
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return SimpleNamespace(text=text, cost_usd=0.0005)


# ---- helpers --------------------------------------------------------------------------------------
def test_history_is_formatted_short_and_only_with_real_roles() -> None:
    messy = [
        {"role": "system", "content": "secret"},
        *HISTORY,
        {"role": "user", "content": "x" * 900},
    ]
    text = format_history(messy)
    assert (
        "secret" not in text
        and text.startswith("user: What kinds")
        and len(text.splitlines()[-1]) < 520
    )
    assert format_history([]) == ""


@pytest.mark.parametrize(
    ("raw", "ok"),
    [
        ("How many topics does GitLab Team Workshops cover?", True),
        ('"How many topics are there?"', True),
        ("", False),
        ("line one\nline two", False),
        ("x" * 400, False),
        ("Sure! " + "blah " * 80, False),  # an essay/answer, far longer than the question
    ],
)
def test_clean_question_accepts_only_a_short_single_line(raw: str, ok: bool) -> None:
    assert (clean_question(raw, "How many topics are there?") is not None) is ok


# ---- rewrite --------------------------------------------------------------------------------------
def test_a_follow_up_is_rewritten_using_the_history() -> None:
    model = FakeModel("How many topics are available in GitLab's Team Workshops?")
    out = rewrite_query(QUESTION, HISTORY, complete_fn=model)
    assert out.text == "How many topics are available in GitLab's Team Workshops?" and out.used
    assert out.cost_usd == 0.0005 and out.calls == 1
    sent = json.dumps(model.calls[0][0])
    assert (
        "Team Workshops" in sent and QUESTION in sent
    )  # the model saw the history and the question


def test_without_history_there_is_no_model_call() -> None:
    model = FakeModel("never used")
    out = rewrite_query(QUESTION, [], complete_fn=model)
    assert out.text == QUESTION and not out.used and model.calls == []


@pytest.mark.parametrize("reply", ["", "two\nlines", "x" * 500, LLMError("down")])
def test_a_bad_or_failed_rewrite_falls_back_to_the_original(reply: Any) -> None:
    out = rewrite_query(QUESTION, HISTORY, complete_fn=FakeModel(reply))
    assert out.text == QUESTION and not out.used


def test_an_unchanged_rewrite_is_not_counted_as_used() -> None:
    assert not rewrite_query(QUESTION, HISTORY, complete_fn=FakeModel(QUESTION)).used


# ---- multi-query ----------------------------------------------------------------------------------
def test_rephrasings_are_deduplicated_capped_and_never_include_the_original() -> None:
    reply = {
        "queries": [
            "refund policy eligibility",
            "REFUND POLICY ELIGIBILITY",
            "When can I request a refund?",
            "money back rules",
            "return conditions",
        ]
    }
    got, cost, calls = expand_queries(
        "When can I request a refund?", 3, complete_fn=FakeModel(reply)
    )
    assert (
        got == ["refund policy eligibility", "money back rules", "return conditions"] and calls == 1
    )


@pytest.mark.parametrize(
    "reply", ["not json", {"queries": "nope"}, {"other": 1}, "", LLMError("x")]
)
def test_bad_multi_query_output_means_no_extra_queries(reply: Any) -> None:
    assert expand_queries("q?", 3, complete_fn=FakeModel(reply))[0] == []


def test_zero_rephrasings_costs_nothing() -> None:
    model = FakeModel({"queries": ["a"]})
    assert expand_queries("q?", 0, complete_fn=model) == ([], 0.0, 0) and model.calls == []


def test_code_fenced_json_is_accepted() -> None:
    reply = '```json\n{"queries": ["money back rules"]}\n```'
    assert expand_queries("When can I request a refund?", 2, complete_fn=FakeModel(reply))[0] == [
        "money back rules"
    ]


# ---- HyDE -----------------------------------------------------------------------------------------
def test_hyde_returns_a_made_up_passage_capped_in_length() -> None:
    out = hyde_passage("Why 30 days?", complete_fn=FakeModel("The notice period is 30 days. " * 80))
    assert out.used and len(out.text) <= 700 and out.text.startswith("The notice period")


def test_a_useless_hyde_reply_falls_back_to_the_question() -> None:
    out = hyde_passage("Why 30 days?", complete_fn=FakeModel("ok"))
    assert out.text == "Why 30 days?" and not out.used


# ---- the cache ------------------------------------------------------------------------------------
def test_the_cache_makes_the_second_call_free_and_survives_a_restart(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "cache.jsonl"
    model = FakeModel("How many topics does Team Workshops cover?")
    first = rewrite_query(QUESTION, HISTORY, complete_fn=model, cache=TransformCache(path))
    again = rewrite_query(
        QUESTION, HISTORY, complete_fn=model, cache=TransformCache(path)
    )  # a "new run"
    assert (
        len(model.calls) == 1
        and again.text == first.text
        and again.cost_usd == 0.0
        and again.calls == 0
    )


def test_the_cache_key_depends_on_the_input_and_the_model() -> None:
    k = TransformCache.key
    assert k("rewrite", "haiku", "a") != k("rewrite", "haiku", "b") != k("rewrite", "sonnet", "b")
    assert k("rewrite", "haiku", "a") != k("hyde", "haiku", "a")
