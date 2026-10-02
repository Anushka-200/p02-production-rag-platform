"""Streaming: the streamed LLM call, the NOT_FOUND hold-back, citation validation after the stream."""

import json
import uuid
from types import SimpleNamespace

import pytest
from litellm import exceptions as exc

from app.answer import NOT_FOUND, REFUSAL, answer_stream
from app.config import Settings
from app.llm import LLMError, LLMResult, LLMStream
from app.retrieval.context import Context, Source

S = Settings(_env_file=None, llm_max_retries=2)  # type: ignore[call-arg]


def chunk(text: str | None):  # type: ignore[no-untyped-def]
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def fake_completion(pieces: list[str | None], fail_first: int = 0):  # type: ignore[no-untyped-def]
    state = {"calls": 0}

    def call(**kw):  # type: ignore[no-untyped-def]
        state["calls"] += 1
        assert kw["stream"] is True and kw["stream_options"] == {"include_usage": True}
        if state["calls"] <= fail_first:
            raise exc.RateLimitError("slow", "p", "m")
        return iter([chunk(p) for p in pieces])

    call.state = state  # type: ignore[attr-defined]
    return call


def build(chunks, messages=None):  # type: ignore[no-untyped-def]
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=7))


def make_stream(pieces, fail_first=0, **kw):  # type: ignore[no-untyped-def]
    return LLMStream(
        [{"role": "user", "content": "q"}], settings=S, completion_fn=fake_completion(pieces, fail_first),
        cost_fn=lambda **_: 0.001, build_fn=build, sleep_fn=lambda _: None, **kw,
    )  # fmt: skip


# ---- LLMStream ---------------------------------------------------------------------------------
def test_pieces_arrive_in_order_and_the_result_is_filled_after_the_last_one() -> None:
    stream = make_stream(["Hel", None, "lo ", "world"])
    assert list(stream) == ["Hel", "lo ", "world"]  # empty pieces are skipped
    r = stream.result
    assert (
        r is not None and r.text == "Hello world" and (r.input_tokens, r.output_tokens) == (100, 7)
    )
    assert r.cost_usd == 0.001 and stream.first_piece_s is not None


def test_a_temporary_error_before_the_first_piece_is_retried() -> None:
    stream = make_stream(["ok"], fail_first=2)
    assert list(stream) == ["ok"] and stream.result.attempts == 3  # type: ignore[union-attr]


def test_it_gives_up_after_the_retries_with_a_readable_error() -> None:
    with pytest.raises(LLMError, match="failed after 3 attempts"):
        list(make_stream(["ok"], fail_first=9))


def test_an_empty_stream_is_an_error_not_silence() -> None:
    def empty(**kw):  # type: ignore[no-untyped-def]
        return iter([])

    stream = LLMStream([{"role": "user", "content": "q"}], settings=S, completion_fn=empty)
    with pytest.raises(LLMError, match="empty stream"):
        list(stream)


# ---- answer_stream -----------------------------------------------------------------------------
def make_context() -> Context:
    sources = [
        Source(
            f"S{i}",
            uuid.uuid4(),
            uuid.uuid4(),
            f"d{i}.md",
            f"Doc > {i}",
            None,
            [f"s{i}"],
            f"text {i}",
            10,
        )
        for i in (1, 2)
    ]
    return Context("<source id='S1'>text 1</source>", sources, 20)


class FakeStream:
    """Stands in for LLMStream: yields the pieces and then exposes .result."""

    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces
        self.result: LLMResult | None = None

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield from self.pieces
        self.result = LLMResult("".join(self.pieces), "m", 100, 20, 0.002, 0.4, 1)


def run(pieces, complete_fn=None):  # type: ignore[no-untyped-def]
    events = list(
        answer_stream(
            "q?",
            make_context(),
            stream_fn=lambda *a, **k: FakeStream(pieces),
            complete_fn=complete_fn or (lambda *a, **k: pytest.fail("no fallback expected")),
        )  # fmt: skip
    )
    tokens = [e["text"] for e in events if e["type"] == "token"]
    return tokens, events[-1]


def test_a_normal_answer_streams_token_by_token_then_reports_validated_citations() -> None:
    tokens, done = run(["Twenty ", "days ", "per year ", "[S1]."])
    assert (
        "".join(tokens) == "Twenty days per year [S1]."
        and done["type"] == "done"
        and not done["replaced"]
    )
    a = done["answer"]
    assert (
        not a.refused
        and [c.id for c in a.citations] == ["S1"]
        and a.cost_usd == 0.002
        and a.llm_calls == 1
    )


def test_not_found_is_never_shown_even_when_split_across_pieces() -> None:
    tokens, done = run(["NOT", "_FO", "UND"])
    assert tokens == [] and done["answer"].refused and done["answer"].text == REFUSAL
    assert "do not contain" in done["answer"].reason


def test_text_that_merely_starts_like_not_found_is_shown() -> None:
    tokens, done = run(["NOT ", "everyone ", "qualifies [S2]."])
    assert "".join(tokens) == "NOT everyone qualifies [S2]." and not done["answer"].refused


def test_the_first_pieces_are_held_back_only_until_the_decision_is_possible() -> None:
    tokens, _ = run(["Tw", "enty [S1]."])
    assert tokens[0] == "Tw"  # 'T' cannot be the start of NOT_FOUND: shown at once


def test_an_invented_citation_triggers_the_validated_fallback() -> None:
    good = LLMResult(
        json.dumps({"supported": True, "answer": "Twenty [S1].", "citations": ["S1"]}),
        "m",
        50,
        10,
        0.001,
        0.2,
        1,
    )
    tokens, done = run(["Twenty days [S9]."], complete_fn=lambda *a, **k: good)
    assert done["replaced"] and not done["answer"].refused and done["answer"].text == "Twenty [S1]."
    assert done["answer"].llm_calls == 2 and done["answer"].cost_usd == pytest.approx(
        0.003
    )  # both calls paid


def test_a_streamed_answer_with_no_citation_at_all_is_replaced_too() -> None:
    good = LLMResult(
        json.dumps({"supported": False, "answer": "", "citations": []}), "m", 50, 10, 0.001, 0.2, 1
    )
    _, done = run(["Probably twenty."], complete_fn=lambda *a, **k: good)
    assert done["replaced"] and done["answer"].refused


def test_no_sources_means_an_immediate_refusal_without_any_model_call() -> None:
    events = list(
        answer_stream("q?", Context("", [], 0), stream_fn=lambda *a, **k: pytest.fail("no call"))
    )
    assert len(events) == 1 and events[0]["answer"].refused and events[0]["answer"].llm_calls == 0


def test_a_failure_before_any_output_ends_in_a_refusal_not_a_crash() -> None:
    class Broken:
        result = None

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise LLMError("provider down")
            yield  # pragma: no cover

    events = list(answer_stream("q?", make_context(), stream_fn=lambda *a, **k: Broken()))
    assert events[-1]["answer"].refused and "model call failed" in events[-1]["answer"].reason


def test_the_streaming_prompt_demands_not_found_and_declares_sources_untrusted() -> None:
    from app.answer import STREAM_SYSTEM

    assert NOT_FOUND in STREAM_SYSTEM and "untrusted" in STREAM_SYSTEM
