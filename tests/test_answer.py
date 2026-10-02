import json
import uuid

import pytest

from app.answer import REFUSAL, answer_question
from app.llm import LLMError, LLMResult
from app.retrieval.context import Context, Source


def make_context(n: int = 2) -> Context:
    sources = [
        Source(
            f"S{i}",
            uuid.uuid4(),
            uuid.uuid4(),
            f"doc{i}.md",
            f"Doc > Sec {i}",
            "5" if i == 1 else None,
            [f"sec-{i}"],
            f"text {i}",
            20,
        )
        for i in range(1, n + 1)
    ]
    text = "\n\n".join(f'<source id="{s.id}">\n{s.text}\n</source>' for s in sources)
    return Context(text, sources, 40)


def reply(**payload: object) -> LLMResult:
    return LLMResult(json.dumps(payload), "m", 100, 30, 0.002, 0.5, 1)


class Script:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes, self.calls = list(outcomes), []

    def __call__(self, messages, **kw):  # type: ignore[no-untyped-def]
        self.calls.append((messages, kw))
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_a_supported_answer_comes_back_with_resolved_citations() -> None:
    fake = Script(reply(supported=True, answer="Twenty days per year [S1].", citations=["S1"]))
    a = answer_question("How many days?", make_context(), complete_fn=fake)
    assert not a.refused and a.text == "Twenty days per year [S1]." and a.llm_calls == 1
    assert [(c.id, c.document, c.section, c.pages, c.anchors) for c in a.citations] == [
        ("S1", "doc1.md", "Doc > Sec 1", "5", ["sec-1"])
    ]
    assert a.cost_usd == pytest.approx(0.002) and fake.calls[0][1]["temperature"] == 0.0


def test_the_prompt_declares_sources_untrusted_and_contains_the_sources_and_question() -> None:
    fake = Script(reply(supported=True, answer="A [S1].", citations=["S1"]))
    answer_question("What is it?", make_context(), complete_fn=fake)
    system, user = fake.calls[0][0]
    assert "untrusted" in system["content"] and "ONLY" in system["content"]
    assert '<source id="S1">' in user["content"] and user["content"].endswith(
        "Question: What is it?"
    )


def test_an_unsupported_answer_becomes_the_standard_refusal() -> None:
    fake = Script(reply(supported=False, answer="I think it might be 20", citations=[]))
    a = answer_question("Unknown?", make_context(), complete_fn=fake)
    assert a.refused and a.text == REFUSAL and a.citations == [] and "do not contain" in a.reason


def test_no_sources_means_refusal_without_calling_the_model() -> None:
    fake = Script()
    a = answer_question("Anything?", Context("", [], 0), complete_fn=fake)
    assert a.refused and a.llm_calls == 0 and fake.calls == [] and a.cost_usd == 0.0


def test_a_citation_to_a_source_that_was_never_sent_is_repaired_once() -> None:
    fake = Script(
        reply(supported=True, answer="Twenty days [S7].", citations=["S7"]),  # S7 does not exist
        reply(supported=True, answer="Twenty days [S1].", citations=["S1"]),
    )
    a = answer_question("How many?", make_context(), complete_fn=fake)
    assert (
        not a.refused and a.repaired and a.llm_calls == 2 and a.cost_usd == pytest.approx(0.004)
    )  # both calls counted
    repair_message = fake.calls[1][0][-1]["content"]
    assert (
        "do not exist" in repair_message and "S7" in repair_message
    )  # the model is told exactly why


def test_a_marker_in_the_text_that_is_not_in_the_citations_list_is_still_checked() -> None:
    fake = Script(
        reply(supported=True, answer="Fact [S1] and invented [S9].", citations=["S1"]),
        reply(supported=True, answer="Fact [S1].", citations=["S1"]),
    )
    assert not answer_question("q", make_context(), complete_fn=fake).refused


def test_a_supported_answer_without_any_citation_is_rejected() -> None:
    fake = Script(
        reply(supported=True, answer="Twenty days.", citations=[]),
        reply(supported=True, answer="Twenty days [S2].", citations=["S2"]),
    )
    a = answer_question("q", make_context(), complete_fn=fake)
    assert a.repaired and [c.id for c in a.citations] == ["S2"]


def test_invalid_json_twice_ends_in_a_refusal_not_a_broken_answer() -> None:
    bad = LLMResult("Sure! The answer is twenty.", "m", 100, 10, 0.001, 0.1, 1)
    a = answer_question("q", make_context(), complete_fn=Script(bad, bad))
    assert a.refused and a.text == REFUSAL and "stayed invalid" in a.reason and a.llm_calls == 2


def test_code_fences_around_the_json_are_accepted() -> None:
    fenced = LLMResult(
        '```json\n{"supported": true, "answer": "Yes [S1].", "citations": ["S1"]}\n```',
        "m",
        1,
        1,
        0.0,
        0.1,
        1,
    )
    assert not answer_question("q", make_context(), complete_fn=Script(fenced)).refused


def test_a_failing_model_call_refuses_gracefully() -> None:
    a = answer_question("q", make_context(), complete_fn=Script(LLMError("provider down")))
    assert a.refused and "model call failed" in a.reason
