import json

import pytest

from app.llm import LLMError, LLMResult
from evals.judge import Judge, JudgeError, faithfulness_score


def reply(payload, cost=0.01):  # type: ignore[no-untyped-def]
    return LLMResult(
        payload if isinstance(payload, str) else json.dumps(payload), "judge", 100, 50, cost, 0.5, 1
    )


class Script:
    def __init__(self, *outcomes) -> None:  # type: ignore[no-untyped-def]
        self.outcomes, self.calls = list(outcomes), []

    def __call__(self, messages, **kw):  # type: ignore[no-untyped-def]
        self.calls.append((messages, kw))
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_faithfulness_is_supported_claims_over_all_claims() -> None:
    claims = [
        {"claim": "Born in Germany", "supported": True},
        {"claim": "Born on 20 March", "supported": False},
    ]
    assert faithfulness_score(claims) == 0.5  # the Ragas example: Einstein, 1 of 2
    assert faithfulness_score([{"supported": True}] * 3) == 1.0
    assert faithfulness_score([]) is None  # no claims (e.g. a refusal): not applicable


def test_the_judge_returns_the_score_computed_by_code_from_the_models_claims(tmp_path) -> None:  # type: ignore[no-untyped-def]
    fake = Script(reply({"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": True},
                                    {"claim": "c", "supported": False}, {"claim": "d", "supported": True}]}))  # fmt: skip
    v = Judge(fake, model="m").faithfulness("Four claims [S1].", "context text")
    assert (
        v.score == pytest.approx(0.75)
        and v.detail["supported"] == 3
        and v.detail["total"] == 4
        and v.cost_usd == 0.01
    )
    messages, kw = fake.calls[0]
    assert kw["model"] == "m" and kw["temperature"] is None  # Sonnet 5.5 accepts only its default
    assert (
        "<context>\ncontext text" in messages[1]["content"] and "[S1]" not in messages[1]["content"]
    )  # markers removed
    assert "DATA" in messages[0]["content"]  # told not to obey text inside the inputs


def test_correctness_maps_labels_to_scores() -> None:
    for label, score in [("correct", 1.0), ("partial", 0.5), ("incorrect", 0.0)]:
        fake = Script(reply({"label": label, "reason": "because"}))
        v = Judge(fake, model="m").correctness("q?", "answer", "gold")
        assert v.score == score and v.detail["label"] == label


def test_an_invalid_judge_answer_is_retried_once_then_fails_clearly() -> None:
    fake = Script(reply("not json"), reply({"label": "correct", "reason": "ok"}))
    assert Judge(fake, model="m").correctness("q", "a", "g").score == 1.0 and len(fake.calls) == 2
    with pytest.raises(JudgeError, match="label"):
        Judge(Script(reply({"label": "maybe"}), reply({"label": "maybe"})), model="m").correctness(
            "q", "a", "g"
        )
    with pytest.raises(JudgeError, match="judge call failed"):
        Judge(Script(LLMError("down")), model="m").correctness("q", "a", "g")


def test_verdicts_are_cached_on_disk_by_content(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "judge.jsonl"
    first = Script(reply({"label": "correct", "reason": "ok"}))
    Judge(first, model="m", cache_path=path).correctness("q", "answer", "gold")
    second = Script()  # must not be called
    again = Judge(second, model="m", cache_path=path).correctness(
        "q", "answer", "gold"
    )  # a NEW object, same file
    assert again.cached and again.score == 1.0 and again.cost_usd == 0.0 and second.calls == []
    changed = Script(reply({"label": "incorrect", "reason": "no"}))
    assert (
        Judge(changed, model="m", cache_path=path)
        .correctness("q", "different answer", "gold")
        .score
        == 0.0
    )


def test_a_different_judge_model_does_not_reuse_anothers_verdicts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "judge.jsonl"
    Judge(
        Script(reply({"label": "correct", "reason": ""})), model="m1", cache_path=path
    ).correctness("q", "a", "g")
    other = Script(reply({"label": "incorrect", "reason": ""}))
    assert (
        Judge(other, model="m2", cache_path=path).correctness("q", "a", "g").score == 0.0
        and len(other.calls) == 1
    )


# ---- the synthetic calibration set ---------------------------------------------------------------
def test_swap_number_makes_the_claim_false_and_leaves_other_text_alone() -> None:
    from evals.calibrate_judge import swap_number

    assert swap_number("Employees get 20 vacation days") == "Employees get 47 vacation days"
    assert (
        swap_number("You can contribute up to $4,300 in 2025.")
        == "You can contribute up to $8607 in 2025."
    )
    assert swap_number("No digits here") is None


def test_cases_cover_four_variants_per_item_deterministically() -> None:
    from evals.calibrate_judge import build_cases
    from evals.schema import EvalItem, GoldRef

    items = [
        EvalItem(id=f"a-single-{i:03d}", tenant="A", kind="single", question=f"q{i}", answerable=True, corpus="gitlab",
                 gold=[GoldRef(doc=f"d{i}.md", locator="s")], gold_answer=f"Answer {i} is {i + 10} days.")
        for i in range(6)
    ]  # fmt: skip
    passages = {(f"d{i}.md", "s"): f"passage {i}" for i in range(6)}
    cases = build_cases(items, passages, n=4, seed=1)
    assert len(cases) == 16 and {c.variant for c in cases} == {
        "faithful",
        "number",
        "extra_claim",
        "other",
    }
    assert [c.answer for c in cases] == [
        c.answer for c in build_cases(items, passages, n=4, seed=1)
    ]  # reproducible
    faithful = next(c for c in cases if c.variant == "faithful")
    assert faithful.answer == faithful.gold and faithful.context.startswith("passage")
    number = next(c for c in cases if c.variant == "number")
    assert number.answer != number.gold


def test_summary_reports_detection_and_false_alarm_rates() -> None:
    from evals.calibrate_judge import summarize

    rows = [
        {"variant": "faithful", "faith": 1.0, "label": "correct"},
        {"variant": "faithful", "faith": 0.5, "label": "correct"},  # a false alarm
        {"variant": "number", "faith": 0.5, "label": "incorrect"},
        {"variant": "number", "faith": 1.0, "label": "correct"},  # missed
        {"variant": "extra_claim", "faith": 0.75, "label": "correct"},
        {"variant": "other", "faith": 0.0, "label": "incorrect"},
    ]
    s = summarize(rows)
    assert s["false_alarm_rate"] == 0.5 and s["detection_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert (
        s["correctness_not_correct_for_number"] == 0.5
        and s["correctness_incorrect_for_other"] == 1.0
    )


def test_a_none_temperature_is_not_sent_to_the_provider() -> None:
    from app.config import Settings
    from app.llm import complete

    seen: dict = {}  # type: ignore[type-arg]

    def provider(**kw):  # type: ignore[no-untyped-def]
        seen.update(kw)
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    complete(
        [{"role": "user", "content": "q"}],
        temperature=None,
        settings=s,
        completion_fn=provider,
        cost_fn=lambda **_: 0.0,
    )
    assert "temperature" not in seen
    complete(
        [{"role": "user", "content": "q"}],
        temperature=0.3,
        settings=s,
        completion_fn=provider,
        cost_fn=lambda **_: 0.0,
    )
    assert seen["temperature"] == 0.3
