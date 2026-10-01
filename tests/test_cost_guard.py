import argparse

import pytest

from app.cost_guard import (
    BudgetExceeded,
    CostGuard,
    SpendDeclined,
    add_cost_args,
    estimate_llm_cost,
    guard_from_args,
)

HAIKU = (
    "anthropic/claude-haiku-4-5-20251001"  # $1 per million input tokens, $5 output (LiteLLM table)
)


def quiet(projected: float | None, **kw: object) -> tuple[CostGuard, list[str]]:
    said: list[str] = []
    return CostGuard("test job", projected, say=said.append, **kw), said  # type: ignore[arg-type]


def test_estimate_uses_the_model_price_table() -> None:
    assert estimate_llm_cost(HAIKU, 1_000_000, 1_000_000) == pytest.approx(6.0)
    assert estimate_llm_cost(HAIKU, 1000, 100, calls=10) == pytest.approx(10 * (0.001 + 0.0005))


def test_cached_input_is_cheaper_when_the_model_has_a_cache_price() -> None:
    plain = estimate_llm_cost(HAIKU, 3000, 60, calls=1000)
    cached = estimate_llm_cost(HAIKU, 3000, 60, calls=1000, cached_input_tokens=2700)
    assert plain is not None and cached is not None and cached < plain * 0.6


def test_unknown_model_gives_no_estimate() -> None:
    assert estimate_llm_cost("nobody/unknown-model", 100, 100) is None


def test_small_projection_proceeds_without_asking() -> None:
    guard, said = quiet(0.10, ask=lambda _: pytest.fail("must not ask"))
    guard.confirm()
    assert "projected $0.10" in said[0]


def test_big_projection_needs_a_yes() -> None:
    guard, _ = quiet(5.0, ask=lambda _: "y")
    guard.confirm()  # typed yes: fine
    declined, _ = quiet(5.0, ask=lambda _: "n")
    with pytest.raises(SpendDeclined, match="--yes"):
        declined.confirm()


def test_assume_yes_skips_the_question_but_still_prints() -> None:
    guard, said = quiet(5.0, assume_yes=True, ask=lambda _: pytest.fail("must not ask"))
    guard.confirm()
    assert "projected $5.00" in said[0]


def test_no_keyboard_means_refuse_not_spend() -> None:
    def no_input(_: str) -> str:
        raise EOFError

    guard, _ = quiet(5.0, ask=no_input)
    with pytest.raises(SpendDeclined):
        guard.confirm()


def test_unknown_price_always_needs_confirmation() -> None:
    guard, said = quiet(None, ask=lambda _: "n")
    with pytest.raises(SpendDeclined):
        guard.confirm()
    assert "UNKNOWN" in said[0]


def test_real_spend_over_the_hard_limit_stops_the_script() -> None:
    guard, _ = quiet(1.0)  # default hard limit = 2x projection = $2
    for _ in range(19):
        guard.record(0.10)  # $1.90 so far: fine
    with pytest.raises(BudgetExceeded, match="over the \\$2.00 limit"):
        guard.record(0.20)


def test_unknown_call_costs_are_counted_not_treated_as_zero() -> None:
    guard, _ = quiet(1.0)
    guard.record(None)
    guard.record(0.5)
    assert guard.unknown_cost_calls == 1 and "1 calls with unknown price" in guard.summary()


def test_command_line_flags() -> None:
    parser = argparse.ArgumentParser()
    add_cost_args(parser)
    guard = guard_from_args(parser.parse_args(["--yes", "--max-usd", "3"]), "job", 1.0)
    assert guard.assume_yes and guard.max_usd == 3.0
