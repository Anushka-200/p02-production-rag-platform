from types import SimpleNamespace

import pytest
from litellm import exceptions as exc

from app.config import Settings
from app.llm import LLMError, backoff_delay, complete

SETTINGS = Settings(_env_file=None, llm_max_retries=3)  # type: ignore[call-arg]


def fake_response(text: str = "hi") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class Script:
    """A fake model: raises or returns the scripted outcomes in order, counting calls."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, **_: object) -> object:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def rate_limit() -> exc.RateLimitError:
    return exc.RateLimitError("slow down", "anthropic", "m")


def run(script: Script, sleeps: list[float] | None = None):
    return complete(
        [{"role": "user", "content": "x"}],
        settings=SETTINGS,
        completion_fn=script,
        cost_fn=lambda **_: 0.001,
        sleep_fn=(sleeps.append if sleeps is not None else lambda _: None),
    )


def test_success_returns_text_tokens_cost() -> None:
    r = run(Script(fake_response("hello")))
    assert (r.text, r.input_tokens, r.output_tokens, r.cost_usd, r.attempts) == (
        "hello",
        10,
        5,
        0.001,
        1,
    )


def test_retries_temporary_errors_then_succeeds() -> None:
    sleeps: list[float] = []
    script = Script(rate_limit(), rate_limit(), fake_response())
    r = run(script, sleeps)
    assert r.attempts == 3 and script.calls == 3
    assert len(sleeps) == 2  # waited between the attempts


def test_gives_up_after_max_retries_with_clear_error() -> None:
    script = Script(*[rate_limit() for _ in range(4)])
    with pytest.raises(LLMError, match="failed after 4 attempts"):
        run(script)
    assert script.calls == 4  # 1 try + 3 retries


def test_non_retryable_error_fails_immediately() -> None:
    script = Script(exc.AuthenticationError("bad key", "anthropic", "m"))
    with pytest.raises(LLMError, match="AuthenticationError"):
        run(script)
    assert script.calls == 1


def test_missing_price_gives_none_not_zero() -> None:
    def no_price(**_: object) -> float:
        raise ValueError("unknown model")

    r = complete(
        [{"role": "user", "content": "x"}],
        settings=SETTINGS,
        completion_fn=Script(fake_response()),
        cost_fn=no_price,
    )
    assert r.cost_usd is None


def test_backoff_grows_but_is_capped_and_jittered() -> None:
    for attempt, ceiling in [(1, 1.0), (2, 2.0), (3, 4.0), (10, 20.0)]:
        delays = [backoff_delay(attempt) for _ in range(200)]
        assert max(delays) <= ceiling
        assert len(set(delays)) > 1  # random, not a fixed number
