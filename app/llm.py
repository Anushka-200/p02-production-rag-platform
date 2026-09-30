"""The one place every LLM call goes through.

Why a wrapper: timeouts, retries, cost and latency capture live here once, instead of being
re-implemented (and forgotten) in every module. Swapping the model or adding caching later
is a change in this file only.
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import exceptions as llm_exc

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Temporary problems: trying again can work. Everything else (bad request, wrong key) is our
# mistake and would fail the same way every time, so retrying only wastes time and money.
RETRYABLE = (
    llm_exc.RateLimitError,
    llm_exc.APIConnectionError,
    llm_exc.Timeout,
    llm_exc.ServiceUnavailableError,
    llm_exc.InternalServerError,
)


class LLMError(RuntimeError):
    """Raised when a call still fails after all retries, or fails in a way retries cannot fix."""


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None if LiteLLM has no price for this model yet; never silently 0
    latency_s: float
    attempts: int


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 20.0) -> float:
    """Seconds to wait before retry number `attempt` (1, 2, 3...).

    Exponential: the ceiling doubles each time (1, 2, 4...) up to `cap`.
    Full jitter: pick a random point below the ceiling, so many clients don't retry in lockstep.
    """
    return random.uniform(0, min(cap, base * 2 ** (attempt - 1)))


def complete(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    settings: Settings | None = None,
    # Injected so tests can run without network, real sleeping or a real key.
    completion_fn: Callable[..., Any] = litellm.completion,
    cost_fn: Callable[..., float] = litellm.completion_cost,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> LLMResult:
    s = settings or get_settings()
    model = model or s.llm_default_model
    api_key = s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else None
    max_attempts = s.llm_max_retries + 1
    started = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        try:
            response = completion_fn(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=s.llm_timeout_s,
                api_key=api_key,
                num_retries=0,  # we retry ourselves, so nothing happens behind our back
            )
            break
        except RETRYABLE as exc:
            if attempt == max_attempts:
                raise LLMError(f"{model} failed after {attempt} attempts: {exc}") from exc
            delay = backoff_delay(attempt)
            logger.warning(
                "llm call failed, retrying",
                extra={
                    "model": model,
                    "attempt": attempt,
                    "retry_in_s": round(delay, 2),
                    "error": type(exc).__name__,
                },
            )
            sleep_fn(delay)
        except Exception as exc:  # not retryable: fail fast with a readable message
            raise LLMError(f"{model} call failed ({type(exc).__name__}): {exc}") from exc

    try:
        cost = cost_fn(completion_response=response)
    except Exception:
        cost = None
        logger.warning("no price known for model", extra={"model": model})

    result = LLMResult(
        text=response.choices[0].message.content or "",
        model=model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        cost_usd=cost,
        latency_s=round(time.perf_counter() - started, 3),
        attempts=attempt,
    )
    logger.info(
        "llm call",
        extra={
            "model": model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": cost,
            "latency_s": result.latency_s,
            "attempts": attempt,
        },
    )
    return result
