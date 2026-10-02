"""The one place every LLM call goes through.

Why a wrapper: timeouts, retries, cost and latency capture live here once, instead of being
re-implemented (and forgotten) in every module. Swapping the model or adding caching later
is a change in this file only.
"""

import logging
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import exceptions as llm_exc

from app.config import Settings, get_settings
from app.tracing import observation

logger = logging.getLogger(__name__)
# LiteLLM logs every call at INFO in its own format; we already log one JSON line per call.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

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
    cached_input_tokens: int = 0  # part of the input served from the prompt cache


def _cached_tokens(usage: Any) -> int:
    """Tokens read from the prompt cache; providers name this differently, so try each."""
    details = getattr(usage, "prompt_tokens_details", None)
    return int(
        getattr(usage, "cache_read_input_tokens", 0) or getattr(details, "cached_tokens", 0) or 0
    )


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 20.0) -> float:
    """Seconds to wait before retry number `attempt` (1, 2, 3...).

    Exponential: the ceiling doubles each time (1, 2, 4...) up to `cap`.
    Full jitter: pick a random point below the ceiling, so many clients don't retry in lockstep.
    """
    return random.uniform(0, min(cap, base * 2 ** (attempt - 1)))


def complete(
    messages: list[dict[str, Any]],
    *,
    name: str = "llm-call",
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = 0.0,  # None: do not send it (some models accept only their default)
    settings: Settings | None = None,
    # Injected so tests can run without network, real sleeping or a real key.
    completion_fn: Callable[..., Any] = litellm.completion,
    cost_fn: Callable[..., float] = litellm.completion_cost,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> LLMResult:
    s = settings or get_settings()
    model = model or s.llm_default_model
    with observation(
        as_type="generation",
        name=name,
        model=model,
        input=messages,
        model_parameters={"max_tokens": max_tokens, "temperature": temperature},
    ) as obs:
        result = _call_with_retries(
            messages, model, max_tokens, temperature, s, completion_fn, cost_fn, sleep_fn
        )
        if obs is not None:
            obs.update(
                output=result.text,
                usage_details={"input": result.input_tokens, "output": result.output_tokens},
                cost_details={"total": result.cost_usd} if result.cost_usd is not None else None,
                metadata={"attempts": result.attempts, "latency_s": result.latency_s},
            )
    return result


def _call_with_retries(
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: float | None,
    s: Settings,
    completion_fn: Callable[..., Any],
    cost_fn: Callable[..., float],
    sleep_fn: Callable[[float], None],
) -> LLMResult:
    api_key = s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else None
    max_attempts = s.llm_max_retries + 1
    started = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        try:
            extra = {} if temperature is None else {"temperature": temperature}
            response = completion_fn(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=s.llm_timeout_s,
                api_key=api_key,
                num_retries=0,  # we retry ourselves, so nothing happens behind our back
                **extra,
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
        cached_input_tokens=_cached_tokens(response.usage),
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


class LLMStream:
    """A streamed model call: iterate it to get the text pieces as they arrive.

    After the iteration finishes, `.result` holds the full text, token counts and cost (the same
    LLMResult as complete()). Retries happen only BEFORE the first piece arrives; an error after
    that is raised, because the user has already seen part of the answer.
    """

    def __init__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 700,
        temperature: float = 0.0,
        settings: Settings | None = None,
        completion_fn: Callable[..., Any] = litellm.completion,
        cost_fn: Callable[..., float] = litellm.completion_cost,
        build_fn: Callable[..., Any] = litellm.stream_chunk_builder,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._s = settings or get_settings()
        self._messages, self._max_tokens, self._temperature = messages, max_tokens, temperature
        self._model = model or self._s.llm_default_model
        self._completion, self._cost, self._build, self._sleep = (
            completion_fn,
            cost_fn,
            build_fn,
            sleep_fn,
        )
        self.result: LLMResult | None = None
        self.first_piece_s: float | None = None

    def _open(self) -> tuple[Iterator[Any], Any, int]:
        """Start the call; returns (iterator over chunks, the first chunk, attempts used)."""
        api_key = (
            self._s.anthropic_api_key.get_secret_value() if self._s.anthropic_api_key else None
        )
        for attempt in range(1, self._s.llm_max_retries + 2):
            try:
                response = self._completion(
                    model=self._model, messages=self._messages, max_tokens=self._max_tokens,
                    temperature=self._temperature, timeout=self._s.llm_timeout_s, api_key=api_key,
                    num_retries=0, stream=True, stream_options={"include_usage": True},
                )  # fmt: skip
                iterator = iter(response)
                return iterator, next(iterator), attempt
            except StopIteration:
                raise LLMError(f"{self._model} returned an empty stream") from None
            except RETRYABLE as exc:
                if attempt == self._s.llm_max_retries + 1:
                    raise LLMError(f"{self._model} failed after {attempt} attempts: {exc}") from exc
                self._sleep(backoff_delay(attempt))
            except Exception as exc:
                raise LLMError(f"{self._model} call failed ({type(exc).__name__}): {exc}") from exc
        raise LLMError("unreachable")  # pragma: no cover

    def __iter__(self) -> Iterator[str]:
        started = time.perf_counter()
        iterator, first, attempts = self._open()
        chunks, pieces = [first], []
        for chunk in _chain(first, iterator):
            if chunk is not first:
                chunks.append(chunk)
            delta = chunk.choices[0].delta.content if getattr(chunk, "choices", None) else None
            if delta:
                if self.first_piece_s is None:
                    self.first_piece_s = round(time.perf_counter() - started, 3)
                pieces.append(delta)
                yield delta
        full = self._build(chunks, messages=self._messages)
        try:
            cost: float | None = self._cost(completion_response=full)
        except Exception:
            cost = None
        usage = full.usage
        self.result = LLMResult(
            text="".join(pieces), model=self._model, input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens, cost_usd=cost,
            latency_s=round(time.perf_counter() - started, 3), attempts=attempts,
            cached_input_tokens=_cached_tokens(usage),
        )  # fmt: skip
        logger.info(
            "llm stream",
            extra={
                "model": self._model,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "cost_usd": cost,
                "first_piece_s": self.first_piece_s,
                "latency_s": self.result.latency_s,
            },
        )


def _chain(first: Any, rest: Iterator[Any]) -> Iterator[Any]:
    yield first
    yield from rest
