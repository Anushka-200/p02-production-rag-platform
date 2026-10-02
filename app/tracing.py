"""Langfuse tracing. With no keys configured, every function here quietly does nothing.

Why a no-op fallback: tests, CI and a fresh clone must work without a Langfuse account.
The client is created lazily (on first use), never at import time.
"""

from contextlib import AbstractContextManager, nullcontext
from functools import lru_cache
from typing import Any

from langfuse import Langfuse

from app.config import get_settings
from app.pii import redact_any


def _mask(*, data: Any, **_: Any) -> Any:
    """Langfuse calls this on every input/output before upload: scrub emails, cards, phones..."""
    return redact_any(data)


@lru_cache
def get_client() -> Langfuse | None:
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    return Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key.get_secret_value(),
        host=s.langfuse_host,
        mask=_mask,
    )


def observation(**kwargs: Any) -> AbstractContextManager[Any]:
    """A traced step. Use as_type="generation" for an LLM call, "span" for other work."""
    client = get_client()
    if client is None:
        return nullcontext(None)
    return client.start_as_current_observation(**kwargs)


def flush() -> None:
    """Send buffered traces now. Call before a short script exits."""
    client = get_client()
    if client is not None:
        client.flush()
