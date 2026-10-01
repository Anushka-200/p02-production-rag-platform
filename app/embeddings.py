"""Embeddings: turn text into numbers so "similar meaning" becomes "close together".

Two implementations behind one small interface, so experiments only change settings:
- LocalEmbedder: runs on this machine through fastembed (ONNX, no GPU, free).
- LiteLLMEmbedder: calls a hosted model through LiteLLM (cost is tracked per call).

Documents and queries are embedded through different methods: some models (BGE among them)
are trained to treat "a question" and "a passage that answers it" slightly differently.
"""

import logging
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Protocol

import litellm

from app.config import Settings, get_settings
from app.llm import RETRYABLE, backoff_delay

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """An embedding call still failed after retries, or failed in a way retries cannot fix."""


class Embedder(Protocol):
    model: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def vector_literal(vector: list[float]) -> str:
    """A vector in the text form Postgres accepts: '[0.1,0.2,0.3]' (use with %s::vector)."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class LocalEmbedder:
    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        batch_size: int = 16,
    ) -> None:
        from fastembed import TextEmbedding  # imported here: heavy, and tests with fakes skip it

        self.model, self.dimension, self.batch_size = model, dimension, batch_size
        self._model = TextEmbedding(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.passage_embed(texts, batch_size=self.batch_size)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()


class LiteLLMEmbedder:
    """Hosted embedding model through LiteLLM, with the same retry rules as app.llm."""

    def __init__(
        self,
        model: str,
        dimension: int,
        batch_size: int = 64,
        max_retries: int = 3,
        # injected so tests run without network, sleeping or a key
        embedding_fn: Callable[..., Any] = litellm.embedding,
        cost_fn: Callable[..., float] = litellm.completion_cost,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model, self.dimension, self.batch_size = model, dimension, batch_size
        self.max_retries = max_retries
        self._embed, self._cost, self._sleep = embedding_fn, cost_fn, sleep_fn
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.cost_known = True  # becomes False if any call had no known price

    def _call(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(1, self.max_retries + 2):
            try:
                started = time.perf_counter()
                response = self._embed(model=self.model, input=batch)
                break
            except RETRYABLE as exc:
                if attempt == self.max_retries + 1:
                    raise EmbeddingError(
                        f"{self.model} failed after {attempt} attempts: {exc}"
                    ) from exc
                self._sleep(backoff_delay(attempt))
            except Exception as exc:
                raise EmbeddingError(
                    f"{self.model} call failed ({type(exc).__name__}): {exc}"
                ) from exc
        tokens = getattr(response.usage, "total_tokens", 0) or 0
        try:
            cost: float | None = self._cost(completion_response=response)
        except Exception:
            cost = None
        self.total_tokens += tokens
        if cost is None:
            self.cost_known = False
        else:
            self.total_cost_usd += cost
        logger.info(
            "embedding call",
            extra={
                "model": self.model,
                "texts": len(batch),
                "tokens": tokens,
                "cost_usd": cost,
                "latency_s": round(time.perf_counter() - started, 3),
                "attempts": attempt,
            },
        )
        # the API may return items out of order: sort by their index
        items = sorted(response.data, key=lambda item: item["index"])
        return [item["embedding"] for item in items]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._call(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._call([text])[0]


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    if s.embedding_provider == "local":
        return LocalEmbedder(s.embedding_model, s.embedding_dimension, s.embedding_batch_size)
    return LiteLLMEmbedder(s.embedding_model, s.embedding_dimension, s.embedding_batch_size)


@lru_cache
def model_token_counter(model_name: str) -> Callable[[str], int]:
    """Count tokens exactly as the embedding model does, to plug into the chunkers.

    The model silently cuts input beyond its maximum (512 tokens for BGE), so a chunk bigger than
    that loses its tail. Counting with the model's own tokenizer prevents that.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer.no_truncation()  # we want the true length, not a capped one

    def count(text: str) -> int:
        return len(tokenizer.encode(text).ids)

    return count


def get_token_counter(settings: Settings | None = None) -> Callable[[str], int]:
    """The token counter the chunkers should use for the configured embedding model.

    Local model: its real tokenizer (exact). Hosted model: the cheap estimate, because we have
    no tokenizer for it; keep chunk sizes comfortably below the provider's input limit.
    """
    from app.ingest.chunking import approx_tokens

    s = settings or get_settings()
    if s.embedding_provider == "local":
        return model_token_counter(s.embedding_model)
    return approx_tokens
