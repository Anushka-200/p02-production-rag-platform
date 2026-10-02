"""Cross-encoder reranking: a model reads the question and one chunk TOGETHER and scores it.

The first-stage search compares the question and each chunk separately (fast, approximate). A
cross-encoder sees both at once (accurate, slow: one model run per pair), so it only reorders the
best few candidates. Two interchangeable implementations: local and hosted (Cohere via LiteLLM).
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import litellm

from app.config import Settings, get_settings
from app.llm import RETRYABLE, backoff_delay
from app.retrieval.fusion import Fused


class RerankError(RuntimeError):
    pass


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage (bigger = more relevant), in the same order."""
        ...


class LocalReranker:
    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2", batch_size: int = 16) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # heavy import, loaded on use

        self.name, self.batch_size = model, batch_size
        self._model = TextCrossEncoder(model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.rerank(query, passages, batch_size=self.batch_size)]


class CohereReranker:
    """Hosted reranker through LiteLLM (needs COHERE_API_KEY). Same retry rules as app.llm."""

    def __init__(
        self,
        model: str = "cohere/rerank-english-v3.0",
        max_retries: int = 3,
        rerank_fn: Callable[..., Any] = litellm.rerank,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.name, self.max_retries, self._rerank, self._sleep = (
            model,
            max_retries,
            rerank_fn,
            sleep_fn,
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self._rerank(
                    model=self.name, query=query, documents=passages, top_n=len(passages)
                )
                break
            except RETRYABLE as exc:
                if attempt == self.max_retries + 1:
                    raise RerankError(
                        f"{self.name} failed after {attempt} attempts: {exc}"
                    ) from exc
                self._sleep(backoff_delay(attempt))
            except Exception as exc:
                raise RerankError(f"{self.name} call failed ({type(exc).__name__}): {exc}") from exc
        scores = [0.0] * len(passages)
        for item in response.results:  # results come sorted by score: map them back by index
            scores[item["index"]] = float(item["relevance_score"])
        return scores


def get_reranker(settings: Settings | None = None) -> Reranker | None:
    s = settings or get_settings()
    if s.reranker_provider == "local":
        return LocalReranker(s.reranker_model)
    if s.reranker_provider == "cohere":
        return CohereReranker(s.cohere_rerank_model)
    return None


@dataclass
class Reranked:
    fused: Fused
    rerank_score: float | None  # None when no reranker ran
    rank: int  # final position, 1 = best


def _passage(f: Fused, max_words: int) -> str:
    c = f.candidate
    text = f"{c.context_blurb}\n\n{c.text}" if c.context_blurb else c.text
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def rerank(
    fused: list[Fused],
    query: str,
    reranker: Reranker | None,
    *,
    candidates: int = 20,
    top: int = 8,
    max_words: int = 220,
) -> list[Reranked]:
    """Rerank the first `candidates` items of the fused list and keep the best `top`.

    No reranker: the fused order is kept (so every later stage works with or without one).
    Ties keep the earlier (fused) order, so results never reshuffle.
    """
    pool = fused[:candidates]
    if reranker is None or not pool:
        return [Reranked(f, None, i) for i, f in enumerate(pool[:top], start=1)]
    scores = reranker.score(query, [_passage(f, max_words) for f in pool])
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    return [Reranked(pool[i], scores[i], rank) for rank, i in enumerate(order[:top], start=1)]
