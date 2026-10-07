"""One call that runs the chosen retrieval mode and reports what each stage took."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psycopg

from app.config import Settings, get_settings
from app.embeddings import Embedder
from app.retrieval.fusion import Fused, fuse
from app.retrieval.search import (
    Scope,
    dense_search,
    keyword_query,
    keyword_search,
    rare_keyword_query,
)


@dataclass
class Retrieval:
    items: list[Fused]
    mode: str
    keywords: str  # the OR query that was sent to the keyword search
    timings_ms: dict[str, float] = field(default_factory=dict)
    # Filled by retrieve_expanded (app/retrieval/expand.py); defaults mean "no transforms ran".
    question: str = ""  # the question that was searched and should be answered (standalone)
    queries: list[str] = field(default_factory=list)  # every query variant that was searched
    hyde: str | None = None  # the made-up passage, if HyDE ran
    transform_cost_usd: float = 0.0


def retrieve(
    conn: psycopg.Connection,
    embedder: Embedder,
    question: str,
    scope: Scope,
    *,
    settings: Settings | None = None,
    mode: str | None = None,
    top: int | None = None,
) -> Retrieval:
    """dense, keyword or hybrid. All modes return the same Fused shape for the later stages."""
    s = settings or get_settings()
    mode = mode or s.retrieval_mode
    k_each, timings = s.retrieval_candidates, {}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        result = fn()
        timings[name] = round((time.perf_counter() - started) * 1000, 2)
        return result

    dense, keyword = [], []
    if mode in {"dense", "hybrid"}:
        vector = timed("embed_ms", lambda: embedder.embed_query(question))
        dense = timed("dense_ms", lambda: dense_search(conn, vector, scope, k_each))
    if mode in {"keyword", "hybrid"}:
        keyword = timed(
            "keyword_ms",
            lambda: keyword_search(conn, question, scope, k_each, max_df=s.keyword_max_df),
        )
    weights = {
        "dense_weight": s.rrf_dense_weight if mode != "keyword" else 0.0,
        "keyword_weight": s.rrf_keyword_weight if mode != "dense" else 0.0,
    }
    items = timed("fuse_ms", lambda: fuse(dense, keyword, k=s.rrf_k, **weights))
    shown_query = (
        rare_keyword_query(conn, question, scope, s.keyword_max_df)
        if mode != "dense"
        else keyword_query(question)
    )
    return Retrieval(items[: top or len(items)], mode, shown_query, timings)
