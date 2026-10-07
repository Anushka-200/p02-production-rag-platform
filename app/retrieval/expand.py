"""Retrieval with query transforms: rewrite -> several searches -> merge.

    question + chat history
        | rewrite (only when there is history)      -> standalone question
        | multi-query (optional)                    -> + N rephrasings
        | HyDE (optional)                           -> + a dense search by a made-up answer
        v
    one hybrid search per query variant -> merged by rank (RRF) -> rerank, context, answer

With every switch off this is exactly `retrieve` (same results, no model call), so turning the
transforms on is always a measurable, reversible change.
"""

import time
from collections.abc import Callable
from typing import Any

import psycopg

from app.config import Settings, get_settings
from app.embeddings import Embedder
from app.llm import complete
from app.retrieval.fusion import Fused, fuse, merge_fused
from app.retrieval.retrieve import Retrieval, retrieve
from app.retrieval.search import Scope, dense_search
from app.retrieval.transform import (
    TransformCache,
    expand_queries,
    hyde_passage,
    rewrite_query,
)
from app.retrieval.tree import tree_candidates


def retrieve_expanded(
    conn: psycopg.Connection,
    embedder: Embedder,
    question: str,
    scope: Scope,
    *,
    history: list[dict[str, str]] | None = None,
    settings: Settings | None = None,
    mode: str | None = None,
    complete_fn: Callable[..., Any] = complete,
    cache: TransformCache | None = None,
) -> Retrieval:
    s = settings or get_settings()
    history = history or []
    use_rewrite = s.query_rewrite and bool(history)
    if not (use_rewrite or s.multi_query > 0 or s.hyde or s.summary_tree):
        base = retrieve(conn, embedder, question, scope, settings=s, mode=mode)
        base.question, base.queries = question, [question]
        return base

    started = time.perf_counter()
    cost = 0.0
    standalone = question
    if use_rewrite:
        rewritten = rewrite_query(
            question, history, complete_fn=complete_fn, model=s.transform_model, cache=cache
        )
        standalone, cost = rewritten.text, cost + rewritten.cost_usd
    variants = [standalone]
    if s.multi_query > 0:
        extra, extra_cost, _ = expand_queries(
            standalone, s.multi_query, complete_fn=complete_fn, model=s.transform_model, cache=cache
        )
        variants += extra
        cost += extra_cost
    transform_ms = round((time.perf_counter() - started) * 1000, 1)

    results = [retrieve(conn, embedder, q, scope, settings=s, mode=mode) for q in variants]
    lists: list[list[Fused]] = [r.items for r in results]
    hyde_text = None
    if s.hyde:
        hyde = hyde_passage(
            standalone, complete_fn=complete_fn, model=s.transform_model, cache=cache
        )
        cost += hyde.cost_usd
        if hyde.used:
            hyde_text = hyde.text
            vector = embedder.embed_query(hyde.text)
            found = dense_search(conn, vector, scope, s.retrieval_candidates)
            lists.append(fuse(found, [], k=s.rrf_k))  # a dense-only list, same Fused shape
    if s.summary_tree:
        # a broad question matches a SUMMARY; the real chunks under the best summaries join the pool
        under = tree_candidates(
            conn, embedder.embed_query(standalone), scope,
            top_k=s.summary_top_k, per_summary=s.summary_members_per,
        )  # fmt: skip
        if under:
            lists.append(fuse(under, [], k=s.rrf_k))
    items = lists[0] if len(lists) == 1 else merge_fused(lists, k=s.rrf_k)
    first = results[0]
    return Retrieval(
        items, first.mode, first.keywords, {**first.timings_ms, "transform_ms": transform_ms},
        question=standalone, queries=variants, hyde=hyde_text, transform_cost_usd=round(cost, 6),
    )  # fmt: skip
