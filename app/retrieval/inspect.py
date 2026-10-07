"""The retrieval inspector: show every candidate with every score, for debugging and trust.

Question in -> every candidate the search found, with its dense rank/distance, keyword rank/score,
RRF score, rerank score and whether it made it into the answer's context. No LLM call, no cost.
"""

from collections.abc import Callable
from typing import Any

import psycopg

from app.config import Settings, get_settings
from app.embeddings import Embedder, get_token_counter
from app.llm import complete
from app.retrieval.context import build_context
from app.retrieval.expand import retrieve_expanded
from app.retrieval.rerank import Reranker, rerank
from app.retrieval.search import Scope


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _pages(start: int | None, end: int | None) -> str | None:
    if start is None:
        return None
    return str(start) if end in (None, start) else f"{start}-{end}"


def inspect_retrieval(
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker | None,
    question: str,
    scope: Scope,
    *,
    settings: Settings | None = None,
    mode: str | None = None,
    preview_chars: int = 300,
    history: list[dict[str, str]] | None = None,
    complete_fn: Callable[..., Any] = complete,
) -> dict[str, Any]:
    s = settings or get_settings()
    retrieval = retrieve_expanded(
        conn, embedder, question, scope, history=history, settings=s, mode=mode,
        complete_fn=complete_fn,
    )  # fmt: skip
    question = retrieval.question or question
    scored = rerank(  # every reranked candidate, so the inspector can show all of their scores
        retrieval.items, question, reranker,
        candidates=s.rerank_candidates, top=s.rerank_candidates, max_words=s.rerank_max_words,
    )  # fmt: skip
    kept = scored[: s.rerank_top]  # what the answer prompt actually draws from
    context = build_context(kept, s.context_budget_tokens, get_token_counter(s))
    rerank_info = {r.fused.candidate.chunk_id: r for r in scored}
    in_context = {src.chunk_id: src.id for src in context.sources}
    candidates = []
    for f in retrieval.items:
        c = f.candidate
        rr = rerank_info.get(c.chunk_id)
        candidates.append(
            {
                "fused_rank": f.rank,
                "rrf_score": round(f.rrf_score, 5),
                "dense_rank": f.dense_rank,
                "dense_distance": _round(f.dense_distance, 4),
                "keyword_rank": f.keyword_rank,
                "keyword_score": _round(f.keyword_score, 4),
                "rerank_rank": rr.rank
                if rr
                else None,  # None: below the candidate cap, never reranked
                "kept_for_answer": bool(rr and rr.rank <= s.rerank_top),
                "rerank_score": _round(rr.rerank_score if rr else None, 3),
                "context_label": in_context.get(
                    c.chunk_id
                ),  # "S1".. if it reached the answer prompt
                "chunk_id": str(c.chunk_id),
                "document": c.source_uri,
                "section": c.section_path,
                "pages": _pages(c.page_start, c.page_end),
                "anchors": c.anchors,
                "text_preview": c.text[:preview_chars],
            }
        )
    return {
        "question": question,
        "mode": retrieval.mode,
        "keyword_query": retrieval.keywords,
        "standalone_question": retrieval.question,
        "queries": retrieval.queries,
        "hyde_passage": retrieval.hyde,
        "transform_cost_usd": retrieval.transform_cost_usd,
        "timings_ms": retrieval.timings_ms,
        "context_tokens": context.tokens,
        "context_sources": len(context.sources),
        "candidates": candidates,
    }
