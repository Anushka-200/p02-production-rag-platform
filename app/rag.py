"""The whole question-answering flow in one call: retrieve > rerank > assemble > answer."""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import psycopg

from app.answer import Answer, answer_question, answer_stream
from app.config import Settings, get_settings
from app.embeddings import Embedder, get_token_counter
from app.llm import complete
from app.retrieval.context import Context, build_context
from app.retrieval.rerank import Reranked, Reranker, rerank
from app.retrieval.retrieve import Retrieval, retrieve
from app.retrieval.search import Scope


@dataclass
class RagResult:
    question: str
    answer: Answer
    retrieval: Retrieval
    reranked: list[Reranked]
    context: Context
    timings_ms: dict[str, float] = field(default_factory=dict)


def ask(
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker | None,
    question: str,
    scope: Scope,
    *,
    settings: Settings | None = None,
    complete_fn: Callable[..., Any] = complete,
) -> RagResult:
    s = settings or get_settings()
    timings: dict[str, float] = {}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        result = fn()
        timings[name] = round((time.perf_counter() - started) * 1000, 1)
        return result

    retrieval = timed("retrieve_ms", lambda: retrieve(conn, embedder, question, scope, settings=s))
    reranked = timed(
        "rerank_ms",
        lambda: rerank(
            retrieval.items, question, reranker,
            candidates=s.rerank_candidates, top=s.rerank_top, max_words=s.rerank_max_words,
        ),
    )  # fmt: skip
    counter = get_token_counter(s)
    context = timed("context_ms", lambda: build_context(reranked, s.context_budget_tokens, counter))
    answer = timed("answer_ms", lambda: answer_question(question, context, complete_fn=complete_fn))
    return RagResult(question, answer, retrieval, reranked, context, timings)


def ask_stream(
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker | None,
    question: str,
    scope: Scope,
    *,
    settings: Settings | None = None,
    stream_fn: Callable[..., Any] | None = None,
    complete_fn: Callable[..., Any] = complete,
    memory_block: str = "",
) -> Iterator[dict[str, Any]]:
    """The streaming flow. Events: sources (as soon as retrieval is done), token..., done.

    The `done` event carries the validated answer and the timings, including time to first token.
    """
    from app.llm import LLMStream

    s = settings or get_settings()
    started = time.perf_counter()
    timings: dict[str, float] = {}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        t = time.perf_counter()
        result = fn()
        timings[name] = round((time.perf_counter() - t) * 1000, 1)
        return result

    retrieval = timed("retrieve_ms", lambda: retrieve(conn, embedder, question, scope, settings=s))
    reranked = timed(
        "rerank_ms",
        lambda: rerank(
            retrieval.items, question, reranker,
            candidates=s.rerank_candidates, top=s.rerank_top, max_words=s.rerank_max_words,
        ),
    )  # fmt: skip
    context = timed(
        "context_ms", lambda: build_context(reranked, s.context_budget_tokens, get_token_counter(s))
    )
    yield {
        "type": "sources",
        "sources": [
            {
                "id": x.id,
                "document": x.document,
                "section": x.section,
                "pages": x.pages,
                "anchors": x.anchors,
            }
            for x in context.sources
        ],
    }
    first_token_ms = None
    for event in answer_stream(
        question, context, stream_fn=stream_fn or LLMStream, complete_fn=complete_fn,
        memory_block=memory_block,
    ):  # fmt: skip
        if event["type"] == "token" and first_token_ms is None:
            first_token_ms = round((time.perf_counter() - started) * 1000, 1)
        if event["type"] == "done":
            timings["first_token_ms"] = first_token_ms if first_token_ms is not None else 0.0
            timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
            event["result"] = RagResult(
                question, event["answer"], retrieval, reranked, context, timings
            )
        yield event
