"""The whole question-answering flow in one call: retrieve > rerank > assemble > answer."""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import psycopg

from app.agent.loop import AgentResult, run_agent
from app.agent.tools import ToolBox
from app.answer import Answer, answer_question, answer_stream
from app.config import Settings, get_settings
from app.embeddings import Embedder, get_token_counter
from app.llm import complete
from app.retrieval.context import Context, build_context
from app.retrieval.expand import retrieve_expanded
from app.retrieval.parents import expand_to_parents
from app.retrieval.rerank import Reranked, Reranker, rerank
from app.retrieval.retrieve import Retrieval
from app.retrieval.search import Scope
from app.retrieval.transform import rewrite_query


@dataclass
class RagResult:
    question: str
    answer: Answer
    retrieval: Retrieval
    reranked: list[Reranked]
    context: Context
    timings_ms: dict[str, float] = field(default_factory=dict)
    agent: AgentResult | None = None


def _gather(
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker | None,
    question: str,
    scope: Scope,
    s: Settings,
    history: list[dict[str, str]] | None,
    complete_fn: Callable[..., Any],
    timed: Callable[[str, Callable[[], Any]], Any],
) -> tuple[Retrieval, list[Reranked], AgentResult | None, str]:
    """Find the evidence for a question. Returns (retrieval, ranked hits, agent run, question).

    With the agent switched on, a planner decides per question which tools to use; if the agent
    fails or finds nothing, the plain pipeline below runs, so turning the agent on can never leave
    a question without the search it would have had anyway."""
    if s.agent_enabled:
        try:
            standalone = question
            if s.query_rewrite and history:
                standalone = timed(
                    "rewrite_ms",
                    lambda: rewrite_query(
                        question, history, complete_fn=complete_fn, model=s.transform_model
                    ).text,
                )  # fmt: skip
            toolbox = ToolBox(conn, embedder, reranker, scope, s, complete_fn)
            agent = timed(
                "agent_ms",
                lambda: run_agent(
                    standalone, toolbox, complete_fn=complete_fn, model=s.agent_model,
                    max_rounds=s.agent_max_rounds, max_tool_calls=s.agent_max_tool_calls,
                    max_cost_usd=s.agent_max_cost_usd, per_sub=s.agent_per_sub,
                    total=s.rerank_top,
                ),
            )  # fmt: skip
            if agent.evidence:
                found = Retrieval(
                    [h.fused for h in agent.evidence], "agent", "", {}, question=standalone,
                    queries=[x.text for x in agent.sub_questions],
                    transform_cost_usd=agent.cost_usd,
                )  # fmt: skip
                return found, agent.evidence, agent, standalone
        except Exception:
            logging.getLogger(__name__).exception("agent failed; using the plain pipeline")
    retrieval = timed(
        "retrieve_ms",
        lambda: retrieve_expanded(
            conn, embedder, question, scope, history=history, settings=s, complete_fn=complete_fn
        ),
    )
    standalone = retrieval.question or question
    reranked = timed(
        "rerank_ms",
        lambda: rerank(
            retrieval.items, standalone, reranker,
            candidates=s.rerank_candidates, top=s.rerank_top, max_words=s.rerank_max_words,
        ),
    )  # fmt: skip
    return retrieval, reranked, None, standalone


def ask(
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker | None,
    question: str,
    scope: Scope,
    *,
    settings: Settings | None = None,
    complete_fn: Callable[..., Any] = complete,
    history: list[dict[str, str]] | None = None,
) -> RagResult:
    s = settings or get_settings()
    timings: dict[str, float] = {}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        result = fn()
        timings[name] = round((time.perf_counter() - started) * 1000, 1)
        return result

    retrieval, reranked, agent, question = _gather(
        conn, embedder, reranker, question, scope, s, history, complete_fn, timed
    )
    counter = get_token_counter(s)
    if s.parent_expansion:
        reranked = timed(
            "parents_ms",
            lambda: expand_to_parents(
                conn, reranked, scope, max_parent_tokens=s.parent_max_tokens, count=counter
            ),
        )  # fmt: skip
    context = timed("context_ms", lambda: build_context(reranked, s.context_budget_tokens, counter))
    answer = timed("answer_ms", lambda: answer_question(question, context, complete_fn=complete_fn))
    return RagResult(question, answer, retrieval, reranked, context, timings, agent)


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
    history: list[dict[str, str]] | None = None,
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

    retrieval, reranked, agent, question = _gather(
        conn, embedder, reranker, question, scope, s, history, complete_fn, timed
    )
    if s.parent_expansion:
        reranked = timed(
            "parents_ms",
            lambda: expand_to_parents(
                conn, reranked, scope, max_parent_tokens=s.parent_max_tokens,
                count=get_token_counter(s),
            ),
        )  # fmt: skip
    context = timed(
        "context_ms", lambda: build_context(reranked, s.context_budget_tokens, get_token_counter(s))
    )
    if agent is not None:
        yield {
            "type": "agent",
            "steps": [x.as_dict() for x in agent.steps],
            "stopped": agent.stopped,
        }
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
                question, event["answer"], retrieval, reranked, context, timings, agent
            )
        yield event
