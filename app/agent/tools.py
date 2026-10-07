"""The agent's tools: four READ-ONLY ways to search, and nothing else.

    search       meaning + keywords (hybrid), the default
    exact        keywords only: ids, error messages, names, numbers
    broad        several rephrasings and an imagined answer: for vaguely worded questions
    connections  follows facts between named things: "what is connected to X?" questions

"Least agency": the agent may only choose a tool from this list and supply a search STRING. The
organisation and user are fixed in the ToolBox by the backend (from the verified login); a tool has
no parameter through which the model could name another tenant, run SQL, call a URL or write
anything. A tool that fails returns no results, it never raises into the planner.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg

from app.config import Settings
from app.embeddings import Embedder
from app.llm import complete
from app.retrieval.expand import retrieve_expanded
from app.retrieval.rerank import Reranked, Reranker, rerank
from app.retrieval.search import Scope

logger = logging.getLogger(__name__)

TOOLS: dict[str, str] = {
    "search": "meaning and keyword search over the documents. Use it by default.",
    "exact": "keyword-only search. Use it when the question contains an exact term: an id, an "
    "error message, a product or file name, a number.",
    "broad": "tries several rephrasings plus an imagined answer. Use it when the question is "
    "vaguely worded or a plain search found nothing.",
    "connections": "follows facts between named things (teams, tools, processes). Use it for "
    "questions about how things are connected: who uses, owns, escalates to, depends on.",
}
MIN_QUERY, MAX_QUERY = 3, 300

_FLAGS_OFF = {
    "query_rewrite": False, "multi_query": 0, "hyde": False, "summary_tree": False,
    "graph_search": False,
}  # fmt: skip
_TOOL_SETTINGS: dict[str, dict[str, Any]] = {
    "search": {"retrieval_mode": "hybrid"},
    "exact": {"retrieval_mode": "keyword"},
    "broad": {"retrieval_mode": "hybrid", "multi_query": 3, "hyde": True},
    "connections": {"retrieval_mode": "hybrid", "graph_search": True},
}


@dataclass
class ToolResult:
    hits: list[Reranked]
    cost_usd: float = 0.0
    error: str = ""


class ToolBox:
    """Everything a tool needs, fixed by the backend. The model never sees or sets any of it."""

    def __init__(
        self,
        conn: psycopg.Connection,
        embedder: Embedder,
        reranker: Reranker | None,
        scope: Scope,
        settings: Settings,
        complete_fn: Callable[..., Any] = complete,
    ) -> None:
        self._conn, self._embedder, self._reranker = conn, embedder, reranker
        self._scope, self._settings, self._complete = scope, settings, complete_fn

    @staticmethod
    def valid_query(query: object) -> str | None:
        if not isinstance(query, str):
            return None
        text = " ".join(query.replace("\x00", "").split())
        return text if MIN_QUERY <= len(text) <= MAX_QUERY else None

    def run(self, tool: str, query: str, *, top: int = 5) -> ToolResult:
        """Run one tool. Unknown tool or unusable query = a refusal, never an exception."""
        if tool not in TOOLS:
            return ToolResult([], error=f"unknown tool {tool!r}")
        text = self.valid_query(query)
        if text is None:
            return ToolResult([], error="unusable query")
        s = self._settings.model_copy(update={**_FLAGS_OFF, **_TOOL_SETTINGS[tool]})
        try:
            found = retrieve_expanded(
                self._conn, self._embedder, text, self._scope, settings=s,
                complete_fn=self._complete,
            )  # fmt: skip
            hits = rerank(
                found.items, text, self._reranker, candidates=s.rerank_candidates, top=top,
                max_words=s.rerank_max_words,
            )  # fmt: skip
        except Exception:
            logger.exception("agent tool failed", extra={"tool": tool})
            return ToolResult([], error="tool failed")
        return ToolResult(hits, cost_usd=found.transform_cost_usd)
