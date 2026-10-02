"""The HTTP API. Phase 2: chat over Server-Sent Events and the retrieval inspector.

Run: uv run uvicorn app.api.main:app --port 8001
"""

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.auth import Principal, dev_principal
from app.config import Settings, get_settings
from app.embeddings import Embedder, get_embedder, get_token_counter
from app.llm import complete
from app.rag import ask_stream
from app.retrieval.inspect import inspect_retrieval
from app.retrieval.rerank import Reranker, get_reranker

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)  # who is asking is NOT in the body


class InspectRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str | None = Field(default=None, pattern="^(dense|keyword|hybrid)$")


def sse(event: str, data: dict[str, Any]) -> str:
    """One Server-Sent Event: the browser (or curl) receives it the moment it is written."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _done_payload(event: dict[str, Any]) -> dict[str, Any]:
    """What the browser gets at the end of the stream: the validated answer and what it cost."""
    a = event["answer"]
    return {
        "text": a.text,
        "refused": a.refused,
        "reason": a.reason,
        "replaced": event["replaced"],  # True: the streamed text was corrected after validation
        "citations": [vars(c) for c in a.citations],
        "model": a.model,
        "cost_usd": round(a.cost_usd, 6),
        "llm_calls": a.llm_calls,
        "timings_ms": event["result"].timings_ms,
    }


def _warm_up(s: Settings, embedder: Embedder, reranker: Reranker | None) -> None:
    """Run each model once at startup, so the first USER request does not pay the cold-start cost
    (measured: rerank 2 s cold vs ~1 s warm, tokenizer load 0.7 s)."""
    embedder.embed_query("warm up")
    if reranker is not None:
        reranker.score("warm up", ["warm up text"])
    get_token_counter(s)("warm up")


def create_app(
    settings: Settings | None = None,
    *,
    connect: Callable[[], psycopg.Connection] | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    stream_fn: Callable[..., Any] | None = None,
    complete_fn: Callable[..., Any] = complete,
) -> FastAPI:
    """Everything the API needs can be injected, so tests run without models, network or a key."""
    s = settings or get_settings()
    state: dict[str, Any] = {
        "embedder": embedder,
        "reranker": reranker,
        "loaded": embedder is not None,
    }

    def open_connection() -> psycopg.Connection:
        if connect is not None:
            return connect()
        return psycopg.connect(s.database_url("app"), autocommit=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not state["loaded"]:  # models take ~20 s to load: do it once, at startup
            state["embedder"], state["reranker"] = get_embedder(s), get_reranker(s)
            _warm_up(
                s, state["embedder"], state["reranker"]
            )  # the first call of each model is slow
        yield

    app = FastAPI(title="P02 RAG platform", lifespan=lifespan)

    def principal(request: Request) -> Principal:
        return dev_principal(request, s, open_connection)

    def admin(who: Principal = Depends(principal)) -> Principal:
        if who.role != "admin":
            raise HTTPException(403, "admin only")
        return who

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(body: ChatRequest, who: Principal = Depends(principal)) -> StreamingResponse:
        def events() -> Iterator[str]:
            conn = (
                open_connection()
            )  # opened here, not as a dependency: streaming outlives dependencies
            try:
                for event in ask_stream(
                    conn, state["embedder"], state["reranker"], body.question, who.scope,
                    settings=s, stream_fn=stream_fn, complete_fn=complete_fn,
                ):  # fmt: skip
                    if event["type"] == "done":
                        yield sse("done", _done_payload(event))
                    else:
                        yield sse(event["type"], {k: v for k, v in event.items() if k != "type"})
            except Exception:
                logger.exception("chat failed")
                yield sse(
                    "error", {"message": "something went wrong; please try again"}
                )  # no internals leaked
            finally:
                conn.close()

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )  # fmt: skip

    @app.post("/admin/inspect")
    def inspect(body: InspectRequest, who: Principal = Depends(admin)) -> dict[str, Any]:
        conn = open_connection()
        try:
            return inspect_retrieval(
                conn,
                state["embedder"],
                state["reranker"],
                body.question,
                who.scope,
                settings=s,
                mode=body.mode,
            )
        finally:
            conn.close()

    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()
