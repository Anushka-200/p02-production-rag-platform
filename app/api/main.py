"""The HTTP API. Phase 2: chat over Server-Sent Events and the retrieval inspector.

Run: uv run uvicorn app.api.main:app --port 8001
"""

import logging
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psycopg
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

from app.admin import list_admin_documents, list_chunks, list_jobs
from app.api.auth import Principal, current_principal
from app.api.security import TokenError, create_token, hash_password, verify_password
from app.api.sse import done_payload as _done_payload
from app.api.sse import sse
from app.api.whatsapp import build_whatsapp_router
from app.api.widget import WidgetContext, build_widget_router, register_cors
from app.channels.whatsapp import Deps as WhatsAppDeps
from app.chat_history import (
    HistoryError,
    add_message,
    conversation_exists,
    create_conversation,
    delete_conversation,
    get_messages,
    list_conversations,
    recent_messages,
)
from app.config import Settings, get_settings
from app.db.tenant import check_role_is_restricted, tenant_connection
from app.documents import DocumentError, NotAllowed, list_documents, set_visibility
from app.embeddings import Embedder, get_embedder, get_token_counter
from app.ingest.upload import UploadError, enqueue_upload
from app.llm import complete
from app.memory import (
    delete_all_memories,
    delete_memory,
    extract_facts,
    format_memory_block,
    list_memories,
    relevant_memories,
    store_facts,
)
from app.rag import ask_stream
from app.ratelimit import RateLimiter, default_limiter
from app.retrieval.inspect import inspect_retrieval
from app.retrieval.rerank import Reranker, get_reranker
from app.semantic_cache import make_writer
from app.widget import list_handoffs, list_leads, resolve_handoff, transcript

logger = logging.getLogger(__name__)
WIDGET_DIR = Path(__file__).resolve().parents[2] / "widget"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)  # who is asking is NOT in the body
    conversation_id: uuid.UUID | None = None  # omit to start a new conversation

    @field_validator("question")
    @classmethod
    def no_nul_bytes(cls, v: str) -> str:
        return v.replace("\x00", "")  # Postgres text cannot hold NUL; it would be a 500


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=200)


# Hash of a throwaway password: checked when the email is unknown, so "no such user" and "wrong
# password" take the same time and answer the same way (no way to find out who has an account).
_DUMMY_HASH = hash_password("not-a-real-password")


class VisibilityRequest(BaseModel):
    visibility: str = Field(pattern="^(private|org|public)$")


class InspectRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)  # to try a rewrite
    mode: str | None = Field(default=None, pattern="^(dense|keyword|hybrid)$")


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
    memory_complete_fn: Callable[..., Any] | None = None,
    queue: Any = None,
    limiter: RateLimiter | None = None,
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

    cache_writer = make_writer(open_connection, s) if s.semantic_cache else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if connect is None:  # tests inject their own connections; real runs must be restricted
            with psycopg.connect(s.database_url("app")) as probe:
                check_role_is_restricted(probe)
        if not state["loaded"]:  # models take ~20 s to load: do it once, at startup
            state["embedder"], state["reranker"] = get_embedder(s), get_reranker(s)
            _warm_up(
                s, state["embedder"], state["reranker"]
            )  # the first call of each model is slow
        yield

    app = FastAPI(title="P02 RAG platform", lifespan=lifespan)

    if s.webhook_only:
        # For a public tunnel during WhatsApp testing: nothing but the webhook (and health) may be
        # reached from outside, so demo accounts, documents and the admin API stay private.
        @app.middleware("http")
        async def only_the_webhook(request: Request, call_next: Callable[..., Any]) -> Any:
            path = request.url.path
            if path == "/health" or path.startswith("/channels/whatsapp/"):
                return await call_next(request)
            return JSONResponse({"detail": "not found"}, status_code=404)

    # The embeddable widget: its own router (anonymous visitors) and per-site CORS.
    widget_ctx = WidgetContext(
        settings=s, open_connection=open_connection, state=state,
        limiter=limiter or default_limiter(s.redis_url),
        stream_fn=stream_fn, complete_fn=complete_fn,
    )  # fmt: skip
    register_cors(app, widget_ctx)
    app.include_router(build_widget_router(widget_ctx))
    app.include_router(
        build_whatsapp_router(
            WhatsAppDeps(
                settings=s,
                open_connection=open_connection,
                state=state,
                limiter=widget_ctx.limiter,
                complete_fn=complete_fn,
            )  # fmt: skip
        )
    )

    @app.get("/widget.js", include_in_schema=False)
    def widget_script() -> FileResponse:
        return FileResponse(
            WIDGET_DIR / "p02-widget.js", media_type="application/javascript",
            headers={"Cache-Control": "public, max-age=300"},
        )  # fmt: skip

    def principal(request: Request) -> Principal:
        return current_principal(request, s, open_connection)

    def admin(who: Principal = Depends(principal)) -> Principal:
        if who.role != "admin":
            raise HTTPException(403, "admin only")
        return who

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/auth/login")
    def login(body: LoginRequest) -> dict[str, str]:
        conn = open_connection()
        try:
            rows = conn.execute("SELECT * FROM login_lookup(%s)", (body.email.strip(),)).fetchall()
        finally:
            conn.close()
        # The same email may exist in several organisations (different passwords). Check every
        # match, so an unknown email costs the same time as a known one; log in only on EXACTLY one
        # match, never guess which organisation was meant.
        candidates = rows or [(None, None, None, _DUMMY_HASH)]
        matches = [
            r for r in candidates if verify_password(body.password, r[3]) and r[0] is not None
        ]
        if len(matches) != 1:
            raise HTTPException(401, "wrong email or password")
        user_id, org_id, role, _ = matches[0]
        try:
            token = create_token(s, user_id, org_id, role)
        except TokenError as exc:
            logger.error("login unavailable: %s", exc)
            raise HTTPException(503, "login is not configured") from exc
        return {"access_token": token, "token_type": "bearer"}

    def job_queue() -> Any:
        if queue is not None:
            return queue
        from redis import Redis
        from rq import Queue

        return Queue(s.ingest_queue, connection=Redis.from_url(s.redis_url))

    @app.post("/documents", status_code=202)
    def upload_document(
        file: UploadFile = File(...),  # noqa: B008
        visibility: str = Form("private", pattern="^(private|org|public)$"),
        who: Principal = Depends(principal),
    ) -> dict[str, Any]:
        """Accept a file for ingestion. The owner is the logged-in user; it starts private."""
        if who.scope.user_id is None:  # anonymous visitors can never upload
            raise HTTPException(401, "login required")
        if visibility == "public" and who.role != "admin":
            raise HTTPException(403, "only an organisation admin can upload public documents")
        limit = s.max_upload_mb * 1_000_000
        raw = file.file.read(
            limit + 1
        )  # read at most one byte over the limit, never the whole stream
        try:
            with tenant_connection(open_connection, who.scope, write=True) as conn:
                result = enqueue_upload(
                    conn, job_queue(), org_id=who.scope.org_id, owner_id=who.scope.user_id,
                    visibility=visibility, filename=file.filename or "upload", raw=raw, settings=s,
                )  # fmt: skip
        except UploadError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "job_id": str(result.job_id),
            "status": result.status,
            "duplicate": not result.created,
        }

    @app.get("/jobs/{job_id}")
    def job_status(job_id: uuid.UUID, who: Principal = Depends(principal)) -> dict[str, Any]:
        with tenant_connection(open_connection, who.scope) as conn:
            row = conn.execute(
                "SELECT status, stage, error, attempts, result FROM ingestion_jobs "
                "WHERE id = %s AND org_id = %s AND owner_id = %s",
                (job_id, who.scope.org_id, who.scope.user_id),
            ).fetchone()
        if row is None:
            raise HTTPException(404, "job not found")
        status, stage, error, attempts, result = row
        return {
            "status": status,
            "stage": stage,
            "error": error,
            "attempts": attempts,
            "result": result,
        }

    @app.get("/documents")
    def documents(who: Principal = Depends(principal)) -> list[dict[str, Any]]:
        with tenant_connection(open_connection, who.scope) as conn:
            return list_documents(conn, who.scope)

    @app.patch("/documents/{document_id}/visibility")
    def document_visibility(
        document_id: uuid.UUID, body: VisibilityRequest, who: Principal = Depends(principal)
    ) -> dict[str, str]:
        try:
            with tenant_connection(open_connection, who.scope, write=True) as conn:
                set_visibility(
                    conn, who.scope, document_id, body.visibility, is_admin=who.role == "admin"
                )
        except NotAllowed as exc:
            raise HTTPException(403, str(exc)) from exc
        except DocumentError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"visibility": body.visibility}

    def remember(who: Principal, user_message_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        """Background step after the answer is sent: learn lasting facts from the user."""
        try:
            with tenant_connection(open_connection, who.scope, write=True) as conn:
                window = recent_messages(
                    conn, who.scope, conversation_id, max_turns=3, budget_tokens=800
                )
                known = [m["fact"] for m in list_memories(conn, who.scope)[:20]]
                facts, cost = extract_facts(
                    [m["content"] for m in window if m["role"] == "user"],
                    known,
                    complete_fn=memory_complete_fn or complete,
                )
                stored = store_facts(
                    conn, who.scope, state["embedder"], facts,
                    source_message_id=user_message_id, settings=s,
                )  # fmt: skip
            logger.info(
                "memory updated",
                extra={"added": stored.added, "merged": stored.merged, "cost_usd": round(cost, 6)},
            )
        except Exception:
            logger.exception("memory update failed")  # never affects the chat the user already got

    @app.post("/chat")
    def chat(body: ChatRequest, who: Principal = Depends(principal)) -> StreamingResponse:
        # (1) one short write transaction: the conversation, the user's message, the memories to use
        with tenant_connection(open_connection, who.scope, write=True) as conn:
            if body.conversation_id is not None:
                if not conversation_exists(conn, who.scope, body.conversation_id):
                    raise HTTPException(404, "conversation not found")
                conversation_id = body.conversation_id
            else:
                conversation_id = create_conversation(conn, who.scope, body.question)
            history = (  # earlier turns, read BEFORE this question is stored (used for rewriting)
                recent_messages(
                    conn,
                    who.scope,
                    conversation_id,
                    max_turns=s.history_max_turns,
                    budget_tokens=s.history_budget_tokens,
                )  # fmt: skip
                if s.query_rewrite and body.conversation_id is not None
                else []
            )
            user_message_id = add_message(conn, who.scope, conversation_id, "user", body.question)
            notes = (
                relevant_memories(conn, state["embedder"], body.question, who.scope, settings=s)
                if s.memory_enabled
                else []
            )

        def events() -> Iterator[str]:
            # (2) the answer, on a read-only connection that is limited to the tenant's rows
            # (opened here, not as a dependency: streaming outlives dependencies)
            try:
                with tenant_connection(open_connection, who.scope) as conn:
                    for event in ask_stream(
                        conn, state["embedder"], state["reranker"], body.question, who.scope,
                        settings=s, stream_fn=stream_fn, complete_fn=complete_fn,
                        memory_block=format_memory_block(notes), history=history,
                        cache_writer=cache_writer, use_cache=body.conversation_id is None,
                    ):  # fmt: skip
                        if event["type"] != "done":
                            yield sse(
                                event["type"], {k: v for k, v in event.items() if k != "type"}
                            )
                            continue
                        payload = _done_payload(event)
                        try:  # (3) save the answer; a failure here must not lose the answer
                            with tenant_connection(open_connection, who.scope, write=True) as w:
                                add_message(
                                    w, who.scope, conversation_id, "assistant", payload["text"],
                                    citations=payload["citations"], refused=payload["refused"],
                                )  # fmt: skip
                        except Exception:
                            logger.exception("could not save the assistant message")
                        payload["conversation_id"] = str(conversation_id)
                        yield sse("done", payload)
            except Exception:
                logger.exception("chat failed")
                yield sse(
                    "error", {"message": "something went wrong; please try again"}
                )  # no internals leaked

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=(
                BackgroundTask(remember, who, user_message_id, conversation_id)
                if s.memory_enabled
                else None
            ),
        )  # fmt: skip

    @app.get("/conversations")
    def conversations(who: Principal = Depends(principal)) -> list[dict[str, Any]]:
        with tenant_connection(open_connection, who.scope) as conn:
            return list_conversations(conn, who.scope)

    @app.get("/conversations/{conversation_id}")
    def conversation(
        conversation_id: uuid.UUID, who: Principal = Depends(principal)
    ) -> list[dict[str, Any]]:
        try:
            with tenant_connection(open_connection, who.scope) as conn:
                return get_messages(conn, who.scope, conversation_id)
        except HistoryError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/conversations/{conversation_id}")
    def remove_conversation(
        conversation_id: uuid.UUID, who: Principal = Depends(principal)
    ) -> dict[str, bool]:
        with tenant_connection(open_connection, who.scope, write=True) as conn:
            if not delete_conversation(conn, who.scope, conversation_id):
                raise HTTPException(404, "conversation not found")
        return {"deleted": True}

    @app.get("/memories")
    def memories(who: Principal = Depends(principal)) -> list[dict[str, Any]]:
        with tenant_connection(open_connection, who.scope) as conn:
            return list_memories(conn, who.scope)

    @app.delete("/memories/{memory_id}")
    def remove_memory(memory_id: uuid.UUID, who: Principal = Depends(principal)) -> dict[str, bool]:
        with tenant_connection(open_connection, who.scope, write=True) as conn:
            if not delete_memory(conn, who.scope, memory_id):
                raise HTTPException(404, "memory not found")
        return {"deleted": True}

    @app.delete("/memories")
    def remove_all_memories(who: Principal = Depends(principal)) -> dict[str, int]:
        with tenant_connection(open_connection, who.scope, write=True) as conn:
            return {"deleted": delete_all_memories(conn, who.scope)}

    @app.get("/admin/leads")
    def admin_leads(who: Principal = Depends(admin)) -> list[dict[str, Any]]:
        with tenant_connection(open_connection, who.scope) as conn:
            return list_leads(conn, who.scope)

    @app.get("/admin/handoffs")
    def admin_handoffs(who: Principal = Depends(admin)) -> list[dict[str, Any]]:
        with tenant_connection(open_connection, who.scope) as conn:
            return list_handoffs(conn, who.scope)

    @app.get("/admin/handoffs/{session_id}")
    def admin_handoff_transcript(
        session_id: uuid.UUID, who: Principal = Depends(admin)
    ) -> list[dict[str, Any]]:
        with tenant_connection(open_connection, who.scope) as conn:
            return transcript(conn, who.scope, session_id)

    @app.post("/admin/handoffs/{session_id}/resolve")
    def admin_resolve_handoff(
        session_id: uuid.UUID, who: Principal = Depends(admin)
    ) -> dict[str, bool]:
        with tenant_connection(open_connection, who.scope, write=True) as conn:
            if not resolve_handoff(conn, who.scope, session_id):
                raise HTTPException(404, "no such handoff")
        return {"resolved": True}

    @app.get("/me")
    def me(who: Principal = Depends(principal)) -> dict[str, str]:
        return {"email": who.email, "role": who.role}

    @app.get("/admin/documents")
    def admin_documents(who: Principal = Depends(admin)) -> list[dict[str, Any]]:
        with tenant_connection(open_connection, who.scope) as conn:
            return list_admin_documents(conn, who.scope)

    @app.get("/admin/jobs")
    def admin_jobs(who: Principal = Depends(admin)) -> list[dict[str, Any]]:
        with tenant_connection(open_connection, who.scope) as conn:
            return list_jobs(conn, who.scope)

    @app.get("/admin/documents/{document_id}/chunks")
    def admin_chunks(
        document_id: uuid.UUID,
        level: int | None = Query(None, ge=0, le=2),
        limit: int = Query(50, ge=1, le=200),
        who: Principal = Depends(admin),
    ) -> list[dict[str, Any]]:
        with tenant_connection(open_connection, who.scope) as conn:
            return list_chunks(conn, who.scope, document_id, level=level, limit=limit)

    @app.post("/admin/inspect")
    def inspect(body: InspectRequest, who: Principal = Depends(admin)) -> dict[str, Any]:
        with tenant_connection(open_connection, who.scope) as conn:
            return inspect_retrieval(
                conn,
                state["embedder"],
                state["reranker"],
                body.question,
                who.scope,
                settings=s,
                mode=body.mode,
                history=body.history,
                complete_fn=complete_fn,
            )

    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()
