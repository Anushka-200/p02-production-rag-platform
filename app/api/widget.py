"""HTTP endpoints for the embeddable chat widget (anonymous website visitors).

URL shape: /widget/{site_key}/<action>. The site key is in the PATH (not a header) because the
browser's CORS "preflight" request cannot carry custom headers, yet the server must already know
which site's allowed origins to check.

Every request goes through the same gate (`visitor`): site key exists and is enabled, the browser
Origin is on that site's allow-list, then a per-visitor rate limit. Visitors are anonymous: they
read PUBLIC documents only (see migrations 0006-0008) and their chats live in widget_* tables
keyed by site.
"""

import logging
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import psycopg
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.api.auth import Principal, normalise_origin, site_principal
from app.api.sse import done_payload, sse
from app.config import Settings
from app.db.tenant import tenant_connection
from app.rag import ask_stream
from app.ratelimit import RateLimiter
from app.widget import (
    WidgetError,
    add_feedback,
    add_lead,
    add_message,
    create_session,
    record_usage,
    request_handoff,
    session_exists,
    spent_today,
)

logger = logging.getLogger(__name__)

DISCLOSURE = (
    "You are chatting with an AI assistant. It answers from this company's public information; "
    "if it cannot help, you can ask for a person."
)
BUSY = (
    "The assistant is unavailable right now. Please leave your details and a person will get "
    "back to you."
)


class ChatBody(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: uuid.UUID | None = None

    @field_validator("question")
    @classmethod
    def no_nul(cls, v: str) -> str:
        return v.replace("\x00", "")


class LeadBody(BaseModel):
    session_id: uuid.UUID
    name: str = Field(max_length=200)
    email: str = Field(max_length=300)
    phone: str | None = Field(default=None, max_length=100)
    message: str | None = Field(default=None, max_length=2000)
    consent: bool = False


class HandoffBody(BaseModel):
    session_id: uuid.UUID
    reason: str | None = Field(default=None, max_length=600)


class FeedbackBody(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    rating: int
    comment: str | None = Field(default=None, max_length=1000)


@dataclass
class WidgetContext:
    settings: Settings
    open_connection: Callable[[], psycopg.Connection]
    state: dict[str, Any]  # holds the loaded embedder and reranker
    limiter: RateLimiter
    stream_fn: Callable[..., Any] | None = None
    complete_fn: Callable[..., Any] | None = None


CITATION_FIELDS = ("id", "document", "section", "pages", "updated")


def client_ip(request: Request, trust_proxy: bool) -> str:
    """The visitor's address. Behind a trusted proxy the real one is the first X-Forwarded-For."""
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def allowed_origins(ctx: WidgetContext, site_key: str) -> set[str]:
    """The origins a site key accepts (empty if the key is unknown or disabled)."""
    conn = ctx.open_connection()
    try:
        row = conn.execute(
            "SELECT allowed_origins, enabled FROM sites WHERE site_key = %s", (site_key,)
        ).fetchone()
    finally:
        conn.close()
    return {normalise_origin(o) for o in row[0]} if row and row[1] else set()


def register_cors(app: FastAPI, ctx: WidgetContext) -> None:
    """Per-site CORS. A browser lets a web page read a response from another server only if that
    server says so; here the answer depends on WHICH site key is in the URL."""

    @app.middleware("http")
    async def widget_cors(request: Request, call_next: Callable[..., Any]) -> Response:
        parts = request.url.path.split("/")  # ["", "widget", "<key>", "<action>"]
        if len(parts) < 4 or parts[1] != "widget":
            return await call_next(request)  # type: ignore[no-any-return,unused-ignore]
        origin = request.headers.get("origin")
        ok = bool(origin) and normalise_origin(origin or "") in await run_in_threadpool(
            allowed_origins, ctx, parts[2]
        )
        cors = (
            {
                "Access-Control-Allow-Origin": origin or "",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "content-type",
                "Access-Control-Max-Age": "600",
                "Vary": "Origin",
            }
            if ok
            else {"Vary": "Origin"}
        )
        if request.method == "OPTIONS":  # the browser's permission check before a real request
            return Response(status_code=204 if ok else 403, headers=cors)
        response: Response = await call_next(request)
        response.headers.update(cors)
        return response


def build_widget_router(ctx: WidgetContext) -> APIRouter:
    router = APIRouter(prefix="/widget")
    s = ctx.settings

    def visitor(site_key: str, request: Request) -> Principal:
        """The gate every widget request passes: key, origin, rate limit."""
        who = site_principal(site_key, request.headers.get("origin"), ctx.open_connection)
        assert who.site is not None
        allowed, retry = ctx.limiter.hit(
            f"{who.site.id}:{client_ip(request, s.trust_proxy)}", who.site.rate_limit_per_minute
        )
        if not allowed:
            raise HTTPException(429, "too many requests, please slow down",
                                headers={"Retry-After": str(retry)})  # fmt: skip
        return who

    @router.get("/{site_key}/config")
    def config(site_key: str, request: Request) -> dict[str, Any]:
        who = visitor(site_key, request)
        assert who.site is not None
        return {"name": who.site.name, "disclosure": DISCLOSURE, "handoff": True}

    @router.post("/{site_key}/chat")
    def chat(site_key: str, body: ChatBody, request: Request) -> StreamingResponse:
        who = visitor(site_key, request)
        assert who.site is not None
        scope = who.scope
        with tenant_connection(ctx.open_connection, scope, write=True) as conn:
            if spent_today(conn, scope) >= who.site.daily_budget_usd:
                logger.warning("site budget reached", extra={"site_id": str(who.site.id)})
                raise HTTPException(429, BUSY)
            if body.session_id is not None and session_exists(conn, scope, body.session_id):
                session_id = body.session_id
            else:  # a new visitor, or a session id that is not theirs: start a fresh session
                session_id = create_session(conn, scope)
            add_message(conn, scope, session_id, "user", body.question)

        def events() -> Iterator[str]:
            try:
                with tenant_connection(ctx.open_connection, scope) as conn:
                    for event in ask_stream(
                        conn, ctx.state["embedder"], ctx.state["reranker"], body.question, scope,
                        settings=s, stream_fn=ctx.stream_fn,
                        complete_fn=ctx.complete_fn or _complete(),
                    ):  # fmt: skip
                        if event["type"] in {"sources", "agent"}:
                            continue  # the agent trace is internal; citations come at the end
                        if event["type"] != "done":
                            yield sse(
                                event["type"], {k: v for k, v in event.items() if k != "type"}
                            )
                            continue
                        full = done_payload(event)
                        message_id = None
                        try:
                            with tenant_connection(ctx.open_connection, scope, write=True) as w:
                                message_id = add_message(
                                    w, scope, session_id, "assistant", full["text"],
                                    citations=full["citations"], refused=full["refused"],
                                )  # fmt: skip
                                record_usage(w, scope, full["cost_usd"])
                        except Exception:
                            logger.exception("could not save the widget answer")
                        yield sse("done", {
                            "text": full["text"],
                            "refused": full["refused"],
                            "citations": [
                                {k: c.get(k) for k in CITATION_FIELDS}
                                for c in full["citations"]
                            ],
                            "session_id": str(session_id),
                            "message_id": str(message_id) if message_id else None,
                            "suggest_handoff": full["refused"],
                        })  # fmt: skip
            except Exception:
                logger.exception("widget chat failed")
                yield sse("error", {"message": "something went wrong; please try again"})

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )  # fmt: skip

    def _write(
        site_key: str, request: Request, action: Callable[[psycopg.Connection, Principal], Any]
    ) -> Any:
        who = visitor(site_key, request)
        try:
            with tenant_connection(ctx.open_connection, who.scope, write=True) as conn:
                return action(conn, who)
        except WidgetError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/{site_key}/lead")
    def lead(site_key: str, body: LeadBody, request: Request) -> dict[str, Any]:
        lead_id = _write(
            site_key, request,
            lambda conn, who: add_lead(
                conn, who.scope, body.session_id, name=body.name, email=body.email,
                phone=body.phone, message=body.message, consent=body.consent,
            ),
        )  # fmt: skip
        return {"saved": True, "id": str(lead_id)}

    @router.post("/{site_key}/handoff")
    def handoff(site_key: str, body: HandoffBody, request: Request) -> dict[str, Any]:
        ok = _write(
            site_key, request,
            lambda conn, who: request_handoff(conn, who.scope, body.session_id, body.reason),
        )  # fmt: skip
        if not ok:
            raise HTTPException(422, "unknown session")
        return {
            "requested": True,
            "message": "Thanks. A person will follow up; please leave your contact details.",
        }

    @router.post("/{site_key}/feedback")
    def feedback(site_key: str, body: FeedbackBody, request: Request) -> dict[str, bool]:
        _write(
            site_key, request,
            lambda conn, who: add_feedback(
                conn, who.scope, body.session_id, body.message_id, body.rating, body.comment
            ),
        )  # fmt: skip
        return {"saved": True}

    return router


def _complete() -> Callable[..., Any]:
    from app.llm import complete

    return complete
