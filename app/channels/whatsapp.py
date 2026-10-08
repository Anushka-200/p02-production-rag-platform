"""The WhatsApp conversation: who is this phone, what did they say, what do we answer.

One incoming message goes through these steps (`handle_message`):

  1. throttle      too many messages from this phone in a minute -> polite refusal
  2. de-duplicate  Twilio may deliver the same message twice (MessageSid): answer only once
  3. route         which company's assistant is this phone talking to?
                     - first contact: "join <keyword>" (the sandbox shares one number), or the
                       dedicated number the message was sent TO
                     - afterwards: remembered by a keyed HASH of the phone number (never the number)
  4. commands      HELP, HUMAN (ask for a person), "YES <name>" (consent to be contacted), STOP
                     (erase everything about this phone)
  5. question      public documents only (the same anonymous scope as the website widget), the same
                     daily budget per site, answers and cost stored with the widget tables

        "join acme"                     -> "Hi! I'm Acme's AI assistant ... send HELP for options"
        "When do you open?"             -> "We open at 8:30 every weekday. [S1]\\n\\nSources: Hours"
        "human"                         -> "... reply YES <your name> so we may contact you here"
        "yes Jo"                        -> lead saved with this number, handoff flagged for staff
        "stop"                          -> everything about this phone is deleted
"""

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg

from app.budget import record as record_spend
from app.budget import run_cost
from app.budget import status as budget_status
from app.channels.twilio import normalise_number, phone_hash
from app.config import Settings
from app.db.tenant import tenant_connection
from app.rag import ask
from app.ratelimit import RateLimiter
from app.retrieval.search import Scope
from app.widget import (
    WidgetError,
    add_lead,
    add_message,
    create_session,
    record_usage,
    request_handoff,
    session_exists,
    spent_today,
)

logger = logging.getLogger(__name__)

MAX_BODY = 1000
PER_MINUTE_PER_PHONE = 10
JOIN = re.compile(r"^\s*(?:join|start)\s+([\w-]{2,40})\s*$", re.IGNORECASE)
YES = re.compile(r"^\s*yes\s+(.{1,100})$", re.IGNORECASE)
HUMAN_WORDS = {"human", "person", "agent", "operator", "representative"}

DISCLOSURE = "I'm an AI assistant. I answer from the company's public information."
HELP = (
    "You can ask me a question in your own words.\n"
    "HUMAN - ask for a person to follow up\n"
    "STOP - delete everything we hold about this chat"
)
NO_CHANNEL = "Hi! To start, send: START <company code>"
UNKNOWN_CODE = "Sorry, I don't recognise that code. Please check it and try again."
SLOW_DOWN = "You are sending messages very fast. Please wait a minute and try again."
BUSY = "The assistant is unavailable right now. Reply HUMAN and a person will get back to you."
CONSENT_ASK = (
    "We'll pass this chat to a person. To let them contact you on this number, reply: "
    "YES <your name>. We keep your name and number only for that. (Or just keep chatting.)"
)
ERASED = (
    "Done. Everything we held about this chat has been deleted. Send START <code> to begin again."
)


@dataclass
class Channel:
    id: uuid.UUID
    org_id: uuid.UUID
    site_id: uuid.UUID
    name: str
    budget: float
    enabled: bool


@dataclass
class Contact:
    phone_hash: str
    channel_id: uuid.UUID
    widget_session_id: uuid.UUID | None
    state: str
    idle_hours: float


@dataclass
class Deps:
    settings: Settings
    open_connection: Callable[[], psycopg.Connection]
    state: dict[str, Any]  # loaded embedder and reranker
    limiter: RateLimiter
    complete_fn: Callable[..., Any]


# ---- routing tables (no RLS: they hold only a hashed phone and ids) ----------------------------
_CHANNEL_SQL = (
    "SELECT c.id, c.org_id, c.site_id, s.name, s.daily_budget_usd, (c.enabled AND s.enabled) "
    "FROM whatsapp_channels c JOIN sites s ON s.id = c.site_id WHERE "
)


def _channel(row: Any) -> Channel | None:
    return Channel(row[0], row[1], row[2], row[3], float(row[4]), bool(row[5])) if row else None


def channel_by_keyword(conn: psycopg.Connection, keyword: str) -> Channel | None:
    return _channel(
        conn.execute(_CHANNEL_SQL + "lower(c.keyword) = lower(%s)", (keyword,)).fetchone()
    )


def channel_by_number(conn: psycopg.Connection, to_number: str) -> Channel | None:
    if not to_number:
        return None
    return _channel(
        conn.execute(_CHANNEL_SQL + "c.to_number = %s", (normalise_number(to_number),)).fetchone()
    )


def channel_by_id(conn: psycopg.Connection, channel_id: uuid.UUID) -> Channel | None:
    return _channel(conn.execute(_CHANNEL_SQL + "c.id = %s", (channel_id,)).fetchone())


def first_delivery(conn: psycopg.Connection, message_sid: str) -> bool:
    """True the first time a MessageSid is seen, False for Twilio's redelivery."""
    if not message_sid:
        return True
    row = conn.execute(
        "INSERT INTO whatsapp_seen (message_sid) VALUES (%s) ON CONFLICT DO NOTHING RETURNING 1",
        (message_sid,),
    ).fetchone()
    return row is not None


def get_contact(conn: psycopg.Connection, phash: str) -> Contact | None:
    row = conn.execute(
        "SELECT phone_hash, channel_id, widget_session_id, state, "
        "extract(epoch FROM now() - last_seen_at) / 3600 "
        "FROM whatsapp_contacts WHERE phone_hash = %s",
        (phash,),
    ).fetchone()
    return Contact(row[0], row[1], row[2], row[3], float(row[4])) if row else None


def save_contact(conn: psycopg.Connection, phash: str, channel: Channel) -> None:
    """First contact, or the phone joined a different company: start clean with that channel."""
    conn.execute(
        "INSERT INTO whatsapp_contacts (phone_hash, channel_id, org_id, site_id) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (phone_hash) DO UPDATE SET channel_id = EXCLUDED.channel_id, "
        "org_id = EXCLUDED.org_id, site_id = EXCLUDED.site_id, widget_session_id = NULL, "
        "state = 'chat', last_seen_at = now()",
        (phash, channel.id, channel.org_id, channel.site_id),
    )


def update_contact(
    conn: psycopg.Connection,
    phash: str,
    *,
    session_id: uuid.UUID | None = None,
    state: str | None = None,
) -> None:
    conn.execute(
        "UPDATE whatsapp_contacts SET last_seen_at = now(), "
        "widget_session_id = coalesce(%s, widget_session_id), state = coalesce(%s, state) "
        "WHERE phone_hash = %s",
        (session_id, state, phash),
    )


# ---- the conversation ----------------------------------------------------------------------------
def handle_message(
    deps: Deps, from_number: str, to_number: str, body: str, message_sid: str
) -> str:
    """Returns the reply text, or "" when nothing should be sent (a duplicate delivery)."""
    s = deps.settings
    if s.jwt_secret is None:
        raise RuntimeError("JWT_SECRET is not set (it also keys the phone-number hash)")
    text = body.replace("\x00", "").strip()[:MAX_BODY]
    phash = phone_hash(s.jwt_secret.get_secret_value(), from_number)

    allowed, _ = deps.limiter.hit(f"wa:{phash}", PER_MINUTE_PER_PHONE)
    if not allowed:
        return SLOW_DOWN

    conn = deps.open_connection()
    try:
        if not first_delivery(conn, message_sid):
            return ""
        contact = get_contact(conn, phash)
        channel = channel_by_id(conn, contact.channel_id) if contact else None

        joined = JOIN.match(text)
        if joined or contact is None:
            return _first_contact(
                deps, conn, phash, contact, channel, joined.group(1) if joined else None, to_number
            )
        if channel is None or not channel.enabled:
            return BUSY
        scope = Scope(channel.org_id, None, channel.site_id)
        return _converse(
            deps, conn, phash, contact, channel, scope, normalise_number(from_number), text
        )
    finally:
        conn.close()


def _first_contact(
    deps: Deps, conn: psycopg.Connection, phash: str, contact: Contact | None,
    current: Channel | None, keyword: str | None, to_number: str,
) -> str:  # fmt: skip
    channel = channel_by_keyword(conn, keyword) if keyword else channel_by_number(conn, to_number)
    if channel is None:
        return UNKNOWN_CODE if keyword else NO_CHANNEL
    if not channel.enabled:
        return BUSY
    if contact is not None and current is not None and current.id != channel.id:
        # Switching company: the old conversation must not be left behind, because after the
        # switch the user has no way left to ask for its deletion.
        _erase(deps, conn, phash, contact, Scope(current.org_id, None, current.site_id))
    save_contact(conn, phash, channel)
    welcome = f"Hi! This is {channel.name}. {DISCLOSURE}\n\n{HELP}"
    return welcome if keyword else f"{welcome}\n\nPlease send your question again."


def _converse(
    deps: Deps, conn: psycopg.Connection, phash: str, contact: Contact, channel: Channel,
    scope: Scope, number: str, text: str,
) -> str:  # fmt: skip
    lowered = text.lower()
    if lowered in {"stop", "unsubscribe", "delete"}:
        return _erase(deps, conn, phash, contact, scope)
    if lowered in {"help", "?"}:
        return HELP
    if not text:
        return HELP

    with tenant_connection(deps.open_connection, scope, write=True) as w:
        session_id = _session(w, scope, contact, deps.settings)
        update_contact(w, phash, session_id=session_id)

        if lowered in HUMAN_WORDS:
            request_handoff(w, scope, session_id, "asked for a person on WhatsApp")
            update_contact(w, phash, state="awaiting_consent")
            return CONSENT_ASK
        consent = YES.match(text)
        if consent and contact.state == "awaiting_consent":
            try:
                add_lead(
                    w, scope, session_id, name=consent.group(1), email=None, phone=number,
                    message="WhatsApp handoff", consent=True,
                )  # fmt: skip
            except WidgetError as exc:
                return str(exc)
            update_contact(w, phash, state="chat")
            return "Thank you. A person will contact you on this number."

        if (
            spent_today(w, scope) >= channel.budget
            or budget_status(w, scope, deps.settings).exhausted
        ):
            logger.warning("site budget reached", extra={"site_id": str(scope.site_id)})
            return BUSY
        add_message(w, scope, session_id, "user", text)

    with tenant_connection(deps.open_connection, scope) as r:  # the slow part: no write lock held
        result = ask(
            r, deps.state["embedder"], deps.state["reranker"], text, scope,
            settings=deps.settings, complete_fn=deps.complete_fn,
        )  # fmt: skip
    answer = result.answer
    with tenant_connection(deps.open_connection, scope, write=True) as w:
        citations = [
            {"id": c.id, "document": c.document, "section": c.section, "pages": c.pages}
            for c in answer.citations
        ]
        add_message(
            w,
            scope,
            session_id,
            "assistant",
            answer.text,
            citations=citations,
            refused=answer.refused,
        )
        cost = run_cost(result)
        record_usage(w, scope, cost)
        record_spend(
            w, scope, deps.settings, cost_usd=cost, input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
        )  # fmt: skip
    if answer.refused:
        return f"{answer.text}\n\nReply HUMAN to talk to a person."
    sources = sorted({str(c["section"] or c["document"]) for c in citations})
    return answer.text + (f"\n\nSources: {' · '.join(sources)}" if sources else "")


def _session(
    w: psycopg.Connection, scope: Scope, contact: Contact, settings: Settings
) -> uuid.UUID:
    """The contact's current widget session, or a fresh one after a long silence."""
    if (
        contact.widget_session_id
        and contact.idle_hours < settings.whatsapp_session_hours
        and session_exists(w, scope, contact.widget_session_id)
    ):
        return contact.widget_session_id
    return create_session(w, scope, channel="whatsapp")


def _erase(deps: Deps, conn: psycopg.Connection, phash: str, contact: Contact, scope: Scope) -> str:
    """STOP: delete the phone's routing row and its whole conversation, leads and messages."""
    if contact.widget_session_id:
        with tenant_connection(deps.open_connection, scope, write=True) as w:
            w.execute(
                "DELETE FROM feedback WHERE session_id = %s AND org_id = %s",
                (contact.widget_session_id, scope.org_id),
            )
            w.execute(  # messages and leads go with the session (ON DELETE CASCADE)
                "DELETE FROM widget_sessions WHERE id = %s AND org_id = %s AND site_id = %s",
                (contact.widget_session_id, scope.org_id, scope.site_id),
            )
    conn.execute("DELETE FROM whatsapp_contacts WHERE phone_hash = %s", (phash,))
    return ERASED
