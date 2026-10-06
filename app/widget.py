"""Data for the embeddable widget: visitor sessions, messages, leads, handoffs, feedback, usage.

Everything here takes a connection that is ALREADY scoped (app/db/tenant.py): for a visitor the
scope carries the org and the site, and row-level security allows only that site's rows. The WHERE
clauses repeat the org and site, a second lock.

A visitor has no account. Their identity is a random session id the browser keeps; it is only a
bearer key for THEIR OWN session's rows on THEIR site.
"""

import json
import re
import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.retrieval.search import Scope

_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s.]{2,}$")
MAX_TEXT = {"name": 120, "phone": 40, "message": 1000, "reason": 300, "comment": 500}


class WidgetError(Exception):
    """A clear, user-safe reason a widget request was refused."""


def _need_site(scope: Scope) -> uuid.UUID:
    if scope.site_id is None:
        raise WidgetError("this action needs a site")
    return scope.site_id


# ---- sessions and messages ------------------------------------------------------------------
def create_session(conn: psycopg.Connection, scope: Scope, channel: str = "web") -> uuid.UUID:
    row = conn.execute(
        "INSERT INTO widget_sessions (org_id, site_id, channel) VALUES (%s, %s, %s) RETURNING id",
        (scope.org_id, _need_site(scope), channel),
    ).fetchone()
    return row[0]  # type: ignore[index]


def session_exists(conn: psycopg.Connection, scope: Scope, session_id: uuid.UUID) -> bool:
    row = conn.execute(
        "UPDATE widget_sessions SET last_seen_at = now() "
        "WHERE id = %s AND org_id = %s AND site_id = %s RETURNING id",
        (session_id, scope.org_id, _need_site(scope)),
    ).fetchone()
    return row is not None


def add_message(
    conn: psycopg.Connection,
    scope: Scope,
    session_id: uuid.UUID,
    role: str,
    content: str,
    *,
    citations: list[dict[str, Any]] | None = None,
    refused: bool = False,
) -> uuid.UUID:
    row = conn.execute(
        "INSERT INTO widget_messages (session_id, org_id, site_id, role, content, citations, "
        "refused) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s) RETURNING id",
        (
            session_id, scope.org_id, _need_site(scope), role, content,
            json.dumps(citations or [], default=str), refused,
        ),
    ).fetchone()  # fmt: skip
    return row[0]  # type: ignore[index]


# ---- handoff, leads, feedback ---------------------------------------------------------------
def _clean(value: str | None, field: str) -> str | None:
    value = (value or "").strip().replace("\x00", "")
    if len(value) > MAX_TEXT[field]:
        raise WidgetError(f"{field} is too long (max {MAX_TEXT[field]} characters)")
    return value or None


def request_handoff(
    conn: psycopg.Connection, scope: Scope, session_id: uuid.UUID, reason: str | None
) -> bool:
    """Flag the session for a human. Returns False if the session is not this visitor's."""
    row = conn.execute(
        "UPDATE widget_sessions SET handoff_requested = true, handoff_reason = %s, "
        "handoff_at = coalesce(handoff_at, now()), handoff_resolved_at = NULL "
        "WHERE id = %s AND org_id = %s AND site_id = %s RETURNING id",
        (_clean(reason, "reason"), session_id, scope.org_id, _need_site(scope)),
    ).fetchone()
    return row is not None


def add_lead(
    conn: psycopg.Connection,
    scope: Scope,
    session_id: uuid.UUID,
    *,
    name: str,
    email: str | None,
    phone: str | None = None,
    message: str | None = None,
    consent: bool,
) -> uuid.UUID:
    if not consent:
        raise WidgetError("we can only save your details with your consent")
    name_clean, email = _clean(name, "name"), (email or "").strip()
    if not name_clean:
        raise WidgetError("please give your name")
    if email and not _EMAIL.match(email):
        raise WidgetError("that email address does not look right")
    if not email and not (phone or "").strip():
        raise WidgetError("please give an email address or a phone number")
    if not session_exists(conn, scope, session_id):
        raise WidgetError("unknown session")
    row = conn.execute(
        "INSERT INTO leads (org_id, site_id, session_id, name, email, phone, message, consent) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, true) RETURNING id",
        (
            scope.org_id, _need_site(scope), session_id, name_clean, email.lower() or None,
            _clean(phone, "phone"), _clean(message, "message"),
        ),
    ).fetchone()  # fmt: skip
    return row[0]  # type: ignore[index]


def add_feedback(
    conn: psycopg.Connection,
    scope: Scope,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    rating: int,
    comment: str | None,
) -> None:
    if rating not in (-1, 1):
        raise WidgetError("rating must be 1 (helpful) or -1 (not helpful)")
    owns = conn.execute(
        "SELECT 1 FROM widget_messages WHERE id = %s AND session_id = %s AND org_id = %s "
        "AND site_id = %s AND role = 'assistant'",
        (message_id, session_id, scope.org_id, _need_site(scope)),
    ).fetchone()
    if owns is None:
        raise WidgetError("unknown message")
    conn.execute(
        "INSERT INTO feedback (org_id, site_id, session_id, message_id, rating, comment) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (scope.org_id, scope.site_id, session_id, message_id, rating, _clean(comment, "comment")),
    )


# ---- the per-site daily budget --------------------------------------------------------------
def spent_today(conn: psycopg.Connection, scope: Scope) -> float:
    row = conn.execute(
        "SELECT cost_usd FROM site_usage WHERE site_id = %s AND day = current_date",
        (_need_site(scope),),
    ).fetchone()
    return float(row[0]) if row else 0.0


def record_usage(conn: psycopg.Connection, scope: Scope, cost_usd: float) -> None:
    conn.execute(
        "INSERT INTO site_usage (site_id, org_id, requests, cost_usd) VALUES (%s, %s, 1, %s) "
        "ON CONFLICT (site_id, day) DO UPDATE SET requests = site_usage.requests + 1, "
        "cost_usd = site_usage.cost_usd + EXCLUDED.cost_usd",
        (_need_site(scope), scope.org_id, round(cost_usd, 6)),
    )


# ---- staff views (a member connection; the API requires an admin) ---------------------------
def list_leads(conn: psycopg.Connection, who: Scope) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT l.id, l.created_at, l.name, l.email, l.phone, l.message, l.session_id, "
            "s.name AS site "
            "FROM leads l JOIN sites s ON s.id = l.site_id WHERE l.org_id = %s "
            "ORDER BY l.created_at DESC LIMIT 200",
            (who.org_id,),
        ).fetchall()


def list_handoffs(conn: psycopg.Connection, who: Scope) -> list[dict[str, Any]]:
    """Sessions where a visitor asked for a person, newest first, with contact details if given."""
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT ws.id, ws.channel, ws.handoff_at, ws.handoff_reason, ws.handoff_resolved_at, "
            "(SELECT coalesce(l.email, l.phone) FROM leads l WHERE l.session_id = ws.id "
            "ORDER BY l.created_at DESC LIMIT 1) AS email, "
            "(SELECT count(*) FROM widget_messages m WHERE m.session_id = ws.id) AS messages "
            "FROM widget_sessions ws WHERE ws.org_id = %s AND ws.handoff_requested "
            "ORDER BY ws.handoff_resolved_at IS NOT NULL, ws.handoff_at DESC LIMIT 200",
            (who.org_id,),
        ).fetchall()


def transcript(conn: psycopg.Connection, who: Scope, session_id: uuid.UUID) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT role, content, refused, created_at FROM widget_messages "
            "WHERE session_id = %s AND org_id = %s ORDER BY created_at",
            (session_id, who.org_id),
        ).fetchall()


def resolve_handoff(conn: psycopg.Connection, who: Scope, session_id: uuid.UUID) -> bool:
    return (
        conn.execute(
            "UPDATE widget_sessions SET handoff_resolved_at = now() "
            "WHERE id = %s AND org_id = %s AND handoff_requested",
            (session_id, who.org_id),
        ).rowcount
        == 1
    )
