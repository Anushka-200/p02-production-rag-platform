"""Who is asking? DEV-ONLY stand-in until real JWT login (Phase 3, task 3.3).

The organisation and user must come from a VERIFIED login, never from the request body or the model.
Until JWT exists, a header names a user who is looked up in the database, and ONLY when the
environment is "dev". In any other environment every request is refused.
"""

from collections.abc import Callable
from dataclasses import dataclass

import psycopg
from fastapi import HTTPException, Request

from app.config import Settings
from app.retrieval.search import Scope


@dataclass(frozen=True)
class Principal:
    email: str
    role: str
    scope: Scope


def dev_principal(
    request: Request, settings: Settings, connect: Callable[[], psycopg.Connection]
) -> Principal:
    if settings.environment != "dev":
        raise HTTPException(401, "login is not available: dev authentication is disabled here")
    email = request.headers.get("X-Dev-User", "").strip().lower()
    if not email:
        raise HTTPException(401, "missing X-Dev-User header (dev login)")
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, org_id, role FROM users WHERE lower(email) = %s LIMIT 2", (email,)
        ).fetchall()
    finally:
        conn.close()
    if len(rows) != 1:  # unknown, or the same email in several organisations (ambiguous)
        raise HTTPException(401, "unknown user")
    user_id, org_id, role = rows[0]
    return Principal(email, role, Scope(org_id, user_id))
