"""Who is asking? The organisation and user come from a VERIFIED login, never from the request
body or the model.

`token_principal`: `Authorization: Bearer <JWT>` from POST /auth/login. The token's signature and
expiry are checked, then the user is looked up again in the database (so a deleted user or changed
role takes effect at once) and the token's org must match the user's real org.

`dev_principal`: a development shortcut (header X-Dev-User) that works ONLY when the environment is
"dev". In any other environment it refuses every request.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass

import psycopg
from fastapi import HTTPException, Request

from app.api.security import TokenError, read_token
from app.config import Settings
from app.retrieval.search import Scope


@dataclass(frozen=True)
class SiteInfo:
    """What the API needs to know about an embedded site: its limits."""

    id: uuid.UUID
    name: str
    daily_budget_usd: float
    rate_limit_per_minute: int


@dataclass(frozen=True)
class Principal:
    email: str
    role: str  # "admin" | "member" | "anonymous"
    scope: Scope
    site: SiteInfo | None = None  # set for anonymous visitors: which embedded site they came from

    @property
    def site_id(self) -> uuid.UUID | None:
        return self.site.id if self.site else None


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


def token_principal(
    request: Request, settings: Settings, connect: Callable[[], psycopg.Connection]
) -> Principal:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(401, "missing bearer token")
    try:
        user_id, org_id = read_token(settings, token.strip())
    except TokenError as exc:
        raise HTTPException(401, "invalid or expired token") from exc  # no detail: nothing to probe
    conn = connect()
    try:
        row = conn.execute(
            "SELECT org_id, email, role FROM users WHERE id = %s", (user_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[0] != org_id:  # deleted user, or a token whose org does not match
        raise HTTPException(401, "invalid or expired token")
    return Principal(row[1], row[2], Scope(org_id, user_id))


def current_principal(
    request: Request, settings: Settings, connect: Callable[[], psycopg.Connection]
) -> Principal:
    """A bearer token wins; the dev header is only a fallback, and only in dev."""
    if request.headers.get("Authorization"):
        return token_principal(request, settings, connect)
    return dev_principal(request, settings, connect)


def normalise_origin(origin: str) -> str:
    return origin.strip().lower().rstrip("/")


def site_principal(
    site_key: str, origin: str | None, connect: Callable[[], psycopg.Connection]
) -> Principal:
    """An anonymous visitor of an embedded site: the org comes from the site key, the user is None.

    The key is public (it is in the page's HTML), so it proves nothing about the visitor. It only
    selects which org's PUBLIC documents may be read. Checks: the key exists and is enabled, and the
    browser's Origin header is on the site's allow-list, so another website cannot reuse the key.
    """
    conn = connect()
    try:
        row = conn.execute(
            "SELECT id, org_id, allowed_origins, enabled, daily_budget_usd, "
            "rate_limit_per_minute, name "
            "FROM sites WHERE site_key = %s",
            (site_key.strip(),),
        ).fetchone()
    finally:
        conn.close()
    if row is None or not row[3]:
        raise HTTPException(401, "unknown or disabled site")
    allowed = {normalise_origin(o) for o in row[2]}
    if origin is None or normalise_origin(origin) not in allowed:
        raise HTTPException(403, "this site is not allowed from this origin")
    site = SiteInfo(row[0], str(row[6]), float(row[4]), int(row[5]))
    return Principal("anonymous", "anonymous", Scope(row[1], None, site.id), site=site)
