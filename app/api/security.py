"""Passwords and login tokens.

Passwords: scrypt (memory-hard, in Python's standard library), random salt per user, constant-time
compare. Tokens: signed JWTs (HS256). A token only proves "the server issued this to user X"; the
backend still looks the user up on every request, so a deleted user or changed role takes effect
immediately instead of when the token expires.
"""

import base64
import hashlib
import hmac
import secrets
import time
import uuid

import jwt

from app.config import Settings

_N, _R, _P = 2**14, 8, 1  # scrypt cost: ~50 ms per hash on a laptop
MIN_SECRET_CHARS = 32


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str | None) -> bool:
    """False for a wrong password, a missing hash or a malformed hash. Never raises."""
    try:
        scheme, salt_b64, digest_b64 = (stored or "").split("$")
        if scheme != "scrypt":
            return False
        salt, digest = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
        candidate = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P)
    except ValueError:
        return False
    return hmac.compare_digest(candidate, digest)


class TokenError(Exception):
    """The token is missing, malformed, expired, forged or not meant for us."""


def _secret(settings: Settings) -> str:
    if settings.jwt_secret is None:
        raise TokenError("JWT_SECRET is not set")
    secret = settings.jwt_secret.get_secret_value()
    if len(secret) < MIN_SECRET_CHARS:
        raise TokenError(f"JWT_SECRET must be at least {MIN_SECRET_CHARS} characters")
    return secret


def create_token(settings: Settings, user_id: uuid.UUID, org_id: uuid.UUID, role: str) -> str:
    now = int(time.time())
    claims = {
        "sub": str(user_id),
        "org": str(org_id),
        "role": role,
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": now + settings.jwt_ttl_minutes * 60,
    }
    return jwt.encode(claims, _secret(settings), algorithm="HS256")


def read_token(settings: Settings, token: str) -> tuple[uuid.UUID, uuid.UUID]:
    """Return (user_id, org_id) from a valid token, else raise TokenError."""
    try:
        claims = jwt.decode(
            token,
            _secret(settings),
            algorithms=["HS256"],  # pinned: never let the token choose its own algorithm ("none")
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "sub", "org", "iss"]},
        )
        return uuid.UUID(claims["sub"]), uuid.UUID(claims["org"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise TokenError(str(exc)) from exc
