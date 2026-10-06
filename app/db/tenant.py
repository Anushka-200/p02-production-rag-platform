"""Tell Postgres who is asking, so row-level security can hide everyone else's rows.

Two settings, app.org_id and app.user_id, are read by the policies in migration 0004.

Two ways to set them, matching how the connection is used:
- `tenant_connection` (API): the settings last only for ONE TRANSACTION (set_config(..., true) is
  the parameter-friendly SET LOCAL). A pooled connection that is reused cannot carry the previous
  tenant over, because the transaction always ends (rollback) before the connection is returned.
- `set_session_tenant` (worker): a worker opens its own connection per job and closes it, so
  session-level settings are safe there; a transaction would be too short (it autocommits).
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import psycopg

from app.retrieval.search import Scope

_SET = (
    "SELECT set_config('app.org_id', %s, %s), set_config('app.user_id', %s, %s), "
    "set_config('app.site_id', %s, %s)"
)


def _user(scope: Scope) -> str:
    """The user id as text; an anonymous visitor has none, which the policies read as NULL."""
    return "" if scope.user_id is None else str(scope.user_id)


def _site(scope: Scope) -> str:
    return "" if scope.site_id is None else str(scope.site_id)


def set_session_tenant(conn: psycopg.Connection, scope: Scope) -> None:
    conn.execute(_SET, (str(scope.org_id), False, _user(scope), False, _site(scope), False))


@contextmanager
def tenant_connection(
    connect: Callable[[], psycopg.Connection], scope: Scope, *, write: bool = False
) -> Iterator[psycopg.Connection]:
    """A connection whose queries see only `scope`'s rows.

    Read-only by default: always rolled back. With write=True the work is committed when the block
    ends normally and rolled back if it raises.
    """
    conn = connect()
    try:
        conn.rollback()  # a connection handed back mid-transaction must not keep stale settings
        conn.autocommit = False
        conn.execute(_SET, (str(scope.org_id), True, _user(scope), True, _site(scope), True))
        yield conn
        if write:
            conn.commit()
    finally:
        try:
            conn.rollback()  # ends the transaction, which also drops the tenant settings
        finally:
            conn.close()


def check_role_is_restricted(conn: psycopg.Connection) -> None:
    """Refuse to run as a role that ignores row-level security (superuser, BYPASSRLS, table owner).

    RLS silently does nothing for such a role, so a wrong DATABASE_URL would turn isolation off
    without any error. Better to fail at startup.
    """
    row = conn.execute(
        "SELECT r.rolsuper, r.rolbypassrls, "
        "EXISTS (SELECT 1 FROM pg_class c WHERE c.relname = 'chunks' AND c.relowner = r.oid) "
        "FROM pg_roles r WHERE r.rolname = current_user"
    ).fetchone()
    if row is None or any(row):
        raise RuntimeError(
            "database role bypasses row-level security (superuser, BYPASSRLS or table owner); "
            "the API must connect as rag_app"
        )
