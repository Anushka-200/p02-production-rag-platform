"""Tiny migration runner: applies migrations/NNNN_name.sql files once each, in order.

Why not Alembic: the schema is small and plain SQL is easy to read, review and explain.
Run as the owner role (only it may change tables): `make migrate`.
"""

from pathlib import Path

import psycopg
from psycopg import sql

from app.config import get_settings
from app.log import setup_logging

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
LOCK_ID = 727001  # arbitrary number; makes two runners take turns instead of racing


def apply_migrations(conn: psycopg.Connection, schema: str | None = None) -> list[str]:
    """Apply pending migrations; return the versions applied now. `schema` is for tests."""
    conn.autocommit = True
    if schema:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    applied_now: list[str] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
            done = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s", (path.stem,)
            ).fetchone()
            if done:
                continue
            conn.execute(path.read_text())  # whole file = one transaction: all or nothing
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
            applied_now.append(path.stem)
    return applied_now


def main() -> None:
    setup_logging()
    with psycopg.connect(get_settings().database_url("owner")) as conn:
        applied = apply_migrations(conn)
    print("applied:", applied or "nothing to do")


if __name__ == "__main__":
    main()
