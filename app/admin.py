"""Read-only views for the admin console: ingestion status, jobs and chunk previews.

All of them run on a tenant connection, so row-level security decides which rows are visible;
the org filter in the SQL is a second lock. An admin sees the org's shared documents and their own
private ones and jobs, never a colleague's private document.
"""

import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.retrieval.search import Scope

PREVIEW_CHARS = 400
MAX_ROWS = 200


def list_admin_documents(conn: psycopg.Connection, who: Scope) -> list[dict[str, Any]]:
    """One row per document: where it stands in ingestion and how many chunks searchable."""
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT d.id, d.title, d.source_uri, d.source_type, d.visibility, d.status, d.error, "
            "d.active_version, d.updated_at, u.email AS owner, "
            "(SELECT count(*) FROM chunks c WHERE c.document_id = d.id "
            " AND c.version = d.active_version AND c.level = 0) AS chunks "
            "FROM documents d JOIN users u ON u.id = d.owner_id "
            "WHERE d.org_id = %s ORDER BY d.updated_at DESC LIMIT %s",
            (who.org_id, MAX_ROWS),
        ).fetchall()


def list_jobs(conn: psycopg.Connection, who: Scope) -> list[dict[str, Any]]:
    """The user's own recent ingestion jobs (jobs are private to their owner)."""
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT id, source_uri, visibility, status, stage, attempts, error, created_at, "
            "finished_at, (result -> 'scan' ->> 'flagged')::boolean AS flagged_by_scan "
            "FROM ingestion_jobs WHERE org_id = %s AND owner_id = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (who.org_id, who.user_id, MAX_ROWS),
        ).fetchall()


def list_chunks(
    conn: psycopg.Connection,
    who: Scope,
    document_id: uuid.UUID,
    *,
    level: int | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Chunk previews of a document's ACTIVE version, in reading order (what retrieval can find)."""
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT c.id, c.level, c.chunk_index, c.section_path, c.page_start, c.page_end, "
            "c.anchors, length(c.text) AS chars, left(c.text, %s) AS preview, "
            "left(c.context_blurb, %s) AS blurb, (c.embedding IS NOT NULL) AS embedded "
            "FROM chunks c "
            "JOIN documents d ON d.id = c.document_id AND c.version = d.active_version "
            "WHERE c.document_id = %s AND c.org_id = %s "
            "AND (%s::int IS NULL OR c.level = %s::int) "
            "ORDER BY c.level, c.chunk_index LIMIT %s",
            (
                PREVIEW_CHARS,
                PREVIEW_CHARS,
                document_id,
                who.org_id,
                level,
                level,
                min(limit, MAX_ROWS),
            ),
        ).fetchall()
