"""Saving a parsed, chunked and embedded document into Postgres, as a new VERSION.

Two steps, so the cheap check happens before the expensive LLM and embedding work:
1. plan_ingest(): is this upload new, changed, unchanged or a duplicate?  (no writes)
2. write_version(): in ONE transaction, insert the new chunks, flip the document to the new
   version, delete the old chunks. A reader sees the old version or the new one, never a mix.

Dedupe never crosses users: telling someone "this file already exists" when another user uploaded
it would leak that the file exists, so duplicates are only looked for within one owner.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import psycopg

from app.embeddings import vector_literal
from app.ingest.chunking import Chunk

logger = logging.getLogger(__name__)

Action = Literal["ingest", "skip_unchanged", "skip_duplicate"]


class IngestError(RuntimeError):
    """Writing a version failed; the previous version (if any) is untouched."""


def file_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass
class Plan:
    action: Action
    document_id: uuid.UUID | None  # existing document, if any
    next_version: int
    duplicate_of: uuid.UUID | None = None


@dataclass
class IngestResult:
    document_id: uuid.UUID
    version: int
    chunks_written: int
    chunks_deleted: int
    created: bool  # True for the first version of a document


def plan_ingest(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    owner_id: uuid.UUID,
    source_uri: str,
    content_hash: str,
) -> Plan:
    """Decide what to do with an upload, before spending any money on it."""
    row = conn.execute(
        "SELECT id, content_hash, active_version, status FROM documents "
        "WHERE org_id = %s AND owner_id = %s AND source_uri = %s",
        (org_id, owner_id, source_uri),
    ).fetchone()
    if row:
        doc_id, old_hash, active_version, status = row
        if old_hash == content_hash and status == "ready":
            return Plan("skip_unchanged", doc_id, active_version)
        return Plan("ingest", doc_id, active_version + 1)  # changed, or an earlier attempt failed
    twin = conn.execute(  # same bytes under another name, same owner only
        "SELECT id FROM documents WHERE org_id = %s AND owner_id = %s AND content_hash = %s "
        "AND status = 'ready'",
        (org_id, owner_id, content_hash),
    ).fetchone()
    if twin:
        return Plan("skip_duplicate", None, 1, duplicate_of=twin[0])
    return Plan("ingest", None, 1)


_INSERT_CHUNK = (
    "INSERT INTO chunks (org_id, owner_id, visibility, document_id, version, level, parent_id, "
    "chunk_index, page_start, page_end, section_path, text, context_blurb, content_hash, anchors, "
    "embedding, embedding_model) "
    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s) RETURNING id"
)


def _chunk_params(
    c: Chunk,
    ids: tuple[uuid.UUID, uuid.UUID, str, uuid.UUID, int],
    parent_id: uuid.UUID | None,
    embedding: list[float] | None,
    model: str | None,
) -> tuple[object, ...]:
    org_id, owner_id, visibility, document_id, version = ids
    return (
        org_id, owner_id, visibility, document_id, version, c.level, parent_id, c.index,
        c.page_start, c.page_end, " > ".join(c.heading_path), c.text, c.context_blurb,
        hashlib.sha256(c.text.encode()).hexdigest(), list(c.anchors),
        vector_literal(embedding) if embedding is not None else None,
        model if embedding is not None else None,
    )  # fmt: skip


def write_version(
    conn: psycopg.Connection,
    plan: Plan,
    *,
    org_id: uuid.UUID,
    owner_id: uuid.UUID,
    visibility: str,
    source_uri: str,
    source_type: str,
    content_hash: str,
    title: str,
    chunks: list[Chunk],
    embeddings: list[list[float] | None],
    embedding_model: str,
    source_updated_at: datetime | None = None,
) -> IngestResult:
    """Write one new version atomically. `embeddings` lines up with `chunks` (None for parents)."""
    if len(chunks) != len(embeddings):
        raise ValueError("embeddings must line up with chunks")
    try:
        with conn.transaction():
            # Two workers ingesting the same document take turns instead of racing.
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{org_id}:{owner_id}:{source_uri}",)
            )
            created = plan.document_id is None
            if created:
                document_id = conn.execute(
                    "INSERT INTO documents (org_id, owner_id, visibility, title, source_type, "
                    "source_uri, content_hash, status, source_updated_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,'processing',%s) RETURNING id",
                    (
                        org_id,
                        owner_id,
                        visibility,
                        title,
                        source_type,
                        source_uri,
                        content_hash,
                        source_updated_at,
                    ),
                ).fetchone()[0]  # type: ignore[index]
            else:
                document_id = plan.document_id
                # Visibility changes first: chunks must match their document's row (foreign key).
                conn.execute(
                    "UPDATE documents SET visibility = %s, title = %s, source_type = %s, "
                    "status = 'processing', error = NULL, source_updated_at = %s WHERE id = %s",
                    (visibility, title, source_type, source_updated_at, document_id),
                )
            ids = (org_id, owner_id, visibility, document_id, plan.next_version)

            # Parents first: children point at their ids.
            parent_ids: dict[int, uuid.UUID] = {}
            for chunk, vector in zip(chunks, embeddings, strict=True):
                if chunk.level != 1:
                    continue
                parent_ids[chunk.index] = conn.execute(
                    _INSERT_CHUNK, _chunk_params(chunk, ids, None, vector, embedding_model)
                ).fetchone()[0]  # type: ignore[index]
            for chunk, vector in zip(chunks, embeddings, strict=True):
                if chunk.level == 1:
                    continue
                parent = (
                    parent_ids.get(chunk.parent_index) if chunk.parent_index is not None else None
                )
                conn.execute(
                    _INSERT_CHUNK, _chunk_params(chunk, ids, parent, vector, embedding_model)
                )

            deleted = conn.execute(
                "DELETE FROM chunks WHERE document_id = %s AND version < %s",
                (document_id, plan.next_version),
            ).rowcount
            conn.execute(
                "UPDATE documents SET active_version = %s, content_hash = %s, status = 'ready', "
                "updated_at = now() WHERE id = %s",
                (plan.next_version, content_hash, document_id),
            )
    except Exception as exc:
        # The transaction rolled back, so the old version is intact. Record why, outside it.
        _mark_failed(conn, org_id, owner_id, source_uri, str(exc))
        raise IngestError(f"{source_uri}: {exc}") from exc
    return IngestResult(document_id, plan.next_version, len(chunks), deleted, created)


def _mark_failed(
    conn: psycopg.Connection, org_id: uuid.UUID, owner_id: uuid.UUID, source_uri: str, error: str
) -> None:
    try:
        conn.execute(
            "UPDATE documents SET status = 'failed', error = %s "
            "WHERE org_id = %s AND owner_id = %s AND source_uri = %s",
            (error[:1000], org_id, owner_id, source_uri),
        )
    except Exception:  # never hide the original error behind a bookkeeping error
        logger.exception("could not record the failure")
