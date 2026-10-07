"""Incremental sync: bring the documents in step with a source, doing only the difference.

    known pages (our table)      current pages (the connector)
    uri     fingerprint          uri     fingerprint
    a.md    111          ----->  a.md    111        unchanged  (nothing read, nothing embedded)
    b.md    222          ----->  b.md    999        CHANGED    (re-read, re-chunked, re-embedded)
    c.md    333          ----->  (gone)             DELETED    (document and derived data removed)
    (new)                        d.md    444        ADDED
    e.md    222 / org    ----->  e.md    222 / private   PERMISSION  (visibility flips, no re-embed)

Why it is built this way:
- comparing fingerprints needs no content: a sync of 4,000 unchanged pages costs one listing;
- every page is handled on its own and is safe to retry: a failure leaves that page's record
  unchanged so the next sync tries again; running the same sync twice changes nothing;
- a change creates a NEW VERSION of the document (the existing versioned write): readers never see
  a half-indexed page; derived data (summaries, graph facts) is removed by database triggers;
- MASS-DELETE GUARD: if one sync would remove more than a fifth of what is known (and more than
  20 pages) it stops and changes nothing. A wrong path prefix, an empty checkout or a source
  outage looks exactly like "everything was deleted"; an operator must confirm that on purpose.
"""

import logging
import uuid
from collections.abc import Callable
from typing import Any

import psycopg

from app.config import Settings
from app.connectors.base import Connector, SourceItem, SyncReport
from app.embeddings import Embedder
from app.ingest.chunking import TokenCounter
from app.ingest.pipeline import run_pipeline
from app.ingest.store import file_hash
from app.ingest.upload import EXTENSIONS
from app.llm import complete

logger = logging.getLogger(__name__)

IngestFn = Callable[[SourceItem, bytes], uuid.UUID | None]
MAX_DELETE_FRACTION = 0.2
MIN_DELETES_FOR_GUARD = 20


def make_pipeline_ingest(
    conn: psycopg.Connection,
    org_id: uuid.UUID,
    owner_id: uuid.UUID,
    *,
    settings: Settings,
    embedder: Embedder,
    complete_fn: Callable[..., Any] = complete,
    count: TokenCounter | None = None,
) -> IngestFn:
    """Ingest one page in-process (the same pipeline the upload worker runs), no queue involved."""

    def ingest(item: SourceItem, raw: bytes) -> uuid.UUID | None:
        suffix = "." + item.uri.rsplit(".", 1)[-1].lower() if "." in item.uri else ""
        job = {
            "org_id": org_id, "owner_id": owner_id, "source_uri": item.uri,
            "source_type": EXTENSIONS.get(suffix, "md"), "visibility": item.visibility,
            "content_hash": file_hash(raw), "source_updated_at": item.modified(),
        }  # fmt: skip
        result = run_pipeline(
            conn, job, raw, settings=settings, embedder=embedder, complete_fn=complete_fn,
            count=count, set_stage=lambda stage: None,
        )  # fmt: skip
        doc = result.get("document_id")
        return uuid.UUID(doc) if doc else None

    return ingest


def _known(
    conn: psycopg.Connection, org_id: uuid.UUID, connector: str
) -> dict[str, tuple[Any, ...]]:
    rows = conn.execute(
        "SELECT uri, source_hash, visibility, document_id, state FROM connector_items "
        "WHERE org_id = %s AND connector = %s",
        (org_id, connector),
    ).fetchall()
    return {r[0]: r for r in rows}


def plan_sync(
    known: dict[str, tuple[Any, ...]], current: dict[str, SourceItem]
) -> tuple[list[SourceItem], list[SourceItem], list[SourceItem], list[str], int]:
    """(added, changed, permission_changed, deleted_uris, unchanged_count) from two listings."""
    added, changed, permissions, unchanged = [], [], [], 0
    for uri, item in current.items():
        row = known.get(uri)
        if row is None or row[4] == "deleted":
            added.append(item)
        elif row[1] != item.source_hash:
            changed.append(item)
        elif row[2] != item.visibility:
            permissions.append(item)
        else:
            unchanged += 1
    deleted = [uri for uri, row in known.items() if row[4] == "active" and uri not in current]
    return added, changed, permissions, deleted, unchanged


def _upsert_item(
    conn: psycopg.Connection,
    org_id: uuid.UUID,
    connector: str,
    item: SourceItem,
    doc_id: uuid.UUID | None,
) -> None:
    conn.execute(
        "INSERT INTO connector_items (org_id, connector, uri, source_hash, visibility, "
        "document_id, state, synced_at) VALUES (%s, %s, %s, %s, %s, %s, 'active', now()) "
        "ON CONFLICT (org_id, connector, uri) DO UPDATE SET source_hash = EXCLUDED.source_hash, "
        "visibility = EXCLUDED.visibility, document_id = coalesce(EXCLUDED.document_id, "
        "connector_items.document_id), state = 'active', synced_at = now()",
        (org_id, connector, item.uri, item.source_hash, item.visibility, doc_id),
    )


def sync_connector(
    conn: psycopg.Connection,
    connector: Connector,
    *,
    org_id: uuid.UUID,
    owner_id: uuid.UUID,
    ingest: IngestFn,
    dry_run: bool = False,
    allow_mass_delete: bool = False,
) -> SyncReport:
    report = SyncReport(connector.name, connector.ref, dry_run=dry_run)
    known = _known(conn, org_id, connector.name)
    current = {item.uri: item for item in connector.items()}
    added, changed, permissions, deleted, report.unchanged = plan_sync(known, current)
    report.added = [i.uri for i in added]
    report.changed = [i.uri for i in changed]
    report.permission_changes = [i.uri for i in permissions]
    report.deleted = deleted
    active = sum(1 for r in known.values() if r[4] == "active")
    if (
        len(deleted) > MIN_DELETES_FOR_GUARD
        and len(deleted) > MAX_DELETE_FRACTION * active
        and not allow_mass_delete
    ):
        report.aborted = (
            f"would delete {len(deleted)} of {active} known pages; check the source and path"
        )
        report.added, report.changed, report.deleted, report.permission_changes = [], [], [], []
        return report
    if dry_run:
        return report

    run_id = conn.execute(
        "INSERT INTO connector_runs (org_id, connector, source_ref) "
        "VALUES (%s, %s, %s) RETURNING id",
        (org_id, connector.name, connector.ref),
    ).fetchone()[0]  # type: ignore[index]
    for item in [*added, *changed]:
        try:
            doc_id = ingest(item, item.read())
            _upsert_item(conn, org_id, connector.name, item, doc_id)
        except Exception as exc:  # one bad page must not stop the rest; it is retried next sync
            logger.warning(
                "sync failed for a page", extra={"uri": item.uri, "error": str(exc)[:200]}
            )
            report.failed.append((item.uri, f"{type(exc).__name__}: {str(exc)[:120]}"))
    failed = {u for u, _ in report.failed}
    report.added = [u for u in report.added if u not in failed]
    report.changed = [u for u in report.changed if u not in failed]
    for (
        item
    ) in permissions:  # a visibility flip: no re-embedding; derived data is purged by triggers
        doc_id = known[item.uri][3]
        if doc_id is not None:
            conn.execute(
                "UPDATE documents SET visibility = %s, updated_at = now() "
                "WHERE id = %s AND org_id = %s",
                (item.visibility, doc_id, org_id),
            )
        _upsert_item(conn, org_id, connector.name, item, None)
    for uri in deleted:
        doc_id = known[uri][3]
        if doc_id is not None:  # chunks, facts and summaries go with it (cascade and triggers)
            conn.execute("DELETE FROM documents WHERE id = %s AND org_id = %s", (doc_id, org_id))
        conn.execute(
            "UPDATE connector_items SET state = 'deleted', document_id = NULL, synced_at = now() "
            "WHERE org_id = %s AND connector = %s AND uri = %s",
            (org_id, connector.name, uri),
        )
    conn.execute(
        "UPDATE connector_runs SET finished_at = now(), added = %s, changed = %s, unchanged = %s, "
        "deleted = %s, permission_changes = %s, failed = %s WHERE id = %s",
        (len(report.added), len(report.changed), report.unchanged, len(report.deleted),
         len(report.permission_changes), len(report.failed), run_id),
    )  # fmt: skip
    return report
