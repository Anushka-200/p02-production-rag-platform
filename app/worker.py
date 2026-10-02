"""The ingestion worker: runs jobs from the queue and keeps their status in Postgres.

Failure policy:
- A bad FILE (corrupt, scanned, hash mismatch) fails at once with a readable message: retrying
  cannot help.
- Anything else (database hiccup, provider outage) is TRANSIENT: the job goes back to 'queued' with
  the error recorded and the exception is re-raised so RQ retries it after a delay. After
  job_max_attempts tries it is marked 'failed'.
Run it with: uv run python -m app.worker
"""

import json
import logging
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.config import Settings, get_settings
from app.cost_guard import BudgetExceeded
from app.db.tenant import set_session_tenant
from app.embeddings import Embedder, get_embedder
from app.ingest.chunking import TokenCounter
from app.ingest.models import ParseError
from app.ingest.pipeline import run_pipeline
from app.ingest.store import file_hash
from app.ingest.upload import UploadError
from app.llm import complete
from app.log import setup_logging
from app.retrieval.search import Scope

logger = logging.getLogger(__name__)

# The file is the problem, or retrying would only spend more money: never retry these.
PERMANENT = (ParseError, UploadError, BudgetExceeded)


def _update(conn: psycopg.Connection, job_id: uuid.UUID, **fields: Any) -> None:
    assignments = ", ".join(f"{name} = %s" for name in fields)
    values = [json.dumps(v) if name == "result" else v for name, v in fields.items()]
    conn.execute(f"UPDATE ingestion_jobs SET {assignments} WHERE id = %s", (*values, job_id))  # noqa: S608


def run_job(
    conn: psycopg.Connection,
    job_id: uuid.UUID,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    complete_fn: Callable[..., Any] = complete,
    count: TokenCounter | None = None,
) -> str:
    """Run one job. Returns its final status; re-raises transient errors so RQ can retry."""
    s = settings or get_settings()
    conn.autocommit = True
    with conn.cursor(row_factory=dict_row) as cur:
        job = cur.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
    if job is None:
        raise LookupError(f"job {job_id} does not exist")
    if job["status"] == "succeeded":
        return "succeeded"  # a duplicate delivery of a finished job is a no-op

    attempts = job["attempts"] + 1
    _update(conn, job_id, status="running", stage="parse", attempts=attempts, started_at=_now(conn))
    try:
        raw = Path(job["raw_path"]).read_bytes()
        if file_hash(raw) != job["content_hash"]:
            raise UploadError("the stored file does not match its checksum")
        embedder = embedder or get_embedder(s)
        result = run_pipeline(
            conn, job, raw, settings=s, embedder=embedder, complete_fn=complete_fn, count=count,
            set_stage=lambda stage: _update(conn, job_id, stage=stage),
        )  # fmt: skip
        document_id = result.get("document_id")
        _update(conn, job_id, status="succeeded", stage="done", error=None, result=result,
                finished_at=_now(conn), document_id=document_id)  # fmt: skip
        logger.info("job succeeded", extra={"job_id": str(job_id), "result": result})
        return "succeeded"
    except PERMANENT as exc:
        _update(conn, job_id, status="failed", error=str(exc), finished_at=_now(conn))
        logger.warning(
            "job failed (file problem)", extra={"job_id": str(job_id), "error": str(exc)}
        )
        return "failed"
    except Exception as exc:
        if attempts >= s.job_max_attempts:
            _update(conn, job_id, status="failed", finished_at=_now(conn),
                    error=f"gave up after {attempts} attempts: {exc}"[:1000])  # fmt: skip
            logger.error("job failed after retries", extra={"job_id": str(job_id)}, exc_info=True)
            return "failed"
        _update(
            conn, job_id, status="queued", stage="queued", error=f"attempt {attempts}: {exc}"[:1000]
        )
        logger.warning("job will be retried", extra={"job_id": str(job_id), "attempt": attempts})
        raise


def _now(conn: psycopg.Connection) -> Any:
    return conn.execute("SELECT now()").fetchone()[0]  # type: ignore[index]


# ---- RQ entry points ----------------------------------------------------------------------------
_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def _shared_embedder(settings: Settings) -> Embedder:
    """The embedding model takes ~18 s to load: load it once per worker process."""
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = get_embedder(settings)
        return _embedder


def process_job(job_id: str, org_id: str, owner_id: str) -> str:
    """What RQ runs. Opens a database connection as the limited worker role.

    The queue payload carries the tenant (set by the backend when it enqueued, from the verified
    login), because the worker cannot even read its own job row until it has said whose it is.
    """
    settings = get_settings()
    with psycopg.connect(settings.database_url("worker")) as conn:
        set_session_tenant(conn, Scope(uuid.UUID(org_id), uuid.UUID(owner_id)))
        return run_job(
            conn, uuid.UUID(job_id), settings=settings, embedder=_shared_embedder(settings)
        )


def main() -> None:
    import sys

    from redis import Redis
    from rq import Queue, Worker

    setup_logging()
    settings = get_settings()
    connection = Redis.from_url(settings.redis_url)
    worker = Worker([Queue(settings.ingest_queue, connection=connection)], connection=connection)
    worker.work(
        burst="--burst" in sys.argv
    )  # --burst: stop when the queue is empty (scripts, tests)


if __name__ == "__main__":
    main()
