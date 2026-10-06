"""Accepting an upload: validate it, store the original, create a job row, enqueue it.

The API (Phase 2/3) calls enqueue_upload() and answers immediately; the worker does the slow part.
"""

import hashlib
import io
import os
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePath

import psycopg
from rq import Queue, Retry

from app.config import Settings, get_settings
from app.ingest.store import file_hash

EXTENSIONS = {  # .mdx is Markdown with components: the Markdown parser handles it
    ".pdf": "pdf", ".docx": "docx", ".md": "md", ".markdown": "md", ".mdx": "md",
    ".txt": "txt", ".html": "html", ".htm": "html",
}  # fmt: skip
RETRY_INTERVALS = [10, 60, 300]  # seconds before retry 1, 2, 3 (backoff)


class UploadError(ValueError):
    """The upload is not acceptable. The message is meant for the user."""


def validate_upload(filename: str, raw: bytes, max_bytes: int) -> str:
    """Return the source type, or raise UploadError. Checks name, size and file signature."""
    name = PurePath(filename).name
    source_type = EXTENSIONS.get(PurePath(name).suffix.lower())
    if source_type is None:
        raise UploadError(
            f"unsupported file type: {name!r} (allowed: {', '.join(sorted(EXTENSIONS))})"
        )
    if not raw:
        raise UploadError("the file is empty")
    if len(raw) > max_bytes:
        raise UploadError(f"the file is too large (limit {max_bytes // 1_000_000} MB)")
    # Magic bytes: the extension is only a claim; the content must agree with it.
    if source_type == "pdf" and not raw.startswith(b"%PDF-"):
        raise UploadError("this is not a real PDF file")
    if source_type == "docx":
        try:
            names = zipfile.ZipFile(io.BytesIO(raw)).namelist()
        except zipfile.BadZipFile:
            raise UploadError("this is not a real DOCX file") from None
        if "word/document.xml" not in names:
            raise UploadError("this is not a real DOCX file")
    if source_type in {"md", "txt", "html"} and b"\x00" in raw[:8192]:
        raise UploadError("this looks like a binary file, not text")
    return source_type


def normalize_source_uri(uri: str) -> str:
    """A document's identity label: a clean relative path like "benefits/_index.md".

    It is only a label (the stored file is named by its hash), but it still must not contain
    "..", absolute paths or control characters.
    """
    parts = [p for p in uri.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    clean = "/".join(parts)
    if not clean or len(clean) > 500 or any(ord(ch) < 32 for ch in clean):
        raise UploadError("invalid file name")
    return clean


def save_raw(upload_dir: str, org_id: uuid.UUID, content_hash: str, raw: bytes) -> Path:
    """Store the original bytes under a name made of their hash (so equal files share a path)."""
    path = Path(upload_dir) / str(org_id) / content_hash[:2] / content_hash
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)  # write to a temp file, then rename: no half files
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    os.replace(tmp, path)
    return path


@dataclass
class EnqueueResult:
    job_id: uuid.UUID
    status: str
    created: bool  # False when this exact upload was already known


def enqueue_upload(
    conn: psycopg.Connection,
    queue: Queue,
    *,
    org_id: uuid.UUID,
    owner_id: uuid.UUID,
    visibility: str,
    filename: str,
    raw: bytes,
    settings: Settings | None = None,
    source_uri: str | None = None,  # bulk sources pass the path; users' uploads use the file name
) -> EnqueueResult:
    s = settings or get_settings()
    source_type = validate_upload(filename, raw, s.max_upload_mb * 1_000_000)
    source_uri = normalize_source_uri(source_uri or PurePath(filename).name)
    content_hash = file_hash(raw)
    raw_path = save_raw(s.upload_dir, org_id, content_hash, raw)
    key = hashlib.sha256(f"{owner_id}|{source_uri}|{content_hash}".encode()).hexdigest()

    row = conn.execute(
        "INSERT INTO ingestion_jobs (org_id, owner_id, source_uri, source_type, visibility, "
        "content_hash, raw_path, idempotency_key) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT (org_id, idempotency_key) DO NOTHING RETURNING id",
        (org_id, owner_id, source_uri, source_type, visibility, content_hash, str(raw_path), key),
    ).fetchone()
    created = row is not None
    if row is None:  # the same upload again
        job_id, status = conn.execute(
            "SELECT id, status FROM ingestion_jobs WHERE org_id = %s AND idempotency_key = %s",
            (org_id, key),
        ).fetchone()  # type: ignore[misc]
        if status != "failed":
            return EnqueueResult(job_id, status, created=False)  # already queued, running or done
        conn.execute(  # a failed job can be retried by uploading the file again
            "UPDATE ingestion_jobs SET status = 'queued', stage = 'queued', error = NULL, "
            "attempts = 0, finished_at = NULL WHERE id = %s",
            (job_id,),
        )
    else:
        job_id = row[0]
    conn.commit()  # the worker must be able to SEE the job row the moment it is queued
    queue.enqueue(
        "app.worker.process_job",
        str(job_id),
        str(org_id),
        str(owner_id),
        retry=Retry(max=s.job_max_attempts - 1, interval=RETRY_INTERVALS),
        job_timeout=1800,
        job_id=str(job_id),  # one Redis job per database job: never queued twice
    )
    return EnqueueResult(job_id, "queued", created=created)
