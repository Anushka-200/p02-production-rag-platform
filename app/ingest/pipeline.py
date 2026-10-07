"""The ingestion pipeline for one file: parse > clean > chunk > (blurbs) > embed > save.

Pure orchestration: it takes the connection, embedder and model function as arguments, so tests
run it with fakes and the worker runs it with the real ones.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psycopg

from app.config import Settings
from app.cost_guard import CostGuard
from app.embeddings import Embedder, get_token_counter
from app.ingest.chunkers import chunk_document
from app.ingest.chunking import TokenCounter
from app.ingest.contextual import (
    BlurbCache,
    BlurbStats,
    add_blurbs,
    embedding_text,
    project_blurb_cost,
)
from app.ingest.models import ParsedDocument, ParseError
from app.ingest.parsers import parse
from app.ingest.preprocess import preprocess
from app.ingest.scan import scan_text, summarize
from app.ingest.store import plan_ingest, write_version

logger = logging.getLogger(__name__)


def run_pipeline(
    conn: psycopg.Connection,
    job: dict[str, Any],
    raw: bytes,
    *,
    settings: Settings,
    embedder: Embedder,
    complete_fn: Callable[..., Any],
    count: TokenCounter | None,
    set_stage: Callable[[str], None],
) -> dict[str, Any]:
    plan = plan_ingest(
        conn,
        org_id=job["org_id"],
        owner_id=job["owner_id"],
        source_uri=job["source_uri"],
        content_hash=job["content_hash"],
    )
    if plan.action != "ingest":  # nothing to do, and nothing was spent
        return {
            "action": plan.action,
            "document_id": str(plan.document_id) if plan.document_id else None,
            "duplicate_of": str(plan.duplicate_of) if plan.duplicate_of else None,
        }

    set_stage("parse")
    parsed = preprocess(parse(job["source_type"], raw, job["source_uri"]))
    scan = scan_summary(raw, job["source_type"], parsed)
    if scan["flagged"]:
        logger.warning("document flagged by the ingestion scan",
                       extra={"source_uri": job["source_uri"], "scan": scan})  # fmt: skip
    if not parsed.sections:
        raise ParseError("no readable text was found in the file")

    set_stage("chunk")
    counter = count or get_token_counter(settings)
    chunks = chunk_document(parsed, settings, counter)

    blurb_cost = 0.0
    if settings.blurbs_enabled:
        set_stage("blurb")
        guard = CostGuard(
            f"blurbs for {job['source_uri']}",
            project_blurb_cost([(parsed, chunks)], settings.llm_default_model, counter),
            max_usd=settings.blurb_max_usd_per_document,
            assume_yes=True,  # unattended: the hard stop is the protection, not a prompt
            say=lambda _: None,
        )
        stats = BlurbStats()
        cache = BlurbCache(Path(settings.upload_dir).parent / "cache" / "blurbs.jsonl")
        chunks = add_blurbs(
            parsed,
            chunks,
            complete_fn=complete_fn,
            guard=guard,
            stats=stats,
            cache=cache,
            count=counter,
        )
        blurb_cost = stats.cost_usd

    set_stage("embed")
    children = [c for c in chunks if c.level == 0]
    texts = []
    for c in children:
        text = embedding_text(c)
        if c.context_blurb and counter(text) > settings.embedding_max_tokens:
            text = c.text  # blurb + chunk would be cut by the model: embed the chunk alone
        texts.append(text)
    vectors = iter(embedder.embed_documents(texts))
    embeddings = [next(vectors) if c.level == 0 else None for c in chunks]  # parents: no vector

    set_stage("save")
    written = write_version(
        conn,
        plan,
        org_id=job["org_id"],
        owner_id=job["owner_id"],
        visibility=job["visibility"],
        source_uri=job["source_uri"],
        source_type=job["source_type"],
        content_hash=job["content_hash"],
        source_updated_at=job.get("source_updated_at"),
        title=parsed.title,
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedder.model,
    )
    return {
        "action": "created" if written.created else "updated",
        "document_id": str(written.document_id),
        "version": written.version,
        "chunks": len(chunks),
        "children": len(children),
        "parents": len(chunks) - len(children),
        "chunks_replaced": written.chunks_deleted,
        "preprocess": parsed.metadata.get("preprocess", {}),
        "scan": scan,
        "blurb_cost_usd": round(blurb_cost, 5),
        "embedding_model": embedder.model,
    }


def scan_summary(raw: bytes, source_type: str, parsed: ParsedDocument) -> dict[str, Any]:
    """Scan what a reader would NOT see (the raw file) as well as what will be indexed.

    Text formats are scanned raw, so comments, hidden CSS and invisible characters are still there.
    PDF/DOCX are binary: the extracted text and the title are scanned instead.
    """
    if source_type in {"md", "txt", "html"}:
        text = raw.decode("utf-8", errors="replace")
    else:
        text = "\n".join(s.text for s in parsed.sections)
    title = parsed.metadata.get("original_title", parsed.title)
    return summarize(scan_text(f"{title}\n{text}"))
