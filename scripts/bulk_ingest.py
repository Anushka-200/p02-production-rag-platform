"""Bulk-ingest the corpora through the REAL queue, worker, embedding model and database.

Tenant A = the GitLab handbook slice + 3 PDFs + the sample DOCX; Tenant B = the PostHog handbook.
No model spend (blurbs stay off). Safe to re-run: unchanged files are skipped by the pipeline.
Run: uv run python -m scripts.bulk_ingest [--tenant a|b|all] [--limit N]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import psycopg
from redis import Redis
from rq import Queue

from app.config import get_settings
from app.ingest.parsers import parse_markdown
from app.ingest.preprocess import preprocess
from app.ingest.upload import enqueue_upload
from evals.corpus import GITLAB_SLICE, PDFS, RAW

GITLAB = RAW / "gitlab-handbook/content/handbook"
POSTHOG = RAW / "posthog-handbook/contents/handbook"
ORGS = {"a": ("Tenant A (GitLab handbook)", "admin@tenant-a.example"),
        "b": ("Tenant B (PostHog handbook)", "admin@tenant-b.example")}  # fmt: skip


def markdown_files(
    root: Path, prefix: str, dirs: list[str] | None = None
) -> list[tuple[str, Path]]:
    found = sorted(
        f for d in (dirs or [""]) for f in (root / d).rglob("*") if f.suffix in {".md", ".mdx"}
    )
    return [(f"{prefix}/{f.relative_to(root).as_posix()}", f) for f in found]


def has_text(path: Path) -> bool:
    return bool(preprocess(parse_markdown(path.read_bytes(), path.name)).sections)


def progress(owner: psycopg.Connection, job_ids: list, total: int, started: float) -> str:
    rows = dict(owner.execute(
        "SELECT status, count(*) FROM ingestion_jobs WHERE id = ANY(%s) GROUP BY status", (job_ids,)
    ).fetchall())  # fmt: skip  # only THIS run's jobs
    done = rows.get("succeeded", 0) + rows.get("failed", 0)
    rate = done / max(time.time() - started, 1)
    eta = (total - done) / rate / 60 if rate else float("inf")
    return f"{done}/{total} finished ({rows.get('failed', 0)} failed), {rate * 60:.0f} jobs/min, about {eta:.0f} min left"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", choices=["a", "b", "all"], default="all")
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N files per tenant (testing)"
    )
    args = parser.parse_args()
    s = get_settings()
    owner = psycopg.connect(s.database_url("owner"), autocommit=True)
    app_conn = psycopg.connect(s.database_url("app"), autocommit=True)
    queue = Queue(s.ingest_queue, connection=Redis.from_url(s.redis_url))

    plan: dict[str, list[tuple[str, Path]]] = {}
    if args.tenant in {"a", "all"}:
        plan["a"] = (
            markdown_files(GITLAB, "gitlab", GITLAB_SLICE)
            + [(name, RAW / "hr-pdfs" / name) for name in PDFS.values()]
            + [("acme-benefits-policy.docx", Path("data/samples/acme-benefits-policy.docx"))]
        )
    if args.tenant in {"b", "all"}:
        plan["b"] = markdown_files(POSTHOG, "posthog")

    org_ids, jobs, skipped_empty = [], [], 0
    for key, files in plan.items():
        name, email = ORGS[key]
        org = owner.execute(
            "INSERT INTO orgs (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
            (name,),
        ).fetchone()[0]
        user = owner.execute(
            "INSERT INTO users (org_id, email, role) VALUES (%s, %s, 'admin') "
            "ON CONFLICT (org_id, email) DO UPDATE SET role = 'admin' RETURNING id",
            (org, email),
        ).fetchone()[0]
        org_ids.append(org)
        for uri, path in files[: args.limit]:
            if path.suffix in {".md", ".mdx"} and not has_text(path):
                skipped_empty += 1  # index stubs with no text: nothing to ingest, not an error
                continue
            r = enqueue_upload(
                app_conn, queue, org_id=org, owner_id=user, visibility="org",
                filename=path.name, raw=path.read_bytes(), settings=s, source_uri=uri,
            )  # fmt: skip
            jobs.append(r.job_id)
        print(f"tenant {key}: {len(files[: args.limit])} files planned")
    print(f"enqueued {len(jobs)} jobs ({skipped_empty} empty index pages skipped)\n")

    started = time.time()
    worker = subprocess.Popen(
        [sys.executable, "-m", "app.worker", "--burst"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    while worker.poll() is None:
        time.sleep(30)
        print(
            f"[{(time.time() - started) / 60:5.1f} min] {progress(owner, jobs, len(jobs), started)}",
            flush=True,
        )
    print(
        f"\nworker finished in {(time.time() - started) / 60:.1f} min (exit code {worker.returncode})"
    )

    totals = owner.execute(
        "SELECT o.name, count(DISTINCT d.id), count(*) FILTER (WHERE c.level = 0), count(*) FILTER (WHERE c.level = 1), "
        "count(*) FILTER (WHERE c.embedding IS NOT NULL) FROM orgs o LEFT JOIN documents d ON d.org_id = o.id "
        "LEFT JOIN chunks c ON c.document_id = d.id WHERE o.id = ANY(%s) GROUP BY o.name ORDER BY o.name",
        (org_ids,),
    ).fetchall()
    print(f"\n{'tenant':32} {'docs':>5} {'children':>9} {'parents':>8} {'vectors':>8}")
    for row in totals:
        print(f"{row[0]:32} {row[1]:5} {row[2]:9} {row[3]:8} {row[4]:8}")
    failed = owner.execute(
        "SELECT source_uri, error FROM ingestion_jobs WHERE org_id = ANY(%s) AND status = 'failed'",
        (org_ids,),
    ).fetchall()
    print(f"\nfailed jobs: {len(failed)}")
    for uri, error in failed[:15]:
        print(f"  {uri}: {error}")


if __name__ == "__main__":
    main()
