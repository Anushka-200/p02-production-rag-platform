"""LIVE run: ingest a small Tenant A slice through the REAL queue, worker, embedding model and DB.

No model spend (blurbs stay off). Safe to re-run: identical files are skipped.
Run: uv run python -m scripts.live_ingest
"""

import subprocess
import sys
import time
from pathlib import Path

import psycopg
from redis import Redis
from rq import Queue

from app.config import get_settings
from app.db.tenant import set_session_tenant
from app.ingest.upload import enqueue_upload
from app.retrieval.search import Scope

ROOT = Path("data/raw/gitlab-handbook/content/handbook/total-rewards")
PAGES = [
    "benefits/_index.md",
    "benefits/modern-health.md",
    "benefits/parental-leave-toolkit.md",
    "benefits/general-and-entity-benefits/financial-wellness.md",
    "benefits/general-and-entity-benefits/canada-corp-benefits.md",
    "benefits/general-and-entity-benefits/gitlab-ireland-ltd.md",
    "compensation/_index.md",
    "compensation/compensation-review-cycle/_index.md",
]
EXTRA = [  # one PDF and one DOCX, so every parser runs for real
    (
        "opm-health-insurance-overview.pdf",
        Path("data/raw/hr-pdfs/opm-health-insurance-overview.pdf"),
    ),
    ("acme-benefits-policy.docx", Path("data/samples/acme-benefits-policy.docx")),
]


def main() -> None:
    s = get_settings()
    owner = psycopg.connect(s.database_url("owner"), autocommit=True)
    org = owner.execute(
        "INSERT INTO orgs (name) VALUES ('Tenant A (GitLab handbook)') "
        "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id"
    ).fetchone()[0]
    user = owner.execute(
        "INSERT INTO users (org_id, email, role) VALUES (%s, 'admin@tenant-a.example', 'admin') "
        "ON CONFLICT (org_id, email) DO UPDATE SET role = 'admin' RETURNING id",
        (org,),
    ).fetchone()[0]

    files = [(f"gitlab/total-rewards/{p}", (ROOT / p)) for p in PAGES] + EXTRA
    app_conn = psycopg.connect(s.database_url("app"), autocommit=True)  # the API's limited role
    queue = Queue(s.ingest_queue, connection=Redis.from_url(s.redis_url))
    job_ids = []
    set_session_tenant(app_conn, Scope(org, user))  # row-level security: say whose rows these are
    for uri, path in files:
        r = enqueue_upload(
            app_conn, queue, org_id=org, owner_id=user, visibility="org",
            filename=path.name, raw=path.read_bytes(), settings=s, source_uri=uri,
        )  # fmt: skip
        job_ids.append(r.job_id)
        print(f"enqueued {uri}  ({'new' if r.created else 'already known'}, status {r.status})")

    print("\nstarting the real worker (burst mode: it stops when the queue is empty)...")
    started = time.time()
    worker = subprocess.Popen([sys.executable, "-m", "app.worker", "--burst"])
    worker.wait()
    print(f"worker finished in {time.time() - started:.0f}s (exit code {worker.returncode})\n")

    print(f"{'status':10} {'stage':6} {'tries':5} {'chunks':>6}  document")
    for job_id, (uri, _) in zip(job_ids, files, strict=True):
        status, stage, attempts, error, result = owner.execute(
            "SELECT status, stage, attempts, error, result FROM ingestion_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        n = (result or {}).get("children", "-")
        print(
            f"{status:10} {stage:6} {attempts:5} {n!s:>6}  {uri}"
            + (f"   ERROR: {error}" if error else "")
        )


if __name__ == "__main__":
    main()
