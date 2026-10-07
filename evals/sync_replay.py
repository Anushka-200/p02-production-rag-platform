"""Replay real edits from a public handbook's git history into a scratch org, one sync per step.

Measures, per step, how many pages/chunks were re-embedded versus what a full rebuild would embed.
Local embeddings only: no model cost. Run as the DB OWNER.

Run: uv run python -m evals.sync_replay --org "Scratch Replay" --email replay@scratch.example
"""

import argparse
import subprocess
import time

import psycopg

from app.config import get_settings
from app.connectors.git import GitConnector
from app.connectors.sync import make_pipeline_ingest, sync_connector
from app.embeddings import get_embedder, get_token_counter
from app.log import setup_logging

REPO = "data/raw/gitlab-handbook"
PATH = "content/handbook/people-group"
STEPS = ["2026-08-01", "2026-08-15", "2026-09-01", "2026-09-15", "2026-10-01"]


def rev_at(date: str) -> str:
    out = subprocess.run(
        ["git", "-C", REPO, "rev-list", "-1", f"--before={date}", "HEAD", "--", PATH],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True)
    parser.add_argument("--email", required=True)
    args = parser.parse_args()
    setup_logging("ERROR")
    s = get_settings()
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    org = conn.execute("SELECT id FROM orgs WHERE name = %s", (args.org,)).fetchone()
    owner = conn.execute(
        "SELECT id FROM users WHERE lower(email) = lower(%s)", (args.email,)
    ).fetchone()
    if org is None or owner is None:
        raise SystemExit("create the org and user first with scripts.create_user")
    ingest = make_pipeline_ingest(
        conn, org[0], owner[0], settings=s, embedder=get_embedder(s), count=get_token_counter(s)
    )  # fmt: skip
    print(
        "| step (date) | commit | changed+added pages | unchanged | chunks embedded | total chunks | embedded vs full rebuild | seconds |"
    )  # noqa: E501
    print("|---|---|---|---|---|---|---|---|")
    for date in STEPS:
        rev = rev_at(date)
        before = conn.execute(
            "SELECT count(*) FROM chunks WHERE org_id = %s", (org[0],)
        ).fetchone()[0]  # type: ignore[index]
        started = time.perf_counter()
        connector = GitConnector(
            REPO, rev, name="git:replay", path_prefix=PATH, uri_prefix="gitlab/people-group"
        )
        r = sync_connector(conn, connector, org_id=org[0], owner_id=owner[0], ingest=ingest)
        seconds = time.perf_counter() - started
        # chunks written by this step = chunks of the pages this step added or changed
        touched = r.added + r.changed
        embedded = conn.execute(
            "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.org_id = %s AND d.source_uri = ANY(%s) AND c.version = d.active_version AND c.level = 0",
            (org[0], touched),
        ).fetchone()[0]  # type: ignore[index]
        total = conn.execute(
            "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.org_id = %s AND c.version = d.active_version AND c.level = 0",
            (org[0],),
        ).fetchone()[0]  # type: ignore[index]
        pct = f"{embedded}/{total} = {100 * embedded / max(total, 1):.0f}%"
        print(
            f"| {date} | {rev[:8]} | {len(touched)} | {r.unchanged} | {embedded} | {total} | {pct} | {seconds:.0f} |"
        )  # noqa: E501
        del before


if __name__ == "__main__":
    main()
