"""Sync a git repository (a handbook, a docs site) into an organisation. Runs as the DB OWNER.

Only what changed since the last sync is read, re-chunked and re-embedded. Use --dry-run first.

Run: uv run python -m scripts.sync_connector --org "Tenant A (GitLab handbook)" --name git:gitlab \
        --repo data/raw/gitlab-handbook --rev HEAD --path content/handbook/people-group \
        --uri-prefix gitlab/people-group [--visibility org] [--rule people-group/private/=private] [--dry-run]
"""

import argparse

import psycopg

from app.config import get_settings
from app.connectors.git import GitConnector
from app.connectors.sync import make_pipeline_ingest, sync_connector
from app.embeddings import get_embedder, get_token_counter
from app.log import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True)
    parser.add_argument(
        "--owner-email", help="owner of the synced documents (default: first admin)"
    )
    parser.add_argument("--name", required=True, help="connector id, e.g. git:gitlab")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--path", default="", help="only pages under this path in the repo")
    parser.add_argument("--uri-prefix", default="")
    parser.add_argument("--visibility", default="org", choices=["private", "org", "public"])
    parser.add_argument("--rule", action="append", default=[], help="PREFIX=VISIBILITY, repeatable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-mass-delete", action="store_true")
    args = parser.parse_args()
    setup_logging("WARNING")
    s = get_settings()
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    org = conn.execute("SELECT id FROM orgs WHERE name = %s", (args.org,)).fetchone()
    if org is None:
        raise SystemExit(f"no organisation named {args.org!r}")
    owner = conn.execute(
        "SELECT id FROM users WHERE org_id = %s AND (%s::text IS NULL AND role = 'admin' "
        "OR lower(email) = lower(%s)) ORDER BY created_at LIMIT 1",
        (org[0], args.owner_email, args.owner_email),
    ).fetchone()
    if owner is None:
        raise SystemExit("no admin user to own the documents; pass --owner-email")
    rules = [tuple(r.split("=", 1)) for r in args.rule]
    connector = GitConnector(
        args.repo, args.rev, name=args.name, path_prefix=args.path, uri_prefix=args.uri_prefix,
        default_visibility=args.visibility, visibility_rules=rules,  # type: ignore[arg-type]
    )  # fmt: skip
    ingest = make_pipeline_ingest(
        conn, org[0], owner[0], settings=s, embedder=get_embedder(s), count=get_token_counter(s)
    )  # fmt: skip
    report = sync_connector(
        conn, connector, org_id=org[0], owner_id=owner[0], ingest=ingest, dry_run=args.dry_run,
        allow_mass_delete=args.allow_mass_delete,
    )  # fmt: skip
    print(report.summary())
    for uri, why in report.failed[:10]:
        print("  FAILED", uri, why)


if __name__ == "__main__":
    main()
