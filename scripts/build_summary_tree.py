"""Build (or refresh) the summary tree for one organisation. Runs as the DB OWNER (an admin tool).

Each visibility group (public / org / each user's private) gets its own tree, so no summary ever
mixes content that different people may see. A group whose chunks have not changed since the last
build is skipped. The tree costs one small-model call per cluster of about 8 chunks, plus a few
more for the upper level: the cost guard prints the projection first.

Run: uv run python -m scripts.build_summary_tree --org "Tenant A (GitLab handbook)" [--yes]
"""

import argparse
import math

import psycopg

from app.config import get_settings
from app.cost_guard import add_cost_args, estimate_llm_cost, guard_from_args
from app.embeddings import get_embedder
from app.ingest.summarytree import (
    CLUSTER_SIZE,
    build_nodes,
    is_current,
    load_groups,
    sample_group,
    store_tree,
)
from app.log import setup_logging


def projected_calls(sizes: list[int]) -> int:
    """About n/8 summaries per group, plus ~15% for the upper level."""
    return sum(math.ceil(n / CLUSTER_SIZE) for n in sizes) * 115 // 100 + len(sizes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True)
    parser.add_argument(
        "--owner-email", help="owner of the org-wide and public summaries (default: first admin)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="build from a random sample of this many chunks per group (a cheap trial, not a real tree)",
    )
    add_cost_args(parser)
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
        raise SystemExit("no admin user to own the shared summaries; pass --owner-email")
    all_groups = load_groups(conn, org[0])
    if (
        args.sample
    ):  # a trial: keep it from replacing a real tree (it would hash differently anyway)
        all_groups = [sample_group(g, args.sample) for g in all_groups]
    groups = [g for g in all_groups if not is_current(conn, org[0], owner[0], g)]
    print("groups:", {g.key: len(g.ids) for g in all_groups})
    print("up to date, skipped:", [g.key for g in all_groups if g not in groups] or "none")
    guard = guard_from_args(
        args, "summary tree",
        estimate_llm_cost(s.llm_default_model, 2200, 200, calls=projected_calls([len(g.ids) for g in groups])),
    )  # fmt: skip
    guard.confirm()
    embedder = get_embedder(s)
    for group in groups:
        nodes, cost = build_nodes(group, embedder, on_call=guard.record)
        written = store_tree(conn, org[0], owner[0], group, nodes, embedder.model)
        by_level = {lv: sum(n.level == lv for n in nodes) for lv in (2, 3)}
        state = "written" if written else "unchanged (skipped)"
        print(
            f"  {group.key:<40} {len(group.ids):>5} chunks -> {by_level} summaries, ${cost:.4f}, {state}"
        )
    print(f"total spent ${guard.spent_usd:.4f}")


if __name__ == "__main__":
    main()
