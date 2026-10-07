"""Read one slice of an organisation's chunks for facts and store them as graph edges.

The slice is chosen by document path prefixes (a section of a handbook). Chunks already read are
skipped, so a run can be repeated or resumed. Runs as the DB OWNER (an admin tool).

Run: uv run python -m scripts.build_graph_slice --org "Tenant B (PostHog handbook)" \
        --prefix posthog/cs-and-onboarding --prefix posthog/onboarding --prefix posthog/support [--yes]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg

from app.config import get_settings
from app.cost_guard import add_cost_args, estimate_llm_cost, guard_from_args
from app.ingest.graph import extract_triples, load_slice_chunks, store_triples
from app.log import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True)
    parser.add_argument(
        "--prefix", action="append", required=True, help="document path prefix; repeatable"
    )
    parser.add_argument("--workers", type=int, default=4)
    add_cost_args(parser)
    args = parser.parse_args()
    setup_logging("WARNING")
    s = get_settings()
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    org = conn.execute("SELECT id FROM orgs WHERE name = %s", (args.org,)).fetchone()
    if org is None:
        raise SystemExit(f"no organisation named {args.org!r}")
    todo = load_slice_chunks(conn, org[0], args.prefix)
    print(f"{len(todo)} chunks to read in {args.prefix}", flush=True)
    guard = guard_from_args(
        args, "graph slice", estimate_llm_cost(s.llm_default_model, 650, 250, calls=len(todo))
    )  # fmt: skip
    guard.confirm()
    edges = done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_triples, row[1]): row for row in todo}
        for future in as_completed(futures):
            row = futures[future]
            triples, cost = future.result()
            guard.record(cost)
            store_triples(conn, org[0], row[0], row[2], row[3], triples)
            edges += len(triples)
            done += 1
            if done % 50 == 0:
                print(
                    f"  {done}/{len(todo)} chunks read, {edges} facts, ${guard.spent_usd:.3f}",
                    flush=True,
                )
    print(f"done: {done} chunks, {edges} facts, spent ${guard.spent_usd:.4f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
