"""Experiment 7 (task 6.8): HNSW ef_search sweep, with the production setting (iterative scan) on.

Per setting: top-10 overlap with an EXACT search (index scans off) and SQL latency p50/p95.
Free: local query embeddings, SQL only. Run as the DB owner against the dev database.

Run: uv run python -m evals.hnsw_sweep
"""

import json
import statistics
import time

import psycopg

from app.config import get_settings
from app.embeddings import get_embedder, vector_literal
from app.retrieval.search import _ALLOWED, Scope

EF_VALUES = [10, 20, 40, 100, 200, 400]
SQL = (
    "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id "  # noqa: S608
    f"WHERE {_ALLOWED} AND c.embedding IS NOT NULL ORDER BY c.embedding <=> %(vec)s::vector LIMIT 10"
)
TENANTS = {"A": "Tenant A (GitLab handbook)", "B": "Tenant B (PostHog handbook)"}


def main() -> None:
    s = get_settings()
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder = get_embedder(s)
    scopes = {}
    for key, name in TENANTS.items():
        org, user = conn.execute(
            "SELECT o.id, u.id FROM orgs o JOIN users u ON u.org_id = o.id WHERE o.name = %s",
            (name,),
        ).fetchone()  # type: ignore[misc]
        scopes[key] = Scope(org, user)
    items = [json.loads(line) for line in open("evals/qa.jsonl")]
    items = [i for i in items if i["kind"] != "follow_up"]
    vecs = embedder.embed_documents([i["question"] for i in items])
    rows = list(zip(items, vecs, strict=True))

    def run(ef: int | None, repeats: int = 3) -> tuple[list[set[object]], list[float]]:
        found, times = [], []
        for item, vec in rows:
            params = {
                "vec": vector_literal(vec),
                "org": scopes[item["tenant"]].org_id,
                "user": scopes[item["tenant"]].user_id,
            }  # noqa: E501
            best = float("inf")
            for _ in range(repeats):
                with conn.transaction():
                    if ef is None:
                        conn.execute("SET LOCAL enable_indexscan = off")
                        conn.execute("SET LOCAL enable_bitmapscan = off")
                    else:
                        conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef)}")
                        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                    t = time.perf_counter()
                    got = conn.execute(SQL, params).fetchall()
                    best = min(best, time.perf_counter() - t)
            found.append({r[0] for r in got})
            times.append(best * 1000)
        return found, times

    exact, exact_t = run(None, repeats=1)
    print(
        f"{len(rows)} questions, {conn.execute('SELECT count(*) FROM chunks').fetchone()[0]} chunks in the table"
    )  # type: ignore[index]
    print("| ef_search | top-10 overlap with exact | SQL p50 ms | SQL p95 ms |\n|---|---|---|---|")
    for ef in EF_VALUES:
        found, times = run(ef)
        overlap = statistics.mean(len(a & b) / 10 for a, b in zip(found, exact, strict=True))
        qs = statistics.quantiles(times, n=20)
        print(f"| {ef} | {overlap:.3f} | {statistics.median(times):.1f} | {qs[18]:.1f} |")
    qs = statistics.quantiles(exact_t, n=20)
    print(f"| exact (no index) | 1.000 | {statistics.median(exact_t):.1f} | {qs[18]:.1f} |")


if __name__ == "__main__":
    main()
