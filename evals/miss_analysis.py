"""Why is Recall@5 not 1? List every answerable question whose gold place is NOT in the top 5, with the evidence.

Runs the production retrieval (hybrid + rerank) on every answerable, non-follow-up question and, for each miss, records:
the gold rank under each stage (dense, keyword, hybrid, hybrid+rerank), whether the gold DOCUMENT is anywhere in the top 50,
what actually ranked at the top, and the question kind. Free (local models). The classification is done by reading the
output (evals/results/misses.json), not by a script, so nothing is tuned to the data.

Run: uv run python -m evals.miss_analysis
"""

import json
from pathlib import Path

import psycopg

from app.config import get_settings
from app.embeddings import get_embedder
from app.retrieval.rerank import get_reranker
from evals.metrics import covers, first_ranks
from evals.run import _ranker, scopes_from_db
from evals.schema import EvalItem

OUT = Path("evals/results/misses.json")


def main() -> None:
    s = get_settings()
    items = [EvalItem.model_validate_json(line) for line in open("evals/qa.jsonl")]
    items = [i for i in items if i.answerable and i.gold and i.kind != "follow_up"]
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker = get_embedder(s), get_reranker(s)
    scopes = scopes_from_db(conn)
    rankers = {
        m: _ranker(m, conn, embedder, reranker, scopes, s)
        for m in ("dense", "keyword", "hybrid", "hybrid+rerank")
    }
    rows, total = [], 0
    for n, item in enumerate(items, 1):
        res = {m: r(item) for m, r in rankers.items()}
        ranks = {m: first_ranks(v, item.gold) for m, v in res.items()}
        final = ranks["hybrid+rerank"]
        total += 1
        if all(r is not None and r <= 5 for r in final):  # every gold place found in the top 5
            continue
        top = res["hybrid+rerank"]
        rows.append({
            "id": item.id, "kind": item.kind, "tenant": item.tenant, "question": item.question,
            "gold": [g.model_dump() for g in item.gold], "gold_answer": item.gold_answer[:300],
            "ranks": ranks,
            "gold_doc_in_top50": [any(c.source_uri == g.doc for c in top) for g in item.gold],
            "top5": [{"doc": c.source_uri, "section": c.section_path, "text": c.text[:160].replace("\n", " ")} for c in top[:5]],
            "gold_chunks": [{"doc": c.source_uri, "section": c.section_path, "anchors": c.anchors, "text": c.text[:160].replace("\n", " ")}
                            for c in top[:50] if any(covers(c, g) for g in item.gold)][:2],
        })  # fmt: skip
        if n % 20 == 0:
            print(f"  {n}/{len(items)}", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
    by_kind: dict[str, list[int]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(1)
    print(
        f"{len(rows)} of {total} questions miss at least one gold place in the top 5; by kind: { {k: len(v) for k, v in by_kind.items()} }"
    )
    print("gold found at ranks 6-50:", sum(1 for r in rows if any(x is not None and x > 5 for x in r["ranks"]["hybrid+rerank"])),
          "| never in the top 50:", sum(1 for r in rows if any(x is None for x in r["ranks"]["hybrid+rerank"])))  # fmt: skip
    print("wrote", OUT)


if __name__ == "__main__":
    main()
