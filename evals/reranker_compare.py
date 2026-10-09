"""Which cross-encoder should re-order the top 10? Measured on the DEV questions only (evals/splits.py).

Same candidates for every model: the production hybrid top 30; each model re-orders the first K (K=10 is production). Metrics are the
real eval metrics (gold place or accepted alternative in the top k). Also the cost: seconds per question for K=10.
Free (local models). Test questions are NOT used here: a model is chosen on dev and confirmed once on test at the end.

Run: uv run python -m evals.reranker_compare [--models a,b] [--k 10]
"""

import argparse
import json
import statistics
import time
from typing import Any

import psycopg

from app.config import get_settings
from app.embeddings import get_embedder
from app.retrieval.rerank import LocalReranker, _passage
from app.retrieval.retrieve import retrieve
from evals.metrics import first_ranks, item_metrics
from evals.run import scopes_from_db
from evals.schema import EvalItem
from evals.splits import split_of

MODELS = [
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-turbo-en",
    "BAAI/bge-reranker-base",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()
    s = get_settings()
    items = [EvalItem.model_validate_json(line) for line in open("evals/qa.jsonl")]
    items = [
        i
        for i in items
        if i.answerable and i.gold and not i.history and split_of(i.id) == args.split
    ]
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, scopes = get_embedder(s), scopes_from_db(conn)
    fused: dict[str, list[Any]] = {
        i.id: retrieve(
            conn, embedder, i.question, scopes[i.tenant], settings=s, mode="hybrid"
        ).items[:30]
        for i in items
    }
    print(f"{len(items)} {args.split} questions (answerable, standalone)")
    print(f"| model | R@1 | R@5 | R@10 | MRR | seconds / question (K={args.k}) |")
    print("|---|---|---|---|---|---|")
    base = [i for i in items]
    rows = {}
    for name in args.models.split(","):
        model = LocalReranker(name)
        per, seconds = [], []
        for item in base:
            top = fused[item.id]
            t = time.perf_counter()
            scores = model.score(
                item.question, [_passage(f, s.rerank_max_words) for f in top[: args.k]]
            )
            seconds.append(time.perf_counter() - t)
            order = sorted(range(args.k), key=lambda i, sc=scores: (-sc[i], i))  # type: ignore[misc]
            ranked = [top[i].candidate for i in order] + [f.candidate for f in top[args.k :]]
            per.append(item_metrics(first_ranks(ranked, item.gold)))
        m = {
            k: statistics.mean(p[k] for p in per)
            for k in ("recall@1", "recall@5", "recall@10", "mrr")
        }
        rows[name] = {**m, "seconds": statistics.mean(seconds)}
        print(
            f"| {name.split('/')[-1]} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} | {statistics.mean(seconds):.2f} |",
            flush=True,
        )
        del model
    json.dump(rows, open(f"evals/results/reranker-compare-{args.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
