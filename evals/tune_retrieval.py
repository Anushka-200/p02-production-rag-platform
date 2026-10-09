"""Two general retrieval settings, tuned WITHOUT overfitting: rerank candidates and the keyword weight in RRF.

Why these two (from reading the misses, evals/results/misses.json): many gold chunks sit at fused rank 11-40, where the
reranker never looks (it re-orders only the first RERANK_CANDIDATES=10), and the weak keyword list can push the gold down.
Both are global settings, not per-question rules. To avoid fitting the questions: the 125 answerable non-follow-up
questions are split in two halves by a hash of their id; a setting only counts as a gain if it helps BOTH halves.
Free (local models). Run: uv run python -m evals.tune_retrieval
"""

import hashlib
import statistics
import time
from typing import Any

import psycopg

from app.config import get_settings
from app.embeddings import get_embedder
from app.retrieval.rerank import _passage, get_reranker
from app.retrieval.retrieve import retrieve
from evals.metrics import first_ranks, item_metrics
from evals.run import scopes_from_db
from evals.schema import EvalItem

WEIGHTS = (0.3, 0.6)  # 0.3 is today's default
CANDIDATES = (0, 10, 20, 30)  # 0 = no reranker


def half(item_id: str) -> str:
    return "A" if int(hashlib.sha1(item_id.encode()).hexdigest(), 16) % 2 == 0 else "B"


def main() -> None:
    s = get_settings()
    items = [EvalItem.model_validate_json(line) for line in open("evals/qa.jsonl")]
    items = [i for i in items if i.answerable and i.gold and i.kind != "follow_up"]
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker = get_embedder(s), get_reranker(s)
    assert reranker is not None
    scopes = scopes_from_db(conn)
    print(
        f"{len(items)} questions: half A {sum(half(i.id) == 'A' for i in items)}, half B {sum(half(i.id) == 'B' for i in items)}"
    )
    rows: dict[tuple[float, int, str], list[dict[str, float]]] = {}
    secs: dict[int, float] = {}
    for w in WEIGHTS:
        sw = s.model_copy(update={"rrf_keyword_weight": w})
        spent, pairs = 0.0, 0
        for item in items:
            fused = retrieve(
                conn, embedder, item.question, scopes[item.tenant], settings=sw, mode="hybrid"
            ).items
            top = fused[:30]
            t = time.perf_counter()
            scores = reranker.score(item.question, [_passage(f, s.rerank_max_words) for f in top])
            spent += time.perf_counter() - t
            pairs += len(top)
            for k in CANDIDATES:
                if k == 0:
                    order = [f.candidate for f in fused]
                else:
                    ranked = sorted(range(k), key=lambda i, sc=scores: (-sc[i], i))  # type: ignore[misc]
                    order = [top[i].candidate for i in ranked] + [f.candidate for f in fused[k:]]
                m = item_metrics(first_ranks(order, item.gold))
                for h in (half(item.id), "all"):
                    rows.setdefault((w, k, h), []).append(m)
        secs[0] = spent / pairs  # seconds per reranked pair
        print(f"weight {w}: reranker {1000 * spent / pairs:.0f} ms per pair", flush=True)
    per_pair = secs[0]
    print(
        "\n| keyword weight | rerank candidates | R@5 half A | R@5 half B | R@5 all | R@1 all | MRR all | added rerank s |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for w in WEIGHTS:
        for k in CANDIDATES:

            def avg(h: str, name: str, w: float = w, k: int = k) -> float:
                return statistics.mean(m[name] for m in rows[(w, k, h)])

            print(
                f"| {w}{' (current)' if w == 0.3 and k == 10 else ''} | {k or 'none'} | {avg('A', 'recall@5'):.3f} | {avg('B', 'recall@5'):.3f} | {avg('all', 'recall@5'):.3f} | {avg('all', 'recall@1'):.3f} | {avg('all', 'mrr'):.3f} | {k * per_pair:.2f} |"
            )  # noqa: E501
    _: Any = None


if __name__ == "__main__":
    main()
