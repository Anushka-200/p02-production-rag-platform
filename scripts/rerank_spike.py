"""SPIKE (task 2.13): how fast can the reranker be before it starts costing quality?

For each configuration this measures, on the same headline eval questions:
  - latency: rerank time per question (p50 / p95), model already loaded and warm
  - quality: Recall@1, Recall@5, MRR after reranking (informal: the questions are unreviewed)
and estimates the time to first token = retrieval (64 ms) + rerank + context (18 ms) + the model's first token (~900 ms).
RUN THIS ON AN IDLE MACHINE: other CPU work distorts the timings.
Run: uv run python -m scripts.rerank_spike
"""

import statistics
import time

import psycopg

from app.config import get_settings
from app.embeddings import get_embedder
from app.retrieval.rerank import LocalReranker
from app.retrieval.retrieve import retrieve
from evals.metrics import evaluate
from evals.run import percentile, reorder_with_reranker, scopes_from_db, split_items
from evals.schema import EvalItem

OTHER_STAGES_MS = 64 + 18 + 900  # retrieve + context + LLM time to first token (measured in B-012)
# (label, model, candidates, max_words, batch_size, threads, sort_by_length)
CONFIGS = [
    ("current: 20 cand, 220 words", "Xenova/ms-marco-MiniLM-L-6-v2", 20, 220, 16, None, False),
    ("+ length-sorted batches", "Xenova/ms-marco-MiniLM-L-6-v2", 20, 220, 16, None, True),
    ("+ sorted, 6 threads", "Xenova/ms-marco-MiniLM-L-6-v2", 20, 220, 16, 6, True),
    ("sorted, 20 cand, 120 words", "Xenova/ms-marco-MiniLM-L-6-v2", 20, 120, 16, None, True),
    ("sorted, 15 cand, 160 words", "Xenova/ms-marco-MiniLM-L-6-v2", 15, 160, 16, None, True),
    ("sorted, 10 cand, 220 words", "Xenova/ms-marco-MiniLM-L-6-v2", 10, 220, 16, None, True),
    ("sorted, 10 cand, 120 words", "Xenova/ms-marco-MiniLM-L-6-v2", 10, 120, 16, None, True),
    ("sorted, 5 cand, 160 words", "Xenova/ms-marco-MiniLM-L-6-v2", 5, 160, 16, None, True),
    (
        "jina turbo, sorted, 10 cand, 160",
        "jinaai/jina-reranker-v1-turbo-en",
        10,
        160,
        16,
        None,
        True,
    ),
]


def main() -> None:
    s = get_settings()
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, scopes = get_embedder(s), scopes_from_db(conn)
    items = [EvalItem.model_validate_json(line) for line in open("evals/qa.jsonl")]
    headline, _ = split_items(items)
    headline = [i for i in headline if i.answerable]
    print(f"retrieving once for {len(headline)} questions...")
    fused = {
        i.id: retrieve(
            conn, embedder, i.question, scopes[i.tenant], settings=s, mode="hybrid"
        ).items
        for i in headline
    }
    rankings = {i.id: [f.candidate for f in fused[i.id]] for i in headline}
    none = evaluate(headline, lambda it: rankings[it.id]).overall
    print(
        f"no reranker (hybrid): R@1 {none['recall@1']:.2f} R@5 {none['recall@5']:.2f} MRR {none['mrr']:.2f}\n"
    )
    print(
        f"{'configuration':36} | {'rerank p50':>10} {'p95':>6} | {'R@1':>5} {'R@5':>5} {'MRR':>5} | est. first token"
    )
    models: dict[tuple, LocalReranker] = {}
    for label, model, n, words, batch, threads, sort in CONFIGS:
        key = (model, batch, threads, sort)
        if key not in models:
            models[key] = LocalReranker(
                model, batch_size=batch, threads=threads, sort_by_length=sort
            )
            models[key].score("warm up", ["warm up text"] * 4)
        reranker, times, ranked = models[key], [], {}
        for item in headline:
            t = time.perf_counter()
            ranked[item.id] = reorder_with_reranker(
                fused[item.id], item.question, reranker, candidates=n, max_words=words
            )
            times.append((time.perf_counter() - t) * 1000)
        m = evaluate(headline, lambda it, r=ranked: r[it.id]).overall
        p50 = statistics.median(times)
        print(
            f"{label:36} | {p50:8.0f}ms {percentile(times, 0.95):4.0f}ms | {m['recall@1']:5.2f} {m['recall@5']:5.2f} {m['mrr']:5.2f} | {(p50 + OTHER_STAGES_MS) / 1000:.2f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
