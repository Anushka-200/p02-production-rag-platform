"""Choosing the semantic-cache threshold (task 6.7) from real questions. Free: local embeddings only.

Pretend the cache already holds an answer for every OTHER eval question. For each question, find its
nearest neighbour. At threshold t the cache would serve that neighbour's answer when similarity >= t:
  correct hit  the neighbour has the same gold location (same answer is right)
  wrong hit    different gold, or the question is unanswerable (a WRONG or unsupported answer is served)
We want the highest hit rate with (almost) no wrong hits.

Run: uv run python -m evals.cache_threshold
"""

import json

import numpy as np

from app.config import get_settings
from app.embeddings import get_embedder

THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.98]


def main() -> None:
    rows = [json.loads(line) for line in open("evals/qa.jsonl")]
    rows = [r for r in rows if r["kind"] != "follow_up"]  # the cache never serves follow-ups
    vecs = np.array(get_embedder(get_settings()).embed_documents([r["question"] for r in rows]))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1)
    gold = [{(g["doc"], g["locator"]) for g in r["gold"]} for r in rows]
    nearest = sims.argmax(axis=1)
    best = sims.max(axis=1)
    same = [bool(gold[i] & gold[j]) and rows[i]["answerable"] for i, j in enumerate(nearest)]
    para = [i for i, r in enumerate(rows) if r["kind"] == "paraphrase"]
    print(f"{len(rows)} questions; {len(para)} are paraphrases of another question")
    print("| threshold | cache hits | correct | WRONG | paraphrases served | wrong share of hits |")
    print("|---|---|---|---|---|---|")
    for t in THRESHOLDS:
        hit = best >= t
        correct = int(sum(h and s for h, s in zip(hit, same, strict=True)))
        wrong = int(hit.sum()) - correct
        served = sum(1 for i in para if hit[i] and same[i])
        share = f"{wrong / hit.sum():.0%}" if hit.sum() else "-"
        print(
            f"| {t:.2f} | {int(hit.sum())} | {correct} | {wrong} | {served}/{len(para)} | {share} |"
        )
    # the closest WRONG pairs: what do dangerous near-misses look like?
    bad = sorted(((best[i], i, nearest[i]) for i in range(len(rows)) if not same[i]), reverse=True)[
        :6
    ]
    print("\nclosest pairs that must NOT share an answer:")
    for s, i, j in bad:
        print(f"  {s:.3f}  {rows[i]['question'][:70]!r}  vs  {rows[j]['question'][:70]!r}")


if __name__ == "__main__":
    main()
