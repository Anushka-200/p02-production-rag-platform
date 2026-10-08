"""Experiment 4 (task 6.5): contextual retrieval on all of Tenant A. COSTS MONEY (owner approved up to $7.50).

Same chunks twice: plain, and with a short Haiku-written blurb in front ("this chunk is from the Time
Off policy, about vacation days") that is embedded together with the chunk. Dense search in memory,
the real eval metrics, Tenant A answerable non-follow-up questions. Blurbs are cached on disk
(data/cache/blurbs-p6.jsonl), so a rerun or a crash does not pay twice.

Run: uv run python -m evals.contextual_experiment --yes
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.cost_guard import CostGuard
from app.embeddings import get_embedder, get_token_counter
from app.ingest.chunking_structured import chunk_structured
from app.ingest.contextual import (
    BlurbCache,
    BlurbStats,
    add_blurbs,
    embedding_text,
    project_blurb_cost,
)
from app.ingest.parsers import parse
from app.ingest.preprocess import preprocess
from evals.chunk_experiments import tenant_a_files
from evals.metrics import evaluate
from evals.schema import EvalItem

MAX_USD = 9.0  # hard stop; the owner approved about $7.50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    s = get_settings()
    count, embedder = get_token_counter(s), get_embedder(s)
    items = [EvalItem(**json.loads(line)) for line in open("evals/qa.jsonl")]
    items = [i for i in items if i.tenant == "A" and i.kind != "follow_up"]
    docs = []
    for uri, kind, raw in tenant_a_files():
        doc = preprocess(parse(kind, raw, uri))
        docs.append((uri, doc, chunk_structured(doc, 400, 50, 1200, count)))
    projected = project_blurb_cost([(d, c) for _, d, c in docs], s.llm_default_model, count) or 0.0
    guard = CostGuard(
        "contextual blurbs, all of Tenant A", projected, max_usd=MAX_USD, assume_yes=args.yes
    )
    guard.confirm()
    cache, stats = BlurbCache(Path("data/cache/blurbs-p6.jsonl")), BlurbStats()
    started = time.perf_counter()
    plain: list[Any] = []
    blurbed: list[Any] = []
    for n, (uri, doc, chunks) in enumerate(docs, 1):
        done = add_blurbs(doc, chunks, cache=cache, guard=guard, stats=stats, count=count)
        for before, after in zip(chunks, done, strict=True):
            if before.level == 0:
                before.source_uri = after.source_uri = uri
                plain.append(before)
                blurbed.append(after)
        if n % 20 == 0:
            print(
                f"  {n}/{len(docs)} documents, ${stats.cost_usd:.2f} so far, {time.perf_counter() - started:.0f}s",
                flush=True,
            )  # noqa: E501
    print(f"blurbs: {stats.calls} calls, {stats.cache_hits} cache hits, {stats.failures} failures, "
          f"${stats.cost_usd:.2f}, {stats.cached_input_tokens} cached input tokens", flush=True)  # fmt: skip
    qvec = dict(
        zip(
            (i.id for i in items),
            embedder.embed_documents([i.question for i in items]),
            strict=True,
        )
    )
    print("| variant | chunks | R@1 | R@5 | R@10 | MRR | nDCG@10 |\n|---|---|---|---|---|---|---|")
    for name, chunks in (("plain chunks", plain), ("with Haiku blurb", blurbed)):
        matrix = np.array(
            embedder.embed_documents([embedding_text(c) for c in chunks]), dtype=np.float32
        )

        def results(
            item: EvalItem, matrix: np.ndarray = matrix, chunks: list[Any] = chunks
        ) -> list[Any]:  # type: ignore[type-arg]
            return [chunks[i] for i in np.argsort(-(matrix @ np.array(qvec[item.id])))[:50]]

        o = evaluate(items, results).overall
        print(f"| {name} | {len(chunks)} | {o['recall@1']:.2f} | {o['recall@5']:.2f} | {o['recall@10']:.2f} | "
              f"{o['mrr']:.2f} | {o['ndcg@10']:.2f} |", flush=True)  # fmt: skip
    print(guard.summary())


if __name__ == "__main__":
    main()
