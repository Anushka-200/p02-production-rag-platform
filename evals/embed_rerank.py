"""Does the better embedding model still win once the reranker is applied? (follow-up to B-033)

Same 3,547 Tenant A chunks and cached vectors as evals/embedding_experiments.py. For each model: dense top-K, then the
production cross-encoder reranker re-orders those K, scored with the real eval metrics on the 87 Tenant A questions.
K=10 is production's setting (RERANK_CANDIDATES); K=20 shows whether more candidates help. Dense only (no
keyword leg), because the keyword leg lives in the database. Free; the reranker is local.

Run: uv run python -m evals.embed_rerank
"""

import hashlib
import json
import time
from typing import Any

import numpy as np

from app.config import get_settings
from app.embeddings import LocalEmbedder, get_token_counter
from app.ingest.chunking_structured import chunk_structured
from app.ingest.parsers import parse
from app.ingest.preprocess import preprocess
from app.retrieval.rerank import get_reranker
from evals.chunk_experiments import tenant_a_files
from evals.embedding_experiments import CACHE_DIR, MODELS, PREFIXES
from evals.metrics import evaluate
from evals.schema import EvalItem

NAMES = ["BAAI/bge-small-en-v1.5", "snowflake/snowflake-arctic-embed-m"]


def main() -> None:
    s = get_settings()
    count = get_token_counter(s)
    reranker = get_reranker(s)
    assert reranker is not None
    items = [EvalItem(**json.loads(line)) for line in open("evals/qa.jsonl")]
    items = [i for i in items if i.tenant == "A" and i.kind != "follow_up"]
    chunks: list[Any] = []
    for uri, kind, raw in tenant_a_files():
        for c in chunk_structured(preprocess(parse(kind, raw, uri)), 400, 50, 1200, count):
            if c.level == 0:
                c.source_uri = uri
                chunks.append(c)
    print(f"{len(items)} questions, {len(chunks)} chunks")
    print(
        "| model | candidates K | R@1 | R@5 | R@10 | MRR | rerank s / question |\n|---|---|---|---|---|---|---|"
    )
    for name in NAMES:
        qp, dp = PREFIXES.get(name, ("", ""))
        tag = name if not (qp or dp) else f"{name}|{qp}|{dp}"
        matrix = np.load(CACHE_DIR / f"emb_{hashlib.sha1(tag.encode()).hexdigest()[:8]}.npy")
        assert len(matrix) == len(chunks), "cache does not match the chunks"
        model = LocalEmbedder(name, MODELS[name], 16)
        qvec = {i.id: np.array(model.embed_query(qp + i.question)) for i in items}
        del model
        scored: dict[
            str, tuple[list[Any], list[float]]
        ] = {}  # question -> (dense top-20, reranker scores of them)
        spent = 0.0
        for item in (
            items
        ):  # the reranker sees each question's top 20 once; K=10 reuses the first 10 scores
            order = np.argsort(-(matrix @ qvec[item.id]))[:20]
            top = [chunks[i] for i in order]
            started = time.perf_counter()
            scores = reranker.score(
                item.question, [" ".join(c.text.split()[: s.rerank_max_words]) for c in top]
            )
            spent += time.perf_counter() - started
            scored[item.id] = (top, scores)
        for k in (0, 10, 20):  # 0 = dense only, no reranker

            def results(
                item: EvalItem, k: int = k, matrix: np.ndarray = matrix, qvec: dict = qvec
            ) -> list[Any]:  # type: ignore[type-arg]
                order = np.argsort(-(matrix @ qvec[item.id]))[:50]
                dense = [chunks[i] for i in order]
                if k == 0:
                    return dense
                top, scores = scored[item.id]  # noqa: B023
                ranked = [
                    c for _, c in sorted(zip(scores[:k], top[:k], strict=True), key=lambda p: -p[0])
                ]
                return ranked + dense[k:]

            o = evaluate(items, results).overall
            label = "dense only" if k == 0 else str(k)
            secs = "-" if k == 0 else f"{spent / len(items) * k / 20:.2f}"
            print(
                f"| {name.split('/')[-1]} | {label} | {o['recall@1']:.2f} | {o['recall@5']:.2f} | {o['recall@10']:.2f} | {o['mrr']:.2f} | {secs} |",
                flush=True,
            )  # noqa: E501


if __name__ == "__main__":
    main()
