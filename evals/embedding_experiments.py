"""Experiment 5 (task 6.6): compare local embedding models on all of Tenant A (ADR 006 checklist).

Same chunks for every model (production chunker, bge tokenizer), dense search in memory, the real eval
metrics. Reports the whole set AND the PDF/DOCX questions apart (the most specialised text), plus
embedding speed. Each model keeps its own vector cache. Free; slow (about 10-40 min per model).

Run: uv run python -m evals.embedding_experiments [--models name,name]
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.embeddings import LocalEmbedder, get_token_counter
from app.ingest.chunking_structured import chunk_structured
from app.ingest.contextual import embedding_text
from app.ingest.parsers import parse
from app.ingest.preprocess import preprocess
from evals.chunk_experiments import tenant_a_files
from evals.metrics import evaluate
from evals.schema import EvalItem

MODELS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "snowflake/snowflake-arctic-embed-m": 768,
}
CACHE_DIR = Path("data/cache")
# (query prefix, document prefix) the model's authors require or recommend; fastembed adds none of them.
# BGE v1.5 works without an instruction. A first nomic run WITHOUT its prefixes scored R@5 0.67 (unfair).
PREFIXES = {
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "snowflake/snowflake-arctic-embed-m": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()
    count = get_token_counter(get_settings())
    items = [EvalItem(**json.loads(line)) for line in open("evals/qa.jsonl")]
    items = [i for i in items if i.tenant == "A" and i.kind != "follow_up"]
    chunks: list[Any] = []
    for uri, kind, raw in tenant_a_files():
        for c in chunk_structured(preprocess(parse(kind, raw, uri)), 400, 50, 1200, count):
            if c.level == 0:
                c.source_uri = uri
                chunks.append(c)
    texts = [embedding_text(c) for c in chunks]
    print(f"{len(items)} questions, {len(chunks)} chunks")
    print(
        "| model | dim | R@1 | R@5 | R@10 | MRR | PDF/DOCX R@5 (n) | markdown R@5 | embed s / 1000 chunks | query ms |"
    )  # noqa: E501
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in args.models.split(","):
        model = LocalEmbedder(name, MODELS[name], 16)
        qp, dp = PREFIXES.get(name, ("", ""))
        tag = name if not (qp or dp) else f"{name}|{qp}|{dp}"
        cache_file = CACHE_DIR / f"emb_{hashlib.sha1(tag.encode()).hexdigest()[:8]}.npy"
        started = time.perf_counter()
        if cache_file.exists():
            matrix, per_1000 = np.load(cache_file), float("nan")
        else:
            matrix = np.array(model.embed_documents([dp + t for t in texts]), dtype=np.float32)
            per_1000 = (time.perf_counter() - started) / len(texts) * 1000
            np.save(cache_file, matrix)
        t = time.perf_counter()
        qvec = {i.id: np.array(model.embed_query(qp + i.question)) for i in items}
        q_ms = (time.perf_counter() - t) / len(items) * 1000

        def results(item: EvalItem, matrix: np.ndarray = matrix, qvec: dict = qvec) -> list[Any]:  # type: ignore[type-arg]  # noqa: E501
            order = np.argsort(-(matrix @ qvec[item.id]))[:50]
            return [chunks[i] for i in order]

        rep = evaluate(items, results)
        o = rep.overall
        pdf = [
            s for k, s in rep.subsets.items() if k.startswith("corpus=") and k != "corpus=gitlab"
        ]
        n_pdf = int(sum(s["n"] for s in pdf))
        pdf_r5 = sum(s["recall@5"] * s["n"] for s in pdf) / max(n_pdf, 1)
        md = rep.subsets.get("corpus=gitlab", {})
        print(f"| {name} | {MODELS[name]} | {o['recall@1']:.2f} | {o['recall@5']:.2f} | {o['recall@10']:.2f} | "
              f"{o['mrr']:.2f} | {pdf_r5:.2f} ({n_pdf}) | {md.get('recall@5', 0):.2f} | {per_1000:.0f} | {q_ms:.0f} |", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
