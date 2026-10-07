"""Experiments 1 and 2 (tasks 6.2, 6.3): chunker A vs B, then chunk size / overlap, on all of Tenant A.

Each config re-chunks the same files, embeds with the production model, and searches in memory
(dense only, so the chunking is the only thing that changes). Scored with the real eval metrics
on Tenant A's answerable, non-follow-up questions. Embeddings are cached by text, so overlapping
configs do not pay twice. Free (local model); slow (about 5-10 min per config).

Run: uv run python -m evals.chunk_experiments [--configs name,name]
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.embeddings import get_embedder, get_token_counter
from app.ingest.chunking import chunk_recursive
from app.ingest.chunking_structured import chunk_structured
from app.ingest.contextual import embedding_text
from app.ingest.parsers import parse
from app.ingest.preprocess import preprocess
from evals.corpus import GITLAB_SLICE, PDFS, RAW
from evals.metrics import evaluate
from evals.schema import EvalItem
from scripts.bulk_ingest import GITLAB, markdown_files

CACHE = Path("data/cache/chunk_experiments_embeds.npz")


@dataclass
class Cfg:
    strategy: str  # recursive | structured
    tokens: int
    overlap: int
    parent: int = 1200


CONFIGS: dict[str, Cfg] = {
    "recursive-400/50": Cfg("recursive", 400, 50),
    "structured-400/50 (production)": Cfg("structured", 400, 50),
    "structured-200/25": Cfg("structured", 200, 25),
    "structured-300/40": Cfg("structured", 300, 40),
    "structured-400/0": Cfg("structured", 400, 0),
    "structured-400/100": Cfg("structured", 400, 100),
    "structured-500/50": Cfg("structured", 500, 50),
}


def tenant_a_files() -> list[tuple[str, str, bytes]]:
    files = [(u, "md", p.read_bytes()) for u, p in markdown_files(GITLAB, "gitlab", GITLAB_SLICE)]
    files += [(n, "pdf", (RAW / "hr-pdfs" / n).read_bytes()) for n in PDFS.values()]
    files.append(
        (
            "acme-benefits-policy.docx",
            "docx",
            Path("data/samples/acme-benefits-policy.docx").read_bytes(),
        )
    )
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(CONFIGS))
    args = parser.parse_args()
    s = get_settings()
    count, embedder = get_token_counter(s), get_embedder(s)
    items = [EvalItem(**json.loads(line)) for line in open("evals/qa.jsonl")]
    items = [i for i in items if i.tenant == "A" and i.kind != "follow_up"]
    qvec = dict(
        zip(
            (i.id for i in items),
            embedder.embed_documents([i.question for i in items]),
            strict=True,
        )
    )
    parsed = [(u, preprocess(parse(t, raw, u))) for u, t, raw in tenant_a_files()]
    cache: dict[str, np.ndarray] = {}
    if CACHE.exists():
        with np.load(CACHE) as z:
            cache = {k: z[k] for k in z.files}
    print(f"{len(items)} questions, {len(parsed)} documents")
    print(
        "| config | chunks | tokens median | R@1 | R@5 | R@10 | MRR | nDCG@10 |\n|---|---|---|---|---|---|---|---|"
    )
    for name in args.configs.split(","):
        cfg = CONFIGS[name]
        chunks: list[Any] = []
        for uri, doc in parsed:
            made = (chunk_recursive(doc, cfg.tokens, cfg.overlap, count) if cfg.strategy == "recursive"
                    else chunk_structured(doc, cfg.tokens, cfg.overlap, cfg.parent, count))  # fmt: skip
            for c in made:
                if c.level == 0:
                    c.source_uri = uri
                    chunks.append(c)
        texts = [embedding_text(c) for c in chunks]
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = sorted({(k, t) for k, t in zip(keys, texts, strict=True) if k not in cache})
        started = time.perf_counter()
        for i in range(0, len(missing), 256):
            batch = missing[i : i + 256]
            for (k, _), v in zip(
                batch, embedder.embed_documents([t for _, t in batch]), strict=True
            ):
                cache[k] = np.array(v, dtype=np.float32)
        np.savez(CACHE, **cache)
        matrix = np.stack([cache[k] for k in keys])

        def results(
            item: EvalItem, matrix: np.ndarray = matrix, chunks: list[Any] = chunks
        ) -> list[Any]:
            order = np.argsort(-(matrix @ np.array(qvec[item.id])))[:50]
            return [chunks[i] for i in order]

        rep = evaluate(items, results)
        o = rep.overall
        sizes = sorted(count(c.text) for c in chunks)
        print(f"| {name} | {len(chunks)} | {sizes[len(sizes) // 2]} | {o['recall@1']:.2f} | {o['recall@5']:.2f} | "
              f"{o['recall@10']:.2f} | {o['mrr']:.2f} | {o['ndcg@10']:.2f} |  (embedded {len(missing)} new in {time.perf_counter() - started:.0f}s)", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
