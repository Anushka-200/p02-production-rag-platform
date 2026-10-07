"""Compare our chunker with LangChain's splitters on the Tenant A people-group slice (ADR 010).

Same pages, same embedding model, dense search in memory (numpy), document-level hit: is a gold
document among the top-k chunks' documents? Informal: 14 questions, unreviewed gold. No model cost.

Run: uv run --with langchain-text-splitters python -m evals.chunker_compare
"""

import json
import statistics
from collections.abc import Callable
from pathlib import Path

import numpy as np
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import get_settings
from app.embeddings import get_embedder, get_token_counter
from app.ingest.chunking_structured import chunk_structured
from app.ingest.contextual import embedding_text
from app.ingest.parsers import parse_markdown

ROOT = Path("data/raw/gitlab-handbook")
SLICE = ROOT / "content/handbook/people-group"
PREFIX = "gitlab/people-group/"
TOKENS, OVERLAP = 400, 50


def pages() -> dict[str, bytes]:
    return {PREFIX + str(p.relative_to(SLICE)): p.read_bytes() for p in sorted(SLICE.rglob("*.md"))}


def questions() -> list[dict]:  # type: ignore[type-arg]
    rows = [json.loads(line) for line in open("evals/qa.jsonl")]
    return [
        q for q in rows
        if q["answerable"] and q["kind"] != "follow_up"
        and any(g["doc"].startswith(PREFIX) for g in q["gold"])
    ]  # fmt: skip


def main() -> None:
    s = get_settings()
    count = get_token_counter(s)
    embedder = get_embedder(s)
    raw = pages()

    def ours() -> list[tuple[str, str]]:
        out = []
        for uri, data in raw.items():
            doc = parse_markdown(data, uri)
            out += [
                (uri, embedding_text(c))
                for c in chunk_structured(doc, TOKENS, OVERLAP, 1200, count)
                if c.level == 0
            ]  # noqa: E501
        return out

    def recursive() -> list[tuple[str, str]]:
        sp = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN, chunk_size=TOKENS, chunk_overlap=OVERLAP, length_function=count
        )
        return [(u, t) for u, d in raw.items() for t in sp.split_text(d.decode("utf-8", "replace"))]

    def headers() -> list[tuple[str, str]]:
        hs = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")])
        sp = RecursiveCharacterTextSplitter(
            chunk_size=TOKENS, chunk_overlap=OVERLAP, length_function=count
        )
        out = []
        for uri, data in raw.items():
            for part in sp.split_documents(hs.split_text(data.decode("utf-8", "replace"))):
                path = " > ".join(v for k, v in sorted(part.metadata.items()))
                out.append((uri, f"{path}\n{part.page_content}"))
        return out

    variants: dict[str, Callable[[], list[tuple[str, str]]]] = {
        "ours (structured, heading-aware)": ours,
        "LangChain recursive (markdown separators)": recursive,
        "LangChain header split + recursive": headers,
    }
    qs = questions()
    qvec = np.array(embedder.embed_documents([q["question"] for q in qs]))
    print(f"{len(raw)} pages, {len(qs)} questions\n")
    print("| chunker | chunks | tokens mean / p95 / max | hit@1 | hit@5 | MRR |")
    print("|---|---|---|---|---|---|")
    for name, build in variants.items():
        chunks = [(u, t) for u, t in build() if t.strip()]
        sizes = [count(t) for _, t in chunks]
        vecs = np.array(embedder.embed_documents([t for _, t in chunks]))
        hit1 = hit5 = 0
        rr = 0.0
        for q, v in zip(qs, qvec, strict=True):
            order = np.argsort(-(vecs @ v))
            gold = {g["doc"] for g in q["gold"]}
            seen: list[str] = []
            for i in order[:50]:
                if chunks[i][0] not in seen:
                    seen.append(chunks[i][0])
            rank = next((r for r, d in enumerate(seen, 1) if d in gold), None)
            hit1 += rank == 1
            hit5 += rank is not None and rank <= 5
            rr += 1 / rank if rank else 0
        p95 = int(np.percentile(sizes, 95))
        print(f"| {name} | {len(chunks)} | {statistics.mean(sizes):.0f} / {p95} / {max(sizes)} | "
              f"{hit1 / len(qs):.2f} | {hit5 / len(qs):.2f} | {rr / len(qs):.2f} |")  # fmt: skip


if __name__ == "__main__":
    main()
