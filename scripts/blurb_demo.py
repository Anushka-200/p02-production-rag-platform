"""Try contextual blurbs on a few REAL pages with the live model (costs a few cents).

Run: uv run python -m scripts.blurb_demo [--yes] [--max-usd 0.5]
"""

import argparse
from pathlib import Path

from app.cost_guard import add_cost_args, guard_from_args
from app.embeddings import get_token_counter
from app.ingest.chunking_structured import chunk_structured
from app.ingest.contextual import BlurbCache, BlurbStats, add_blurbs, project_blurb_cost
from app.ingest.parsers import parse_markdown
from app.ingest.preprocess import preprocess
from app.log import setup_logging

ROOT = Path("data/raw/gitlab-handbook/content/handbook/total-rewards")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_cost_args(parser)
    args = parser.parse_args()
    setup_logging("WARNING")
    count = get_token_counter()

    docs = []
    for f in sorted(ROOT.rglob("*.md")):
        d = preprocess(parse_markdown(f.read_bytes(), f.name))
        tokens = sum(count(s.text) for s in d.sections)
        chunks = chunk_structured(d, 400, 50, 1200, count)
        if sum(c.level == 0 for c in chunks) >= 3:
            docs.append((tokens, d, chunks))
    docs.sort(key=lambda x: x[0])
    big = next(x for x in reversed(docs) if x[0] < 9000)  # the biggest page under ~9k tokens
    picks = [docs[len(docs) // 3], big]
    items = [(d, c) for _, d, c in picks]

    from app.config import get_settings

    model = get_settings().llm_default_model
    projected = project_blurb_cost(items, model, count)
    guard = guard_from_args(args, "blurb demo", projected)
    guard.confirm()

    stats = BlurbStats()
    cache = BlurbCache(Path("data/cache/blurbs.jsonl"))
    for tokens, d, chunks in picks:
        before = (stats.calls, stats.cached_input_tokens, stats.cost_usd)
        out = add_blurbs(d, chunks, guard=guard, stats=stats, cache=cache, count=count)
        kids = [c for c in out if c.level == 0]
        print(f"\n=== {d.title} | doc {tokens} tokens | {len(kids)} child chunks")
        print(
            f"    calls {stats.calls - before[0]}, "
            f"cached input tokens {stats.cached_input_tokens - before[1]}, "
            f"cost ${stats.cost_usd - before[2]:.4f}"
        )
        for c in kids[:2]:
            print(f"    CHUNK : {c.text[:110]!r}")
            print(f"    BLURB : {c.context_blurb!r}")
    print("\n", guard.summary(), "| cache hits:", stats.cache_hits, "| failures:", stats.failures)


if __name__ == "__main__":
    main()
