"""Experiment (Phase 5 block 3): does reading the PARENT section put the answer in front of the model?

No model calls, so it is free and exactly repeatable. For every answerable single / multi-hop /
paraphrase question the production pipeline runs up to the context the answer model would be given
(hybrid search, rerank, optional parent swap, context budget), and we check:

    any gold   at least one of the question's gold places is inside the context
    all gold   EVERY gold place is inside (matters for multi-hop questions)
    tokens     how much of the 3000-token budget the context uses (dilution / cost)
    sources    how many source blocks the answer model has to read

Run: uv run python -m evals.context_experiments
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import psycopg

from app.config import Settings, get_settings
from app.embeddings import get_embedder, get_token_counter
from app.log import setup_logging
from app.retrieval.context import build_context
from app.retrieval.expand import retrieve_expanded
from app.retrieval.parents import expand_to_parents
from app.retrieval.rerank import get_reranker, rerank
from evals.run import _page_range, scopes_from_db
from evals.schema import EvalItem, GoldRef

QA = Path(__file__).parent / "qa.jsonl"
RESULTS = Path(__file__).parent / "results"
GROUPS = ("single", "multi_hop", "paraphrase", "broad", "relationship")
CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {"parent_expansion": False},
    "parents<=400": {"parent_expansion": True, "parent_max_tokens": 400},
    "parents<=700": {"parent_expansion": True, "parent_max_tokens": 700},
    "parents<=1200": {"parent_expansion": True, "parent_max_tokens": 1200},
    "tree": {"parent_expansion": False, "summary_tree": True},
    "tree-small": {
        "parent_expansion": False,
        "summary_tree": True,
        "summary_top_k": 3,
        "summary_members_per": 2,
    },
    "graph": {"parent_expansion": False, "graph_search": True},
    "tree+graph": {"parent_expansion": False, "summary_tree": True, "graph_search": True},
}


def covered(gold: GoldRef, sources: list[Any]) -> bool:
    return any(
        s.document == gold.doc
        and (
            gold.locator in s.anchors
            or (
                gold.locator.startswith("page:")
                and s.pages
                and int(gold.locator[5:]) in _page_range(s.pages)
            )
        )
        for s in sources
    )


def run(
    name: str,
    conn: psycopg.Connection,
    embedder: Any,
    reranker: Any,
    items: list[EvalItem],
    scopes: dict[str, Any],
    base: Settings,
) -> dict[str, Any]:
    s = base.model_copy(update=CONFIGS[name])
    count = get_token_counter(s)
    out: dict[str, Any] = {}
    for group in GROUPS:
        members = [i for i in items if i.kind == group and i.answerable]
        any_hit = all_hit = 0
        tokens: list[int] = []
        sources_n: list[int] = []
        for item in members:
            scope = scopes[item.tenant]
            r = retrieve_expanded(conn, embedder, item.question, scope, settings=s)
            ranked = rerank(
                r.items,
                item.question,
                reranker,
                candidates=s.rerank_candidates,
                top=s.rerank_top,
                max_words=s.rerank_max_words,
            )
            if s.parent_expansion:
                ranked = expand_to_parents(
                    conn, ranked, scope, max_parent_tokens=s.parent_max_tokens, count=count
                )
            ctx = build_context(ranked, s.context_budget_tokens, count)
            hits = [covered(g, ctx.sources) for g in item.gold]
            any_hit += any(hits)
            all_hit += all(hits)
            tokens.append(ctx.tokens)
            sources_n.append(len(ctx.sources))
        n = len(members)
        out[group] = {
            "n": n, "any_gold": round(any_hit / n, 3), "all_gold": round(all_hit / n, 3),
            "tokens": round(sum(tokens) / n), "sources": round(sum(sources_n) / n, 1),
        }  # fmt: skip
    return out


def main() -> None:
    setup_logging("WARNING")
    s = get_settings()
    items = [
        EvalItem.model_validate_json(line) for line in QA.read_text().splitlines() if line.strip()
    ]
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker = get_embedder(s), get_reranker(s)
    scopes = scopes_from_db(conn)
    embedder.embed_query("warm up")
    report: dict[str, Any] = {}
    wanted = [c for c in os.environ.get("CONFIGS", ",".join(CONFIGS)).split(",") if c]
    for name in wanted:
        started = time.perf_counter()
        report[name] = run(name, conn, embedder, reranker, items, scopes, s)
        print(f"done {name} ({time.perf_counter() - started:.0f} s)", flush=True)
    print(f"\n{'config':14} " + " | ".join(f"{g:>34}" for g in GROUPS))
    print(f"{'':14} " + " | ".join(f"{'n  any  all  tokens  sources':>34}" for _ in GROUPS))
    for name, d in report.items():
        cells = [
            f"{d[g]['n']:>2} {d[g]['any_gold']:5.2f} {d[g]['all_gold']:4.2f} {d[g]['tokens']:7d} {d[g]['sources']:8.1f}".rjust(
                34
            )
            for g in GROUPS
        ]
        print(f"{name:14} " + " | ".join(cells))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{time.strftime('%Y%m%d-%H%M')}-parents.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
