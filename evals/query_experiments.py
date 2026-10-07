"""Experiment (Phase 5 block 2): do query rewriting, multi-query and HyDE find more of the right places?

Each configuration is the production pipeline (hybrid search, then reranking of the best 10) with
the transforms switched on or off. For every question the CHAT HISTORY is passed along (only
follow-ups have any). Results are reported per question group, because each technique is meant
to help ONE group and may hurt another:

    follow_up    10   rewriting should help (these questions only make sense with the history)
    paraphrase   24   multi-query and HyDE should help (different words than the document)
    single       61   should not get worse
    multi_hop     8   may or may not move

Transform outputs are cached on disk (data/cache/transforms.jsonl), so repeating a run costs nothing
and gives identical results. Informal: the eval set is LLM-drafted and not human-reviewed.

Run: uv run python -m evals.query_experiments [--yes] [--configs baseline,rewrite,...]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import psycopg

from app.config import Settings, get_settings
from app.cost_guard import add_cost_args, estimate_llm_cost, guard_from_args
from app.embeddings import get_embedder
from app.log import setup_logging
from app.retrieval.expand import retrieve_expanded
from app.retrieval.rerank import get_reranker
from app.retrieval.search import Candidate
from app.retrieval.transform import TransformCache
from evals.metrics import evaluate
from evals.run import reorder_with_reranker, scopes_from_db
from evals.schema import EvalItem

QA = Path(__file__).parent / "qa.jsonl"
RESULTS = Path(__file__).parent / "results"
CACHE = Path("data/cache/transforms.jsonl")
OFF = {"query_rewrite": False, "multi_query": 0, "hyde": False}  # every config states all three
CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {**OFF},
    "rewrite": {**OFF, "query_rewrite": True},
    "multi-3": {**OFF, "multi_query": 3},
    "hyde": {**OFF, "hyde": True},
    "all": {"query_rewrite": True, "multi_query": 3, "hyde": True},
}
GROUPS = ("follow_up", "paraphrase", "single", "multi_hop")


def run_config(
    name: str, conn: psycopg.Connection, embedder: Any, reranker: Any, items: list[EvalItem],
    scopes: dict[str, Any], base: Settings, cache: TransformCache,
) -> dict[str, Any]:  # fmt: skip
    s = base.model_copy(update=CONFIGS[name])
    spent: list[float] = []
    ms: list[float] = []

    def results(item: EvalItem) -> list[Candidate]:
        started = time.perf_counter()
        r = retrieve_expanded(
            conn, embedder, item.question, scopes[item.tenant], history=item.history, settings=s,
            cache=cache,
        )  # fmt: skip
        spent.append(r.transform_cost_usd)
        ranked = reorder_with_reranker(
            r.items, r.question or item.question, reranker,
            candidates=s.rerank_candidates, max_words=s.rerank_max_words,
        )  # fmt: skip
        ms.append((time.perf_counter() - started) * 1000)
        return ranked

    out: dict[str, Any] = {}
    for group in GROUPS:
        members = [i for i in items if i.kind == group and i.answerable]
        if members:
            rep = evaluate(members, results)
            out[group] = {
                "n": rep.n,
                **{k: rep.overall[k] for k in ("recall@1", "recall@5", "recall@10", "mrr")},
            }
    out["_transform_usd"] = round(sum(spent), 4)  # 0 when answered from the cache
    out["_p50_ms"] = round(sorted(ms)[len(ms) // 2], 0)
    return out


def render(report: dict[str, dict[str, Any]]) -> str:
    lines = [f"{'config':10} " + " | ".join(f"{g:>22}" for g in GROUPS) + " |  p50 ms"]
    lines.append(f"{'':10} " + " | ".join(f"{'n  R@1  R@5  MRR':>22}" for _ in GROUPS) + " |")
    for name, d in report.items():
        cells = []
        for g in GROUPS:
            c = d.get(g)
            cells.append(
                f"{c['n']:>2} {c['recall@1']:5.2f} {c['recall@5']:4.2f} {c['mrr']:5.2f}".rjust(22)
                if c
                else "-".rjust(22)
            )
        lines.append(f"{name:10} " + " | ".join(cells) + f" | {d['_p50_ms']:7.0f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--label", default="query-transforms")
    add_cost_args(parser)
    args = parser.parse_args()
    setup_logging("WARNING")
    s = get_settings()
    names = [c for c in args.configs.split(",") if c]
    items = [
        EvalItem.model_validate_json(line) for line in QA.read_text().splitlines() if line.strip()
    ]
    n_calls = len(items) * 2 * len(names)  # an upper bound; cached calls are free
    guard = guard_from_args(
        args, "query transforms", estimate_llm_cost(s.llm_default_model, 300, 80, calls=n_calls)
    )
    guard.confirm()
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker, cache = get_embedder(s), get_reranker(s), TransformCache(CACHE)
    scopes = scopes_from_db(conn)
    embedder.embed_query("warm up")
    report: dict[str, dict[str, Any]] = {}
    for name in names:
        report[name] = run_config(name, conn, embedder, reranker, items, scopes, s, cache)
        guard.record(report[name]["_transform_usd"])
        print(f"done {name} ({report[name]['_p50_ms']:.0f} ms p50)", flush=True)
    print("\n" + render(report))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{time.strftime('%Y%m%d-%H%M')}-{args.label}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}; transform spend ${guard.spent_usd:.4f}")


if __name__ == "__main__":
    main()
