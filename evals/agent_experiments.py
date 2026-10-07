"""Experiment (Phase 5 block 5): does letting an agent CHOOSE the search beat one fixed search?

For every answerable question, both paths run up to the context the answer model would be given:

    plain   hybrid search -> rerank -> context          (what the system did before agents)
    agent   plan -> chosen tools -> check -> context     (real planner and checker calls)

Scored by whether the gold place(s) are in that context (any / all), per question group, plus what
the agent costs: its own model calls, search calls, extra seconds, which tools it picked, and why
it stopped. The planner/checker calls are cached on disk, so a repeat run is free and identical.
Informal: the eval set is LLM-drafted; the relationship questions are derived from the graph.

Run: uv run python -m evals.agent_experiments [--yes]
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psycopg

from app.agent.loop import WEAK_SCORE, run_agent
from app.agent.tools import ToolBox
from app.config import get_settings
from app.cost_guard import add_cost_args, estimate_llm_cost, guard_from_args
from app.embeddings import get_embedder, get_token_counter
from app.llm import complete
from app.log import setup_logging
from app.retrieval.context import build_context
from app.retrieval.expand import retrieve_expanded
from app.retrieval.rerank import get_reranker, rerank
from app.retrieval.transform import TransformCache
from evals.context_experiments import covered
from evals.run import scopes_from_db
from evals.schema import EvalItem

QA = Path(__file__).parent / "qa.jsonl"
RESULTS = Path(__file__).parent / "results"
GROUPS = ("single", "paraphrase", "multi_hop", "broad", "relationship")


def cached(cache: TransformCache) -> Any:
    """A model function that remembers its answers on disk (temperature is 0, so they are stable)."""

    def complete_fn(messages: list[dict[str, Any]], **kw: Any) -> Any:
        key = TransformCache.key("agent", kw.get("model"), json.dumps(messages, sort_keys=True))
        if (hit := cache.get(key)) is not None:
            return SimpleNamespace(text=hit["text"], cost_usd=0.0)
        result = complete(messages, **kw)
        cache.put(key, {"text": result.text})
        return result

    return complete_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    add_cost_args(parser)
    args = parser.parse_args()
    setup_logging("WARNING")
    s = get_settings().model_copy(update={"query_rewrite": False, "agent_enabled": False})
    items = [
        EvalItem.model_validate_json(line) for line in QA.read_text().splitlines() if line.strip()
    ]
    todo = [i for i in items if i.kind in GROUPS and i.answerable]
    guard = guard_from_args(
        args,
        "agent experiment",
        estimate_llm_cost(s.llm_default_model, 700, 120, calls=2 * len(todo)),
    )
    guard.confirm()
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker = get_embedder(s), get_reranker(s)
    scopes = scopes_from_db(conn)
    count = get_token_counter(s)
    cache = TransformCache("data/cache/agent.jsonl")
    model_fn = cached(cache)
    embedder.embed_query("warm up")
    rows: dict[str, list[dict[str, Any]]] = {g: [] for g in GROUPS}
    tools: Counter[str] = Counter()
    stops: Counter[str] = Counter()
    for item in todo:
        scope = scopes[item.tenant]
        t0 = time.perf_counter()
        plain = rerank(
            retrieve_expanded(conn, embedder, item.question, scope, settings=s).items, item.question,
            reranker, candidates=s.rerank_candidates, top=s.rerank_top, max_words=s.rerank_max_words,
        )  # fmt: skip
        plain_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        toolbox = ToolBox(conn, embedder, reranker, scope, s, model_fn)
        run = run_agent(
            item.question, toolbox, complete_fn=model_fn, max_cost_usd=s.agent_max_cost_usd
        )
        agent_ms = (time.perf_counter() - t0) * 1000
        guard.record(run.cost_usd)
        for step in run.steps:
            if step.kind == "search" and step.tool:
                tools[step.tool] += 1
        stops[run.stopped] += 1
        model_calls = sum(1 for x in run.steps if x.kind in {"plan", "replan", "check"})
        # "gated": escalate to the agent only when the plain search looks weak (its best rerank score)
        best = plain[0].rerank_score if plain and plain[0].rerank_score is not None else None
        weak = not plain or (best is not None and best < WEAK_SCORE)
        row: dict[str, Any] = {
            "local_ms": agent_ms
            - plain_ms,  # search work only; model-call latency is not in a cached run
            "model_calls": model_calls,
            "calls": run.tool_calls,
            "cost": run.cost_usd,
            "escalated": weak,
        }
        for name, ranked in (
            ("plain", plain),
            ("agent", run.evidence or plain),
            ("gated", (run.evidence or plain) if weak else plain),
        ):
            sources = build_context(ranked, s.context_budget_tokens, count).sources
            hits = [covered(g, sources) for g in item.gold]
            row[name] = (any(hits), all(hits))
        rows[item.kind].append(row)
    header = (
        f"\n{'group':12} {'n':>3} | {'plain':>11} | {'always agent':>12} | {'gated':>11} | "
        f"escalated | model calls/q (always -> gated) | est. extra s/q (always -> gated)"
    )
    print(header + "\n" + " " * 17 + "(any / all gold in context)")
    report: dict[str, Any] = {}
    for g, rs in rows.items():
        if not rs:
            continue
        n = len(rs)
        cell = {
            k: (sum(r[k][0] for r in rs) / n, sum(r[k][1] for r in rs) / n)
            for k in ("plain", "agent", "gated")
        }
        share = sum(r["escalated"] for r in rs) / n
        calls_always = sum(r["model_calls"] for r in rs) / n
        calls_gated = sum(r["model_calls"] for r in rs if r["escalated"]) / n
        # assumption: one small-model call is about 1 s (Haiku, 100-200 output tokens)
        sec_always = sum(r["local_ms"] / 1000 + r["model_calls"] for r in rs) / n
        sec_gated = sum(r["local_ms"] / 1000 + r["model_calls"] for r in rs if r["escalated"]) / n
        report[g] = {
            "n": n, **{f"{k}_{m}": v[i] for k, v in cell.items() for i, m in enumerate(("any", "all"))},
            "escalated": share, "model_calls_always": calls_always, "model_calls_gated": calls_gated,
            "est_extra_s_always": sec_always, "est_extra_s_gated": sec_gated,
        }  # fmt: skip
        print(
            f"{g:12} {n:>3} | {cell['plain'][0]:4.2f} /{cell['plain'][1]:5.2f} | {cell['agent'][0]:5.2f} /{cell['agent'][1]:5.2f} | "
            f"{cell['gated'][0]:4.2f} /{cell['gated'][1]:5.2f} | {share:8.0%} | {calls_always:11.1f} -> {calls_gated:4.1f}          | "
            f"{sec_always:8.1f} -> {sec_gated:4.1f}"
        )
    print("\ntools chosen:", dict(tools), "| stopped by:", dict(stops))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{time.strftime('%Y%m%d-%H%M')}-agent.json"
    path.write_text(json.dumps({"groups": report, "tools": tools, "stops": stops}, indent=2))
    print(f"wrote {path}; agent model spend ${guard.spent_usd:.4f}")


if __name__ == "__main__":
    main()
