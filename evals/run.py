"""`make eval`: score the pipeline on the eval set.

Stage 1 (free)         retrieval metrics for dense / keyword / hybrid / hybrid+rerank
Stage 2 (--answers)    the full pipeline on every question: refusals, context hits, latency, cost per answer
Stage 3 (--judge)      faithfulness and correctness by the LLM judge
Headline numbers EXCLUDE follow-up questions (they need chat history: Phase 5); they are reported apart.
Run: uv run python -m evals.run [--answers] [--judge] [--limit N] [--label NAME] [--log] [--yes]
"""

import argparse
import json
import math
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg

from app.config import Settings, get_settings
from app.cost_guard import add_cost_args, estimate_llm_cost, guard_from_args
from app.embeddings import get_embedder
from app.log import setup_logging
from app.rag import ask
from app.retrieval.fusion import Fused
from app.retrieval.rerank import Reranker, get_reranker, rerank
from app.retrieval.retrieve import retrieve
from app.retrieval.search import Candidate, Scope
from evals.judge import Judge
from evals.metrics import evaluate
from evals.schema import EvalItem

MODES = ["dense", "keyword", "hybrid", "hybrid+rerank"]
TENANTS = {"A": "Tenant A (GitLab handbook)", "B": "Tenant B (PostHog handbook)"}


# ---- pure helpers (unit-tested) -----------------------------------------------------------------
def percentile(values: Sequence[float], p: float) -> float:
    """The p-th percentile (0..1) by the nearest-rank method; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(math.ceil(p * len(ordered)) - 1, 0)]


def reorder_with_reranker(
    fused: list[Fused], question: str, reranker: Reranker | None, *, candidates: int, max_words: int
) -> list[Candidate]:
    """The ranking after reranking: the reranked pool first, then the rest in fused order."""
    pool = fused[:candidates]
    scored = rerank(
        fused, question, reranker, candidates=len(pool), top=len(pool), max_words=max_words
    )
    return [r.fused.candidate for r in scored] + [f.candidate for f in fused[len(pool) :]]


def split_items(items: Sequence[EvalItem]) -> tuple[list[EvalItem], list[EvalItem]]:
    """(headline items, follow-ups). Follow-ups cannot work without chat history (Phase 5)."""
    return [i for i in items if i.kind != "follow_up"], [i for i in items if i.kind == "follow_up"]


@dataclass
class AnswerRecord:
    item_id: str
    kind: str
    answerable: bool
    refused: bool
    text: str
    reason: str
    cost_usd: float
    timings_ms: dict[str, float]
    gold_in_context: bool | None
    citations: list[str] = field(default_factory=list)
    faithfulness: float | None = None
    correctness: float | None = None


def answer_metrics(records: Sequence[AnswerRecord]) -> dict[str, Any]:
    head = [r for r in records if r.kind != "follow_up"]
    answerable = [r for r in head if r.answerable]
    unanswerable = [r for r in head if not r.answerable]
    pre_llm = [
        r.timings_ms.get("retrieve_ms", 0)
        + r.timings_ms.get("rerank_ms", 0)
        + r.timings_ms.get("context_ms", 0)
        for r in head
    ]
    total = [sum(r.timings_ms.values()) for r in head]
    faith = [r.faithfulness for r in head if r.faithfulness is not None and not r.refused]
    correct = [r.correctness for r in answerable if r.correctness is not None]
    out: dict[str, Any] = {
        "n": len(head),
        "refusal_accuracy": round(sum(r.refused for r in unanswerable) / len(unanswerable), 3)
        if unanswerable
        else None,
        "wrongly_refused_rate": round(sum(r.refused for r in answerable) / len(answerable), 3)
        if answerable
        else None,
        "gold_in_context_rate": round(
            sum(bool(r.gold_in_context) for r in answerable) / len(answerable), 3
        )
        if answerable
        else None,
        "latency_ms_p50": round(percentile(total, 0.5)),
        "latency_ms_p95": round(percentile(total, 0.95)),
        "pre_llm_ms_p50": round(percentile(pre_llm, 0.5)),
        "pre_llm_ms_p95": round(percentile(pre_llm, 0.95)),
        "cost_per_answer_usd": round(sum(r.cost_usd for r in records) / len(records), 5)
        if records
        else 0.0,
    }
    if faith:
        out["faithfulness"] = round(sum(faith) / len(faith), 3)
        out["faithfulness_n"] = len(faith)
    if correct:
        out["correctness"] = round(
            sum(correct) / len(correct), 3
        )  # refusals of answerable questions count as 0
    return out


# ---- the run -----------------------------------------------------------------------------------
def gold_in_context(item: EvalItem, sources: Sequence[Any]) -> bool:
    return any(
        s.document == g.doc and (g.locator in s.anchors or (g.locator.startswith("page:") and s.pages is not None
                                  and int(g.locator[5:]) in _page_range(s.pages)))
        for s in sources for g in item.gold
    )  # fmt: skip


def _page_range(pages: str) -> range:
    first, _, last = pages.partition("-")
    return range(int(first), int(last or first) + 1)


def scopes_from_db(conn: psycopg.Connection) -> dict[str, Scope]:
    out = {}
    for tenant, name in TENANTS.items():
        org, user = conn.execute(
            "SELECT o.id, u.id FROM orgs o JOIN users u ON u.org_id = o.id WHERE o.name = %s",
            (name,),
        ).fetchone()  # type: ignore[misc]
        out[tenant] = Scope(org, user)
    return out


def _ranker(
    mode: str, conn: psycopg.Connection, embedder: Any, reranker: Reranker | None,
    scopes: dict[str, Scope], s: Settings,
) -> Any:  # fmt: skip
    """A function question -> ranked chunks for one configuration (a factory, so `mode` is bound)."""
    base = "hybrid" if mode == "hybrid+rerank" else mode

    def results(item: EvalItem) -> list[Candidate]:
        fused = retrieve(
            conn, embedder, item.question, scopes[item.tenant], settings=s, mode=base
        ).items
        if mode == "hybrid+rerank":
            return reorder_with_reranker(
                fused,
                item.question,
                reranker,
                candidates=s.rerank_candidates,
                max_words=s.rerank_max_words,
            )
        return [f.candidate for f in fused]

    return results


def retrieval_report(
    conn: psycopg.Connection, embedder: Any, reranker: Reranker | None, items: Sequence[EvalItem],
    scopes: dict[str, Scope], s: Settings, modes: Sequence[str],
) -> dict[str, Any]:  # fmt: skip
    headline, follow = split_items(items)
    out: dict[str, Any] = {}
    for mode in modes:
        results = _ranker(mode, conn, embedder, reranker, scopes, s)
        started = time.perf_counter()
        rep = evaluate(headline, results)
        follow_rep = evaluate(follow, results) if follow else None
        out[mode] = {
            "headline": rep.overall, "n": rep.n,
            "follow_up": follow_rep.overall if follow_rep else None, "n_follow_up": len(follow),
            "subsets": rep.subsets, "seconds": round(time.perf_counter() - started, 1),
        }  # fmt: skip
    return out


def run_answers(
    conn: psycopg.Connection, embedder: Any, reranker: Reranker | None, items: Sequence[EvalItem],
    scopes: dict[str, Scope], s: Settings, guard: Any,
) -> list[AnswerRecord]:  # fmt: skip
    records = []
    for n, item in enumerate(items, start=1):
        r = ask(conn, embedder, reranker, item.question, scopes[item.tenant], settings=s)
        guard.record(r.answer.cost_usd)
        records.append(AnswerRecord(
            item.id, item.kind, item.answerable, r.answer.refused, r.answer.text, r.answer.reason, r.answer.cost_usd,
            r.timings_ms, gold_in_context(item, r.context.sources) if item.answerable else None,
            [c.id for c in r.answer.citations],
        ))  # fmt: skip
        r_context[item.id] = r.context.text
        if n % 10 == 0:
            print(f"  answered {n}/{len(items)}  (spent ${guard.spent_usd:.3f})", flush=True)
    return records


r_context: dict[str, str] = {}  # item id -> the context the model saw (for the judge)


def run_judge(
    items: Sequence[EvalItem],
    records: list[AnswerRecord],
    judge: Judge,
    guard: Any,
    workers: int = 5,
) -> None:
    by_id = {i.id: i for i in items}

    def one(rec: AnswerRecord) -> None:
        item = by_id[rec.item_id]
        if not rec.refused:
            f = judge.faithfulness(rec.text, r_context[rec.item_id])
            rec.faithfulness = f.score
            guard.record(f.cost_usd)
        if item.answerable and item.gold_answer:
            if rec.refused:
                rec.correctness = 0.0  # a wrongly refused answerable question is a miss
            else:
                c = judge.correctness(item.question, rec.text, item.gold_answer)
                rec.correctness = c.score
                guard.record(c.cost_usd)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, records))


def render(report: dict[str, Any]) -> str:
    lines = [
        f"{'config':14} {'n':>3} | {'R@1':>5} {'R@5':>5} {'R@10':>5} {'R@50':>5} | {'MRR':>5} {'nDCG@10':>7} | follow-up R@5"
    ]
    for mode, d in report["retrieval"].items():
        h = d["headline"]
        fu = f"{d['follow_up']['recall@5']:.2f} (n={d['n_follow_up']})" if d["follow_up"] else "-"
        lines.append(
            f"{mode:14} {d['n']:3} | {h['recall@1']:5.2f} {h['recall@5']:5.2f} {h['recall@10']:5.2f} {h['recall@50']:5.2f} | {h['mrr']:5.2f} {h['ndcg@10']:7.2f} | {fu}"
        )
    if "answers" in report:
        lines.append("\nfull pipeline (hybrid + rerank + answer), headline questions only:")
        for key, value in report["answers"].items():
            lines.append(f"  {key:24} {value}")
    return "\n".join(lines)


def append_log_row(path: Path, label: str, report: dict[str, Any], s: Settings) -> None:
    best = report["retrieval"].get("hybrid+rerank") or next(iter(report["retrieval"].values()))
    a = report.get("answers", {})
    header = "\n## Eval runs (appended by `make eval`; LLM-drafted, UNREVIEWED questions)\n| date | label | config | n | Recall@5 | MRR | nDCG@10 | faithfulness | correctness | refusal acc | p50 / p95 ms | $/answer |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    text = path.read_text() if path.exists() else ""
    if "## Eval runs" not in text:
        text += header
    cfg = f"{s.chunk_strategy} {s.chunk_tokens}t, {s.retrieval_mode}, rerank {s.rerank_candidates}/{s.rerank_top} {s.reranker_provider}"
    row = (
        f"| {datetime.now():%Y-%m-%d %H:%M} | {label} | {cfg} | {best['n']} | {best['headline']['recall@5']:.2f} | {best['headline']['mrr']:.2f} | "
        f"{best['headline']['ndcg@10']:.2f} | {a.get('faithfulness', '-')} | {a.get('correctness', '-')} | {a.get('refusal_accuracy', '-')} | "
        f"{a.get('latency_ms_p50', '-')} / {a.get('latency_ms_p95', '-')} | {a.get('cost_per_answer_usd', '-')} |\n"
    )
    path.write_text(text + row)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_cost_args(parser)
    parser.add_argument(
        "--answers",
        action="store_true",
        help="run the full pipeline on every question (costs cents)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="faithfulness and correctness by the judge (needs --answers)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N questions (testing)"
    )
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--log", action="store_true", help="append a row to docs/benchmarks.md")
    args = parser.parse_args()
    setup_logging("WARNING")
    s = get_settings()
    items = [EvalItem.model_validate_json(line) for line in open("evals/qa.jsonl")][: args.limit]
    n = len(items)
    projected = 0.0
    if args.answers:
        projected += estimate_llm_cost(s.llm_default_model, 4200, 220, calls=n) or 0
    if args.judge:
        projected += (estimate_llm_cost(s.llm_judge_model, 1500, 300, calls=n) or 0) + (
            estimate_llm_cost(s.llm_judge_model, 500, 80, calls=n) or 0
        )
    guard = guard_from_args(args, f"eval ({args.label})", projected)
    guard.confirm()

    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker = get_embedder(s), get_reranker(s)
    scopes = scopes_from_db(conn)
    modes = [m for m in args.modes.split(",") if m]
    print(f"stage 1: retrieval metrics on {n} questions, modes {modes}")
    report: dict[str, Any] = {
        "label": args.label,
        "n_items": n,
        "retrieval": retrieval_report(conn, embedder, reranker, items, scopes, s, modes),
    }
    records: list[AnswerRecord] = []
    if args.answers:
        print("stage 2: full pipeline on every question")
        records = run_answers(conn, embedder, reranker, items, scopes, s, guard)
        if args.judge:
            print("stage 3: judge")
            run_judge(items, records, Judge(cache_path=Path("data/cache/judge.jsonl")), guard)
        report["answers"] = answer_metrics(records)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = Path(f"evals/results/{stamp}-{args.label}.json")
    out.write_text(
        json.dumps(
            {
                **report,
                "config": s.model_dump(
                    mode="json",
                    exclude={
                        "anthropic_api_key",
                        "langfuse_secret_key",
                        "rag_owner_password",
                        "rag_app_password",
                        "rag_worker_password",
                    },
                ),
                "records": [asdict(r) for r in records],
            },
            indent=1,
            default=str,
        )
    )
    print("\n" + render(report))
    print(f"\nwrote {out}\n{guard.summary()}")
    if args.log:
        append_log_row(Path("docs/benchmarks.md"), args.label, report, s)
        print("appended a row to docs/benchmarks.md")


if __name__ == "__main__":
    main()
