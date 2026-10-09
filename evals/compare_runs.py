"""Side-by-side summary of eval result files, restricted to one split (dev by default).

Answer-level numbers from the saved records: correctness (a refusal of an answerable question counts 0), faithfulness, wrongly refused,
refusal accuracy on the unanswerable ones, gold place in the context, cost and latency, and correctness for the hard kinds.
Run: uv run python -m evals.compare_runs LABEL_OR_FILE ... [--split dev|test|all]
"""

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any

from evals.splits import split_of


def load(arg: str) -> dict[str, Any]:
    path = (
        Path(arg)
        if arg.endswith(".json")
        else Path(sorted(glob.glob(f"evals/results/*-{arg}.json"))[-1])
    )
    return json.load(open(path))  # type: ignore[no-any-return]


def summarise(res: dict[str, Any], split: str) -> dict[str, float]:
    recs = [r for r in res["records"] if split == "all" or split_of(r["item_id"]) == split]
    ans = [r for r in recs if r["answerable"]]
    un = [r for r in recs if not r["answerable"]]
    corr = [0.0 if r["refused"] else (r["correctness"] or 0.0) for r in ans]
    faith = [r["faithfulness"] for r in recs if r["faithfulness"] is not None]
    out = {
        "n answerable": len(ans), "n unanswerable": len(un),
        "correctness": statistics.mean(corr) if corr else 0.0,
        "faithfulness": statistics.mean(faith) if faith else 0.0,
        "wrongly refused": sum(r["refused"] for r in ans) / max(len(ans), 1),
        "refusal accuracy": sum(r["refused"] for r in un) / max(len(un), 1),
        "gold in context": sum(1 for r in ans if r["gold_in_context"]) / max(len(ans), 1),
        "cost / answer $": statistics.mean(r["cost_usd"] for r in recs),
        "latency p50 ms": statistics.median(r["timings_ms"].get("total_ms", sum(r["timings_ms"].values())) for r in recs),
    }  # fmt: skip
    for kind in ("single", "paraphrase", "multi_hop", "broad", "relationship", "follow_up"):
        ks = [
            (0.0 if r["refused"] else (r["correctness"] or 0.0)) for r in ans if r["kind"] == kind
        ]
        if ks:
            out[f"correct: {kind} (n={len(ks)})"] = statistics.mean(ks)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = parser.parse_args()
    cols = {r: summarise(load(r), args.split) for r in args.runs}
    names = list(next(iter(cols.values())))
    print(f"split: {args.split}")
    print("| metric | " + " | ".join(args.runs) + " |\n|---|" + "---|" * len(args.runs))
    for name in sorted(
        {n for c in cols.values() for n in c}, key=lambda n: names.index(n) if n in names else 99
    ):
        print(
            f"| {name} | "
            + " | ".join(f"{c[name]:.3f}" if name in c else "-" for c in cols.values())
            + " |"
        )


if __name__ == "__main__":
    main()
