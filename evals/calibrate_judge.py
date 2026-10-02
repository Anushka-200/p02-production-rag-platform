"""Synthetic calibration of the judge: does it notice DELIBERATE corruption of a known-good answer?

Not human calibration (the human review was skipped): it measures sensitivity to local errors,
not agreement with a person. Cases are built from drafted eval items:
  faithful      the gold answer, judged against its own source passage   (expect: fully supported, correct)
  number        the gold answer with its first number changed            (expect: unsupported claim, not correct)
  extra_claim   the gold answer plus a claim the passage never makes     (expect: unsupported claim)
  other         another question's gold answer                           (expect: incorrect)
Run: uv run python -m evals.calibrate_judge [--n 15] [--yes]
"""

import argparse
import json
import random
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.cost_guard import add_cost_args, estimate_llm_cost, guard_from_args
from app.log import setup_logging
from evals import corpus as C
from evals.judge import Judge
from evals.schema import EvalItem

EXTRA_CLAIMS = [
    "Employees at every location also receive a 50% annual bonus.",
    "This benefit is available to all contractors in all countries without exception.",
    "The policy was introduced in 1998 and has never been changed.",
    "Managers must report every use of it to the CEO personally.",
]
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def swap_number(text: str) -> str | None:
    """Change the first number in the text (so the claim becomes false); None if there is no number."""
    m = _NUMBER.search(text)
    if not m:
        return None
    original = m.group(0)
    digits = original.replace(",", "")
    changed = str(int(float(digits)) * 2 + 7) if "." not in digits else str(float(digits) + 7)
    return text[: m.start()] + changed + text[m.end() :]


@dataclass
class Case:
    item_id: str
    variant: str  # faithful | number | extra_claim | other
    question: str
    answer: str
    context: str
    gold: str


def build_cases(
    items: Sequence[EvalItem], passages: dict[tuple[str, str], str], n: int = 15, seed: int = 7
) -> list[Case]:
    usable = sorted(
        (
            i
            for i in items
            if i.answerable
            and i.kind == "single"
            and i.gold_answer
            and (i.gold[0].doc, i.gold[0].locator) in passages
        ),
        key=lambda i: i.id,
    )
    rng = random.Random(seed)
    chosen = rng.sample(usable, min(n, len(usable)))
    cases: list[Case] = []
    for k, item in enumerate(chosen):
        context = passages[(item.gold[0].doc, item.gold[0].locator)]
        base = dict(item_id=item.id, question=item.question, context=context, gold=item.gold_answer)
        cases.append(Case(variant="faithful", answer=item.gold_answer, **base))
        swapped = swap_number(item.gold_answer)
        if swapped:  # a "number" case exists only when there really is a number to change
            cases.append(Case(variant="number", answer=swapped, **base))
        cases.append(
            Case(
                variant="extra_claim",
                answer=f"{item.gold_answer} {EXTRA_CLAIMS[(k + 1) % 4]}",
                **base,
            )
        )
        other = chosen[(k + 1) % len(chosen)]
        cases.append(Case(variant="other", answer=other.gold_answer, **base))
    return cases


def summarize(rows: list[dict]) -> dict:  # type: ignore[type-arg]
    """rows: {variant, faith (float|None), label (str|None)} -> detection rates."""

    def mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    by = {
        v: [r for r in rows if r["variant"] == v]
        for v in ("faithful", "number", "extra_claim", "other")
    }
    faith = {
        v: mean([r["faith"] for r in rs if r["faith"] is not None])
        for v, rs in by.items()
        if v != "other"
    }
    corrupted = [r for v in ("number", "extra_claim") for r in by[v] if r["faith"] is not None]
    clean = [r for r in by["faithful"] if r["faith"] is not None]
    return {
        "faithfulness_mean_by_variant": faith,
        "detection_rate": round(sum(r["faith"] < 1.0 for r in corrupted) / len(corrupted), 3)
        if corrupted
        else None,
        "false_alarm_rate": round(sum(r["faith"] < 1.0 for r in clean) / len(clean), 3)
        if clean
        else None,
        "correctness_correct_for_faithful": round(
            sum(r["label"] == "correct" for r in by["faithful"]) / len(by["faithful"]), 3
        )
        if by["faithful"]
        else None,
        "correctness_not_correct_for_number": round(
            sum(r["label"] != "correct" for r in by["number"]) / len(by["number"]), 3
        )
        if by["number"]
        else None,
        "correctness_incorrect_for_other": round(
            sum(r["label"] == "incorrect" for r in by["other"]) / len(by["other"]), 3
        )
        if by["other"]
        else None,
        "n_cases": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_cost_args(parser)
    parser.add_argument("--n", type=int, default=15)
    args = parser.parse_args()
    setup_logging("WARNING")
    s = get_settings()
    items = [EvalItem.model_validate_json(line) for line in open("evals/qa.jsonl")]
    print("loading passages (about 30 s)...")
    passages = {
        (p.doc, p.locator): p.text
        for corpus in (C.load_gitlab(), C.load_posthog(), C.load_pdfs())
        for ps in corpus.values()
        for p in ps
    }
    cases = build_cases(items, passages, args.n)
    projected = (estimate_llm_cost(s.llm_judge_model, 1300, 260, calls=len(cases)) or 0) + (
        estimate_llm_cost(s.llm_judge_model, 450, 80, calls=len(cases)) or 0
    )
    guard = guard_from_args(args, "judge calibration", projected)
    guard.confirm()
    judge = Judge(cache_path=Path("data/cache/judge.jsonl"))

    def run(case: Case) -> dict:  # type: ignore[type-arg]
        f = judge.faithfulness(case.answer, case.context)
        c = judge.correctness(case.question, case.answer, case.gold)
        guard.record(f.cost_usd + c.cost_usd)
        return {
            "item": case.item_id,
            "variant": case.variant,
            "faith": f.score,
            "label": c.detail["label"],
            "answer": case.answer[:200],
            "unsupported_claims": [
                x["claim"] for x in f.detail.get("claims", []) if not x["supported"]
            ],
        }

    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(run, cases))
    report = summarize(rows)
    Path("evals/results/judge_calibration.json").write_text(
        json.dumps({"summary": report, "rows": rows}, indent=1)
    )
    print(json.dumps(report, indent=1))
    for r in rows[:8]:
        print(f"  {r['variant']:12} faith={r['faith']} label={r['label']:9} {r['answer'][:90]!r}")
    print(guard.summary())


if __name__ == "__main__":
    main()
