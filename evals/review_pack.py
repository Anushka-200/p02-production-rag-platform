"""Build review cards so every eval question can be checked against its source (task: eval review).

For each question: the question and gold answer, the gold passage text with an automatic check that every supporting
quote really appears in it, and the best passages that the production search finds BESIDES the gold (candidate alternative
answers: if one of them answers the question equally well, the question has two valid places and the gold must say so).
The cards are read by the reviewer; decisions go in evals/review_decisions.jsonl and are applied by evals.apply_review.
Free (local models). Run: uv run python -m evals.review_pack [--per-batch 10]
"""

import argparse
import json
from pathlib import Path

import psycopg

from app.config import get_settings
from app.embeddings import get_embedder
from app.retrieval.rerank import get_reranker
from evals import corpus as C
from evals.metrics import covers
from evals.run import _ranker, scopes_from_db
from evals.schema import EvalItem

OUT = Path("evals/results/review")
GOLD_CHARS, OTHER_CHARS = 1100, 520


def passage_index() -> dict[tuple[str, str], C.Passage]:
    index: dict[tuple[str, str], C.Passage] = {}
    for loader in (C.load_gitlab, C.load_posthog, C.load_pdfs):
        for passages in loader().values():
            for p in passages:
                index[(p.doc, p.locator)] = p
    return index


def card(
    item: EvalItem, index: dict[tuple[str, str], C.Passage], others: list[dict[str, str]]
) -> str:
    lines = [f"### {item.id}  [{item.kind}, tenant {item.tenant}]", f"**Q:** {item.question}"]
    if item.history:
        lines.append(
            "history: " + " / ".join(f"{m['role']}: {m['content'][:160]}" for m in item.history)
        )
    lines.append(f"**Gold answer:** {item.gold_answer}")
    gold_texts = [index[(g.doc, g.locator)].text for g in item.gold if (g.doc, g.locator) in index]
    unsupported = [
        q for q in item.support if not any(C.contains_verbatim(t, q) for t in gold_texts)
    ]
    for n, g in enumerate(item.gold, 1):
        p = index.get((g.doc, g.locator))
        if p is None:
            lines.append(f"- GOLD {n}: {g.doc} #{g.locator}  **MISSING FROM CORPUS**")
            continue
        flag = (
            "(all support quotes are in the gold passages)"
            if not unsupported and item.support
            else f"support quotes NOT in ANY gold passage: {len(unsupported)} of {len(item.support)}"
        )
        text = p.text if len(p.text) <= GOLD_CHARS else p.text[:GOLD_CHARS] + " [...]"
        lines.append(
            f"- GOLD {n}: {g.doc} #{g.locator} | path: {p.path} | {flag}\n  > {' '.join(text.split())}"
        )
    for o in others:
        lines.append(f"- OTHER ({o['doc']} | {o['section']}): {o['text']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-batch", type=int, default=10)
    parser.add_argument("--file", default="evals/qa.jsonl")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    s = get_settings()
    items = [EvalItem.model_validate_json(line) for line in open(args.file)]
    index = passage_index()
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    embedder, reranker = get_embedder(s), get_reranker(s)
    rank = _ranker("hybrid+rerank", conn, embedder, reranker, scopes_from_db(conn), s)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cards, auto = [], {}
    for n, item in enumerate(items, 1):
        others: list[dict[str, str]] = []
        if item.kind != "follow_up":
            seen = set()
            for c in rank(item):
                if any(covers(c, g) for g in item.gold):
                    continue
                key = (c.source_uri, c.section_path)
                if key in seen:
                    continue
                seen.add(key)
                others.append(
                    {
                        "doc": c.source_uri,
                        "section": c.section_path[-70:],
                        "text": " ".join(c.text.split())[:OTHER_CHARS],
                    }
                )
                if len(others) == (3 if item.answerable else 4):
                    break
        text = card(item, index, others)
        cards.append(text)
        auto[item.id] = {"missing_gold": [g.locator for g in item.gold if (g.doc, g.locator) not in index],
                         "bad_support": "NOT found" in text}  # fmt: skip
        if n % 20 == 0:
            print(f"  {n}/{len(items)}", flush=True)
    for b in range(0, len(cards), args.per_batch):
        (out / f"batch-{b // args.per_batch + 1:02d}.md").write_text(
            "\n\n".join(cards[b : b + args.per_batch])
        )
    (out / "auto-checks.json").write_text(json.dumps(auto, indent=1))
    print(f"{len(cards)} cards in {-(-len(cards) // args.per_batch)} batches -> {out}; "
          f"missing gold: {sum(bool(a['missing_gold']) for a in auto.values())}, quotes not found: {sum(a['bad_support'] for a in auto.values())}")  # fmt: skip


if __name__ == "__main__":
    main()
