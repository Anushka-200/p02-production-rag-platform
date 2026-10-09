"""Apply the review decisions (evals/review_decisions.jsonl) to the eval set and verify every gold place exists.

Decisions: keep (optionally add accepted alternatives), fix (new question / gold answer / gold places), drop (removed
from qa.jsonl, kept in evals/qa_dropped.jsonl with the reason). Every reviewed item gets reviewed=true and
reviewer="claude": a MODEL review against the source text, NOT a human one, and the docs say so.
Verification: each gold place and each alternative must match a real chunk of that document (a heading anchor, or a page
inside a chunk's page range for PDFs); otherwise the script stops and lists the bad places.

Run: uv run python -m evals.apply_review [--dry-run]
"""

import argparse
import json
from pathlib import Path

import psycopg

from app.config import get_settings
from evals.schema import EvalItem

QA = Path("evals/qa.jsonl")
DECISIONS = Path("evals/review_decisions.jsonl")
DROPPED = Path("evals/qa_dropped.jsonl")
REVIEW_DATE = "2026-10-09"


def place_exists(conn: psycopg.Connection, doc: str, locator: str) -> bool:
    if locator.startswith("page:"):
        page = int(locator[5:])
        return conn.execute(
            "SELECT 1 FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.source_uri = %s AND c.level = 0 "
            "AND c.page_start <= %s AND coalesce(c.page_end, c.page_start) >= %s LIMIT 1", (doc, page, page),
        ).fetchone() is not None  # fmt: skip
    return conn.execute(
        "SELECT 1 FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.source_uri = %s AND c.level = 0 "
        "AND %s = ANY(c.anchors) LIMIT 1", (doc, locator),
    ).fetchone() is not None  # fmt: skip


def apply(items: list[dict], decisions: dict[str, dict]) -> tuple[list[dict], list[dict]]:  # type: ignore[type-arg]
    kept, dropped = [], []
    for item in items:
        d = decisions[item["id"]]
        if d["v"] == "drop":
            dropped.append(
                {
                    **item,
                    "dropped_reason": d["note"],
                    "reviewer": "claude",
                    "review_date": REVIEW_DATE,
                }
            )
            continue
        new = dict(item)
        notes = []
        if d.get("question"):
            notes.append(f"question changed; original: {item['question']!r}")
            new["question"] = d["question"]
        if d.get("answer"):
            notes.append(f"gold answer changed; original: {item['gold_answer']!r}")
            new["gold_answer"] = d["answer"]
        if d.get("gold"):
            notes.append(
                f"gold places changed; original: {[g['doc'] + '#' + g['locator'] for g in item['gold']]}"
            )
            new["gold"] = [
                {"doc": g["doc"], "locator": g["locator"], "alt": g.get("alt", [])}
                for g in d["gold"]
            ]
        for index, places in d.get("alt", {}).items():
            gold = dict(new["gold"][int(index)])
            gold["alt"] = [*gold.get("alt", []), *places]
            new["gold"] = [*new["gold"][: int(index)], gold, *new["gold"][int(index) + 1 :]]
        if d.get("note"):
            notes.append(d["note"])
        new.update(
            reviewed=True,
            reviewer="claude",
            review_notes=" | ".join(notes) + f" [reviewed by Claude {REVIEW_DATE}]"
            if notes
            else f"reviewed by Claude {REVIEW_DATE}",
        )  # noqa: E501
        kept.append(new)
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--items", default=str(QA), help="the file of questions to review")
    parser.add_argument("--decisions", default=str(DECISIONS))
    parser.add_argument(
        "--append",
        action="store_true",
        help="add the kept items to qa.jsonl instead of replacing it",
    )
    args = parser.parse_args()
    items = [json.loads(line) for line in Path(args.items).read_text().splitlines()]
    decisions = {d["id"]: d for d in map(json.loads, Path(args.decisions).read_text().splitlines())}
    assert {i["id"] for i in items} == set(decisions), "every item needs exactly one decision"
    kept, dropped = apply(items, decisions)
    conn = psycopg.connect(get_settings().database_url("owner"), autocommit=True)
    bad = []
    for item in kept:
        EvalItem(**item)  # schema check
        for g in item["gold"]:
            for p in [g, *g.get("alt", [])]:
                if not place_exists(conn, p["doc"], p["locator"]):
                    bad.append((item["id"], p["doc"], p["locator"]))
    print(
        f"{len(kept)} kept/fixed, {len(dropped)} dropped; gold places that do not exist: {len(bad)}"
    )
    for b in bad:
        print("  MISSING", b)
    if bad or args.dry_run:
        raise SystemExit(1 if bad else 0)
    if args.append:
        have = {json.loads(line)["id"] for line in QA.read_text().splitlines()}
        assert not have & {i["id"] for i in kept}, "ids already in the eval set"
        with QA.open("a") as f:
            f.writelines(json.dumps(i) + "\n" for i in kept)
        with DROPPED.open("a") as f:
            f.writelines(json.dumps(i) + "\n" for i in dropped)
    else:
        QA.write_text("".join(json.dumps(i) + "\n" for i in kept))
        DROPPED.write_text("".join(json.dumps(i) + "\n" for i in dropped))
    print(f"wrote {QA} and {DROPPED}")


if __name__ == "__main__":
    main()
