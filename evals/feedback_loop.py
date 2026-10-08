"""Feedback loop (task 6.14): widget thumbs become CANDIDATE eval questions, reviewed by a human.

    visitor clicks thumbs-down on an answer  ->  export  ->  evals/feedback_candidates.jsonl
    a person reads the candidate, finds where the answer really is, and runs --promote with the gold
    location  ->  the question joins evals/qa.jsonl (still reviewed:false until the person says so)

Why not add them automatically: a thumb says an answer was bad, not what the right answer is, and a
visitor's text can contain personal data or an attack. So the question and comment are redacted
(app/pii.py), the item has NO gold until a person supplies one, and it is kept out of qa.jsonl (the
file `make eval` scores) until promoted.

Run: uv run python -m evals.feedback_loop --org "Acme" --tenant A            # export candidates
     uv run python -m evals.feedback_loop --promote fb-1a2b3c4d --doc gitlab/x.md --locator some-heading
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.pii import redact
from evals.schema import EvalItem

CANDIDATES = Path("evals/feedback_candidates.jsonl")
QA = Path("evals/qa.jsonl")

QUERY = (
    "SELECT f.id, f.rating, f.comment, a.content AS answer, a.refused, a.citations, "
    "(SELECT u.content FROM widget_messages u WHERE u.session_id = a.session_id AND u.role = 'user' "
    " AND u.created_at <= a.created_at ORDER BY u.created_at DESC LIMIT 1) AS question "
    "FROM feedback f JOIN widget_messages a ON a.id = f.message_id "
    "WHERE f.org_id = %s ORDER BY f.created_at"
)


def to_item(row: dict[str, Any], tenant: str) -> dict[str, Any] | None:
    """One feedback row -> a candidate eval item, or None when there is nothing usable."""
    question = redact(row["question"] or "").text.strip()
    if len(question) < 5:
        return None
    thumbs_down = row["rating"] < 0
    # A refusal the visitor liked suggests an unanswerable question; every other case is a question
    # the corpus should be able to answer (a person confirms when reviewing).
    unanswerable = bool(row["refused"]) and not thumbs_down
    cited = sorted({c.get("document", "") for c in row["citations"] or [] if c.get("document")})
    item = {
        "id": f"fb-{str(row['id'])[:8]}",
        "tenant": tenant,
        "kind": "unanswerable" if unanswerable else "single",
        "question": question,
        "answerable": not unanswerable,
        "gold": [],  # a person fills this in
        "corpus": "feedback",
        "tags": ["feedback"],
        "flags": [f"feedback:{'down' if thumbs_down else 'up'}"]
        + (["answer-was-refusal"] if row["refused"] else []),  # noqa: E501
        "source": cited,
        "reviewed": False,
        "review_notes": redact(row["comment"] or "").text,
    }
    EvalItem(**item)  # raises if the shape is wrong
    return item


def export(
    conn: psycopg.Connection, org_id: uuid.UUID, tenant: str, path: Path = CANDIDATES
) -> int:
    """Append candidates not already exported; returns how many were new."""
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(QUERY, (org_id,)).fetchall()
    known = (
        {json.loads(line)["id"] for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    fresh = []
    for row in rows:
        item = to_item(row, tenant)
        if item and item["id"] not in known and item["id"] not in {i["id"] for i in fresh}:
            fresh.append(item)
    if fresh:
        with path.open("a") as f:
            f.writelines(json.dumps(i) + "\n" for i in fresh)
    return len(fresh)


def promote(
    item_id: str, doc: str, locator: str, conn: psycopg.Connection,
    candidates: Path = CANDIDATES, qa: Path = QA,
) -> dict[str, Any]:  # fmt: skip
    """Move one reviewed candidate into the eval set with the gold location a person found."""
    items = [json.loads(line) for line in candidates.read_text().splitlines()]
    item = next((i for i in items if i["id"] == item_id), None)
    if item is None:
        raise SystemExit(f"no candidate {item_id!r}")
    if conn.execute("SELECT 1 FROM documents WHERE source_uri = %s", (doc,)).fetchone() is None:
        raise SystemExit(f"no document {doc!r}: the gold location must exist")
    if any(json.loads(line)["id"] == item_id for line in qa.read_text().splitlines()):
        raise SystemExit(f"{item_id} is already in the eval set")
    item["gold"] = [{"doc": doc, "locator": locator}]
    item["tags"] = [*item["tags"], "from-feedback"]
    EvalItem(**item)
    with qa.open("a") as f:
        f.write(json.dumps(item) + "\n")
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org")
    parser.add_argument("--tenant", choices=["A", "B"], default="A")
    parser.add_argument("--promote")
    parser.add_argument("--doc")
    parser.add_argument("--locator")
    args = parser.parse_args()
    conn = psycopg.connect(get_settings().database_url("owner"), autocommit=True)
    if args.promote:
        if not (args.doc and args.locator):
            raise SystemExit("--promote needs --doc and --locator")
        print("promoted:", promote(args.promote, args.doc, args.locator, conn)["id"])
        return
    org = conn.execute("SELECT id FROM orgs WHERE name = %s", (args.org,)).fetchone()
    if org is None:
        raise SystemExit(f"no organisation named {args.org!r}")
    print(f"{export(conn, org[0], args.tenant)} new candidates written to {CANDIDATES}")


if __name__ == "__main__":
    main()
