"""Thumbs -> candidate eval questions: redacted, no gold, never auto-merged, promotable only by a person."""

import json
from pathlib import Path

import pytest

from evals.feedback_loop import export, promote, to_item
from tests.test_widget import Rig


def row(**kw):  # type: ignore[no-untyped-def]
    base = {"id": "1a2b3c4d-0000-0000-0000-000000000000", "rating": -1, "comment": "wrong", "answer": "a",
            "refused": False, "citations": [{"document": "gitlab/x.md"}], "question": "When do you open?"}  # fmt: skip
    return {**base, **kw}


def test_a_thumbs_down_becomes_an_answerable_candidate_without_gold() -> None:
    item = to_item(row(), "A")
    assert item["id"] == "fb-1a2b3c4d" and item["answerable"] and item["gold"] == []  # type: ignore[index]
    assert item["reviewed"] is False and item["flags"] == ["feedback:down"]  # type: ignore[index]
    assert item["source"] == ["gitlab/x.md"]  # type: ignore[index]


def test_a_liked_refusal_is_a_candidate_unanswerable_question() -> None:
    item = to_item(row(rating=1, refused=True, citations=[]), "A")
    assert item["kind"] == "unanswerable" and item["answerable"] is False  # type: ignore[index]
    disliked = to_item(row(rating=-1, refused=True, citations=[]), "A")
    assert disliked["answerable"] is True and "answer-was-refusal" in disliked["flags"]  # type: ignore[index]


def test_personal_data_in_the_question_and_the_comment_is_redacted() -> None:
    item = to_item(
        row(question="My email is jo@example.com, when do you open?", comment="call 415-555-0132"),
        "A",
    )
    text = json.dumps(item)
    assert "jo@example.com" not in text and "555-0132" not in text


def test_empty_or_tiny_questions_are_skipped() -> None:
    assert to_item(row(question=""), "A") is None and to_item(row(question=None), "A") is None


def test_export_reads_real_feedback_and_never_exports_twice(db_schema, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    rig = Rig(db_schema)
    first = rig.done(rig.chat())
    r = rig.post("feedback", {"session_id": first["session_id"], "message_id": first["message_id"],
                              "rating": -1, "comment": "I wanted the weekend hours"})  # fmt: skip
    assert r.status_code in (200, 201, 204), r.text
    path = tmp_path / "c.jsonl"
    assert export(db_schema, rig.ann.org_id, "A", path) == 1
    assert export(db_schema, rig.ann.org_id, "A", path) == 0  # idempotent
    item = json.loads(path.read_text())
    assert (
        item["question"] == "When do you open?"
        and item["review_notes"] == "I wanted the weekend hours"
    )


def test_promote_needs_a_real_document_and_adds_the_gold_but_stays_unreviewed(
    db_schema, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    rig = Rig(db_schema)
    first = rig.done(rig.chat())
    rig.post(
        "feedback",
        {"session_id": first["session_id"], "message_id": first["message_id"], "rating": -1},
    )
    cand, qa = tmp_path / "c.jsonl", tmp_path / "qa.jsonl"
    qa.write_text("")
    export(db_schema, rig.ann.org_id, "A", cand)
    item_id = json.loads(cand.read_text())["id"]
    with pytest.raises(SystemExit, match="must exist"):
        promote(item_id, "no/such.md", "x", db_schema, cand, qa)
    uri = db_schema.execute("SELECT source_uri FROM documents LIMIT 1").fetchone()[0]
    done = promote(item_id, uri, "opening-hours", db_schema, cand, qa)
    saved = json.loads(qa.read_text())
    assert (
        saved["gold"] == [{"doc": uri, "locator": "opening-hours"}] and saved["reviewed"] is False
    )
    assert "from-feedback" in done["tags"]
    with pytest.raises(SystemExit, match="already"):
        promote(item_id, uri, "opening-hours", db_schema, cand, qa)
