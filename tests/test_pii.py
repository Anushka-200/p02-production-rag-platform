"""PII redaction: pattern-based entities are replaced, ordinary text and our own ids are not."""

import json
import logging

import pytest

from app.log import JsonFormatter
from app.pii import redact, redact_any
from app.tracing import _mask


@pytest.mark.parametrize(
    ("text", "entity"),
    [
        ("mail ann.lee@acme.com now", "EMAIL_ADDRESS"),
        ("card 4111 1111 1111 1111", "CREDIT_CARD"),
        ("call (415) 555-0132", "PHONE_NUMBER"),
        ("IBAN DE89 3704 0044 0532 0130 00", "IBAN_CODE"),
        ("host 203.0.113.42", "IP_ADDRESS"),
    ],
)
def test_personal_data_is_replaced_by_its_label(text: str, entity: str) -> None:
    result = redact(text)
    assert entity in result.found and f"<{entity}>" in result.text


@pytest.mark.parametrize(
    "text",
    [
        "I prefer short answers.",
        "Section 4.2.1 on page 12, version 3.12.1",
        "3782fdc2-5512-4ee3-9a55-0eba42121297",  # our own job/org ids
        "gitlab/people-group/2024-05-01-notes.md",
        "",
    ],
)
def test_ordinary_text_and_our_own_ids_are_left_alone(text: str) -> None:
    assert redact(text).text == text and redact(text).found == ()


def test_names_are_not_redacted_because_the_name_detector_is_unreliable() -> None:
    # measured in B-016: the small model tags "Email" as a person and "Priya" as an organisation
    assert redact("Priya works in the Berlin office").text == "Priya works in the Berlin office"


def test_redact_any_walks_nested_data_and_keeps_other_types() -> None:
    data = {"input": [{"role": "user", "content": "mail bob@x.com"}], "n": 3, "ok": True}
    out = redact_any(data)
    assert out["input"][0]["content"] == "mail <EMAIL_ADDRESS>"
    assert out["n"] == 3 and out["ok"] is True


def test_the_langfuse_mask_scrubs_prompts_before_upload() -> None:
    assert _mask(data={"prompt": "my card 4111 1111 1111 1111"}) == {
        "prompt": "my card <CREDIT_CARD>"
    }


def test_log_lines_are_redacted_but_ids_and_timestamps_are_not() -> None:
    record = logging.LogRecord("t", logging.INFO, "f", 1, "user wrote to ann@x.com", (), None)
    record.job_id = "3782fdc2-5512-4ee3-9a55-0eba42121297"
    record.question = "my phone is (415) 555-0132"
    line = json.loads(JsonFormatter().format(record))
    assert line["message"] == "user wrote to <EMAIL_ADDRESS>"
    assert line["question"] == "my phone is <PHONE_NUMBER>"
    assert line["job_id"] == "3782fdc2-5512-4ee3-9a55-0eba42121297"
    assert line["ts"].startswith("20")
    unredacted = json.loads(JsonFormatter(redact_pii=False).format(record))
    assert "ann@x.com" in unredacted["message"]
