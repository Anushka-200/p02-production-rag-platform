import json
import logging

import pytest

from app.log import log_context, setup_logging


def last_line(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_line_is_valid_json_with_extra_fields(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    logging.getLogger("t").info("ingested", extra={"doc_id": "d1", "chunks": 12})
    line = last_line(capsys)
    assert line["message"] == "ingested"
    assert line["level"] == "INFO"
    assert line["doc_id"] == "d1" and line["chunks"] == 12
    assert "ts" in line


def test_context_fields_are_added(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    token = log_context.set({"org_id": "acme"})
    try:
        logging.getLogger("t").info("hello")
    finally:
        log_context.reset(token)
    assert last_line(capsys)["org_id"] == "acme"


def test_exception_is_included(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logging.getLogger("t").exception("failed")
    assert "RuntimeError: boom" in str(last_line(capsys)["exception"])


def test_calling_setup_twice_does_not_duplicate(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    setup_logging("INFO")
    logging.getLogger("t").info("once")
    assert len(capsys.readouterr().out.strip().splitlines()) == 1
