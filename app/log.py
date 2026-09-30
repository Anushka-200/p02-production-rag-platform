"""Structured JSON logging: one JSON object per line, so tools can filter on fields.

Named log.py (not logging.py) so it never shadows the standard library module.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# Set once per request or job (e.g. org_id, request_id); every log line in that scope includes it.
log_context: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)

# Attributes every LogRecord already has; anything else came from `extra=` and is a custom field.
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **(log_context.get() or {}),
        }
        entry.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        # default=str keeps logging from crashing on values JSON cannot encode (UUIDs, datetimes).
        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]  # replace, so calling this twice never duplicates lines
    root.setLevel(level.upper())
