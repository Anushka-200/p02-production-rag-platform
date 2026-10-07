"""Server-Sent Events helpers shared by the member chat and the widget chat."""

import json
from typing import Any


def sse(event: str, data: dict[str, Any]) -> str:
    """One Server-Sent Event: the browser (or curl) receives it the moment it is written."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def done_payload(event: dict[str, Any]) -> dict[str, Any]:
    """What the browser gets at the end of the stream: the validated answer and what it cost."""
    a = event["answer"]
    agent = event["result"].agent
    return {
        "text": a.text,
        "refused": a.refused,
        "reason": a.reason,
        "replaced": event["replaced"],  # True: the streamed text was corrected after validation
        "citations": [vars(c) for c in a.citations],
        "model": a.model,
        "cost_usd": round(a.cost_usd, 6),
        "llm_calls": a.llm_calls,
        "timings_ms": event["result"].timings_ms,
        "agent": [x.as_dict() for x in agent.steps] if agent else None,
    }
