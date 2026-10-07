"""Query transforms: three ways to turn the user's words into better search input.

    rewrite_query  "How many topics are there for these sessions?" + chat history
                   -> "How many topics are available in GitLab's Team Workshops?"
    expand_queries "When can I request a refund?"
                   -> ["refund policy eligibility", "how long after purchase can I get money back"]
    hyde_passage   "Why is the notice period 30 days?"
                   -> "The notice period is 30 days so the team can hand over work..." (made up)

Each is one small model call. The outputs are ONLY used as search input: they are never shown to the
user and never given to the answer model as instructions, so a manipulated rewrite can at worst
make a search worse. Every output is validated and, if unusable, the ORIGINAL question is used
(graceful degradation: a failing transform must never break a question that would have worked).

An optional disk cache (key = kind + model + input) makes repeated experiments free and exactly
repeatable.
"""

import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.llm import LLMError, complete

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 300
MAX_HYDE_CHARS = 700
HISTORY_TURNS = 3  # the last three messages pairs are enough to resolve "it", "that", "these"
HISTORY_CHARS = 500  # each past message is cut to this length before it is sent


@dataclass
class Transformed:
    text: str  # what to search with
    cost_usd: float = 0.0
    calls: int = 0
    used: bool = False  # False: the original question was kept (nothing to do, or the model failed)


class TransformCache:
    """Append-only JSONL file; the whole file is loaded once. Safe to share between threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.data[row["key"]] = row["value"]

    @staticmethod
    def key(kind: str, model: str | None, payload: str) -> str:
        return hashlib.sha256(f"{kind}|{model}|{payload}".encode()).hexdigest()

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            self.data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")


def format_history(history: list[dict[str, str]]) -> str:
    """The last few messages as 'user: ...' / 'assistant: ...' lines, each cut short."""
    lines = [
        f"{m['role']}: {' '.join(str(m['content']).split())[:HISTORY_CHARS]}"
        for m in history[-2 * HISTORY_TURNS :]
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    return "\n".join(lines)


def _ask(
    kind: str, system: str, user: str, *, complete_fn: Callable[..., Any], model: str | None,
    cache: TransformCache | None, max_tokens: int,
) -> tuple[str | None, float, int]:  # fmt: skip
    """One model call with caching. Returns (text or None on failure, cost, calls made)."""
    key = TransformCache.key(kind, model, system + "\n" + user)
    if cache is not None and (hit := cache.get(key)) is not None:
        return str(hit["text"]), 0.0, 0
    try:
        result = complete_fn(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            name=f"query-{kind}", max_tokens=max_tokens, temperature=0.0,
            **({"model": model} if model else {}),
        )  # fmt: skip
    except LLMError as exc:
        logger.warning("query transform failed", extra={"kind": kind, "error": str(exc)[:200]})
        return None, 0.0, 0
    if cache is not None:
        cache.put(key, {"text": result.text})
    return str(result.text), float(result.cost_usd or 0.0), 1


# ---- 1. rewrite ------------------------------------------------------------
REWRITE_SYSTEM = (
    "You rewrite the user's LAST question so it can be understood without the conversation.\n"
    "Rules: resolve pronouns and references ('it', 'they', 'these sessions') using the "
    "conversation; keep the user's language; do not answer the question; do not add facts that are "
    "not in the conversation; if the question is already complete and standalone, return it "
    "unchanged. Return ONLY the question, on one line."
)


def clean_question(text: str | None, original: str) -> str | None:
    """A usable one-line question, or None. Rejects empty, multi-line, huge or answer-like text."""
    if not text:
        return None
    line = text.strip().strip("\"'`“”").strip()
    if not line or "\n" in line or len(line) > MAX_QUESTION_CHARS:
        return None
    if len(line) > 3 * max(len(original), 40):  # an answer or an essay, not a question
        return None
    return line


def rewrite_query(
    question: str,
    history: list[dict[str, str]],
    *,
    complete_fn: Callable[..., Any] = complete,
    model: str | None = None,
    cache: TransformCache | None = None,
) -> Transformed:
    if not history:
        return Transformed(question)  # nothing to resolve: no model call
    context = format_history(history)
    if not context:
        return Transformed(question)
    user = f"Conversation so far:\n{context}\n\nLast question: {question}"
    text, cost, calls = _ask(
        "rewrite", REWRITE_SYSTEM, user, complete_fn=complete_fn, model=model, cache=cache,
        max_tokens=120,
    )  # fmt: skip
    rewritten = clean_question(text, question)
    if rewritten is None:
        return Transformed(question, cost, calls)
    return Transformed(rewritten, cost, calls, used=rewritten != question)


# ---- 2. multi-query ------------------------------------------------------------
EXPAND_SYSTEM = (
    "You help a search engine over company documents. Given a question, write {n} DIFFERENT "
    "rephrasings that use different words (synonyms, other ways the document might say it). Keep "
    "the meaning; do not add new facts. "
    'Return JSON only: {{"queries": ["...", "..."]}}'
)


def expand_queries(
    question: str,
    n: int,
    *,
    complete_fn: Callable[..., Any] = complete,
    model: str | None = None,
    cache: TransformCache | None = None,
) -> tuple[list[str], float, int]:
    """Up to n rephrasings (never the original), plus cost and calls."""
    if n <= 0:
        return [], 0.0, 0
    text, cost, calls = _ask(
        "expand", EXPAND_SYSTEM.format(n=n), question, complete_fn=complete_fn, model=model,
        cache=cache, max_tokens=250,
    )  # fmt: skip
    try:
        raw = json.loads(
            re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
        )
        candidates = raw["queries"] if isinstance(raw, dict) else []
    except (json.JSONDecodeError, KeyError, TypeError):
        return [], cost, calls
    seen = {question.strip().lower()}
    out: list[str] = []
    for c in candidates if isinstance(candidates, list) else []:
        q = clean_question(str(c), question)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:n], cost, calls


# ---- 3. HyDE ------------------------------------------------------------
HYDE_SYSTEM = (
    "Write a short passage (about 70 words) of the kind that would appear in a company's "
    "documentation and would answer the question. Write it as the document itself, in plain prose: "
    "no preamble, do not mention the question, do not say you are guessing. Invent plausible "
    "specifics if needed; this passage is only used to find similar real text."
)


def hyde_passage(
    question: str,
    *,
    complete_fn: Callable[..., Any] = complete,
    model: str | None = None,
    cache: TransformCache | None = None,
) -> Transformed:
    text, cost, calls = _ask(
        "hyde", HYDE_SYSTEM, question, complete_fn=complete_fn, model=model, cache=cache,
        max_tokens=200,
    )  # fmt: skip
    passage = " ".join((text or "").split())[:MAX_HYDE_CHARS]
    if len(passage) < 20:
        return Transformed(question, cost, calls)
    return Transformed(passage, cost, calls, used=True)
