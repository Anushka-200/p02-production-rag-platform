"""Ingestion scan: look for text that is aimed at the AI instead of at a human reader.

Anyone who can upload a file (or edit a wiki page) can plant instructions in it ("indirect prompt
injection"). The scan is a smoke alarm, not a wall: it FLAGS suspicious documents so a person can
review them, and the result is stored with the ingestion job. Documents are not silently dropped,
because a legitimate security handbook may quote an attack phrase.

What it looks for (each finding has a kind, a severity and a short snippet):
- instruction_phrase   "ignore previous instructions", "note to the AI assistant", ...
- hidden_comment       an HTML comment that contains such a phrase (invisible when rendered)
- invisible_chars      zero-width / bidi / Unicode "tag" characters (text a human cannot see)
- hidden_style         HTML made invisible with CSS (display:none, font-size:0, white text)
- fake_source_tag      "<source ...>" in the text: an attempt to fake one of our source blocks
- external_image       a remote image or link carrying data out when it is rendered

Separately from flagging, hidden text (comments, invisible CSS, zero-width characters) is removed
before indexing by the parsers, so it never reaches the model even if nobody reviews the flag.
"""

import re
from dataclasses import asdict, dataclass

SNIPPET_CHARS = 90

_INSTRUCTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+|the\s+|your\s+)*(?:previous|prior|above|earlier)\s+"
    r"(?:instructions?|rules?|prompts?|messages?)"
    r"|disregard\s+(?:all\s+|any\s+|the\s+|your\s+)*(?:previous|prior|above|earlier)?\s*"
    r"(?:instructions?|rules?|guidelines)"
    r"|(?:note|message|instructions?)\s+(?:to|for)\s+(?:the\s+)?(?:ai|llm|assistant|chatbot|model|language model)"  # noqa: E501
    r"|(?m:^)\s*(?:system|developer)\s*(?:prompt|message|override)?\s*:"
    r"|new\s+instructions?\s*:"
    r"|you\s+are\s+now\s+(?:in\s+)?(?:a|an|the)?\s*\w+\s*(?:mode|assistant)"
    r"|do\s+not\s+(?:tell|reveal|mention|inform)\s+(?:the\s+)?user"
    r"|(?:reveal|print|repeat|output)\s+(?:your\s+|the\s+)?(?:system\s+prompt|hidden\s+instructions)"
    r"|when\s+(?:the\s+user\s+|anyone\s+|someone\s+)?asks?\s+about\b[^.\n]{0,80}\b(?:answer|reply|respond|say)\b",
    re.IGNORECASE,
)
_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
# zero-width, bidi controls, and the Unicode "tag" block used to smuggle invisible ASCII
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]")
# Only inline style="..." attributes count (CSS-looking text in Mermaid charts or code is normal).
_STYLE_ATTR = re.compile(r"""style\s*=\s*(["'])(.*?)\1""", re.IGNORECASE | re.DOTALL)
_HIDING_CSS = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|pt|em|rem)?\b",
    re.IGNORECASE,
)
_WHITE_TEXT = re.compile(r"(?<![-\w])color\s*:\s*(?:#fff(?:fff)?\b|white\b)", re.IGNORECASE)
# Our own source blocks look like <source id="S1" document="...">; a bare </source> is never normal.
# HTML5 video uses <source src="..."> and is not matched.
_SOURCE_TAG = re.compile(
    r"</\s*source\s*>|<\s*source\b(?=[^>]*\b(?:id|document|section)\s*=)", re.IGNORECASE
)
# Only an image whose URL carries data in a query string can leak anything when rendered.
_EXTERNAL_IMAGE = re.compile(
    r"!\[[^\]]*\]\(\s*https?://[^)\s]*\?[^)\s]*="
    r"|<img\b[^>]*\bsrc\s*=\s*[\"']?https?://[^\"'>\s]*\?[^\"'>\s]*=",
    re.IGNORECASE,
)
TAG_CHARS_MIN = 8  # an emoji flag uses about 6 tag characters; a smuggled sentence uses many more
INVISIBLE_LINE_SHARE = 0.2


@dataclass(frozen=True)
class Finding:
    kind: str
    severity: str  # "high" | "medium"
    snippet: str


def _snippet(text: str) -> str:
    return " ".join(text.split())[:SNIPPET_CHARS]


def reveal_invisible(text: str) -> str:
    """The text with invisible characters removed: what a hidden payload says once assembled."""
    return _INVISIBLE.sub("", text)


def scan_text(text: str) -> list[Finding]:
    findings: list[Finding] = []
    seen_in_comments: set[str] = set()
    for m in _COMMENT.finditer(text):
        hit = _INSTRUCTION.search(m.group(1))
        if hit:
            seen_in_comments.add(hit.group(0).lower())
            findings.append(Finding("hidden_comment", "high", _snippet(m.group(1))))

    revealed = reveal_invisible(text)
    tag_chars = sum(1 for c in text if 0xE0000 <= ord(c) <= 0xE007F)
    payload_lines = [
        line
        for line in text.split("\n")
        if (n := len(_INVISIBLE.findall(line))) >= 3 and n / len(line) >= INVISIBLE_LINE_SHARE
    ]
    if tag_chars >= TAG_CHARS_MIN or payload_lines:
        # Stray zero-width characters are copy-paste debris and are ignored; a line made mostly of
        # them, or a run of Unicode tag characters, is hidden text. Its words show up once removed.
        hit = _INSTRUCTION.search(revealed)
        what = (
            f"{tag_chars} tag characters" if tag_chars else f"{len(payload_lines)} hidden line(s)"
        )
        findings.append(
            Finding(
                "invisible_chars",
                "high" if hit else "medium",
                what + (f"; reveals: {_snippet(hit.group(0))}" if hit else ""),
            )
        )

    for m in _INSTRUCTION.finditer(revealed):
        if m.group(0).lower() not in seen_in_comments:
            start = max(0, m.start() - 20)
            findings.append(
                Finding("instruction_phrase", "high", _snippet(revealed[start : m.end() + 60]))
            )
    for attr in _STYLE_ATTR.finditer(text):
        css = attr.group(2)
        if _HIDING_CSS.search(css) or (_WHITE_TEXT.search(css) and "background" not in css.lower()):
            findings.append(
                Finding("hidden_style", "medium", _snippet(text[attr.start() : attr.end() + 60]))
            )
            break
    if tag := _SOURCE_TAG.search(text):
        findings.append(
            Finding(
                "fake_source_tag", "high", _snippet(text[max(0, tag.start() - 20) : tag.end() + 80])
            )
        )
    if image := _EXTERNAL_IMAGE.search(text):
        findings.append(
            Finding("external_image", "medium", _snippet(text[image.start() : image.end() + 80]))
        )
    return findings


def summarize(findings: list[Finding], limit: int = 10) -> dict[str, object]:
    """What is stored with the ingestion job: small and JSON-safe."""
    return {
        "flagged": bool(findings),
        "high": sum(f.severity == "high" for f in findings),
        "findings": [asdict(f) for f in findings[:limit]],
    }
