"""Cleaning after parsing and before chunking.

Each step was chosen from measurements on the real corpus (see PLAN.md / task 1.5):
- PDF page furniture ("6 Publication 15-B (2026)") repeats on every page and would be embedded
  into every chunk, so it is removed.
- Exact duplicate sections inside one document are dropped (7 of 3,276 in the GitLab slice).
- Cross-document duplicates are NOT handled: only 4 paragraphs repeated on 3+ pages, not worth it.
- Tables are exposed as blocks so the chunkers never cut through one (see split_blocks/split_table).
"""

import re
from collections import defaultdict
from dataclasses import dataclass, replace

from app.ingest.models import ParsedDocument, Section

FURNITURE_MIN_PAGES = 5  # short PDFs are left alone: too little evidence to call anything repeated
FURNITURE_PAGE_FRACTION = 0.4  # a short paragraph shape on 40%+ of pages is a header or footer
FURNITURE_MAX_CHARS = 100  # real content paragraphs are longer; this protects against false hits
DUPLICATE_MIN_WORDS = 20  # tiny repeated sections ("See above.") are not worth the risk

_PARAGRAPH_BREAK = re.compile(r"\n{2,}")
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿"), None)  # zero-width characters


def clean_text(text: str) -> str:
    """Tidy characters that look like nothing but break matching."""
    text = text.translate(_INVISIBLE).replace(" ", " ")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _shape(paragraph: str) -> str:
    """Paragraph with digits replaced by '#': page 6 and page 7 footers get the same shape."""
    return re.sub(r"\d+", "#", " ".join(paragraph.split()))


def remove_page_furniture(doc: ParsedDocument) -> tuple[ParsedDocument, int]:
    """Drop short paragraphs that repeat on many pages of a PDF. Returns (document, count)."""
    pages = doc.metadata.get("pages", 0)
    if doc.source_type != "pdf" or pages < FURNITURE_MIN_PAGES:
        return doc, 0

    pages_with_shape: dict[str, set[int]] = defaultdict(set)
    for section in doc.sections:
        for paragraph in _PARAGRAPH_BREAK.split(section.text):
            if len(paragraph) <= FURNITURE_MAX_CHARS and "\n" not in paragraph:
                pages_with_shape[_shape(paragraph)].add(section.page or 0)
    furniture = {
        shape
        for shape, seen_on in pages_with_shape.items()
        if len(seen_on) >= max(FURNITURE_MIN_PAGES, FURNITURE_PAGE_FRACTION * pages)
    }

    removed, kept = 0, []
    for section in doc.sections:
        paragraphs = []
        for paragraph in _PARAGRAPH_BREAK.split(section.text):
            if _shape(paragraph) in furniture and "\n" not in paragraph:
                removed += 1
            else:
                paragraphs.append(paragraph)
        text = "\n\n".join(paragraphs).strip()
        if text:  # a section that held only furniture disappears
            kept.append(replace(section, text=text))
    return replace(doc, sections=kept), removed


def drop_duplicate_sections(doc: ParsedDocument) -> tuple[ParsedDocument, int]:
    """Drop later sections whose text is identical to an earlier one in the same document."""
    seen: set[str] = set()
    kept: list[Section] = []
    for section in doc.sections:
        key = " ".join(section.text.lower().split())
        if len(key.split()) >= DUPLICATE_MIN_WORDS and key in seen:
            continue
        seen.add(key)
        kept.append(section)
    return replace(doc, sections=kept), len(doc.sections) - len(kept)


def preprocess(doc: ParsedDocument) -> ParsedDocument:
    """Run every cleaning step and record what was removed in doc.metadata["preprocess"]."""
    doc = replace(
        doc,
        sections=[s for s in (replace(s, text=clean_text(s.text)) for s in doc.sections) if s.text],
    )
    doc, furniture = remove_page_furniture(doc)
    doc, duplicates = drop_duplicate_sections(doc)
    stats = {"furniture_paragraphs_removed": furniture, "duplicate_sections_dropped": duplicates}
    return replace(doc, metadata={**doc.metadata, "preprocess": stats})


# ---- blocks and tables: what the chunkers need so they never cut through a table ---------------
@dataclass
class Block:
    kind: str  # "paragraph" | "table" | "code"
    text: str


def split_blocks(text: str) -> list[Block]:
    """Cut section text into paragraphs, pipe tables and fenced code blocks, in order."""
    blocks: list[Block] = []
    current: list[str] = []
    kind = "paragraph"

    def flush() -> None:
        if current:
            blocks.append(Block(kind, "\n".join(current).strip()))
            current.clear()

    in_code = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")) and stripped.count("```") < 2:
            if not in_code:
                flush()
                kind = "code"
            current.append(line)
            if in_code:
                flush()
                kind = "paragraph"
            in_code = not in_code
        elif in_code:
            current.append(line)
        elif stripped.startswith("|"):
            if kind != "table":
                flush()
                kind = "table"
            current.append(line)
        elif not stripped:
            flush()
            kind = "paragraph"
        else:
            if kind == "table":  # a table ends at the first non-table line
                flush()
                kind = "paragraph"
            current.append(line)
    flush()
    return [b for b in blocks if b.text]


def split_table(table: str, max_chars: int) -> list[str]:
    """Split a long pipe table into parts of at most max_chars. Every part repeats the header.

    A single row longer than the limit stays whole: cutting a row would destroy its meaning.
    """
    if len(table) <= max_chars:
        return [table]
    lines = table.split("\n")
    header, rows = lines[:2], lines[2:]  # header row + the |---| separator row
    if len(rows) == 0:
        return [table]
    head_text = "\n".join(header)
    parts: list[str] = []
    current: list[str] = []
    size = len(head_text)
    for row in rows:
        if current and size + 1 + len(row) > max_chars:
            parts.append("\n".join([head_text, *current]))
            current, size = [], len(head_text)
        current.append(row)
        size += 1 + len(row)
    if current:
        parts.append("\n".join([head_text, *current]))
    return parts
