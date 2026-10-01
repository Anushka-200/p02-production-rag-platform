"""PDF parser: text with page numbers, headings by font, tables as pipe tables.

PDFs have no real structure, only text placed at positions with a font and size. So:
- Page numbers come from the file itself, which is what makes "page 12" citations possible.
- A line is a heading if it is bold and clearly bigger than the body text. Bigger = higher level.
- Tables are found by PyMuPDF's table finder; their text is excluded from the normal text so it
  is not stored twice.
- A section never spans pages: each Section carries exactly one page number.

Steps: (1) read every page into blocks and tables, (2) learn the body font size, (3) turn each
page into "runs" (heading / body paragraph / table), (4) build sections from the runs.
"""

import unicodedata
from collections import Counter
from pathlib import PurePath
from typing import Any

import pymupdf

from app.ingest.common import join_wrapped_lines, slugify, table_to_markdown
from app.ingest.models import ParsedDocument, ParseError, Section

HEADING_MIN_RATIO = 1.25  # heading font must be at least 25% bigger than the body font
MIN_TEXT_CHARS = 20  # fewer characters than this in the whole file = no text layer (scanned)

Line = tuple[str, float, bool]  # (text, font size, is bold)
Run = tuple[str, str, float]  # (kind: "heading" | "body" | "table", text, font size)


def _is_bold(span: dict[str, Any]) -> bool:
    return "bold" in span["font"].lower() or bool(span["flags"] & 16)


def _read_page(page: pymupdf.Page, size_chars: Counter[float]) -> tuple[list[Any], int]:
    """Return the page's items in reading order: ("text", [Line, ...]) and ("table", markdown).

    Text inside a table is skipped (the table is emitted instead). Each table is placed just
    before the first text block that starts below it.
    """
    try:
        tables = page.find_tables().tables
    except Exception:  # the table finder can fail on odd pages; the text is still useful
        tables = []
    table_rects = [pymupdf.Rect(t.bbox) for t in tables]

    items: list[tuple[float, str, Any]] = []  # (y position, kind, payload)
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:  # 0 = text block; images are skipped
            continue
        x0, y0, x1, y1 = block["bbox"]
        center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any(rect.contains(center) for rect in table_rects):
            continue
        lines: list[Line] = []
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            # NFKC turns ligatures back into letters ("ﬁ" -> "fi"), else "benefits" never matches.
            text = unicodedata.normalize("NFKC", "".join(s["text"] for s in line["spans"])).strip()
            size = round(max(s["size"] for s in spans), 1)
            size_chars[size] += len(text)
            lines.append((text, size, all(_is_bold(s) for s in spans)))
        if lines:
            items.append((y0, "text", lines))

    for table, rect in zip(tables, table_rects, strict=True):
        markdown = table_to_markdown(table.extract())
        if markdown:
            position = next((i for i, it in enumerate(items) if it[0] > rect.y0), len(items))
            items.insert(position, (rect.y0, "table", markdown))
    return [(kind, payload) for _, kind, payload in items], len(tables)


def _runs(items: list[Any], is_heading: Any) -> list[Run]:
    """Group a page's lines into headings, body paragraphs and tables.

    A block is one paragraph, so body lines are only joined within their block. Heading lines
    of the same size are joined across blocks: "Simple Cafeteria Plans for Small" + "Businesses".
    """
    runs: list[Run] = []
    for kind, payload in items:
        if kind == "table":
            runs.append(("table", payload, 0.0))
            continue
        block_runs: list[tuple[str, list[str], float]] = []
        for text, size, bold in payload:
            k = "heading" if is_heading((text, size, bold)) else "body"
            if block_runs and block_runs[-1][0] == k and (k == "body" or block_runs[-1][2] == size):
                block_runs[-1][1].append(text)
            else:
                block_runs.append((k, [text], size))
        for k, lines, size in block_runs:
            if k == "heading" and runs and runs[-1][0] == "heading" and runs[-1][2] == size:
                runs[-1] = ("heading", runs[-1][1] + " " + " ".join(lines), size)
            elif k == "heading":
                runs.append(("heading", " ".join(lines), size))
            else:
                runs.append(("body", join_wrapped_lines(lines), size))
    return runs


def parse_pdf(raw: bytes, filename: str = "") -> ParsedDocument:
    try:
        doc = pymupdf.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise ParseError(f"cannot open PDF: {exc}") from exc
    if doc.needs_pass:
        raise ParseError("PDF is password protected")

    # (1) read every page
    size_chars: Counter[float] = Counter()
    page_items, n_tables = [], 0
    for page in doc.pages():
        items, found = _read_page(page, size_chars)
        page_items.append(items)
        n_tables += found
    if sum(size_chars.values()) < MIN_TEXT_CHARS:
        raise ParseError("PDF has no extractable text (scanned images? OCR is not enabled)")

    # (2) the body font size is the one most characters are set in
    body_size = size_chars.most_common(1)[0][0]

    def is_heading(line: Line) -> bool:
        return line[2] and line[1] >= body_size * HEADING_MIN_RATIO

    # (3) runs per page
    page_runs = [_runs(items, is_heading) for items in page_items]
    heading_sizes = sorted(
        {r[2] for runs in page_runs for r in runs if r[0] == "heading"}, reverse=True
    )
    level_of = {size: i + 1 for i, size in enumerate(heading_sizes)}  # biggest font = level 1

    # (4) sections. Title: file metadata, else the first heading, else the filename.
    first_heading = next((r[1] for runs in page_runs for r in runs if r[0] == "heading"), "")
    title = (
        (doc.metadata.get("title") or "").strip()
        or first_heading
        or PurePath(filename).stem
        or "Untitled"
    )
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []

    for page_no, runs in enumerate(page_runs, start=1):
        paragraphs: list[str] = []

        def flush(page_no: int = page_no, paragraphs: list[str] = paragraphs) -> None:
            text = "\n\n".join(paragraphs).strip()
            if text:
                last = stack[-1][1] if stack else title
                sections.append(
                    Section(
                        heading_path=(title, *[h for _, h in stack]),
                        anchor=f"p{page_no}-{slugify(last)}",
                        text=text,
                        page=page_no,
                    )
                )
            paragraphs.clear()

        for kind, text, size in runs:
            if kind != "heading":
                paragraphs.append(text)
                continue
            flush()
            if text == title:  # the document's own title heading is the root, not a child
                stack.clear()
                continue
            level = level_of[size]
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        flush()

    return ParsedDocument(
        title=title,
        source_type="pdf",
        sections=sections,
        metadata={"pages": len(doc), "tables": n_tables, "body_font_size": body_size},
    )
