"""DOCX parser: headings from styles, lists, and tables as pipe tables.

Word files have no fixed pages (pagination is decided when the file is printed or opened), so
Section.page stays None. Headers and footers live outside the body and are not read.
"""

import io
from collections import Counter
from pathlib import PurePath

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingest.common import slugify, table_to_markdown
from app.ingest.models import ParsedDocument, ParseError, Section


def _heading_level(style_name: str) -> int | None:
    """ "Heading 2" -> 2, "Title" -> 1, anything else -> None."""
    if style_name == "Title":
        return 1
    if style_name.startswith("Heading "):
        digits = style_name.removeprefix("Heading ").strip()
        return int(digits) if digits.isdigit() else None
    return None


def _style_name(paragraph: Paragraph) -> str:
    return paragraph.style.name if paragraph.style is not None and paragraph.style.name else ""


def _is_list_item(paragraph: Paragraph) -> bool:
    ppr = paragraph._p.pPr
    return _style_name(paragraph).startswith("List") or (ppr is not None and ppr.numPr is not None)


def _table_rows(table: Table) -> list[list[str]]:
    rows = []
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:
            if cell._tc in seen:  # a merged cell is returned once per column it spans
                continue
            seen.add(cell._tc)
            cells.append(cell.text)
        rows.append(cells)
    return rows


def parse_docx(raw: bytes, filename: str = "") -> ParsedDocument:
    try:
        document = docx.Document(io.BytesIO(raw))
    except Exception as exc:
        raise ParseError(f"cannot open DOCX: {exc}") from exc

    # Paragraphs and tables in the order they appear in the document.
    blocks: list[tuple[str, int | None, str]] = []  # (kind, heading level, text)
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            level = _heading_level(_style_name(paragraph))
            if level:
                blocks.append(("heading", level, text))
            elif _is_list_item(paragraph):
                blocks.append(("body", None, "- " + text))
            else:
                blocks.append(("body", None, text))
        elif child.tag == qn("w:tbl"):
            markdown = table_to_markdown(_table_rows(Table(child, document)))
            if markdown:
                blocks.append(("table", None, markdown))

    first_heading = next((b[2] for b in blocks if b[0] == "heading"), "")
    title = (
        (document.core_properties.title or "").strip()
        or first_heading
        or PurePath(filename).stem
        or "Untitled"
    )

    sections: list[Section] = []
    seen: Counter[str] = Counter()
    stack: list[tuple[int, str]] = []
    pending: list[str] = []
    anchor = slugify(title)

    def flush() -> None:
        text = "\n\n".join(pending).strip()
        if text:
            sections.append(
                Section(heading_path=(title, *[h for _, h in stack]), anchor=anchor, text=text)
            )
        pending.clear()

    for kind, level, text in blocks:
        if kind != "heading" or level is None:
            pending.append(text)
            continue
        flush()
        if level == 1 and text == title:
            stack.clear()
        else:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        anchor = slugify(text)
        seen[anchor] += 1
        if seen[anchor] > 1:
            anchor = f"{anchor}-{seen[anchor] - 1}"
    flush()

    return ParsedDocument(title=title, source_type="docx", sections=sections)
