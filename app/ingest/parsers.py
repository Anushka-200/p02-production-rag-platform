"""Parsers for Markdown (incl. Hugo/MDX handbooks), plain text and HTML.

All return a ParsedDocument. Design choices:
- Headings define sections, and the heading path is kept ("Benefits > Parental leave").
- Markdown text is sliced from the original, not re-rendered, so tables and lists survive.
- Build-time markup (Hugo shortcodes, MDX imports, JSX tags) is removed; the text inside is kept.
"""

import re
from collections import Counter
from pathlib import PurePath
from typing import Any

import yaml
from markdown_it import MarkdownIt
from selectolax.parser import HTMLParser

from app.ingest.common import slugify
from app.ingest.docx import parse_docx
from app.ingest.models import ParsedDocument, Section
from app.ingest.pdf import parse_pdf

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
# {{% name args %}} / {{< name args >}} and their closing forms
_SHORTCODE = re.compile(r"\{\{[%<]\s*(/?)\s*([\w/\-.]+)([^}]*?)[%>]\}\}")
_TITLE_ATTR = re.compile(r'(?:title|summary)\s*=\s*"([^"]*)"')
# JSX component tags start with a capital letter (<PrivateLink ...>, </Tab>, <Image />)
_JSX_TAG = re.compile(r"</?[A-Z][A-Za-z0-9]*\b[^<>]*?/?>")
_MDX_IMPORT = re.compile(r"^(import|export)\s.*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# Hugo's escaped examples: {{</* name */>}} is a shortcode shown as literal text on the site
_ESCAPED_SHORTCODE = re.compile(r"\{\{[%<]/\*.*?\*/[%>]\}\}")

_md = MarkdownIt()
_HIDDEN_NODE_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|pt|em|rem)?\b"
    r"|color\s*:\s*(?:#fff(?:fff)?\b|white\b)",
    re.IGNORECASE,
)


def decode(raw: bytes) -> str:
    """Bytes -> text. Never fails: bad bytes become the replacement character."""
    return raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        meta = None
    return (meta if isinstance(meta, dict) else {}), text[m.end() :]


def _clean_line(line: str) -> str | None:
    """Strip Hugo shortcodes and JSX tags from one line; None means drop the line (MDX import)."""
    if _MDX_IMPORT.match(line):
        return None

    def shortcode(m: re.Match[str]) -> str:
        # An opening alert/details keeps its title ("Warning:"); everything else vanishes.
        if m.group(1):
            return ""
        t = _TITLE_ATTR.search(m.group(3))
        return f"{t.group(1)}: " if t else ""

    line = _ESCAPED_SHORTCODE.sub("", line)
    line = _SHORTCODE.sub(shortcode, line)
    return _JSX_TAG.sub("", line)


def _drop_comments(line: str, in_comment: bool) -> tuple[str | None, bool]:
    """Remove <!-- ... --> (also across lines). Returns (line or None if nothing is left, state)."""
    kept: list[str] = []
    pos = 0
    while pos < len(line):
        if in_comment:
            end = line.find("-->", pos)
            if end == -1:
                pos = len(line)
            else:
                in_comment, pos = False, end + 3
        else:
            start = line.find("<!--", pos)
            if start == -1:
                kept.append(line[pos:])
                pos = len(line)
            else:
                kept.append(line[pos:start])
                in_comment, pos = True, start + 4
    out = "".join(kept)
    return (out if out.strip() or not line.strip() else None), in_comment


def _strip_markup(text: str) -> str:
    """Remove Hugo shortcodes, MDX imports, JSX tags; keep inner text. Code fences are untouched."""
    out: list[str] = []
    in_fence = False
    in_comment = False  # inside an HTML comment: invisible to readers, so never indexed
    fence_start = 0  # index in `out` of the line that opened the current fence
    for line in text.split("\n"):
        if not in_fence:
            visible, in_comment = _drop_comments(line, in_comment)
            if visible is None:
                continue
            line = visible
        fence = _FENCE.match(line)
        if fence and line.count(fence.group(1)) < 2:  # 2+ on one line = inline code, not a fence
            in_fence = not in_fence
            fence_start = len(out)
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            cleaned = _clean_line(line)
            if cleaned is not None:
                out.append(cleaned)
    if in_fence:
        # Real pages sometimes forget to close a fence. Treat it as unclosed and clean what follows,
        # otherwise the rest of the page would be left as raw markup.
        for i in range(fence_start + 1, len(out)):
            out[i] = _clean_line(out[i]) or ""
    return "\n".join(out)


def parse_markdown(raw: bytes | str, filename: str = "") -> ParsedDocument:
    text = decode(raw) if isinstance(raw, bytes) else raw
    meta, body = _split_front_matter(text)
    body = _strip_markup(body)
    title = str(meta.get("title") or "").strip()

    lines = body.split("\n")
    # (line index where the heading is, level, heading text) for every real heading
    tokens = _md.parse(body)
    headings: list[tuple[int, int, str]] = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.map:
            headings.append((tok.map[0], int(tok.tag[1]), tokens[i + 1].content.strip()))

    if not title:
        first_h1 = next((h[2] for h in headings if h[1] == 1), "")
        title = first_h1 or PurePath(filename).stem or "Untitled"

    sections: list[Section] = []
    seen: Counter[str] = Counter()
    stack: list[tuple[int, str]] = []  # heading path so far: (level, text)

    def add(path: tuple[str, ...], anchor: str, chunk_lines: list[str]) -> None:
        body_text = "\n".join(chunk_lines).strip()
        if body_text:
            sections.append(Section(heading_path=path, anchor=anchor, text=body_text))

    # Text before the first heading belongs to the title itself.
    first = headings[0][0] if headings else len(lines)
    add((title,), slugify(title), lines[:first])

    for n, (start, level, htext) in enumerate(headings):
        end = headings[n + 1][0] if n + 1 < len(headings) else len(lines)
        if level == 1 and htext.lower() == title.lower():
            stack = []  # the page's own h1 is the root, not a child of it
        else:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, htext))
        anchor = slugify(htext)
        seen[anchor] += 1
        if seen[anchor] > 1:  # repeated heading text gets a suffix, like browsers do
            anchor = f"{anchor}-{seen[anchor] - 1}"
        path = (title, *[h for _, h in stack])
        add(path, anchor, lines[start + 1 : end])

    return ParsedDocument(title=title, source_type="md", sections=sections, metadata=meta)


def parse_text(raw: bytes | str, filename: str = "") -> ParsedDocument:
    text = (decode(raw) if isinstance(raw, bytes) else raw).strip()
    title = PurePath(filename).stem or "Untitled"
    sections = [Section(heading_path=(title,), anchor=slugify(title), text=text)] if text else []
    return ParsedDocument(title=title, source_type="txt", sections=sections)


_DROP_TAGS = "script, style, noscript, nav, header, footer, aside, form, iframe, svg"
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "tr"}
_NESTED_PARENTS = {"li", "blockquote", "tr", "pre"}


def parse_html(raw: bytes | str, filename: str = "") -> ParsedDocument:
    html = decode(raw) if isinstance(raw, bytes) else raw
    tree = HTMLParser(html)
    for node in tree.css(_DROP_TAGS):
        node.decompose()
    for node in tree.css("[style], [hidden]"):  # text a human cannot see is not content
        style = node.attributes.get("style") or ""
        if "hidden" in node.attributes or _HIDDEN_NODE_STYLE.search(style):
            node.decompose()
    title_node = tree.css_first("title")
    title = (
        (title_node.text(strip=True) if title_node else "") or PurePath(filename).stem or "Untitled"
    )
    container = tree.css_first("main") or tree.css_first("article") or tree.body
    if container is None:
        return ParsedDocument(title=title, source_type="html", sections=[])

    sections: list[Section] = []
    seen: Counter[str] = Counter()
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    cur_path: tuple[str, ...] = (title,)
    cur_anchor = slugify(title)

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(heading_path=cur_path, anchor=cur_anchor, text=text))
        buf.clear()

    # traverse() walks in document order; css("a, b") would return all a's, then all b's.
    for node in container.traverse():
        if node.tag not in _BLOCK_TAGS:
            continue
        # A <p> inside an <li> is already covered by the <li>: skip nested duplicates.
        parent, nested = node.parent, False
        while parent is not None and parent != container:
            if parent.tag in _NESTED_PARENTS:
                nested = True
                break
            parent = parent.parent
        if nested:
            continue
        tag = node.tag
        if tag and len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            flush()
            level, htext = int(tag[1]), node.text(strip=True)
            if not htext:
                continue
            if level == 1 and htext.lower() == title.lower():
                stack = []
            else:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, htext))
            cur_path = (title, *[h for _, h in stack])
            cur_anchor = slugify(htext)
            seen[cur_anchor] += 1
            if seen[cur_anchor] > 1:
                cur_anchor = f"{cur_anchor}-{seen[cur_anchor] - 1}"
        elif tag == "tr":
            cells = [c.text(strip=True) for c in node.css("th, td")]
            if any(cells):
                buf.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        elif tag == "li":
            buf.append("- " + node.text(separator=" ", strip=True))
        else:
            text = node.text(separator=" ", strip=True)
            if text:
                buf.append(text)
    flush()
    return ParsedDocument(title=title, source_type="html", sections=sections)


PARSERS = {
    "md": parse_markdown,
    "txt": parse_text,
    "html": parse_html,
    "pdf": parse_pdf,
    "docx": parse_docx,
}


def parse(source_type: str, raw: bytes, filename: str = "") -> ParsedDocument:
    try:
        return PARSERS[source_type](raw, filename)
    except KeyError:
        raise ValueError(f"no parser for source type {source_type!r}") from None
