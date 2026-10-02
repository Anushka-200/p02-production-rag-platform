"""Context assembly: turn the best chunks into labelled, budgeted, safely wrapped prompt text.

Each chunk becomes a block the answer can cite:
    <source id="S1" document="gitlab/..." section="Benefits > Time off" pages="5-6"> ... </source>
Rules: best first; stop at the token budget; skip duplicates; never return nothing just because
the best chunk is big (truncate it); and neutralise any text that could close a block early,
because document text is UNTRUSTED (it may contain instructions or fake tags).
"""

import re
import uuid
from dataclasses import dataclass, field

from app.ingest.chunking import TokenCounter, approx_tokens
from app.retrieval.rerank import Reranked


@dataclass
class Source:
    id: str  # "S1", "S2"... the label the answer cites
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document: str  # source_uri
    section: str
    pages: str | None
    anchors: list[str]
    text: str
    tokens: int


@dataclass
class Context:
    text: str  # all source blocks, ready for the prompt
    sources: list[Source]
    tokens: int
    dropped: int = 0  # candidates left out (budget, duplicates)
    notes: list[str] = field(default_factory=list)


def _pages(start: int | None, end: int | None) -> str | None:
    if start is None:
        return None
    return str(start) if end in (None, start) else f"{start}-{end}"


def _attr(value: str) -> str:
    """A label safe inside an attribute: quotes and '<' could break it ('>' is fine)."""
    return value.replace('"', "'").replace("<", "(")


def _safe(text: str) -> str:
    """Stop the document text from closing or opening our blocks."""
    return re.sub(r"<(/?)source", r"&lt;\1source", text, flags=re.IGNORECASE)


def _block(source_id: str, document: str, section: str, pages: str | None, text: str) -> str:
    page_attr = f' pages="{pages}"' if pages else ""
    doc, sec = _attr(document), _attr(section)
    head = f'<source id="{source_id}" document="{doc}" section="{sec}"{page_attr}>'
    return f"{head}\n{text}\n</source>"


def build_context(
    items: list[Reranked], budget_tokens: int = 3000, count: TokenCounter = approx_tokens
) -> Context:
    sources: list[Source] = []
    blocks: list[str] = []
    used, dropped, seen_ids = 0, 0, set()
    notes: list[str] = []
    for item in items:
        c = item.fused.candidate
        text = _safe(c.text)
        if c.chunk_id in seen_ids or any(text in s.text for s in sources):
            dropped += 1  # the same chunk, or one already fully contained in a better one
            continue
        sid = f"S{len(sources) + 1}"
        pages = _pages(c.page_start, c.page_end)
        block = _block(sid, c.source_uri, c.section_path, pages, text)
        size = count(block)
        if used + size > budget_tokens:
            if not sources:  # never an empty context: cut the best chunk to fit
                head = count(_block(sid, c.source_uri, c.section_path, pages, ""))
                words, keep = text.split(), []
                for word in words:
                    if count(" ".join([*keep, word])) + head > budget_tokens:
                        break
                    keep.append(word)
                text = " ".join(keep)
                block, size = _block(sid, c.source_uri, c.section_path, pages, text), 0
                size = count(block)
                notes.append(f"{sid} was cut to fit the budget")
            else:
                dropped += 1
                continue
        seen_ids.add(c.chunk_id)
        sources.append(
            Source(
                sid,
                c.chunk_id,
                c.document_id,
                c.source_uri,
                c.section_path,
                pages,
                c.anchors,
                text,
                size,
            )
        )
        blocks.append(block)
        used += size
    return Context("\n\n".join(blocks), sources, used, dropped, notes)
