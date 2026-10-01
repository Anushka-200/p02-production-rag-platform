"""Chunker A: recursive splitting by tokens, inside each section.

Chunks are what we embed and search. This is the simple baseline strategy:
- A section that fits the size limit becomes one chunk (small sections are NOT merged).
- A bigger section is cut at the most natural boundary that works: between blocks (paragraph,
  table, code), then between sentences, then between words. Tables and code are never cut
  mid-row / mid-line unless one row or line alone is bigger than the limit.
- Pieces are packed up to `chunk_tokens`. When a section needs several chunks, the next chunk
  starts with the tail of the previous one ("overlap") so an idea on the boundary survives.
- Chunks never cross a section boundary, so each keeps one heading path, one anchor, one page.
Chunker B (task 1.7) differs: it merges small sections and links children to parent sections.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from app.ingest.models import ParsedDocument
from app.ingest.preprocess import Block, split_blocks, split_table

TokenCounter = Callable[[str], int]

# A sentence ends at . ! or ? followed by whitespace and something that starts a new sentence.
# Simple on purpose: "Dept. of" may be split wrongly, which only makes a slightly smaller piece.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[•-])")


def approx_tokens(text: str) -> int:
    """Cheap estimate: 1 token is about 4 characters for English prose.

    Tables and code use more tokens per character, so this under-counts them. Task 1.8 swaps in
    the real tokenizer of the chosen embedding model through the TokenCounter parameter.
    """
    return max(1, math.ceil(len(text) / 4))


@dataclass
class Chunk:
    text: str
    heading_path: tuple[str, ...]
    anchor: str
    page_start: int | None
    page_end: int | None
    token_count: int
    index: int  # position of the chunk in its document, starting at 0
    level: int = 0  # 0 = child chunk (searched); chunker B adds 1 = parent, 2 = summary
    anchors: tuple[str, ...] = ()  # every section anchor the chunk covers (B can merge several)
    parent_index: int | None = None  # index of this child's parent chunk, if it has one


@dataclass
class _Piece:
    sep: str  # what goes before this piece when it follows another one in the same chunk
    text: str


def _words_split(text: str, limit: int, count: TokenCounter) -> list[str]:
    """Last resort for text with no sentence or line breaks: pack words up to the limit."""
    pieces, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and count(candidate) > limit:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _block_pieces(block: Block, limit: int, count: TokenCounter) -> list[_Piece]:
    """Cut one block into pieces that each fit the limit, as gently as possible."""
    if count(block.text) <= limit:
        return [_Piece("\n\n", block.text)]

    if block.kind == "table":
        parts = split_table(block.text, limit, measure=count)  # every part repeats the header
        return [_Piece("\n\n", p) for p in parts]

    if block.kind == "code":
        units, sep = block.text.split("\n"), "\n"
    else:
        units, sep = _SENTENCE_END.split(block.text), " "

    pieces: list[_Piece] = []
    for unit in units:
        if count(unit) <= limit:
            parts = [unit]
        else:  # one sentence or line alone is too big: fall back to words
            parts = _words_split(unit, limit, count)
        pieces.extend(_Piece(sep, p) for p in parts)
    pieces[0].sep = "\n\n"  # the first piece of a block is separated from the previous block
    return pieces


def _join(pieces: list[_Piece]) -> str:
    return "".join((p.sep if i else "") + p.text for i, p in enumerate(pieces))


def _overlap_tail(pieces: list[_Piece], overlap: int, count: TokenCounter) -> list[_Piece]:
    """The last whole pieces whose total size stays within `overlap` tokens."""
    tail: list[_Piece] = []
    size = 0
    for piece in reversed(pieces):
        size += count(piece.text)
        if size > overlap:
            break
        tail.insert(0, piece)
    return tail


def split_text(
    text: str, chunk_tokens: int, overlap_tokens: int, count: TokenCounter = approx_tokens
) -> list[str]:
    """Split one section's text into chunk texts (the heart of chunker A)."""
    if count(text) <= chunk_tokens:
        return [text]
    pieces: list[_Piece] = []
    for block in split_blocks(text):
        pieces.extend(_block_pieces(block, chunk_tokens, count))

    chunks: list[str] = []
    current: list[_Piece] = []
    size = 0
    for piece in pieces:
        piece_size = count(piece.text)
        if current and size + count(piece.sep) + piece_size > chunk_tokens:
            chunks.append(_join(current))
            current = _overlap_tail(current, overlap_tokens, count)
            size = sum(count(p.text) for p in current)
            if size + piece_size > chunk_tokens:  # the overlap would not leave room: skip it
                current, size = [], 0
        current.append(piece)
        size += piece_size
    chunks.append(_join(current))
    return chunks


def chunk_recursive(
    doc: ParsedDocument,
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
    count: TokenCounter = approx_tokens,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        for text in split_text(section.text, chunk_tokens, overlap_tokens, count):
            chunks.append(
                Chunk(
                    text=text,
                    heading_path=section.heading_path,
                    anchor=section.anchor,
                    page_start=section.page,
                    page_end=section.page,
                    token_count=count(text),
                    index=len(chunks),
                    anchors=(section.anchor,),
                )
            )
    return chunks
