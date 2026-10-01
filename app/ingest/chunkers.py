"""Picks the chunking strategy from settings, so experiments only change configuration."""

from app.config import Settings, get_settings
from app.ingest.chunking import Chunk, TokenCounter, approx_tokens, chunk_recursive
from app.ingest.chunking_structured import chunk_structured
from app.ingest.models import ParsedDocument


def chunk_document(
    doc: ParsedDocument,
    settings: Settings | None = None,
    count: TokenCounter = approx_tokens,
) -> list[Chunk]:
    s = settings or get_settings()
    if s.chunk_strategy == "recursive":
        return chunk_recursive(doc, s.chunk_tokens, s.chunk_overlap_tokens, count)
    return chunk_structured(doc, s.chunk_tokens, s.chunk_overlap_tokens, s.parent_tokens, count)
