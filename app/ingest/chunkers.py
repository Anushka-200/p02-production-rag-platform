"""Picks the chunking strategy from settings, so experiments only change configuration."""

from app.config import Settings, get_settings
from app.embeddings import get_token_counter
from app.ingest.chunking import Chunk, TokenCounter, chunk_recursive
from app.ingest.chunking_structured import chunk_structured
from app.ingest.models import ParsedDocument


def chunk_document(
    doc: ParsedDocument,
    settings: Settings | None = None,
    count: TokenCounter | None = None,
) -> list[Chunk]:
    s = settings or get_settings()
    count = count or get_token_counter(s)  # the embedding model's own tokenizer for local models
    if s.chunk_strategy == "recursive":
        return chunk_recursive(doc, s.chunk_tokens, s.chunk_overlap_tokens, count)
    return chunk_structured(doc, s.chunk_tokens, s.chunk_overlap_tokens, s.parent_tokens, count)
