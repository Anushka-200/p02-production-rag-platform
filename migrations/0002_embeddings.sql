-- Dense search: the embedding column and its HNSW index, plus the anchors a chunk covers.
-- vector(384) matches the default local model (BAAI/bge-small-en-v1.5). A model with another
-- size needs its own migration and a full re-embed.

-- Chunker B can merge several sections into one chunk; the eval's "gold" locator is a section
-- anchor, so a chunk must remember every anchor it covers.
ALTER TABLE chunks ADD COLUMN anchors text[] NOT NULL DEFAULT '{}';

-- Only child chunks (level 0) are embedded. Parents (level 1) are fetched by id, never searched.
ALTER TABLE chunks ADD COLUMN embedding vector(384);
ALTER TABLE chunks ADD COLUMN embedding_model text;
ALTER TABLE chunks ADD CONSTRAINT embedding_has_model
    CHECK (embedding IS NULL OR embedding_model IS NOT NULL);  -- never mix up which model wrote it

-- HNSW with cosine distance: queries must use the <=> operator to use this index.
-- m = connections per chunk, ef_construction = build quality (pgvector defaults; tuned in Phase 6).
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
