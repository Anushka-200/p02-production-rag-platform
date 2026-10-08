-- Semantic cache: a finished answer is kept, keyed by the MEANING of the question, so a near-identical
-- question later is answered without searching or calling the model.
--
-- Safety rules (docs/adr/012-semantic-cache.md):
--   * An entry belongs to one org AND one user. Other users of the org may read it only when
--     `shareable` is true: nothing in the context the model saw came from a private document.
--   * `doc_refs` lists the documents (and versions) the answer was built from. A lookup re-checks that
--     every one still exists, is the active version, and is visible to the asker; otherwise the entry
--     is stale and is ignored (and deleted).
--   * Anonymous visitors (widget, WhatsApp) never read or write the cache: no user, no policy match.

CREATE TABLE semantic_cache (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id     uuid NOT NULL,
    user_id    uuid NOT NULL,
    shareable  boolean NOT NULL DEFAULT false,
    question   text NOT NULL,
    embedding  vector(384) NOT NULL,
    answer     jsonb NOT NULL,                 -- {text, citations, sources, model}
    doc_refs   jsonb NOT NULL,                 -- [{"document_id": "...", "version": 3}]
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (user_id, org_id) REFERENCES users (id, org_id) ON DELETE CASCADE
);
CREATE INDEX semantic_cache_org_idx ON semantic_cache (org_id, created_at DESC);
-- No HNSW index: a tenant's cache is small (capped in code); an exact scan inside the org is exact and fast.

ALTER TABLE semantic_cache ENABLE ROW LEVEL SECURITY;
CREATE POLICY read_own_or_shared ON semantic_cache FOR SELECT
    USING (org_id = app_org() AND app_user() IS NOT NULL AND (user_id = app_user() OR shareable));
CREATE POLICY write_own ON semantic_cache FOR ALL
    USING (org_id = app_org() AND user_id = app_user())
    WITH CHECK (org_id = app_org() AND user_id = app_user());
