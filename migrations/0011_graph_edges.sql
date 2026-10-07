-- GraphRAG slice: facts of the form (subject) --relation--> (object), each pointing at the chunk
-- that states it. The "graph" is simply this table: a node is a normalised name, an edge is a row,
-- and walking the graph is a join. (A graph database such as Neo4j is the next step if the
-- project needs deep traversal; this slice compares the IDEA against plain retrieval.)
--
-- Same privacy rule as summaries: an edge is DERIVED from a chunk, so it carries that chunk's
-- visibility and owner, follows the same read/write policies, is deleted with the chunk (FK), and
-- is deleted by a trigger when the chunk loses visibility (the fact would otherwise outlive the
-- permission to read it).

CREATE TABLE graph_edges (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       uuid NOT NULL,
    owner_id     uuid NOT NULL,
    visibility   text NOT NULL,
    chunk_id     uuid NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,   -- the evidence
    subject      text NOT NULL,        -- normalised: lower case, no leading "the", single spaces
    subject_type text NOT NULL DEFAULT 'other',
    relation     text NOT NULL CHECK (relation ~ '^[a-z_]{2,40}$'),
    object       text NOT NULL,
    object_type  text NOT NULL DEFAULT 'other',
    created_at   timestamptz NOT NULL DEFAULT now(),
    CHECK (subject <> object)
);
CREATE INDEX graph_edges_subject_idx ON graph_edges (org_id, subject);
CREATE INDEX graph_edges_object_idx ON graph_edges (org_id, object);
CREATE INDEX graph_edges_chunk_idx ON graph_edges (chunk_id);

-- Which chunks were already read for facts (a chunk with no facts must not be re-read every run).
CREATE TABLE graph_extracted (
    chunk_id uuid PRIMARY KEY REFERENCES chunks (id) ON DELETE CASCADE,
    org_id   uuid NOT NULL,
    edges    integer NOT NULL DEFAULT 0
);

ALTER TABLE graph_edges ENABLE ROW LEVEL SECURITY;
CREATE POLICY read_visible ON graph_edges FOR SELECT
    USING (org_id = app_org() AND (
        visibility = 'public'
        OR (app_user() IS NOT NULL AND (visibility = 'org' OR owner_id = app_user()))));
CREATE POLICY write_own ON graph_edges FOR ALL
    USING (org_id = app_org() AND owner_id = app_user())
    WITH CHECK (org_id = app_org() AND owner_id = app_user());
ALTER TABLE graph_extracted ENABLE ROW LEVEL SECURITY;
CREATE POLICY own_org ON graph_extracted FOR ALL
    USING (org_id = app_org() AND app_user() IS NOT NULL)
    WITH CHECK (org_id = app_org() AND app_user() IS NOT NULL);

CREATE FUNCTION purge_graph_on_visibility_change() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    DELETE FROM graph_edges WHERE chunk_id = NEW.id;
    DELETE FROM graph_extracted WHERE chunk_id = NEW.id;   -- so it is read again, under its new level
    RETURN NEW;
END $$;
CREATE TRIGGER chunk_visibility_purges_graph AFTER UPDATE OF visibility ON chunks
    FOR EACH ROW WHEN (OLD.level = 0 AND OLD.visibility IS DISTINCT FROM NEW.visibility)
    EXECUTE FUNCTION purge_graph_on_visibility_change();
