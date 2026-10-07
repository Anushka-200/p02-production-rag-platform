-- RAPTOR-style summary tree: summaries of groups of chunks, searchable like chunks.
--
-- A summary is stored as a CHUNK (level 2 = summary of chunks, level 3 = summary of summaries) of a
-- synthetic document per visibility group, so it reuses the chunk table, the HNSW index, row-level
-- security and versioning. `summary_members` records which chunks each summary was written from.
--
-- PRIVACY RULE (the dangerous part of summaries): a summary is a copy of its sources' content. It
-- must disappear the moment ANY source chunk is deleted, replaced by a re-index, or loses its
-- visibility. Triggers below do that for every code path, so no application code can forget it.

ALTER TABLE documents DROP CONSTRAINT documents_source_type_check;
ALTER TABLE documents ADD CONSTRAINT documents_source_type_check
    CHECK (source_type IN ('pdf', 'docx', 'md', 'txt', 'html', 'summary'));
ALTER TABLE chunks DROP CONSTRAINT chunks_level_check;
ALTER TABLE chunks ADD CONSTRAINT chunks_level_check CHECK (level IN (0, 1, 2, 3));

CREATE TABLE summary_members (
    summary_id uuid NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    member_id  uuid NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    org_id     uuid NOT NULL,
    owner_id   uuid NOT NULL,   -- copied from the summary so row-level security needs no join
    visibility text NOT NULL,
    PRIMARY KEY (summary_id, member_id)
);
CREATE INDEX summary_members_member_idx ON summary_members (member_id);

ALTER TABLE summary_members ENABLE ROW LEVEL SECURITY;
CREATE POLICY read_visible ON summary_members FOR SELECT
    USING (org_id = app_org() AND (
        visibility = 'public'
        OR (app_user() IS NOT NULL AND (visibility = 'org' OR owner_id = app_user()))));
CREATE POLICY write_own ON summary_members FOR ALL
    USING (org_id = app_org() AND owner_id = app_user())
    WITH CHECK (org_id = app_org() AND owner_id = app_user());

-- Delete every summary that (directly or through other summaries) contains this chunk. SECURITY
-- DEFINER because a member re-indexing THEIR document must be able to remove a summary owned by
-- someone else: it only removes derived data, which can always be rebuilt.
CREATE FUNCTION purge_summaries_containing(p_chunk uuid) RETURNS integer
    LANGUAGE sql SECURITY DEFINER AS $$
    WITH RECURSIVE hit(id) AS (
        SELECT m.summary_id FROM summary_members m WHERE m.member_id = p_chunk
        UNION
        SELECT m.summary_id FROM summary_members m JOIN hit h ON m.member_id = h.id
    ), gone AS (
        DELETE FROM chunks WHERE id IN (SELECT id FROM hit) RETURNING 1
    )
    SELECT count(*)::integer FROM gone
$$;
REVOKE ALL ON FUNCTION purge_summaries_containing(uuid) FROM PUBLIC;

CREATE FUNCTION purge_on_chunk_delete() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    PERFORM purge_summaries_containing(OLD.id);
    RETURN OLD;
END $$;
CREATE TRIGGER chunk_deleted_purges_summaries BEFORE DELETE ON chunks
    FOR EACH ROW WHEN (OLD.level = 0) EXECUTE FUNCTION purge_on_chunk_delete();

CREATE FUNCTION purge_on_visibility_change() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    PERFORM purge_summaries_containing(NEW.id);
    RETURN NEW;
END $$;
CREATE TRIGGER chunk_visibility_purges_summaries AFTER UPDATE OF visibility ON chunks
    FOR EACH ROW WHEN (OLD.level = 0 AND OLD.visibility IS DISTINCT FROM NEW.visibility)
    EXECUTE FUNCTION purge_on_visibility_change();
