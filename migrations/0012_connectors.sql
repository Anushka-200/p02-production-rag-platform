-- Connectors: keeping documents in step with a source that changes (a wiki, a git repo).
--
-- `connector_items` remembers, for every page a connector has seen, the source's own fingerprint of
-- its content (for git: the blob id). Next sync compares fingerprints WITHOUT reading the files, so
-- only pages that really changed are fetched, parsed and embedded. `connector_runs` is the audit
-- trail of each sync (what was added, changed, removed) for the admin console.

CREATE TABLE connector_items (
    org_id      uuid NOT NULL,
    connector   text NOT NULL,                -- e.g. 'git:gitlab-handbook'
    uri         text NOT NULL,                -- the page's stable identity inside the org
    source_hash text NOT NULL,                -- the source's fingerprint of the content
    visibility  text NOT NULL CHECK (visibility IN ('private', 'org', 'public')),
    document_id uuid REFERENCES documents (id) ON DELETE SET NULL,
    state       text NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'deleted')),
    synced_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (org_id, connector, uri)
);

CREATE TABLE connector_runs (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      uuid NOT NULL,
    connector   text NOT NULL,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    source_ref  text,                         -- e.g. the git commit that was synced
    added       integer NOT NULL DEFAULT 0,
    changed     integer NOT NULL DEFAULT 0,
    unchanged   integer NOT NULL DEFAULT 0,
    deleted     integer NOT NULL DEFAULT 0,
    permission_changes integer NOT NULL DEFAULT 0,
    failed      integer NOT NULL DEFAULT 0,
    aborted     text,                         -- why a run stopped early (mass-delete guard, ...)
    cost_usd    numeric(10, 6) NOT NULL DEFAULT 0
);

ALTER TABLE connector_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE connector_runs ENABLE ROW LEVEL SECURITY;
-- Operational metadata: members of the org may read it (the API requires an admin); the sync runs
-- as the database owner or the worker role acting for the org.
CREATE POLICY org_read ON connector_items FOR SELECT USING (org_id = app_org() AND app_user() IS NOT NULL);
CREATE POLICY org_read ON connector_runs FOR SELECT USING (org_id = app_org() AND app_user() IS NOT NULL);
