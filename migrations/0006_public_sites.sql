-- Public documents and anonymous visitors (widget, WhatsApp).
--
-- A visitor on a customer's website is not an org member. They get an ANONYMOUS scope: the org
-- (from the site key) but NO user. Anonymous sees only documents marked 'public'; never
-- org-internal ('org') or private ones. Members see public documents too.

ALTER TABLE documents DROP CONSTRAINT documents_visibility_check;
ALTER TABLE documents ADD CONSTRAINT documents_visibility_check
    CHECK (visibility IN ('private', 'org', 'public'));
ALTER TABLE ingestion_jobs DROP CONSTRAINT ingestion_jobs_visibility_check;
ALTER TABLE ingestion_jobs ADD CONSTRAINT ingestion_jobs_visibility_check
    CHECK (visibility IN ('private', 'org', 'public'));

-- Old rule: org docs were visible to anyone with the org set. With anonymous visitors that would
-- leak, so 'org' now also requires a real user (app_user() IS NOT NULL). 'public' needs only the org.
DROP POLICY tenant_isolation ON documents;
DROP POLICY tenant_isolation ON chunks;
CREATE POLICY tenant_isolation ON documents
    USING (org_id = app_org() AND (
        visibility = 'public'
        OR (app_user() IS NOT NULL AND (visibility = 'org' OR owner_id = app_user()))))
    WITH CHECK (org_id = app_org() AND owner_id = app_user());
CREATE POLICY tenant_isolation ON chunks
    USING (org_id = app_org() AND (
        visibility = 'public'
        OR (app_user() IS NOT NULL AND (visibility = 'org' OR owner_id = app_user()))))
    WITH CHECK (org_id = app_org() AND owner_id = app_user());
-- WITH CHECK is unchanged: nobody can write without a user, so anonymous visitors can never
-- create or change documents, chunks, jobs, conversations or memories.

-- A "site" is one place that embeds the widget. The site key is PUBLIC (it sits in the page's HTML);
-- it only says which org's public documents to use. Real protection = the origin allow-list,
-- the enabled switch, rate limits and budgets, and the fact that anonymous scope sees only public data.
CREATE TABLE sites (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          uuid NOT NULL REFERENCES orgs (id),
    name            text NOT NULL,
    site_key        text NOT NULL UNIQUE,
    allowed_origins text[] NOT NULL DEFAULT '{}',   -- e.g. {https://www.acme.com}
    enabled         boolean NOT NULL DEFAULT true,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Looked up before any tenant is known (like users): no RLS, read-only for the app roles.
DO $$
DECLARE r text;
BEGIN
    FOREACH r IN ARRAY ARRAY['rag_app', 'rag_worker'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
            EXECUTE format('REVOKE ALL ON sites FROM %I', r);
            EXECUTE format('GRANT SELECT ON sites TO %I', r);
        END IF;
    END LOOP;
END $$;
