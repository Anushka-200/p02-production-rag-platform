-- Row-level security: the DATABASE refuses to show a tenant another tenant's rows.
-- Every request/job tells Postgres who it is with two settings (taken from the verified login,
-- never from the request body):  app.org_id, app.user_id.
--
-- Fail closed: if a setting is missing, current_setting(..., true) returns NULL, and
-- "org_id = NULL" is never true, so no rows match. Nothing falls back to "show all".
--
-- RLS does not apply to the table owner (rag_owner, used only for migrations) or superusers.
-- The API/worker roles are neither (NOBYPASSRLS, not owners); the API checks this at startup.

CREATE FUNCTION app_org() RETURNS uuid LANGUAGE sql STABLE AS
    $$ SELECT nullif(current_setting('app.org_id', true), '')::uuid $$;
CREATE FUNCTION app_user() RETURNS uuid LANGUAGE sql STABLE AS
    $$ SELECT nullif(current_setting('app.user_id', true), '')::uuid $$;

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingestion_jobs ENABLE ROW LEVEL SECURITY;

-- A row is visible when it is in my org AND (shared with the org OR mine).
-- WITH CHECK: I may only write rows into my own org, owned by me.
CREATE POLICY tenant_isolation ON documents
    USING (org_id = app_org() AND (visibility = 'org' OR owner_id = app_user()))
    WITH CHECK (org_id = app_org() AND owner_id = app_user());

CREATE POLICY tenant_isolation ON chunks
    USING (org_id = app_org() AND (visibility = 'org' OR owner_id = app_user()))
    WITH CHECK (org_id = app_org() AND owner_id = app_user());

-- Jobs are private to their owner, like the document they create.
CREATE POLICY tenant_isolation ON ingestion_jobs
    USING (org_id = app_org() AND owner_id = app_user())
    WITH CHECK (org_id = app_org() AND owner_id = app_user());

-- Login: password hash (scrypt) for JWT login. Nullable: seeded/bulk users cannot log in until set.
ALTER TABLE users ADD COLUMN password_hash text;

-- orgs/users are looked up BEFORE the tenant is known (login), so they have no RLS; instead the
-- app and worker roles may only read them. They hold no document content.
-- Login must read the password hash, which the app role may not see. A SECURITY DEFINER function
-- runs with the owner's rights and returns only the one row for the email asked about.
CREATE FUNCTION login_lookup(p_email text)
    RETURNS TABLE (id uuid, org_id uuid, role text, password_hash text)
    LANGUAGE sql STABLE SECURITY DEFINER AS
    $$ SELECT u.id, u.org_id, u.role, u.password_hash FROM users u WHERE lower(u.email) = lower(p_email) $$;
REVOKE ALL ON FUNCTION login_lookup(text) FROM PUBLIC;

DO $$
DECLARE r text;
BEGIN
    FOREACH r IN ARRAY ARRAY['rag_app', 'rag_worker'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
            EXECUTE format('REVOKE ALL ON orgs, users FROM %I', r);
            EXECUTE format('GRANT SELECT ON orgs TO %I', r);
            EXECUTE format('GRANT SELECT (id, org_id, email, role, created_at) ON users TO %I', r);
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') THEN
        GRANT EXECUTE ON FUNCTION login_lookup(text) TO rag_app;
    END IF;
END $$;
