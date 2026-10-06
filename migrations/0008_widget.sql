-- Widget visitors: sessions, messages, leads, feedback, per-site usage.
--
-- Visitors have no user, so the user-private chat tables cannot hold their chats. Their data is
-- keyed by SITE instead: a third setting, app.site_id, says which embedded site the request came
-- from, and the policies match on it. A visitor of one site can never touch another site's rows.
-- Members of the org may READ these rows (support staff follow up on leads and handoffs); the API
-- limits that to org admins.

CREATE FUNCTION app_site() RETURNS uuid LANGUAGE sql STABLE AS
    $$ SELECT nullif(current_setting('app.site_id', true), '')::uuid $$;

ALTER TABLE sites
    ADD COLUMN daily_budget_usd numeric(8, 2) NOT NULL DEFAULT 1.00,   -- stop answering when spent
    ADD COLUMN rate_limit_per_minute integer NOT NULL DEFAULT 20;      -- per visitor address

CREATE TABLE widget_sessions (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),  -- random: acts as the visitor's session key
    org_id               uuid NOT NULL,
    site_id              uuid NOT NULL REFERENCES sites (id),
    created_at           timestamptz NOT NULL DEFAULT now(),
    last_seen_at         timestamptz NOT NULL DEFAULT now(),
    handoff_requested    boolean NOT NULL DEFAULT false,              -- "I want to talk to a person"
    handoff_reason       text,
    handoff_at           timestamptz,
    handoff_resolved_at  timestamptz,
    UNIQUE (id, org_id, site_id)
);
CREATE INDEX widget_sessions_handoff_idx ON widget_sessions (org_id, handoff_requested, handoff_at DESC);

CREATE TABLE widget_messages (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id uuid NOT NULL,
    org_id     uuid NOT NULL,
    site_id    uuid NOT NULL,
    role       text NOT NULL CHECK (role IN ('user', 'assistant')),
    content    text NOT NULL,
    citations  jsonb NOT NULL DEFAULT '[]',
    refused    boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (session_id, org_id, site_id)
        REFERENCES widget_sessions (id, org_id, site_id) ON DELETE CASCADE
);
CREATE INDEX widget_messages_session_idx ON widget_messages (session_id, created_at);

-- A lead is contact data the visitor GAVE us on purpose; consent must be recorded as true.
CREATE TABLE leads (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id     uuid NOT NULL,
    site_id    uuid NOT NULL,
    session_id uuid NOT NULL,
    name       text NOT NULL CHECK (length(name) BETWEEN 1 AND 120),
    email      text NOT NULL CHECK (length(email) BETWEEN 3 AND 254),
    phone      text CHECK (phone IS NULL OR length(phone) <= 40),
    message    text CHECK (message IS NULL OR length(message) <= 1000),
    consent    boolean NOT NULL CHECK (consent),
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (session_id, org_id, site_id)
        REFERENCES widget_sessions (id, org_id, site_id) ON DELETE CASCADE
);
CREATE INDEX leads_org_idx ON leads (org_id, created_at DESC);

-- Thumbs up/down on an answer: the first step of the feedback loop into the eval set.
CREATE TABLE feedback (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id     uuid NOT NULL,
    site_id    uuid,
    session_id uuid,
    message_id uuid,
    rating     smallint NOT NULL CHECK (rating IN (-1, 1)),
    comment    text CHECK (comment IS NULL OR length(comment) <= 500),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX feedback_org_idx ON feedback (org_id, created_at DESC);

-- What a site has spent today: the budget cap reads this, every answer adds to it.
CREATE TABLE site_usage (
    site_id  uuid NOT NULL REFERENCES sites (id),
    org_id   uuid NOT NULL,
    day      date NOT NULL DEFAULT current_date,
    requests integer NOT NULL DEFAULT 0,
    cost_usd numeric(10, 6) NOT NULL DEFAULT 0,
    PRIMARY KEY (site_id, day)
);

DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['widget_sessions', 'widget_messages', 'leads', 'feedback', 'site_usage'] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        -- the visitor's side: only rows of THIS site (and this org)
        EXECUTE format($p$CREATE POLICY site_access ON %I FOR ALL
            USING (org_id = app_org() AND site_id = app_site())
            WITH CHECK (org_id = app_org() AND site_id = app_site())$p$, t);
        -- the staff side: any logged-in member of the org may read (the API requires an admin)
        EXECUTE format($p$CREATE POLICY member_read ON %I FOR SELECT
            USING (org_id = app_org() AND app_user() IS NOT NULL)$p$, t);
    END LOOP;
END $$;
-- Staff may mark a handoff as resolved.
CREATE POLICY member_resolve ON widget_sessions FOR UPDATE
    USING (org_id = app_org() AND app_user() IS NOT NULL)
    WITH CHECK (org_id = app_org() AND app_user() IS NOT NULL);
