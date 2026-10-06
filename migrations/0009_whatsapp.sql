-- WhatsApp channel (via Twilio).
--
-- A phone number is personal data. We never store it to RECOGNISE a returning user: only a keyed
-- hash (HMAC with a server secret) is kept, so the table cannot be turned back into phone numbers
-- without the secret. The number itself is stored only if the user explicitly consents to be
-- contacted by a person (it then lives in `leads`, where staff can see it).
--
-- These routing tables are looked up BEFORE the tenant is known (the message arrives first, the
-- company is found second), so, like `sites` and `users`, they carry no row-level security; they
-- contain only a hash and ids.

-- Which WhatsApp entry point belongs to which site: a join keyword (shared sandbox number) and/or
-- a dedicated number.
CREATE TABLE whatsapp_channels (
    id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id    uuid NOT NULL REFERENCES orgs (id),
    site_id   uuid NOT NULL REFERENCES sites (id),
    keyword   text NOT NULL UNIQUE CHECK (keyword = lower(keyword) AND keyword ~ '^[a-z0-9-]{2,40}$'),
    to_number text UNIQUE,
    enabled   boolean NOT NULL DEFAULT true
);

-- One row per phone (hashed): which channel it talks to, its current widget session, its state.
CREATE TABLE whatsapp_contacts (
    phone_hash        text PRIMARY KEY,
    channel_id        uuid NOT NULL REFERENCES whatsapp_channels (id) ON DELETE CASCADE,
    org_id            uuid NOT NULL,
    site_id           uuid NOT NULL,
    widget_session_id uuid,                         -- the conversation (widget_sessions, channel 'whatsapp')
    state             text NOT NULL DEFAULT 'chat' CHECK (state IN ('chat', 'awaiting_consent')),
    created_at        timestamptz NOT NULL DEFAULT now(),
    last_seen_at      timestamptz NOT NULL DEFAULT now()
);

-- Twilio redelivers a message when it does not get an answer in time: remember what we handled.
CREATE TABLE whatsapp_seen (
    message_sid text PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- A WhatsApp session is marked as such, so staff can tell where a handoff came from.
ALTER TABLE widget_sessions ADD COLUMN channel text NOT NULL DEFAULT 'web'
    CHECK (channel IN ('web', 'whatsapp'));

-- A WhatsApp lead has a phone number but no email: one of the two is enough.
ALTER TABLE leads ALTER COLUMN email DROP NOT NULL;
ALTER TABLE leads DROP CONSTRAINT leads_email_check;
ALTER TABLE leads ADD CONSTRAINT leads_email_check
    CHECK (email IS NULL OR length(email) BETWEEN 3 AND 254);
ALTER TABLE leads ADD CONSTRAINT leads_has_contact CHECK (email IS NOT NULL OR phone IS NOT NULL);

-- The API may read channels (set up by the owner) but not change them; contacts and the seen-list
-- it writes itself.
DO $$
DECLARE r text;
BEGIN
    FOREACH r IN ARRAY ARRAY['rag_app', 'rag_worker'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
            EXECUTE format('REVOKE ALL ON whatsapp_channels FROM %I', r);
            EXECUTE format('GRANT SELECT ON whatsapp_channels TO %I', r);
        END IF;
    END LOOP;
END $$;
