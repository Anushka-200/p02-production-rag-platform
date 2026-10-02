-- Chat history and long-term memory. Both are PRIVATE TO THE USER (not just to the org): an org
-- admin cannot read a colleague's conversations or memories, and RLS enforces it.
--
-- Three separate stores on purpose:
--   chunks / documents  what the COMPANY knows (shared, cited)
--   conversations/messages   what was SAID in a chat (transcript, windowed into prompts later)
--   memories            what we LEARNED about the user (short facts, viewable and deletable)

CREATE TABLE conversations (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id     uuid NOT NULL,
    user_id    uuid NOT NULL,
    title      text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (user_id, org_id) REFERENCES users (id, org_id) ON DELETE CASCADE,
    UNIQUE (id, org_id, user_id)  -- target of messages' composite foreign key
);
CREATE INDEX conversations_user_idx ON conversations (org_id, user_id, updated_at DESC);

CREATE TABLE messages (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid NOT NULL,
    org_id          uuid NOT NULL,
    user_id         uuid NOT NULL,
    role            text NOT NULL CHECK (role IN ('user', 'assistant')),
    content         text NOT NULL,
    citations       jsonb NOT NULL DEFAULT '[]',   -- [{id, document, section, pages}] for assistant turns
    refused         boolean NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now(),
    -- A message can only belong to a conversation of the SAME org and user.
    FOREIGN KEY (conversation_id, org_id, user_id)
        REFERENCES conversations (id, org_id, user_id) ON DELETE CASCADE
);
CREATE INDEX messages_conversation_idx ON messages (conversation_id, created_at);

CREATE TABLE memories (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id            uuid NOT NULL,
    user_id           uuid NOT NULL,
    fact              text NOT NULL CHECK (length(fact) BETWEEN 3 AND 300),
    embedding         vector(384) NOT NULL,
    embedding_model   text NOT NULL,
    source_message_id uuid REFERENCES messages (id) ON DELETE SET NULL,  -- where we learned it
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (user_id, org_id) REFERENCES users (id, org_id) ON DELETE CASCADE
);
CREATE INDEX memories_user_idx ON memories (org_id, user_id);
-- No HNSW index: one user has at most a few hundred memories, so an exact scan is faster and exact.

ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;

CREATE POLICY private_to_user ON conversations
    USING (org_id = app_org() AND user_id = app_user())
    WITH CHECK (org_id = app_org() AND user_id = app_user());
CREATE POLICY private_to_user ON messages
    USING (org_id = app_org() AND user_id = app_user())
    WITH CHECK (org_id = app_org() AND user_id = app_user());
CREATE POLICY private_to_user ON memories
    USING (org_id = app_org() AND user_id = app_user())
    WITH CHECK (org_id = app_org() AND user_id = app_user());
