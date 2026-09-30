-- Core tables. Row-level security is added in Phase 3, but the columns it needs
-- (org_id, owner_id, visibility) exist from the start so we never restructure.
-- The embedding column and HNSW index are added after the embedding model is chosen (task 1.8).

CREATE TABLE orgs (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name       text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id     uuid NOT NULL REFERENCES orgs (id),
    email      text NOT NULL,
    role       text NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member')),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (org_id, email),
    -- Lets other tables prove "this user belongs to this org" with a composite foreign key.
    UNIQUE (id, org_id)
);

CREATE TABLE documents (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id            uuid NOT NULL,
    owner_id          uuid NOT NULL,
    visibility        text NOT NULL DEFAULT 'private' CHECK (visibility IN ('private', 'org')),
    title             text NOT NULL,
    source_type       text NOT NULL CHECK (source_type IN ('pdf', 'docx', 'md', 'txt', 'html')),
    source_uri        text NOT NULL,          -- filename or wiki path; identity of a document
    content_hash      text NOT NULL,          -- same hash on re-upload = nothing to do
    active_version    integer NOT NULL DEFAULT 0,  -- which chunk version search should use
    status            text NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    error             text,
    source_updated_at timestamptz,            -- "last updated" shown in citations
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (owner_id, org_id) REFERENCES users (id, org_id),
    UNIQUE (org_id, owner_id, source_uri),
    UNIQUE (id, org_id),
    UNIQUE (id, org_id, owner_id, visibility)  -- target of chunks' composite foreign key
);

CREATE TABLE chunks (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id        uuid NOT NULL,
    owner_id      uuid NOT NULL,   -- copied from the document so RLS needs no join
    visibility    text NOT NULL,   -- copied too; the foreign key below keeps copies consistent
    document_id   uuid NOT NULL,
    version       integer NOT NULL,
    level         smallint NOT NULL DEFAULT 0 CHECK (level IN (0, 1, 2)),  -- 0 child, 1 parent, 2 summary
    parent_id     uuid REFERENCES chunks (id) ON DELETE SET NULL,
    chunk_index   integer NOT NULL,
    page_start    integer,
    page_end      integer,
    section_path  text NOT NULL DEFAULT '',
    text          text NOT NULL,
    context_blurb text NOT NULL DEFAULT '',   -- LLM-written situating sentence (contextual retrieval)
    content_hash  text NOT NULL,
    -- Keyword index: section path counts most (A), blurb + text least (D). Always in sync with the text.
    tsv           tsvector GENERATED ALWAYS AS (
                      setweight(to_tsvector('english', section_path), 'A') ||
                      setweight(to_tsvector('english', context_blurb || ' ' || text), 'D')
                  ) STORED,
    created_at    timestamptz NOT NULL DEFAULT now(),
    -- Changing a document's visibility flows to its chunks, so permissions can never drift.
    FOREIGN KEY (document_id, org_id, owner_id, visibility)
        REFERENCES documents (id, org_id, owner_id, visibility)
        ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (document_id, version, level, chunk_index),
    CHECK (page_start IS NULL OR page_end IS NULL OR page_end >= page_start)
);

CREATE INDEX chunks_tsv_idx    ON chunks USING gin (tsv);
CREATE INDEX chunks_doc_idx    ON chunks (org_id, document_id, version);
CREATE INDEX chunks_parent_idx ON chunks (parent_id);
CREATE INDEX documents_org_idx ON documents (org_id, status);

CREATE TABLE ingestion_jobs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          uuid NOT NULL,
    document_id     uuid NOT NULL,
    idempotency_key text NOT NULL,   -- org + content hash: the same upload twice is one job
    status          text NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    attempts        integer NOT NULL DEFAULT 0,
    error           text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    started_at      timestamptz,
    finished_at     timestamptz,
    FOREIGN KEY (document_id, org_id) REFERENCES documents (id, org_id) ON DELETE CASCADE,
    UNIQUE (org_id, idempotency_key)
);
