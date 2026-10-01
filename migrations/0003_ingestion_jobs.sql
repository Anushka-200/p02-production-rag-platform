-- Ingestion jobs now hold everything the worker needs, so Redis only has to carry the job id.
-- (If Redis is flushed, no job information is lost.) The table was unused before this migration.
DELETE FROM ingestion_jobs;

-- A first-time upload has no document row yet; the worker creates it when it saves version 1.
ALTER TABLE ingestion_jobs ALTER COLUMN document_id DROP NOT NULL;

ALTER TABLE ingestion_jobs
    ADD COLUMN owner_id     uuid NOT NULL,
    ADD COLUMN source_uri   text NOT NULL,    -- the file's name; identity of the document
    ADD COLUMN source_type  text NOT NULL CHECK (source_type IN ('pdf', 'docx', 'md', 'txt', 'html')),
    ADD COLUMN visibility   text NOT NULL CHECK (visibility IN ('private', 'org')),
    ADD COLUMN content_hash text NOT NULL,    -- sha256 of the uploaded bytes
    ADD COLUMN raw_path     text NOT NULL,    -- where the original file is stored
    ADD COLUMN stage        text NOT NULL DEFAULT 'queued',  -- queued|parse|chunk|blurb|embed|save
    ADD COLUMN result       jsonb,            -- what the run did (counts, cost, action)
    ADD CONSTRAINT ingestion_jobs_owner_fk
        FOREIGN KEY (owner_id, org_id) REFERENCES users (id, org_id);
