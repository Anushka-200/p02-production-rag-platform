#!/bin/bash
# Runs once, on first start of an empty data volume. Creates the extension and three login roles:
#   rag_owner  - owns the tables; used only for migrations
#   rag_app    - the API; can read/write rows, cannot change table structure, cannot bypass RLS
#   rag_worker - the ingestion worker; same limits as rag_app
# NOBYPASSRLS + not owner means row-level security policies always apply to rag_app/rag_worker.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username postgres --dbname rag <<SQL
CREATE EXTENSION IF NOT EXISTS vector;

CREATE ROLE rag_owner  LOGIN PASSWORD '${RAG_OWNER_PASSWORD}'  NOSUPERUSER NOBYPASSRLS;
CREATE ROLE rag_app    LOGIN PASSWORD '${RAG_APP_PASSWORD}'    NOSUPERUSER NOBYPASSRLS;
CREATE ROLE rag_worker LOGIN PASSWORD '${RAG_WORKER_PASSWORD}' NOSUPERUSER NOBYPASSRLS;

ALTER DATABASE rag OWNER TO rag_owner;
ALTER SCHEMA public OWNER TO rag_owner;
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO rag_app, rag_worker;

-- Tables that rag_owner creates later automatically become readable/writable by the app roles.
ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag_app, rag_worker;
ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO rag_app, rag_worker;
SQL
