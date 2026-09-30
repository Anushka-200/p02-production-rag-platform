# ADR 001: Vector store

Status: proposed (2026-09-30)

## Context
Need dense + keyword search and tenant isolation in one system, self-hostable locally and on RDS.

## Options
- pgvector (Postgres): one DB for vectors, full-text and RLS; HNSW; limits at very large scale.
- Qdrant: fast, rich filtering; separate service, no RLS, keyword search weaker.
- Pinecone: managed, no ops; vendor cost, isolation by namespace only.

## Decision
Leading candidate: pgvector. Confirm with Phase 1 measurements (filtered HNSW recall, iterative scans).

## Consequences
TBD after benchmark. Queue choice (RQ over Celery for simplicity) is recorded here too.
