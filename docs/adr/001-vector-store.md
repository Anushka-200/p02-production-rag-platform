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

## Measured (benchmarks B-005..B-007, 2026-10-02)
At 7,156 vectors in a two-tenant shared index, a filtered dense query takes ~1-2 ms (HNSW index used). Without care, the tenant filter
returned only 35 of 50 requested rows on average (min 8). `SET LOCAL hnsw.iterative_scan = relaxed_order` returns all 50 with
results identical to exact search and ~1.3 ms p50. Every filtered vector query must set it (to be enforced in the retrieval layer).

## Consequences
TBD after benchmark. Queue choice (RQ over Celery for simplicity) is recorded here too.
