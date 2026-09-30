# ADR 003: Hybrid fusion and reranker

Status: proposed (2026-09-30)

## Context
Dense misses exact tokens; keyword misses vocabulary mismatch; neither reads query and chunk together.

## Options
- Fusion: RRF (k, weights) vs weighted score sum.
- Keyword: Postgres ts_rank vs true BM25 extension.
- Reranker: local bge-reranker vs Cohere hosted vs none.

## Decision
RRF; local bge-reranker; Cohere benchmarked on the same eval. Keep reranker only if Recall@5 gain justifies latency.

## Consequences
TBD after experiment 3.
