# ADR 003: Hybrid fusion and reranker

Status: proposed (2026-09-30)

## Context
Dense misses exact tokens; keyword misses vocabulary mismatch; neither reads query and chunk together.

## Options
- Fusion: RRF (k, weights) vs weighted score sum.
- Keyword: Postgres ts_rank / ts_rank_cd (not BM25) vs a true-BM25 extension; query as AND vs OR of words (OR expected to win on recall).
- Reranker: local bge-reranker vs Cohere hosted vs none.

## Decision
RRF; local bge-reranker; Cohere benchmarked on the same eval. Keep reranker only if Recall@5 gain justifies latency.

## Early evidence (informal, 2026-10-02: benchmarks B-008)
Equal-weight RRF of dense + OR-keyword (ts_rank_cd) scored 0.64 gold-in-top-5 vs 0.79 for dense alone on 61 unreviewed questions. Cause:
ts_rank_cd has no IDF, so common words add noise. Dropping words in >2% of chunks + keyword weight 0.5 restored 0.79. Open: true BM25, tuned
weights, and exact-term eval questions (the current set favours dense by construction).

## Consequences
TBD after experiment 3. Risk: a large cross-encoder on CPU may break the 2.5 s budget; measured in a Phase 2 spike.

## Update 2026-10-02 (B-014)
Reranking only the top 10 fused candidates keeps quality (R@5 0.83, MRR 0.69) and halves rerank latency (about 1.9 s to 0.96 s); length-sorted batching is on. Estimated first token 1.94 s against the 2.5 s target.
