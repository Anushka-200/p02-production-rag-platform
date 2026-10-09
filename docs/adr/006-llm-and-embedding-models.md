# ADR 006: LLM and embedding models

Status: accepted (2026-10-08) with one open decision: arctic-embed-m beats the current bge-small (B-033), switch pending

## Context
Cheap default tier via LiteLLM; Anthropic has no embedding model.

## Options
Haiku 4.5 answers/rewrites/blurbs/memory, Sonnet 5.5 judge; local open embedding vs one hosted embedding; all-OpenAI.

## Decision
Haiku 4.5 + Sonnet 5.5 judge; local embedding as default, one hosted model for the quality-vs-cost experiment. Verify current model names and prices when finalising.

## Consequences
TBD after experiment 5. Constraint: indexed `vector` columns allow at most 2,000 dimensions (`halfvec` 4,000).

## Embedding model: checklist before experiment 5 (task 6.6)
1. Data domain: handbooks = general business/HR English; IRS/OPM PDFs = tax/benefits guidance (most specialised). Test the PDF questions separately.
2. Measure on our eval set (Recall@5, speed, cost); do not guess.
3. Shortlist: bge-small (default now), bge-base (768-d), nomic-embed-text-v1.5 (768-d, 8,192-token input), one hosted model (OpenAI, Voyage or Cohere), maybe a finance/law variant for PDFs.
4. Check the MTEB leaderboard first; verify licences from primary files (bge-small/base MIT, nomic Apache-2.0 per fastembed metadata).
5. A model switch = new migration (vector size) + re-embed (~16 min for the Tenant A slice).
6. Query and chunks must use the same model per index. Two models only with separate columns/indexes and fused results, and only if the PDF gain is large (about +10 pts Recall@5).

## Embedding model: measured (2026-10-08, B-033)
Four local models compared on all of Tenant A with the real eval metrics. snowflake-arctic-embed-m (768-d, Apache-2.0)
scored R@5 0.81 / MRR 0.72 against 0.75 / 0.62 for the current bge-small-en-v1.5 (384-d), with no PDF/DOCX advantage for any
model. bge-base and nomic did not beat it. Recommendation: switch to arctic-m after confirming the gain survives the reranker
(re-run retrieval with `hybrid+rerank`) and accepting the cost (vector(768) migration, full re-embed, 1.8x slower embedding).
bge-small stays the default until that is done.

### Update 2026-10-09 (B-034)
With the production reranker, bge-small and arctic-embed-m both reach R@5 0.81 and MRR 0.66, so the gap measured in B-033 disappears.
**Decision: keep bge-small-en-v1.5.** No migration and no re-embed; revisit only if the reranker is replaced or removed.
