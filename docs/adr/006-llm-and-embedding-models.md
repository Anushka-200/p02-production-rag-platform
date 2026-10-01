# ADR 006: LLM and embedding models

Status: proposed (2026-09-30)

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
