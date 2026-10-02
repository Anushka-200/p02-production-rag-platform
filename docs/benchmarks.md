# Benchmarks (every number here was measured; date, setup and the command that produced it are given)

Machine: Intel i7-8750H (6 cores / 12 threads), 8 GB RAM, WSL2, no GPU. Postgres 16 + pgvector 0.8.6 in Docker.
Embedding: BAAI/bge-small-en-v1.5 (384 dimensions) through fastembed on CPU.

## B-001 Embedding batch size (2026-10-01, task 1.8)
Same 96 real chunks, local model. Default batch of 256 swapped this 8 GB machine to disk (~2 s per chunk, ~138 min for the
Tenant A slice). Smaller batches fixed it:

| batch size | chunks/s | projected time for 4.1k chunks |
|---|---|---|
| 8 | 4.2 | 16.4 min |
| **16 (chosen)** | **4.3** | **15.8 min** |
| 32 | 3.8 | 18.0 min |
| 64 | 3.8 | 18.0 min |

## B-002 Token counter for chunking (2026-10-01, task 1.8)
Tenant A slice, chunker B, 400-token target. The cheap "chars / 4" estimate vs the embedding model's real tokenizer:

| counter used while chunking | child chunks | over 400 real tokens | over 512 (truncated by the model) |
|---|---|---|---|
| chars / 4 | 3,404 | 238 | **29** |
| real tokenizer (chosen) | 3,301 | 0 | 0 |

## B-003 Chunker A vs B on the Tenant A slice (2026-10-01, task 1.7)
| | children | parents | median tokens | under 50 tokens |
|---|---|---|---|---|
| A recursive (400/50) | 4,135 | 0 | 171 | 10.4% |
| B structured (400/50, parent 1200) | 3,404 | 695 | 279 | 1.9% |
| B + merge across branches | 3,132 | 740 | 306 | 0.9% |
(Retrieval quality for A vs B is measured in Phase 6, experiment 1.)

## B-004 Prompt caching for contextual blurbs (2026-10-01, task 1.9)
Haiku 4.5, one 8,566-token document read 3 times: call 1 $0.01075 (cache write, 1.25x), calls 2-3 $0.0009 (cache read, 0.1x).
Live demo on a real 8,971-token page, 39 chunks: 244,847 tokens read from the cache, $0.082 vs $0.11 projected.
Projected blurb cost for Tenant A (3,301 chunks): ~$7.45. PostHog (3,606 chunks): ~$8.85 (not generated).

## B-005 Bulk ingest, full Tenant A slice + all of PostHog (2026-10-02, task 1.16)
601 files through the real queue, worker, embedding model and Postgres, blurbs off (no API spend). 0 failed jobs.
About 55 minutes of active time (the laptop slept for ~13 h in the middle; the job resumed on its own), roughly 2 chunks/s
end to end (embedding alone ran at ~4 chunks/s; the rest is parsing, chunking and database writes).

| tenant | documents | child chunks | parent chunks | vectors |
|---|---|---|---|---|
| A (GitLab slice, 3 PDFs, 1 DOCX) | 217 | 3,550 | 712 | 3,550 |
| B (PostHog handbook) | 384 | 3,606 | 773 | 3,606 |

Storage: chunks table with indexes 58 MB; HNSW index 14 MB; GIN (keyword) index 5.7 MB.

## B-006 Query latency at 7,156 vectors (2026-10-02, task 1.16)
60 real eval questions, dense search filtered to one tenant, LIMIT 50:
question embedding p50 7.3 ms / p95 10.2 ms; SQL search p50 1.0-1.5 ms / p95 ~2 ms. The planner uses the HNSW index.

## B-007 Filtered HNSW search: the shared-index problem (2026-10-02, task 1.16)
A tenant filter is applied AFTER the index finds its ~40 nearest neighbours across ALL tenants. Same 60 questions, one tenant:

| setting | rows returned of 50 (min / avg) | SQL p50 / p95 | top-10 overlap with exact search |
|---|---|---|---|
| default (ef_search 40) | **8 / 35.2** | 1.0 / 1.3 ms | 0.99 |
| ef_search 100 | 20 / 48.1 | 1.4 / 1.7 ms | 1.00 |
| **hnsw.iterative_scan = relaxed_order** | **50 / 50.0** | 1.3 / 2.8 ms | **1.00** |
| hnsw.iterative_scan = strict_order | 50 / 50.0 | 1.3 / 2.5 ms | 1.00 |

Decision: every filtered vector query runs with `SET LOCAL hnsw.iterative_scan = relaxed_order` (ADR 001, ADR 005).

## B-008 INFORMAL preview: dense vs keyword vs hybrid (2026-10-02, tasks 2.2-2.3)
**Caveats first:** 61 single-fact questions, drafted by an LLM and NOT human-reviewed; gold chunk in the top 5; one corpus. A preview to
steer work, not a result. The official metrics come from tasks 2.9-2.11.

| variant | gold in top 5 |
|---|---|
| dense alone | 0.79 |
| keyword alone (OR of all question words, ts_rank_cd) | 0.28 |
| hybrid RRF, equal weights | **0.64 (worse than dense alone)** |
| keyword with words in >2% of chunks dropped | 0.39 |
| hybrid, those words dropped, keyword weight 0.5 | 0.79 (equal to dense alone) |

Reading: (1) `ts_rank_cd` has no inverse document frequency, so common words ("how", "many", "get") dominate the OR query and add noise;
dropping them helps. (2) A noisy second list hurts RRF unless it is down-weighted. (3) The eval questions were drafted to AVOID copying the
passage's words, which favours dense search; exact-term questions (codes, names, numbers), where keyword search shines, are under-represented.
Next: IDF-aware keyword query or true BM25 (ADR 003 experiment 3), tune the RRF weights, add ~10 exact-term questions to the eval set.
Stage timings on the real index: embed ~8 ms, dense ~4 ms, keyword ~66 ms (a full scan of the text index for common words), fuse 0.2 ms.

### B-008b Sensitivity sweep (same informal setup; per-tenant document-frequency cutoff x keyword weight; dense alone 0.79)
| cutoff | keyword alone | hybrid w=0.3 | w=0.5 | w=1.0 |
|---|---|---|---|---|
| 2% | 0.31 | **0.79** | 0.74 | 0.69 |
| 5% | 0.36 | 0.75 | 0.74 | 0.69 |
| 10% | 0.38 | 0.75 | 0.69 | 0.62 |
| 20% | 0.30 | 0.79 | 0.79 | 0.62 |
Hybrid never beat dense on this set. Provisional defaults: cutoff 2%, keyword weight 0.3. The set cannot show hybrid's benefit
(questions avoid the passage's words); exact-term questions are needed before concluding anything (ADR 003, experiment 3).

## B-009 Reranker latency on this CPU (2026-10-02, task 2.4)
94 real candidates (avg 171 words), local cross-encoders through fastembed, batch 16 (first run measured 94 pairs; per-pair figures recomputed):

| model | size | ms per pair | 20 candidates | 50 candidates |
|---|---|---|---|---|
| Xenova/ms-marco-MiniLM-L-6-v2 (default) | 80 MB | ~71 | ~1.4 s | ~3.5 s |
| jinaai/jina-reranker-v1-turbo-en | 150 MB | ~88 | ~1.8 s | ~4.4 s |
| Xenova/ms-marco-MiniLM-L-12-v2 | 120 MB | ~120 | ~2.4 s | ~6.0 s |
| BAAI/bge-reranker-base (1 GB) | not completed | | | |
Reranking 50 candidates breaks the 2.5 s first-token budget on this machine; the default reranks 20 (task 2.13 tunes: fewer candidates,
shorter passages, ONNX quantisation, or the hosted reranker).

## B-010 End-to-end baseline, live (2026-10-02, tasks 2.4-2.6) - INFORMAL, 12 drafted unreviewed questions
Real retrieval (hybrid, 20 reranked, top 8, 3,000-token context), real Haiku 4.5 answers, real database.
- Cost: $0.042 for 12 questions = **~$0.0035 per answer** (about 3,500 input and 150 output tokens each).
- Stage latency (typical): retrieve 20-75 ms (first call 450 ms warm-up), **rerank ~1.0-1.1 s**, context 5 ms, LLM 1.9-4.7 s (not streaming yet).
- Quality (not a metric): 4 of 8 answerable questions were answered with the gold section in context; 2 were refused although the gold
  section was in context; 3 of 4 "unanswerable" questions were refused; 1 "unanswerable" question was answered with citations to a real
  passage, so that item is probably mislabelled (an example of what the skipped human review would have caught).
- Reranker scores (MiniLM logits) are not comparable across questions (range about -9 to +6), so a fixed score threshold for refusals needs
  calibration on the eval set (task 2.11); it stays disabled until then.

## B-011 INFORMAL retrieval metrics preview (2026-10-02, task 2.9)
79 answerable LLM-drafted, UNREVIEWED questions; chunker B; hybrid weights 1.0 / 0.3; Tenant A and B scored against their own corpus.
Not a result (the official run is task 2.11); a sanity check that the metric code works on real data.

| mode | Recall@1 | Recall@5 | Recall@10 | Recall@50 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| dense | 0.41 | 0.66 | 0.75 | 0.84 | 0.56 | 0.59 |
| keyword | 0.02 | 0.29 | 0.41 | 0.58 | 0.15 | 0.20 |
| hybrid | 0.36 | 0.66 | 0.75 | 0.84 | 0.52 | 0.57 |

By subset (hybrid, Recall@5): single-fact 0.79 (n=61), multi-hop 0.56 (n=8, partial credit), **follow-up 0.00 (n=10)**: a follow-up like
"How many topics are available for these sessions?" cannot be retrieved without the chat history (query rewriting, Phase 5). IRS Pub 969 1.00,
Pub 15-B 0.67, GitLab 0.62, PostHog 0.69. Recall@50 = 0.84 is the ceiling for the reranker (it can only reorder what was found);
the follow-ups alone account for 10/79 = 12.7% of the missing 16%.

## B-012 Real API smoke test (2026-10-02, tasks 2.7-2.8)
Real uvicorn server, real models, real database, real Haiku call, one question, WARM server (models loaded and warmed at startup, server ready in 12 s).
Cold first request for comparison: first token 5.4 s (rerank 2.0 s cold, tokenizer load 0.76 s) -> warm-up at startup added.

| stage | warm |
|---|---|
| retrieve (embed + dense + keyword + fuse) | 64 ms |
| **rerank (20 candidates, local MiniLM-L6)** | **1,919 ms** |
| context assembly | 18 ms |
| LLM time to first token | ~0.9 s |
| **time to first token (total)** | **2.9 s** (target < 2.5 s) |
| answer complete | 5.7 s |
| LLM cost | $0.0036 |
The reranker is about two thirds of the wait. Task 2.13: fewer candidates (10), shorter passages, quantised ONNX, or the hosted reranker.

## B-013 Judge calibration, SYNTHETIC (2026-10-02, task 2.10)
Judge: Claude Sonnet 5.5 (default temperature: it accepts only that), cost $0.35 for 48 cases. Not human calibration (the human review was skipped).
Cases are built from 15 drafted items: the gold answer ("faithful"), the gold answer with its first number changed ("number", only when a number
exists: 3 cases), the gold answer plus a claim the passage never makes ("extra_claim"), and another question's answer ("other").

| check | result |
|---|---|
| corrupted answers with faithfulness < 1.0 | 100% (18 of 18) |
| swapped number labelled not-correct | 100% (3 of 3) |
| another question's answer labelled incorrect | 100% (15 of 15) |
| faithful (gold) answer labelled correct | 100% (15 of 15) |
| **faithful (gold) answers scored below 1.0 (false alarms)** | **47% (7 of 15); mean faithfulness 0.83** |

Reading: the judge reliably catches deliberate local errors. It is also STRICT: it flags inferences that the gold answer states but the passage
does not directly say (the gold answers are LLM-drafted; only their proof sentence was verified). So ~0.83 is the practical ceiling for
"faithful" on this data, and faithfulness numbers must be read against that baseline (and compared between configurations, not to 0.90).
The first version of the calibration set was flawed (12 of 15 "number" cases had no number to swap and fell back to an extra claim); fixed.

## B-014 Reranker latency spike (2026-10-02, task 2.13)
MiniLM-L6 cross-encoder on this CPU, 69 answerable questions, retrieval done once. Estimated first token = rerank p50 + 982 ms (answer model).

| configuration | rerank p50 | R@1 | R@5 | MRR | est. first token |
|---|---|---|---|---|---|
| no reranker (hybrid) | 0 | 0.41 | 0.76 | 0.60 | |
| 20 cand, 220 words (old default) | 1955 ms | 0.54 | 0.82 | 0.69 | 2.94 s |
| sorted, 20 cand, 120 words | 1335 ms | 0.48 | 0.80 | 0.65 | 2.32 s |
| **sorted, 10 cand, 220 words (new default)** | **962 ms** | 0.54 | 0.83 | 0.69 | **1.94 s** |
| sorted, 5 cand, 160 words | 372 ms | 0.49 | 0.76 | 0.65 | 1.35 s |
| jina turbo, 10 cand, 160 | 1047 ms | 0.45 | 0.79 | 0.64 | 2.03 s |

Finding: only the top 10 fused candidates matter, so halving the candidates costs no quality and meets the 2.5 s target. Cutting words hurts quality; 5 candidates falls back to no-rerank quality; 6 threads and the jina model did not help. Informal: unreviewed eval set. Defaults changed: `rerank_candidates=10`, `reranker_sort_by_length=True`.

## B-015 Injection defences and isolation attacks (2026-10-02, tasks 3.6-3.8)
| check | result |
|---|---|
| Adversarial isolation suite (`make adversarial`): forged ids/headers/tokens, other tenants' document ids, hostile questions, 8 uploaded attack documents | **47 attacks, 0 leaks** |
| Ingestion scan on the 8 attack documents (comment, zero-width, Unicode tags, CSS-hidden, title, visible, forged `<source>`, exfil image) | 8 of 8 flagged |
| Ingestion scan false positives on the real corpus (4,582 Markdown/HTML files) | first version 102 flagged (2.2%); after tightening each rule to the attack shape **12 flagged (0.26%)**, all benign (CDN image URLs with query strings, one lab describing a system prompt) |
| Live model (Haiku 4.5, real prompt), 8 attack documents (`scripts/injection_live.py`) | **obeyed 0 of 8**; cost $0.0038 |

Findings: (1) the cleaning step itself ASSEMBLED a zero-width-spaced instruction into readable text; fixed by dropping lines that are mostly invisible characters. (2) A document title lands in every chunk's heading path, so an instruction-like title is replaced. (3) My first test counted the attacker's own question echoed in the prompt as a leak; the harness now inspects retrieved context only. Informal: simple attacks and one model; the red-team pass in Phase 7 uses garak/promptfoo.

## Eval runs (appended by `make eval`; LLM-drafted, UNREVIEWED questions)
| date | label | config | n | Recall@5 | MRR | nDCG@10 | faithfulness | correctness | refusal acc | p50 / p95 ms | $/answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-02 11:49 | baseline-1 | structured 400t, hybrid, rerank 20/8 local | 69 | 0.82 | 0.69 | 0.72 | 0.954 | 0.841 | 0.6 | 4396 / 5366 | 0.00344 |
