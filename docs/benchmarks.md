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

## B-016 Memory thresholds and PII redaction (2026-10-02, tasks 3.10-3.11)
Real embeddings (bge-small) and the real extractor (Haiku 4.5); `scripts/memory_live.py`, `make`-free, cost $0.0018.

| pair (cosine) | similarity | should |
|---|---|---|
| "prefers short answers" / "likes brief answers" (paraphrase) | 0.882 | merge |
| "lives in Berlin" / "lives in Munich" (update) | 0.875 | merge |
| "works in the Berlin office" / "...Munich office" (update) | 0.904 | merge |
| "is a nurse" / "works in a hospital" (related, different fact) | 0.883 | keep both |
| "prefers short answers" / "prefers answers in German" | 0.841 | keep both |
| unrelated facts | 0.61-0.64 | keep both |
| question vs a relevant fact | 0.54-0.73 | use |
| question vs an irrelevant fact | 0.44-0.55 | skip |

Finding: the model squeezes similarities into a narrow band, so cosine cannot separate "update" from "related". Chosen: merge at **0.90** (a wrong merge LOSES a fact; a missed merge only leaves a harmless duplicate the user can delete) and use a memory when relevance is at least **0.60**. Known limits: "Berlin -> Munich" (0.875) is NOT merged, so a stale fact can remain until the user deletes it; style preferences ("short answers") are not topical, so they are only used when the question is similar to them. Both are Phase 6 experiments (always-on preferences, LLM-decided replacement).
Extractor on 4 messages: kept "nurse in Berlin", "prefers short answers", "works night shifts"; the email and phone number were NOT stored; the injection attempt and a weather chat produced nothing.

PII redaction (Presidio, pattern recognisers only), `python -m evals.pii_check`, 51 hand-written cases: **26/26 planted items caught** (email 6, phone 6, card 5, IBAN 3, IP 3, SSN 3), **0/28 harmless sentences changed** (versions, dates, ids, prices, paths). Informal: I wrote the cases. Name/place/org detection (spaCy small NER) was measured and rejected: it tagged "Email" as a person and "Priya" as an organisation.

## B-017 Live two-tenant run (2026-10-06, task 3.13)
`make demo-tenants`: real API (ENVIRONMENT=prod, restricted DB role), real RQ worker, real embeddings and Haiku 4.5, two companies (Initech with two users, Hooli with one). **22 checks, 0 failed**, answer spend $0.0049 (memory extraction not included, about $0.0005 per turn). Covers login, upload and ingestion (4 documents, one with a hidden instruction: flagged), cited answers per company, no cross-company or cross-colleague leaks (secret code words, private note, document ids, forged token), sharing and un-sharing, conversation history, memory (learned, private, deleted).

Bugs the live run found that 465 passing tests had missed: (1) `process_job` set the tenant before switching the worker connection to autocommit, so EVERY worker job crashed (fixed by `prepare_worker_connection`, regression test added); (2) Presidio warnings flowed back through the redacting log formatter and called Presidio again (a feedback loop that doubled the test-suite time and flooded the worker log; fixed with a log-level setting and a re-entrancy guard); (3) there was no upload endpoint or way to create a user with a password (added `POST /documents`, `GET /jobs/{id}`, `scripts/create_user.py`); (4) `enqueue_upload` queued the job before committing it, so a fast worker could not see the row (commit first).

## B-018 Widget in a real browser, channel isolation, test-suite speed (2026-10-06, tasks 4.7, 4.9)
- **Real browser** (headless Chromium via Playwright; real API under uvicorn; the demo page on a SECOND origin; model faked): 6 tests pass 5 runs in a row. They cover ask, cited answer, thumbs, session continuity across a page reload, handoff plus consent-gated lead, markup shown as text (`window.__pwned` stays undefined), another website blocked by the browser's own CORS, and a 375 px phone viewport. Breaking the widget (answers rendered as HTML) or the origin check makes the matching test fail.
- **Two flaky-test races found and fixed** (the test acted before the widget's final event had stored the session id). Lesson: a browser test must wait for the event that proves the state it needs, not for text that merely appears earlier in the stream.
- **Channel tests** (`tests/test_channels.py`, 6): no channel shows another company's or any internal text; staff see only their company's leads and handoffs from BOTH channels; one site's budget or limit does not affect another; web and WhatsApp spend add up in one total; a session id from one company is useless on another's widget.
- **A real privacy gap found by thinking through these tests:** a phone that switched company (`START other`) left its old conversation and lead behind with no way left to delete them. Now the switch erases the old data first (test fails without it).
- **Budgets are in whole cents** (`numeric(8,2)`); a budget of 0.0001 silently becomes 0.00 and blocks every request.
- **Suite speed: 37 min -> 6.5 min** (559 tests). The cause was `TRUNCATE` (about 6.5 s per test on this drive because of file syncs, not the schema creation or the logging loop I first suspected). One shared schema per run plus ordered `DELETE` takes milliseconds.
- **Chromium on a machine without root:** its four missing system libraries (libnss3, libnspr4, libasound2) were unpacked from `apt-get download` into `~/.cache/chromium-libs`, which the test module adds to `LD_LIBRARY_PATH`.

## B-019 Query rewriting, multi-query, HyDE (2026-10-07, tasks 5.2, 5.3)
Production pipeline (hybrid search, rerank the best 10) with each transform on or off, per question group. Results are Recall@5 / MRR; `evals/query_experiments.py`, transform calls cached so a repeat run is free and identical. Informal: the eval set is LLM-drafted and not reviewed; the paraphrase group (24, reworded to share few words with the original, often stiffly formal, so a harsh test) is new.

| config | follow-up (n=10) R@5 / MRR | paraphrase (24) R@5 / MRR | single (61) R@5 / MRR | multi-hop (8) R@5 / MRR | latency p50 |
|---|---|---|---|---|---|
| baseline | 0.00 / 0.00 | 0.67 / 0.43 | 0.85 / 0.69 | 0.62 / 0.68 | 1.1 s |
| **rewrite** | **0.60 / 0.47** | 0.67 / 0.43 | 0.85 / 0.69 | 0.62 / 0.68 | 1.1 s |
| multi-query (3) | 0.00 / 0.01 | 0.62 / 0.44 | 0.85 / 0.71 | 0.56 / 0.59 | 2.8 s |
| HyDE | 0.00 / 0.01 | 0.62 / 0.42 | 0.89 / 0.72 | 0.62 / 0.70 | 3.5 s |
| all three | 0.60 / 0.49 | 0.62 / 0.44 | 0.87 / 0.71 | 0.56 / 0.59 | 1.3 s* |

*the "all" run reused cached model outputs from the earlier configs, so its latency is not comparable; the other latencies include live model calls (which add about 0.3 to 2.3 s).

**Reading it (n is small: one paraphrase question = 4 points, one multi-hop question = 12):**
- **Rewriting works where it should and costs nothing elsewhere:** follow-ups go from 0 of 10 found in the top 5 to 6 of 10; every other group is identical (no history, no model call). It leaves 4 unfixed: 3 where the history never names what "they"/"this" refers to (the model correctly leaves the question unchanged) and 1 resolved correctly but ranked 51st.
- **Multi-query and HyDE did NOT help paraphrases** (0.67 -> 0.62, one question worse) and add 1.7 to 2.4 s. HyDE gained 2-3 single questions (0.85 -> 0.89) which is within noise at n=61; multi-query may hurt multi-hop (one question). **No demonstrable benefit for the cost: left OFF by default.** They stay available as switches, to be retried on a reviewed, larger set in task 6.9.
- Why paraphrases stay hard (0.67 vs 0.85 for the originals): the reworded text is the stiff vocabulary a model produces ("remuneration", "contractual obligations"), which sits no closer to the document than the original did; a human re-wording may behave differently.
- **Bugs the tests caught while building:** the admin inspector called the live model instead of the injected one (it used a real model in a test); three of my own tests depended on tie-breaking by random ids and failed intermittently until they asserted "in the top two" instead of "first".

## B-020 Parent expansion, and the summary tree (2026-10-07, tasks 5.4-5.6)
**Parent expansion** (swap a hit for its larger parent section), informal, eval set unreviewed.
- *Context only* (free, no model; `evals/context_experiments.py`): is the gold place in the context the answer model sees?

| config | single: any gold | multi-hop: ALL gold | context tokens | sources |
|---|---|---|---|---|
| baseline | 0.87 | 0.50 | 2539 | 7.9 |
| parents <= 400 tokens | 0.87 | 0.50 | 2539 | 7.9 (no parent is that small: identical) |
| parents <= 700 | 0.87 | 0.50 | 2611 | 7.4 |
| parents <= 1200 | 0.85 | 0.50 | 2634 | 4.7 (big parents crowd out other sources) |
- *Answer quality* (real model + judge, 118 questions, `baseline-2` vs `parents-700`): gold place in context 0.828 vs 0.828; correctness 0.774 vs 0.774; faithfulness 0.947 vs 0.956; cost per answer $0.0035 either way. 72 of 118 contexts and 52 answers changed, and 8 questions changed correctness in BOTH directions (4 better, 4 worse), net zero.
- **Decision: implemented, OFF by default.** A parent cannot add a place that retrieval did not find; it only widens the text around it, and at a 700-token cap that neither helped nor hurt measurably. Retest on a reviewed set, or only for questions where the first search looks incomplete.

**Summary tree (RAPTOR-style), measured on Tenant B (2026-10-07):** the full tree for Tenant B (3,606 chunks) = **409 first-level + 62 second-level summaries for $1.26** (a 8-minute-per-100-calls sequential build, about 25 minutes). Gold place in the context the answer model sees (free, `evals/context_experiments.py`; n=12 broad, 20 relationship, 61 single; one question = 8, 5 and 1.6 points):

| config | single any | broad any / all | relationship any / all |
|---|---|---|---|
| baseline | 0.87 | 0.83 / 0.33 | 0.55 / 0.20 |
| tree (5 summaries x 4 chunks) | 0.85 | 0.75 / 0.33 | 0.40 / 0.05 |
| tree-small (3 x 2) | 0.87 | 0.75 / 0.33 | 0.50 / 0.20 |
| tree + graph | 0.85 | 0.75 / 0.33 | 0.65 / 0.15 |

**The tree did NOT help on this corpus at any dose** (broad 0.83 -> 0.75, one question; relationship worse). Consistent with the expectation that hybrid search plus reranking already finds scattered passages when the question shares words with them; summaries are a signpost for THEMATIC questions with no shared words, and our broad questions were drafted from section headings so they do share words. Also seen: the second-level summaries are grab-bags, the retrieval run-to-run noise is about one question (paraphrase baseline read 0.71 and 0.67 in two runs), and the tree adds latency and cost for no measured gain. **Decision: implemented, tested, privacy-proofed, OFF by default.** Retest if broad, truly thematic questions become a real use case (the better first experiment: group by document or section path BEFORE clustering).
- Real trial on 240 sampled chunks of Tenant B: 27 first-level + 4 second-level summaries for **$0.08**. The summaries keep names, numbers and policy terms. The second-level summaries are grab-bags of unrelated topics (k-means over a mixed handbook does not always find clean topics).
- Projected cost of the full trees (cost guard): Tenant A **$1.64**, Tenant B **$1.66**.
- 12 new BROAD questions (answer spread over 3-5 sections of one document, 9 of them Tenant B) drafted for $0.018; kept out of the headline.
- Privacy by construction: summaries are built per visibility group, never mixing; triggers delete a summary as soon as any source chunk is deleted, re-indexed or made less visible (3 tests fail if the triggers are removed); visitors see only public summaries.
- A tooling fix: new question groups (paraphrase, broad) no longer enter the answer headline (it had silently grown from 84 to 108 questions, so the `baseline-2` and `parents-700` rows above are on 108 and must not be compared with older rows).

## B-021 GraphRAG slice (2026-10-07, tasks 5.7-5.9)
Slice: Tenant B customer-facing sections (`cs-and-onboarding`, `onboarding`, `support`): **375 chunks -> 1,532 facts** (subject --relation--> object, each pointing at the chunk that states it) for **$0.66** (about $0.0018 per chunk, 4 parallel workers, 8 minutes). Facts are kept in a Postgres table (`graph_edges`), not a graph database: this slice compares the IDEA, and deep traversal belongs to a later project.

20 RELATIONSHIP questions were generated from real two-step fact chains in that graph (the question names the START of the chain, asks about the END, and may not name the middle) for $0.025. **Caveat that matters:** they were derived from the same extraction the graph search uses, so the graph finding them is partly by construction, and not all 20 truly need two passages (some can be answered from one). Treat the size of the gain as an upper bound.

Gold place in the context the answer model would see (free, no model, `evals/context_experiments.py`):

| group (n) | baseline any / all | + graph any / all | change |
|---|---|---|---|
| relationship (20) | 0.55 / 0.20 | **0.85 / 0.35** | +30 / +15 points |
| single (61) | 0.87 / 0.87 | 0.87 / 0.87 | none |
| multi-hop (8, mostly outside the slice) | 0.88 / 0.50 | 0.88 / 0.50 | none |
| paraphrase (24) | 0.71 / 0.71 | 0.67 / 0.67 | -1 question |
| **broad (12)** | 0.83 / 0.33 | **0.58 / 0.33** | **-25 points** |

**Reading it:** following facts finds the second passage that shares no words with the question (the thing chunk search cannot do), but the extra chunks it adds crowd out the right ones for broad questions (a broad question names things that have many facts, so the walk returns plenty of loosely related chunks). One technique, a big win and a clear loss, depending on the QUESTION TYPE: this is the case for choosing a method per question (a router, block 5), not for switching a technique on globally.

Findings on the data: the extraction is usable but noisy. The same real-world thing appears under several names ("support team", "customer support team", "support engineers"): names are normalised (case, articles, underscores) but not resolved to one entity, so some chains break. A model sometimes wrote names in snake_case, which would have split nodes; fixed by normalising before storing. Cost scales linearly with chunks: the 3,600-chunk tenant would cost about $6.50 to read in full.

## B-022 Agentic retrieval (2026-10-07, tasks 5.10, 5.11)
The agent adds searches to the plain one; measured by whether the gold place(s) are in the context the answer model sees (free of answer cost; the planner and checker are real Haiku 4.5 calls, cached). Informal: eval set unreviewed, relationship questions derived from the graph.

**First design (the planner REPLACED the search): worse than no agent.** single 0.87 -> 0.77, paraphrase 0.67 -> 0.54, multi-hop 0.88/0.50 -> 0.75/0.25, with 2 to 10 extra seconds. Two faults found by inspecting what it did: (1) single-part questions got only 4 sources against the plain pipeline's 8 (a per-part cap meant for multi-part questions), (2) the planner rewrote clear questions into keyword strings that rerank worse. **Fix = a design rule: the plain search of the question as asked ALWAYS runs first and keeps its top 5; the agent may only ADD evidence.**

**Corrected design, always on / gated (agent only when the plain search looks weak):**

| group (n) | plain any / all | always agent | gated | share escalated | est. extra s per question (always -> gated) |
|---|---|---|---|---|---|
| single (61) | 0.87 / 0.87 | 0.90 / 0.90 | 0.89 / 0.89 | 11% | 4.3 -> 0.5 |
| paraphrase (24) | 0.67 / 0.67 | 0.71 / 0.71 | 0.71 / 0.71 | 62% | 3.7 -> 2.1 |
| multi-hop (8) | 0.88 / 0.50 | 0.88 / 0.50 | 0.88 / 0.50 | 25% | 3.9 -> 0.9 |
| broad (12) | 0.83 / 0.33 | 0.92 / 0.42 | 0.92 / 0.33 | 25% | 3.9 -> 1.1 |
| relationship (20) | 0.55 / 0.20 | 0.60 / 0.20 | 0.55 / 0.20 | 15% | 2.6 -> 0.2 |

Cost of the always-on agent: about $0.003 per question (planner and checker). Tools chosen over 125 questions: search 286 (the plain search plus extras), exact 24, broad 15, connections 5; stopped by: 54 hit the 4-search cap, 60 enough, 11 round cap, which shows the planner is over-eager (it proposes extras for questions that did not need them).
**Calibration of the gate** (single, paraphrase and unanswerable questions): a best rerank score below -1 flags 59% of the questions that missed their gold place and 18% of those that found it. The score cannot spot unanswerable questions (their median best score, 2.19, is nearly as high as good matches': 2.74), so it triggers a second look, never a conclusion. Latency figures use an assumption of about 1 s per small-model call; the first uncached run measured roughly 9 s of extra latency per question for the always-on agent.
**Decision:** built, tested and **off by default** (`AGENT_ENABLED`); when on, gated by default (`AGENT_ONLY_WHEN_WEAK`). A modest, measured gain (+2 single, +4 paraphrase, +9 broad any-gold) for roughly half a second per ordinary question; worth switching on after a human-reviewed eval confirms it.

## Eval runs (appended by `make eval`; LLM-drafted, UNREVIEWED questions)
| date | label | config | n | Recall@5 | MRR | nDCG@10 | faithfulness | correctness | refusal acc | p50 / p95 ms | $/answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-02 11:49 | baseline-1 | structured 400t, hybrid, rerank 20/8 local | 69 | 0.82 | 0.69 | 0.72 | 0.954 | 0.841 | 0.6 | 4396 / 5366 | 0.00344 |
| 2026-10-07 12:49 | parents-700 | structured 400t, hybrid, rerank 10/8 local | 69 | 0.83 | 0.69 | 0.72 | 0.956 | 0.774 | 0.6 | 3968 / 4832 | 0.00357 |
| 2026-10-07 12:52 | baseline-2 | structured 400t, hybrid, rerank 10/8 local | 69 | 0.83 | 0.69 | 0.72 | 0.947 | 0.774 | 0.6 | 3979 / 4983 | 0.00351 |

## B-023 Incremental sync of a living wiki (2026-10-07, tasks 5.12, 5.13)
Real run, GitLab handbook `people-group` (67 Markdown pages), local embeddings, scratch org. Sync 1 = the repo as of
2026-08-01; sync 2 = HEAD (2026-09-24). Wall-clock on this CPU.

| Run | Pages read + embedded | Skipped (fingerprint equal) | Time |
|---|---|---|---|
| Full build (sync 1) | 67 | 0 | 4 min 51 s |
| Incremental to HEAD (sync 2) | 32 changed | 35 | 1 min 57 s |
| Sync again, nothing changed | 0 | 67 | seconds |

Incremental sync took 40% of the full-build time, and the saving grows with the size and quietness of the wiki
(these 67 pages are an unusually busy folder: 48% changed in 8 weeks). Model cost: $0 (no LLM calls, no blurbs).

## B-024 Replaying real handbook history (2026-10-07, task 5.15)
`evals/sync_replay.py`: GitLab `people-group` (Tenant A source), five syncs at fortnightly points of the real git
history into a scratch org. Local embeddings, $0. "Embedded" = chunks of the pages that sync added or changed.

| Step | Pages changed | Unchanged | Chunks embedded / total | Share of a full rebuild | Seconds |
|---|---|---|---|---|---|
| 2026-08-01 (first build) | 67 | 0 | 1072 / 1072 | 100% | 151 |
| 2026-08-15 | 12 | 55 | 255 / 1071 | 24% | 48 |
| 2026-09-01 | 7 | 60 | 257 / 1071 | 24% | 39 |
| 2026-09-15 | 22 | 45 | 609 / 1070 | 57% | 85 |
| 2026-10-01 | 4 | 63 | 176 / 1070 | 16% | 24 |

Four incremental steps re-embedded 1297 chunks against 4283 for four full rebuilds (30%) and took 196 s against about
600 s. Limit seen: the unit is the whole page, so editing a few lines of a long page re-embeds all its chunks (7 pages
= 257 chunks). Re-using the embedding of every chunk whose own text hash is unchanged is the next saving (not built).

## B-025 Library chunkers vs ours; LangGraph vs plain Python (2026-10-07, task 5.16, ADR 010)
**Chunkers** (`evals/chunker_compare.py`): GitLab people-group, 67 pages, same embedding model, in-memory dense search,
document-level hit, 14 questions with unreviewed gold. Informal; $0.

| Chunker | Chunks | Tokens mean / p95 / max | hit@1 | hit@5 | MRR |
|---|---|---|---|---|---|
| Ours (structured, heading-aware, heading path embedded) | 1069 | 251 / 386 / 400 | 0.71 | 1.00 | 0.83 |
| LangChain recursive (Markdown separators) | 1008 | 259 / 382 / 399 | 0.79 | 0.93 | 0.87 |
| LangChain header split + recursive | 1491 | 180 / 368 / 413 | 0.71 | 1.00 | 0.84 |

One question is 7 points of hit rate, so these are the same within noise. The library output has no page numbers,
anchors, parent-child levels or token-exact limits (header split overshoots to 413); ours keeps them.

**Orchestration** (`scripts/langgraph_spike.py`, 111 lines incl. CLI): retrieve > assemble > answer as a LangGraph graph
whose nodes call our own functions, 8 Tenant A questions, live model, $0.059 for both paths.

| | Plain Python (`rag.ask`) | LangGraph |
|---|---|---|
| Same cited documents | | 8 / 8 |
| Latency p50 / mean | 3.50 s / 3.67 s | 3.33 s / 3.58 s (difference = model noise) |
| Rewrite, agent, tree, graph, streaming | built in | would each need a node and routing: not ported |

## B-026 Phase 6 block 1: baseline, retrieval modes, HNSW ef_search (2026-10-07, tasks 6.1, 6.4, 6.8)
Data: current dev database (11,685 chunks incl. two scratch orgs), 150-question eval set, 69 answerable headline
questions, unreviewed gold (informal).

**6.1 Baseline per answer** (row baseline-2 above, full pipeline, Haiku 4.5, local embeddings/reranker):
cost 0.00351 USD per answer; end-to-end p50 3.98 s / p95 4.98 s (retrieval + local rerank ~1 s, the rest is the model call).

**6.4 Retrieval modes** (`python -m evals.run --label modes-p6`, no model cost):

| mode | R@1 | R@5 | R@10 | R@50 | MRR | nDCG@10 | paraphrase R@5 |
|---|---|---|---|---|---|---|---|
| dense | 0.46 | 0.76 | 0.86 | 0.94 | 0.64 | 0.67 | 0.62 |
| keyword | 0.02 | 0.33 | 0.47 | 0.66 | 0.17 | 0.23 | 0.08 |
| hybrid (RRF) | 0.41 | 0.76 | 0.86 | 0.94 | 0.60 | 0.65 | 0.62 |
| hybrid + rerank | 0.54 | 0.83 | 0.86 | 0.94 | 0.69 | 0.72 | 0.67 |

Findings: the reranker is the clear win (+7 points R@5, +13 points R@1 over hybrid). Hybrid did NOT beat dense here
(the brief's +10 points target is not met on this set): keyword search alone is weak on natural-language questions
(R@5 0.33) and adds noise at rank 1. Cohere rerank was not compared (no key).

**6.8 HNSW ef_search sweep** (`evals/hnsw_sweep.py`, 140 questions, iterative scan on, as in production):
top-10 overlap with an exact search is 1.000 for every ef_search from 10 to 400, SQL p50 13.0-13.2 ms in every case
(exact search 13.5 ms). Conclusion: at 10k vectors with tenant filters ef_search does not matter; keep the default (40).
The cost is dominated by the join and filter, not the graph walk. Revisit at 100k+ chunks.

## B-027 Chunker A vs B and chunk size / overlap (2026-10-07, tasks 6.2, 6.3)
`evals/chunk_experiments.py`: all of Tenant A (226 documents), same embedding model (bge-small), dense search in memory,
real eval metrics on 87 Tenant A answerable non-follow-up questions (informal: unreviewed gold; one question is about 1 point).
No blurbs, no rerank, so chunking is the only change. Free.

| config | chunks | tokens median | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| A recursive 400/50 | 4370 | 171 | 0.33 | 0.67 | 0.75 | 0.52 | 0.55 |
| **B structured 400/50 (production)** | 3547 | 275 | 0.42 | 0.75 | 0.89 | 0.62 | 0.67 |
| B 200/25 | 6896 | 153 | 0.42 | 0.69 | 0.84 | 0.58 | 0.62 |
| B 300/40 | 4640 | 219 | 0.46 | 0.73 | 0.86 | 0.64 | 0.67 |
| B 400/0 | 3515 | 271 | 0.44 | 0.76 | 0.89 | 0.63 | 0.68 |
| B 400/100 | 3628 | 282 | 0.41 | 0.77 | 0.89 | 0.61 | 0.67 |
| B 500/50 | 2969 | 319 | 0.48 | 0.79 | 0.90 | 0.66 | 0.70 |

Findings: (1) the heading-aware chunker beats plain recursive splitting by +8 points R@5 and +14 at R@10: this is the
one large, reliable difference. (2) Size and overlap barely matter between 300 and 500 tokens (R@5 0.73-0.79, within
noise at n=87); very small chunks (200) are worse (more chunks, less context each). (3) Overlap 0/50/100 makes no
measurable difference, so overlap can be 0 and save ~1% of chunks. (4) 500 is nominally best but is close to the
embedding model's 512-token limit (heading prefix + text can be cut), so we keep 400/50 and do not claim a win.

## B-028 Semantic cache: threshold and behaviour (2026-10-08, task 6.7)
`evals/cache_threshold.py` (free, local embeddings): pretend the cache holds an answer for every other eval
question (140 questions, follow-ups excluded). For each question the nearest neighbour is the entry it would hit.
Correct = the neighbour has the same gold location; wrong = different gold or the question is unanswerable.

| threshold | hits | correct | WRONG | paraphrases served (of 24) |
|---|---|---|---|---|
| 0.80 | 49 | 35 | 14 | 17 |
| 0.85 | 27 | 22 | 5 | 11 |
| 0.88 | 8 | 8 | 0 | 4 |
| 0.90 | 6 | 6 | 0 | 3 |
| **0.93 (chosen)** | 2 | 2 | 0 | 1 |
| 0.96 | 2 | 2 | 0 | 1 |
| 0.98 | 0 | 0 | 0 | 0 |

The closest pair that must NOT share an answer is 0.874 ("how does the support team handle customer issues" vs
"what resources should support use when investigating customer issues"). At 0.85 29% of hits would be wrong. So the
threshold sits above the first wrong pair with a margin (0.93), and the local bge-small model means paraphrases
mostly fall below it: the cache pays for repeats, not for rewordings. A better embedding model (6.6) could move this.
Hit latency in the API test: about 10 ms total (6 ms lookup) and $0, against a fresh answer at p50 3.98 s and 0.35 cents.
Live run against the real stack (`scripts/cache_live.py`, real model, $0.006): first ask 11.2 s (CPU was busy with
experiments) and $0.0032; the same question again 19 ms and $0, same citations; a lower-case rewording 19 ms and $0
(similarity above 0.93); the same words asked as the OTHER tenant: a miss, answered fresh (and correctly refused, since
Tenant B has no such document). One cache row saved, shareable because only org-visible documents were used.

## B-029 Per-tenant budget: the runaway-cost test (2026-10-08, task 6.13)
Fake model, every fresh answer costs $0.002, org cap set to $0.005 (`tests/test_budget.py`). A script hammers `/chat` 50 times:

| | result |
|---|---|
| requests answered | 3 (spend 0.002, 0.004, 0.006) |
| requests refused with HTTP 429 | 47 |
| model calls made | 3 (blocked requests never reach retrieval or the model) |
| spend at the end | $0.006: the cap plus the one answer that was already running |
| other tenant, same moment | unaffected, answers normally |
| alert | one warning at 80% ($0.004), one "exhausted" error at 100%, each logged once per day |
Without the check the same loop would have spent $0.10 (50 x $0.002), i.e. 20x the cap. Production default cap: $5 per org per
day (`ORG_DAILY_BUDGET_USD`, 0 = unlimited, per-org override in `orgs.daily_budget_usd`). Covers member chat, widget and WhatsApp.
Known limit: parallel requests already in flight when the cap is crossed still finish (overshoot <= parallelism x one answer).

## B-030 Long context vs RAG (2026-10-08, task 6.11)
`evals/long_context.py`. Corpus: Tenant A `finance` folder, 150,080 tokens, put whole into Haiku 4.5's prompt (prompt-cached).
17 eval questions whose gold is in that folder; same judge and (unreviewed) gold answers for both sides. Long context is
told the right folder for free, which RAG (searching all of Tenant A) is not: a generous setup for long context. Informal, n=17.

| approach | correctness strict | correctness lenient | cost per answer | latency median | context |
|---|---|---|---|---|---|
| RAG (8 chunks) | 0.91 | 0.91 | $0.0033 | 4.2 s | ~3,000 tokens |
| Long context (whole folder) | 0.35 | 0.62 | $0.0597 | 46.8 s | 150,080 tokens |

"Strict" counts an answer that opens with the refusal phrase as a refusal; "lenient" judges the whole text, because the
model often said "I couldn't find this" and then gave related facts (11 of 17 answers). First question $0.20 (cache write),
the other 16 average $0.051 (cache reads, 10% price). Spent $0.43.
Findings: RAG was 18x cheaper per answer, 11x faster, and more accurate on this set; with 150k tokens in the prompt the model
missed answers that were in the context. Long context would only be worth it for a small corpus that fits comfortably and
for questions that need the whole document. Caveat: one prompt wording and one model; a larger model may do better.
(An earlier run of this script scored RAG as 0.00 because the saved answers file lacks judge scores; fixed by re-judging.)

## B-031 History compression for follow-up rewriting (2026-10-08, task 6.12)
`evals/history_experiments.py`. The 10 eval follow-ups were buried in 8-turn conversations (unrelated turns taken from other
eval questions and their saved answers). Recall of hybrid+rerank on the REWRITTEN question; n=10, so one question = 10 points.

| referenced turn | strategy | R@5 | R@1 | rewrite prompt chars | model cost / question |
|---|---|---|---|---|---|
| last turn | none (no rewrite) | 0.00 | 0.00 | 0 | $0 |
| last turn | window-3 (production) | 0.60 | 0.40 | 1,328 | $0.0005 |
| last turn | wide (10 turns) | 0.60 | 0.40 | 4,027 | $0.0011 |
| last turn | summary + last 2 turns | 0.70 | 0.50 | 1,250 | $0.0016 |
| 4 turns ago | none | 0.00 | 0.00 | 0 | $0 |
| 4 turns ago | window-3 (production) | 0.00 | 0.00 | 1,595 | $0.0006 |
| 4 turns ago | wide (10 turns) | 0.20 | 0.20 | 4,035 | $0.0012 |
| 4 turns ago | summary + last 2 turns | 0.10 | 0.10 | 1,465 | $0.0016 |

Findings: rewriting is what matters (0.00 without it). When the referenced turn is recent, the production window already works
and compression changes one question at most (0.60 vs 0.70). When the reference is 4 turns back the window loses it (0.00); a
60-word summary or a 10-turn window recovers only 1-2 of 10, so a short summary drops the very detail the rewriter needs.
Wide costs 3x the prompt for the same quality on recent references. Decision: keep window-3; no claim that compression helps.
Spent $0.11. Peak memory 1.1 GB.
