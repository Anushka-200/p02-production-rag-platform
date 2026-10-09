# Learning log (P02 Production RAG Platform)

What I learned, in the order it happened, with the mistake that taught it. Numbers are in `docs/benchmarks.md`.

## Phase 0-1: foundations and ingestion (2026-09-30 to 2026-10-01)
- **Embeddings are the index.** Chunk size and the heading path embedded with each chunk matter more than clever search: the heading-aware chunker beat the plain recursive one by 8 points of Recall@5.
- **Batch size can swap a laptop to death.** Embedding batch 256 on 8 GB RAM took 138 min; batch 8 took 16 min (B-001). Measure before you scale.
- **A "licence check" is reading the LICENSE file**, not a summary. Each source's terms are recorded in `docs/data.md` before anything was downloaded.
- **A parser must return page numbers or citations are decoration.** PDF passages are keyed by page; Markdown by heading anchor.

## Phase 2: retrieval and answers
- **Hybrid search is not automatically better.** On meaning-heavy questions keyword search added nothing over dense; the reranker (a cross-encoder reading question + passage together) was the real win (+5 to +7 points Recall@5 depending on the eval set; R@1 0.43 to 0.59 on the reviewed set).
- **Refuse by default.** The answer prompt cites sources or says "I couldn't find this"; a checker rejects invented citations and any image or link that is not in a source.
- **An LLM judge needs calibration** against hand-labelled answers before its numbers mean anything.

## Phase 3: tenancy
- **Never trust the prompt for who someone is.** Row-level security in Postgres, with the org and user set from the verified token per transaction, so a forgotten `WHERE` cannot leak. Policies fail closed when the setting is missing.
- **Live runs find what unit tests miss.** The first real two-tenant run found that every worker job crashed (tenant set before autocommit) although 465 tests passed (B-017).
- **Cleaning can create an attack.** Removing zero-width characters assembled an instruction out of letters that were spaced apart; the fix was dropping lines that are mostly invisible characters.

## Phase 4: channels
- **A public key is not a secret.** The widget's site key is in the page HTML; what protects the org is the origin allow-list, public-only documents, rate limits and a budget cap.
- **One "can see" policy also means "can delete".** A test that tried to delete as a visitor showed reads and writes need separate policies.

## Phase 5-6: advanced retrieval and experiments
- **Most fancy techniques did not beat the simple pipeline** (multi-query, HyDE, summary tree, GraphRAG, agent, long context). Each is built, tested and OFF; the value is knowing when to switch one on and having the measurement to prove it.
- **Incremental sync saves 70% of embedding work** on a real handbook history, but the unit is a page: editing three lines re-embeds the whole page.
- **Long context is not free**: 18x the cost and 11x the latency for lower correctness (B-030).
- **My eval was the weakest part.** The first set was drafted by an LLM and unreviewed: about a quarter of the questions were flawed, and "refusal accuracy 0.6" and "follow-ups R@5 0.10" were the eval's fault (the runner never passed chat history). After review: refusal 0.96, follow-ups 0.76. Lesson: audit the ruler before trusting the measurement.
- **Dev gains do not always survive a hidden test.** A prompt change gained 3.8 points on the dev split and 0.6 (noise) on the test split; I reverted it. Tune on dev, look at test once.
- **Three parallel jobs froze a 6 GB laptop twice.** Rule: one heavy job at a time, watch the memory.

## Phase 7: security
- **Attack with a deterministic judge.** Plant canary secrets and injected markers; a breach is a string match, not an opinion. 83 attacks in 14 families, 0 breaches (B-039, B-040).
- **A clean result can hide weak tests.** 12 of 17 harder attacks were refused only because the corpus had no matching content; a stronger test puts the attack's topic inside the corpus.
- **Retention is a feature you have to build.** Nothing deleted old visitor chats until `app/retention.py`.
- **Audit your dependencies.** `pip-audit` found 3 advisories in `cryptography`, pinned below the fix by a transitive dependency; an override fixed it.
- **Check what is in git before the first push.** 605 runtime upload files and a personal email address were in history; rewritten out (123 commits) before any push.
