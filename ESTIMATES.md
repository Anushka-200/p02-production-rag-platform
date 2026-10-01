# ESTIMATES: P02 Production RAG Platform
Status: drafted 2026-09-30. Your O/M/P columns are empty on purpose: fill them for a phase **before** starting it.
My own estimates are sealed (git-ignored file `.estimates_claude_view.json`) and are revealed at the end of each phase, next to your numbers and the actuals.
Rules: PERT = (O + 4M + P) / 6. `x mult` is pre-filled from `03_ESTIMATION_TRACKER.md` section B (retrieval 2.0, agent 2.0, integration 1.4, testing-evals 1.1, optimization 1.2, others 1.0; debugging +15% buffer is added at phase level). Familiarity is my suggestion (Some/New); change it if wrong. Estimates are your hours with Claude writing the code. From Phase 1 on, note learning/Q&A and build time separately (Phase 0 lesson). Tasks marked SPIKE are timeboxed because neither of us can size them yet. Note the start time before each phase. If elapsed time passes 150% of a task's planned hours, stop and diagnose.
## Phase 0: Foundations
Start time: 17:02 (2026-09-30). Owner's phase-level estimate (all 8 tasks together): O 0.5 / M 0.75 / P 1 h, PERT 0.75 h.

**Claude's sealed view (revealed at phase end):** 8 tasks, PERT 9.32 h, with multipliers 9.82 h. **Actual 1.35 h** = 1.8x the owner's estimate and 0.14x Claude's. Claude's numbers assumed each task takes about 1 h of hands-on time even with AI writing the code; real build time was minutes per task. The owner's numbers were closer in size but missed learning and Q&A time.

| # | Task | Category | Familiarity | O | M | P | PERT | x mult | Planned h | Learning h | Build h | Actual total h | Ratio | Why it was off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | Repo scaffold: uv, ruff, mypy, pytest, pyproject, .env.example, Makefile skeleton | setup | Some | | | | | 1.0 | | | | 0.35 | | 17:02-17:23 |
| 0.2 | docker-compose: Postgres+pgvector (5434), Redis (6380), DB roles | infra | Some | | | | | 1.0 | | | | 0.30 | | 17:23-17:41 |
| 0.3 | Settings (pydantic-settings) + structured JSON logging | setup | Some | | | | | 1.0 | | | | 0.10 | | 17:41-17:47 |
| 0.4 | LiteLLM wrapper: timeout, retry+backoff+jitter, cost capture; new Anthropic key with spend limit | llm-logic | New | | | | | 1.0 | | | | 0.20 | | 17:47-17:59, live run later |
| 0.5 | Langfuse tracing wired + live run and fix | integration | New | | | | | 1.4 | | | | 0.30 | | 17:59-18:17, incl. live run + Langfuse v2 API |
| 0.6 | CI skeleton (GitHub Actions: lint + tests) | infra | Some | | | | | 1.0 | | | | | | 18:17-18:23 together with 0.7, 0.8 (0.10 h) |
| 0.7 | Verify licences (GitLab, PostHog, HR PDFs) -> docs/data.md | data | Some | | | | | 1.0 | | | | | | |
| 0.8 | Download and pin corpora (shallow clone, sizes, D: drive) | data | Some | | | | | 1.0 | | | | | | |
| | **Phase total** | | | 0.5 | 0.75 | 1 | 0.75 | | 0.75 | 1.23 | 0.12 | 1.35 | 1.8 | Owner: underestimated; did not budget learning and Q&A (learning:build about 10:1, split derived from that ratio). Start 17:02, end 18:23 |

## Phase 1: Ingestion + eval set v0
Start time: 19:01 (2026-09-30). Owner's phase-level estimate (all 16 tasks together): O 1 / M 1.25 / P 1.5 h, PERT 1.25 h.

| # | Task | Category | Familiarity | O | M | P | PERT | x mult | Planned h | Learning h | Build h | Actual total h | Ratio | Why it was off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.1 | Read pgvector README (HNSW params) + Postgres full-text docs | learning | New | | | | | 1.0 | | | | 0.85 | | 19:01-19:52; mostly Q&A (filtering, candidates, AND vs OR) |
| 1.2 | Schema migrations: tables + GIN index (HNSW moved to 1.8) | infra | Some | | | | | 1.0 | | | | 0.63 | | 19:52-20:30; includes the plan review done in between |
| 1.3 | Parsers: MD / TXT / HTML with sections | retrieval | New | | | | | 2.0 | | | | | | |
| 1.4 | Parsers: PDF with page numbers, DOCX with tables | retrieval | New | | | | | 2.0 | | | | | | |
| 1.5 | Preprocessing: boilerplate removal, dedupe, table handling | data | Some | | | | | 1.0 | | | | | | |
| 1.6 | Chunker A: recursive by tokens + overlap + tests | retrieval | New | | | | | 2.0 | | | | | | |
| 1.7 | Chunker B: heading-aware + parent-child linking + tests | retrieval | New | | | | | 2.0 | | | | | | |
| 1.8 | Embedding wrapper (local + hosted), batching; embedding column + HNSW index | retrieval | New | | | | | 2.0 | | | | | | |
| 1.9 | Contextual blurb generation (Haiku + prompt caching) + tests | retrieval | New | | | | | 2.0 | | | | | | |
| 1.10 | Persist chunks; versioned re-index; dedupe on re-upload (idempotency) | retrieval | New | | | | | 2.0 | | | | | | |
| 1.11 | RQ worker + job status + failure handling | infra | Some | | | | | 1.0 | | | | | | |
| 1.12 | Live run + fix: ingest a small tenant A slice end to end | integration | New | | | | | 1.4 | | | | | | |
| 1.13 | Eval set: LLM drafts 80 + 15 candidates (with tagged subsets) | data | Some | | | | | 1.0 | | | | | | |
| 1.14 | Eval set: YOUR review of all items, set reviewed flag | data | Some | | | | | 1.0 | | | | | | |
| 1.15 | Tests: ingestion unit + one integration test | testing-evals | New | | | | | 1.1 | | | | | | |
| 1.16 | Live run + fix: bulk-ingest full corpus (cost and time check) | integration | New | | | | | 1.4 | | | | | | |
| 1.17 | Cost guard: projected cost + confirmation above $0.50 for model-calling scripts | llm-logic | New | | | | | 1.0 | | | | | | |
| | **Phase total** | | | 1 | 1.25 | 1.5 | 1.25 | | 1.25 | | | | | |

## Phase 2: Retrieval + answers
| # | Task | Category | Familiarity | O | M | P | PERT | x mult | Planned h | Learning h | Build h | Actual total h | Ratio | Why it was off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.1 | Read RRF explainer, contextual retrieval post + cookbook, Ragas faithfulness docs | learning | New | | | | | 1.0 | | | | | | |
| 2.2 | Dense and keyword search SQL | retrieval | New | | | | | 2.0 | | | | | | |
| 2.3 | RRF fusion (k, weights) + unit tests | retrieval | New | | | | | 2.0 | | | | | | |
| 2.4 | Reranker: local bge + Cohere adapter | retrieval | New | | | | | 2.0 | | | | | | |
| 2.5 | Context assembly with token budget | retrieval | New | | | | | 2.0 | | | | | | |
| 2.6 | Grounded answer prompt: citations, refusal, Pydantic output, repair-then-retry | llm-logic | New | | | | | 1.0 | | | | | | |
| 2.7 | Streaming answers (SSE /chat) | llm-logic | New | | | | | 1.0 | | | | | | |
| 2.8 | Retrieval inspector endpoint | retrieval | New | | | | | 2.0 | | | | | | |
| 2.9 | Metrics code: Recall@K, MRR, nDCG + tests | testing-evals | New | | | | | 1.1 | | | | | | |
| 2.10 | LLM judge (faithfulness, correctness) + calibration on 20 hand labels | testing-evals | New | | | | | 1.1 | | | | | | |
| 2.11 | make eval end to end: baseline per retrieval mode | testing-evals | New | | | | | 1.1 | | | | | | |
| 2.12 | Live run + fix: full query flow on real corpus | integration | New | | | | | 1.4 | | | | | | |
| 2.13 | SPIKE (timebox 1 h): reranker latency on CPU (candidate count, model size, ONNX) | optimization | New | | | | | 1.2 | | | | | | |
| | **Phase total** | | | | | | | | | | | | | |

## Phase 3: Tenancy, auth, memory, PII
| # | Task | Category | Familiarity | O | M | P | PERT | x mult | Planned h | Learning h | Build h | Actual total h | Ratio | Why it was off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.1 | Read Postgres RLS docs, OWASP GenAI Top 10 (2026), multi-tenant RAG posts | learning | New | | | | | 1.0 | | | | | | |
| 3.2 | RLS policies, WITH CHECK, non-bypass roles, composite FKs | infra | Some | | | | | 1.0 | | | | | | |
| 3.3 | JWT auth, orgs/users/roles, per-request app.org_id context | integration | New | | | | | 1.4 | | | | | | |
| 3.4 | Private vs shared docs; permission-aware retrieval; filtered-HNSW / iterative scan test | retrieval | New | | | | | 2.0 | | | | | | |
| 3.5 | SQL-level RLS tests | testing-evals | New | | | | | 1.1 | | | | | | |
| 3.6 | Write adversarial docs (hidden text, metadata, instruction-like text) | data | Some | | | | | 1.0 | | | | | | |
| 3.7 | Adversarial isolation suite: forged tenant, modified doc ids, injection | testing-evals | New | | | | | 1.1 | | | | | | |
| 3.8 | Injection defences: delimiting, output checker, ingestion scan | llm-logic | New | | | | | 1.0 | | | | | | |
| 3.9 | Chat history storage + windowing | llm-logic | New | | | | | 1.0 | | | | | | |
| 3.10 | Memory: extraction, merge/dedupe, store, view/delete API | llm-logic | New | | | | | 1.0 | | | | | | |
| 3.11 | Presidio PII redaction on logs + memory (+ hit-rate check) | integration | New | | | | | 1.4 | | | | | | |
| 3.12 | Memory tests | testing-evals | New | | | | | 1.1 | | | | | | |
| 3.13 | Live run + fix: two tenants end to end | integration | New | | | | | 1.4 | | | | | | |
| 3.14 | Pooled-connection test: tenant setting never carries over; unset org fails closed | testing-evals | New | | | | | 1.1 | | | | | | |
| | **Phase total** | | | | | | | | | | | | | |

## Phase 4: Channels
| # | Task | Category | Familiarity | O | M | P | PERT | x mult | Planned h | Learning h | Build h | Actual total h | Ratio | Why it was off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.1 | Streamlit chat UI | integration | New | | | | | 1.4 | | | | | | |
| 4.2 | Streamlit admin: inspector, ingestion status, chunk previews | integration | New | | | | | 1.4 | | | | | | |
| 4.3 | Widget: single-script JS bundle, site key, CORS, rate limit | integration | New | | | | | 1.4 | | | | | | |
| 4.4 | Twilio sandbox setup + webhook signature verification | integration | New | | | | | 1.4 | | | | | | |
| 4.5 | WhatsApp flow: session-to-org mapping, replies | integration | New | | | | | 1.4 | | | | | | |
| 4.6 | Lead capture tool + human-handoff flag | llm-logic | New | | | | | 1.0 | | | | | | |
| 4.7 | Live run + fix: widget on a demo page | integration | New | | | | | 1.4 | | | | | | |
| 4.8 | Live run + fix: WhatsApp via tunnel | integration | New | | | | | 1.4 | | | | | | |
| 4.9 | Channel tests | testing-evals | New | | | | | 1.1 | | | | | | |
| 4.10 | Public visibility + anonymous scope for widget/WhatsApp, with leak tests | infra | New | | | | | 1.0 | | | | | | |
| | **Phase total** | | | | | | | | | | | | | |

## Phase 5: Advanced retrieval
| # | Task | Category | Familiarity | O | M | P | PERT | x mult | Planned h | Learning h | Build h | Actual total h | Ratio | Why it was off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.1 | Learn: RAG From Scratch (multi-query, HyDE, RAPTOR videos) | learning | New | | | | | 1.0 | | | | | | |
| 5.2 | Query rewriting + tests | llm-logic | New | | | | | 1.0 | | | | | | |
| 5.3 | Multi-query + HyDE behind flags | retrieval | New | | | | | 2.0 | | | | | | |
| 5.4 | Parent expansion in pipeline | retrieval | New | | | | | 2.0 | | | | | | |
| 5.5 | RAPTOR-style summary tree build | retrieval | New | | | | | 2.0 | | | | | | |
| 5.6 | Summary-tree retrieval integration | retrieval | New | | | | | 2.0 | | | | | | |
| 5.7 | SPIKE (timebox 3 h): GraphRAG learning, Neo4j course / LazyGraphRAG | learning | New | | | | | 1.0 | | | | | | |
| 5.8 | GraphRAG slice: entity + relationship extraction on one section | retrieval | New | | | | | 2.0 | | | | | | |
| 5.9 | Graph query + comparison harness (20 relationship questions) | retrieval | New | | | | | 2.0 | | | | | | |
| 5.10 | Agentic RAG loop: decompose, re-search, tool choice, step cap | agent | New | | | | | 2.0 | | | | | | |
| 5.11 | Agent tests and trace checks | testing-evals | New | | | | | 1.1 | | | | | | |
| 5.12 | Connector interface + git handbook connector | integration | New | | | | | 1.4 | | | | | | |
| 5.13 | Change detection (hash), incremental re-index, deletes | integration | New | | | | | 1.4 | | | | | | |
| 5.14 | Permission sync + last-updated in citations | integration | New | | | | | 1.4 | | | | | | |
| 5.15 | Live run + fix: replay handbook git history | integration | New | | | | | 1.4 | | | | | | |
| 5.16 | LangChain / LangGraph spike (ADR 010) | integration | New | | | | | 1.4 | | | | | | |
| 5.17 | Sync tests | testing-evals | New | | | | | 1.1 | | | | | | |
| | **Phase total** | | | | | | | | | | | | | |

## Phase 6: Evaluate and optimize
| # | Task | Category | Familiarity | O | M | P | PERT | x mult | Planned h | Learning h | Build h | Actual total h | Ratio | Why it was off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6.1 | Baseline cost per answer + p50/p95 from Langfuse traces | testing-evals | New | | | | | 1.1 | | | | | | |
| 6.2 | Exp 1: chunker A vs B | optimization | New | | | | | 1.2 | | | | | | |
| 6.3 | Exp 2: chunk size / overlap sweep | optimization | New | | | | | 1.2 | | | | | | |
| 6.4 | Exp 3: dense vs keyword vs hybrid vs hybrid+rerank (+ Cohere) | optimization | New | | | | | 1.2 | | | | | | |
| 6.5 | Exp 4: contextual retrieval | optimization | New | | | | | 1.2 | | | | | | |
| 6.6 | Exp 5: two embedding models | optimization | New | | | | | 1.2 | | | | | | |
| 6.7 | Exp 6: semantic cache + tenant-scoped cache tests | optimization | New | | | | | 1.2 | | | | | | |
| 6.8 | Exp 7: HNSW ef_search sweep | optimization | New | | | | | 1.2 | | | | | | |
| 6.9 | Exp: rewriting, multi-query, HyDE | optimization | New | | | | | 1.2 | | | | | | |
| 6.10 | Exp: hierarchical, RAPTOR, GraphRAG, agentic results | optimization | New | | | | | 1.2 | | | | | | |
| 6.11 | Long context vs RAG experiment | optimization | New | | | | | 1.2 | | | | | | |
| 6.12 | History / context compression experiment | optimization | New | | | | | 1.2 | | | | | | |
| 6.13 | Per-tenant token budgets + alert + runaway-cost test | infra | Some | | | | | 1.0 | | | | | | |
| 6.14 | Feedback loop: widget thumbs into eval set | integration | New | | | | | 1.4 | | | | | | |
| 6.15 | Benchmark charts, docs/benchmarks.md, finalize ADRs 001-010 | docs-video | Some | | | | | 1.0 | | | | | | |
| | **Phase total** | | | | | | | | | | | | | |

## Phase 7: Security pass
| # | Task | Category | Familiarity | O | M | P | PERT | x mult | Planned h | Learning h | Build h | Actual total h | Ratio | Why it was off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7.1 | SPIKE (timebox 4 h): PortSwigger LLM labs, garak / promptfoo docs | learning | New | | | | | 1.0 | | | | | | |
| 7.2 | Red-team campaign run: attacks tried vs blocked | testing-evals | New | | | | | 1.1 | | | | | | |
| 7.3 | Fix top red-team findings | llm-logic | New | | | | | 1.0 | | | | | | |
| 7.4 | OWASP GenAI Top 10 mapping table (ADR 005) | docs-video | Some | | | | | 1.0 | | | | | | |
| 7.5 | Data-policy note + AI disclosure line | docs-video | Some | | | | | 1.0 | | | | | | |
| 7.6 | Secrets and personal-data scan of whole history | testing-evals | New | | | | | 1.1 | | | | | | |
| | **Phase total** | | | | | | | | | | | | | |

## Phase 8: Package
| # | Task | Category | Familiarity | O | M | P | PERT | x mult | Planned h | Learning h | Build h | Actual total h | Ratio | Why it was off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8.1 | README: impact line, tables, Mermaid, design decisions, limitations | docs-video | Some | | | | | 1.0 | | | | | | |
| 8.2 | DEMO_SCRIPT, LEARNING_LOG, docs/architecture.mmd | docs-video | Some | | | | | 1.0 | | | | | | |
| 8.3 | Full compose (api, worker, postgres, redis, ui) + make targets | infra | Some | | | | | 1.0 | | | | | | |
| 8.4 | CI: lint + tests + 15-question mocked eval | infra | Some | | | | | 1.0 | | | | | | |
| 8.5 | Record 4-minute demo video | docs-video | Some | | | | | 1.0 | | | | | | |
| 8.6 | Edit and publish video | docs-video | Some | | | | | 1.0 | | | | | | |
| 8.7 | Fresh-clone quick-start run (live run + fix) | integration | New | | | | | 1.4 | | | | | | |
| 8.8 | ESTIMATES.md project summary + tracker row | docs-video | Some | | | | | 1.0 | | | | | | |
| | **Phase total** | | | | | | | | | | | | | |


## Time log (IST; Claude records the start when the owner says go; the END only when the owner says the task is finished)
| Task | Start | End | Note |
|---|---|---|---|
| 1.3 | 20:30 (09-30) | n/a | owner reported ~0.5 h active |
| 1.4 | n/a | 2026-10-01 18:04 | end recorded when the code walkthrough was delivered; start not captured |
| 1.5 | 2026-10-01 18:04 | 2026-10-01 19:10 | end given by owner; overlaps 1.6 work done while the owner read 1.5 |
| 1.6 | 2026-10-01 18:57 | | owner said go while reading 1.5 |
| 1.7 | 2026-10-01 19:19 | | owner said go while reading 1.6 (1.6 end left open until the owner says) |
