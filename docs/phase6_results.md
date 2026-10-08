# Phase 6 results: every experiment, what changed, what we decided

All numbers are from `docs/benchmarks.md` (B-numbers in the last column). **Caveat for every row:** the eval set is drafted by an LLM from
the documents and is **not human-reviewed** (`reviewed: false`), the samples are small (one question is 1 to 12 points depending on the
group), and most retrieval numbers are on Tenant A's 87 non-follow-up questions. Treat differences under about 5 points as noise.
"Default" is what the code does today.

| # | Experiment | Before -> after | Cost / latency | Decision (default) | Ref |
|---|---|---|---|---|---|
| 6.1 | Baseline | R@5 0.83 (hybrid + rerank), faithfulness 0.95, correctness 0.77 | $0.0035 per answer, p50 3.98 s, p95 4.98 s | measured | B-026 |
| 6.4 | Dense vs keyword vs hybrid vs +rerank | R@5 0.76 / 0.33 / 0.76 / **0.83**; R@1 0.46 -> 0.54 with rerank | rerank about 1 s, $0 (local) | hybrid + rerank ON; hybrid alone did not beat dense | B-026 |
| 6.8 | HNSW `ef_search` | top-10 identical to exact search at every setting 10 to 400 | SQL p50 about 13 ms flat | keep default 40 | B-026 |
| 6.2 | Chunker A vs B | R@5 0.67 -> **0.75** (R@10 0.75 -> 0.89) | none | structured chunker ON | B-027 |
| 6.3 | Chunk size and overlap | R@5 0.69 (200), 0.73 (300), 0.75 (400), 0.79 (500); overlap 0/50/100 no change | 500 is near the 512-token embedding limit | keep 400/50 | B-027 |
| 6.5 | Contextual retrieval (Haiku blurbs) | R@5 0.75 -> 0.79, R@1 0.42 -> 0.46 (3 to 4 questions) | about $7 per 3,500 chunks, 80 min | OFF: small gain, not worth the cost here | B-032 |
| 6.6 | Embedding model | R@5 0.75 (bge-small) / 0.79 (bge-base) / 0.68 (nomic) / **0.81 (arctic-m)** | arctic 1.8x slower to embed, 2x vector size | recommend arctic-m; not switched, re-check with rerank | B-033 |
| 6.7 | Semantic cache | threshold 0.93: 0 wrong hits (first wrong pair at 0.874) | hit: 19 ms and $0 vs about 4 s and 0.35 cents | built, OFF by default (`SEMANTIC_CACHE`) | B-028, ADR 012 |
| 6.9 | Query rewriting | follow-up R@5 0.00 -> **0.60** | about $0.0005 only on follow-ups | ON | B-019 |
| 6.9 | Multi-query, HyDE | paraphrase R@5 0.67 -> 0.62; single 0.85 -> 0.85/0.89 | +1.7 to 2.4 s | OFF: no demonstrable benefit | B-019 |
| 6.10 | Parent expansion | correctness 0.774 -> 0.774, gold in context unchanged | none | built, OFF | B-020 |
| 6.10 | Summary tree (RAPTOR-style) | broad 0.83 -> 0.75, relationship 0.55 -> 0.40 | $1.26 to build Tenant B | built, OFF: did not help | B-020 |
| 6.10 | GraphRAG slice | relationship 0.55 -> **0.85**, but broad 0.83 -> 0.58 | $0.66 for 375 chunks | built, OFF: helps one question type, hurts another | B-021 |
| 6.10 | Agentic retrieval (plain first, agent adds) | single 0.87 -> 0.90, paraphrase 0.67 -> 0.71, broad 0.83 -> 0.92 | about $0.003 always-on; gated: +0.5 s on ordinary questions | built, OFF; gated when on | B-022 |
| 6.11 | Long context vs RAG | correctness 0.91 (RAG) vs 0.35 strict / 0.62 lenient (150k-token prompt) | $0.0033 vs $0.0597, 4.2 s vs 46.8 s | RAG stays | B-030 |
| 6.12 | History compression | recent reference 0.60 -> 0.70 (one question); 4 turns back: 0.00 / 0.20 (10-turn window) / 0.10 (summary) | summary costs 3x the rewrite | keep the 3-turn window | B-031 |
| 6.13 | Per-tenant budget | runaway loop of 50 requests stopped after 3; other tenant unaffected | $0.10 would have been spent without it | ON ($5 per org per day, 0 = off) | B-029 |
| 6.14 | Feedback loop | thumbs -> redacted candidate questions, no gold until a person adds it | none | built; a person promotes items | tests |

## Reading the whole table
- **What moved the numbers:** the heading-aware chunker (+8 R@5), the reranker (+7 R@5), query rewriting for follow-ups (0 -> 0.60),
  and, still unapplied, the embedding model (+6 R@5). Everything else was inside the noise or helped one question type while hurting another.
- **What cost money for little gain:** contextual blurbs (about $7), the summary tree ($1.26 per tenant), the long-context prompt (18x per answer).
- **What was a negative result and why it matters:** hybrid not beating dense, multi-query/HyDE, parent expansion, the tree and long context
  all fail to beat the simple pipeline on these questions. Each is built and tested, so a client corpus where it helps (exact IDs, long
  documents, relationship questions) can switch it on and re-measure.
- **Open items:** review the eval set (`reviewed: true`), confirm arctic-m with the reranker, and analyse why R@5 is not higher
  (about 6% of questions never find the gold page in the top 50).
