# ADR 010: Framework (LangChain) vs plain Python for the RAG pipeline

Status: proposed (2026-09-30)

## Context
Many client posts ask for LangChain or LlamaIndex. This project needs full control of tenant filtering,
per-candidate scores (retrieval inspector), versioned re-indexing and SQL-level RRF. The shared standard says
"no magic". Libraries for single jobs (LiteLLM, PyMuPDF, pgvector, Presidio) are still used.

## Options
- LangChain/LlamaIndex owning the pipeline: fast start, many connectors; hides intermediate scores, adds an
  abstraction where the tenant filter could be skipped, API churn.
- Plain Python with small libraries: more code, every step visible and testable.
- Hybrid: plain Python core, framework only where it saves real time.

## Decision
Plain Python with small libraries for the core pipeline. A 1-2 h spike in Phase 5 re-implements the
retrieve-and-answer chain in LangChain/LangGraph and compares lines of code, visibility of intermediate
scores, tenant-filter placement and latency on the same eval subset. The result is recorded in
docs/benchmarks.md and this ADR is updated.

## Consequences
More code to write and test ourselves. A measured answer for clients who ask "why not LangChain?".
Revisit if the spike shows a clear win.
