# ADR 010: Framework (LangChain) vs plain Python for the RAG pipeline

Status: accepted (2026-10-07), spike measured

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

## Spike result (2026-10-07, B-025)
- **Chunking:** LangChain's recursive and header splitters gave the same retrieval as ours on the Tenant A slice
  (hit@5 0.93-1.00, 14 questions) but produce no page numbers, anchors or parent levels, which citations and
  parent expansion need. Keep ours.
- **Orchestration:** the LangGraph version of the main flow matched plain Python (same cited documents 8/8, same
  latency within noise) and is about the same length. It adds a graph/state abstraction, but no new capability for
  a linear flow. Its real value would be for branching flows (our agent loop), which already work in plain code.
  The tenant scope must travel in graph state; keep it set by the caller, not by a node.
- **Decision stands:** plain Python core, libraries for single jobs. A client who asks for LangChain can be offered
  LangGraph around these functions, as the spike shows, in about an hour.
