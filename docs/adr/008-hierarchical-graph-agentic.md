# ADR 008: Hierarchical, GraphRAG and agentic RAG

Status: proposed (2026-09-30)

## Context
Some questions need surrounding context, whole-corpus themes, relationships or several searches.

## Options
Parent-child; RAPTOR summary tree; GraphRAG slice; agent loop with step cap.

## Decision
Build all four as flags. GraphRAG is a comparison slice only (one handbook section, ~20 relationship questions); depth belongs to P10. Agent has step and token caps; tools take the org from the backend.

## Consequences
TBD after experiments.

## Hierarchical retrieval: what was built and decided (Phase 5 block 3, 2026-10-07; B-020)
- **Parent expansion** (`app/retrieval/parents.py`): a hit is swapped for its parent section (capped at 700 tokens, one source per parent, read under the asker's own visibility). Measured: no gain in gold-in-context (0.828 either way) or correctness (0.774 either way); 8 questions flipped in both directions. **Off by default** (`PARENT_EXPANSION`). A parent cannot find anything new, only show more around what was found.
- **Summary tree** (`app/ingest/summarytree.py`, `app/retrieval/tree.py`, migration 0010): summaries are stored as chunks of a synthetic document per visibility group (levels 2 and 3), linked to their members in `summary_members`. Search finds a summary, then **walks down to the real chunks under it**; a summary is never shown to the answer model or cited, so citations still point at real pages and exact figures are not lost to summarising.
- **Privacy is enforced by the database, not by code that could forget:** (1) a summary is only built from chunks of one visibility group (a private user's chunks never enter an org summary); (2) triggers delete every summary above a source chunk when that chunk is deleted, replaced by a re-index, or loses visibility (`purge_summaries_containing`, SECURITY DEFINER so a member's re-index can remove a summary owned by someone else); (3) the same read/write policies apply to summary rows; (4) visibility is re-checked on the chunks under a summary.
- **Staleness:** a tree is a snapshot. Any change to a source removes the affected summaries at once (safe), but they are only rebuilt by `scripts/build_summary_tree.py`; unchanged groups are skipped. Incremental refresh belongs to block 6 (connectors and sync).
- **Cost:** about one small-model call per 8 chunks (about $1.65 per 3,500-chunk tenant with Haiku 4.5), plus embeddings. **Open:** retrieval benefit on broad questions is not yet measured on the full corpus (approval of about $3.3 pending); `SUMMARY_TREE` stays off until it is.
- **Observed weakness:** second-level summaries are broad grab-bags; the first level is coherent. A better grouping (by document or section path before clustering) is the obvious next experiment.
