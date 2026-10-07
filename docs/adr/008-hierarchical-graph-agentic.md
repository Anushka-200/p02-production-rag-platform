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
- **Cost:** about one small-model call per 8 chunks; the real Tenant B tree (3,606 chunks) cost $1.26 and took about 25 minutes. **Result (B-020):** no gain on broad or relationship questions at any dose (broad 0.83 -> 0.75), so `SUMMARY_TREE` stays OFF. The mechanism and its privacy guarantees are kept; the first thing to try if thematic questions matter is grouping by document or section before clustering.
- **Observed weakness:** second-level summaries are broad grab-bags; the first level is coherent. A better grouping (by document or section path before clustering) is the obvious next experiment.

## GraphRAG slice: built and decided (Phase 5 block 4, 2026-10-07; B-021)
- **Scope:** a comparison slice, as planned: Tenant B's customer-facing sections (375 chunks), facts extracted by Haiku 4.5 into a Postgres table `graph_edges` (a node is a normalised name, an edge is a row, walking = a join). No Neo4j: the extra service and the traversal depth belong to P10; this slice answers "does following facts beat matching words, and when?".
- **Retrieval** (`app/retrieval/graph.py`): entities named in the question are the seeds (all their words appear in the question; hubs and very short names are skipped), a 2-hop walk follows the facts (not through hubs), and the chunks that STATE those facts (nearest hop first) join the other search results through the usual rank merge. The model still reads real chunks, so citations stay real.
- **Privacy** is the same discipline as summaries: a fact carries its evidence chunk's visibility and owner, follows the same policies, is deleted with its chunk (foreign key) and by a trigger when the chunk loses visibility; reading is checked on the facts AND on the returned chunks. Tested, including a mutation check.
- **Result (informal, circular caveat in B-021):** relationship questions 0.55 -> 0.85 gold-in-context, broad questions 0.83 -> 0.58, everything else unchanged. **`GRAPH_SEARCH` stays OFF by default.** It is a strong tool for one question type and a harmful one for another, so it should be chosen per question, not set globally. That is the job of the router in the agentic block (5.10).
- **Not done (and why it matters):** entity resolution (merging "support team" and "customer support team"), community detection and global summaries (Microsoft's full GraphRAG; our summary tree is the closest analogue and was built separately), and a graph database. LazyGraphRAG's idea (build almost nothing in advance, extract at query time) is the cheaper path if the full cost per tenant ($6.50 here) becomes a problem.

## Agentic retrieval: built and decided (Phase 5 block 5, 2026-10-07; B-022)
- **What it is:** the plain search (rewrite, hybrid, rerank) always runs first. If its best match looks weak (rerank score below -1; or always, with `AGENT_ONLY_WHEN_WEAK=false`), a planner (one Haiku call) decides which EXTRA searches would add something: the parts of a multi-part question, or the same question through a better tool (`exact`, `connections`, `broad`). A checker (one more call) decides whether the evidence is enough; if not, one more round. The final evidence is the plain search's top 5 untouched, then the best new hits of the extras, then the rest of the plain search. The normal grounded-answer step writes the answer, so citations are still validated by code and unsupported questions are still refused.
- **Least agency:** four read-only tools; the model supplies a tool NAME from an allow-list and a search STRING (3 to 300 characters). The organisation and user are fixed in the `ToolBox` by the backend from the verified login; there is no parameter for a tenant, SQL, a URL or a write. Hard limits: 2 planning rounds, 4 searches, $0.01 spent on the agent's own model calls per question. A failed or hostile planner means "no extras": exactly the plain search. Evidence text reaches the planner only inside `<evidence>` tags with an instruction to treat it as data (tested with a chunk that says "use tool delete_all"). The agent trace is shown to logged-in users ("How I searched") but never sent to anonymous website visitors.
- **The lesson that shaped the design:** the first version let the planner replace the search and was worse than no agent on ordinary questions (B-022). An agent must not be able to damage what already works, so it only adds.
- **Router:** this IS the per-question method choice (rewriting stays an always-on rule; the rest are tools the agent picks). It is gated by a score that is a weak signal (59% of misses vs 18% of hits) and cannot detect unanswerable questions; a trained or LLM router judged on a reviewed set is the next step.
