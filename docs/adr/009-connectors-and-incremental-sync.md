# ADR 009: Connectors and incremental sync

Status: accepted, implemented in Phase 5 (2026-10-08): fingerprint sync re-embeds only changed pages (B-023, B-024)

## Context
Wikis change daily; re-indexing everything is wasteful and stale answers are wrong answers.

## Options
Full re-index on schedule; per-page hash diff; webhook-driven.

## Decision
Connector interface (fetch, modified time, permissions); per-page/section content hash; re-index only changes; delete removed pages; sync permissions; show last-updated in citations. Demo by replaying GitLab handbook git history.

## Consequences
TBD.
