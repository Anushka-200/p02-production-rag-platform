# ADR 012: Semantic cache

Status: accepted (2026-10-08), off by default (`SEMANTIC_CACHE=true` to enable)

## Context
Many questions repeat. A cache hit skips retrieval, reranking and the model call: about 10 ms and $0 instead of
about 4 s and 0.35 cents. But in a multi-tenant system a cache is the most likely place for a leak: it stores
ANSWERS, which can contain private knowledge, and it is looked up by a fuzzy match.

## Options
- No cache: safest, slowest, most expensive.
- Cache keyed by exact question text: safe, but almost never hits (people word things differently).
- Semantic cache (embedding similarity) scoped by tenant and user, revalidated against the documents: chosen.
- A global or per-org shared cache with no per-user rule: rejected, leaks private-document answers to colleagues.

## Decision
- Table `semantic_cache` (migration 0013) with row-level security: a row is readable by its owner, and by
  other members of the SAME org only if `shareable` (every document the model saw was non-private).
- On every lookup, re-check that each source document is still ready, still the same active version, and
  still visible to the asker. A re-indexed, deleted or newly private document makes the entry stale.
- Only signed-in users, only standalone questions: never inside a conversation (a follow-up depends on the
  chat), never when personal memory notes were added to the prompt, never for refusals, never for anonymous
  visitors (widget, WhatsApp).
- Threshold 0.93 cosine, chosen from data (B-028): the nearest wrong pair in our eval set is 0.874.
- 24-hour expiry and a cap of 2,000 entries per org (oldest dropped).

## Consequences
- Hits are real answers served in about 10 ms with no model cost. On our eval set only exact and near-exact
  repeats pass the safe threshold (1 of 24 paraphrases), so the realistic saving is for repeated questions.
- A conversation turn never benefits from the cache, even when it is a complete question. Next step: key
  conversation turns by the rewritten standalone question.
- Cached answers do not reflect a changed prompt or model: they expire after 24 hours, and the cache should be
  cleared when the answer prompt changes.
- Tests prove: no cross-tenant hit, no cross-user hit for private sources, staleness on re-index/delete/visibility
  change, follow-ups and refusals excluded, row-level security as a second lock.
