# ADR 007: Query rewriting and expansion

Status: proposed (2026-09-30)

## Context
Follow-ups are not standalone; vocabulary mismatch hurts recall.

## Options
None; rewrite with history; multi-query; HyDE.

## Decision
Rewrite follow-ups with a small model; multi-query and HyDE behind flags, kept only if they raise Recall per added latency/cost.

## Consequences
TBD. Cache keys use the rewritten question.

## Result and decision (Phase 5 block 2, 2026-10-07; benchmark B-019)
- **Query rewriting: ON by default.** It runs only when the conversation has history, so a first question costs nothing extra. Follow-up questions went from Recall@5 0.00 to 0.60 (6 of 10); no other group changed. Cost: one small-model call (about $0.0005, 0.3 s) per follow-up. The rewrite is used ONLY as search input and as the question the answer model sees; it is never shown as an instruction, and an unusable rewrite (empty, multi-line, an essay) falls back to the original question.
- **Multi-query and HyDE: implemented, OFF by default** (`MULTI_QUERY=0`, `HYDE=false`). On the 24-question paraphrase group neither helped (0.67 to 0.62) and they add 1.7 to 2.4 s at the median. Retest on a human-reviewed, larger set in task 6.9; consider using them only as a fallback when the first search looks weak (low rerank score), which is the "corrective retrieval" idea the agent in block 5 will own.
- **Where history comes from:** the member chat reads the last turns of the stored conversation (before the new question is saved). The widget and WhatsApp do not pass history yet, so rewriting does not apply to them.
- **Known limit:** when the history does not name what "they" or "this" means, no rewrite can recover it (3 of the 4 remaining misses).
