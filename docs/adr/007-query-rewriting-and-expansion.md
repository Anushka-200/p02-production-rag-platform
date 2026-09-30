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
