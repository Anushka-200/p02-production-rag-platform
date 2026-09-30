# ADR 004: Memory design

Status: proposed (2026-09-30)

## Context
Chat history, long-term user memory and RAG documents have different scope, lifetime and deletion needs.

## Options
Store facts in the chunk index; separate memories table; external memory service.

## Decision
Separate memories table with RLS by user; LLM extraction after each turn; Presidio redaction; view/delete API; injected as a labelled block.

## Consequences
TBD: extraction precision, wrong-fact handling.
