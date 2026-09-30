# ADR 002: Chunking strategy

Status: proposed (2026-09-30)

## Context
Chunking is fixed at ingestion; a wrong choice means re-embedding everything.

## Options
A recursive by tokens (with overlap); B heading/structure-aware; C parent-child; D semantic.

## Decision
Ship A and B, configurable. Choose by Recall@5 on the eval (size/overlap sweep). Parent-child covered in ADR 008.

## Consequences
TBD after experiments 1-2.
