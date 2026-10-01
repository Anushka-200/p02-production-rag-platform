# ADR 002: Chunking strategy

Status: proposed (2026-09-30)

## Context
Chunking is fixed at ingestion; a wrong choice means re-embedding everything.

## Options
A recursive by tokens (with overlap); B heading/structure-aware; C parent-child; D semantic.

## Decision
Ship A and B, configurable. Choose by Recall@5 on the eval (size/overlap sweep). Parent-child covered in ADR 008.

## Build vs reuse
We wrote chunkers A and B ourselves (tables, page/anchor metadata, parent-child, ablation flags). A comparison against LangChain's splitters on the same slice is planned in task 5.16; results go here.

## Consequences
TBD after experiments 1-2.
