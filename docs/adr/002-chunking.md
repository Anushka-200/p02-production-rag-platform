# ADR 002: Chunking strategy

Status: accepted (2026-10-08): structured 400/50 kept; +8 points R@5 over recursive, size and overlap barely matter (B-003, B-027)

## Context
Chunking is fixed at ingestion; a wrong choice means re-embedding everything.

## Options
A recursive by tokens (with overlap); B heading/structure-aware; C parent-child; D semantic.

## Decision
Ship A and B, configurable. Choose by Recall@5 on the eval (size/overlap sweep). Parent-child covered in ADR 008.

## Build vs reuse
We wrote chunkers A and B ourselves (tables, page/anchor metadata, parent-child, ablation flags). A comparison against LangChain's splitters on the same slice is planned in task 5.16; results go here.

## Finding from the first live run (2026-10-01, task 1.12)
On a flat page (all `##` sections are siblings, e.g. GitLab's Modern Health page) chunker B produced 8 children and no parents:
each `##` is its own topic branch, so nothing merges and a 44-token section stays a 44-token chunk. B only merges where there
are sub-sections or a title-level intro. The `merge_across_branches=True` variant (median 306 tokens, 0.9% under 50 tokens on
the Tenant A slice, vs 279 and 1.9%) must be part of experiment 1 before B's default is final.

## Consequences
TBD after experiments 1-2.
