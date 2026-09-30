# ADR 006: LLM and embedding models

Status: proposed (2026-09-30)

## Context
Cheap default tier via LiteLLM; Anthropic has no embedding model.

## Options
Haiku 4.5 answers/rewrites/blurbs/memory, Sonnet 5.5 judge; local open embedding vs one hosted embedding; all-OpenAI.

## Decision
Haiku 4.5 + Sonnet 5.5 judge; local embedding as default, one hosted model for the quality-vs-cost experiment. Verify current model names and prices when finalising.

## Consequences
TBD after experiment 5.
