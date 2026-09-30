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
