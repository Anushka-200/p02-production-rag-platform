# ADR 005: Tenant isolation

Status: proposed (2026-09-30)

## Context
Model output must never decide tenant; a forgotten WHERE must not leak.

## Options
App-level filter/namespace; Postgres RLS; database per tenant.

## Decision
RLS with FORCE, non-bypass roles, org_id/user_id set from the verified JWT per request; WITH CHECK on writes; composite FKs. OWASP GenAI Top 10 (2026) mapping table lives here.

## Consequences
TBD: adversarial suite results, filtered-HNSW behaviour.
