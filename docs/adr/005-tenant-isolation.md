# ADR 005: Tenant isolation

Status: proposed (2026-09-30)

## Context
Model output must never decide tenant; a forgotten WHERE must not leak.

## Options
App-level filter/namespace; Postgres RLS; database per tenant.

## Decision
RLS with FORCE, non-bypass roles, org_id/user_id set from the verified JWT per request; WITH CHECK on writes; composite FKs. Settings are transaction-scoped (SET LOCAL) and policies fail closed when unset. Anonymous channels (widget, WhatsApp) see only visibility = 'public'. OWASP GenAI Top 10 (2026) mapping table lives here.

## Consequences
TBD: adversarial suite results, filtered-HNSW behaviour.

## Implementation notes (Phase 3, 2026-10-02; migration 0004, `app/db/tenant.py`, `app/api/auth.py`)
- **Policies** on `documents`, `chunks`, `ingestion_jobs`: `org_id = app_org() AND (visibility = 'org' OR owner_id = app_user())`, plus `WITH CHECK` on writes. Jobs are visible to their owner only.
- **Fail closed:** `app_org()`/`app_user()` return NULL when the setting is missing, so no rows match. Measured on the real corpus: no tenant set = 0 of 8641 chunks; tenant A = 4262, tenant B = 4379 (they add up, no overlap).
- **FORCE ROW LEVEL SECURITY is NOT used.** It would also bind the table owner, which only runs migrations and tests; the policies would then need an owner exemption, which cancels the point. Instead the API and worker are non-owner, NOBYPASSRLS roles, and the API **refuses to start** if its role is a superuser, has BYPASSRLS or owns the tables (`check_role_is_restricted`).
- **Per-request tenant:** `set_config('app.org_id', ..., true)` (transaction-local) inside a transaction that is always rolled back, so a reused pooled connection cannot keep the previous tenant (tested). The worker sets session-level settings because it opens and closes its own connection per job; the tenant travels in the queue payload.
- **Login:** `orgs`/`users` have no RLS (the tenant is unknown at login). The app and worker roles can only SELECT non-secret columns; password hashes are reachable only through the SECURITY DEFINER function `login_lookup(email)`. JWTs (HS256, pinned algorithm, issuer, expiry) are re-checked against the `users` table on every request, and the token's org must equal the user's real org.
- **Known gap:** the keyword document-frequency cache is keyed by org but computed from the rows visible to the first caller (statistics only, no text). Revisit with private documents in block 3.
