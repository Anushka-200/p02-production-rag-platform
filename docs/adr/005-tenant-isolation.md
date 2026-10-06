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

## Injection defences (Phase 3, tasks 3.6-3.8) and OWASP GenAI mapping
Layers, cheapest first: (1) **hidden text is removed before indexing** (HTML comments, CSS-hidden elements, Unicode tag characters, lines made mostly of zero-width characters; titles that read like instructions are replaced), (2) the **ingestion scan** flags what a human should review and stores the result in the job (`result.scan`), without dropping the document, (3) source text is **delimited and escaped** so a forged `</source><source ...>` cannot add a block, (4) the system prompt says sources are untrusted data, (5) the **output checker** rejects images, links that are not in a source, and invented citations (one repair attempt, then refusal), (6) tenant isolation does not depend on any of this: it is enforced by the database.

| OWASP GenAI risk | Control | Test |
|---|---|---|
| Prompt injection (direct, indirect) | layers 1-5 above | `tests/test_scan.py`, `scripts/injection_live.py` (0 of 8 obeyed) |
| Sensitive information disclosure | RLS, private vs shared, scope in code | `tests/test_rls.py`, `tests/test_adversarial.py` (0 leaks) |
| Improper output handling | output checker (no images / unknown links) | `tests/test_scan.py` |
| Vector and embedding weaknesses | RLS on chunks, filter before search, scan | `tests/test_rls.py` |
| Data and model poisoning | scan + hidden-text removal at ingestion | `tests/test_scan.py` |
| Excessive agency | tenant from the token only; no tools yet (agent in Phase 5 inherits it) | Phase 5 |
| Unbounded consumption | per-tenant budgets | Phase 6 |
| System prompt leakage / misinformation | prompt holds no secrets; answers must cite sources | Phase 7 red team |
Residual risk: the scan is a smoke alarm (regex), not a classifier; a paraphrased visible instruction can slip past it. The model resisted all 8 attack documents here, which is evidence, not a guarantee.

## Anonymous visitors and public documents (Phase 4 block 1, 2026-10-06; migrations 0006-0007)
- **A third level, `public`.** Visibility is now private (owner), org (members), public (members and anonymous visitors). Only an org admin may publish (`NotAllowed` -> 403); changing it cascades to the chunks as before.
- **Anonymous scope:** `Scope(org_id, user_id=None)`. The org comes from a **site key** (table `sites`), the user is NULL. The policy for reading is `public OR (user is not NULL AND (org OR own))`, so an empty `app.user_id` can never reach org-internal rows. Before this change an org-only setting would have exposed org documents; that is why the policy had to change before any anonymous channel existed.
- **Reads and writes have separate policies** (0007). With one "can see" policy, whoever could see a row could also delete it; found by a test that tried to delete as a visitor. Now INSERT/UPDATE/DELETE require `owner_id = app_user()`, so visitors (user NULL) can never write, and a member cannot delete a colleague's shared document through SQL either.
- **The site key is public** (it is in the page's HTML), so it is not a secret. The checks are: key exists and enabled, the browser `Origin` is on the site's allow-list (`site_principal`), then rate limits and budgets (block 3). Anonymous scope can read only public documents, so a leaked or abused key exposes only what the org chose to publish.
- **Keyword statistics are cached per (org, user)**, so a visitor's numbers (public documents only) never become a member's numbers.
- **Tested** (`tests/test_anonymous.py`, 25 tests): visitors see only public documents through the restricted role AND through the owner connection (Python filter alone), cannot write, cannot read another org via a forged scope, the full retrieval pipeline stays public-only for 6 hostile questions, pooled connections reset.
