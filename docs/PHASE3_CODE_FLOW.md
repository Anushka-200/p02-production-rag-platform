# Phase 3 code flow: who is asking, what they may see, what we remember

Written 2026-10-06 for catching up. Every step names the file and the function. Read it top to bottom:
it follows one user from "an account exists" to "a remembered fact is deleted".

## Map of what Phase 3 added

| Area | Files |
|---|---|
| Database rules | `migrations/0004_rls.sql` (row-level security, login function), `migrations/0005_chat_memory.sql` (chats, memories) |
| Who is asking | `app/api/security.py`, `app/api/auth.py`, `app/api/main.py` (login + dependencies) |
| Telling the database | `app/db/tenant.py` |
| Documents and sharing | `app/documents.py`, `app/api/main.py` (upload, list, visibility) |
| Hidden-instruction defences | `app/ingest/scan.py`, `app/ingest/parsers.py`, `app/ingest/preprocess.py`, `app/answer.py` (`unsafe_output`), `app/retrieval/context.py` (escaping) |
| Chat history | `app/chat_history.py` |
| Memory | `app/memory.py` |
| Personal-data redaction | `app/pii.py`, `app/log.py`, `app/tracing.py` |
| Worker | `app/worker.py` (tenant per job) |
| Scripts | `scripts/create_user.py`, `scripts/two_tenant_demo.py`, `scripts/injection_live.py`, `scripts/memory_live.py` |
| Attack data and checks | `evals/adversarial/` (8 attack documents), `evals/pii_check.py` |

## A. One-time setup

1. `docker/postgres/init/01-roles.sh` (Phase 0) creates three login roles: `rag_owner` (runs migrations, owns tables), `rag_app` (the API) and `rag_worker` (ingestion). `rag_app` and `rag_worker` are NOT owners and have NOBYPASSRLS, so row-level security applies to them.
2. `app/db/migrate.py: apply_migrations` runs `migrations/0004_rls.sql` as the owner:
   - `app_org()` / `app_user()`: read the settings `app.org_id` / `app.user_id`; NULL when missing.
   - `ENABLE ROW LEVEL SECURITY` + a policy on `documents`, `chunks`, `ingestion_jobs`: a row is visible when `org_id = app_org()` and (`visibility = 'org'` or `owner_id = app_user()`). `WITH CHECK` limits writes to your own org and owner.
   - `users.password_hash` column; the app roles cannot read it and cannot insert/update/delete `users` or `orgs`.
   - `login_lookup(email)`: a `SECURITY DEFINER` function, the only way the API reads a password hash.
3. `migrations/0005_chat_memory.sql`: tables `conversations`, `messages`, `memories`, each with a policy `org_id = app_org() AND user_id = app_user()` (private to the USER, not just the org).
4. `scripts/create_user.py: create_user` (run as the owner): hashes the password with `app/api/security.py: hash_password` and inserts the org and user. The API cannot create accounts.

## B. API start

5. `app/api/main.py: create_app` builds the app. In its `lifespan`, `app/db/tenant.py: check_role_is_restricted` connects as `rag_app` and refuses to start if that role is a superuser, has BYPASSRLS or owns `chunks`. Then the models are loaded and warmed up (`_warm_up`).
6. `app/log.py: setup_logging` installs `JsonFormatter`, which redacts personal data in every log line (see section I).

## C. Login: `POST /auth/login`

7. `app/api/main.py: login`. The body is checked by `LoginRequest` (length limits).
8. `login` runs `SELECT * FROM login_lookup(%s)` with the email as a parameter (never pasted into the SQL). The function returns `(id, org_id, role, password_hash)` for each user with that email.
9. For each row, `app/api/security.py: verify_password` re-hashes the typed password with the stored salt (scrypt) and compares in constant time. Unknown email: the same work is done against `_DUMMY_HASH`, so the answer and the time look the same.
10. Exactly one match is required (the same email can exist in two orgs). Zero or several: 401 "wrong email or password".
11. `app/api/security.py: create_token` signs a JWT (HS256) with claims `sub` (user id), `org`, `role`, `iss`, `iat`, `exp` (60 minutes) using `JWT_SECRET`, which must be 32+ characters (`_secret`).

## D. Every later request: who is asking, and what the database may show

12. FastAPI calls `principal` (in `create_app`) which calls `app/api/auth.py: current_principal`. A bearer token wins; the `X-Dev-User` header (`dev_principal`) works only when `ENVIRONMENT=dev`.
13. `token_principal` calls `app/api/security.py: read_token`: pinned algorithm HS256, issuer, expiry, required claims. Any failure is a generic 401.
14. `token_principal` then loads the user from `users` by id (so a deleted user stops working at once), requires the token's org to equal the user's real org, and takes the role from the database (the token's role claim is ignored). Result: `Principal(email, role, Scope(org_id, user_id))`.
15. The endpoint opens `app/db/tenant.py: tenant_connection(open_connection, who.scope)`:
    - `conn.rollback()` then `autocommit = False`,
    - `SELECT set_config('app.org_id', ..., true), set_config('app.user_id', ..., true)` (transaction-local),
    - the endpoint's SQL runs; every query on `chunks`, `documents`, `ingestion_jobs`, `conversations`, `messages`, `memories` is filtered by the policies,
    - at the end `rollback()` (read-only) or `commit()` (`write=True`) drops the settings before the connection is reused. That is the pooled-connection safety.
16. The Python search code also filters by scope (`app/retrieval/search.py: dense_search`, `keyword_search`; `app/documents.py: list_documents`). Two independent locks.

## E. Upload and ingestion: `POST /documents`

17. `app/api/main.py: upload_document` reads at most `max_upload_mb + 1 byte` of the file; the owner is the logged-in user and the default visibility is `private`.
18. Inside `tenant_connection(..., write=True)` it calls `app/ingest/upload.py: enqueue_upload`:
    - `validate_upload`: allowed type, non-empty, size, magic bytes,
    - `save_raw`: stores the file by content hash,
    - inserts an `ingestion_jobs` row (idempotency key = owner + path + content hash; the same file twice is one job),
    - `conn.commit()` so the worker can see the row, then `queue.enqueue("app.worker.process_job", job_id, org_id, owner_id)`.
19. The worker (`app/worker.py: main`, RQ) runs `process_job(job_id, org_id, owner_id)`:
    - `prepare_worker_connection`: autocommit first, then `app/db/tenant.py: set_session_tenant` (the worker could not even read its own job row before saying whose it is),
    - `run_job` marks the job running and calls `app/ingest/pipeline.py: run_pipeline`.
20. `run_pipeline` steps, in order:
    1. `plan_ingest` (skip if unchanged or a duplicate),
    2. `app/ingest/parsers.py: parse` for the file type. Hidden text is removed here: `_strip_markup` / `_drop_comments` (HTML comments in Markdown), `parse_html` (elements with `display:none`, `visibility:hidden`, `font-size:0`, white text, or the `hidden` attribute),
    3. `app/ingest/preprocess.py: preprocess`: `neutralise_title` (a title that reads like an instruction becomes "Untitled document"), `clean_text` (drops lines that are mostly invisible characters, deletes Unicode tag characters; a payload cannot be "assembled" by stripping zero-width characters), PDF furniture and duplicate-section removal,
    4. `pipeline.py: scan_summary` calls `app/ingest/scan.py: scan_text` on the RAW file and the original title: instruction phrases, hidden comments, invisible characters, hidden CSS, forged `<source>` tags, data-carrying image URLs. The result (`flagged`, findings) is stored in the job result and logged; the document is not dropped,
    5. chunk, optional blurbs, embed, `write_version` (all rows carry org, owner, visibility).
21. `GET /jobs/{id}` (`job_status`) returns the job only to its owner (same-org colleagues get 404).

## F. Sharing: `PATCH /documents/{id}/visibility` and `GET /documents`

22. `app/api/main.py: document_visibility` -> `app/documents.py: set_visibility`: one `UPDATE documents ... WHERE id = %s AND owner_id = %s`. The foreign key from `chunks` is `ON UPDATE CASCADE`, so every chunk gets the new visibility in the same transaction. Not found, someone else's, or hidden by RLS all give the same 404.
23. `list_documents` returns what the user may see (RLS and a Python filter).

## G. One chat turn: `POST /chat`

24. `app/api/main.py: chat`, step 1 (one short write transaction, `tenant_connection(write=True)`):
    1. `app/chat_history.py: conversation_exists` or `create_conversation` (the title is the first question, 60 characters),
    2. `add_message` stores the user's message,
    3. `app/memory.py: relevant_memories` embeds the question and returns up to 3 of this user's facts with similarity >= 0.60,
    4. `format_memory_block` wraps them in `<user_notes>` (labelled unverified, never to be cited; `<` is escaped).
25. Step 2, `events()` (a read-only `tenant_connection`): `app/rag.py: ask_stream`
    1. `app/retrieval/retrieve.py: retrieve` (dense + keyword + RRF; filtered HNSW with iterative scan),
    2. `app/retrieval/rerank.py: rerank` (10 candidates, B-014),
    3. `app/retrieval/context.py: build_context` (token budget; `_safe` escapes any `<source` inside chunk text so a forged tag cannot add a source block),
    4. `app/answer.py: answer_stream` builds the prompt with `_user_prompt(context, question, memory_block)` and streams tokens (SSE `sources`, `token`, `done`),
    5. at the end the text is validated: every `[S#]` must exist and `unsafe_output` rejects images and links that are not in a source; if invalid, `answer_question` / `_check` produces a corrected answer (one repair attempt, then refusal).
26. Step 3: on the `done` event the assistant message and its citations are saved (`add_message`, citations JSON), a failure here is logged and does not lose the answer; the payload gets `conversation_id`.
27. Step 4, after the response is sent (Starlette `BackgroundTask` -> `remember` in `create_app`), only if `memory_enabled`:
    1. `app/chat_history.py: recent_messages` (last turns that fit the token budget, always starting with a user message),
    2. `app/memory.py: list_memories` (known facts),
    3. `extract_facts`: a Haiku call that sees only the USER's messages and the known facts and returns JSON; each fact passes `acceptable` (size; `scan_text` rejects instruction-like facts = memory poisoning; `app/pii.py: redact` rejects facts that contain personal data); max 5 per turn; any failure means "nothing learned",
    4. `store_facts`: embeds each fact, finds the user's nearest memory; similarity >= 0.90 replaces its wording, otherwise inserts; keeps at most 200 per user (oldest-updated dropped).

## H. Viewing and deleting your data

28. `GET /conversations`, `GET /conversations/{id}`, `DELETE /conversations/{id}` (`app/chat_history.py: list_conversations`, `get_messages`, `delete_conversation`; messages are deleted by cascade).
29. `GET /memories`, `DELETE /memories/{id}`, `DELETE /memories` (`app/memory.py: list_memories`, `delete_memory`, `delete_all_memories`). All are private to the user by RLS and by an explicit filter.

## I. Logs and traces

30. `app/log.py: JsonFormatter.format` passes every field (except `ts`, `level`, `logger`) through `app/pii.py: redact_any`; `redact` runs Presidio pattern recognisers (email, phone, card, IBAN, IP, SSN, ...) only if the text contains a digit or `@`, and skips our own UUIDs. A re-entrancy guard (`_busy`) stops the redactor's own log lines from calling it again.
31. `app/tracing.py: _mask` does the same for every Langfuse input/output before upload.

## J. Which test proves which step

| Step | Test file |
|---|---|
| RLS policies, pool safety, startup role check, user-private chats and memories | `tests/test_rls.py` |
| Passwords, tokens, login, duplicate emails | `tests/test_auth.py` |
| Upload API, job visibility | `tests/test_upload_api.py` |
| Scan, hidden-text removal, output checker | `tests/test_scan.py` |
| 47 attacks, 0 leaks | `tests/test_adversarial.py` (`make adversarial`) |
| History window, privacy | `tests/test_chat_history.py` |
| Extraction filters, merge, cap, relevance | `tests/test_memory.py` |
| Chat + memory over HTTP | `tests/test_chat_memory_api.py` |
| Redaction, logs, traces | `tests/test_pii.py` |
| Worker connection order (regression) | `tests/test_worker.py` |
| Everything live, 22 checks | `scripts/two_tenant_demo.py` |

## K. Known gaps (carried to the next phases)

- No admin "invite user / reset password" flow; accounts are created by `scripts/create_user.py`.
- RLS stops forgotten filters; it cannot stop a full SQL injection (the tenant is a session setting). All SQL uses placeholders.
- The scan is regex-based (smoke alarm). Paraphrased visible instructions can pass it.
- Memory: a correction below 0.90 similarity leaves the stale fact; style preferences are only used when similar to the question.
- History is stored but not yet used in prompts (query rewriting is Phase 5).
- Login has no rate limit or lockout yet (Phase 7).
