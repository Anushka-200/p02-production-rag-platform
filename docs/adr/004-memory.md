# ADR 004: Memory design

Status: accepted, implemented in Phase 3 (2026-10-02)

## Context
Chat history, long-term user memory and RAG documents have different scope, lifetime and deletion needs.

## Options
Store facts in the chunk index; separate memories table; external memory service.

## Decision
Separate memories table with RLS by user; LLM extraction after each turn; Presidio redaction; view/delete API; injected as a labelled block.

## Consequences
TBD: extraction precision, wrong-fact handling.

## Implementation notes (Phase 3, 2026-10-02; `app/memory.py`, `app/chat_history.py`, `app/pii.py`, migration 0005)
- **Three stores:** documents (company knowledge, cited), `conversations`/`messages` (transcript, windowed into prompts), `memories` (one-line facts). Chats and memories are private to the USER: RLS policy `org_id = app_org() AND user_id = app_user()`, so even an org admin cannot read a colleague's chat.
- **Extraction:** after the answer is sent (a background task, no added latency) a Haiku call reads only the user's own recent messages plus the known facts and returns JSON. Failure or bad JSON = "nothing learned", never an error. About $0.0005 per turn.
- **Filters before storing:** reject facts that read like instructions (memory poisoning: the ingestion scan's rules), facts containing personal data (they are dropped, not stored in redacted form), and wrong sizes. 5 facts per turn, 200 per user.
- **Merge:** embedding cosine >= 0.90 replaces the old wording; thresholds come from B-016 (bge-small compresses similarities, so 0.88 would have merged a nurse with a hospital worker).
- **Use:** facts with relevance >= 0.60 (top 3) go into the prompt in a `<user_notes>` block that says they are unverified, must never be cited and are not evidence; facts are escaped.
- **PII:** Presidio pattern recognisers only (B-016); applied to memory facts, every log line (`JsonFormatter`) and every Langfuse trace (`mask`). Our own UUIDs are exempt so logs stay debuggable.
- **Open issues:** a contradicting update below 0.90 leaves the stale fact; style preferences are not topical so they are only used when similar to the question; the transcript is stored un-redacted (the user owns it and can delete it) while logs and traces are redacted; history is stored but not yet used in prompts (query rewriting in Phase 5).
