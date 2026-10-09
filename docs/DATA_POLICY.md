# Data policy and AI disclosure

Plain-language note for a client's legal or security reviewer. Written 2026-10-09; it describes what the code does today, and says where it does not.

## What personal data the platform holds

| Data | Table(s) | Whose | Who can read it | Kept for | How it is deleted |
|---|---|---|---|---|---|
| Staff accounts (email, password hash, role) | `users` | employees | the org's admins; the hash only via the login function | while the account exists | admin removes the account (manual) |
| Uploaded documents and their chunks | `documents`, `chunks`, raw files | the org | by visibility: owner / members / public | until the org deletes them | `DELETE /documents/{id}` removes chunks and index entries |
| Member chat history | `conversations`, `messages` | the member | that member only | until the member deletes it | `DELETE /conversations/{id}` |
| Long-term memory (facts and preferences) | `memories` | the member | that member only; viewable and deletable | until deleted | `DELETE /memories/{id}`, delete-all |
| Website visitor chats | `widget_sessions`, `widget_messages` | anonymous visitors | org staff (admins) | 90 days after the last message (`RETENTION_VISITOR_DAYS`) | `make purge` (cron) |
| Leads (name, email, phone, message) | `leads` | visitors who agreed | org admins | 365 days (`RETENTION_LEAD_DAYS`) | `make purge`; consent tick is also a database CHECK |
| Thumbs-up/down and comments | `feedback` | visitors, members | org staff | 90 days | `make purge` |
| WhatsApp routing | `whatsapp_contacts` | visitors | the system only | 90 days after last message | `make purge`, or the user sends `STOP` |
| WhatsApp phone numbers | stored only in `leads`, after `YES <name>` | visitors | org admins | as leads | as leads, or `STOP` |
| Cached answers | `semantic_cache` | the org / the user | no one directly | 24 h TTL, off by default | expires; each hit is re-checked against the user's current documents |
| Spend counters | `org_usage`, `site_usage` | the org | org admins | indefinitely (no personal content) | with the org |
| Traces and logs | Langfuse, JSON logs | everyone | the operator | set by the operator's Langfuse project | operator action |

Visitors get **no** long-term memory: nothing is learned about people who never logged in. WhatsApp numbers are kept as a keyed hash for routing; the number itself exists only in a lead the person agreed to.

## What goes to third parties
- Question text and retrieved passages go to the LLM provider (Anthropic, via LiteLLM) to produce an answer. Embedding and reranking run locally; no document text goes to an embedding vendor.
- Optional: Twilio (WhatsApp messages) and Langfuse (traces). A client who must keep data out of a provider's hands can run the LLM through their own cloud account; the model is a setting.
- Memory: a fact that is itself personal data (emails, card numbers, etc.) is not stored at all.

## What is not done (honest limits)
- Retention for **staff** conversations is user-controlled, not time-based. A client with a legal period can add it to `app/retention.py`.
- Backups, Langfuse and provider-side logs follow their own retention; the operator must set them.
- Account removal and org deletion are manual SQL/admin work, not a self-service flow.
- No data-subject request tooling beyond per-item delete (`STOP`, delete conversation, delete memory). Export is not built.
- Not legal advice; a client's counsel decides the real retention periods. The periods above are defaults in config.

## AI disclosure (EU AI Act, Article 50)
People must be told they are talking to an AI system. Every channel says so before or at the first message, and a test (`tests/test_retention.py`) fails if any of the three loses the words "AI assistant":

| Channel | Wording |
|---|---|
| Web app | "You are chatting with an AI assistant. It answers only from your organisation's documents and cites its sources; please check them before relying on an answer." |
| Website widget | "You are chatting with an AI assistant. It answers from this company's public information; if it cannot help, you can ask for a person." |
| WhatsApp | "I'm an AI assistant. I answer from the company's public information." (sent with the welcome) |

Answers carry citations, the user can reach a person (handoff flag), and lead capture needs an explicit consent tick.
