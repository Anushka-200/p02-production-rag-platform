# Demo script (about 4 minutes)

Goal of the video: a client sees that this answers from THEIR documents, cites the page, refuses when it does not know, keeps companies apart, and has been attacked on purpose. Record at 1080p; terminal font large.

## Before recording (once)
1. `cp .env.example .env`, fill the passwords (`openssl rand -hex 16`), `JWT_SECRET` (`openssl rand -hex 32`) and your `ANTHROPIC_API_KEY` (a new key with its own spend limit).
2. `make demo` (builds, starts everything, creates `demo@demo.example`). Open http://localhost:8502.
3. Have two sample documents ready: one PDF with page numbers (e.g. a benefits policy) and `data/samples/acme-benefits-policy.docx`.

## Script
| Time | Say | Show |
|---|---|---|
| 0:00 | "Companies keep answers in documents nobody can search. This is a multi-tenant document Q&A platform: every answer cites its source and refuses when it can't find one." | README top: impact line and the architecture diagram |
| 0:20 | "Log in as the company's admin, upload a policy PDF." | UI: upload; Admin page > ingestion status goes queued > parsing > embedding > ready |
| 0:50 | "Ask a question the document answers." Example: "How many vacation days do new employees get?" | Streaming answer with citation `[S1]`; click it: document, page, chunk text |
| 1:20 | "Now one it doesn't answer." Example: "What is the CEO's salary?" | Refusal: "I couldn't find this in the available documents." |
| 1:40 | "A follow-up, the way people really talk." Ask "and for part-time staff?" | Query rewrite shown in the Admin inspector as a standalone question |
| 2:00 | "Why did it pick these passages?" | Admin > retrieval inspector: dense, keyword, fused and rerank scores per candidate |
| 2:30 | "Companies are isolated by the database, not by the prompt." Log in as the other company's user and ask the first company's question | Refusal; then `make adversarial` prints "47 attacks, 0 leaks" |
| 3:00 | "And I attacked it on purpose." `make redteam` (or show B-040) | Table: 83 attacks, 14 families, 0 breaches, mapped to OWASP GenAI Top 10 |
| 3:30 | "Customers can use it on their website or WhatsApp." | `make demo-widget`: the widget answers a visitor, leaves a lead, asks for a human |
| 3:50 | "Numbers: faithfulness 0.95, refusal accuracy 0.96 on near-miss questions, about 0.34 cents per answer, every claim in docs/benchmarks.md." | README results table |

## If something goes wrong on camera
- Answers are slow the first time: the embedding model downloads once (about 130 MB). Do a warm-up question before recording.
- A refusal where you expected an answer is honest behaviour, not a bug: say so, then ask a more direct question.
- Never show `.env`.
