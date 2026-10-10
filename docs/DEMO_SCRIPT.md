# Demo script: what to say and what to click (about 5 minutes)

Validated on 2026-10-10: every step below was run against the real stack (API, worker, Streamlit UI driven in a headless browser, real model). The answers quoted are what it returned; wording varies a little between runs, the facts do not.

## 0. Setup (before you press record)
Three terminals in `~/portfolio/p02-production-rag-platform`, after `make up` (Postgres + Redis):
```bash
make api       # terminal 1: http://localhost:8001
make worker    # terminal 2: the ingestion worker
make ui        # terminal 3: http://localhost:8502
```
Accounts (both password `demo-password-123`):
- `demo-a@demo.example`: admin of **Tenant A**, a company whose documents are GitLab's public handbook (217 pages) plus public HR/tax PDFs.
- `demo-b@demo.example`: admin of **Tenant B**, a different company: PostHog's public handbook (385 pages).

Do one warm-up question first (models load on the first request). Never show `.env`. If you rehearse the upload step, delete the document afterwards (Admin > Ingestion status > Documents), otherwise the second upload is reported as a duplicate.
Files for the upload step: `data/samples/acme-benefits-policy.docx` (a Word policy with a table).

---

## 1. The 30-second pitch (README open on screen)
> "Every company has answers buried in policies and handbooks that nobody can search properly. I built a platform where employees, or a company's website visitors, ask a question in plain English and get an answer that **cites the exact document and page**, or an honest 'I couldn't find this' if the documents don't say. It serves many companies at once, and one company can never see another's data, enforced by the database, not by a promise in a prompt. I measured it: 95% of statements in answers are supported by their sources, 96% correct refusals on tricky unanswerable questions, 0.34 cents per answer, and I attacked it with 83 hostile tests and none got through."

## 2. Upload a document and watch it get ready (about 40 s)
Log in as `demo-a`. Sidebar: **Admin** > tab **Ingestion status** > expander **Upload a document**. Choose the DOCX, "Who can read it" = `org`, **Upload**, press **Refresh**.
> "I'm uploading a Word policy as the company admin. It goes onto a queue and a background worker picks it up: it reads the file keeping page and heading information, cleans it, scans it for hidden instructions, cuts it into passages, turns each passage into numbers (an embedding) and stores it. You can see the job go from queued to running to succeeded, so nothing happens invisibly."

## 3. Ask a question the new document answers (about 40 s)
Sidebar **Chat**. Type: `What is the hotel limit in the UK according to the Acme benefits policy?`
Expected: "According to the Acme Corp Benefits Policy, the hotel limit in the UK is £150 [S1]." about 2 s, about 0.2 cents. Open **Sources (1)**.
> "The answer came from a *table* in the Word file, and the [S1] marker opens the source: document, section, the passage. If I ask it to prove a claim, the proof is one click away. The line under the answer shows time to first word and what this answer cost."

Optional page-number moment: ask `Are de minimis fringe benefits taxable to the employee?` Answer: not taxable, cited to `irs-p15b-fringe-benefits.pdf`, **page 10**.
> "For PDFs the citation carries the page number, so a reviewer can check it in seconds."

## 4. A question it cannot answer (about 20 s)
`What is the CEO's home address?`
Expected: "I couldn't find this in the available documents."
> "This is the most important behaviour for a business. A bot that guesses is a liability. This one refuses, and I measured it: on 52 deliberately tricky questions that look answerable but aren't, it refused 50 of them correctly (96%)."

## 5. A follow-up the way people really talk (about 40 s)
`How many weeks of parental leave do GitLab team members get?` (answer: 16 weeks, cited to the leave-types page). Then in the same chat: `And what about adoption?`
> "The second question means nothing alone. The system rewrites it into a full question using the chat history before searching. Before I added that, follow-ups found the right passage 0% of the time; now about three quarters of the time."

## 6. Why did it pick those passages? (about 50 s)
**Admin** > **Retrieval inspector**. Type `hotel limit in the UK` > **Inspect**.
> "This is the window into the search. Every candidate shows its dense score (meaning), keyword score (exact words), the fused score and finally the reranker score. Here the Word document sits 4th after fusion, and the reranker, a small model that reads the question and passage together, moves it to number 1. That reranker was the single biggest quality gain I measured: right-passage-first went from 43% to 59%."

Point at: the table columns, the `used` column (S1, S3, ... which candidates went to the model), the timings (search is about a tenth of a second plus about a second for reranking; most of the rest is the model writing).

## 7. Companies are kept apart (about 50 s)
Sign out. Log in as `demo-b` (PostHog's company). Ask the same question: `What is the hotel limit in the UK?` Expected: "I couldn't find this in the available documents." Then `How does PostHog handle paid time off?` (answered from PostHog's own handbook: unlimited time off, at least 25 days expected).
> "Same question, different company: refused, because that document belongs to Tenant A. This isn't the model being polite. Every database query runs with the company stamped on it from the login token, and Postgres itself filters rows (row-level security). Even if someone tricks the model, the rows aren't there to be found. I tested this with 47 attacks: forged IDs, forged tokens, other companies' document IDs: zero leaks."

## 8. I attacked it on purpose (about 40 s)
Show `docs/benchmarks.md` B-039/B-040 or run `make redteam` (about 2 min, $0.11).
> "I wrote 83 attacks in 14 families, modelled on the public security test suites (garak, promptfoo, PortSwigger): 'ignore your instructions', hidden instructions inside uploaded files, instructions split across chat turns, base64 and look-alike letters, trying to steal another company's secret, poisoning the cache and the memory. I plant fake secrets and check by exact match whether any leaks. Result: 83 blocked, zero breaches, each mapped to the OWASP list of LLM risks. I'm also honest about what that means: it's evidence, not proof, and a part of the refusals happened because the documents simply held nothing to answer from."

## 9. Website widget and WhatsApp (about 30 s, optional)
`make demo-widget` serves a sample customer site; the widget answers a visitor from *public* documents only, can take a lead with consent and flag "talk to a human". WhatsApp uses the same engine.
> "Customers can put one script tag on their site. Visitors see only what the company marked public, and the widget says up front that it is an AI assistant. WhatsApp is built and tested with signed requests; I have not yet run it with a real phone."

## 10. Close with numbers (about 20 s, README results table)
> "Faithfulness 0.95, correct refusals 96%, recall at 5 of 0.81, 0.34 cents and about 4 seconds per answer, 83 attacks and no breaches, and every number traces to a logged measurement. What I'd do next: deploy it on AWS with Terraform and load-test it."

---

# Talking points: the whole story (for the intro and for questions)

## What it is
A multi-tenant document Q&A platform (RAG: Retrieval-Augmented Generation = look up the relevant passages first, then have the model answer only from them). Orgs > users > documents (private, org-wide, or public). Upload PDF/DOCX/Markdown/HTML/TXT or sync from a git-based wiki; ask in a web app, a website widget or WhatsApp.

## How I built it, in order
1. **Foundations:** Docker Compose with Postgres 16 + pgvector and Redis; three database roles so the app can never change the schema or bypass security; settings only from environment variables; every model call through LiteLLM (swap models without code changes) and traced in Langfuse.
2. **Ingestion:** parsers that keep page numbers and headings; a heading-aware chunker (400 tokens, 50 overlap); cleaning and hidden-text removal; an injection scan; local embeddings (bge-small); background worker with job status and safe retries.
3. **Retrieval and answers:** dense + keyword search merged by Reciprocal Rank Fusion, a local cross-encoder reranker, a token-budgeted context, an answer prompt that must cite or refuse, and an output checker that rejects invented citations, images and foreign links.
4. **Tenancy and memory:** JWT login, row-level security, private/org/public documents, per-user memory (facts it learned about you; you can view and delete it; personal data is never stored).
5. **Channels:** Streamlit chat + admin console, a one-file embeddable widget, WhatsApp via Twilio, lead capture and human handoff.
6. **Advanced retrieval, built behind switches:** query rewriting, multi-query and HyDE, parent chunks, a summary tree, a GraphRAG slice, an agent, a connector with incremental sync of a living wiki.
7. **Evaluation and optimisation:** an 80+15 question plan that grew into 267 reviewed questions with a dev/test split; ablations of everything above; a semantic cache; per-organisation budgets.
8. **Security pass:** an 83-attack red team, an OWASP mapping, a data-retention policy and purge, an "AI assistant" disclosure on every channel, a secrets/personal-data scan of the git history, a dependency audit.
9. **Packaging:** Docker image + Compose for API, worker and UI, CI (lint, types, tests with a real database, a 15-question mocked eval, dependency audit, image build).

## How I improved it (all measured; see `docs/benchmarks.md`)
| What I changed | Effect |
|---|---|
| Heading-aware chunker instead of plain splitting | Recall@5 0.67 > 0.75 |
| Local reranker on top of hybrid search | right passage first 43% > 59%; Recall@5 0.76 > 0.81 |
| Rewrite follow-up questions using chat history | follow-up Recall@5 0.00 > 0.60 (0.76 on the reviewed set) |
| Fixed my own evaluation | refusal accuracy 0.6 > 0.96 and follow-ups 0.10 > 0.76 were the *eval's* fault; about a quarter of the first drafted questions were flawed |
| Incremental sync of a changing wiki | 30% of the embedding work and one third of the time of full rebuilds |
| Semantic cache (off by default) | a hit costs 19 ms and $0 instead of about 4 s and 0.35 cents, with no wrong hits at the chosen threshold |
| RAG vs pasting 150k tokens into the prompt | RAG: higher correctness, 18x cheaper, 11x faster |
| Per-organisation daily budget | a 50-request runaway loop was stopped after 3 requests |

**Things that did not help, kept honestly:** multi-query, HyDE, parent expansion, the summary tree, GraphRAG (helped relationship questions, hurt broad ones), the agent, contextual blurbs (about $7 for +4 points), a bigger embedder, a longer-answer prompt (+3.8 on dev, +0.6 on the hidden test: reverted). Each is built, tested and switched off; a client with a different corpus can switch one on and re-measure.

## Likely questions, with straight answers
- **"How do I know it won't hallucinate?"** It must cite or refuse; a checker rejects invented citations; I measured faithfulness 0.95 with an independent judge model, and 96% correct refusals on near-miss unanswerable questions. It's not zero: about 5% of statements were not fully supported.
- **"Can one customer see another's data?"** Not through this system: the database enforces it. 47 isolation attacks and 83 red-team attacks: no leaks. Anonymous visitors can only ever read documents marked public.
- **"What does it cost?"** About 0.34 cents per answer with Claude Haiku 4.5, local embeddings and reranking (free), budgets per organisation.
- **"How fast?"** About 4 s for a full answer, first words in about 2 s; follow-ups use one extra small model call.
- **"What about our private data and Europe?"** Documents and embeddings stay in your Postgres; only the question and the retrieved passages go to the model provider. A data policy, retention (visitor chats 90 days, leads 365) and an AI disclosure are written down and tested.
- **"What would you not trust yet?"** The eval set was reviewed by a model, not a human (a human spot check is the next step); multi-hop and relationship questions are weaker; WhatsApp is not yet tested with a real phone; there's no cloud deployment yet.
