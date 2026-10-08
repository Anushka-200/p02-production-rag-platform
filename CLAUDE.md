# P02 Production RAG Platform

The shared engineering standard and my working style load automatically from `~/portfolio/CLAUDE.md`
(which imports `ai-portfolio-kit/01_SHARED_CLAUDE.md`). Don't duplicate them here.
The full project brief is `~/portfolio/ai-portfolio-kit/projects/P02_production_rag_platform.md`.

## Project: P02 Production RAG Platform (from the brief)
Goal: Multi-tenant document Q&A. Orgs → users → documents (private or org-shared). Upload PDF/DOCX/MD/TXT/HTML → parse (keep page numbers) → chunk (configurable strategy) → embed → store in Postgres/pgvector + full-text index. Query → retrieval mode (dense | keyword | hybrid-RRF) → rerank → answer with citations [doc, page, chunk] → refuse if unsupported.
Memory: per-user long-term memory (facts/preferences extracted from chats, user can view/delete), separate from chat history and from RAG.
Isolation: Postgres RLS + tenant_id enforced in the backend (never trust the prompt). Adversarial test suite: modified requests, forged tenant ids, prompt injection, instructions hidden inside uploaded docs.
Channels: FastAPI + Next.js (or Streamlit) app, embeddable JS chat widget, WhatsApp (Twilio sandbox) channel. Lead capture + human handoff flag.
Admin: retrieval inspector (query → candidates with dense/keyword/rerank scores), ingestion status, chunk previews.
Workers: ingestion in a background queue (Celery/RQ + Redis) with status.
Eval: evals/qa.jsonl (80 questions with gold doc/page + 15 unanswerable).
v2: Deploy on AWS with Terraform (VPC, ECS Fargate, RDS Postgres w/ pgvector, ALB + HTTPS), GitHub Actions CI/CD, CloudWatch.

## Decisions made before starting (2026-09-29)
**Scope: everything stays in this project.** A later finish is fine; no concept gets skipped or pushed to
another project. That includes the brief's v2 (AWS + Terraform + CI/CD + load test), done after v1.

**Concepts added beyond the brief** (each needs learning material, a design decision and a measured
experiment where it applies):
- **Document preprocessing:** cleaning, boilerplate/nav removal, dedupe, table handling, (OCR if a
  scanned PDF is in the corpus)
- **Query rewriting:** turn follow-ups into standalone questions using chat history
- **Query expansion:** multi-query and HyDE
- **Hierarchical RAG:** parent-child chunks; summary tree (RAPTOR-style)
- **GraphRAG:** as a comparison experiment on relationship questions. P10 in the kit is a GraphRAG
  project, so confirm at planning how deep to go here vs there
- **Agentic RAG:** an agent that decomposes multi-hop questions, decides whether to search again, and
  picks tools
- Also from the brief: contextual retrieval, semantic cache, reranking, memory

**Coverage gaps also assigned to P02** (from the portfolio-wide research, 2026-09-29; details in the brief's
"Coverage gaps to include" section): long context vs RAG experiment; per-tenant token budgets with alerts;
user-feedback loop into the eval set; a structured red-team pass (garak / promptfoo / PortSwigger patterns);
a data-retention note plus an "AI assistant" disclosure line (EU AI Act Article 50).

**Living-wiki RAG (connectors + incremental sync):** most companies keep knowledge in wikis
(Confluence/Notion/SharePoint/GitHub wiki) that change daily. Build a connector interface, change
detection (last-modified / content hash per page and section), re-index only what changed, remove
deleted pages, permission sync (permission-aware retrieval), and "last updated" in citations. Demo it by
replaying real edits from a public handbook's git history.

**Data (verify every licence before downloading anything):**
- Tenant A: GitLab's public handbook (Markdown, git repo, edited daily = the living wiki)
- Tenant B: another company's public handbook on overlapping topics, e.g. PostHog's (for leak tests)
- Public HR/benefits PDFs and DOCX with tables and page numbers (parsing, tables, page citations)
- Adversarial documents we write ourselves (hidden injected instructions)
- Eval set: 80 questions with gold doc/page + 15 unanswerable. **Record which items I reviewed**
  (`reviewed: true`) - P01's eval never did

## Lessons from P01 to apply here
- Every integration step gets a separate ~1 h "live run + fix" task in the estimate.
- Re-estimate the moment the approach or scope changes.
- Record end times immediately; add ~15% for unplanned environment trouble.
- Use the multipliers in `ai-portfolio-kit/03_ESTIMATION_TRACKER.md` (section B), not the defaults.
- When resetting demo data, clear every store together (P01's id-reuse gotcha).
- Before any push: secrets + personal-data scan across the whole history. Push only when I ask.
- Run things for real (Docker, demos) before calling a step done - it found a bug almost every time.

## Heavy jobs: one at a time (added 2026-10-08 after the laptop froze twice)
WSL has about 6 GB of RAM. Three evals/embedding runs in parallel filled it (swap hit 3.4 GB) and froze the machine.
- **Never run more than one heavy job at once** (evals, embedding runs, blurb generation, load tests, the full
  test suite). Run them one after another: in the foreground, or as ONE background job.
- Run `free -h` before each heavy job and report it. Light work (editing, lint, one small test file) may run beside a
  single heavy job; a second heavy job may not.
- Watch the job's RSS (`ps -o rss,cmd -p PID`); if it passes 3.5 GB, stop it and tell the owner.
- Use the caches (`data/cache/*`) so finished work is never redone.

## Environment
- WSL and Docker live on D: (C: is small). P01 is running alongside: Postgres **5433**, n8n **5678**,
  and P01 uses API 8000 / dashboard 8501. **P02 must use other ports** (e.g. Postgres 5434, Redis 6380,
  API 8001, UI 8502).
- Anthropic key: create a new one for P02 in Claude Console with its own spend limit.

## How to start
1. **Learning step first:** the brief's Prompt 0 (9 concepts) plus the added concepts above, each with a
   failure example, plain-language explanations, then a quiz. Split it into two parts if it's long.
2. Then plan mode: PLAN.md + ADRs (the brief's 001-006, plus query rewriting/expansion, hierarchical /
   GraphRAG / agentic, and connectors / incremental sync).
3. Then ESTIMATES.md with my O/M/P per task, your view, and the client quote.
