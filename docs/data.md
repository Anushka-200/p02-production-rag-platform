# Data sources and licences

Checked 2026-09-30 against primary files (the repos' LICENSE files), not summaries.
Rule: verify a licence before downloading. Raw data lives in `data/raw/` (git-ignored) and is
**never committed or redistributed**; the repo holds only download scripts and this note.

| Source | Role | Licence found | Verified how | Our use | Status |
|---|---|---|---|---|---|
| GitLab handbook (`gitlab.com/gitlab-com/content-sites/handbook`, path `content/handbook`) | Tenant A, living wiki (git history, edited daily) | Repo `LICENSE` = MIT ("software and associated documentation files"), copyright GitLab B.V. No separate content licence in the repo. A secondary source says CC BY-SA 4.0; **not confirmed from a primary file** | Read `LICENSE` via GitLab API; README has no licence text | Private indexing and evaluation; short quotes with attribution in README/demo. Do not republish the corpus. Treat as attribution + share-alike to stay safe under either licence | Allowed for our use; open point: confirm CC BY-SA 4.0 |
| PostHog handbook (`github.com/PostHog/posthog.com`, path `contents/handbook`) | Tenant B, overlapping topics for leak tests | Repo `LICENSE` has two parts: **MIT for `/contents/`**; a custom restrictive licence for everything else (site design, layout, non-contents files) | Read `LICENSE` on GitHub; confirmed `handbook/` sits inside `/contents/` | Use only `contents/handbook`. Nothing from outside `/contents/` (no site code, design or logos) | Allowed for `contents/handbook` only |
| Public HR / benefits PDFs with tables and page numbers (`data/raw/hr-pdfs/`) | Parsing, tables, page citations | US federal government publications: not subject to copyright; IRS asks for credit ([Use of content from IRS.gov](https://www.irs.gov/about-irs/use-of-content-from-irsgov)) | IRS policy page read; OPM page's own copyright line not separately checked | `irs-p15b-fringe-benefits.pdf` (IRS Pub 15-B), `irs-p969-health-accounts.pdf` (IRS Pub 969), `opm-health-insurance-overview.pdf` (OPM FEHB overview). Credit the agency in README/demo. The DOL FMLA guides were skipped: DOL's CDN returns 403 to scripted downloads, and we do not work around access blocks | Downloaded; page counts checked in Phase 1 |
| Synthetic benefits-policy DOCX with tables | DOCX parsing test | Our own work | n/a | Committed (task 1.4) | Phase 1 |
| Adversarial documents | Prompt-injection and isolation tests | Written by us | n/a | Committed to the repo (`data/adversarial/`, Phase 3) | Phase 3 |

## Corpus scope (PLAN.md revision 2026-09-30)
Tenant A for LLM-costly steps = GitLab sections people-group, people-policies, total-rewards, hiring, values,
communication, leadership, finance (~220 pages, ~0.8M tokens). Full handbook (~4,200 pages) is ingested without
blurbs for scale tests. The eval set pins the handbook commit it was written against.

## Open points
- Confirm whether GitLab publishes the handbook content under CC BY-SA 4.0 (e.g. a footer or the
  handbook's own "license" page). Until then attribution is given anyway.
- DOCX: no public-domain DOCX with tables was found quickly. In task 1.4 we author a small synthetic benefits-policy DOCX ourselves (our own work, so no licence issue) to test `python-docx` tables.
- Optionally download the DOL FMLA employee/employer guides manually in a browser if more PDFs are wanted.
