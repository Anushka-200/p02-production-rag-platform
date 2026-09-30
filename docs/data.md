# Data sources and licences

Checked 2026-09-30 against primary files (the repos' LICENSE files), not summaries.
Rule: verify a licence before downloading. Raw data lives in `data/raw/` (git-ignored) and is
**never committed or redistributed**; the repo holds only download scripts and this note.

| Source | Role | Licence found | Verified how | Our use | Status |
|---|---|---|---|---|---|
| GitLab handbook (`gitlab.com/gitlab-com/content-sites/handbook`, path `content/handbook`) | Tenant A, living wiki (git history, edited daily) | Repo `LICENSE` = MIT ("software and associated documentation files"), copyright GitLab B.V. No separate content licence in the repo. A secondary source says CC BY-SA 4.0; **not confirmed from a primary file** | Read `LICENSE` via GitLab API; README has no licence text | Private indexing and evaluation; short quotes with attribution in README/demo. Do not republish the corpus. Treat as attribution + share-alike to stay safe under either licence | Allowed for our use; open point: confirm CC BY-SA 4.0 |
| PostHog handbook (`github.com/PostHog/posthog.com`, path `contents/handbook`) | Tenant B, overlapping topics for leak tests | Repo `LICENSE` has two parts: **MIT for `/contents/`**; a custom restrictive licence for everything else (site design, layout, non-contents files) | Read `LICENSE` on GitHub; confirmed `handbook/` sits inside `/contents/` | Use only `contents/handbook`. Nothing from outside `/contents/` (no site code, design or logos) | Allowed for `contents/handbook` only |
| Public HR / benefits PDFs and DOCX with tables and page numbers | Parsing, tables, page citations | Candidate: US federal government publications (US federal works are public domain in the US) | Not yet verified per file | Choose 3-5 files in task 0.8; record URL + licence line per file here | Pending |
| Adversarial documents | Prompt-injection and isolation tests | Written by us | n/a | Committed to the repo (`data/adversarial/`, Phase 3) | Phase 3 |

## Open points
- Confirm whether GitLab publishes the handbook content under CC BY-SA 4.0 (e.g. a footer or the
  handbook's own "license" page). Until then attribution is given anyway.
- PDFs: pick files whose page states the licence or that are US federal works; note each here.
