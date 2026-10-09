# Security review (Phase 7, 2026-10-09)

## 1. Red-team: 83 attacks, 0 breaches
See B-039 and B-040 in `docs/benchmarks.md` and the OWASP mapping in `docs/adr/005-tenant-isolation.md`. Re-run with `make redteam` (about $0.11).

## 2. Secrets and personal-data scan of the whole git history (121 commits, 20 MB, local only, never pushed)
Method: `git log --all -p` searched for key formats (Anthropic, OpenAI, AWS, GitHub, Slack, Twilio SID, private keys, Langfuse), for assignments of 16+ character values to key/secret/token/password names, for added file names (`.env`, keys, credentials), for the owner's name, email and home path, and for emails and phone numbers. No dedicated scanner (gitleaks, trufflehog) was installed; it should be run in CI as well.

| Check | Result |
|---|---|
| Real API keys or tokens in any commit | none found |
| `.env` or key files ever added | none (`.env` is git-ignored; only `.env.example`) |
| Owner's email, name, home path in file content | none |
| Emails and phone numbers in content | all from public corpora (GitLab and PostHog handbooks, public benefit documents) or obvious test fixtures (`ann@x.com`) |
| One token-like string | a `?token=` in a public insurer's URL inside a public benefits PDF; not ours, not a secret |
| **Finding: 605 uploaded-document blobs (13 MB) were tracked** in `data/uploads/` | runtime storage does not belong in git. Fixed going forward: untracked and git-ignored. They remain in history (public corpora under MIT / CC licences, plus our synthetic documents). |
| Commit author metadata | one author with a personal Gmail address in every commit. Normal on GitHub, but a noreply address avoids publishing it. |

Decision for the owner before the first push (history is local, so a rewrite is still cheap): (a) rewrite history to drop `data/uploads/` and map the author to a GitHub noreply address, or (b) push as is. Recommendation: (a), then re-run this scan on the rewritten history.
