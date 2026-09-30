#!/usr/bin/env bash
# Downloads the two handbook corpora into data/raw/ (git-ignored). Licences: see docs/data.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

if [ ! -d data/raw/gitlab-handbook/.git ]; then
  # Partial clone: full history (needed to replay real edits) but file contents fetched on demand.
  git clone --filter=blob:none --no-checkout --quiet \
    https://gitlab.com/gitlab-com/content-sites/handbook.git data/raw/gitlab-handbook
  git -C data/raw/gitlab-handbook sparse-checkout set content/handbook
  git -C data/raw/gitlab-handbook checkout --quiet main
fi

if [ ! -d data/raw/posthog-handbook/.git ]; then
  # Shallow clone: latest version only. Only contents/handbook (the MIT-licensed part).
  git clone --depth 1 --filter=blob:none --no-checkout --quiet \
    https://github.com/PostHog/posthog.com.git data/raw/posthog-handbook
  git -C data/raw/posthog-handbook sparse-checkout set contents/handbook
  git -C data/raw/posthog-handbook checkout --quiet master
fi
echo "done"
