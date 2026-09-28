#!/usr/bin/env bash
set -Eeuo pipefail

REPO_NAME="${1:-financial-stress-prediction}"
OWNER="${2:-khalwaniev}"

git status >/dev/null 2>&1 || {
  git init -b main
}

git add .
git commit -m "Initial portfolio release" || true

if git remote get-url origin >/dev/null 2>&1; then
  echo "origin already configured: $(git remote get-url origin)"
else
  gh auth status
  gh repo create "${OWNER}/${REPO_NAME}" \
    --public \
    --source . \
    --remote origin \
    --description "Leakage-aware tabular ML pipeline for financial stress prediction" \
    --push
fi

echo
echo "Repository:"
echo "https://github.com/${OWNER}/${REPO_NAME}"
