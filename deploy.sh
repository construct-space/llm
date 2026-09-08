#!/usr/bin/env bash
# Deploy api/tinker/ to Caprover app 'llm' (llm.lisaos.dev).
# Not a git root, so we tar the directory and push via --tarFile.
set -euo pipefail

cd "$(dirname "$0")"

TAR=/tmp/tinker-deploy.tar.gz

echo "→ packing $TAR"
tar -czf "$TAR" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.env' \
  --exclude='.DS_Store' \
  --exclude='deploy.sh' \
  .

echo "→ deploying to Caprover app 'llm'"
caprover deploy --appName llm --tarFile "$TAR"

echo "✓ deployed. Tail logs in Caprover dashboard → Apps → llm → App Logs"
echo "  Quick health check: curl https://llm.lisaos.dev/health"
