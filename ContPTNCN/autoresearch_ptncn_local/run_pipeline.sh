#!/bin/bash
set -e

echo "=============================================="
echo "PTNCN Autoresearch With Local LLM"
echo "Started: $(date)"
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
cd "$SCRIPT_DIR"

echo ""
echo "=== Step 1: Checking prerequisites ==="
"$PYTHON_BIN" prepare.py

echo ""
echo "=== Step 2: Creating or switching to experiment branch ==="
BRANCH="autoresearch/ptncn-local-$(date +%Y%m%d)"
cd "$REPO_ROOT"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree is not clean."
  echo "Commit or stash your current changes, or run autoresearch in a separate clone/worktree."
  exit 1
fi
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
cd "$SCRIPT_DIR"
echo "On branch: $BRANCH"

echo ""
echo "=== Step 3: Starting autonomous PTNCN search ==="
echo "Model: ${AUTORESEARCH_MODEL:-rnj:latest}"
echo "Press Ctrl+C to stop."
echo ""
PYTHONUNBUFFERED=1 "$PYTHON_BIN" agent.py
