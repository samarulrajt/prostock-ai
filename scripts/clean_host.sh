#!/usr/bin/env bash
# Drop host caches and the local venv. The app runs in Docker after ./scripts/dev_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Purging pip cache..."
python3 -m pip cache purge 2>/dev/null || true
if [[ -x venv/bin/pip ]]; then
    venv/bin/pip cache purge 2>/dev/null || true
fi

echo "Removing host venv, bytecode, pytest cache..."
rm -rf venv .pytest_cache
find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

echo "Done. Project on disk is source + weights only; runtime is Docker."
du -sh . 2>/dev/null || true
