#!/usr/bin/env bash
# Portable run: Colima/Docker only. No host venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY="$ROOT/deploy"
cd "$DEPLOY"

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created deploy/.env — add OLLAMA_API_KEY if you use a cloud model."
fi

mkdir -p "$ROOT/logs"

if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Stopping whatever is on port 8000 so the container can bind it..."
        lsof -nP -iTCP:8000 -sTCP:LISTEN -t | xargs kill 2>/dev/null || true
        sleep 1
    fi
fi

echo "Building and starting ProStock + n8n + Ollama..."
"$ROOT/scripts/compose.sh" up -d --build

cat <<EOF

Portable stack is up (no Python venv on the host).

  App   http://127.0.0.1:8000
  n8n   http://127.0.0.1:5678
  Copy this folder to another machine with Docker, then run the same script.

Stop:  $ROOT/scripts/compose.sh down
Train (optional, needs a venv):  $ROOT/scripts/setup_app.sh && $ROOT/scripts/train_weekly.sh

EOF
