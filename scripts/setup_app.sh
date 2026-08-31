#!/usr/bin/env bash
# Optional host venv for training only. Daily use is Docker: ./scripts/dev_local.sh
# Recreate the app venv with a Python that can install TensorFlow on Apple Silicon.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        case "$ver" in
            3.1[0-3]) PY="$candidate"; break ;;
        esac
    fi
done

if [[ -z "$PY" ]]; then
    echo "Need Python 3.10–3.13. macOS 3.9 cannot install current TensorFlow."
    echo "Install with:  brew install python@3.12"
    exit 1
fi

echo "Using $($PY --version)"
rm -rf venv
"$PY" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-prod.txt
echo
echo "venv is ready. Next:"
echo "  source venv/bin/activate"
echo "  export OLLAMA_URL=http://127.0.0.1:11434"
echo "  export OLLAMA_MODEL=nemotron-3-ultra:cloud"
echo "  export N8N_WEBHOOK_URL=http://127.0.0.1:5678/webhook/prostock-brief"
echo "  python -m uvicorn main:app --reload --port 8000"
echo
echo "If you use an Ollama cloud key, put it in deploy/.env as OLLAMA_API_KEY=..."
echo "Do not paste ollama.com URLs on the export line."
