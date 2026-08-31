#!/usr/bin/env bash
# Weekly (or on-demand) training for ProStock.
# Install on Linux: copy deploy/prostock-train.{service,timer} and enable the timer.
# Cron alternative (Sunday 03:00):  0 3 * * 0 /path/to/simple-model/scripts/train_weekly.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOCKDIR="$ROOT/.train.lock"
LOGDIR="$ROOT/logs"
ARCHIVEDIR="$ROOT/models/archive"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-4}"
DATE_STAMP="$(date +%Y-%m-%d_%H%M%S)"
LOGFILE="$LOGDIR/train-${DATE_STAMP}.log"

if ! mkdir "$LOCKDIR" 2>/dev/null; then
    echo "Training already running (lock: $LOCKDIR)" >&2
    exit 0
fi
cleanup() { rmdir "$LOCKDIR" 2>/dev/null || true; }
trap cleanup EXIT

mkdir -p "$LOGDIR" "$ARCHIVEDIR"

if [[ -x "$ROOT/venv/bin/python" ]]; then
    PYTHON="$ROOT/venv/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

archive_if_present() {
    local src="$1"
    local name="$2"
    if [[ -f "$src" ]]; then
        cp "$src" "$ARCHIVEDIR/${name}.${DATE_STAMP}"
    fi
}

prune_archives() {
    local pattern="$1"
    local n=0
    ls -1t "$ARCHIVEDIR"/$pattern 2>/dev/null | while read -r f; do
        n=$((n + 1))
        if (( n > KEEP_ARCHIVES )); then
            rm -f "$f"
        fi
    done
}

{
    echo "=== ProStock weekly train $DATE_STAMP ==="
    echo "python: $PYTHON"
    echo "root: $ROOT"

    archive_if_present "$ROOT/pro_model.h5" "pro_model.h5"
    archive_if_present "$ROOT/pro_scaler.pkl" "pro_scaler.pkl"
    prune_archives "pro_model.h5.*"
    prune_archives "pro_scaler.pkl.*"

    "$PYTHON" -u get_tickers.py
    "$PYTHON" -u train_pro_model.py \
        --file all_tickers.txt \
        --epochs "${TRAIN_EPOCHS:-15}" \
        --batch-size "${TRAIN_BATCH_SIZE:-64}" \
        --download-batch "${TRAIN_DOWNLOAD_BATCH:-40}" \
        --max-per-ticker "${TRAIN_MAX_PER_TICKER:-300}" \
        --pause "${TRAIN_PAUSE:-1.0}"

    echo "=== train finished $(date -Iseconds) ==="
    echo "Writing walk-forward baseline metrics..."
    "$PYTHON" -u eval_baselines.py
    "$PYTHON" -u eval_lstm.py || echo "LSTM OOS eval skipped"

    if command -v docker >/dev/null 2>&1 && [[ -f "$ROOT/deploy/docker-compose.yml" ]]; then
        echo "Restarting prostock container so it loads the new weights"
        env_file=()
        if [[ -f "$ROOT/deploy/.env" ]]; then
            env_file=(--env-file "$ROOT/deploy/.env")
        fi
        docker compose -f "$ROOT/deploy/docker-compose.yml" "${env_file[@]}" restart prostock || true
    fi
} 2>&1 | tee -a "$LOGFILE"
