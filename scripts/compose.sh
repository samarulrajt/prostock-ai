#!/usr/bin/env bash
# Run Compose from the deploy/ folder with whichever binary exists.
# Usage (from repo root):  ./scripts/compose.sh up -d
#                          ./scripts/compose.sh down
#                          ./scripts/compose.sh exec ollama ollama pull nemotron-3-ultra:cloud
set -euo pipefail

DEPLOY="$(cd "$(dirname "$0")/../deploy" && pwd)"
cd "$DEPLOY"

ensure_daemon() {
    if docker info >/dev/null 2>&1; then
        return 0
    fi
    echo "Docker daemon is not running."
    if command -v colima >/dev/null 2>&1; then
        echo "Starting Colima (this can take a minute)..."
        if ! colima start; then
            echo "Colima did not start. Clearing a stuck VM and retrying..."
            colima stop -f || true
            colima start
        fi
        if docker info >/dev/null 2>&1; then
            return 0
        fi
        echo "Colima started but Docker still cannot connect."
        echo "Try:  colima start && docker info"
        exit 1
    fi
    echo "Start the engine, then retry:"
    echo "  colima start          # if you use Colima"
    echo "  # or open Docker Desktop"
    exit 1
}

ensure_daemon

if docker compose version >/dev/null 2>&1; then
    exec docker compose "$@"
fi

if command -v docker-compose >/dev/null 2>&1; then
    exec docker-compose "$@"
fi

PLUGIN="$HOME/.docker/cli-plugins/docker-compose"
if [[ -x "$PLUGIN" ]]; then
    exec docker compose "$@"
fi

echo "Docker Compose is not available on this machine."
echo
echo "Docker itself is installed, but the Compose plugin is missing."
echo "That is why 'docker compose up -d' errors with: unknown shorthand flag: 'd'"
echo
echo "Fix (pick one):"
echo
echo "  A) Docker Desktop (recommended on Mac)"
echo "     https://www.docker.com/products/docker-desktop/"
echo "     Install, open it, wait until it says running, then:"
echo "       docker compose version"
echo
echo "  B) Homebrew Compose plugin"
echo "       brew install docker-compose"
echo "       mkdir -p ~/.docker/cli-plugins"
echo "       ln -sf \$(brew --prefix)/bin/docker-compose ~/.docker/cli-plugins/docker-compose"
echo "       docker compose version"
echo
echo "Then from the repo:"
echo "  ./scripts/dev_local.sh"
echo "  or: ./scripts/compose.sh up -d"
exit 1
