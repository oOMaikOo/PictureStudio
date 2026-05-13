#!/usr/bin/env bash
# ============================================================
# start_monitor.sh — Helper script to launch the daemon
# Edit PROFILE_PATH and PYTHON to match your installation.
# ============================================================

# Absolute path to your monitoring profile JSON
PROFILE_PATH="${1:-/opt/picture/profiles/monitor_profile.json}"

# Python with the project virtualenv active
PYTHON="/opt/picture/.venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# If the system Python doesn't have the venv, fall back to project venv
if [ ! -f "$PYTHON" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi

exec "$PYTHON" "$PROJECT_ROOT/scripts/monitor_daemon.py" --profile "$PROFILE_PATH"
