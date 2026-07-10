#!/usr/bin/env bash
# Launch Spark Studio, listening on all interfaces so other machines on the
# LAN can reach it. Override with e.g. ./start.sh --host 127.0.0.1 --port 8000
# On first run this also creates ./env and installs requirements.txt.
set -euo pipefail

cd "$(dirname "$0")"

HOST=0.0.0.0
PORT=7860
NO_SPARKRUN_UPDATE="${SPARK_STUDIO_NO_SPARKRUN_UPDATE:-0}"
DOCTOR=0
UPDATE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --no-sparkrun-update) NO_SPARKRUN_UPDATE=1; shift ;;
        --doctor) DOCTOR=1; shift ;;
        --update) UPDATE=1; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# First run: create the virtualenv and install dependencies automatically.
if [[ ! -x env/bin/python ]]; then
    echo "First run — setting up the Python environment in ./env …"
    if command -v uv >/dev/null 2>&1; then
        uv venv env --python 3.11 || uv venv env
        uv pip install --python env/bin/python -r requirements.txt
    else
        PY=$(command -v python3.11 || command -v python3 || true)
        if [[ -z "$PY" ]]; then
            echo "error: no python3 found — install Python 3.11 (or uv) first" >&2
            exit 1
        fi
        "$PY" -m venv env
        env/bin/pip install --quiet --upgrade pip
        env/bin/pip install -r requirements.txt
    fi
    echo "Environment ready."
fi

# Update mode: pull the latest code + refresh dependencies, then continue
# into a normal start (one command to be current *and* running).
if [[ "$UPDATE" == "1" ]]; then
    OLD_VER=$(cat VERSION 2>/dev/null || echo "?")
    echo "Updating Spark Studio (currently v$OLD_VER)…"
    if git rev-parse --git-dir >/dev/null 2>&1; then
        git pull --ff-only || echo "warning: git pull failed (local changes or diverged branch) — continuing with the current code" >&2
    else
        echo "warning: not a git checkout — cannot self-update; re-clone from GitHub instead" >&2
    fi
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python env/bin/python -r requirements.txt --upgrade --quiet
    else
        env/bin/pip install --quiet -r requirements.txt --upgrade
    fi
    NEW_VER=$(cat VERSION 2>/dev/null || echo "?")
    if [[ "$NEW_VER" != "$OLD_VER" ]]; then
        echo "Updated: v$OLD_VER → v$NEW_VER"
    else
        echo "Up to date (v$NEW_VER)"
    fi
fi

# Doctor mode: print the system health report and exit (no server start).
if [[ "$DOCTOR" == "1" ]]; then
    exec env/bin/python doctor.py
fi

# Keep sparkrun fresh on launch: a bare `sparkrun update` upgrades the tool
# and refreshes recipe registries while staying on the channel last picked
# (in the app's Community tab or via the CLI). Skip with --no-sparkrun-update
# or SPARK_STUDIO_NO_SPARKRUN_UPDATE=1; never block startup on failure.
if [[ "$NO_SPARKRUN_UPDATE" != "1" ]] && command -v sparkrun >/dev/null 2>&1; then
    echo "Updating sparkrun (current channel) …"
    timeout 300 sparkrun update || echo "warning: sparkrun update failed — continuing with $(sparkrun --version 2>/dev/null || echo 'current install')" >&2
fi

# Fail fast with a useful message if the port is taken (uvicorn only errors
# after "Application startup complete", which reads like a successful boot).
if ! env/bin/python - "$PORT" <<'PY'
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
then
    echo "error: port $PORT is already in use." >&2
    holder=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
    [[ -n "${holder:-}" ]] && echo "  in use by: $(ps -p "$holder" -o pid=,cmd= 2>/dev/null)" >&2
    echo "  another Spark Studio may already be running — check http://127.0.0.1:$PORT" >&2
    echo "  or start on a different port: ./start.sh --port $((PORT + 1))" >&2
    exit 1
fi

echo "Spark Studio starting on port $PORT"
echo "  Local:   http://127.0.0.1:$PORT"
if [[ "$HOST" == "0.0.0.0" ]]; then
    for ip in $(hostname -I); do
        [[ "$ip" == *:* ]] && continue  # skip IPv6
        echo "  Network: http://$ip:$PORT"
    done
fi

exec env/bin/python -m uvicorn server:app --host "$HOST" --port "$PORT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
