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
MODE=serve   # serve | desktop | install-launcher | install-service
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --no-sparkrun-update) NO_SPARKRUN_UPDATE=1; shift ;;
        --doctor) DOCTOR=1; shift ;;
        --update) UPDATE=1; shift ;;
        --desktop) MODE=desktop; shift ;;
        --install-launcher) MODE=install-launcher; shift ;;
        --install-service) MODE=install-service; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

APP_DIR="$(pwd)"

# ----- desktop launcher (.desktop file) --------------------------------------
if [[ "$MODE" == "install-launcher" ]]; then
    mkdir -p "$HOME/.local/share/applications"
    DESK="$HOME/.local/share/applications/spark-studio.desktop"
    cat > "$DESK" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Spark Studio
Comment=DGX Spark inference dashboard
Exec=/bin/bash "$APP_DIR/start.sh" --desktop
Icon=$APP_DIR/icon.png
Terminal=false
Categories=Development;Utility;
DESKTOP
    chmod +x "$DESK"
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    echo "Desktop launcher installed: $DESK"
    echo "Find 'Spark Studio' in your application menu — it starts the server if needed and opens the dashboard."
    exit 0
fi

# ----- user-level systemd service ---------------------------------------------
if [[ "$MODE" == "install-service" ]]; then
    mkdir -p "$HOME/.config/systemd/user"
    UNIT="$HOME/.config/systemd/user/spark-studio.service"
    cat > "$UNIT" <<UNITEOF
[Unit]
Description=Spark Studio — DGX Spark inference dashboard
After=network-online.target

[Service]
Type=exec
WorkingDirectory=$APP_DIR
ExecStart=/bin/bash "$APP_DIR/start.sh" --no-sparkrun-update
# Restart=always makes the one-click in-app update work: apply exits cleanly
# and systemd brings the new version up. 'systemctl --user stop' still stops.
Restart=always
RestartSec=5
# Models keep serving across service restarts; the next boot re-adopts them.
Environment=SPARK_STUDIO_KEEP_RUNS_ON_EXIT=1
# Tells the app it may self-restart after an in-app update.
Environment=SPARK_STUDIO_SERVICE=1
# NOTE: user units cannot LOWER oom_score_adj without privilege. For full OOM
# protection of the dashboard, apply the earlyoom fix from the README instead.

[Install]
WantedBy=default.target
UNITEOF
    systemctl --user daemon-reload
    systemctl --user enable spark-studio.service >/dev/null 2>&1 || true
    echo "Service installed + enabled: $UNIT"
    echo
    echo "  start now:        systemctl --user start spark-studio"
    echo "  logs:             journalctl --user -u spark-studio -f"
    echo "  survive logout:   loginctl enable-linger $USER   (may need sudo)"
    echo
    echo "Note: stop any terminal-launched ./start.sh first — they share port $PORT."
    exit 0
fi

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

# Desktop mode (used by the .desktop launcher): make sure the server is up,
# then open the dashboard in the default browser. Never blocks a terminal.
if [[ "$MODE" == "desktop" ]]; then
    URL="http://127.0.0.1:$PORT"
    _serving() { curl -sf -m 2 -o /dev/null "$URL/api/system"; }
    if ! _serving; then
        if systemctl --user is-enabled spark-studio.service >/dev/null 2>&1; then
            systemctl --user start spark-studio.service
        else
            mkdir -p data
            nohup env/bin/python -m uvicorn server:app --host "$HOST" --port "$PORT" \
                >> data/spark-studio.log 2>&1 &
            disown
        fi
        for _ in $(seq 1 60); do _serving && break; sleep 2; done
    fi
    _serving || { echo "Spark Studio did not come up — see data/spark-studio.log" >&2; exit 1; }
    command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || echo "Open: $URL"
    exit 0
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
