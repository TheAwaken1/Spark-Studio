#!/usr/bin/env bash
# Spark Studio one-command installer for NVIDIA DGX Spark.
#
#   Interactive:      bash <(curl -fsSL https://raw.githubusercontent.com/TheAwaken1/Spark-Studio/main/install.sh)
#   Non-interactive:  curl -fsSL https://raw.githubusercontent.com/TheAwaken1/Spark-Studio/main/install.sh | bash
#
# IMPORTANT: when piped (`curl | bash`), stdin is the script itself, so no
# prompts are possible — this script detects that and uses safe defaults
# (Recommended profile, ~/spark-studio, no auto-start). Run it as a file or
# via process substitution to get the interactive prompts.
#
# Options (flags or env vars):
#   --dir <path>        install directory        (SPARK_STUDIO_DIR,     default ~/spark-studio)
#   --basic|--recommended|--full                 (SPARK_STUDIO_PROFILE, default recommended)
#   --repo <url>        git remote               (SPARK_STUDIO_REPO)
#   --no-start          never offer to start the app
#
# Profiles:
#   basic        dashboard + recipes + local runs (Python env only)
#   recommended  basic + sparkrun CLI (community recipes, kernel tuning)
#   full         recommended + llama-benchy + Claude/Codex CLIs (needs npm)

set -euo pipefail

REPO="${SPARK_STUDIO_REPO:-https://github.com/TheAwaken1/Spark-Studio.git}"
DIR="${SPARK_STUDIO_DIR:-$HOME/spark-studio}"
PROFILE="${SPARK_STUDIO_PROFILE:-recommended}"
NO_START=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) DIR="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --basic) PROFILE=basic; shift ;;
        --recommended) PROFILE=recommended; shift ;;
        --full) PROFILE=full; shift ;;
        --no-start) NO_START=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# Piped installs have no usable stdin — everything must default sanely.
INTERACTIVE=0
[[ -t 0 ]] && INTERACTIVE=1

ok()   { printf '✅ %s\n' "$1"; }
warn() { printf '⚠️  %s\n' "$1"; }
fail() { printf '❌ %s\n' "$1" >&2; exit 1; }
ask() { # ask "question" "default(y/n)" — returns 0 for yes; auto-default when piped
    local q="$1" def="${2:-y}" reply
    if [[ "$INTERACTIVE" != "1" ]]; then
        [[ "$def" == "y" ]] && return 0 || return 1
    fi
    read -r -p "$q " reply || reply=""
    reply="${reply:-$def}"
    [[ "$reply" =~ ^[Yy] ]]
}

echo "Spark Studio Installer"
echo "  profile: $PROFILE · directory: $DIR"
echo

# ----- prerequisite checks --------------------------------------------------
[[ "$(uname -s)" == "Linux" ]] || fail "Linux required (this targets NVIDIA DGX Spark)."
ok "Linux detected ($(uname -m))"
[[ "$(uname -m)" == "aarch64" ]] || warn "not aarch64 — fine for x86_64 Linux, but this is tuned for DGX Spark"

command -v git >/dev/null 2>&1 || fail "git is required — install it first (sudo apt install git)."
ok "git found"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    ok "NVIDIA GPU detected${GPU_NAME:+ ($GPU_NAME)}"
else
    warn "nvidia-smi not found — GPU telemetry and fit checks will be disabled"
fi

# Python 3.11+ or uv (uv can provision Python by itself).
if command -v uv >/dev/null 2>&1; then
    ok "uv found ($(uv --version 2>/dev/null | head -1))"
elif command -v python3.11 >/dev/null 2>&1 || python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
    ok "Python 3.11+ found"
else
    warn "neither uv nor Python 3.11+ found"
    if ask "Install uv now (official installer from astral.sh)? [Y/n]" y; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        command -v uv >/dev/null 2>&1 || fail "uv install did not land on PATH — open a new shell and re-run."
        ok "uv installed"
    else
        fail "Python 3.11+ or uv is required."
    fi
fi

command -v docker >/dev/null 2>&1 \
    && ok "Docker found" \
    || warn "Docker not found — bundled Web Search and spark-vllm-docker recipes will be disabled"

if [[ "$PROFILE" == "full" ]]; then
    command -v npm >/dev/null 2>&1 \
        && ok "npm found (needed for Claude/Codex agent CLIs)" \
        || warn "npm not found — agent CLIs will be skipped (install Node.js later to enable them)"
fi

# ----- clone / update ---------------------------------------------------------
echo
if [[ -d "$DIR/.git" ]]; then
    echo "Updating existing install in $DIR …"
    git -C "$DIR" pull --ff-only || warn "git pull failed (local changes?) — continuing with the current checkout"
elif [[ -e "$DIR" && -n "$(ls -A "$DIR" 2>/dev/null)" ]]; then
    fail "$DIR exists and is not a Spark Studio checkout — pick another spot with --dir"
else
    echo "Cloning Spark Studio into $DIR …"
    git clone "$REPO" "$DIR" || fail "clone failed — if the repo is private, authenticate first (gh auth login) or use --repo"
fi
cd "$DIR"

# ----- python environment (start.sh owns the bootstrap; doctor exercises it) --
echo
echo "Setting up the Python environment (first run can take a few minutes)…"
bash start.sh --doctor || true   # non-zero just means a core check warned — report is what matters

# ----- profile extras ---------------------------------------------------------
if [[ "$PROFILE" == "recommended" || "$PROFILE" == "full" ]]; then
    if ! command -v sparkrun >/dev/null 2>&1; then
        if ask "Install sparkrun (community recipes + kernel tuning)? [Y/n]" y; then
            uv tool install sparkrun && ok "sparkrun installed" \
                || warn "sparkrun install failed — run 'uvx sparkrun setup' later"
            echo "   ↳ run 'sparkrun setup' later for the guided cluster/earlyoom wizard"
        fi
    else
        ok "sparkrun already installed"
    fi
fi
if [[ "$PROFILE" == "full" ]]; then
    echo "Installing llama-benchy (benchmark sweeps)…"
    uv pip install --python env/bin/python llama-benchy >/dev/null 2>&1 \
        && ok "llama-benchy installed" || warn "llama-benchy install failed — install later from the Benchmarks tab hint"
    if command -v npm >/dev/null 2>&1; then
        if ask "Install Claude Code + Codex agent CLIs via npm -g? [Y/n]" y; then
            npm install -g @anthropic-ai/claude-code @openai/codex \
                && ok "agent CLIs installed (log in from the Agents tab)" \
                || warn "npm install failed — install the agent CLIs later"
        fi
    fi
fi

# ----- done -------------------------------------------------------------------
echo
ok "Spark Studio is installed in $DIR"
echo
echo "Start it with:"
echo "    cd \"$DIR\" && ./start.sh"
echo
if [[ "$NO_START" != "1" ]] && ask "Start Spark Studio now? [Y/n]" y && [[ "$INTERACTIVE" == "1" ]]; then
    exec ./start.sh
fi
echo "(then open http://127.0.0.1:7860 — the first-run wizard takes it from there)"
