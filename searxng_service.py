"""App-managed SearXNG search engine (Docker).

Spark Studio's web search prefers a SearXNG meta-search backend over the fragile
DuckDuckGo scraper. Rather than depend on a separately-installed instance, we run
the official ``searxng/searxng`` image ourselves as a singleton container bound to
localhost and hand its URL to the discovery code in ``server.py``.

Lifecycle is best-effort: every function swallows its own errors and reflects the
outcome through ``status()`` / module state instead of raising into the caller, so
a missing/broken Docker never takes down the app. Auto-started from the FastAPI
startup hook; also exposed via start/stop endpoints for manual recovery.
"""

from __future__ import annotations

import asyncio
import secrets
import socket
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "data" / "searxng"
SETTINGS_PATH = CONFIG_DIR / "settings.yml"
PORT_PATH = CONFIG_DIR / "port"

CONTAINER_NAME = "spark-searxng"
IMAGE = "searxng/searxng:latest"
CONTAINER_PORT = 8080  # SearXNG listens on 8080 inside the image

# Bump when the generated settings.yml changes so existing installs regenerate
# (and the container is recreated) instead of keeping a stale config.
CONFIG_VERSION = 4
_CONFIG_MARKER = "# spark-studio-config-version:"

# Which engines answer a self-hosted instance shifts over time, so we enable a
# redundant set and let SearXNG aggregate whatever responds. Notably:
#   - bing is disabled: it matches only the first word of the query and returns
#     navigational junk (dictionary pages, google.com) that outranks real hits.
#   - duckduckgo/brave/presearch currently answer reliably without keys;
#     mojeek works but rate-limits after a few rapid queries, hence the backups.
#   - startpage/qwant/karmasearch block or CAPTCHA self-hosted IPs.
# Merged over defaults by name via use_default_settings.
_ENGINE_OVERRIDES = [
    {"name": "bing", "disabled": True},          # first-word-only junk results
    {"name": "mojeek", "disabled": False},
    {"name": "duckduckgo", "disabled": False},
    {"name": "brave", "disabled": False},
    {"name": "presearch", "disabled": False},
    {"name": "startpage", "disabled": True},     # CAPTCHA
    {"name": "qwant", "disabled": True},         # access denied
    {"name": "karmasearch", "disabled": True},   # access denied
    {"name": "wikipedia", "disabled": False},
    {"name": "wikidata", "disabled": False},
]

# Module state, updated by ensure_started()/status(). managed_url() reads these so
# discovery stays non-blocking.
_state = "stopped"  # stopped | starting | ready | error
_error: str | None = None
_port: int | None = None
_url: str | None = None
_lock = asyncio.Lock()


# ----- helpers --------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _run(*args: str, timeout: float = 30.0) -> tuple[int, str]:
    """Run a subprocess, returning (returncode, combined_output). Never raises."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode if proc.returncode is not None else 1, out.decode(errors="replace").strip()
    except FileNotFoundError:
        return 127, "docker not found"
    except asyncio.TimeoutError:
        return 124, "timed out"
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


async def _docker_available() -> bool:
    code, _ = await _run("docker", "version", "--format", "{{.Server.Version}}", timeout=10)
    return code == 0


async def _inspect(fmt: str) -> str | None:
    """docker inspect the container with a Go template; None if it doesn't exist."""
    code, out = await _run("docker", "inspect", "-f", fmt, CONTAINER_NAME, timeout=10)
    if code != 0:
        return None
    return out


async def _container_running() -> bool:
    return (await _inspect("{{.State.Running}}")) == "true"


async def _container_exists() -> bool:
    return (await _inspect("{{.Name}}")) is not None


async def _container_host_port() -> int | None:
    """Host port the running container publishes 8080 on, if any."""
    out = await _inspect(
        '{{(index (index .NetworkSettings.Ports "%d/tcp") 0).HostPort}}' % CONTAINER_PORT
    )
    try:
        return int(out) if out else None
    except (TypeError, ValueError):
        return None


def _existing_secret_key() -> str | None:
    try:
        data = yaml.safe_load(SETTINGS_PATH.read_text()) or {}
        return (data.get("server") or {}).get("secret_key")
    except Exception:  # noqa: BLE001
        return None


def _write_config() -> bool:
    """Generate settings.yml if missing or out of date. Returns True if written.

    Uses SearXNG's ``use_default_settings`` merge so we only override essentials:
    a stable secret key, the disabled bot-detection limiter (localhost JSON API),
    the JSON result format (off by default, required by /api/search), and a curated
    engine set (see _ENGINE_OVERRIDES). Preserves an existing secret key across
    config upgrades so we don't needlessly churn it.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    marker = f"{_CONFIG_MARKER} {CONFIG_VERSION}"
    if SETTINGS_PATH.exists() and marker in SETTINGS_PATH.read_text()[:200]:
        return False
    secret = _existing_secret_key() or secrets.token_hex(32)
    settings = {
        "use_default_settings": True,
        "server": {
            "secret_key": secret,
            "limiter": False,
            "image_proxy": True,
        },
        "search": {
            "formats": ["html", "json"],
        },
        "engines": _ENGINE_OVERRIDES,
    }
    body = yaml.safe_dump(settings, default_flow_style=False, sort_keys=False)
    SETTINGS_PATH.write_text(f"{marker}\n{body}")
    return True


def _desired_port() -> int:
    """Persisted host port, chosen once and reused across restarts."""
    try:
        return int(PORT_PATH.read_text().strip())
    except (OSError, ValueError):
        port = _free_port()
        try:
            PORT_PATH.write_text(str(port))
        except OSError:
            pass
        return port


async def _healthy(port: int, timeout: float = 5.0) -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(
                f"http://127.0.0.1:{port}/search",
                params={"q": "ping", "format": "json"},
            )
        return r.status_code < 400
    except Exception:  # noqa: BLE001
        return False


# ----- public API -----------------------------------------------------------

def managed_url() -> str | None:
    """Base URL of the app-managed SearXNG, or None. Non-blocking (reads state)."""
    return _url if _state == "ready" else None


async def ensure_started(timeout: float = 120.0) -> None:
    """Idempotently bring the SearXNG container up and wait until it answers JSON.

    Safe to call on every startup: no-ops when already healthy, starts a stopped
    container, or pulls+runs the image on first use. Best-effort — records failure
    in module state rather than raising.
    """
    global _state, _error, _port, _url

    async with _lock:
        if _state == "ready" and _port and await _healthy(_port):
            return

        _state, _error = "starting", None

        if not await _docker_available():
            _state, _error = "error", "Docker is not available"
            return

        # Refresh settings.yml before touching the container so a config upgrade
        # forces a recreate with the new engine set.
        config_changed = _write_config()

        port = await _container_host_port() if await _container_running() else _desired_port()
        _port = port
        _url = f"http://127.0.0.1:{port}"

        if config_changed and await _container_exists():
            await _run("docker", "rm", "-f", CONTAINER_NAME, timeout=30)

        if await _container_running():
            if await _wait_healthy(port, timeout=30):
                return
            # Running but unhealthy — recreate from scratch.
            await _run("docker", "rm", "-f", CONTAINER_NAME, timeout=30)

        elif await _container_exists():
            code, out = await _run("docker", "start", CONTAINER_NAME, timeout=30)
            if code != 0:
                # Stale/misconfigured container — recreate.
                await _run("docker", "rm", "-f", CONTAINER_NAME, timeout=30)

        if not await _container_running():
            code, out = await _run(
                "docker", "run", "-d",
                "--name", CONTAINER_NAME,
                "--restart", "unless-stopped",
                "-p", f"127.0.0.1:{port}:{CONTAINER_PORT}",
                # Mount only settings.yml read-only (not the whole dir): the image's
                # entrypoint chowns its config dir to the container user, which would
                # otherwise lock the host out of managing/upgrading the file.
                "-v", f"{SETTINGS_PATH}:/etc/searxng/settings.yml:ro",
                IMAGE,
                timeout=timeout,  # first run pulls the image
            )
            if code != 0:
                _state, _error = "error", f"docker run failed: {out}"
                return

        if await _wait_healthy(port, timeout=timeout):
            return
        _state, _error = "error", "SearXNG did not become healthy in time"


async def _wait_healthy(port: int, timeout: float) -> bool:
    global _state, _error, _url
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if await _healthy(port):
            _url = f"http://127.0.0.1:{port}"
            _state, _error = "ready", None
            return True
        await asyncio.sleep(2)
    return False


async def stop() -> None:
    """Stop the container (best-effort). Leaves it present for a fast restart."""
    global _state
    await _run("docker", "stop", CONTAINER_NAME, timeout=30)
    _state = "stopped"


async def status() -> dict:
    """Live status for the API/UI. Probes health when the container is running."""
    global _state, _url, _port
    docker_ok = await _docker_available()
    exists = docker_ok and await _container_exists()
    running = docker_ok and await _container_running()
    healthy = False
    if running:
        port = await _container_host_port() or _port or _desired_port()
        _port = port
        healthy = await _healthy(port)
        if healthy:
            _url = f"http://127.0.0.1:{port}"
            _state = "ready"
        elif _state == "ready":
            _state = "starting"
    elif _state == "ready":
        _state = "stopped"
    return {
        "docker": docker_ok,
        "exists": exists,
        "running": running,
        "healthy": healthy,
        "state": _state,
        "url": _url if healthy else None,
        "port": _port,
        "error": _error,
    }
