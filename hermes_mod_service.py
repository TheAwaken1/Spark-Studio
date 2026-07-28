"""Lifecycle and sandbox bridge for the optional Hermes Mod skin editor.

Spark Studio installs a pinned upstream npm package under data/, launches it on
an unadvertised loopback port, and exposes it only through a sandboxed iframe.
Its HERMES_HOME is always Agent Lab's isolated profile, never ~/.hermes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import signal
import socket
from collections import deque
from pathlib import Path
from typing import Any

import httpx

import agentlab

PACKAGE_NAME = "hermes-mod"
PACKAGE_VERSION = "0.2.0"
PACKAGE_SPEC = f"{PACKAGE_NAME}@{PACKAGE_VERSION}"
ADDON_DIR = agentlab.APP_DIR / "data" / "addons" / "hermes-mod"
PACKAGE_DIR = ADDON_DIR / "node_modules" / PACKAGE_NAME
SERVER_PATH = PACKAGE_DIR / "server.js"
PACKAGE_JSON = PACKAGE_DIR / "package.json"
LOG_PATH = ADDON_DIR / "hermes-mod.log"

BRIDGE_PREFIX = "/api/hermes-mod/ui"
BRIDGE_HEADER = "X-Spark-Studio-Hermes-Mod"
BRIDGE_TOKEN = secrets.token_urlsafe(32)

_state = "stopped"  # stopped | installing | starting | ready | error
_error: str | None = None
_port: int | None = None
_process: asyncio.subprocess.Process | None = None
_log_task: asyncio.Task | None = None
_lock = asyncio.Lock()
_log_tail: deque[str] = deque(maxlen=80)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _node_executable() -> str | None:
    return shutil.which("node")


def _npm_executable() -> str | None:
    return shutil.which("npm")


def _installed_version() -> str | None:
    try:
        data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
        return str(data.get("version") or "") or None
    except (OSError, ValueError, TypeError):
        return None


def installed() -> bool:
    return SERVER_PATH.is_file() and _installed_version() == PACKAGE_VERSION


def _hermes_runtime() -> tuple[Path | None, Path | None]:
    candidates: list[Path] = []
    override = os.environ.get("HERMES_AGENT_ROOT", "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        [
            Path.home() / ".hermes" / "hermes-agent",
            Path.home() / "hermes-agent",
        ]
    )
    binary = agentlab.find_hermes()
    if binary:
        resolved = Path(binary).expanduser().resolve()
        candidates.extend([resolved.parent.parent, resolved.parent])

    seen: set[Path] = set()
    for root in candidates:
        try:
            root = root.resolve()
        except OSError:
            continue
        if root in seen:
            continue
        seen.add(root)
        if not (root / "hermes_cli" / "skin_engine.py").is_file():
            continue
        for python in (
            root / "venv" / "bin" / "python",
            root / "env" / "bin" / "python",
            root / ".venv" / "bin" / "python",
        ):
            if python.is_file():
                return root, python
        return root, None
    return None, None


def managed_url() -> str | None:
    if _state == "ready" and _port:
        return f"http://127.0.0.1:{_port}"
    return None


async def _run_command(*args: str, timeout: float) -> tuple[int, str]:
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, output.decode(errors="replace").strip()
    except FileNotFoundError:
        return 127, f"{args[0]} is not installed"
    except asyncio.TimeoutError:
        if proc:
            proc.kill()
        return 124, "command timed out"
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


async def install() -> dict[str, Any]:
    """Install the pinned package into Spark Studio's git-ignored data dir."""
    global _state, _error
    async with _lock:
        if installed():
            _state, _error = "stopped", None
            return await status()
        npm = _npm_executable()
        if not npm:
            _state, _error = "error", "npm is not installed"
            return await status()
        _state, _error = "installing", None
        ADDON_DIR.mkdir(parents=True, exist_ok=True)
        code, output = await _run_command(
            npm,
            "install",
            "--prefix",
            str(ADDON_DIR),
            "--save-exact",
            "--no-audit",
            "--no-fund",
            PACKAGE_SPEC,
            timeout=240,
        )
        if code != 0 or not installed():
            _state = "error"
            _error = f"npm install failed: {output[-1200:]}" if output else "npm install failed"
        else:
            _state, _error = "stopped", None
        return await status()


async def _capture_output(stream: asyncio.StreamReader) -> None:
    ADDON_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("ab") as log:
        while True:
            line = await stream.readline()
            if not line:
                return
            log.write(line)
            log.flush()
            _log_tail.append(line.decode(errors="replace").rstrip())


async def _healthy(timeout: float = 2.0) -> bool:
    if not _port:
        return False
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://127.0.0.1:{_port}/api/meta")
        return response.status_code == 200
    except Exception:  # noqa: BLE001
        return False


async def start(timeout: float = 30.0) -> dict[str, Any]:
    """Start the installed editor on an unadvertised loopback port."""
    global _state, _error, _port, _process, _log_task
    async with _lock:
        if _process and _process.returncode is None and await _healthy():
            _state, _error = "ready", None
            return await status()
        if not installed():
            _state, _error = "error", "Skin Studio is not installed"
            return await status()
        node = _node_executable()
        if not node:
            _state, _error = "error", "Node.js is not installed"
            return await status()

        _state, _error = "starting", None
        _port = _free_port()
        root, python = _hermes_runtime()
        env = os.environ.copy()
        env.update(
            {
                "PORT": str(_port),
                "NODE_ENV": "production",
                "HERMES_HOME": str(agentlab.HERMES_HOME),
            }
        )
        if root:
            env["HERMES_AGENT_ROOT"] = str(root)
        if python:
            env["HERMES_PYTHON"] = str(python)
        agentlab.HERMES_HOME.mkdir(parents=True, exist_ok=True)
        ADDON_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _process = await asyncio.create_subprocess_exec(
                node,
                str(SERVER_PATH),
                cwd=str(PACKAGE_DIR),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001
            _state, _error = "error", f"could not start Skin Studio: {exc}"
            return await status()
        assert _process.stdout is not None
        _log_task = asyncio.create_task(_capture_output(_process.stdout))

        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if _process.returncode is not None:
                _state = "error"
                _error = "Skin Studio exited during startup"
                if _log_tail:
                    _error += f": {_log_tail[-1]}"
                return await status()
            if await _healthy():
                _state, _error = "ready", None
                return await status()
            await asyncio.sleep(0.25)
        _state, _error = "error", "Skin Studio did not become ready in time"
        return await status()


async def stop() -> dict[str, Any]:
    """Stop only Hermes Mod; the model engine and Chat remain untouched."""
    global _state, _error, _process, _log_task, _port
    async with _lock:
        process = _process
        if process and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=4)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()
        if _log_task:
            try:
                await asyncio.wait_for(_log_task, timeout=1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                _log_task.cancel()
        _process, _log_task, _port = None, None, None
        _state, _error = "stopped", None
        return await status()


async def status() -> dict[str, Any]:
    global _state, _error
    running = bool(_process and _process.returncode is None)
    healthy = running and await _healthy()
    if healthy:
        _state, _error = "ready", None
    elif _state == "ready":
        _state = "starting" if running else "stopped"
    root, python = _hermes_runtime()
    return {
        "package": PACKAGE_NAME,
        "pinned_version": PACKAGE_VERSION,
        "installed_version": _installed_version(),
        "installed": installed(),
        "node": _node_executable(),
        "npm": _npm_executable(),
        "running": running,
        "healthy": healthy,
        "state": _state,
        "error": _error,
        "profile": str(agentlab.HERMES_HOME),
        "active_skin": agentlab.active_hermes_skin(),
        "hermes_agent_root": str(root) if root else None,
        "hermes_python": str(python) if python else None,
        "skin_engine_found": bool(root),
        "log_tail": list(_log_tail)[-12:],
    }


def _rewrite_html(content: bytes) -> bytes:
    text = content.decode("utf-8")
    text = text.replace('href="/styles.css"', 'href="styles.css"')
    text = text.replace('src="/app.js"', 'src="app.js"')
    return text.encode("utf-8")


_BROWSER_PICKER = r"""async function chooseHeroImage() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/png,image/jpeg,image/gif,image/webp';
  const file = await new Promise((resolve) => {
    input.addEventListener('change', () => resolve(input.files?.[0] || null), { once: true });
    input.addEventListener('cancel', () => resolve(null), { once: true });
    input.click();
  });
  if (!file) {
    setStatus('Image selection canceled', 'normal');
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    setStatus('Choose an image smaller than 8 MB', 'error');
    return;
  }
  const imageData = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener('load', () => resolve(String(reader.result || '')), { once: true });
    reader.addEventListener('error', () => reject(new Error('Could not read that image')), { once: true });
    reader.readAsDataURL(file);
  });
  setHeroGeneratorSource(imageData, file.name);
  setStatus('Generating hero art...', 'normal');
  triggerHeroGeneration();
}

function handleHeroStyleChange"""


def _rewrite_javascript(content: bytes) -> bytes:
    text = content.decode("utf-8")
    text = f"const SPARK_STUDIO_BRIDGE = '{BRIDGE_PREFIX}';\n" + text
    text = text.replace(
        "const response = await fetch(path, {",
        "const response = await fetch(`${SPARK_STUDIO_BRIDGE}${path}`, {",
        1,
    )
    text = text.replace(
        "headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },",
        "headers: { 'Content-Type': 'application/json', "
        f"'{BRIDGE_HEADER}': '{BRIDGE_TOKEN}', ...(options.headers || {{}}) }},",
        1,
    )
    text, count = re.subn(
        r"async function chooseHeroImage\(\) \{.*?\n\}\n\nfunction handleHeroStyleChange",
        _BROWSER_PICKER,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Hermes Mod image-picker patch no longer matches the pinned package")
    text = text.replace(
        "setStatus('Opening system image picker...', 'normal');",
        "setStatus('Opening browser image picker...', 'normal');",
        1,
    )
    return text.encode("utf-8")


def rewrite_response(path: str, content_type: str, content: bytes) -> bytes:
    """Adapt root-relative assets/API calls to Spark Studio's iframe bridge."""
    clean = path.strip("/")
    if clean in {"", "index.html"} and "text/html" in content_type:
        return _rewrite_html(content)
    if clean == "app.js" and "javascript" in content_type:
        return _rewrite_javascript(content)
    return content


def valid_bridge_token(value: str | None) -> bool:
    return bool(value) and secrets.compare_digest(value, BRIDGE_TOKEN)


def _proxy_sync(
    method: str,
    target: str,
    headers: dict[str, str],
    body: bytes,
) -> tuple[int, dict[str, str], bytes]:
    # Activation and Spark Studio's config refresh share this lock, preventing
    # either process from winning with a stale read of config.yaml.
    lock = agentlab._CONFIG_LOCK if "/api/activate/" in target else None
    if lock:
        lock.acquire()
    try:
        response = httpx.request(
            method,
            target,
            headers=headers,
            content=body,
            timeout=45,
        )
    finally:
        if lock:
            lock.release()
    response_headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower() in {"content-type", "cache-control", "content-disposition"}
    }
    return response.status_code, response_headers, response.content


async def proxy_request(
    method: str,
    path: str,
    query: str,
    headers: dict[str, str],
    body: bytes,
) -> tuple[int, dict[str, str], bytes]:
    url = managed_url()
    if not url or not await _healthy():
        raise RuntimeError("Skin Studio is not running")
    clean = path.lstrip("/")
    if ".." in Path(clean).parts:
        raise ValueError("invalid Skin Studio path")
    target = f"{url}/{clean}"
    if query:
        target = f"{target}?{query}"
    return await asyncio.to_thread(_proxy_sync, method, target, headers, body)
