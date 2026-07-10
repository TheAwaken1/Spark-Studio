"""Spark Studio Doctor — one source of truth for system health checks.

Aggregates every dependency / hardware / feature probe the app already knows
how to do (hostinfo, runners, sparkrun_service, agents, benchy) into a single
report consumed three ways:

  - CLI:  ./start.sh --doctor   (or: env/bin/python doctor.py)
  - API:  GET /api/doctor       (powers the first-run wizard + Feature Health)
  - Bug reports: the "Copy Bug Report" flow embeds the same report

Each check returns {id, label, status, detail, fix}:
  status: "ok" — working;  "warn" — optional thing missing (feature degrades);
          "error" — core problem worth fixing before serious use.
Checks never raise: a probe that blows up becomes its own "error" entry so one
broken dependency can't hide the rest of the report.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

APP_DIR = Path(__file__).parent
DEFAULT_PORT = int(os.environ.get("SPARK_STUDIO_PORT", "7860"))


def app_version() -> str:
    try:
        return (APP_DIR / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


# ----------------------------- helpers ------------------------------------

def _which_version(cmd: str, args: list[str] | None = None, timeout: int = 6) -> str | None:
    """Version string of a CLI tool, or None when missing/broken."""
    path = shutil.which(cmd)
    if not path:
        return None
    try:
        res = subprocess.run([path] + (args or ["--version"]),
                             capture_output=True, text=True, timeout=timeout)
        out = (res.stdout or res.stderr or "").strip().splitlines()
        return out[0][:80] if out else "installed"
    except Exception:  # noqa: BLE001
        return None


def _check(id_: str, label: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one probe defensively; a crash becomes an 'error' entry."""
    base = {"id": id_, "label": label, "status": "error", "detail": "", "fix": None}
    try:
        base.update(fn())
    except Exception as e:  # noqa: BLE001
        base["detail"] = f"check failed: {e}"
    return base


def _lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=4)
        for ip in (out.stdout or "").split():
            if ":" not in ip and not ip.startswith("127."):  # skip IPv6 + loopback
                ips.append(ip)
    except Exception:  # noqa: BLE001
        pass
    return ips


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ----------------------------- probes -------------------------------------

def _probe_platform() -> dict[str, Any]:
    arch = platform.machine()
    ok = platform.system() == "Linux"
    return {
        "status": "ok" if ok else "error",
        "detail": f"{platform.system()} {platform.release()} · {arch}"
                  + (" (DGX Spark native)" if arch == "aarch64" else ""),
        "fix": None if ok else "Spark Studio targets Linux (DGX Spark).",
    }


def _probe_python() -> dict[str, Any]:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return {
        "status": "ok" if ok else "warn",
        "detail": f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})",
        "fix": None if ok else "Python 3.11+ recommended — `uv venv env --python 3.11`.",
    }


def _probe_tool(cmd: str, why: str, fix: str, required: bool = False):
    def fn() -> dict[str, Any]:
        ver = _which_version(cmd)
        if ver:
            return {"status": "ok", "detail": ver}
        return {"status": "error" if required else "warn",
                "detail": f"not found — {why}", "fix": fix}
    return fn


def _probe_gpu() -> dict[str, Any]:
    import hostinfo
    host = hostinfo.probe_host(force=True)
    if not host["gpu_count"]:
        return {"status": "error",
                "detail": "no NVIDIA GPU detected (nvidia-smi missing or empty)",
                "fix": "Install the NVIDIA driver; verify with `nvidia-smi`."}
    label = host["summary"]
    if host["is_dgx_spark"]:
        label += " · DGX Spark detected"
    if host["mesh_size"] > 1:
        label += f" · mesh ×{host['mesh_size']}"
    return {"status": "ok", "detail": label}


def _probe_driver() -> dict[str, Any]:
    import hostinfo
    gpus = hostinfo.probe_host()["gpus"]
    if not gpus:
        return {"status": "warn", "detail": "no GPU — driver unknown"}
    drv = gpus[0].get("driver") or "?"
    major = drv.split(".")[0]
    if major.isdigit() and int(major) >= 590:
        # eugr/spark-vllm-docker: 590.x has a CUDAGraph capture deadlock on GB10.
        return {"status": "warn", "detail": f"driver {drv}",
                "fix": "Driver 590.x has a known CUDAGraph deadlock on GB10 — 580.x is recommended."}
    return {"status": "ok", "detail": f"driver {drv}"}


def _probe_memory() -> dict[str, Any]:
    import psutil
    vm = psutil.virtual_memory()
    total, avail = vm.total / 1024**3, vm.available / 1024**3
    status = "ok" if avail > 8 else "warn"
    return {"status": status,
            "detail": f"{avail:.0f} GB free / {total:.0f} GB unified",
            "fix": None if status == "ok" else
            "Little free memory — a resident model may block the next launch (the memory guard will stop it first)."}


def _probe_docker() -> dict[str, Any]:
    if not shutil.which("docker"):
        return {"status": "warn", "detail": "not installed — Web Search container + docker recipes disabled",
                "fix": "Install Docker to enable bundled SearXNG and spark-vllm-docker recipes."}
    try:
        res = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                             capture_output=True, text=True, timeout=8)
        if res.returncode == 0 and res.stdout.strip():
            return {"status": "ok", "detail": f"docker {res.stdout.strip()}"}
        return {"status": "warn", "detail": "installed but daemon not reachable",
                "fix": "Start the Docker daemon (`sudo systemctl start docker`) or add your user to the docker group."}
    except Exception as e:  # noqa: BLE001
        return {"status": "warn", "detail": f"daemon check failed: {e}"}


def _probe_engine(engine: str, hint: str):
    def fn() -> dict[str, Any]:
        from runners import engine_available
        if engine_available(engine):
            return {"status": "ok", "detail": "installed"}
        return {"status": "warn", "detail": "not installed — engine unavailable", "fix": hint}
    return fn


def _probe_sparkrun() -> dict[str, Any]:
    import sparkrun_service
    ver = sparkrun_service.version()
    if ver:
        return {"status": "ok", "detail": f"sparkrun {ver}"}
    return {"status": "warn", "detail": "not installed — Community Recipes disabled",
            "fix": "Run `uvx sparkrun setup` in a terminal (guided cluster wizard)."}


def _probe_agent(which: str) -> dict[str, Any]:
    import agents
    installed = agents.claude_available() if which == "claude" else agents.codex_available()
    if not installed:
        return {"status": "warn",
                "detail": f"not installed — Ask {'Claude' if which == 'claude' else 'Codex'} / Auto-Fix disabled",
                "fix": "npm install -g " + ("@anthropic-ai/claude-code" if which == "claude" else "@openai/codex")}
    home = Path.home()
    logged_in = (
        (home / ".claude" / ".credentials.json").exists() or (home / ".claude" / "session.json").exists()
        if which == "claude" else (home / ".codex" / "auth.json").exists()
    )
    if logged_in:
        return {"status": "ok", "detail": "installed · logged in"}
    return {"status": "warn", "detail": "installed but not logged in",
            "fix": "Log in from the Agents tab (browser OAuth, no API key)."}


def _probe_benchy() -> dict[str, Any]:
    import benchy
    if benchy.available():
        return {"status": "ok", "detail": "installed"}
    return {"status": "warn", "detail": "not installed — full benchmark sweeps disabled",
            "fix": "uv pip install --python env/bin/python llama-benchy"}


def _probe_searxng() -> dict[str, Any]:
    if not shutil.which("docker"):
        return {"status": "warn", "detail": "needs Docker — web search falls back to DuckDuckGo"}
    try:
        res = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "spark-searxng"],
                             capture_output=True, text=True, timeout=8)
        if res.returncode == 0 and res.stdout.strip() == "true":
            return {"status": "ok", "detail": "spark-searxng container running"}
        return {"status": "warn", "detail": "container not running — starts automatically with the app",
                "fix": None}
    except Exception as e:  # noqa: BLE001
        return {"status": "warn", "detail": f"container check failed: {e}"}


def _probe_port(port: int):
    def fn() -> dict[str, Any]:
        if _port_in_use(port):
            return {"status": "ok",
                    "detail": f"port {port} is serving (Spark Studio appears to be running)"}
        return {"status": "warn", "detail": f"port {port} free — app not running",
                "fix": "./start.sh"}
    return fn


# ----------------------------- report -------------------------------------

def run_checks(port: int = DEFAULT_PORT) -> dict[str, Any]:
    checks = [
        _check("platform", "Operating system", _probe_platform),
        _check("python", "Python", _probe_python),
        _check("uv", "uv", _probe_tool("uv", "falls back to pip", "pip install uv")),
        _check("git", "Git", _probe_tool("git", "registry sync & updates need it",
                                         "sudo apt install git", required=True)),
        _check("node", "Node.js", _probe_tool(
            "node", "only needed for Claude/Codex agent features",
            "Install Node.js, then `npm install -g @anthropic-ai/claude-code @openai/codex`")),
        _check("gpu", "NVIDIA GPU", _probe_gpu),
        _check("driver", "NVIDIA driver", _probe_driver),
        _check("memory", "Unified memory", _probe_memory),
        _check("docker", "Docker", _probe_docker),
        _check("vllm", "vLLM engine", _probe_engine(
            "vllm", 'uv pip install --python env/bin/python "vllm" (or use spark-vllm-docker recipes)')),
        _check("sglang", "SGLang engine", _probe_engine(
            "sglang", 'uv pip install --python env/bin/python "sglang[all]"')),
        _check("llamacpp", "llama.cpp engine", _probe_engine(
            "llamacpp", "conda install -c conda-forge llama.cpp (native llama-server)")),
        _check("sparkrun", "sparkrun", _probe_sparkrun),
        _check("claude", "Claude Code agent", lambda: _probe_agent("claude")),
        _check("codex", "Codex agent", lambda: _probe_agent("codex")),
        _check("benchy", "llama-benchy", _probe_benchy),
        _check("searxng", "Web search (SearXNG)", _probe_searxng),
        _check("port", "Dashboard", _probe_port(port)),
    ]
    counts = {"ok": 0, "warn": 0, "error": 0}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    lan = _lan_ips()
    return {
        "version": app_version(),
        "generated_at": time.time(),
        "checks": checks,
        "summary": counts,
        "urls": {
            "local": f"http://127.0.0.1:{port}",
            "lan": [f"http://{ip}:{port}" for ip in lan],
        },
    }


_ICONS = {"ok": "✅", "warn": "⚠️ ", "error": "❌"}


def format_cli(report: dict[str, Any]) -> str:
    lines = [f"Spark Studio Doctor · v{report['version']}", ""]
    for c in report["checks"]:
        lines.append(f"{_ICONS.get(c['status'], '•')} {c['label']}: {c['detail']}")
        if c.get("fix") and c["status"] != "ok":
            lines.append(f"     ↳ {c['fix']}")
    s = report["summary"]
    lines += ["", f"{s['ok']} ok · {s['warn']} warnings · {s['error']} errors", ""]
    lines.append(f"Local:   {report['urls']['local']}")
    for u in report["urls"]["lan"]:
        lines.append(f"LAN:     {u}")
    return "\n".join(lines)


def main() -> int:
    report = run_checks()
    print(format_cli(report))
    return 1 if report["summary"]["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
