"""Helpers for observing sparkrun-managed workloads.

sparkrun workloads deliberately outlive their launcher process: the container
runs `sleep infinity` and the serve command is `docker exec`'d separately,
logging to /tmp/sparkrun_serve.log INSIDE the container. That means neither
the launcher's exit code nor `docker ps` tells you whether the engine is
actually alive — these helpers do.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from typing import Any

# Community recipe refs look like @official/name or @experimental/name.
REF_RE = re.compile(r"@[\w][\w.-]*/[\w][\w.-]*")
# sparkrun prints the job id in brackets: "Job: @exp/x  (tp=1)  [33bb6cc6567d]".
# Newer sparkrun job ids are underscore-joined hex segments
# ("c46c75a711a7ace8_71cd0a0f80c1") — a hex-only pattern truncates them and
# the resulting `sparkrun stop <half-id>` exits 1 while the model keeps serving.
JOBID_RE = re.compile(r"\[([0-9a-f]{6,}(?:_[0-9a-f]{4,})*)\]")
# Containers are named sparkrun_<jobid>_<role> (role: solo/head/worker...).
# The role must start with a letter so greedy matching keeps every hex
# segment of the jobid in group 1 instead of splitting it at an underscore.
CONTAINER_RE = re.compile(r"sparkrun_([0-9a-f]{6,}(?:_[0-9a-f]{4,})*)_([A-Za-z][\w-]*)")

# Job line: `Job: <ref>  (tp=1)  [<jobid>]` — newer sparkrun adds fields
# inside the parens (`(tp=1, pp=1)`), so match anything up to the close-paren.
_JOB_LINE_RE = re.compile(r"^Job:\s+(\S+)\s+\(tp=(\d+)[^)]*\)\s+\[([0-9a-f]+(?:_[0-9a-f]+)*)\]")
_HOST_LINE_RE = re.compile(r"^\s+(\S+)\s+(\d{1,3}(?:\.\d{1,3}){3})\s+(.*)$")
# Engine process names that count as "the model is being served (or loaded)".
_ENGINE_PROC_RE = re.compile(r"\b(vllm|sglang|llama|trtllm|lmdeploy|mlc)\b", re.I)


def sparkrun_bin() -> str | None:
    return shutil.which("sparkrun")


def version(timeout: int = 10) -> str | None:
    """Installed sparkrun version string (e.g. `0.2.40` or `0.3.0-alpha+g1a2b3c4`
    on a preview channel), or None if sparkrun is missing/broken."""
    exe = sparkrun_bin()
    if not exe:
        return None
    try:
        res = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    if res.returncode != 0:
        return None
    out = (res.stdout or res.stderr or "").strip()
    # Typically "sparkrun 0.2.40" — keep just the version token if present.
    m = re.search(r"\d+\.\d+\.\S*", out)
    return m.group(0) if m else (out or None)


# ---- self-update ------------------------------------------------------------
# `sparkrun update` upgrades the uv-tool install and refreshes recipe
# registries. Channel flags opt into preview builds from git: --stable (PyPI,
# default), --beta (develop), --alpha (develop-next / bleeding edge), --yolo
# (alias for --alpha). No flag stays on the currently-remembered channel.
UPDATE_CHANNELS = ("stable", "beta", "alpha", "yolo")

_update_lock = threading.Lock()
_update_state: dict[str, Any] = {
    "running": False,
    "channel": None,       # channel requested for the in-flight/last update
    "ok": None,            # None until first update; then True/False
    "log": [],             # captured stdout+stderr lines of the last update
    "started": None,
    "finished": None,
    "version_before": None,
    "version_after": None,
}


def update_status() -> dict[str, Any]:
    with _update_lock:
        return dict(_update_state, log=list(_update_state["log"]))


def start_update(channel: str | None, timeout: int = 900) -> dict[str, Any]:
    """Kick off `sparkrun update [--<channel>]` in a background thread.

    Returns the initial status snapshot; poll update_status() for progress.
    Raises ValueError if sparkrun is missing, the channel is unknown, or an
    update is already running.
    """
    exe = sparkrun_bin()
    if not exe:
        raise ValueError("sparkrun is not installed")
    if channel and channel not in UPDATE_CHANNELS:
        raise ValueError(f"unknown update channel {channel!r} (expected one of {', '.join(UPDATE_CHANNELS)})")
    with _update_lock:
        if _update_state["running"]:
            raise ValueError("a sparkrun update is already running")
        _update_state.update(
            running=True, channel=channel, ok=None, log=[],
            started=time.time(), finished=None,
            version_before=None, version_after=None,
        )

    cmd = [exe, "update"] + ([f"--{channel}"] if channel else [])

    def _worker() -> None:
        ver_before = version()
        with _update_lock:
            _update_state["version_before"] = ver_before
        ok = False
        lines: list[str] = []
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            for chunk in (res.stdout, res.stderr):
                lines.extend(line.rstrip() for line in (chunk or "").splitlines() if line.strip())
            ok = res.returncode == 0
            if not ok:
                lines.append(f"[exit code {res.returncode}]")
        except subprocess.TimeoutExpired:
            lines.append(f"[timed out after {timeout}s]")
        except Exception as e:  # noqa: BLE001
            lines.append(f"[error: {e}]")
        ver_after = version()
        with _update_lock:
            _update_state.update(
                running=False, ok=ok, log=lines[-400:],
                finished=time.time(), version_after=ver_after,
            )

    threading.Thread(target=_worker, name="sparkrun-update", daemon=True).start()
    return update_status()


def list_recipes(timeout: int = 30) -> list[dict[str, Any]]:
    """`sparkrun list --json` → every launchable recipe across ALL configured
    registries (official, eugr, transitional, …) — far more complete than our
    local mirror of two repos. [] when sparkrun is missing or predates --json."""
    exe = sparkrun_bin()
    if not exe:
        return []
    try:
        res = subprocess.run([exe, "list", "--json"], capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0 or not (res.stdout or "").lstrip().startswith("["):
            return []
        out = []
        for r in json.loads(res.stdout):
            ref = r.get("name") or ""
            if not ref.startswith("@"):
                continue
            try:
                min_nodes = int(r.get("min_nodes") or 1)
            except (TypeError, ValueError):
                min_nodes = 1
            out.append({
                "ref": ref,
                "workload": r.get("file") or ref.rsplit("/", 1)[-1],
                "namespace": (r.get("registry") or ref[1:].split("/", 1)[0]),
                "name": r.get("file") or ref,
                "model": r.get("model"),
                "engine": r.get("runtime"),
                "description": r.get("description") or "",
                "min_nodes": min_nodes,
                "max_nodes": None,
            })
        return out
    except Exception:  # noqa: BLE001
        return []


def _jobs_from_cluster_status(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Map `sparkrun cluster status --json` output to our job shape.

    Schema (sparkrun ClusterStatusResult.to_dict): groups is
    {cluster_id: {meta, containers: [{host, role, status, image}]}}; solo
    single-container jobs land in solo_entries [{cluster_id, meta, host,
    status, image}]. meta carries recipe (the ref), hosts, port, and
    overrides.tensor_parallel."""
    def _job(cid: str, meta: dict[str, Any], members: list[dict[str, Any]]) -> dict[str, Any]:
        jobid = cid.removeprefix("sparkrun_")
        meta = meta or {}
        try:
            tp = int(((meta.get("overrides") or {}).get("tensor_parallel")) or meta.get("tp") or len(members) or 1)
        except (TypeError, ValueError):
            tp = max(len(members), 1)
        hosts = [{"role": m.get("role") or "solo", "ip": m.get("host") or "",
                  "status": m.get("status") or ""} for m in members]
        return {
            "ref": meta.get("recipe") or meta.get("ref") or "",
            "tp": tp,
            "jobid": jobid,
            "hosts": hosts,
            "containers": [f"sparkrun_{jobid}_{h['role']}" for h in hosts],
        }

    out: list[dict[str, Any]] = []
    for cid, group in (doc.get("groups") or {}).items():
        out.append(_job(cid, group.get("meta") or {}, group.get("containers") or []))
    for entry in doc.get("solo_entries") or []:
        out.append(_job(entry.get("cluster_id") or "", entry.get("meta") or {},
                        [{"host": entry.get("host"), "role": "solo", "status": entry.get("status")}]))
    return out


def parse_status(timeout: int = 25) -> list[dict[str, Any]]:
    """Parse sparkrun's container status into
    [{ref, tp, jobid, hosts: [{role, ip, status}], containers: [...]}].

    Primary source: `sparkrun cluster status --json` (the `status` alias
    doesn't accept --json, but the underlying command does — thanks to the
    spark-arena admin for the pointer). Text parsing of `sparkrun status`
    remains as the fallback for older builds."""
    exe = sparkrun_bin()
    if not exe:
        return []
    try:
        res = subprocess.run([exe, "cluster", "status", "--json"],
                             capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0 and (res.stdout or "").lstrip().startswith("{"):
            return _jobs_from_cluster_status(json.loads(res.stdout))
    except Exception:  # noqa: BLE001
        pass  # fall through to the text parser
    try:
        res = subprocess.run([exe, "status"], capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        return []
    if res.returncode != 0:
        return []
    jobs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in (res.stdout or "").splitlines():
        m = _JOB_LINE_RE.match(line)
        if m:
            current = {
                "ref": m.group(1),
                "tp": int(m.group(2)),
                "jobid": m.group(3),
                "hosts": [],
                "containers": [],
            }
            jobs.append(current)
            continue
        if current and not line.strip().startswith(("logs:", "stop:")):
            m = _HOST_LINE_RE.match(line)
            if m:
                role = m.group(1)
                current["hosts"].append({"role": role, "ip": m.group(2), "status": m.group(3).strip()})
                current["containers"].append(f"sparkrun_{current['jobid']}_{role}")
    return jobs


def export_running_recipe(jobid: str, timeout: int = 25) -> dict[str, Any] | None:
    """`sparkrun export running-recipe <jobid> --json` → model/runtime/
    container/defaults (incl. port). None on any failure."""
    exe = sparkrun_bin()
    if not exe:
        return None
    try:
        res = subprocess.run(
            [exe, "export", "running-recipe", jobid, "--json"],
            capture_output=True, text=True, timeout=timeout,
        )
        if res.returncode != 0:
            return None
        return json.loads(res.stdout.strip() or "null")
    except Exception:  # noqa: BLE001
        return None


def serve_alive(container: str, timeout: int = 12) -> bool | None:
    """Is an engine process running inside `container`?

    True  — an engine process (vllm/sglang/...) exists; safe during long model
            loads because the process is present the whole time.
    False — only wrapper processes (sleep infinity, bash, log tails) remain:
            the serve process died while the container stayed Up.
    None  — unknown (docker missing, container gone, or remote host); callers
            must never treat None as dead.
    """
    docker = shutil.which("docker")
    if not docker:
        return None
    try:
        res = subprocess.run([docker, "top", container], capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    if res.returncode != 0:
        return None
    lines = (res.stdout or "").splitlines()[1:]  # skip header
    saw_process = False
    for line in lines:
        if not line.strip():
            continue
        saw_process = True
        # /tmp/sparkrun_serve.log tails would false-positive engine regexes
        # that include "serve"; _ENGINE_PROC_RE deliberately matches engine
        # names only, so log tails and shell wrappers fall through.
        if _ENGINE_PROC_RE.search(line):
            return True
    return False if saw_process else None


def serve_log_tail(container: str, n: int = 200, timeout: int = 15) -> list[str]:
    """Last n lines of the in-container serve log (empty list on failure)."""
    docker = shutil.which("docker")
    if not docker:
        return []
    try:
        res = subprocess.run(
            [docker, "exec", container, "tail", "-n", str(n), "/tmp/sparkrun_serve.log"],
            capture_output=True, text=True, timeout=timeout,
        )
        if res.returncode != 0:
            return []
        return [line.rstrip() for line in (res.stdout or "").splitlines()]
    except Exception:  # noqa: BLE001
        return []


def guess_url(job: dict[str, Any], port: int = 8000) -> str | None:
    """Engine URL for a parsed status job: first host's IP + recipe port
    (containers use host networking)."""
    hosts = job.get("hosts") or []
    if not hosts:
        return None
    return f"http://{hosts[0]['ip']}:{port}"


def tail_pump_cmd(jobid: str | None, container: str | None) -> list[str] | None:
    """Command whose stdout re-streams a live workload's logs, for adoption.
    Prefer `sparkrun logs <jobid>`; fall back to docker exec tail -F."""
    if jobid and sparkrun_bin():
        return [sparkrun_bin(), "logs", jobid]
    if container and shutil.which("docker"):
        return [shutil.which("docker"), "exec", container, "tail", "-n", "200", "-F", "/tmp/sparkrun_serve.log"]
    return None
