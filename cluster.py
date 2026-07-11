"""Spark-mesh (multi-node) view — a friendly UI over sparkrun's truth.

Design rule (from the roadmap): sparkrun is the distributed runtime; Spark
Studio only *reads* what sparkrun and the local box report. No SSH agents,
no per-node daemons — remote nodes get a lightweight reachability probe and
whatever `sparkrun status` says about them, nothing more.

Sources:
  sparkrun cluster list   → saved clusters + default (name, host list)
  sparkrun status         → running jobs with per-host roles/states
  hostinfo / psutil       → full detail for the local node only
"""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import time
from typing import Any

import hostinfo
import sparkrun_service
from runners import runner

# `sparkrun cluster list` row: "* name    host1,host2   description"
_CLUSTER_ROW_RE = re.compile(r"^(\*?)\s*(\S+)\s+((?:\d{1,3}\.){3}\d{1,3}(?:\s*,\s*(?:\d{1,3}\.){3}\d{1,3})*)")


def _local_ips() -> set[str]:
    ips = {"127.0.0.1", "localhost"}
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=4)
        ips.update(ip for ip in (out.stdout or "").split() if ":" not in ip)
    except Exception:  # noqa: BLE001
        pass
    return ips


def clusters() -> list[dict[str, Any]]:
    """Saved sparkrun cluster definitions ([{name, hosts, default}]).
    Uses `cluster list --json` (supported since 0.3.0-alpha) with a text-table
    fallback for older sparkrun builds."""
    exe = sparkrun_service.sparkrun_bin()
    if not exe:
        return []
    try:
        res = subprocess.run([exe, "cluster", "list", "--json"],
                             capture_output=True, text=True, timeout=20)
        if res.returncode == 0 and (res.stdout or "").lstrip().startswith("["):
            import json
            return [{"name": c.get("name"), "hosts": list(c.get("hosts") or []),
                     "default": bool(c.get("default"))} for c in json.loads(res.stdout)]
    except Exception:  # noqa: BLE001
        pass  # fall through to the text parser
    try:
        res = subprocess.run([exe, "cluster", "list"], capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return []
    if res.returncode != 0:
        return []
    out = []
    for line in (res.stdout or "").splitlines():
        m = _CLUSTER_ROW_RE.match(line.strip())
        if not m:
            continue
        out.append({
            "name": m.group(2),
            "hosts": [h.strip() for h in m.group(3).split(",") if h.strip()],
            "default": m.group(1) == "*",
        })
    return out


def _reachable(ip: str, timeout: float = 1.2) -> bool:
    """Remote-node liveness: can we open the SSH port sparkrun itself uses?
    (ICMP ping is often blocked; TCP/22 is what actually matters here.)"""
    try:
        with socket.create_connection((ip, 22), timeout=timeout):
            return True
    except OSError:
        return False


def cluster_info() -> dict[str, Any]:
    """Everything the Cluster page renders in one call."""
    saved = clusters()
    default = next((c for c in saved if c["default"]), saved[0] if saved else None)
    hosts: list[str] = list((default or {}).get("hosts") or [])

    # Merge in spark-vllm-docker's CLUSTER_NODES (autodiscover output) so a
    # mesh configured for docker recipes shows even without a sparkrun cluster.
    host_probe = hostinfo.probe_host()
    for n in host_probe.get("cluster_nodes") or []:
        if n not in hosts:
            hosts.append(n)
    local_ips = _local_ips()
    if not hosts:
        # No cluster configured anywhere: the local box is the whole "cluster".
        lan = sorted(ip for ip in local_ips if ip not in ("127.0.0.1", "localhost"))
        hosts = [lan[0] if lan else "127.0.0.1"]

    jobs = []
    try:
        jobs = sparkrun_service.parse_status()
    except Exception:  # noqa: BLE001
        pass
    role_by_ip: dict[str, dict[str, Any]] = {}
    for job in jobs:
        for h in job.get("hosts") or []:
            role_by_ip[h["ip"]] = {"job": job["ref"], "role": h["role"], "state": h["status"]}

    nodes = []
    for ip in hosts:
        is_local = ip in local_ips
        if is_local:
            node: dict[str, Any] = {
                "ip": ip, "local": True, "online": True,
                "summary": host_probe.get("summary"),
                "memory_total_gb": host_probe.get("total_memory_gb"),
            }
            try:
                import psutil
                node["memory_free_gb"] = round(psutil.virtual_memory().available / 1024 ** 3, 1)
            except Exception:  # noqa: BLE001
                pass
        else:
            node = {"ip": ip, "local": False, "online": _reachable(ip)}
        node["workload"] = role_by_ip.get(ip)
        nodes.append(node)

    online = sum(1 for n in nodes if n["online"])
    tp_options = [{"tp": i, "ok": i <= online,
                   "why": None if i <= online else f"needs {i} online node(s), have {online}"}
                  for i in range(1, max(len(nodes), 1) + 1)]

    active = runner.active()
    return {
        "generated_at": time.time(),
        "sparkrun_installed": sparkrun_service.sparkrun_bin() is not None,
        "cluster_name": (default or {}).get("name"),
        "clusters": saved,
        "nodes": nodes,
        "online_nodes": online,
        "mesh": len(nodes) > 1,
        "tp_options": tp_options,
        "jobs": jobs,
        "local_active_run": active.summary() if active else None,
    }


def readiness(tp: int = 1) -> dict[str, Any]:
    """Pre-launch checks for a tp-node run — plain-English, doctor-style."""
    info = cluster_info()
    checks: list[dict[str, Any]] = []

    def add(id_: str, ok: bool | None, detail: str, fix: str | None = None):
        checks.append({"id": id_, "status": "ok" if ok else ("warn" if ok is None else "error"),
                       "detail": detail, "fix": fix})

    add("sparkrun", info["sparkrun_installed"],
        "sparkrun installed" if info["sparkrun_installed"] else "sparkrun not installed",
        None if info["sparkrun_installed"] else "uvx sparkrun setup")

    enough = info["online_nodes"] >= tp
    add("nodes", enough,
        f"{info['online_nodes']} node(s) online, {tp} needed",
        None if enough else "Bring the offline Spark(s) up, or pick a lower TP.")

    docker_ok = shutil.which("docker") is not None
    if docker_ok:
        try:
            res = subprocess.run(["docker", "info", "--format", "ok"],
                                 capture_output=True, text=True, timeout=8)
            docker_ok = res.returncode == 0
        except Exception:  # noqa: BLE001
            docker_ok = False
    add("docker", docker_ok, "Docker daemon running on this node" if docker_ok
        else "Docker not available on this node", None if docker_ok else "Start Docker before launching.")

    # Memory: on Spark, one resident model at a time. A local active model will
    # be stopped by the pre-launch guard — surface it so nobody is surprised.
    active = info.get("local_active_run")
    if active:
        add("resident", None,
            f"'{active.get('label') or active.get('engine')}' is serving now — the launch guard will stop it first")
    else:
        add("resident", True, "no resident model to displace")

    busy = [j["ref"] for j in info["jobs"]]
    if busy:
        add("jobs", None, f"sparkrun job(s) already running: {', '.join(busy)} — "
                          "launching another may contend for memory")
    else:
        add("jobs", True, "no sparkrun jobs running")

    if tp > 1 and not info["mesh"]:
        add("mesh", False, "only one Spark is configured — multi-node needs a sparkrun cluster",
            "Run `sparkrun cluster create <name> <ip1>,<ip2>` (or `sparkrun setup`) to mesh your Sparks.")

    ok = all(c["status"] != "error" for c in checks)
    return {"tp": tp, "ok": ok, "checks": checks}
