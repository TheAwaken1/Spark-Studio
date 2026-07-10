"""Recovery actions — the "I broke it" safety net.

Every action here is deliberately conservative and reports exactly what it
did, so beginners can click them without fear:

  clear_finished_runs   drop exited runs from the live list (history stays in DB)
  clean_containers      remove orphan sparkrun_* / spark-vllm-* containers that
                        no live run owns — NEVER touches user-managed containers
                        (vllm_node, spark-searxng, …) or jobs sparkrun still reports
  reset_registry        wipe the registry mirror + forged YAML cache (resynced on
                        demand / next boot)
  wipe_db               delete saved recipes, run history, bench + eval history
                        (models on disk are untouched)
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

import db
import registry
import sparkrun_service
from runners import runner

# Containers Spark Studio creates and may safely reap when orphaned. Anything
# else (vllm_node, spark-searxng, user stacks) is out of bounds by prefix.
_OWNED_PREFIXES = ("spark-vllm-", "sparkrun_")


def clear_finished_runs() -> dict[str, Any]:
    """Remove exited runs from the in-memory list. DB history is preserved —
    this is a declutter, not a delete. Also marks any stale 'running' DB rows
    (from crashed sessions) as exited."""
    removed = []
    for rid, run in list(runner.runs.items()):
        if run.status == "exited":
            removed.append(run.label or rid)
            del runner.runs[rid]
    stale = 0
    live_ids = set(runner.runs.keys())
    try:
        for row in db.runs_list_running():
            if row["id"] not in live_ids:
                db.runs_update(row["id"], status="exited", ended_at=db.now())
                stale += 1
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "removed_from_list": removed, "stale_rows_closed": stale}


def _docker_containers() -> list[dict[str, str]]:
    docker = shutil.which("docker")
    if not docker:
        return []
    try:
        res = subprocess.run(
            [docker, "ps", "-a", "--format", "{{.Names}}\t{{.State}}"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:  # noqa: BLE001
        return []
    out = []
    for line in (res.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            out.append({"name": parts[0].strip(), "state": parts[1].strip()})
    return out


def clean_containers() -> dict[str, Any]:
    """Reap orphan containers Spark Studio (or sparkrun) created but nothing
    controls anymore. Model files are never touched."""
    docker = shutil.which("docker")
    if not docker:
        return {"ok": False, "error": "docker not found"}

    # Containers a live run still owns — off limits.
    owned_by_runs: set[str] = set()
    for run in runner.runs.values():
        if run.status == "running":
            owned_by_runs.update(run.managed_containers or [])

    # Jobs sparkrun still reports as active — off limits even if this app
    # doesn't know them (terminal-launched workloads).
    active_jobids: set[str] = set()
    try:
        for job in sparkrun_service.parse_status():
            active_jobids.add(job["jobid"])
    except Exception:  # noqa: BLE001
        pass

    removed, skipped = [], []
    for c in _docker_containers():
        name, state = c["name"], c["state"].lower()
        if not name.startswith(_OWNED_PREFIXES):
            continue  # not ours — never touch
        if name in owned_by_runs:
            skipped.append({"name": name, "why": "owned by a live run"})
            continue
        if name.startswith("sparkrun_"):
            m = sparkrun_service.CONTAINER_RE.match(name)
            jobid = m.group(1) if m else None
            if state == "running" and jobid and jobid in active_jobids:
                skipped.append({"name": name, "why": "sparkrun reports this job as active"})
                continue
            if state == "running" and not active_jobids and sparkrun_service.sparkrun_bin():
                # sparkrun status came back empty — could be a probe hiccup;
                # don't kill a running job on ambiguous evidence.
                skipped.append({"name": name, "why": "could not confirm the job is dead"})
                continue
        try:
            res = subprocess.run([docker, "rm", "-f", name],
                                 capture_output=True, text=True, timeout=60)
            if res.returncode == 0:
                removed.append(name)
            else:
                skipped.append({"name": name, "why": (res.stderr or res.stdout or "rm failed").strip()[:120]})
        except Exception as e:  # noqa: BLE001
            skipped.append({"name": name, "why": str(e)[:120]})
    return {"ok": True, "removed": removed, "skipped": skipped}


def reset_registry() -> dict[str, Any]:
    """Delete the local registry mirrors + forged YAML cache. They re-clone on
    the next sync (kicked off by the caller / next app start)."""
    import docker_recipe
    removed = []
    for path in (registry.REGISTRY_ROOT, docker_recipe.FORGED_DIR):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path))
    registry.reindex()  # empty index until the next sync completes
    return {"ok": True, "removed": removed}


def wipe_db(confirm: bool = False) -> dict[str, Any]:
    """Delete saved recipes, run history, benchmark + tool-eval history.
    Downloaded models and the registry mirror are untouched. Requires
    confirm=True — this is the one action that loses user-created data."""
    if not confirm:
        return {"ok": False, "error": "confirmation required"}
    counts: dict[str, int] = {}
    with db.cur() as c:
        for table in ("benchmarks", "benchy_runs", "tooleval_runs", "runs", "recipes"):
            c.execute(f"DELETE FROM {table}")
            counts[table] = c.rowcount if c.rowcount >= 0 else 0
    try:
        db._conn.execute("VACUUM")
    except Exception:  # noqa: BLE001
        pass
    # In-memory run objects for still-running workloads survive on purpose —
    # wiping the DB should not stop a serving model.
    return {"ok": True, "deleted": counts}
