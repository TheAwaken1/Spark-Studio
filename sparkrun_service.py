"""Helpers pour observer les workloads gérés par sparkrun.

Les workloads sparkrun survivent volontairement à leur processus launcher :
le container exécute `sleep infinity` et la commande serve est lancée
séparément via `docker exec`, en loggant dans /tmp/sparkrun_serve.log À
L'INTÉRIEUR du container. Cela signifie que ni le code de sortie du
launcher ni `docker ps` ne vous disent si l'engine est réellement vivant
— ces helpers le font.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import yaml


BUNDLED_RECIPES_DIR = Path(__file__).resolve().parent / "recipes"
BUNDLED_RECIPE_NAMESPACE = "studio"

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
_ENGINE_PROC_RE = re.compile(r"\b(vllm|sglang|llama|ds4|trtllm|lmdeploy|mlc)\b", re.I)


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


def _bundled_recipes() -> list[dict[str, Any]]:
    """Recipes shipped with Spark Studio but not published in a registry yet."""
    out: list[dict[str, Any]] = []
    if not BUNDLED_RECIPES_DIR.is_dir():
        return out
    for path in sorted(BUNDLED_RECIPES_DIR.glob("*.y*ml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(doc, dict):
            continue
        if not doc.get("model") or not doc.get("command"):
            continue
        raw_metadata = doc.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        raw_defaults = doc.get("defaults")
        defaults: dict[str, Any] = raw_defaults if isinstance(raw_defaults, dict) else {}
        try:
            min_nodes = int(doc.get("min_nodes") or defaults.get("tensor_parallel") or 1)
        except (TypeError, ValueError):
            min_nodes = 1
        try:
            max_nodes = int(doc["max_nodes"]) if doc.get("max_nodes") is not None else None
        except (TypeError, ValueError):
            max_nodes = None
        out.append({
            "ref": f"@{BUNDLED_RECIPE_NAMESPACE}/{path.stem}",
            "workload": path.stem,
            "namespace": BUNDLED_RECIPE_NAMESPACE,
            "name": doc.get("name") or path.stem,
            "model": doc.get("model"),
            "engine": doc.get("runtime") or "vllm",
            "description": doc.get("description") or metadata.get("description") or "",
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "path": str(path),
        })
    return out


def resolve_recipe_target(ref: str) -> str:
    """Map a synthetic @studio ref to its trusted bundled recipe path."""
    prefix = f"@{BUNDLED_RECIPE_NAMESPACE}/"
    if not ref.startswith(prefix):
        return ref
    stem = ref.removeprefix(prefix)
    if not re.fullmatch(r"[\w][\w.-]*", stem):
        return ref
    candidate = (BUNDLED_RECIPES_DIR / f"{stem}.yaml").resolve()
    try:
        candidate.relative_to(BUNDLED_RECIPES_DIR.resolve())
    except ValueError:
        return ref
    return str(candidate) if candidate.is_file() else ref


def canonical_recipe_ref(ref: str) -> str:
    """Convert a bundled recipe file path to its stable ``@studio`` ref.

    Older adopted runs were saved with the absolute local YAML path as their
    sparkrun ref.  The launch API intentionally accepts only recipe names/refs,
    so normalize trusted files under our bundled recipe directory before that
    validation runs.  Paths outside that directory are left unchanged.
    """
    ref = ref.strip()
    if not ref or ref.startswith("@"):
        return ref
    try:
        candidate = Path(ref).expanduser().resolve()
        candidate.relative_to(BUNDLED_RECIPES_DIR.resolve())
    except (OSError, ValueError):
        return ref
    if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}:
        return f"@{BUNDLED_RECIPE_NAMESPACE}/{candidate.stem}"
    return ref


def list_recipes(timeout: int = 30) -> list[dict[str, Any]]:
    """`sparkrun list --json` → every launchable recipe across ALL configured
    registries (official, eugr, transitional, …) — far more complete than our
    local mirror of two repos. Bundled @studio recipes are included even when
    the CLI listing is unavailable."""
    bundled = _bundled_recipes()
    exe = sparkrun_bin()
    if not exe:
        return bundled
    try:
        res = subprocess.run([exe, "list", "--json"], capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0 or not (res.stdout or "").lstrip().startswith("["):
            return bundled
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
                "max_nodes": r.get("max_nodes"),
            })
        known_refs = {recipe["ref"] for recipe in out}
        out.extend(recipe for recipe in bundled if recipe["ref"] not in known_refs)
        return out
    except Exception:  # noqa: BLE001
        return bundled


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


def find_recipes_by_model(model: str, timeout: int = 30) -> list[dict[str, Any]]:
    """Every sparkrun-launchable recipe whose model matches `model` exactly
    (case-insensitive) — the Forge's highest-trust source for 'make this new
    model work': if ANY community registry has validated the model (official,
    eugr, atlas, …), that beats adapting or synthesizing.

    Prefers `sparkrun list --json`; falls back to scanning the synced registry
    cache on disk, because older sparkrun builds omit newer recipe formats
    (e.g. 0.2.40-beta doesn't list recipe_version-2 / alternate-runtime
    recipes that are sitting right there in ~/.cache/sparkrun/registries)."""
    target = (model or "").strip().lower()
    if not target:
        return []
    out = [r for r in list_recipes(timeout)
           if (r.get("model") or "").strip().lower() == target]
    seen = {r["ref"] for r in out}

    # Disk fallback / supplement: registries synced by `sparkrun update`.
    import yaml
    from pathlib import Path
    root = Path.home() / ".cache" / "sparkrun" / "registries"
    if root.is_dir():
        for reg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for yml in sorted(reg_dir.rglob("*.y*ml")):
                try:
                    if yml.stat().st_size > 128 * 1024:
                        continue
                    doc = yaml.safe_load(yml.read_text(encoding="utf-8", errors="replace"))
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(doc, dict):
                    continue
                if (str(doc.get("model") or "")).strip().lower() != target:
                    continue
                ref = f"@{reg_dir.name}/{yml.stem}"
                if ref in seen:
                    continue
                seen.add(ref)
                meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
                desc = (doc.get("description") or meta.get("description") or "").strip()
                try:
                    min_nodes = int(doc.get("min_nodes") or (doc.get("defaults") or {}).get("tensor_parallel") or 1)
                except (TypeError, ValueError):
                    min_nodes = 1
                out.append({
                    "ref": ref,
                    "workload": yml.stem,
                    "namespace": reg_dir.name,
                    "name": doc.get("name") or yml.stem,
                    "model": doc.get("model"),
                    "engine": doc.get("runtime") or "vllm",
                    "description": desc.splitlines()[0][:200] if desc else "",
                    "min_nodes": min_nodes,
                    "max_nodes": int(doc["max_nodes"]) if str(doc.get("max_nodes") or "").isdigit() else None,
                })
    return out


# jobid -> recovered ref, so the docker fallback pays the export + recipe-list
# cost once per orphaned job, not on every watchdog sweep.
_ORPHAN_REF_CACHE: dict[str, str] = {}


def _jobs_from_docker(timeout: int = 15) -> list[dict[str, Any]]:
    """Last-resort job discovery straight from `docker ps`.

    sparkrun's state store can lose a job whose container is still up and
    serving (`cluster status --json` reports zero jobs and the host as idle
    while `sparkrun_<jobid>_solo` answers /v1/models). The containers are the
    ground truth: without this, the dashboard adopts nothing, `runner.active()`
    stays None, and everything downstream — chat, the engine passthrough,
    Hermes' /model row — reports no model while one is plainly running.

    Local-only by construction (docker ps sees this host), which matches how
    adoption uses the result: host networking on 127.0.0.1.
    """
    docker = shutil.which("docker")
    if not docker:
        return []
    try:
        res = subprocess.run([docker, "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        return []
    if res.returncode != 0:
        return []
    grouped: dict[str, list[str]] = {}
    for name in (res.stdout or "").split():
        m = CONTAINER_RE.fullmatch(name.strip())
        if m:
            grouped.setdefault(m.group(1), []).append(m.group(2))
    jobs: list[dict[str, Any]] = []
    for jobid, roles in grouped.items():
        # The container name carries no ref; recover it through the model the
        # job is serving. Best-effort — an adopted job works without a ref
        # (jobid-scoped stop, exported port, model label), it just loses the
        # My Recipes linkage.
        if jobid not in _ORPHAN_REF_CACHE:
            ref = ""
            try:
                export = export_running_recipe(jobid)
                model = (export or {}).get("model") or ""
                if model:
                    matches = find_recipes_by_model(model)
                    ref = matches[0]["ref"] if matches else ""
            except Exception:  # noqa: BLE001
                ref = ""
            _ORPHAN_REF_CACHE[jobid] = ref
        roles.sort(key=lambda role: (role != "head", role != "solo", role))
        jobs.append({
            "ref": _ORPHAN_REF_CACHE[jobid],
            "tp": len(roles),
            "jobid": jobid,
            "hosts": [{"role": role, "ip": "127.0.0.1", "status": "Up"} for role in roles],
            "containers": [f"sparkrun_{jobid}_{role}" for role in roles],
        })
    return jobs


def parse_status(timeout: int = 25) -> list[dict[str, Any]]:
    """Parse sparkrun's container status into
    [{ref, tp, jobid, hosts: [{role, ip, status}], containers: [...]}].

    Primary source: `sparkrun cluster status --json` (the `status` alias
    doesn't accept --json, but the underlying command does — thanks to the
    spark-arena admin for the pointer). Text parsing of `sparkrun status`
    remains as the fallback for older builds; live sparkrun_* containers that
    sparkrun itself no longer reports are the fallback of last resort."""
    exe = sparkrun_bin()
    if not exe:
        return []
    try:
        res = subprocess.run([exe, "cluster", "status", "--json"],
                             capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0 and (res.stdout or "").lstrip().startswith("{"):
            return _jobs_from_cluster_status(json.loads(res.stdout)) or _jobs_from_docker()
    except Exception:  # noqa: BLE001
        pass  # fall through to the text parser
    try:
        res = subprocess.run([exe, "status"], capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        return _jobs_from_docker()
    if res.returncode != 0:
        return _jobs_from_docker()
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
    return jobs or _jobs_from_docker()


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
