"""Probe DGX Spark host capacity and Spark-mesh state.

Single source of truth so:

  - the UI header can show "1× GB10 · 128 GB"
  - Forge can badge each recipe with fits / needs-cluster / too-big
  - prepare_run can pre-flight a recipe before launching (and the UI can
    explain *why* a recipe will or won't run on this box)

DGX Spark layout in practice: one GB10 GPU per box (128 GB unified memory),
multi-box runs go through spark-vllm-docker's autodiscover, which writes
``CLUSTER_NODES`` into ``data/registry/spark-vllm-docker/.env``. We treat
that file as authoritative for mesh size; if it is missing or empty we
assume a 1-Spark deployment.

Cached in-process for ``_TTL`` seconds; call ``refresh()`` to invalidate.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any


SPARK_VLLM_DOCKER_ENV = (
    Path(__file__).parent / "data" / "registry" / "spark-vllm-docker" / ".env"
)
# Names we recognise as "this is a DGX Spark box". GB10 is the production
# Spark chip; older / pre-prod units may report variants.
DGX_SPARK_GPU_RE = re.compile(r"GB10|GB200|DGX|Spark", re.IGNORECASE)

_TTL = 30.0
_cache: dict[str, Any] | None = None
_cache_ts: float = 0.0


# ----------------------------- probing ------------------------------------

def _nvidia_smi_query() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        # GB10 reports memory.total as [N/A]: its 128 GB is unified with system
        # RAM, so nvidia-smi has no discrete VRAM number. Dropping the GPU here
        # (the old behavior) silently broke DGX Spark detection and every
        # fits-this-Spark check — fall back to total system memory instead,
        # which IS the GPU-visible pool on unified-memory boxes.
        memory_gb: float | None = None
        try:
            memory_gb = round(int(parts[2]) / 1024, 1)
        except ValueError:
            try:
                import psutil
                memory_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
            except Exception:  # noqa: BLE001
                memory_gb = 0.0
        gpus.append(
            {
                "index": idx,
                "name": parts[1],
                "memory_gb": memory_gb,
                "driver": parts[3],
            }
        )
    return gpus


def _read_cluster_nodes() -> list[str]:
    if not SPARK_VLLM_DOCKER_ENV.exists():
        return []
    try:
        text = SPARK_VLLM_DOCKER_ENV.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() != "CLUSTER_NODES":
            continue
        val = val.strip().strip('"').strip("'")
        return [n.strip() for n in val.split(",") if n.strip()]
    return []


def probe_host(force: bool = False) -> dict[str, Any]:
    global _cache, _cache_ts
    now = time.time()
    if _cache is not None and not force and (now - _cache_ts) < _TTL:
        return _cache

    gpus = _nvidia_smi_query()
    cluster_nodes = _read_cluster_nodes()
    is_dgx_spark = bool(gpus) and any(
        DGX_SPARK_GPU_RE.search(g["name"]) for g in gpus
    )
    total_memory_gb = round(sum(g["memory_gb"] for g in gpus), 1)
    # In a Spark mesh each box contributes its own GPU(s). We don't have
    # remote nvidia-smi, but GB10 boxes are homogeneous, so multiplying by
    # mesh size is a fair estimate of effective capacity. ``mesh_size``
    # counts the head node, so a single box reports mesh_size=1.
    mesh_size = max(1, len(cluster_nodes))
    effective_gpu_count = len(gpus) * mesh_size
    effective_memory_gb = round(total_memory_gb * mesh_size, 1)

    # Friendly summary string for the UI header.
    if gpus:
        primary = gpus[0]["name"]
        # Trim noisy NVIDIA prefixes.
        for prefix in ("NVIDIA ", "GeForce "):
            if primary.startswith(prefix):
                primary = primary[len(prefix):]
        summary = f"{len(gpus)}× {primary} · {total_memory_gb:g} GB"
        if mesh_size > 1:
            summary += f" · mesh ×{mesh_size} (~{effective_memory_gb:g} GB)"
    else:
        summary = "no GPU detected"

    _cache = {
        "gpus": gpus,
        "gpu_count": len(gpus),
        "total_memory_gb": total_memory_gb,
        "is_dgx_spark": is_dgx_spark,
        "cluster_nodes": cluster_nodes,
        "mesh_size": mesh_size,
        "effective_gpu_count": effective_gpu_count,
        "effective_memory_gb": effective_memory_gb,
        "summary": summary,
        "probed_at": now,
    }
    _cache_ts = now
    return _cache


def refresh() -> dict[str, Any]:
    return probe_host(force=True)


# ----------------------------- fit verdict --------------------------------

def fit_for_recipe(
    *,
    tensor_parallel: int | None = None,
    min_nodes: int | None = None,
    weight_gb: float | None = None,
    cluster_only: bool = False,
    solo_only: bool = False,
) -> dict[str, Any]:
    """Decide whether a recipe will run on the current host.

    Verdicts:
      - ``fits``: all good; safe to launch.
      - ``needs_cluster``: would need more GPUs than this host (or mesh) has.
      - ``too_big``: model weights exceed the GPU memory budget.
      - ``unknown``: nvidia-smi unavailable; can't decide.

    Returned dict always includes ``required_gpus`` and ``available_gpus``
    so the UI can show "needs 2 Sparks, you have 1".
    """
    host = probe_host()
    gpu_count = host["gpu_count"]
    mesh_size = host["mesh_size"]
    effective = host["effective_gpu_count"]
    eff_mem = host["effective_memory_gb"]

    required_gpus = max(int(tensor_parallel or 1), int(min_nodes or 1))

    if not gpu_count:
        return {
            "verdict": "unknown",
            "required_gpus": required_gpus,
            "available_gpus": 0,
            "reason": "nvidia-smi unavailable; can't verify hardware.",
        }

    if cluster_only and mesh_size <= 1:
        return {
            "verdict": "needs_cluster",
            "required_gpus": max(required_gpus, 2),
            "available_gpus": effective,
            "mesh_size": mesh_size,
            "reason": "Recipe is cluster-only — run autodiscover to mesh Sparks.",
        }

    if solo_only and mesh_size > 1:
        return {
            "verdict": "needs_cluster",
            "required_gpus": required_gpus,
            "available_gpus": effective,
            "mesh_size": mesh_size,
            "reason": "Recipe is solo-only but a mesh is configured.",
        }

    if required_gpus > effective:
        short_by = required_gpus - effective
        return {
            "verdict": "needs_cluster",
            "required_gpus": required_gpus,
            "available_gpus": effective,
            "mesh_size": mesh_size,
            "reason": f"Needs {required_gpus} GPU(s); have {effective} (short by {short_by}).",
        }

    if weight_gb and eff_mem and weight_gb > eff_mem * 0.9:
        return {
            "verdict": "too_big",
            "required_gpus": required_gpus,
            "available_gpus": effective,
            "weight_gb": weight_gb,
            "memory_gb": eff_mem,
            "reason": (
                f"Model ~{weight_gb:.0f} GB exceeds GPU memory budget "
                f"(~{eff_mem:.0f} GB available)."
            ),
        }

    return {
        "verdict": "fits",
        "required_gpus": required_gpus,
        "available_gpus": effective,
        "mesh_size": mesh_size,
    }
