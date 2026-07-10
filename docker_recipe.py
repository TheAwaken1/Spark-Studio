"""Render a registry-shaped recipe into a runnable raw_cmd.

We delegate to spark-vllm-docker's canonical orchestrator (``run-recipe.sh``
→ ``run-recipe.py``) instead of reinventing docker invocation. That script
handles image build, model download, mod application, ray/solo cluster
setup, and graceful shutdown — all things our launcher would otherwise
have to reimplement.

Forge plants a ``_registry`` block inside ``args`` for any registry-sourced
recipe. ``prepare_run`` recognises it, materialises the YAML on disk
(adapting the model field if needed), and produces the raw shell command
the runner will spawn.
"""

from __future__ import annotations

import hashlib
import json
import socket
import uuid
from pathlib import Path
from typing import Any

import yaml

REGISTRY_ROOT = Path(__file__).parent / "data" / "registry"
SPARK_VLLM_DOCKER = REGISTRY_ROOT / "spark-vllm-docker"
FORGED_DIR = Path(__file__).parent / "data" / "forged"


def is_registry_shaped(args: dict[str, Any]) -> bool:
    block = args.get("_registry") if isinstance(args, dict) else None
    return isinstance(block, dict) and bool(block.get("container") or block.get("command"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _normalize_for_runner(body: str, fallback_name: str) -> str:
    """Normalize a recipe YAML so spark-vllm-docker's run-recipe.py accepts it.

    Recipe-registry v2 recipes diverge from the v1 schema run-recipe.py
    validates against: they omit top-level ``name``, bury the description
    under ``metadata.description``, and reference ``{model}`` in the command
    template while keeping ``model`` only at the top level (run-recipe.py's
    ``str.format`` only sources from ``defaults``). We patch all three so any
    registry-sourced recipe runs as-is; the recipe_version is also dropped to
    "1" since the result is v1-compatible (silences the runner's warning).
    Returns the original body when no fix-ups are needed.
    """
    try:
        doc = yaml.safe_load(body)
    except yaml.YAMLError:
        return body
    if not isinstance(doc, dict):
        return body

    changed = False
    if not doc.get("name"):
        doc["name"] = fallback_name
        changed = True
    if not doc.get("description"):
        meta = doc.get("metadata")
        if isinstance(meta, dict) and meta.get("description"):
            doc["description"] = str(meta["description"])
            changed = True
    # Lift top-level model into defaults so {model} substitutes in command.
    model = doc.get("model")
    if model:
        defaults = doc.get("defaults")
        if not isinstance(defaults, dict):
            defaults = {}
            doc["defaults"] = defaults
        if "model" not in defaults:
            defaults["model"] = model
            changed = True
    if str(doc.get("recipe_version") or "") not in ("", "1"):
        doc["recipe_version"] = "1"
        changed = True
    if not changed:
        return body
    return yaml.safe_dump(doc, sort_keys=False)


def _yaml_path_for(block: dict[str, Any], requested_model: str | None) -> Path:
    """Return an on-disk YAML path. For exact-match recipes that already pass
    validation we use the cloned file directly; otherwise (adapted recipes, or
    recipe-registry v2 files missing the ``name`` field) we synthesize a
    derived YAML in ``app/data/forged/`` keyed by (origin, model)."""
    origin = block.get("origin") or {}
    if isinstance(origin, str):
        # Agents sometimes return origin as "repo/path" concatenated.
        parts = origin.split("/", 1)
        origin = {"repo": parts[0], "path": parts[1]} if len(parts) == 2 else {}
    repo = origin.get("repo")
    rel = origin.get("path")
    raw_yaml = block.get("raw_yaml") or ""
    upstream_model = block.get("adapted_from_model")

    fallback_name = Path(rel).stem if rel else (block.get("name") or "forged-recipe")

    body = raw_yaml
    if upstream_model and requested_model and upstream_model != requested_model:
        body = body.replace(upstream_model, requested_model)
    body = _normalize_for_runner(body, fallback_name)

    cloned = REGISTRY_ROOT / repo / rel if repo and rel else None
    if not upstream_model and cloned and cloned.exists():
        try:
            if cloned.read_text(encoding="utf-8", errors="replace") == body:
                return cloned
        except OSError:
            pass

    digest = hashlib.sha1(
        json.dumps({"r": repo, "p": rel, "m": requested_model}, sort_keys=True).encode()
    ).hexdigest()[:10]
    FORGED_DIR.mkdir(parents=True, exist_ok=True)
    out = FORGED_DIR / f"{digest}.yaml"
    if not out.exists() or out.read_text(encoding="utf-8", errors="replace") != body:
        out.write_text(body, encoding="utf-8")
    return out


def prepare_run(
    args: dict[str, Any],
    env_extra: dict[str, str] | None = None,
    explicit_port: int | None = None,
) -> tuple[str, list[str], dict[str, str], int]:
    """Build the spawn payload for a registry-shaped recipe.

    Returns (raw_cmd, managed_containers, env, port).
    """
    block = args["_registry"]
    requested_model = args.get("model")

    runner_sh = SPARK_VLLM_DOCKER / "run-recipe.sh"
    if not runner_sh.exists():
        raise RuntimeError(
            "spark-vllm-docker is not synced — click 'Refresh registry' "
            f"or rerun install.js. Expected: {runner_sh}"
        )

    yaml_path = _yaml_path_for(block, requested_model)
    port = explicit_port or _free_port()
    container_name = f"spark-vllm-{uuid.uuid4().hex[:8]}"

    # Pass any extra `env` from the recipe through run-recipe.py's `-e` flag
    # so they reach the container. Caller-provided env_extra wins over recipe.
    recipe_env = dict(block.get("env") or {})
    if env_extra:
        recipe_env.update(env_extra)

    parts = [
        "bash", "./run-recipe.sh",
        "--solo",
        # --setup is idempotent: builds the container image if missing, downloads
        # the HF model if missing, then runs. No-op when both already exist, so
        # safe to pass on every run for true 1-click UX.
        "--setup",
        "--name", container_name,
        "--port", str(port),
        "--host", "127.0.0.1",
    ]

    # Forward any defaults the user overrode in the Launch Settings dialog.
    # run-recipe.py's CLI overrides take precedence over the YAML defaults,
    # which is exactly what we need for per-launch token/memory adjustments.
    block_defaults = block.get("defaults") or {}
    if "max_model_len" in block_defaults:
        parts += ["--max-model-len", str(block_defaults["max_model_len"])]
    if "gpu_memory_utilization" in block_defaults:
        parts += ["--gpu-memory-utilization", str(block_defaults["gpu_memory_utilization"])]
    if "tensor_parallel" in block_defaults:
        parts += ["--tensor-parallel", str(block_defaults["tensor_parallel"])]

    for k, v in recipe_env.items():
        parts += ["-e", _q(f"{k}={v}")]
    parts.append(_q(str(yaml_path)))
    # spark-vllm-docker's helper scripts expect to run from their own repo root
    # because they resolve Dockerfile and metadata paths relative to cwd.
    raw_cmd = f"cd {_q(str(SPARK_VLLM_DOCKER))} && " + " ".join(parts)

    return raw_cmd, [container_name], dict(env_extra or {}), port


def _q(s: str) -> str:
    """shlex.quote but preserves simple identifiers unquoted for readability."""
    if s and all(c.isalnum() or c in "._-+=:/," for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"
