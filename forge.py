"""Recipe Forge: produce runnable recipes for an HF repo.

Order of preference, highest leverage first:

  1. Registry exact match — return the curated YAML verbatim.
  2. Registry similar match — adapt the closest curated recipe (same family
     / quant), substituting the requested HF repo into ``model:`` and
     surfacing how we adapted it.
  3. Synthesized recipe — recipe_brain assembles a real spark-vllm-docker
     shaped YAML for the repo (right container variant for the quant, right
     tool/reasoning parser for the family, the standard flag stack eugr's
     curated recipes use). Goes through the docker path identically to a
     registry match.
  4. Heuristic fallback — flat-args presets for SGLang / llama.cpp paths
     where the docker container set doesn't apply.

Each emitted recipe carries a ``source`` field (``registry`` / ``similar``
/ ``synth`` / ``heuristic``) so the UI can badge it.
"""

from __future__ import annotations

from typing import Any

import yaml

import hostinfo
import recipe_brain
import registry


# ----------------------------- fit annotation ----------------------------

def _registry_recipe_needs(rec: registry.Recipe) -> dict[str, Any]:
    """Extract the resource hints a fit check needs from a registry recipe.

    Reads ``defaults.tensor_parallel`` and the top-level ``min_nodes`` /
    ``cluster_only`` / ``solo_only`` flags. We re-parse ``raw_yaml`` because
    the structured ``Recipe`` only normalises a subset; recipe-registry v2
    files keep ``min_nodes`` outside of ``defaults``.
    """
    needs: dict[str, Any] = {
        "tensor_parallel": int(rec.defaults.get("tensor_parallel", 1) or 1),
        "min_nodes": None,
        "cluster_only": False,
        "solo_only": False,
    }
    try:
        doc = yaml.safe_load(rec.raw_yaml or "") or {}
    except yaml.YAMLError:
        return needs
    if not isinstance(doc, dict):
        return needs
    if doc.get("min_nodes") is not None:
        try:
            needs["min_nodes"] = int(doc["min_nodes"])
        except (TypeError, ValueError):
            pass
    needs["cluster_only"] = bool(doc.get("cluster_only"))
    needs["solo_only"] = bool(doc.get("solo_only"))
    return needs


def _attach_fit(recipes: list[dict[str, Any]], report: dict[str, Any]) -> None:
    """Mutates each recipe in place to add a ``fit`` verdict against the host."""
    weight_gb = report.get("weight_gb") if isinstance(report, dict) else None
    for r in recipes:
        reg = r.get("registry") or {}
        if reg:
            tp = int((reg.get("defaults") or {}).get("tensor_parallel", 1) or 1)
            try:
                doc = yaml.safe_load(reg.get("raw_yaml") or "") or {}
            except yaml.YAMLError:
                doc = {}
            doc = doc if isinstance(doc, dict) else {}
            min_nodes = doc.get("min_nodes")
            cluster_only = bool(doc.get("cluster_only"))
            solo_only = bool(doc.get("solo_only"))
            try:
                min_nodes = int(min_nodes) if min_nodes is not None else None
            except (TypeError, ValueError):
                min_nodes = None
        else:
            tp = int(r.get("args", {}).get("tensor_parallel", 1) or 1)
            min_nodes = None
            cluster_only = False
            solo_only = False
        r["fit"] = hostinfo.fit_for_recipe(
            tensor_parallel=tp,
            min_nodes=min_nodes,
            weight_gb=weight_gb,
            cluster_only=cluster_only,
            solo_only=solo_only,
        )


# ----------------------------- registry path ------------------------------

def _from_registry_recipe(rec: registry.Recipe, requested_repo: str | None, source_kind: str, adapted_from: str | None = None) -> dict[str, Any]:
    """Turn a registry.Recipe into the public Forge recipe shape.

    The registry metadata is embedded at ``args._registry`` so that when this
    recipe round-trips through the existing Run/Save flow it carries enough
    context for ``docker_recipe.prepare_run`` to render a working raw_cmd.
    """
    model = requested_repo or rec.model
    name_suffix = "Registry" if source_kind == "registry" else "Adapted"
    notes_lines: list[str] = []
    if rec.description:
        notes_lines.append(rec.description)
    if source_kind == "similar" and adapted_from:
        notes_lines.append(f"Adapted from registry recipe: {adapted_from}")
    notes_lines.append(f"Source: {rec.source_repo}/{rec.source_path}")
    tags = [source_kind, rec.engine]
    if rec.container:
        tags.append("docker")
    for mod in rec.mods:
        tag = mod.split("/")[-1]
        if tag and tag not in tags:
            tags.append(tag)
    registry_block = {
        "container": rec.container,
        "mods": list(rec.mods or []),
        "command": rec.command,
        "defaults": dict(rec.defaults or {}),
        "raw_yaml": rec.raw_yaml,
        "origin": {"repo": rec.source_repo, "path": rec.source_path},
        "adapted_from_model": rec.model if source_kind == "similar" else None,
    }
    args: dict[str, Any] = {"_registry": registry_block}
    if model:
        args["model"] = model
    return {
        "name": f"{model or rec.name} · {name_suffix}",
        "engine": rec.engine,
        "model": model,
        "args": args,
        "env": dict(rec.env or {}),
        "notes": "\n".join(notes_lines),
        "tags": ",".join(tags),
        "source": source_kind,
        "registry": registry_block,
    }


# ----------------------------- synth path --------------------------------

def _synth_recipe(report: dict[str, Any]) -> dict[str, Any] | None:
    """Wrap recipe_brain.synthesize_recipe into the public Forge shape.

    The synthesized YAML rides through the same ``_registry`` block path
    that exact / similar matches use, so ``docker_recipe.prepare_run`` runs
    it via ``run-recipe.sh`` and gets identical behavior to a curated
    recipe — same image build, same mod application, same launch script.
    """
    try:
        host = hostinfo.probe_host()
    except Exception:  # noqa: BLE001
        host = None
    synth = recipe_brain.synthesize_recipe(report, host=host)
    if synth is None:
        return None

    parsed = synth["parsed"]
    profile = synth["profile"]
    repo = report.get("repo")

    registry_block = {
        "container": parsed["container"],
        "mods": list(parsed["mods"]),
        "command": parsed["command"],
        "defaults": dict(parsed["defaults"]),
        "raw_yaml": synth["raw_yaml"],
        # No origin — this YAML is materialized in app/data/forged/ at run time.
        "origin": None,
        "adapted_from_model": None,
        "build_args": list(parsed.get("build_args") or []),
        "env": dict(parsed.get("env") or {}),
        "synth_profile": profile,
    }

    args: dict[str, Any] = {"_registry": registry_block}
    if repo:
        args["model"] = repo

    tags = ["synth", "vllm", "docker", profile["family"], profile["quant"]]
    for m in parsed["mods"]:
        tag = m.split("/")[-1]
        if tag and tag not in tags:
            tags.append(tag)

    return {
        "name": f"{repo} · Synthesized ({profile['family']}/{profile['quant']})",
        "engine": "vllm",
        "model": repo,
        "args": args,
        "env": dict(parsed.get("env") or {}),
        "notes": profile["rationale"],
        "tags": ",".join(t for t in tags if t),
        "source": "synth",
        "registry": registry_block,
    }


# ----------------------------- heuristic path -----------------------------

def _heuristic(report: dict[str, Any]) -> list[dict[str, Any]]:
    repo = report.get("repo")
    weight_gb = report.get("weight_gb") or 0
    context = report.get("context") or 4096
    arch = (report.get("architecture") or "").lower()
    tags = report.get("tags") or []
    engines = report.get("suggested_engines") or ["vllm"]

    is_moe = any(k in arch for k in ["moe", "mixtral", "deepseek"])
    is_gguf = "gguf" in tags

    recipes: list[dict[str, Any]] = []

    if "vllm" in engines:
        recipes.append({
            "name": f"{repo} · vLLM throughput",
            "engine": "vllm",
            "model": repo,
            "args": {
                "dtype": "auto",
                "max-model-len": min(context, 16384),
                "gpu-memory-utilization": 0.90,
                "enable-chunked-prefill": True,
                "trust-remote-code": True,
            },
            "notes": "Heuristic preset (no registry match). Balanced throughput.",
            "tags": "heuristic,vllm,throughput",
            "source": "heuristic",
        })
        recipes.append({
            "name": f"{repo} · vLLM low-latency",
            "engine": "vllm",
            "model": repo,
            "args": {
                "dtype": "auto",
                "max-model-len": 4096,
                "gpu-memory-utilization": 0.85,
                "max-num-seqs": 8,
                "enforce-eager": True,
                "trust-remote-code": True,
            },
            "notes": "Heuristic preset (no registry match). Eager mode, low TTFT.",
            "tags": "heuristic,vllm,latency",
            "source": "heuristic",
        })
        if weight_gb and weight_gb < 60:
            recipes.append({
                "name": f"{repo} · vLLM long-context",
                "engine": "vllm",
                "model": repo,
                "args": {
                    "dtype": "auto",
                    "max-model-len": min(context, 131072),
                    "gpu-memory-utilization": 0.92,
                    "enable-chunked-prefill": True,
                    "enable-prefix-caching": True,
                    "trust-remote-code": True,
                },
                "notes": "Heuristic preset (no registry match). Long context.",
                "tags": "heuristic,vllm,long-context",
                "source": "heuristic",
            })
        if is_moe:
            recipes.append({
                "name": f"{repo} · vLLM MoE-aware",
                "engine": "vllm",
                "model": repo,
                "args": {
                    "dtype": "auto",
                    "max-model-len": min(context, 8192),
                    "gpu-memory-utilization": 0.88,
                    "enable-expert-parallel": True,
                    "trust-remote-code": True,
                },
                "notes": "Heuristic preset (no registry match). Expert parallelism.",
                "tags": "heuristic,vllm,moe",
                "source": "heuristic",
            })

    if "sglang" in engines:
        recipes.append({
            "name": f"{repo} · SGLang",
            "engine": "sglang",
            "model": repo,
            "args": {
                "context-length": min(context, 16384),
                "mem-fraction-static": 0.88,
                "disable-cuda-graph": False,
                "trust-remote-code": True,
            },
            "notes": "Heuristic preset (no registry match). SGLang default.",
            "tags": "heuristic,sglang",
            "source": "heuristic",
        })

    if is_gguf or "llamacpp" in engines:
        recipes.append({
            "name": f"{repo} · llama.cpp",
            "engine": "llamacpp",
            "model": repo,
            "args": {
                "ctx-size": min(context, 8192),
                "n-gpu-layers": 999,
                "parallel": 4,
                "cont-batching": True,
                # Current llama.cpp takes a value (on|off|auto); a bare
                # --flash-attn would swallow the next flag as its value.
                "flash-attn": "on",
            },
            "notes": "Heuristic preset (no registry match). Full GPU offload.",
            "tags": "heuristic,llamacpp,gguf",
            "source": "heuristic",
        })

    return recipes


# ----------------------------- public API ---------------------------------

def _finalize(out: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    """Give every forged vLLM recipe the full native context (capped at 262144),
    a healthy prefill batch, and auto-detected tool/reasoning parsers, then attach
    the hardware fit verdict."""
    native = report.get("context") if isinstance(report, dict) else None
    for r in out:
        recipe_brain.apply_perf_defaults(r, native_context=native, add_capabilities=True)
    _attach_fit(out, report)
    return out


def forge(report: dict[str, Any]) -> list[dict[str, Any]]:
    repo = report.get("repo")
    out: list[dict[str, Any]] = []

    if repo:
        exact = registry.by_exact_repo(repo)
        if exact:
            for rec in exact:
                out.append(_from_registry_recipe(rec, repo, "registry"))
            return _finalize(out, report)

        # No exact hit — surface adapted curated recipes (if any) AND a
        # freshly synthesized spark-vllm-docker shaped recipe for the repo.
        # Synth comes first because it actually targets the requested model.
        synth = _synth_recipe(report)
        if synth:
            out.append(synth)

        similar = registry.by_similar(repo, k=3)
        for rec in similar:
            out.append(_from_registry_recipe(
                rec, repo, "similar",
                adapted_from=rec.model or rec.name,
            ))

        # Heuristic flat-args presets cover SGLang / llama.cpp paths and act
        # as a fallback for unusual repos.
        out.extend(_heuristic(report))
        return _finalize(out, report)

    return _finalize(_heuristic(report), report)
