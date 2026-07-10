"""Starter-model recommendations for the first-run wizard and Overview.

Ranks launchable candidates from what this box actually has — nothing is
hardcoded to a model name, so recommendations don't rot as the registry moves:

  1. Saved recipes tagged ✓ working   — proven on THIS machine (best signal)
  2. Locally cached HF models          — real on-disk size, no download wait
  3. Synced registry recipes (solo)    — community-validated for DGX Spark

Extra signals: quick-bench history (measured tok/s beats any estimate),
MoE active-parameter counts (the speed lever on bandwidth-bound GB10),
quant-aware weight estimates against unified memory, and
recipe_brain.capabilities_for() for the tool-calling category.

Categories: fastest · best_quality · coding · tool_calling · low_memory.
"""

from __future__ import annotations

import re
import time
from typing import Any

import db
import hostinfo
import models as models_mod
import recipe_brain
import registry


# Total params: the LAST size token in the name (e.g. "Qwen3.6-35B-A3B" → 35).
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb]\b")
# MoE active params: "-A3B" / "-a10b" suffixed token.
_ACTIVE_RE = re.compile(r"[-_][Aa](\d+(?:\.\d+)?)[Bb]\b")
_CODING_RE = re.compile(r"\b(coder|coding|code|codestral|starcoder|devstral)\b", re.I)
# Not chat models — never recommend as a starter (embedding/reranker/safety).
_NOT_CHAT_RE = re.compile(r"embed|rerank|guard|safety|classifier", re.I)

# Bytes per parameter by quant token found in the model name/tags.
_QUANT_BPP = [
    (("nvfp4", "mxfp4", "int4", "4bit", "awq", "gptq", "autoround", "q4"), 0.55),
    (("fp8", "int8", "8bit", "q8"), 1.05),
    (("bf16", "fp16", "f16"), 2.0),
]


def _parse_size(model: str) -> tuple[float | None, float | None]:
    """(total_params_b, active_params_b) from the model name."""
    total = None
    sizes = _PARAMS_RE.findall(model or "")
    active_m = _ACTIVE_RE.search(model or "")
    active = float(active_m.group(1)) if active_m else None
    for s in sizes:
        v = float(s)
        if active is not None and v == active:
            continue  # skip the A3B token when looking for total size
        total = max(total or 0, v)
    return total, active


def _quant_of(model: str) -> str | None:
    low = (model or "").lower()
    for tokens, _ in _QUANT_BPP:
        for t in tokens:
            if t in low:
                return t
    return None


def _est_weight_gb(model: str, total_b: float | None) -> float | None:
    if not total_b:
        return None
    low = (model or "").lower()
    bpp = 2.0
    for tokens, val in _QUANT_BPP:
        if any(t in low for t in tokens):
            bpp = val
            break
    return round(total_b * bpp, 1)


def _bench_tps_by_model() -> dict[str, float]:
    """model id → best measured tok/s from quick-bench history."""
    out: dict[str, float] = {}
    try:
        recipes = {r["id"]: r for r in db.recipes_list()}
        for row in db.bench_list(limit=300):
            tps = row.get("tokens_per_sec")
            rec = recipes.get(row.get("recipe_id"))
            if not tps or not rec:
                continue
            model = rec.get("model")
            if model and tps > out.get(model, 0):
                out[model] = round(float(tps), 1)
    except Exception:  # noqa: BLE001
        pass
    return out


def _collect() -> list[dict[str, Any]]:
    """Merge all sources into one candidate list keyed by model id."""
    host = hostinfo.probe_host()
    mem_budget = (host.get("total_memory_gb") or 128) * 0.75
    tps_by_model = _bench_tps_by_model()

    cached: dict[str, float] = {}
    try:
        for m in models_mod.scan():
            cached[m["repo"]] = m["size_gb"]
    except Exception:  # noqa: BLE001
        pass

    by_model: dict[str, dict[str, Any]] = {}

    def _add(model: str, *, source: str, name: str, recipe: dict[str, Any],
             proven: bool = False) -> None:
        model = (model or "").strip()
        if not model or "/" not in model or _NOT_CHAT_RE.search(model):
            return
        total_b, active_b = _parse_size(model)
        entry = by_model.get(model)
        if entry is None:
            weight = cached.get(model) or _est_weight_gb(model, total_b)
            entry = by_model[model] = {
                "model": model,
                "name": name,
                "source": source,
                "recipe": recipe,
                "proven": proven,
                "cached": model in cached,
                "params_b": total_b,
                "active_params_b": active_b,
                "quant": _quant_of(model),
                "est_weight_gb": weight,
                "tokens_per_sec": tps_by_model.get(model),
                "fits": weight is None or weight <= mem_budget,
                "caps": recipe_brain.capabilities_for(model),
            }
            return
        # Merge signals: a proven saved recipe beats a registry duplicate.
        if proven and not entry["proven"]:
            entry.update(proven=True, source=source, name=name, recipe=recipe)

    # 1) Saved recipes — "working" tag means it served successfully here.
    try:
        for r in db.recipes_list():
            tags = {t.strip() for t in (r.get("tags") or "").split(",")}
            if "fix" in tags and "working" not in tags:
                continue  # known broken
            model = r.get("model") or (r.get("args") or {}).get("model")
            if not model:
                continue
            _add(model, source="your recipe", name=r.get("name") or model,
                 recipe={k: r.get(k) for k in ("id", "name", "engine", "model", "args", "env", "tags", "raw_cmd")},
                 proven="working" in tags)
    except Exception:  # noqa: BLE001
        pass

    # 2) Registry recipes (single-node only — this is a starter flow).
    try:
        import forge as forge_mod
        for rec in registry.all_recipes():
            if rec.min_nodes > 1 or not rec.model:
                continue
            _add(rec.model, source="community recipe",
                 name=rec.name or rec.model,
                 recipe=forge_mod._from_registry_recipe(rec, rec.model, "registry"))
    except Exception:  # noqa: BLE001
        pass

    # 3) Cached models with no recipe anywhere: launchable via a flat vLLM
    #    recipe (save applies context/batch/capability defaults). GGUF caches
    #    are skipped — they need a llama.cpp file path, not a repo id.
    for repo in cached:
        if repo in by_model or "gguf" in repo.lower():
            continue
        _add(repo, source="cached model", name=repo,
             recipe={"name": repo, "engine": "vllm", "model": repo, "args": {}, "env": {},
                     "tags": "starter"})

    return [e for e in by_model.values() if e["fits"]]


def _speed_score(e: dict[str, Any]) -> float:
    """Higher = faster expected generation on GB10 (bandwidth-bound)."""
    if e.get("tokens_per_sec"):
        base = float(e["tokens_per_sec"])          # measured truth
    else:
        active = e.get("active_params_b") or e.get("params_b") or 30.0
        weight_per_tok = max(active, 1.0)
        base = 100.0 / weight_per_tok              # rough: bandwidth / active weights
        if e.get("quant") == "nvfp4":
            base *= 0.4  # NVFP4 kernels aren't FlashInfer-optimized on sm_121 yet
    if e.get("proven"):
        base *= 1.5
    if e.get("cached"):
        base *= 1.3    # no download wait — big deal for a first run
    return base


def _quality_score(e: dict[str, Any]) -> float:
    total = e.get("params_b") or 0.0
    prec = {"bf16": 1.15, "fp16": 1.15, "fp8": 1.1, "int8": 1.05}.get(e.get("quant") or "", 1.0)
    bonus = (1.3 if e.get("proven") else 1.0) * (1.1 if e.get("cached") else 1.0)
    return total * prec * bonus


def _why(e: dict[str, Any], category: str) -> str:
    bits: list[str] = []
    if e.get("proven"):
        bits.append("ran successfully on this Spark")
    if e.get("tokens_per_sec"):
        bits.append(f"measured {e['tokens_per_sec']:.0f} tok/s here")
    if e.get("cached"):
        bits.append("already downloaded")
    if category == "fastest" and e.get("active_params_b"):
        bits.append(f"MoE — only {e['active_params_b']:g}B active per token")
    if category == "best_quality" and e.get("params_b"):
        bits.append(f"{e['params_b']:g}B parameters, fits in unified memory")
    if category == "tool_calling":
        caps = e.get("caps") or {}
        if caps.get("tool_call_parser"):
            bits.append(f"native tool calling ({caps['tool_call_parser']} parser)")
    if category == "low_memory" and e.get("est_weight_gb"):
        bits.append(f"~{e['est_weight_gb']:g} GB — leaves room for other work")
    if e.get("source") == "community recipe":
        bits.append("Spark-validated community recipe")
    return "; ".join(bits) or "fits this Spark"


def _public(e: dict[str, Any], category: str) -> dict[str, Any]:
    return {
        "category": category,
        "model": e["model"],
        "name": e["name"],
        "source": e["source"],
        "reason": _why(e, category),
        "proven": e["proven"],
        "cached": e["cached"],
        "params_b": e["params_b"],
        "active_params_b": e["active_params_b"],
        "quant": e["quant"],
        "est_weight_gb": e["est_weight_gb"],
        "tokens_per_sec": e["tokens_per_sec"],
        "supports_tools": (e.get("caps") or {}).get("supports_tools", False),
        "recipe": e["recipe"],
    }


def recommend(k: int = 3) -> dict[str, Any]:
    """Top-k candidates per category, ranked from live local signals."""
    entries = _collect()
    host = hostinfo.probe_host()

    def top(pool: list[dict[str, Any]], score, category: str) -> list[dict[str, Any]]:
        ranked = sorted(pool, key=score, reverse=True)[:k]
        return [_public(e, category) for e in ranked]

    categories = {
        "fastest": top(entries, _speed_score, "fastest"),
        "best_quality": top(entries, _quality_score, "best_quality"),
        "coding": top([e for e in entries if _CODING_RE.search(e["model"])],
                      _speed_score, "coding"),
        "tool_calling": top([e for e in entries if (e.get("caps") or {}).get("supports_tools")],
                            lambda e: _speed_score(e) * (1.2 if e.get("proven") else 1.0),
                            "tool_calling"),
        "low_memory": top(
            sorted([e for e in entries if (e.get("est_weight_gb") or 0) > 0
                    and (e.get("params_b") or 0) >= 1.5],
                   key=lambda e: e["est_weight_gb"])[: k * 2],
            lambda e: -(e.get("est_weight_gb") or 999), "low_memory"),
    }
    return {
        "generated_at": time.time(),
        "host": host.get("summary"),
        "candidates": len(entries),
        "categories": categories,
    }
