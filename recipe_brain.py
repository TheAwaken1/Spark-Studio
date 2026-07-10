"""Synthesize spark-vllm-docker shaped recipes for arbitrary HF repos.

The eugr/spark-vllm-docker recipe set encodes tribal knowledge: which
container variant pairs with which quant (``vllm-node-mxfp4`` for MXFP4,
``vllm-node-tf5`` for transformers-5 models like GLM-4.7 / Gemma 4),
which tool/reasoning parser belongs to which family, which mods unbreak
which combo, which flag stack each quant needs (MXFP4's CUTLASS +
FLASHINFER + kv-fp8 trio; NVFP4's MoE-cutlass + TRITON_ATTN; native FP8's
flashinfer + fastsafetensors). Forge previously fell back to a flat
``args`` dict when no registry match existed, which couldn't use the
docker path at all. This module captures that knowledge as data and emits
a real v1 spark-vllm-docker YAML for any repo so the same
``run-recipe.sh`` pipeline drives it.

The output is intentionally shaped exactly like eugr's curated recipes:
``recipe_version: "1"``, top-level ``name`` / ``description`` / ``model``
/ ``container`` / ``defaults`` / ``env`` / ``command``, plus ``mods`` and
``build_args`` when relevant.
"""

from __future__ import annotations

import re
from typing import Any

import registry


# Every vLLM recipe should serve the full context the model supports (capped at
# this ceiling) and use a healthy prefill batch. vLLM sizes the KV cache to the
# gpu-memory-utilization budget, so a high max-model-len sets the per-request
# ceiling without itself causing OOM — it just trades some max concurrency.
TARGET_MAX_MODEL_LEN = 262144
TARGET_MAX_NUM_BATCHED_TOKENS = 16384


# ----------------------------- detection ----------------------------------

_TOK_RE = re.compile(r"[a-z0-9.]+")


def _toks(*parts: str | None) -> list[str]:
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        out.extend(_TOK_RE.findall(str(p).lower()))
    return out


def detect_quant(report: dict[str, Any]) -> str:
    """Return the quant id we recognise: mxfp4 / nvfp4 / fp8 / awq /
    int4_autoround / gptq / gguf / unquantized.

    Sources, in order: explicit ``dtype`` from the HF check (which already
    folds in ``quantization_config.quant_method``), HF tags, then repo-name
    heuristics.
    """
    repo = report.get("repo") or ""
    dtype = (report.get("dtype") or "").lower()
    tags = [t.lower() for t in (report.get("tags") or [])]
    blob = " ".join([repo.lower(), dtype, *tags])

    # Exact dtype tokens first (most authoritative).
    if "mxfp4" in dtype or "mxfp4" in blob:
        return "mxfp4"
    if "nvfp4" in dtype or "nvfp4" in blob:
        return "nvfp4"
    if "autoround" in blob and ("int4" in blob or "4bit" in blob or "4-bit" in blob):
        return "int4_autoround"
    if "awq" in blob:
        return "awq"
    if "gptq" in blob:
        return "gptq"
    if "gguf" in blob:
        return "gguf"
    if dtype.startswith("fp8") or "fp8" in blob:
        return "fp8"
    return "unquantized"


def detect_family(report: dict[str, Any]) -> str:
    """Return the family id we recognise. Order matters — more specific
    matches come first."""
    repo = (report.get("repo") or "").lower()
    arch = (report.get("architecture") or "").lower()
    blob = repo + " " + arch

    if "glm-4.7" in blob or "glm47" in blob or "glm-4-7" in blob:
        if "flash" in blob:
            return "glm-4.7-flash"
        return "glm-4.7"
    if "gpt-oss" in blob or "gpt_oss" in blob or "gptoss" in blob:
        return "gpt-oss"
    if "minimax-m2" in blob or "minimax_m2" in blob:
        return "minimax-m2"
    if "nemotron" in blob and "omni" in blob:
        return "nemotron-omni"
    if "nemotron" in blob and "nano" in blob:
        return "nemotron-nano"
    if "nemotron" in blob and "super" in blob:
        return "nemotron-super"
    if "gemma-4" in blob or "gemma4" in blob:
        return "gemma-4"
    if "qwen3.5" in blob or "qwen-3.5" in blob:
        return "qwen3.5"
    if "qwen3-coder" in blob or "qwen3coder" in blob:
        return "qwen3-coder"
    if "qwen3" in blob:
        return "qwen3"
    return "generic"


# ----------------------------- profiles -----------------------------------

# Quant profile = container variant + build args + flag stack + load format.
# Mirrors what the curated eugr recipes settle on for each quant.
QUANT_PROFILES: dict[str, dict[str, Any]] = {
    "mxfp4": {
        "container": "vllm-node-mxfp4",
        "build_args": ["--exp-mxfp4"],
        "extra_flags": [
            "--quantization mxfp4",
            "--mxfp4-backend CUTLASS",
            "--mxfp4-layers moe,qkv,o,lm_head",
            "--attention-backend FLASHINFER",
            "--kv-cache-dtype fp8",
        ],
        "extra_env": {"VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": "1"},
        "load_format": "fastsafetensors",
    },
    "nvfp4": {
        "container": "vllm-node",
        "build_args": [],
        "extra_flags": [
            "--moe-backend cutlass",
            "--kv-cache-dtype fp8",
            "--attention-backend TRITON_ATTN",
        ],
        "extra_env": {},
        "load_format": "fastsafetensors",
    },
    "fp8": {
        "container": "vllm-node",
        "build_args": [],
        "extra_flags": [
            "--kv-cache-dtype fp8",
            "--attention-backend flashinfer",
        ],
        "extra_env": {},
        "load_format": "fastsafetensors",
    },
    "awq": {
        "container": "vllm-node",
        "build_args": [],
        "extra_flags": [],
        "extra_env": {},
        "load_format": "fastsafetensors",
    },
    "int4_autoround": {
        "container": "vllm-node",
        "build_args": [],
        "extra_flags": [],
        "extra_env": {"VLLM_MARLIN_USE_ATOMIC_ADD": "1"},
        "load_format": "fastsafetensors",
    },
    "gptq": {
        "container": "vllm-node",
        "build_args": [],
        "extra_flags": [],
        "extra_env": {},
        "load_format": "fastsafetensors",
    },
    "unquantized": {
        "container": "vllm-node",
        "build_args": [],
        "extra_flags": [],
        "extra_env": {},
        "load_format": "fastsafetensors",
    },
    # gguf intentionally omitted — that path goes through llama.cpp, not vLLM.
}


# Family profile = parsers, chat template, default trust-remote-code, and
# the suggested mod folder names. Mods are looked up in the registry mod
# index at synth time so we only attach mods that actually exist on disk.
FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "glm-4.7": {
        "tool_call_parser": "glm47",
        "reasoning_parser": "glm45",
        "chat_template": None,
        "trust_remote_code": True,
        "needs_tf5": True,
        "mod_hints": [],
    },
    "glm-4.7-flash": {
        "tool_call_parser": "glm47",
        "reasoning_parser": "glm45",
        "chat_template": None,
        "trust_remote_code": True,
        "needs_tf5": True,
        "mod_hints": ["glm-4.7-flash"],
    },
    "gpt-oss": {
        "tool_call_parser": "openai",
        "reasoning_parser": "openai_gptoss",
        "chat_template": None,
        "trust_remote_code": False,
        "needs_tf5": False,
        "mod_hints": [],
    },
    "minimax-m2": {
        "tool_call_parser": "minimax_m2",
        "reasoning_parser": "minimax_m2",
        "chat_template": None,
        "trust_remote_code": False,
        "needs_tf5": False,
        "mod_hints": [],
    },
    "nemotron-nano": {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "nano_v3",
        "extra_flags": ["--reasoning-parser-plugin nano_v3_reasoning_parser.py"],
        "chat_template": None,
        "trust_remote_code": True,
        "needs_tf5": False,
        "mod_hints": ["nemotron-nano"],
    },
    "nemotron-omni": {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "nemotron_v3",
        "chat_template": None,
        "trust_remote_code": True,
        "needs_tf5": False,
        "mod_hints": [],
        "default_model_alias": "nemotron",
        "default_flags": [
            "--served-model-name nemotron",
            "--video-pruning-rate 0.5",
            "--media-io-kwargs '{\"video\": {\"num_frames\": 512, \"fps\": 1}}'",
            "--max-num-seqs 8",
        ],
    },
    "nemotron-super": {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "nemotron_v3",
        "extra_flags": ["--mamba_ssm_cache_dtype float32"],
        "chat_template": None,
        "trust_remote_code": True,
        "needs_tf5": False,
        "mod_hints": ["nemotron-super"],
    },
    "gemma-4": {
        "tool_call_parser": "gemma4",
        "reasoning_parser": "gemma4",
        "chat_template": None,
        "trust_remote_code": False,
        "needs_tf5": True,
        "mod_hints": ["gemma4"],
    },
    "qwen3.5": {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "chat_template": "unsloth.jinja",
        "trust_remote_code": False,
        "needs_tf5": False,
        "mod_hints": ["qwen3-coder-next", "qwen3.5-chat-template"],
    },
    "qwen3-coder": {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": None,
        "chat_template": None,
        "trust_remote_code": False,
        "needs_tf5": False,
        "mod_hints": ["qwen3-coder-next"],
    },
    "qwen3": {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "chat_template": None,
        "trust_remote_code": False,
        "needs_tf5": False,
        "mod_hints": [],
    },
    "generic": {
        "tool_call_parser": None,
        "reasoning_parser": None,
        "chat_template": None,
        "trust_remote_code": True,
        "needs_tf5": False,
        "mod_hints": [],
    },
}


# ----------------------------- mod resolution -----------------------------

def _match_mods(hints: list[str]) -> list[str]:
    """Resolve mod hints to actual paths in the spark-vllm-docker mod
    catalog.

    We only attach mods that are still **actively referenced** by at least
    one curated recipe. eugr keeps deprecated mods in the catalog for
    reference (e.g. ``fix-glm-4.7-flash-AWQ`` patches a vLLM source file
    that has since moved), but commenting them out of every recipe is the
    signal that the patch no longer applies. A naive substring match over
    every available mod re-introduced exactly those deprecated patches and
    crashed the launch with ``Hunk #1 FAILED``. Filtering by
    ``actively_referenced`` keeps the synthesizer in sync with whatever
    eugr currently considers safe.
    """
    if not hints:
        return []
    actively_referenced: set[str] = set()
    for r in registry.all_recipes():
        for m in r.mods:
            actively_referenced.add(m)
    available = {m.name: m.source_path for m in registry.all_mods()
                 if m.source_repo == "spark-vllm-docker"
                 and m.source_path in actively_referenced}
    resolved: list[str] = []
    for hint in hints:
        for name, path in available.items():
            if name == hint or name == f"fix-{hint}" or hint in name:
                if path not in resolved:
                    resolved.append(path)
                break
    return resolved


# ----------------------------- defaults sizing ----------------------------

def _suggested_max_model_len(report: dict[str, Any], weight_gb: float | None,
                             host_memory_gb: float | None) -> int:
    """Default ``max_model_len``: the model's full native context, capped at
    TARGET_MAX_MODEL_LEN. vLLM allocates KV to the memory budget rather than to
    max_model_len, so serving the full context is safe — a smaller ceiling just
    needlessly truncates what the model can do. (weight_gb / host_memory_gb are
    unused now but kept for signature stability.)
    """
    native = int(report.get("context") or TARGET_MAX_MODEL_LEN)
    return max(2048, min(native, TARGET_MAX_MODEL_LEN))


# ----------------------------- synthesis ----------------------------------

def synthesize_recipe(
    report: dict[str, Any],
    host: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a full v1 spark-vllm-docker recipe for the repo in ``report``.

    Returns a dict with:
      - ``raw_yaml``: complete YAML text (run-recipe.py compatible)
      - ``parsed``:   parsed equivalent (so callers can index defaults etc.)
      - ``profile``:  human-readable summary of choices for the UI
      - ``container``, ``mods``, ``defaults``, ``env``, ``build_args``,
        ``command``: shortcuts the registry-block consumers expect

    Returns ``None`` only for genuinely unservable cases (e.g. GGUF, which
    belongs on llama.cpp not vLLM). The caller should fall back to the
    heuristic dict path in that case.
    """
    repo = report.get("repo")
    if not repo:
        return None

    quant = detect_quant(report)
    if quant == "gguf":
        return None  # llama.cpp territory

    family = detect_family(report)
    qprof = QUANT_PROFILES.get(quant, QUANT_PROFILES["unquantized"])
    fprof = FAMILY_PROFILES.get(family, FAMILY_PROFILES["generic"])

    # Container: family-required tf5 wins over quant default.
    container = "vllm-node-tf5" if fprof.get("needs_tf5") else qprof["container"]
    build_args = list(qprof.get("build_args") or [])
    if container == "vllm-node-tf5" and "--pre-tf" not in build_args and "--tf5" not in build_args:
        build_args.append("--pre-tf")

    # Mods.
    mods = _match_mods(list(fprof.get("mod_hints") or []))

    # Sizing.
    weight_gb = report.get("weight_gb")
    host_mem = (host or {}).get("effective_memory_gb") or (host or {}).get("total_memory_gb")
    gpu_count = (host or {}).get("effective_gpu_count") or 1
    max_model_len = _suggested_max_model_len(report, weight_gb, host_mem)
    # tensor_parallel: stay at 1 for solo Sparks; if a mesh is set up and the
    # model is fat, scale up. We'll let run-recipe.py's solo override fire if
    # the user runs solo anyway.
    tp = 1
    if gpu_count >= 2 and weight_gb and host_mem and weight_gb > host_mem * 0.5:
        tp = min(gpu_count, 4)

    defaults: dict[str, Any] = {
        "port": 8000,
        "host": "0.0.0.0",
        "tensor_parallel": tp,
        "gpu_memory_utilization": 0.7,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": TARGET_MAX_NUM_BATCHED_TOKENS,
    }

    env = dict(qprof.get("extra_env") or {})

    # Command assembly. We bake the repo into the serve line (eugr's recipes
    # do the same — easier to read and keeps the substitution surface small).
    flag_lines: list[str] = []

    def add(flag: str) -> None:
        if flag and flag not in flag_lines:
            flag_lines.append(flag)

    # Standard quality-of-life flags every eugr recipe sets.
    add("--enable-prefix-caching")
    add(f"--load-format {qprof['load_format']}")
    if fprof.get("trust_remote_code"):
        add("--trust-remote-code")

    # Quant-specific stack.
    for f in qprof.get("extra_flags") or []:
        add(f)

    # Family defaults that aren't parser/template related.
    for f in fprof.get("default_flags") or []:
        add(f)

    # Family parsers / templates.
    if fprof.get("tool_call_parser"):
        add("--enable-auto-tool-choice")
        add(f"--tool-call-parser {fprof['tool_call_parser']}")
    if fprof.get("reasoning_parser"):
        add(f"--reasoning-parser {fprof['reasoning_parser']}")
    if fprof.get("chat_template"):
        add(f"--chat-template {fprof['chat_template']}")
    for f in fprof.get("extra_flags") or []:
        add(f)

    # Sizing flags.
    add("--gpu-memory-utilization {gpu_memory_utilization}")
    add("--max-model-len {max_model_len}")
    add("--max-num-batched-tokens {max_num_batched_tokens}")
    add("--host {host}")
    add("--port {port}")
    if tp > 1:
        add("-tp {tensor_parallel}")
        add("--distributed-executor-backend ray")

    cmd = "vllm serve " + repo + " \\\n    " + " \\\n    ".join(flag_lines)

    # Compose YAML directly (preserves comments and reads cleanly; we don't
    # need yaml.safe_dump's formatting fidelity here).
    name = f"{repo.split('/')[-1]}-{quant}"
    description = f"Synthesized recipe for {repo} ({family} family · {quant} quant)"
    yaml_lines = [
        "# Auto-synthesized by recipe_brain — modeled on eugr/spark-vllm-docker",
        f"# Family: {family}    Quant: {quant}",
        'recipe_version: "1"',
        f"name: {name}",
        f"description: {description}",
        f"model: {repo}",
        f"container: {container}",
    ]
    if build_args:
        yaml_lines.append("build_args:")
        for a in build_args:
            yaml_lines.append(f"  - {a}")
    yaml_lines.append("mods:" + (" []" if not mods else ""))
    for m in mods:
        yaml_lines.append(f"  - {m}")
    yaml_lines.append("defaults:")
    for k, v in defaults.items():
        if isinstance(v, str):
            yaml_lines.append(f"  {k}: {v}")
        else:
            yaml_lines.append(f"  {k}: {v}")
    yaml_lines.append("env:" + (" {}" if not env else ""))
    for k, v in env.items():
        yaml_lines.append(f'  {k}: "{v}"')
    yaml_lines.append("command: |")
    for line in cmd.splitlines():
        yaml_lines.append(f"  {line}")
    raw_yaml = "\n".join(yaml_lines) + "\n"

    parsed = {
        "recipe_version": "1",
        "name": name,
        "description": description,
        "model": repo,
        "container": container,
        "build_args": build_args,
        "mods": mods,
        "defaults": defaults,
        "env": env,
        "command": cmd,
    }

    profile = {
        "family": family,
        "quant": quant,
        "container": container,
        "mods": mods,
        "served_model_name": fprof.get("default_model_alias"),
        "tool_call_parser": fprof.get("tool_call_parser"),
        "reasoning_parser": fprof.get("reasoning_parser"),
        "chat_template": fprof.get("chat_template"),
        "build_args": build_args,
        "rationale": _explain_choices(family, quant, container, mods, fprof),
    }

    return {
        "raw_yaml": raw_yaml,
        "parsed": parsed,
        "profile": profile,
        "container": container,
        "build_args": build_args,
        "mods": mods,
        "defaults": defaults,
        "env": env,
        "command": cmd,
    }


# ----------------------------- capabilities -------------------------------

def capabilities_for(model: str | None) -> dict[str, Any]:
    """Which tool-call / reasoning parsers a model's family supports.

    Parsers are family-specific — passing the wrong one makes vLLM reject tool
    calls or crash — so an unrecognized family returns None for both, which is
    how "not all models get it" is enforced. ``supports_*`` are convenience
    booleans for the UI toggle.
    """
    family = detect_family({"repo": model or ""})
    fprof = FAMILY_PROFILES.get(family, FAMILY_PROFILES["generic"])
    tool = fprof.get("tool_call_parser")
    reasoning = fprof.get("reasoning_parser")
    return {
        "family": family,
        "tool_call_parser": tool,
        "reasoning_parser": reasoning,
        "supports_tools": bool(tool),
        "supports_reasoning": bool(reasoning),
    }


def apply_perf_defaults(
    recipe: dict[str, Any],
    native_context: int | None = None,
    add_capabilities: bool = True,
) -> dict[str, Any]:
    """Give a vLLM recipe the full native context (capped at TARGET_MAX_MODEL_LEN),
    a healthy prefill batch, and — when ``add_capabilities`` — the right tool /
    reasoning parsers for a recognized model family. Mutates and returns the
    recipe. No-op for non-vLLM, raw_cmd, or sparkrun recipes.

    ``add_capabilities`` is True on recipe creation (Forge) so new recipes come
    with tool/reasoning pre-wired; the editor passes False so its explicit
    on/off toggle stays authoritative and isn't re-added on save.
    """
    if not isinstance(recipe, dict):
        return recipe
    if (recipe.get("engine") or "").lower() != "vllm" or recipe.get("raw_cmd"):
        return recipe

    args = recipe.get("args")
    if not isinstance(args, dict):
        args = {}
        recipe["args"] = args

    max_len = min(int(native_context), TARGET_MAX_MODEL_LEN) if native_context else TARGET_MAX_MODEL_LEN
    max_len = max(2048, max_len)

    # Registry / docker-shaped recipe: values live in the _registry defaults and
    # reach vLLM via run-recipe.sh (max_model_len as a CLI override, batch via
    # the command template's {max_num_batched_tokens} placeholder).
    reg = args.get("_registry")
    if isinstance(reg, dict):
        defaults = reg.get("defaults")
        if not isinstance(defaults, dict):
            defaults = {}
            reg["defaults"] = defaults
        defaults["max_model_len"] = max_len
        defaults["max_num_batched_tokens"] = TARGET_MAX_NUM_BATCHED_TOKENS
        return recipe

    # Flat vLLM recipe: kebab-case flag keys the runner turns into --flags.
    args["max-model-len"] = max_len
    args["max-num-batched-tokens"] = TARGET_MAX_NUM_BATCHED_TOKENS

    if add_capabilities:
        caps = capabilities_for(recipe.get("model") or args.get("model"))
        if caps["tool_call_parser"] and "tool-call-parser" not in args:
            args["enable-auto-tool-choice"] = True
            args["tool-call-parser"] = caps["tool_call_parser"]
        if caps["reasoning_parser"] and "reasoning-parser" not in args:
            args["reasoning-parser"] = caps["reasoning_parser"]
    return recipe


def _explain_choices(family: str, quant: str, container: str, mods: list[str],
                     fprof: dict[str, Any]) -> str:
    bits = [f"detected {family} family + {quant} quant → container {container}"]
    if mods:
        bits.append("applied mods " + ", ".join(m.split("/")[-1] for m in mods))
    if fprof.get("tool_call_parser"):
        bits.append(f"tool parser {fprof['tool_call_parser']}")
    if fprof.get("reasoning_parser"):
        bits.append(f"reasoning parser {fprof['reasoning_parser']}")
    if fprof.get("default_model_alias"):
        bits.append(f"served model name {fprof['default_model_alias']}")
    if fprof.get("chat_template"):
        bits.append(f"chat template {fprof['chat_template']}")
    return "; ".join(bits) + "."
