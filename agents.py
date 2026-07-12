"""Bridge to Claude Code CLI and OpenAI Codex CLI.

We delegate all OAuth to the official CLIs:
  - `claude` (@anthropic-ai/claude-code) — Pro/Max subscription login via `claude /login`
  - `codex`  (@openai/codex)             — ChatGPT login via `codex login`

Both store creds in the user's home dir after login; afterward we just shell
out with a --print/exec flag to ask them questions. No API keys in this app.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess as _subprocess
from typing import AsyncIterator

import registry

_PROMPT_COMMON_HEADER = """You are helping a user run and optimize an LLM inference recipe on NVIDIA DGX Spark
(Grace-Blackwell GB10, sm_121, aarch64, CUDA 13, 128 GB unified memory, NVLink-C2C bandwidth).

THE ENGINE IS: __ENGINE__
ONLY apply flags, schemas, and strategies that are valid for __ENGINE__. Do NOT mix flags from different engines.

Return ONLY a valid JSON object — no markdown fences, no prose outside the JSON:
{
  "diagnosis": "one-paragraph explanation of what went wrong and/or what to change",
  "patched_recipe": { ... },
  "diff_notes": ["bullet: what changed and why"]
}
"""

_PROMPT_VLLM = """
=== vLLM ENGINE RULES ===

vLLM on DGX Spark runs inside spark-vllm-docker container images — NOT via pip install.
PyPI aarch64 torch is CPU-only and vLLM's C extension is linked to CUDA 12; both fail on CUDA 13.

OUTPUT SCHEMA for patched_recipe:
{
  "engine": "vllm",
  "model": "org/model-name",
  "args": {
    "_registry": {
      "container": "vllm-node",
      "mods": [],
      "command": "vllm serve org/model-name --gpu-memory-utilization 0.90 ...",
      "defaults": {"port": 8000, "host": "0.0.0.0", "tensor_parallel": 1, "gpu_memory_utilization": 0.9, "max_model_len": 32768},
      "raw_yaml": "recipe_version: '1'\\nname: Model Name\\nmodel: org/model-name\\ncontainer: vllm-node\\ncommand: |\\n  vllm serve org/model-name --gpu-memory-utilization 0.90\\ndefaults:\\n  port: 8000\\n",
      "origin": {"repo": "recipe-registry", "path": "official-recipes/..."},
      "adapted_from_model": null
    }
  },
  "env": {},
  "raw_cmd": null
}

CONTAINERS — ONLY use names from the "Docker images available locally" list in CONTEXT above.
The three standard spark-vllm-docker names are:
- vllm-node          → standard (default for most models)
- vllm-node-mxfp4    → MXFP4 quantization (2× tok/s for supported models)
- vllm-node-tf5      → GLM / Gemma-4 families (needs transformers 5)

CRITICAL: NEVER invent container names. Never use Docker Hub paths like "sparkrun-eugr-*",
"eugr/*", or any name not in the locally-available images list. These are locally-built images,
not Docker Hub images — Docker will fail to pull them. If the needed image is not in the list,
fall back to the nearest available one (prefer vllm-node for general use).
If a ghcr.io/spark-arena/* image is in the local list, you may use it as raw_cmd instead of _registry.

STRATEGY:
- PREFERRED: Use _registry with raw_yaml. Launcher runs: run-recipe.sh --solo --setup --port <auto> --host 127.0.0.1 <yaml>
- raw_yaml MUST be valid spark-vllm-docker v1 YAML (recipe_version: "1").
- origin MUST be a JSON object {"repo": "...", "path": "..."}, never a plain string.
- If a curated recipe exists in CONTEXT, adopt its raw_yaml verbatim.
- FALLBACK only if docker pipeline itself is broken: set raw_cmd to a docker run command, set args to {}.

CRASH FIXES:
- --enforce-eager          (sm_121 torch.compile fails)
- --max-model-len <lower>  (OOM)
- --kv-cache-dtype fp8     (memory pressure)
- --gpu-memory-utilization 0.85  (OOM)
- "Unknown architecture" / "model type … not supported" / "no model executor": the RUNNER IMAGE
  is stale, not the recipe — new architectures land in vLLM nightlies. Do NOT flag-tweak around it;
  tell the user to update the engine image (vLLM tab → Engine images → "Update to tested nightly",
  i.e. build-and-copy.sh) and relaunch.
- SILENT HANG during load (logs stop right after backend/kernel selection, near-idle CPU):
  the classic signature is an NVFP4 model whose GEMMs select FlashInfer NVFP4 kernels
  ("FlashInferCutlassNvFp4LinearKernel") — not fully working on sm_121. The reliable fix
  is switching to an FP8/AWQ quantization of the same model; do NOT just retry the NVFP4.
  If the hang is mid-weight-load instead, drop --load-format fastsafetensors.

PERFORMANCE / OPTIMIZATION (use when goal is to maximize tok/s):
- --attention-backend FLASHINFER  (the fast default on GB10; every fast community recipe uses it)
  EXCEPTION: NVFP4-quantized models are NOT fully FlashInfer-supported on sm_121 yet — use the
  Marlin backend (or triton) for NVFP4, and prefer FP8/AWQ/MXFP4 quants of the same model today.
- --kv-cache-dtype fp8            (half the KV cache → larger batches; drop it only if quality regresses)
- --max-num-batched-tokens 16384  (batching amortizes memory bandwidth — the #1 lever on Spark; 8192 for gpt-oss-120b)
- --max-model-len: use the model's FULL native context, capped at 262144 (e.g. a 262144-native
  Qwen3 gets 262144; a 131072 model gets 131072). vLLM sizes the KV cache to the
  gpu-memory-utilization budget, NOT to max-model-len, so a high ceiling does not cause paging —
  it only trades some max concurrency. Only lower it to fix an actual OOM at load.
- --gpu-memory-utilization 0.80   (community default; 0.70 for models near capacity — on unified
  memory pushing 0.92 causes system memory pressure, NOT more speed)
- --load-format fastsafetensors   (much faster load; AVOID if weights > 0.85 of available RAM — OOM risk)
- --enable-prefix-caching         (free win for repeated/agentic prompts)
- --quantization mxfp4 --mxfp4-backend CUTLASS --mxfp4-layers moe,qkv,o,lm_head + container vllm-node-mxfp4  (2× tok/s for supported models, e.g. gpt-oss)
- MoE model env vars (set in recipe env, per family):
  VLLM_USE_FLASHINFER_MOE_FP4=1 + VLLM_FLASHINFER_MOE_BACKEND=throughput   (NVFP4 MoE)
  VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1                                    (gpt-oss MXFP4 path)
  VLLM_MARLIN_USE_ATOMIC_ADD=1                                             (Qwen3.6 FP8)
- Tuned Triton MoE kernels: if ~/.cache/sparkrun/tuning/vllm/ has configs for this model, mount it
  and set VLLM_TUNED_CONFIG_FOLDER — untuned fused-MoE kernels are the #1 cause of slow MoE recipes.
  (Generate once via: sparkrun tune vllm <recipe> -H localhost --tp 1)
- --disable-log-requests          (lower CPU overhead at high QPS)
- AVOID --enforce-eager unless required — disables cuda graphs, costs ~20% throughput
"""

_PROMPT_SGLANG = """
=== SGLang ENGINE RULES ===

SGLang on DGX Spark can run natively (if installed) or in a container.
The recipe uses engine="sglang". Do NOT apply vLLM flags or docker pipelines.

OUTPUT SCHEMA for patched_recipe:
{
  "engine": "sglang",
  "model": "org/model-name",
  "args": {
    "model-path": "org/model-name",
    "tp": 1,
    "host": "0.0.0.0",
    "port": 30000,
    "mem-fraction-static": 0.88
  },
  "env": {},
  "raw_cmd": null
}

VALID SGLang FLAGS (flat kebab-case in args, no _registry):
- --model-path org/model-name   (required)
- --tp 1                        (tensor parallel; keep 1 for single-GPU)
- --host 0.0.0.0 --port 30000
- --mem-fraction-static 0.88    (fraction of GPU memory for static allocation)
- --max-running-requests 512
- --attention-backend flashinfer (faster on Blackwell)
- --chunked-prefill-size 8192
- --dtype bfloat16
- --quantization fp8            (for memory savings)
- --enable-p2p-check false      (skip NVLink peer check on single-GPU)
- --trust-remote-code

CRASH FIXES:
- Reduce --mem-fraction-static (OOM)
- Add --dtype bfloat16 (precision issues)
- Add --trust-remote-code (custom model code)

PERFORMANCE / OPTIMIZATION (use when goal is to maximize tok/s):
- --attention-backend flashinfer (the fast backend on Blackwell)
- --chunked-prefill-size 8192
- --max-running-requests 1024
- --mem-fraction-static 0.88   (on unified memory, higher ≠ faster — pushing past ~0.90 causes
  system memory pressure and paging; only raise it if KV-cache capacity is the actual bottleneck)
- --quantization fp8           (halves memory, increases batch capacity)
- Tuned Triton MoE kernels: if ~/.cache/sparkrun/tuning/sglang/ has configs for this model, set
  SGLANG_MOE_CONFIG_DIR to it — untuned fused-MoE kernels are the #1 cause of slow MoE recipes.
  (Generate once via: sparkrun tune sglang <recipe> -H localhost --tp 1)
"""

_PROMPT_LLAMACPP = """
=== llama.cpp ENGINE RULES ===

llama.cpp runs natively as llama-server. It serves GGUF model files from local disk.
Do NOT apply vLLM flags, SGLang flags, or _registry docker blocks. They are invalid here.

OUTPUT SCHEMA for patched_recipe:
{
  "engine": "llamacpp",
  "model": "/path/to/model.gguf",
  "args": {
    "m": "/path/to/model.gguf",
    "n-gpu-layers": 99,
    "ctx-size": 4096,
    "batch-size": 2048,
    "ubatch-size": 512,
    "flash-attn": "on",
    "host": "0.0.0.0",
    "port": 8080
  },
  "env": {},
  "raw_cmd": null
}

VALID llama-server FLAGS (use these long-form keys in args; the launcher also
translates the short aliases c/b/ub/ngl/np/t/fa/ctk/ctv to them automatically):
- "m": "/path/to/model.gguf"  (GGUF model file path — required)
- "n-gpu-layers": 99          (GPU layers; 99 = all on GPU)
- "ctx-size": 4096            (context size)
- "batch-size": 2048          (logical batch size for prompt processing)
- "ubatch-size": 512          (physical micro-batch size)
- "flash-attn": "on"          (FlashAttention takes a VALUE: on|off|auto — never a bare flag)
- "threads": 8                (CPU threads for non-GPU layers)
- "host": "0.0.0.0", "port": 8080
- "parallel": 1               (parallel slots / concurrent requests)
- "mlock": true               (lock model in RAM — good for 128 GB system)
- "cache-type-k": "q8_0", "cache-type-v": "q8_0"  (KV-cache quantization)

DO NOT use: --gpu-memory-utilization, --model-path, --tp, --mem-fraction-static,
            _registry, container, raw_yaml, vllm serve, sglang serve.
These are from other engines and will break llama.cpp.

CRASH FIXES:
- Reduce -c (context OOM)
- Reduce -ngl (partial GPU offload if full offload OOM)
- Check model file path exists on disk

PERFORMANCE / OPTIMIZATION (use when goal is to maximize tok/s):
- "n-gpu-layers": 99          (all layers on GPU — most important flag)
- "flash-attn": "on"          (significant speedup on Blackwell)
- "cache-type-k": "q8_0", "cache-type-v": "q8_0"  (KV-cache quantization — the single biggest
  llama.cpp lever on GB10; near-lossless, halves KV memory and lifts tok/s)
- "ubatch-size": 2048         (large micro-batch = much faster prompt processing on GB10)
- "no-mmap": true             (mmap measurably HURTS on GB10 unified memory — load weights directly)
- "parallel": 4               (parallel slots for concurrent users)
- "mlock": true               (avoid swap on 128 GB system)
- Larger context ("ctx-size" 8192+) only if needed — hurts tok/s if unused
- GB10 sanity reference: gpt-oss-120B ≈ 35 tok/s gen (~1000 tok/s prefill); Qwen3-Coder-30B Q8 ≈ 44 tok/s.
  If measured tok/s is far below the same class of model, config is the problem — not the hardware.
"""

_PROMPT_FOOTER = """
--- CONTEXT (registry recipes synced locally — ground truth) ---
__CONTEXT__
__PERF__
--- RECIPE (current recipe JSON) ---
__RECIPE__

--- LAST LOGS (most recent 300 lines) ---
__LOGS__

--- USER GOAL ---
__GOAL__
"""

_PERF_BLOCK = """
--- MEASURED PERFORMANCE (live benchmark of THIS recipe on THIS machine) ---
__PERF__

Treat these numbers as ground truth. Your patch must plausibly raise tokens/sec.
Do NOT propose changes that only affect startup (load format, download caching) unless
the measured bottleneck is load time. The engine currently serves successfully —
do not change model, engine, or container unless a strictly faster validated variant
exists in the CONTEXT recipes above.
"""


# Build the full prompt dynamically per engine so the agent never confuses engines.
def _build_prompt(engine: str, context: str, recipe_json: str, logs: str, goal: str,
                  perf: str = "") -> str:
    eng = (engine or "vllm").lower().replace("-", "").replace("_", "")
    if "sglang" in eng:
        engine_section = _PROMPT_SGLANG
        engine_label = "sglang"
    elif "llama" in eng or "llamacpp" in eng or "gguf" in eng:
        engine_section = _PROMPT_LLAMACPP
        engine_label = "llamacpp"
    else:
        engine_section = _PROMPT_VLLM
        engine_label = "vllm"

    return (
        _PROMPT_COMMON_HEADER.replace("__ENGINE__", engine_label)
        + engine_section
        + _PROMPT_FOOTER
        .replace("__PERF__", _PERF_BLOCK.replace("__PERF__", perf) if perf else "")
        .replace("__CONTEXT__", context)
        .replace("__RECIPE__", recipe_json)
        .replace("__LOGS__", logs[-20000:] if logs else "")
        .replace("__GOAL__", goal or f"Make this {engine_label} recipe run successfully on NVIDIA DGX Spark.")
    )

# Keep for backwards compat — not used directly anymore
FIX_PROMPT = _PROMPT_COMMON_HEADER + _PROMPT_VLLM + _PROMPT_FOOTER


def _which(cmd: str) -> str | None:
    found = shutil.which(cmd)
    if found:
        return found
    # npm global prefix may not be on PATH when server starts without sourcing .bashrc
    home = os.path.expanduser("~")
    for extra in [
        os.path.join(home, ".npm-global", "bin", cmd),
        os.path.join(home, ".local", "bin", cmd),
        "/usr/local/bin/" + cmd,
    ]:
        if os.path.isfile(extra) and os.access(extra, os.X_OK):
            return extra
    return None


def claude_available() -> bool:
    return _which("claude") is not None


# Cached probe: codex file may exist but crash (e.g. missing linux-arm64 native binary).
# We run it once with --version to confirm it actually works.
_codex_probe: bool | None = None


def codex_available() -> bool:
    global _codex_probe
    if _codex_probe is not None:
        return _codex_probe
    path = _which("codex")
    if path is None:
        _codex_probe = False
        return False
    try:
        r = _subprocess.run([path, "--version"], capture_output=True, timeout=8)
        _codex_probe = r.returncode == 0
    except Exception:
        _codex_probe = False
    return _codex_probe


async def _spawn(cmd: list[str], stdin: str | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    out, err = await proc.communicate(stdin.encode() if stdin is not None else None)
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


async def ask_claude(prompt: str) -> str:
    if not claude_available():
        raise RuntimeError("`claude` CLI not installed. Run `npm install -g @anthropic-ai/claude-code`.")
    # Pipe via stdin to avoid OS arg-length limits on very long prompts.
    # -p / --print = non-interactive mode; --dangerously-skip-permissions
    # suppresses interactive permission prompts in server context.
    code, out, err = await _spawn(
        ["claude", "-p", "--dangerously-skip-permissions"],
        stdin=prompt,
    )
    if code != 0:
        # Fall back to argv style in case the version doesn't support stdin pipe.
        code2, out2, err2 = await _spawn(["claude", "--print", prompt])
        if code2 != 0:
            raise RuntimeError(f"claude failed ({code2}): {err2.strip() or out2.strip()}")
        return out2.strip()
    return out.strip()


async def ask_codex(prompt: str) -> str:
    codex_path = _which("codex")
    if not codex_available():
        if codex_path is not None:
            raise RuntimeError(
                "Codex is installed but cannot run on this platform — "
                "the native binary @openai/codex-linux-arm64 is missing. "
                "Reinstall via: npm install -g @openai/codex@latest"
            )
        raise RuntimeError("`codex` CLI not installed. Run `npm install -g @openai/codex`.")

    # Feed the prompt via stdin ("-") and read ONLY the agent's final message
    # via --output-last-message. The previous temp-file + full-transcript
    # approach made codex spend turns on file reading and left tool chatter in
    # stdout, which regularly corrupted the JSON extraction — the main reason
    # Codex fixes appeared "worse" than Claude's.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        out_path = f.name
    try:
        code, out, err = await _spawn(
            [
                codex_path, "exec",
                "--skip-git-repo-check",
                "--sandbox", "read-only",   # answer-only task: no command execution needed
                "--ephemeral",
                "--output-last-message", out_path,
                "-",
            ],
            stdin=prompt + "\n\nAnswer directly — do not run commands or read files. "
                           "Output ONLY the raw JSON object requested, no markdown fences.",
        )
        last = ""
        try:
            with open(out_path, encoding="utf-8") as fh:
                last = fh.read().strip()
        except OSError:
            pass
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass

    if code != 0 and not last:
        combined = err.strip() or out.strip()
        if "linux-arm64" in combined or "Missing optional dependency" in combined:
            raise RuntimeError(
                "Codex native binary missing for linux/arm64 — "
                "reinstall via: npm install -g @openai/codex@latest"
            )
        raise RuntimeError(f"codex failed ({code}): {combined[:800]}")
    return last or out.strip()


async def login_status() -> dict:
    """Detect whether each CLI has cached credentials (best effort)."""
    claude_home = os.path.expanduser("~/.claude")
    codex_home = os.path.expanduser("~/.codex")
    return {
        "claude": {
            "installed": claude_available(),
            "logged_in": os.path.exists(os.path.join(claude_home, ".credentials.json"))
                       or os.path.exists(os.path.join(claude_home, "session.json")),
        },
        "codex": {
            "installed": codex_available(),
            "logged_in": os.path.exists(os.path.join(codex_home, "auth.json")),
        },
    }


async def login_stream(which: str) -> AsyncIterator[str]:
    """Stream the login flow for `claude` or `codex`. Yields each stdout line.

    Codex supports a one-shot `codex login` that prints a URL for the user to
    visit. Claude Code's subscription OAuth is only available inside the
    interactive REPL (via `/login`), so for Claude we surface instructions
    instead of trying to pipe the REPL.
    """
    if which == "claude":
        yield "Claude Code OAuth is handled by the CLI itself — open any terminal and run:"
        yield ""
        yield "    claude"
        yield ""
        yield "At the prompt, type `/login` and follow the browser flow."
        yield "After you sign in, close the terminal and click Refresh here."
        return
    if which == "codex":
        cmd = ["codex", "login"]
    else:
        raise ValueError(f"unknown agent: {which}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=os.environ.copy(),
    )
    assert proc.stdout is not None
    async for raw in proc.stdout:
        yield raw.decode("utf-8", "replace").rstrip("\n")
    await proc.wait()
    yield f"[exit] code={proc.returncode}"


def _recipe_repo(recipe: dict) -> str | None:
    """Pull the HF repo id out of whatever shape the recipe is in."""
    args = recipe.get("args") or {}
    return (
        recipe.get("model")
        or args.get("model")
        or args.get("model-path")
        or (args.get("_registry") or {}).get("origin", {}).get("path")
    )


def _local_docker_images() -> list[str]:
    """Return image names that are actually present on this machine."""
    try:
        out = _subprocess.check_output(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            text=True, timeout=6,
        )
        return [l.strip() for l in out.splitlines() if l.strip() and "<none>" not in l]
    except Exception:
        return []


def _recipe_schema_doc(max_chars: int = 18000) -> str:
    """The sparkrun RECIPES.md schema reference from the local registry
    mirror (synced on every app start) — authoritative field reference for
    recipe YAML, so agents patch against the real schema instead of guessing."""
    try:
        p = registry.REGISTRY_ROOT / "sparkrun" / "RECIPES.md"
        text = p.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars]
    except OSError:
        return ""


def _build_context(recipe: dict, logs: str) -> str:
    """Inline the matching registry recipe + relevant mod contents + image
    pins so the agent has real ground truth instead of URLs to fetch."""
    repo = _recipe_repo(recipe)
    blocks: list[str] = []

    schema = _recipe_schema_doc()
    if schema:
        blocks.append("=== Recipe YAML schema reference (sparkrun/RECIPES.md, local mirror) ===\n" + schema)

    # Always include locally-built docker images so the agent never hallucinates
    # container names — it must only use images that actually exist on this machine.
    local_images = _local_docker_images()
    vllm_images = [img for img in local_images if any(
        kw in img for kw in ("vllm", "sglang", "spark", "llm")
    )]
    if vllm_images:
        blocks.append(
            "Docker images available locally on this machine (ONLY use these — do NOT invent other names):\n"
            + "\n".join(f"  - {img}" for img in vllm_images)
        )
    else:
        blocks.append(
            "No vLLM/SGLang docker images found locally.\n"
            "Valid container names for spark-vllm-docker: vllm-node, vllm-node-mxfp4, vllm-node-tf5\n"
            "(these are built locally — not Docker Hub images; run build-and-copy.sh to build them)"
        )

    pins = registry.image_pins()
    if pins:
        blocks.append("Pinned image tags from sparkrun/versions.yaml:\n" + json.dumps(pins, indent=2))

    matches: list[registry.Recipe] = []
    if repo:
        matches = registry.by_exact_repo(repo)
        if not matches:
            matches = registry.by_similar(repo, k=2)
    if matches:
        for r in matches:
            blocks.append(
                f"=== Curated recipe ({r.source_repo}/{r.source_path}) ===\n{r.raw_yaml.strip()}"
            )

    log_tail = logs[-8000:] if logs else ""
    mods = registry.relevant_mods(repo, log_tail, k=3)
    for m in mods:
        body_chunks = []
        for fname, content in m.files.items():
            body_chunks.append(f"--- {fname} ---\n{content.strip()}")
        blocks.append(
            f"=== Relevant mod ({m.source_repo}/{m.source_path}) ===\n" + "\n".join(body_chunks)
        )

    if not blocks:
        return "(no curated recipe or mod matches this repo — fall back to general DGX Spark knowledge)"
    return "\n\n".join(blocks)


def diff_against_registry(recipe: dict) -> dict:
    """Agent-free deterministic check. Returns the same shape as fix_recipe.
    Useful when no agent CLI is logged in, or as a quick cross-check."""
    repo = _recipe_repo(recipe)
    if not repo:
        return {
            "diagnosis": "No model id on the recipe — cannot match a curated recipe.",
            "patched_recipe": recipe,
            "diff_notes": [],
        }
    matches = registry.by_exact_repo(repo) or registry.by_similar(repo, k=1)
    if not matches:
        return {
            "diagnosis": (
                f"No curated recipe found for {repo}. "
                "Sync the registry or run with the heuristic recipe."
            ),
            "patched_recipe": recipe,
            "diff_notes": [],
        }
    canonical = matches[0]
    notes = [
        f"Found curated recipe: {canonical.source_repo}/{canonical.source_path}",
        f"Container image: {canonical.container or '(none)'}",
    ]
    if canonical.mods:
        notes.append("Required mods: " + ", ".join(canonical.mods))
    notes.append("Adopt this recipe verbatim and run via run-recipe.sh for best results.")
    patched = {
        "engine": canonical.engine,
        "model": repo,
        "args": {
            "model": repo,
            "_registry": {
                "container": canonical.container,
                "mods": canonical.mods,
                "command": canonical.command,
                "defaults": canonical.defaults,
                "raw_yaml": canonical.raw_yaml,
                "origin": {"repo": canonical.source_repo, "path": canonical.source_path},
                "adapted_from_model": canonical.model if canonical.model != repo else None,
            },
        },
        "env": canonical.env,
    }
    return {
        "diagnosis": (
            f"A curated recipe exists for {repo} ({canonical.source_repo}). "
            "Replacing the current args with the curated recipe."
        ),
        "patched_recipe": patched,
        "diff_notes": notes,
    }


def _extract_fix_json(raw: str) -> dict | None:
    """Find the fix-result JSON object in agent output. Scans balanced {...}
    candidates and prefers ones that actually carry a patched_recipe — much
    more robust than first-'{' … last-'}' when transcripts leak into stdout."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    candidates: list[dict] = []
    i = 0
    while i < len(raw):
        start = raw.find("{", i)
        if start < 0:
            break
        depth = 0
        in_str = False
        esc = False
        for j in range(start, len(raw)):
            ch = raw[j]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == '"' and not esc:
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(raw[start : j + 1])
                            if isinstance(obj, dict):
                                candidates.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = start + 1
                        break
        else:
            break
        if depth != 0:
            i = start + 1
    for obj in reversed(candidates):
        if "patched_recipe" in obj:
            return obj
    for obj in reversed(candidates):
        if "diagnosis" in obj:
            return obj
    return candidates[-1] if candidates else None


_AGENT_TIMEOUT = int(os.environ.get("SPARK_STUDIO_AGENT_TIMEOUT", "420"))


async def fix_recipe(agent: str, recipe: dict, logs: str, goal: str = "", perf: str = "") -> dict:
    # When neither agent CLI is available, use the deterministic registry diff.
    want = agent or ""
    if want == "claude" and not claude_available():
        want = "auto"
    if want == "codex" and not codex_available():
        want = "auto"
    if want == "auto" or (want not in ("claude", "codex")):
        if claude_available():
            want = "claude"
        elif codex_available():
            want = "codex"
        else:
            return diff_against_registry(recipe)

    engine = recipe.get("engine") or "vllm"
    context = _build_context(recipe, logs or "")
    prompt = _build_prompt(
        engine=engine,
        context=context,
        recipe_json=json.dumps(recipe, indent=2),
        logs=logs or "",
        goal=goal,
        perf=perf,
    )
    try:
        raw = await asyncio.wait_for(
            ask_claude(prompt) if want == "claude" else ask_codex(prompt),
            timeout=_AGENT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return {
            "diagnosis": f"{want} did not answer within {_AGENT_TIMEOUT}s (SPARK_STUDIO_AGENT_TIMEOUT).",
            "patched_recipe": recipe,
            "diff_notes": ["agent timed out — recipe unchanged"],
        }
    result = _extract_fix_json(raw)
    if result is None:
        return {"diagnosis": raw.strip()[:4000], "patched_recipe": recipe, "diff_notes": ["agent returned non-JSON output"]}
    result.setdefault("patched_recipe", recipe)
    result.setdefault("diff_notes", [])

    # Validate container name — catch hallucinated Docker Hub paths before they cause a failed pull.
    _VALID_CONTAINERS = {"vllm-node", "vllm-node-mxfp4", "vllm-node-tf5"}
    patched = result.get("patched_recipe") or {}
    reg = (patched.get("args") or {}).get("_registry") or {}
    container = reg.get("container", "")
    local_images = _local_docker_images()
    local_names = {img.split(":")[0] for img in local_images}
    if container and container not in _VALID_CONTAINERS and container not in local_names:
        # Replace with the nearest valid image that is actually built locally.
        fallback = next(
            (c for c in ("vllm-node", "vllm-node-tf5", "vllm-node-mxfp4") if c in local_names),
            "vllm-node",
        )
        reg["container"] = fallback
        notes = result.get("diff_notes") or []
        notes.append(
            f"Container '{container}' not found locally — replaced with '{fallback}'. "
            "Build missing images with build-and-copy.sh if needed."
        )
        result["diff_notes"] = notes

    return result
