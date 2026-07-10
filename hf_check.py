"""HuggingFace model compatibility checker for DGX Spark.

DGX Spark = Grace-Blackwell superchip, 128 GB unified memory. We estimate
memory footprint from config.json and quantization, then flag whether a
model can run on vLLM / SGLang / llama.cpp under that envelope.
"""

from __future__ import annotations

from typing import Any

import httpx

HF_API = "https://huggingface.co/api/models"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{file}"

# ~115 GB usable for model weights + KV cache, leaving headroom.
SPARK_USABLE_GB = 115.0


async def fetch_config(repo: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{HF_API}/{repo}")
        r.raise_for_status()
        meta = r.json()
        cfg = {}
        try:
            r = await client.get(HF_FILE.format(repo=repo, file="config.json"))
            if r.status_code == 200:
                cfg = r.json()
        except Exception:
            pass
        return {"meta": meta, "config": cfg}


def _bytes_per_param(dtype: str) -> float:
    d = (dtype or "").lower()
    if "int4" in d or "q4" in d or "awq" in d or "gptq" in d:
        return 0.5
    if "int8" in d or "fp8" in d or "q8" in d:
        return 1.0
    if "bf16" in d or "fp16" in d or "half" in d:
        return 2.0
    if "fp32" in d or "float32" in d:
        return 4.0
    return 2.0


def _guess_params(meta: dict, config: dict) -> float | None:
    # 1. Siblings / safetensors metadata
    for s in meta.get("safetensors", {}).get("parameters", {}).values() if isinstance(meta.get("safetensors"), dict) else []:
        if isinstance(s, (int, float)):
            return float(s)
    st = meta.get("safetensors") or {}
    if isinstance(st, dict):
        total = st.get("total")
        if isinstance(total, (int, float)):
            return float(total)
    # 2. Parse from name (e.g. "-70B-", "-7b-instruct")
    name = (meta.get("modelId") or meta.get("id") or "") + " " + str(config.get("_name_or_path", ""))
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\b", name)
    if m:
        return float(m.group(1)) * 1e9
    # 3. Estimate from architecture
    hs = config.get("hidden_size") or config.get("n_embd")
    nl = config.get("num_hidden_layers") or config.get("n_layer")
    vocab = config.get("vocab_size")
    if hs and nl:
        return 12 * nl * hs * hs + (vocab or 0) * hs
    return None


def analyze(meta: dict, config: dict) -> dict[str, Any]:
    params = _guess_params(meta, config)
    # NB: parenthesized so a missing tags list can't nullify a real torch_dtype
    # (a or b if cond else None parses as (a or b) if cond else None).
    dtype = config.get("torch_dtype") or (meta["tags"][0] if meta.get("tags") else None)
    quant_cfg = config.get("quantization_config") or {}
    quant_method = quant_cfg.get("quant_method") or quant_cfg.get("bits")
    if quant_method:
        dtype = f"{quant_method}"
    bpp = _bytes_per_param(dtype or "bf16")
    weight_gb = (params * bpp / 1e9) if params else None

    ctx = (
        config.get("max_position_embeddings")
        or config.get("max_seq_len")
        or config.get("seq_length")
        or 4096
    )
    # KV cache per 1k tokens (rough, fp16): 2 * num_layers * num_heads_kv * head_dim * 2 bytes * 1000
    nl = config.get("num_hidden_layers") or 32
    nh = config.get("num_key_value_heads") or config.get("num_attention_heads") or 32
    hd = (config.get("hidden_size") or 4096) // (config.get("num_attention_heads") or 32)
    kv_per_1k = 2 * nl * nh * hd * 2 * 1000 / 1e9  # GB per 1000 tokens

    verdict = "unknown"
    reasons: list[str] = []
    if weight_gb:
        if weight_gb < SPARK_USABLE_GB * 0.4:
            verdict = "excellent"
            reasons.append(f"weights {weight_gb:.1f} GB fit with plenty of room for KV cache")
        elif weight_gb < SPARK_USABLE_GB * 0.7:
            verdict = "good"
            reasons.append(f"weights {weight_gb:.1f} GB fit; long contexts may need reduction")
        elif weight_gb < SPARK_USABLE_GB:
            verdict = "tight"
            reasons.append(f"weights {weight_gb:.1f} GB leave little KV headroom; consider quantization")
        else:
            verdict = "too-large"
            reasons.append(f"weights {weight_gb:.1f} GB exceed the ~{SPARK_USABLE_GB:.0f} GB envelope")

    arch = (config.get("architectures") or [None])[0]
    tags = meta.get("tags", [])
    if "gguf" in tags:
        reasons.append("llama.cpp compatible (GGUF)")

    return {
        "repo": meta.get("modelId") or meta.get("id"),
        "params": params,
        "params_human": _human_params(params) if params else None,
        "dtype": dtype,
        "bytes_per_param": bpp,
        "weight_gb": weight_gb,
        "context": ctx,
        "kv_gb_per_1k": kv_per_1k,
        "max_tokens_at_full_ctx": _estimate_max_tokens(weight_gb, kv_per_1k),
        "architecture": arch,
        "tags": tags,
        "verdict": verdict,
        "reasons": reasons,
        "suggested_engines": _suggest_engines(tags, arch, config),
    }


def _human_params(p: float) -> str:
    if p >= 1e12:
        return f"{p/1e12:.1f}T"
    if p >= 1e9:
        return f"{p/1e9:.1f}B"
    if p >= 1e6:
        return f"{p/1e6:.0f}M"
    return f"{p:.0f}"


def _estimate_max_tokens(weight_gb: float | None, kv_per_1k: float) -> int | None:
    if not weight_gb:
        return None
    remaining = SPARK_USABLE_GB - weight_gb
    if remaining <= 0:
        return 0
    return int((remaining / kv_per_1k) * 1000)


def _suggest_engines(tags: list[str], arch: str | None, config: dict) -> list[str]:
    engines: list[str] = []
    if "gguf" in tags:
        engines.append("llamacpp")
    if arch:
        engines.extend(["vllm", "sglang"])
    if not engines:
        engines = ["vllm"]
    return engines


async def check(repo: str) -> dict[str, Any]:
    data = await fetch_config(repo)
    return analyze(data["meta"], data["config"])
