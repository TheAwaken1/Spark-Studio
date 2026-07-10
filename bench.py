"""Simple benchmark client against an OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx


DEFAULT_PROMPT = "Explain what NVIDIA DGX Spark is in 200 words."


async def benchmark(
    url: str,
    model: str = "local",
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 256,
    runs: int = 3,
) -> dict[str, Any]:
    base = url.rstrip("/").replace("://0.0.0.0", "://127.0.0.1")
    endpoint = f"{base}/v1/chat/completions"
    ttfts: list[float] = []
    tps: list[float] = []
    all_tokens = 0
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=300) as client:
        for i in range(runs):
            try:
                t0 = time.time()
                first = None
                completion_tokens = 0
                usage_tokens: int | None = None
                async with client.stream(
                    "POST",
                    endpoint,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "stream": True,
                        # Engines that support it report exact token counts on the
                        # final chunk; others silently ignore the option.
                        "stream_options": {"include_usage": True},
                    },
                ) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        # Count only content-bearing chunks: the role preamble and
                        # finish chunks aren't tokens, and TTFT should stamp on the
                        # first real token — not on protocol noise.
                        try:
                            obj = json.loads(payload)
                            u = obj.get("usage") or {}
                            if isinstance(u.get("completion_tokens"), int):
                                usage_tokens = u["completion_tokens"]
                            delta = ((obj.get("choices") or [{}])[0].get("delta")) or {}
                            # Reasoning models stream thinking tokens under
                            # `reasoning` (newer vLLM) or `reasoning_content`
                            # (SGLang/older vLLM) — those are generated tokens too.
                            has_content = any(
                                bool(delta.get(k))
                                for k in ("content", "tool_calls", "reasoning", "reasoning_content")
                            )
                        except (json.JSONDecodeError, AttributeError, IndexError):
                            has_content = True  # unknown shape — keep the old behavior
                        if not has_content:
                            continue
                        if first is None:
                            first = time.time()
                        completion_tokens += 1  # approximate: one content chunk ~= one token
                if usage_tokens is not None:
                    completion_tokens = usage_tokens  # exact count from the engine
                t1 = time.time()
                if first is None:
                    # No token-bearing chunk was recognized: fall back to the
                    # whole wall time so tok/s can't divide by ~zero.
                    first = t0
                ttfts.append((first - t0) * 1000.0)
                duration = max(t1 - first, 1e-6)
                tps.append(completion_tokens / duration)
                all_tokens += completion_tokens
            except Exception as e:  # noqa: BLE001
                errors.append(str(e))

    def avg(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {
        "endpoint": endpoint,
        "model": model,
        "runs": runs,
        "tokens_per_sec": avg(tps),
        "ttft_ms": avg(ttfts),
        "completion_tokens": all_tokens,
        "errors": errors,
    }
