"""Simple benchmark client against an OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx


DEFAULT_PROMPT = "Explain what NVIDIA DGX Spark is in 200 words."

# Default ladder for the concurrency sweep: single stream (what one chat
# feels like), a small agent team, and a busy multi-agent box. Community
# "N tok/s serving" claims are usually the aggregate at levels like these,
# so measuring them side by side keeps expectations honest.
DEFAULT_CONCURRENCY = (1, 4, 8)


async def _stream_once(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> tuple[float, int, float]:
    """One generation; returns (ttft_ms, completion_tokens, decode_seconds)."""
    t0 = time.time()
    first: float | None = None
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
            # Count only content-bearing chunks: the role preamble and finish
            # chunks aren't tokens, and TTFT should stamp on the first real
            # token — not on protocol noise.
            try:
                obj = json.loads(payload)
                u = obj.get("usage") or {}
                if isinstance(u.get("completion_tokens"), int):
                    usage_tokens = u["completion_tokens"]
                delta = ((obj.get("choices") or [{}])[0].get("delta")) or {}
                # Reasoning models stream thinking tokens under `reasoning`
                # (newer vLLM) or `reasoning_content` (SGLang/older vLLM) —
                # those are generated tokens too.
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
        # No token-bearing chunk was recognized: fall back to the whole wall
        # time so tok/s can't divide by ~zero.
        first = t0
    return (first - t0) * 1000.0, completion_tokens, max(t1 - first, 1e-6)


async def concurrency_sweep(
    url: str,
    model: str = "local",
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 256,
    streams: tuple[int, ...] | list[int] = DEFAULT_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Aggregate vs per-stream throughput at increasing concurrency.

    Aggregate tok/s divides the batch's total tokens by the wall time of the
    whole batch (first request start → last request end) — the number
    "multi-agent serving" claims quote. Per-stream is that divided by the
    stream count: what each individual chat experiences at that load.
    """
    base = url.rstrip("/").replace("://0.0.0.0", "://127.0.0.1")
    endpoint = f"{base}/v1/chat/completions"
    levels: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=600) as client:
        for n in [max(1, int(s)) for s in streams]:
            t0 = time.time()
            outcomes = await asyncio.gather(
                *(_stream_once(client, endpoint, model, prompt, max_tokens) for _ in range(n)),
                return_exceptions=True,
            )
            wall = max(time.time() - t0, 1e-6)
            ok = [o for o in outcomes if not isinstance(o, BaseException)]
            errors = [str(o) for o in outcomes if isinstance(o, BaseException)]
            tokens = sum(t for _, t, _ in ok)
            aggregate = tokens / wall
            levels.append({
                "streams": n,
                "completed": len(ok),
                "aggregate_tokens_per_sec": aggregate,
                "per_stream_tokens_per_sec": aggregate / n if n else 0.0,
                "ttft_ms": (sum(t for t, _, _ in ok) / len(ok)) if ok else None,
                "completion_tokens": tokens,
                "wall_seconds": wall,
                "errors": errors,
            })
    return levels


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
        for _ in range(runs):
            try:
                ttft_ms, completion_tokens, decode_s = await _stream_once(
                    client, endpoint, model, prompt, max_tokens
                )
                ttfts.append(ttft_ms)
                tps.append(completion_tokens / decode_s)
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
