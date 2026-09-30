"""Wrapper around eugr/llama-benchy for OpenAI-compatible endpoint benchmarking.

Spawns the `llama-benchy` CLI as a subprocess, streams its stdout line-by-line
to a callback, and returns the parsed JSON result file.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable

VENV_BIN = Path(sys.executable).parent


def _resolve_binary() -> str | None:
    local = VENV_BIN / "llama-benchy"
    if local.exists() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which("llama-benchy")


def available() -> bool:
    return _resolve_binary() is not None


def _has_extra_body_key(args: list[str], key: str) -> bool:
    """Return whether caller-supplied llama-benchy args set an extra-body key."""
    for i, arg in enumerate(args):
        if arg != "--extra-body" or i + 1 >= len(args):
            continue
        value = args[i + 1]
        for entry in value.split(","):
            name = entry.split("=", 1)[0].split(":", 1)[0].strip()
            if name == key:
                return True
    return False


async def run(
    *,
    base_url: str,
    model: str,
    tokenizer: str | None = None,
    served_model_name: str | None = None,
    pp: list[int] | None = None,
    tg: list[int] | None = None,
    depth: list[int] | None = None,
    runs: int = 3,
    concurrency: list[int] | None = None,
    latency_mode: str = "generation",
    enable_prefix_caching: bool = False,
    skip_coherence: bool = False,
    no_cache: bool = False,
    extra_args: list[str] | None = None,
    env_extra: dict[str, str] | None = None,
    on_log: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Run llama-benchy. Yields each stdout line via on_log; returns parsed result."""
    binary = _resolve_binary()
    if not binary:
        raise RuntimeError(
            "llama-benchy not installed. `uv pip install --python env/bin/python llama-benchy`"
        )
    out_dir = Path(tempfile.gettempdir())
    out_path = out_dir / f"benchy-{os.urandom(6).hex()}.json"
    cmd = [
        binary,
        "--base-url", base_url,
        "--model", model,
        "--format", "json",
        "--save-result", str(out_path),
        "--runs", str(runs),
        "--latency-mode", latency_mode,
    ]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    for k, vs in (("--pp", pp), ("--tg", tg), ("--depth", depth), ("--concurrency", concurrency)):
        if vs:
            cmd += [k, *map(str, vs)]
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if skip_coherence:
        cmd.append("--skip-coherence")
    if no_cache:
        cmd.append("--no-cache")
    # llama-benchy 0.4.0 requests token IDs in every streamed completion.
    # SGLang rejects that combination (return_token_ids + stream), while its
    # streamed usage event provides the authoritative completion token count.
    # Disable token IDs unless a caller explicitly overrides this compatibility
    # default; the client already has a usage-count fallback.
    supplied_args = extra_args or []
    if not _has_extra_body_key(supplied_args, "return_token_ids"):
        cmd += ["--extra-body", "return_token_ids=false"]
    if supplied_args:
        cmd += supplied_args

    if on_log:
        await on_log(f"$ {' '.join(cmd)}")

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if on_log:
            await on_log(line)
    await proc.wait()

    result: dict[str, Any] | None = None
    if out_path.exists():
        try:
            result = json.loads(out_path.read_text())
        except Exception as e:  # noqa: BLE001
            if on_log:
                await on_log(f"[parse error] {e}")
        try:
            out_path.unlink()
        except OSError:
            pass

    return {"exit_code": proc.returncode or 0, "result": result, "cmd": cmd}
