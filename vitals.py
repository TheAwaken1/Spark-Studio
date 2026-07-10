"""Live DGX Spark hardware telemetry.

Streams GPU + unified-memory stats every 2 s via an async generator.

The GB10 SoC exposes its 128 GB unified memory as ordinary system RAM;
nvidia-smi covers GPU utilisation, temperature, power, and clock.
nvidia-smi mem fields return [N/A] on the GB10, so psutil is used instead.
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from typing import AsyncGenerator

import psutil

_SMI_FIELDS = "utilization.gpu,temperature.gpu,power.draw,clocks.current.graphics"


def _read_smi() -> dict:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={_SMI_FIELDS}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=4,
        ).strip()
        parts = [p.strip() for p in raw.split(",")]

        def _f(s: str) -> float | None:
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        def _pick(idx: int) -> float | None:
            v = parts[idx] if idx < len(parts) else None
            return None if v in (None, "[N/A]", "N/A", "") else _f(v)

        return {
            "gpu_util":  _pick(0),
            "gpu_temp":  _pick(1),
            "gpu_power": _pick(2),
            "gpu_clock": _pick(3),
        }
    except Exception:
        return {"gpu_util": None, "gpu_temp": None, "gpu_power": None, "gpu_clock": None}


def _read_system() -> dict:
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    return {
        "mem_used_gb":  round(mem.used  / 1024 ** 3, 1),
        "mem_total_gb": round(mem.total / 1024 ** 3, 1),
        "mem_pct":      round(mem.percent, 1),
        "cpu_pct":      round(cpu, 1),
    }


def snapshot() -> dict:
    """One-shot telemetry sample (same fields as the stream) — used to give
    the optimizer agent the machine's live state alongside benchmark numbers."""
    return {**_read_smi(), **_read_system(), "ts": time.time()}


async def stream_vitals(interval: float = 2.0) -> AsyncGenerator[dict, None]:
    """Yield a telemetry dict every *interval* seconds (default 2 s)."""
    loop = asyncio.get_event_loop()
    # Prime the cpu_percent baseline (first call always returns 0.0)
    psutil.cpu_percent(interval=None)
    while True:
        smi = await loop.run_in_executor(None, _read_smi)
        sys_info = await loop.run_in_executor(None, _read_system)
        yield {**smi, **sys_info, "ts": time.time()}
        await asyncio.sleep(interval)
