"""Import history that already exists on this machine into the usage tables.

Runs on startup and is safe to repeat. Only local files are read, and nothing
is sent anywhere. Each source is optional: a machine without Hermes or without
earlier engine runs simply imports nothing from it.

Sources:
- Hermes agent (``~/.hermes/state.db``, or ``$HERMES_HOME``): per-session,
  per-model totals. Hermes keeps no per-request timing, so these rows show
  tokens and cache only.
- Engine run logs (``data/run-logs/*.log``): vLLM's 10-second throughput lines
  give historical tok/s, running requests and prefix-cache hit rate.

Hermes history is imported only up to the first request the live recorder
saw, so the same request is never counted twice.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import db
import usage

_THROUGHPUT = re.compile(
    r"INFO (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) \[loggers\.py:\d+\] Engine \d+: "
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs"
    r"(?:, GPU KV cache usage: [\d.]+%)?"
    r"(?:, Prefix cache hit rate: ([\d.]+)%)?"
)


def hermes_state_path() -> Path:
    home = os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")
    return Path(home) / "state.db"


def _hermes_rows(path: Path) -> list[sqlite3.Row]:
    try:
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    src.row_factory = sqlite3.Row
    try:
        return src.execute(
            "SELECT session_id, model, billing_base_url, task, api_call_count, "
            "input_tokens, output_tokens, cache_read_tokens, reasoning_tokens, "
            "first_seen, last_seen FROM session_model_usage"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        src.close()


def hermes_rows_to_events(rows: list[sqlite3.Row], node: str, cutoff: float | None) -> list[dict[str, Any]]:
    """Map Hermes per-session totals to usage rows. ``cutoff`` is the start of the
    earliest live-recorded request; anything last seen after it is skipped."""
    events = []
    for r in rows:
        last_seen = r["last_seen"] or 0.0
        if cutoff is not None and last_seen >= cutoff:
            continue
        calls = r["api_call_count"] or 0
        new_in = r["input_tokens"] or 0
        out_tokens = r["output_tokens"] or 0
        cached = r["cache_read_tokens"] or 0
        if not (calls or new_in or out_tokens or cached):
            continue
        # Hermes reports uncached input separately from cache reads, so the
        # full prompt is the sum of both.
        prompt = new_in + cached
        key = "|".join(str(r[k] or "") for k in ("session_id", "model", "billing_base_url", "task"))
        task = r["task"] or "chat"
        events.append({
            "id": "hermes-" + hashlib.sha1(key.encode()).hexdigest()[:16],
            "started_at": float(r["first_seen"] or last_seen or 0.0),
            "ended_at": float(last_seen or r["first_seen"] or 0.0),
            "node": node,
            "client_ip": "",
            "engine_run_id": "backfill:hermes",
            "model": r["model"] or "",
            "endpoint": f"imported · hermes {task} · {calls} calls",
            "stream": 0,
            "status": "ok",
            "error": None,
            "prompt_tokens": prompt or None,
            "completion_tokens": out_tokens,
            "cached_tokens": cached if prompt else None,
            "cache_hit_pct": round(100.0 * cached / prompt, 1) if prompt else None,
            "thinking": 1 if (r["reasoning_tokens"] or 0) > 0 else 0,
            "concurrency": 1,
            "ttft_ms": None,
            "decode_secs": None,
            "tok_s": None,
            "peak_tok_s": None,
            "samples": None,
        })
    return events


def import_hermes(path: Path | None = None, node: str | None = None) -> int:
    path = path or hermes_state_path()
    if not path.is_file():
        return 0
    rows = _hermes_rows(path)
    if not rows:
        return 0
    cutoff = db.usage_first_live_start()
    events = hermes_rows_to_events(rows, node or usage.node_name("127.0.0.1"), cutoff)
    for event in events:
        db.usage_insert(event)
    return len(events)


def _year_for(month: int, day: int, hour: int, minute: int, second: int, mtime: float) -> float | None:
    """Log lines carry no year. Use the file's modification year, stepping back
    one year when the stamp would fall after the file was last written."""
    year = datetime.fromtimestamp(mtime).year
    for candidate in (year, year - 1):
        try:
            ts = datetime(candidate, month, day, hour, minute, second).timestamp()
        except ValueError:
            return None
        if ts <= mtime + 86400:
            return ts
    return None


def parse_throughput_lines(lines, mtime: float, stem: str) -> list[dict[str, Any]]:
    rows = []
    for line in lines:
        m = _THROUGHPUT.search(line)
        if not m:
            continue
        month, day, hour, minute, second = (int(m.group(i)) for i in range(1, 6))
        ts = _year_for(month, day, hour, minute, second, mtime)
        if ts is None:
            continue
        prompt_tps, gen_tps, running, waiting, hit = m.group(6, 7, 8, 9, 10)
        rows.append({
            "id": f"log-{stem}-{int(ts)}",
            "ts": ts,
            "engine_run_id": f"backfill:{stem}",
            "source": "log",
            "running": int(running),
            "waiting": int(waiting),
            "gen_tps": float(gen_tps),
            "prompt_tps": float(prompt_tps),
            "cache_hit_pct": float(hit) if hit is not None else None,
        })
    return rows


def import_run_logs(log_dir: Path | None = None) -> int:
    if log_dir is None:
        import runners
        log_dir = runners.RUN_LOG_DIR
    if not log_dir.is_dir():
        return 0
    total = 0
    for path in sorted(log_dir.glob("*.log")):
        try:
            mtime = path.stat().st_mtime
            with path.open("r", errors="replace") as fh:
                rows = parse_throughput_lines(fh, mtime, path.stem)
        except OSError:
            continue
        if rows:
            total += db.engine_sample_insert_many(rows)
    return total


def run_all() -> dict[str, int]:
    """Import every local source. Errors in one source don't stop the others."""
    counts = {"hermes_sessions": 0, "engine_samples": 0, "failed_pruned": 0}
    # Aborted and failed requests are kept for a while, then removed.
    keep_days = float(os.environ.get("SPARK_STUDIO_USAGE_FAILED_KEEP_DAYS", "30"))
    try:
        counts["failed_pruned"] = db.usage_prune_failed(time.time() - keep_days * 86400)
    except Exception as exc:  # noqa: BLE001
        print(f"[usage] prune skipped: {exc}", file=sys.stderr, flush=True)
    try:
        counts["hermes_sessions"] = import_hermes()
    except Exception as exc:  # noqa: BLE001
        print(f"[usage] hermes import skipped: {exc}", file=sys.stderr, flush=True)
    try:
        counts["engine_samples"] = import_run_logs()
    except Exception as exc:  # noqa: BLE001
        print(f"[usage] run-log import skipped: {exc}", file=sys.stderr, flush=True)
    if any(counts[k] for k in ("hermes_sessions", "engine_samples")):
        print(f"[usage] imported history: {counts}", flush=True)
    return counts
