"""Token usage tracking for traffic that passes through Spark Studio's proxies.

Every OpenAI-compatible request routed through ``/api/engine/v1`` (Hermes, other
machines on the tailnet, scripts) and the dashboard chat is recorded here: which
machine sent it, which model, prompt and completion tokens, cache hits, whether
the model was thinking, tokens per second, and how many streams were running at
the time. In-flight requests live in memory for the live view; finished requests
are written to the ``usage_events`` table.

Only traffic that goes through Spark Studio is seen. Requests that hit an engine
port directly, or frontier APIs, are not.
"""

from __future__ import annotations

import bisect
import json
import os
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from typing import Any

import db

# Per-second buckets kept per request, and token timestamps kept for peak
# calculation. Long generations are truncated rather than growing without bound.
MAX_SAMPLES = 1200
MAX_TOKEN_TIMES = 20000
LIVE_WINDOW_SECS = 5.0
# Streamed text kept per live request so its token count can be refreshed.
MAX_LIVE_TEXT = 300_000

SESSION_START = time.time()

_lock = threading.Lock()
_live: dict[str, "UsageRecorder"] = {}
_session_peak_streams = 0
_session_peak_gen = 0.0


def parse_node_names(raw: str) -> dict[str, str]:
    """``"198.51.100.3=node-a, 198.51.100.4=node-b"`` -> {ip: name}. Bad entries are skipped."""
    names: dict[str, str] = {}
    for part in (raw or "").split(","):
        ip, sep, name = part.partition("=")
        ip, name = ip.strip(), name.strip()
        if sep and ip and name:
            names[ip] = name
    return names


NODE_NAMES = parse_node_names(os.environ.get("SPARK_STUDIO_NODE_NAMES", ""))


# Tailscale hostnames by address, refreshed in the background so a request never
# waits on the tailscale CLI. Empty when tailscale isn't installed or running.
_ts_names: dict[str, str] = {}
_ts_self_ips: set[str] = set()
_ts_started = False
_ts_lock = threading.Lock()


def parse_tailscale_status(data: dict[str, Any]) -> dict[str, str]:
    """{address: hostname} for this machine and its peers, from `tailscale status --json`."""
    nodes: list[dict[str, Any]] = []
    if isinstance(data.get("Self"), dict):
        nodes.append(data["Self"])
    peers = data.get("Peer")
    if isinstance(peers, dict):
        nodes.extend(p for p in peers.values() if isinstance(p, dict))
    names: dict[str, str] = {}
    for node in nodes:
        host = node.get("HostName") or str(node.get("DNSName") or "").split(".")[0]
        for ip in node.get("TailscaleIPs") or []:
            if isinstance(ip, str) and host:
                names[ip] = host
    return names


def _refresh_tailscale_names() -> None:
    try:
        out = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            data = json.loads(out.stdout)
            names = parse_tailscale_status(data)
            _ts_names.clear()
            _ts_names.update(names)
            self_node = data.get("Self") if isinstance(data.get("Self"), dict) else {}
            _ts_self_ips.clear()
            _ts_self_ips.update(str(ip) for ip in self_node.get("TailscaleIPs") or [])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass


def start_background() -> None:
    """Begin refreshing tailscale names. Called once from server startup."""
    global _ts_started
    with _ts_lock:
        if _ts_started:
            return
        _ts_started = True

    def loop() -> None:
        while True:
            _refresh_tailscale_names()
            time.sleep(60)

    threading.Thread(target=loop, daemon=True, name="tailscale-names").start()


def node_name(ip: str | None) -> str:
    """Friendly machine name for a client address. Order: SPARK_STUDIO_NODE_NAMES,
    loopback, the tailnet hostname, then the address itself."""
    if not ip:
        return "unknown"
    if ip in NODE_NAMES:
        return NODE_NAMES[ip]
    if ip in ("127.0.0.1", "::1", "localhost") or ip in _ts_self_ips:
        # The Spark itself, however it is reached (loopback or its tailnet address).
        return "this box"
    return _ts_names.get(ip, ip)


def prepare_stream_body(body: bytes, path: str) -> bytes:
    """Ask the engine to report usage on streamed chat completions.

    Without ``include_usage`` the engine sends no token counts for a stream, so
    prompt and cache numbers would be lost. Set only when the caller did not
    choose its own ``stream_options``. Anything that isn't a streamed chat body
    is returned unchanged.
    """
    if path != "chat/completions" or not body:
        return body
    try:
        payload = json.loads(body)
    except ValueError:
        return body
    if not isinstance(payload, dict) or not payload.get("stream"):
        return body
    if "stream_options" in payload:
        return body
    payload["stream_options"] = {"include_usage": True}
    return json.dumps(payload).encode()


def request_meta(body: bytes) -> tuple[str, bool]:
    """(model, stream) from a request body. Missing or non-JSON bodies give ("", False)."""
    try:
        payload = json.loads(body) if body else {}
    except ValueError:
        return "", False
    if not isinstance(payload, dict):
        return "", False
    return str(payload.get("model") or ""), bool(payload.get("stream"))


def _note_finished(record: dict[str, Any]) -> None:
    """Track the fastest finished request this session (exact tokens over wall time)."""
    global _session_peak_gen
    tok_s = record.get("tok_s")
    if tok_s is not None:
        with _engine_lock:
            _session_peak_gen = max(_session_peak_gen, float(tok_s))


class UsageRecorder:
    """Tracks one request from start to finish.

    Feed it the raw response bytes as they arrive (``feed``). Streamed responses
    are parsed as server-sent events; plain responses are read as one JSON body.
    ``finish`` writes the row and removes the request from the live view.
    """

    def __init__(
        self,
        *,
        client_ip: str | None,
        engine_run_id: str | None,
        model: str,
        endpoint: str,
        stream: bool,
        engine_url: str | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.engine_url = engine_url
        self._text_parts: list[str] = []
        self._text_chars = 0
        self.live_tokens: int | None = None  # exact count of the text so far, from the engine
        self._tools: dict[int, dict[str, Any]] = {}
        self.client_ip = client_ip or ""
        self.node = node_name(client_ip)
        self.engine_run_id = engine_run_id or ""
        self.model = model or ""
        self.endpoint = endpoint
        self.stream = bool(stream)
        self.started_at = time.time()
        self.ended_at: float | None = None
        self.status = "running"
        self.error = ""
        self.saw_done = False  # the engine's [DONE] marker arrived
        self.phase = "waiting"
        self.thinking = False
        self.content_tokens = 0
        self.first_token_at: float | None = None
        self.last_token_at: float | None = None
        self.token_times: list[float] = []
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.cached_tokens: int | None = None
        self.concurrency = 1
        self._in_think_tag = False
        self._sse_buf = b""
        self._body = bytearray()
        self._register()

    # -- live registry ---------------------------------------------------

    def _register(self) -> None:
        global _session_peak_streams
        with _lock:
            same_engine = sum(1 for r in _live.values() if r.engine_run_id == self.engine_run_id)
            self.concurrency = same_engine + 1
            _live[self.id] = self
            _session_peak_streams = max(_session_peak_streams, len(_live))

    def _unregister(self) -> None:
        with _lock:
            _live.pop(self.id, None)

    # -- feeding -----------------------------------------------------------

    def feed(self, chunk: bytes) -> None:
        if self.status != "running" or not chunk:
            return
        if self.stream:
            self._feed_sse(chunk)
        else:
            self._body.extend(chunk)

    def _feed_sse(self, chunk: bytes) -> None:
        self._sse_buf += chunk
        *lines, self._sse_buf = self._sse_buf.split(b"\n")
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                self.saw_done = True
                continue
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if isinstance(obj, dict):
                self._take_event(obj)

    def _take_event(self, obj: dict[str, Any]) -> None:
        usage = obj.get("usage")
        if isinstance(usage, dict):
            self._take_usage(usage)
        now = time.time()
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            content = delta.get("content") or choice.get("text") or ""
            if content and "<think>" in content:
                self.thinking = True
                self._in_think_tag = True
            if reasoning:
                self.thinking = True
                self.phase = "thinking"
                self._token(now)
                self._keep_text(reasoning)
            if content:
                if "</think>" in content:
                    self._in_think_tag = False
                self.phase = "thinking" if self._in_think_tag else "answering"
                self._token(now)
                self._keep_text(content)
            for tool in delta.get("tool_calls") or []:
                if isinstance(tool, dict):
                    self._take_tool_delta(tool)

    def _keep_text(self, text: str) -> None:
        if self._text_chars < MAX_LIVE_TEXT:
            self._text_parts.append(text)
            self._text_chars += len(text)

    def live_text(self) -> str:
        return "".join(self._text_parts)

    def _take_tool_delta(self, tool: dict[str, Any]) -> None:
        """Streamed tool calls arrive in pieces: the name once, then the arguments."""
        index = tool.get("index", 0)
        entry = self._tools.setdefault(index if isinstance(index, int) else 0, {"name": "", "args_chars": 0})
        fn = tool.get("function") or {}
        if isinstance(fn, dict):
            if fn.get("name"):
                entry["name"] = str(fn["name"])
            args = fn.get("arguments")
            if isinstance(args, str):
                entry["args_chars"] += len(args)
        self.phase = "tool call"

    def tool_calls_done(self) -> list[dict[str, Any]]:
        return [{"name": v["name"], "args_chars": v["args_chars"]} for v in self._tools.values() if v["name"]]

    def tools_in_progress(self) -> list[str]:
        return [v["name"] for v in self._tools.values() if v["name"]] if self.phase == "tool call" else []

    def _take_usage(self, usage: dict[str, Any]) -> None:
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if isinstance(prompt, int):
            self.prompt_tokens = prompt
        if isinstance(completion, int):
            self.completion_tokens = completion
        if isinstance(cached, int):
            self.cached_tokens = cached

    def _token(self, now: float) -> None:
        self.content_tokens += 1
        if self.first_token_at is None:
            self.first_token_at = now
        self.last_token_at = now
        if len(self.token_times) < MAX_TOKEN_TIMES:
            self.token_times.append(now)

    # -- finishing ---------------------------------------------------------

    def finish(self, status: str = "ok", error: str = "") -> dict[str, Any] | None:
        if self.status != "running":
            return None
        if not self.stream and status == "ok":
            self._take_body(bytes(self._body))
        self.status = status
        self.error = error or ""
        self.ended_at = time.time()
        self.phase = "done"
        record = self._record()
        db.usage_insert(record)
        calls = self.tool_calls_done()
        if calls:
            db.usage_tool_calls_insert(self.id, self.started_at, self.node, self.model, calls)
        self._unregister()
        _note_finished(record)
        return record

    def _take_body(self, body: bytes) -> None:
        try:
            obj = json.loads(body) if body else {}
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        if isinstance(obj.get("usage"), dict):
            self._take_usage(obj["usage"])
        for choice in obj.get("choices") or []:
            msg = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(msg, dict) and (msg.get("reasoning_content") or msg.get("reasoning")):
                self.thinking = True
            for tool in (msg.get("tool_calls") or []) if isinstance(msg, dict) else []:
                if isinstance(tool, dict):
                    self._take_tool_delta({**tool, "index": len(self._tools)})
        # A plain response arrives all at once, so decode time is the whole call.
        self.first_token_at = self.started_at
        self.last_token_at = self.ended_at or time.time()
        self.content_tokens = self.completion_tokens or 0

    def _record(self) -> dict[str, Any]:
        ended = self.ended_at or time.time()
        completion = self.completion_tokens if self.completion_tokens is not None else self.content_tokens
        decode_secs = None
        tok_s = None
        ttft_ms = None
        if self.first_token_at is not None:
            ttft_ms = round((self.first_token_at - self.started_at) * 1000, 1)
        if self.first_token_at is not None:
            # Wall time from the first token to the end of the response. Chunk
            # arrival times are bunched when the engine batches tokens, so they
            # can't be used as the denominator.
            decode_secs = round(max(0.0, ended - self.first_token_at), 3)
            # Under a second, the window is too short to give a meaningful rate.
            if self.status == "ok" and completion > 1 and decode_secs >= 1.0:
                tok_s = round(completion / decode_secs, 1)
        cache_pct = None
        if self.prompt_tokens and self.cached_tokens is not None:
            cache_pct = round(100.0 * self.cached_tokens / self.prompt_tokens, 1)
        buckets: list[int] = []
        for t in self.token_times:
            idx = int(t - self.started_at)
            if idx >= MAX_SAMPLES:
                break
            while len(buckets) <= idx:
                buckets.append(0)
            buckets[idx] += 1
        return {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": ended,
            "node": self.node,
            "client_ip": self.client_ip,
            "engine_run_id": self.engine_run_id,
            "model": self.model,
            "endpoint": self.endpoint,
            "stream": int(self.stream),
            "status": self.status,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": completion,
            "cached_tokens": self.cached_tokens,
            "cache_hit_pct": cache_pct,
            "thinking": int(self.thinking),
            "concurrency": self.concurrency,
            "ttft_ms": ttft_ms,
            "decode_secs": decode_secs,
            "tok_s": tok_s,
            # Fastest request = its own exact rate; the engine's counter is not used
            # for per-request peaks (see note in README).
            "peak_tok_s": tok_s,
            "samples": buckets,
        }


def _recent_series(token_times: list[float], now: float, seconds: int = 40) -> list[int]:
    """Tokens per second for the last ``seconds`` seconds, oldest first."""
    counts = [0] * seconds
    for t in reversed(token_times):
        age = int(now - t)
        if age >= seconds:
            break
        if age >= 0:
            counts[seconds - 1 - age] += 1
    return counts


def _live_rate(r: "UsageRecorder", now: float) -> float | None:
    """Tokens per second so far for a streaming request: exact count when the
    engine has counted the text, otherwise none."""
    if r.live_tokens is None or r.first_token_at is None:
        return None
    elapsed = now - r.first_token_at
    if elapsed < 1.0:
        return None
    return round(r.live_tokens / elapsed, 1)


async def refresh_live_tokens(client: Any) -> None:
    """Ask each live request's engine to count the text produced so far. Uses the
    engine's own tokenizer, so the count matches what it generated."""
    with _lock:
        rows = list(_live.values())
    for r in rows:
        if not r.engine_url or not r._text_parts:
            continue
        try:
            resp = await client.post(
                f"{r.engine_url}/tokenize",
                json={"model": r.model, "prompt": r.live_text()},
                timeout=3,
            )
        except Exception:  # noqa: BLE001
            continue
        if resp.status_code == 200:
            try:
                count = (resp.json() or {}).get("count")
            except ValueError:
                continue
            if isinstance(count, int) and count >= 0:
                r.live_tokens = count


def live_snapshot() -> dict[str, Any]:
    """Requests in flight, plus the engine's exact current rate.

    Per-stream numbers are activity only: the engine packs several tokens into
    one streamed chunk when it speculates, so a chunk rate would understate the
    real tok/s. Exact tok/s is the engine-wide counter.
    """
    now = time.time()
    with _lock:
        rows = list(_live.values())
    streams = []
    for r in rows:
        streams.append({
            "id": r.id,
            "node": r.node,
            "client_ip": r.client_ip,
            "model": r.model,
            "endpoint": r.endpoint,
            "engine_run_id": r.engine_run_id,
            "phase": r.phase,
            "thinking": r.thinking,
            "tokens_so_far": r.live_tokens if r.live_tokens is not None else r.content_tokens,
            "tokens_exact": r.live_tokens is not None,
            "tok_s": _live_rate(r, now),
            "tools": r.tools_in_progress(),
            "recent": _recent_series(r.token_times, now),
            "started_at": r.started_at,
            "elapsed": round(now - r.started_at, 1),
        })
    streams.sort(key=lambda s: s["started_at"])
    with _engine_lock:
        engine_history = list(_engine_history)
        session_peak = _session_peak_gen
    return {
        "streams_now": len(streams),
        "tok_s_now": round(sum(s["tok_s"] or 0.0 for s in streams), 1) if any(s["tok_s"] for s in streams) else None,
        "tools_running": [name for s in streams for name in s["tools"]],
        "session_peak_streams": _session_peak_streams,
        "session_peak_tok_s": round(session_peak, 1),
        "session_start": SESSION_START,
        "streams": streams,
        "engine": engine_history[-1] if engine_history else None,
        "engine_history": engine_history[-60:],
    }


# ----- engine-side metrics ----------------------------------------------------
# A running vLLM engine exposes Prometheus counters at /metrics. Sampling them
# captures every request the engine serves, including calls that bypass the
# proxy. Counters are cumulative, so the sampler works on the change between
# two polls.

_METRIC_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$")
# Engines name their metrics with their own prefix (vllm:, tensorfold:, ...), so
# match on the part after the prefix.
_METRIC_NAMES = {
    "running": ("num_requests_running",),
    "waiting": ("num_requests_waiting",),
    "prompt": ("prompt_tokens_total",),
    "gen": ("generation_tokens_total",),
    "cache_q": ("prefix_cache_queries_total", "gpu_prefix_cache_queries_total"),
    "cache_h": ("prefix_cache_hits_total", "gpu_prefix_cache_hits_total"),
}


def parse_engine_metrics(text: str) -> dict[str, float]:
    """Pick the counters the sampler needs out of a Prometheus /metrics page.
    Values are summed across label sets. Missing counters are left out."""
    sums: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC_LINE.match(line.strip())
        if not m:
            continue
        name = m.group(1)
        base = name.split(":", 1)[1] if ":" in name else name
        sums[base] = sums.get(base, 0.0) + float(m.group(2))
    out: dict[str, float] = {}
    for key, names in _METRIC_NAMES.items():
        for name in names:
            if name in sums:
                out[key] = sums[name]
                break
    return out


class EngineSampler:
    """Turns two counter readings into per-second rates for one engine run."""

    def __init__(self) -> None:
        self._prev: dict[str, tuple[float, dict[str, float]]] = {}

    def observe(self, run_id: str, counters: dict[str, float], now: float) -> dict[str, Any] | None:
        prev = self._prev.get(run_id)
        self._prev[run_id] = (now, counters)
        if prev is None or now <= prev[0]:
            return None
        dt = now - prev[0]
        before = prev[1]

        def delta(key: str) -> float | None:
            if key not in counters or key not in before:
                return None
            # A counter that went down means the engine restarted; count nothing for that step.
            return max(0.0, counters[key] - before[key])

        gen = delta("gen")
        prompt = delta("prompt")
        dq = delta("cache_q")
        dh = delta("cache_h")
        hit = round(100.0 * dh / dq, 1) if dq and dh is not None else None
        return {
            "id": uuid.uuid4().hex[:16],
            "ts": now,
            "engine_run_id": run_id,
            "source": "live",
            "running": int(counters.get("running", 0)),
            "waiting": int(counters.get("waiting", 0)),
            "gen_tps": round(gen / dt, 1) if gen is not None else None,
            "peak_gen_tps": round(gen / dt, 1) if gen is not None else None,
            "prompt_tps": round(prompt / dt, 1) if prompt is not None else None,
            "cache_hit_pct": hit,
        }


sampler = EngineSampler()
_engine_lock = threading.Lock()
_gen_ring: deque[tuple[float, float | None]] = deque(maxlen=7200)  # two hours of one-second ticks
_engine_history: deque[dict[str, Any]] = deque(maxlen=300)


def note_engine_sample(row: dict[str, Any]) -> None:
    """Keep recent one-second engine ticks in memory for the live view and peaks."""
    global _session_peak_gen
    with _engine_lock:
        _gen_ring.append((row["ts"], row["gen_tps"]))
        _engine_history.append({
            "ts": row["ts"],
            "running": row["running"],
            "waiting": row["waiting"],
            "gen_tps": row["gen_tps"],
            "prompt_tps": row["prompt_tps"],
            "cache_hit_pct": row["cache_hit_pct"],
        })


class EnginePersist:
    """Stores one row per window of one-second ticks, about 8,600 rows a day per
    engine rather than 86,400. Each stored row keeps the window's mean and its
    busiest second."""

    def __init__(self, window: int = 10) -> None:
        self.window = window
        self._rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if self._rows and self._rows[-1]["engine_run_id"] != row["engine_run_id"]:
            self._rows = []
        self._rows.append(row)
        if len(self._rows) < self.window:
            return None
        rows, self._rows = self._rows, []
        prompts = [r["prompt_tps"] for r in rows if r["prompt_tps"] is not None]
        hits = [r["cache_hit_pct"] for r in rows if r["cache_hit_pct"] is not None]
        last = rows[-1]
        return {
            "id": uuid.uuid4().hex[:16],
            "ts": last["ts"],
            "engine_run_id": last["engine_run_id"],
            "source": "live",
            "running": last["running"],
            "waiting": last["waiting"],
            "gen_tps": None,
            "peak_gen_tps": None,
            "prompt_tps": round(sum(prompts) / len(prompts), 1) if prompts else None,
            "cache_hit_pct": hits[-1] if hits else None,
        }


engine_persist = EnginePersist()
