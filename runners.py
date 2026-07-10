"""Subprocess lifecycle manager for vLLM, SGLang, and llama.cpp.

Each run gets a unique id, a line-buffered stdout/stderr pump, an async log
queue, and a live ring buffer. The frontend streams logs over SSE and can
send SIGTERM/SIGKILL via the /runs/{id}/stop endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import db
import oomguard
import sparkrun_service


VENV_BIN = Path(sys.executable).parent

# Model-name extraction from shell commands, for run labels.
_MODEL_FROM_CMD_RES = [
    re.compile(r"(?:vllm|sglang|llama-server|llamacpp)?\s*serve\s+([\w@./:-]+)"),
    re.compile(r"--model(?:-path)?[=\s]+([\w@./:-]+)"),
    re.compile(r"\bmodel:\s*([\w@./:-]+)"),
    re.compile(r"sparkrun\s+run\s+(@[\w./-]+)"),
]


def _guess_label(args: dict[str, Any] | None, raw_cmd: str | None) -> str | None:
    """Best-effort human name for a run: the model id from args, else the
    first model-looking token in the command."""
    if isinstance(args, dict):
        for key in ("model", "model-path", "hf-repo"):
            if args.get(key):
                return str(args[key])
    for rx in _MODEL_FROM_CMD_RES:
        m = rx.search(raw_cmd or "")
        if m and not m.group(1).startswith("-"):
            return m.group(1)
    return None


def _resolve(binary: str) -> str | None:
    """Find a binary: venv bin dir first, then PATH, then common conda locations."""
    local = VENV_BIN / binary
    if local.exists() and os.access(local, os.X_OK):
        return str(local)
    found = shutil.which(binary)
    if found:
        return found
    # conda may not be on PATH when the server is started without sourcing .bashrc
    home = Path.home()
    for conda_bin in [
        home / "miniconda3" / "bin" / binary,
        home / "anaconda3" / "bin" / binary,
        home / "miniforge3" / "bin" / binary,
        Path("/opt/conda/bin") / binary,
    ]:
        if conda_bin.exists() and os.access(conda_bin, os.X_OK):
            return str(conda_bin)
    return None


ENGINE_BINARIES: dict[str, list[str]] = {
    "vllm": ["vllm"],
    "sglang": [],  # python -m sglang.launch_server — checked via module import
    "llamacpp": ["llama-server"],
    "sparkrun": ["sparkrun"],
}

ENGINE_INSTALL_HINTS: dict[str, str] = {
    "vllm": 'uv pip install --python env/bin/python "vllm"',
    "sglang": 'uv pip install --python env/bin/python "sglang[all]"',
    "llamacpp": (
        "llama.cpp server not on PATH. Either install the native binary "
        "(`conda install -c conda-forge llama.cpp`) or install the pip server: "
        '`uv pip install --python env/bin/python "llama-cpp-python[server]"`'
    ),
    "sparkrun": "run `uvx sparkrun setup` in a terminal (guided cluster setup)",
}


def _module_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def engine_available(engine: str) -> bool:
    if engine == "sglang":
        return _module_available("sglang")
    for binary in ENGINE_BINARIES.get(engine, []):
        if _resolve(binary):
            return True
    return False


class EngineMissing(RuntimeError):
    def __init__(self, engine: str):
        self.engine = engine
        super().__init__(
            f"{engine} is not installed in this launcher. "
            f"Hint: {ENGINE_INSTALL_HINTS.get(engine, '')}"
        )


class MemoryTooTight(RuntimeError):
    """Raised by the pre-launch guard when a model won't fit in available
    unified memory. Carries a user-facing message the API surfaces as HTTP 507."""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


READY_PATTERNS = [
    "Application startup complete",
    "Uvicorn running on",
    "HTTP 200 OK",
]

URL_RE = re.compile(r"http://(?:[0-9]{1,3}\.){3}[0-9]{1,3}:\d+|http://localhost:\d+")
DOCKER_NAME_RE = re.compile(r"(?:^|\s)(?:--name|--container-name)\s+([A-Za-z0-9][A-Za-z0-9_.-]*)")


def _mem_used_gb() -> float | None:
    """Used system RAM in GB. On the GB10's unified memory this is where model
    weights land, so a before/after delta approximates the model's footprint."""
    try:
        import psutil
        return round(psutil.virtual_memory().used / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001
        return None


def _mem_total_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:  # noqa: BLE001
        return None


def _mem_available_gb() -> float | None:
    """Memory that could be handed to a new allocation (free + reclaimable) —
    the right number for 'will another model fit?'."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:  # noqa: BLE001
        return None


# gpu-memory-utilization (vLLM) / mem-fraction-static (SGLang): the fraction of
# the WHOLE unified pool the engine pre-fills with weights+KV cache. On DGX
# Spark this — not the model's file size — is what a model actually costs.
_MEM_FRACTION_RE = re.compile(
    r"(?:gpu[-_]memory[-_]utilization|mem[-_]fraction[-_]static)[=\s]+([0-9]*\.?[0-9]+)"
)


def _parse_mem_fraction(raw_cmd: str | None, args: dict[str, Any] | None) -> float | None:
    text = raw_cmd or ""
    if isinstance(args, dict):
        for k, v in args.items():
            text += f" {k}={v}"
    m = _MEM_FRACTION_RE.search(text)
    if m:
        try:
            frac = float(m.group(1))
            if 0 < frac <= 1:
                return frac
        except ValueError:
            pass
    return None


@dataclass
class Run:
    id: str
    engine: str
    recipe_id: int | None
    cmd: list[str]
    env: dict[str, str]
    raw_cmd: str | None = None
    managed_containers: list[str] = field(default_factory=list)
    # Shell words to run on stop for workloads that outlive their launcher
    # process (e.g. `sparkrun stop <name>` — killing `sparkrun run` merely
    # detaches from logs, the model keeps serving).
    stop_cmd: list[str] | None = None
    proc: subprocess.Popen | None = None
    port: int | None = None
    status: str = "starting"
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    exit_code: int | None = None
    # True once the user asked for this run to stop — lets the UI distinguish
    # a requested shutdown (SIGTERM/SIGKILL → negative exit code) from a crash.
    stop_requested: bool = False
    # Detached runs (sparkrun): the workload outlives the pumped process, so a
    # dead launcher/log-tail must not finalize the run — the watchdog does.
    detached: bool = False
    # sparkrun metadata: ref, tp, jobid, pump_cmd (for tail respawn), …
    meta: dict[str, Any] = field(default_factory=dict)
    # Last status tag written for recipe_id ("working"/"fix") — write-once guard.
    recipe_tagged: str | None = None
    # Watchdog state: consecutive URL-probe failures / docker-top "dead" hits.
    probe_failures: int = 0
    serve_dead_count: int = 0
    # PID adopted from a previous server session (no Popen handle exists).
    adopted_pid: int | None = None
    # Human-facing name for run lists (model id / recipe ref) — the hex run id
    # means nothing to a person scanning for "which run was gemma?".
    label: str | None = None
    ring: deque[str] = field(default_factory=lambda: deque(maxlen=4000))
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    url: str | None = None
    ready: bool = False
    # Load telemetry: RAM used when the run launched vs when the engine first
    # answered, and when that happened — powers the "loaded in Xs · +Y GB"
    # readout on run cards. None for adopted runs (no launch baseline).
    ready_at: float | None = None
    mem_at_start_gb: float | None = None
    mem_at_ready_gb: float | None = None

    def mark_ready(self) -> None:
        """Flip to ready; on the FIRST readiness stamp load time + RAM delta.
        (The watchdog may clear and re-grant `ready` later — those don't
        re-stamp, the interesting number is the initial model load.)"""
        was_ready = self.ready
        self.ready = True
        if was_ready or self.ready_at is not None:
            return
        self.ready_at = time.time()
        self.mem_at_ready_gb = _mem_used_gb()
        load_secs = round(self.ready_at - self.started_at, 1)
        self.meta["load_secs"] = load_secs
        if self.mem_at_start_gb is not None and self.mem_at_ready_gb is not None:
            self.meta["ram_delta_gb"] = round(self.mem_at_ready_gb - self.mem_at_start_gb, 1)
        try:
            db.runs_update(self.id, meta_json=json.dumps({k: v for k, v in self.meta.items() if k != "pump_cmd"}))
        except Exception:  # noqa: BLE001
            pass
        delta = self.meta.get("ram_delta_gb")
        self.publish(
            f"[ready] model loaded in {load_secs:.0f}s"
            + (f" — RAM {delta:+.1f} GB ({self.mem_at_ready_gb:.1f} GB used)" if delta is not None else "")
        )

    def publish(self, line: str) -> None:
        self.ring.append(line)
        if not self.ready and any(p in line for p in READY_PATTERNS):
            self.mark_ready()
        if self.engine == "sparkrun" and not self.meta.get("jobid"):
            self._capture_sparkrun_jobid(line)
        # "Uvicorn running on http://..." is the authoritative bind address —
        # always override any pre-set URL with it. Other URL hits only fill in
        # when we don't have one yet.
        m = URL_RE.search(line)
        if m:
            url = m.group(0).replace("0.0.0.0", "127.0.0.1")
            if "Uvicorn running on" in line or not self.url:
                self.url = url
                # Update port too so /api/runs reports the real one.
                try:
                    self.port = int(url.rsplit(":", 1)[1])
                except (ValueError, IndexError):
                    pass
        dead: list[asyncio.Queue] = []
        for q in list(self.subscribers):
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.subscribers.remove(q)

    def _capture_sparkrun_jobid(self, line: str) -> None:
        """Grab the sparkrun job id from launcher output the first time it
        appears, then scope stop to that job and record its containers."""
        m = sparkrun_service.CONTAINER_RE.search(line) or sparkrun_service.JOBID_RE.search(line)
        if not m:
            return
        jobid = m.group(1)
        self.meta["jobid"] = jobid
        # A jobid-scoped stop can't take down other adopted jobs the way a
        # workload-name (or --all) stop could.
        exe = sparkrun_service.sparkrun_bin()
        if exe:
            self.stop_cmd = [exe, "stop", jobid]
        for cm in sparkrun_service.CONTAINER_RE.finditer(line):
            name = cm.group(0)
            if name not in self.managed_containers:
                self.managed_containers.append(name)
        try:
            db.runs_update(self.id, meta_json=json.dumps(self.meta))
        except Exception:  # noqa: BLE001
            pass

    def outcome(self) -> str:
        """Human-facing status: running/starting pass through; exited runs
        split into stopped (user-requested), failed (crash), or exited (clean)."""
        if self.status != "exited":
            return self.status
        if self.stop_requested:
            return "stopped"
        if self.exit_code:
            return "failed"
        return "exited"

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "engine": self.engine,
            "recipe_id": self.recipe_id,
            "status": self.status,
            "outcome": self.outcome(),
            "port": self.port,
            "url": self.url,
            "ready": self.ready,
            "pid": self.proc.pid if self.proc else None,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
            "cmd": " ".join(shlex.quote(c) for c in self.cmd),
            "raw_cmd": self.raw_cmd,
            "ref": self.meta.get("ref"),
            "tp": self.meta.get("tp"),
            "containers": list(self.managed_containers),
            "detached": self.detached,
            "label": self.label or self.meta.get("ref"),
            "load_secs": self.meta.get("load_secs"),
            "ram_delta_gb": self.meta.get("ram_delta_gb"),
        }


class Runner:
    """Manages all engine subprocesses."""

    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _running_model_runs(self) -> list[Run]:
        """Runs we own that hold unified/GPU memory. Excludes external
        registrations (someone else's endpoint — stopping it frees nothing here)."""
        out = []
        for r in self.runs.values():
            if r.status != "running" or r.engine not in ("vllm", "sglang", "llamacpp", "sparkrun"):
                continue
            if r.proc is None and not r.managed_containers and not r.detached:
                continue  # external/registered endpoint, not a local workload
            out.append(r)
        return out

    def _preflight_memory(
        self, *, engine: str, raw_cmd: str | None, args: dict[str, Any] | None, skip: bool,
    ) -> list[str]:
        """Make room for a new model before launching. Returns log lines to
        attach to the new run; raises MemoryTooTight if it can't fit."""
        msgs: list[str] = []
        if skip or os.environ.get("SPARK_STUDIO_NO_MEMORY_GUARD") == "1":
            return msgs
        total = _mem_total_gb()
        if not total:
            return msgs  # no psutil / not Linux — don't gate launches blindly

        frac = _parse_mem_fraction(raw_cmd, args)
        # vLLM/SGLang (and sparkrun, which wraps vLLM) pre-fill KV cache to
        # frac × total. llama.cpp allocates roughly weights+ctx, not the whole
        # pool, so we don't predict a large footprint for it (guard still stops
        # co-resident models, just won't hard-block on a fit estimate).
        need = (frac or 0.80) * total if engine in ("vllm", "sglang", "sparkrun") else None

        # 1) Only one model fits — stop any other resident model first.
        others = self._running_model_runs()
        for r in others:
            label = r.label or r.engine
            msgs.append(f"[guard] stopping running model '{label}' to free unified memory for the new launch")
            print(f"[guard] pre-launch: stopping {label} ({r.id}) to free memory", flush=True)
            try:
                self.stop(r.id)
            except Exception as e:  # noqa: BLE001
                print(f"[guard] stop of {r.id} raised {e}", flush=True)

        # 2) Wait for the freed memory to actually drain — only when we stopped
        #    something (container teardown + page reclaim lag, the exact gap
        #    that caused the OOM before). If there was nothing of ours to stop
        #    and it still doesn't fit, waiting wouldn't help — fail fast below.
        if others:
            timeout = int(os.environ.get("SPARK_STUDIO_MEM_GUARD_TIMEOUT", "120"))
            deadline = time.time() + timeout
            prev = None
            while time.time() < deadline:
                avail = _mem_available_gb()
                if avail is None:
                    break
                if need is not None and avail >= need:
                    break
                if need is None:  # no fit target: wait until reclaim stabilizes
                    if prev is not None and avail <= prev + 1.0:
                        break
                    prev = avail
                time.sleep(1.5)
            avail = _mem_available_gb()
            if avail is not None:
                msgs.append(f"[guard] {avail:.0f} GB unified memory available after freeing previous model(s)")

        # 3) Final fit check for pool-filling engines.
        if need is not None:
            avail = _mem_available_gb() or 0.0
            if avail < need * 0.92:  # small tolerance for measurement jitter
                raise MemoryTooTight(
                    f"Not enough unified memory to launch this model: it needs about "
                    f"{need:.0f} GB (gpu-memory-utilization {frac or 0.80:.2f} × {total:.0f} GB pool) "
                    f"but only {avail:.0f} GB is available. On DGX Spark each model fills most of the "
                    f"128 GB pool, so only one runs at a time. Stop other GPU workloads, lower "
                    f"gpu-memory-utilization in the recipe, or relaunch with force to override "
                    f"(or set SPARK_STUDIO_NO_MEMORY_GUARD=1)."
                )
        return msgs

    def start(
        self,
        engine: str,
        args: dict[str, Any],
        env_extra: dict[str, str] | None = None,
        recipe_id: int | None = None,
        raw_cmd: str | None = None,
        port: int | None = None,
        managed_containers: list[str] | None = None,
        stop_cmd: list[str] | None = None,
        detached: bool = False,
        meta: dict[str, Any] | None = None,
        skip_memory_guard: bool = False,
    ) -> Run:
        # Pre-launch memory guard: on DGX Spark each model fills most of the
        # 128 GB unified pool, so a second model launched before the first is
        # freed → OOM (and earlyoom may take the dashboard with it). Stop any
        # other resident model, wait for its memory to actually reclaim, and
        # refuse a launch that still won't fit. Raises MemoryTooTight.
        guard_msgs = self._preflight_memory(
            engine=engine, raw_cmd=raw_cmd, args=args, skip=skip_memory_guard,
        )
        # Raw-command mode: skip engine-availability checks, spawn the verbatim
        # shell string. Ideal for DGX Spark docker recipes (`docker run ... &&
        # docker exec vllm_node vllm serve ...`). We still watch the output
        # stream for readiness markers and URLs.
        if raw_cmd:
            cmd = ["bash", "-lc", raw_cmd]
            extracted = self._extract_container_names(raw_cmd)
            if managed_containers:
                # Caller-supplied names take precedence but merge regex hits.
                merged = list(managed_containers)
                for n in extracted:
                    if n not in merged:
                        merged.append(n)
                managed_containers = merged
            else:
                managed_containers = extracted
        else:
            if not engine_available(engine):
                raise EngineMissing(engine)
            if port is None:
                port = args.get("port") or _free_port()
            cmd = self._build_cmd(engine, args, port)
            managed_containers = list(managed_containers or [])
        env = os.environ.copy()
        # Ensure the venv bin dir comes first on PATH so engine binaries find
        # their python/scripts siblings without needing `source activate`.
        env["PATH"] = f"{VENV_BIN}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(VENV_BIN.parent)
        if env_extra:
            env.update({k: str(v) for k, v in env_extra.items()})

        run_id = uuid.uuid4().hex[:12]
        run = Run(
            id=run_id,
            engine=engine,
            recipe_id=recipe_id,
            cmd=cmd,
            env=env,
            raw_cmd=raw_cmd,
            managed_containers=managed_containers,
            stop_cmd=stop_cmd,
            port=port,
            detached=detached,
            meta=meta or {},
        )
        run.label = _guess_label(args, raw_cmd) or (meta or {}).get("ref")
        for m in guard_msgs:
            run.publish(m)
        run.mem_at_start_gb = _mem_used_gb()
        if port is not None:
            run.url = f"http://127.0.0.1:{port}"

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
                # Own process group so stop() can signal the whole tree —
                # bash -lc wrappers otherwise leave their children running.
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise EngineMissing(engine) from e
        run.proc = proc
        # A directly-spawned engine is the model process itself: mark it a
        # preferred OOM victim so memory pressure kills the (relaunchable)
        # engine before the dashboard. Docker/sparkrun runs (raw_cmd) launch
        # the model in a container we don't own, so there's no local pid to
        # tag — those rely on the earlyoom --prefer fix instead.
        if not raw_cmd:
            oomguard.deprioritize(proc.pid)
        run.status = "running"
        self.runs[run_id] = run
        db.runs_insert(
            {
                "id": run_id,
                "recipe_id": recipe_id,
                "engine": engine,
                "status": "running",
                "pid": proc.pid,
                "port": port,
                "cmd": run.summary()["cmd"],
                "meta_json": json.dumps(run.meta) if run.meta else None,
            }
        )

        threading.Thread(target=self._pump, args=(run,), daemon=True).start()
        return run

    def _pump(self, run: Run) -> None:
        assert run.proc is not None
        try:
            for line in run.proc.stdout:  # type: ignore[union-attr]
                run.publish(line.rstrip("\n"))
        except Exception as e:  # noqa: BLE001
            run.publish(f"[runner error] {e}")
        finally:
            run.proc.wait()
            if run.status == "exited":
                # Already finalized elsewhere (watchdog / finalize()).
                pass
            elif run.detached and not run.stop_requested:
                # sparkrun workloads outlive their launcher/log tail: the
                # pumped process dying says nothing about the engine. Leave
                # the run running — the watchdog owns lifecycle from here
                # (and respawns the tail).
                run.publish("[launcher] log stream ended — workload may still be running; watchdog keeps monitoring")
            else:
                run.exit_code = run.proc.returncode
                self._capture_managed_container_logs(run)
                self._cleanup_run(run)
                run.ended_at = time.time()
                run.status = "exited"
                db.runs_update(run.id, status="exited", ended_at=db.now(), exit_code=run.exit_code)
                run.publish(f"[exit] code={run.exit_code}")
                self._tag_recipe_from_outcome(run)
                for q in list(run.subscribers):
                    try:
                        q.put_nowait("__EOF__")
                    except Exception:
                        pass

    def _tag_recipe_from_outcome(self, run: Run) -> None:
        """Server-side failure tagging: mark the recipe broken when its run
        crashed. 'working' is granted by the watchdog on readiness, and a
        user-stopped run keeps whatever tag it already earned."""
        if not run.recipe_id or run.outcome() != "failed" or run.recipe_tagged == "fix":
            return
        try:
            db.recipes_set_status_tag(run.recipe_id, ok=False)
            run.recipe_tagged = "fix"
        except Exception:  # noqa: BLE001
            pass

    def stop(self, run_id: str, force: bool = False) -> bool:
        run = self.runs.get(run_id)
        if not run:
            return False
        # Console breadcrumb: every stop request is deliberate — make the
        # source traceable when a workload disappears unexpectedly.
        print(f"[stop] requested for {run_id} ({run.label or run.engine}, force={force})", flush=True)
        if not run.proc and run.adopted_pid:
            # Re-adopted plain-process run from a previous session: no Popen
            # handle, but we know the pid (verified alive at adoption).
            run.stop_requested = True
            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                os.killpg(os.getpgid(run.adopted_pid), sig)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(run.adopted_pid, sig)
                except ProcessLookupError:
                    pass
            self.finalize(run, exit_code=-int(sig), reason="stopped by user (adopted run)")
            return True
        if not run.proc:
            return False
        if run.proc.poll() is not None and not (run.detached and run.status == "running"):
            return True
        run.stop_requested = True
        # For docker-based runs, stop/kill the container immediately so the
        # wrapper bash exits quickly instead of blocking on `docker run`.
        if run.managed_containers:
            self._stop_docker_containers(run, force=force)
        # Workloads that survive their launcher (sparkrun) need an explicit
        # stop command. Run it in a thread with output captured into the run
        # log — a silent failure here means the model keeps serving with no
        # way for the user to tell why Stop "did nothing".
        if run.stop_cmd:
            threading.Thread(target=self._run_stop_cmd, args=(run,), daemon=True).start()
        if run.detached and (run.proc is None or run.proc.poll() is not None):
            # Detached run whose log tail already died: nothing left to signal
            # locally — the container/stop_cmd teardown above is the real stop.
            self.finalize(run, exit_code=0, reason="stopped by user")
            return True
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            # Signal the whole group: raw_cmd runs are bash wrappers whose
            # children (sparkrun, docker attach, …) ignore a lone SIGTERM to
            # the wrapper — or survive it entirely.
            os.killpg(os.getpgid(run.proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                run.proc.send_signal(sig)
            except ProcessLookupError:
                return True
        return True

    def _run_stop_cmd(self, run: Run) -> None:
        """Execute run.stop_cmd synchronously, streaming its output into the
        run log so failures are visible instead of silent. When the stop
        command fails (bad/stale jobid, sparkrun error), fall back to
        force-removing the workload's containers directly — Stop must mean
        stopped, not "the stop command exited 1"."""
        run.publish(f"[stop] $ {' '.join(run.stop_cmd)}")
        ok = False
        try:
            res = subprocess.run(
                run.stop_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
                env=run.env or None,
            )
            for line in (res.stdout or "").splitlines():
                if line.strip():
                    run.publish(f"[stop] {line.rstrip()}")
            ok = res.returncode == 0
            if not ok:
                run.publish(f"[stop] stop command exited {res.returncode} — force-removing containers")
        except subprocess.TimeoutExpired:
            run.publish("[stop] stop command timed out after 180s — force-removing containers")
        except Exception as e:  # noqa: BLE001
            run.publish(f"[stop] failed to execute: {e} — force-removing containers")
        if not ok:
            ok = self._force_remove_workload_containers(run)
        run.publish(
            "[stop] workload stopped" if ok
            else "[stop] could not confirm teardown — workload may still be running (check `sparkrun status`)"
        )

    def _force_remove_workload_containers(self, run: Run) -> bool:
        """Last-resort teardown when the workload's own stop command failed:
        `docker rm -f` every local container that belongs to this run, found
        by recorded name or by jobid substring in a sparkrun_* container name
        (substring, because job ids recorded by older builds may be truncated
        halves of the real underscore-joined id). Returns True when nothing
        belonging to the run is left running locally."""
        docker = shutil.which("docker")
        if not docker:
            return False
        names = set(run.managed_containers)
        jobid = (run.meta or {}).get("jobid")
        try:
            res = subprocess.run(
                [docker, "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=20,
            )
            live = [n.strip() for n in (res.stdout or "").splitlines() if n.strip()]
        except Exception:  # noqa: BLE001
            live = []
        if jobid:
            names.update(n for n in live if n.startswith("sparkrun_") and jobid in n)
        if not names:
            return False
        ok = True
        for name in sorted(names):
            try:
                res = subprocess.run(
                    [docker, "rm", "-f", name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=60,
                )
                out = (res.stdout or "").strip()
                if res.returncode == 0:
                    run.publish(f"[stop] removed container {name}")
                elif "No such container" in out:
                    pass  # already gone — that's the outcome we wanted
                else:
                    ok = False
                    run.publish(f"[stop] docker rm -f {name}: {out}")
            except Exception as e:  # noqa: BLE001
                ok = False
                run.publish(f"[stop] docker rm -f {name} failed: {e}")
        return ok

    def _stop_docker_containers(self, run: Run, force: bool = False) -> None:
        """Fire-and-forget docker stop/kill; stop() must return immediately."""
        docker = shutil.which("docker")
        if not docker:
            return
        for name in run.managed_containers:
            try:
                subprocess.Popen(
                    [docker, "kill" if force else "stop", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:  # noqa: BLE001
                pass

    def _extract_container_names(self, raw_cmd: str) -> list[str]:
        names = []
        for name in DOCKER_NAME_RE.findall(raw_cmd or ""):
            if name not in names:
                names.append(name)
        return names

    def _cleanup_run(self, run: Run) -> None:
        if run.managed_containers:
            self._cleanup_docker_containers(run)

    def _capture_managed_container_logs(self, run: Run) -> None:
        docker = shutil.which("docker")
        if not docker or not run.managed_containers:
            return
        for name in run.managed_containers:
            try:
                inspect = subprocess.run(
                    [docker, "inspect", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                if inspect.returncode != 0:
                    continue
                res = subprocess.run(
                    [docker, "logs", "--tail", "200", name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=20,
                )
                lines = [line.rstrip() for line in (res.stdout or "").splitlines() if line.strip()]
                if not lines:
                    continue
                run.publish(f"[container:{name}] recent logs")
                for line in lines:
                    run.publish(f"[container:{name}] {line}")
            except Exception as e:  # noqa: BLE001
                run.publish(f"[container:{name}] failed to read logs: {e}")

    def _cleanup_docker_containers(self, run: Run) -> None:
        docker = shutil.which("docker")
        if not docker:
            return
        for name in run.managed_containers:
            try:
                res = subprocess.run(
                    [docker, "rm", "-f", name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=20,
                )
                out = (res.stdout or "").strip()
                if res.returncode == 0:
                    run.publish(f"[cleanup] removed container {name}")
                elif out and "No such container" not in out:
                    run.publish(f"[cleanup] docker rm -f {name}: {out}")
            except Exception as e:  # noqa: BLE001
                run.publish(f"[cleanup] docker rm -f {name} failed: {e}")

    def list(self) -> list[dict[str, Any]]:
        return [r.summary() for r in self.runs.values()]

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def active(self) -> Run | None:
        """Return the most recent running run, for the chat/canvas target."""
        live = [r for r in self.runs.values() if r.status == "running"]
        if not live:
            return None
        return max(live, key=lambda r: r.started_at)

    def register_external(self, engine: str, name: str, url: str) -> Run:
        """Register a managed-by-someone-else endpoint (e.g. spark-vllm-docker)
        as a Run so chat/bench/agent-fix flows can target it uniformly."""
        run_id = uuid.uuid4().hex[:12]
        cmd = [f"[external] {name}", url]
        run = Run(id=run_id, engine=engine, recipe_id=None, cmd=cmd, env={}, port=None)
        run.label = name
        run.url = url.rstrip("/").replace("0.0.0.0", "127.0.0.1")
        run.status = "running"
        # Already serving when registered — there's no load to time.
        run.ready_at = run.started_at
        run.ring.append(f"[external] registered {name} at {url}")
        self.runs[run_id] = run
        db.runs_insert(
            {
                "id": run_id,
                "recipe_id": None,
                "engine": engine,
                "status": "running",
                "pid": None,
                "port": None,
                "cmd": f"[external] {name} {url}",
            }
        )
        return run

    def adopt(
        self,
        run_id: str,
        *,
        engine: str,
        ref: str | None = None,
        jobid: str | None = None,
        containers: list[str] | None = None,
        stop_cmd: list[str] | None = None,
        recipe_id: int | None = None,
        url: str | None = None,
        port: int | None = None,
        started_at: float | None = None,
        adopted_pid: int | None = None,
        pump_cmd: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        cmd_desc: str | None = None,
        label: str | None = None,
    ) -> Run | None:
        """Re-attach to a workload that survived a server restart, reusing its
        old run id so history and recipe links stay intact."""
        if run_id in self.runs:
            return self.runs[run_id]
        if jobid and any((r.meta or {}).get("jobid") == jobid for r in self.runs.values()):
            return None  # this job is already adopted under another run id
        run = Run(
            id=run_id,
            engine=engine,
            recipe_id=recipe_id,
            cmd=[cmd_desc or f"[adopted] {ref or run_id}"],
            env={},
            managed_containers=list(containers or []),
            stop_cmd=stop_cmd,
            port=port,
            detached=engine == "sparkrun",
            meta={**(meta or {}), **({"ref": ref} if ref else {}), **({"jobid": jobid} if jobid else {})},
            adopted_pid=adopted_pid,
        )
        run.url = url
        run.status = "running"
        run.label = label or _guess_label(None, cmd_desc) or ref
        if started_at:
            run.started_at = started_at
        # The engine was already up when we re-attached: there is no load to
        # time or RAM baseline to diff. Pre-stamping ready_at makes mark_ready
        # keep whatever load stats the original run recorded in meta.
        run.ready_at = run.started_at
        if pump_cmd:
            run.meta["pump_cmd"] = pump_cmd
        run.publish(f"[adopted] re-attached to {ref or run_id} after restart")
        self.runs[run_id] = run
        if pump_cmd:
            self._spawn_tail(run, pump_cmd)
        return run

    def _spawn_tail(self, run: Run, pump_cmd: list[str]) -> None:
        try:
            proc = subprocess.Popen(
                pump_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            run.proc = proc
            run.meta["tail_spawned_at"] = time.time()
            threading.Thread(target=self._pump, args=(run,), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            run.publish(f"[adopted] could not attach log stream: {e}")

    def respawn_tail(self, run: Run) -> None:
        """Restart the log tail for a detached run whose pump died (rate-limited
        by the caller). No-op while a tail is still alive."""
        if run.status != "running" or not run.detached:
            return
        if run.proc and run.proc.poll() is None:
            return
        pump_cmd = run.meta.get("pump_cmd")
        if not pump_cmd:
            jobid = run.meta.get("jobid")
            container = run.managed_containers[0] if run.managed_containers else None
            pump_cmd = sparkrun_service.tail_pump_cmd(jobid, container)
            if pump_cmd:
                run.meta["pump_cmd"] = pump_cmd
        if pump_cmd:
            self._spawn_tail(run, pump_cmd)

    def finalize(self, run: Run, exit_code: int | None, reason: str = "", teardown: bool = False) -> None:
        """Terminal-state a run from outside its pump (watchdog / stop of an
        adopted run): set exited, persist, tag the recipe, close streams.
        teardown=True also stops the zombie workload (dead engine inside a
        still-Up container) so a relaunch starts clean."""
        if run.status == "exited":
            return
        run.exit_code = exit_code
        run.ended_at = time.time()
        run.status = "exited"
        if reason:
            run.publish(f"[watchdog] {reason}")
        if teardown:
            if run.managed_containers:
                self._stop_docker_containers(run, force=False)
            if run.stop_cmd:
                threading.Thread(target=self._run_stop_cmd, args=(run,), daemon=True).start()
        try:
            db.runs_update(run.id, status="exited", ended_at=db.now(), exit_code=exit_code)
        except Exception:  # noqa: BLE001
            pass
        self._tag_recipe_from_outcome(run)
        if run.proc and run.proc.poll() is None:
            try:
                os.killpg(os.getpgid(run.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    run.proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
        for q in list(run.subscribers):
            try:
                q.put_nowait("__EOF__")
            except Exception:  # noqa: BLE001
                pass

    def shutdown_all(self) -> None:
        """Stop every running workload on app shutdown (Ctrl+C) so models
        don't linger on the GPU after the dashboard is gone. Synchronous —
        called from the FastAPI shutdown hook via a thread."""
        running = [r for r in self.runs.values() if r.status == "running"]
        for run in running:
            label = run.label or run.meta.get("ref") or run.engine
            print(f"[shutdown] stopping {label} ({run.id}) …", flush=True)
            # Mark exited first so concurrently-dying pump threads skip their
            # own finalization (they check run.status).
            run.stop_requested = True
            run.status = "exited"
            run.exit_code = 0
            run.ended_at = time.time()
            try:
                db.runs_update(run.id, status="exited", ended_at=db.now(), exit_code=0)
            except Exception:  # noqa: BLE001
                pass
            if run.managed_containers:
                self._stop_docker_containers(run, force=False)
            pid = None
            if run.proc and run.proc.poll() is None:
                pid = run.proc.pid
            elif run.adopted_pid:
                pid = run.adopted_pid
            if pid:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
            if run.stop_cmd:
                # Synchronous: sparkrun/docker teardown must be issued before
                # the process exits (the spawned stop continues on its own).
                self._run_stop_cmd(run)
            print(f"[shutdown] {label} stopped", flush=True)

    def unregister_external(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if not run or run.proc is not None:
            return False
        run.status = "exited"
        run.stop_requested = True
        run.ended_at = time.time()
        db.runs_update(run_id, status="exited", ended_at=db.now(), exit_code=0)
        return True

    def _build_cmd(self, engine: str, args: dict[str, Any], port: int) -> list[str]:
        if engine == "vllm":
            return self._vllm(args, port)
        if engine == "sglang":
            return self._sglang(args, port)
        if engine == "llamacpp":
            return self._llamacpp(args, port)
        raise ValueError(f"unknown engine: {engine}")

    def _vllm(self, args: dict[str, Any], port: int) -> list[str]:
        model = args.get("model")
        if not model:
            raise ValueError("vllm recipe requires 'model'")
        binary = _resolve("vllm") or "vllm"
        cmd = [binary, "serve", str(model), "--port", str(port), "--host", "127.0.0.1"]
        for k, v in args.items():
            # Underscore-prefixed keys are app metadata (_registry, _sparkrun,
            # _spark_yaml, _no_*), never engine flags.
            if k in ("model", "port", "host") or k.startswith("_"):
                continue
            cmd += self._flagify(k, v)
        return cmd

    def _sglang(self, args: dict[str, Any], port: int) -> list[str]:
        model = args.get("model") or args.get("model-path")
        if not model:
            raise ValueError("sglang recipe requires 'model' or 'model-path'")
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", str(model),
            "--port", str(port),
            "--host", "127.0.0.1",
        ]
        for k, v in args.items():
            if k in ("model", "model-path", "port", "host") or k.startswith("_"):
                continue
            cmd += self._flagify(k, v)
        return cmd

    # llama-server registers exact option spellings: `--c` / `--ngl` / `--ub` are
    # rejected with "invalid argument" — only `-c`/`--ctx-size` etc. exist. Recipes
    # (and the fix agents) commonly use the short names as args keys, so translate
    # them to the canonical long flags before flagifying.
    _LLAMACPP_SHORT_KEYS = {
        "c": "ctx-size",
        "b": "batch-size",
        "ub": "ubatch-size",
        "ngl": "n-gpu-layers",
        "np": "parallel",
        "t": "threads",
        "fa": "flash-attn",
        "cb": "cont-batching",
        "ctk": "cache-type-k",
        "ctv": "cache-type-v",
    }

    def _llamacpp(self, args: dict[str, Any], port: int) -> list[str]:
        model = args.get("model") or args.get("m")
        if not model:
            raise ValueError("llama.cpp recipe requires 'model' (gguf path or HF id)")
        binary = _resolve("llama-server") or "llama-server"
        cmd = [binary, "--port", str(port), "--host", "127.0.0.1"]
        # HF repo shorthand: user/repo:quant — but a relative file path like
        # models/foo.gguf also contains one "/", so file-looking values stay -m.
        if isinstance(model, str) and model.count("/") == 1 and not model.lower().endswith(".gguf"):
            cmd += ["-hf", model]
        else:
            cmd += ["-m", str(model)]
        for k, v in args.items():
            if k in ("model", "m", "port", "host") or k.startswith("_"):
                continue
            k = self._LLAMACPP_SHORT_KEYS.get(k.lstrip("-"), k)
            # Current llama.cpp takes a value for --flash-attn (on|off|auto);
            # a bare `--flash-attn` swallows the next flag as its value.
            if k == "flash-attn" and isinstance(v, bool):
                v = "on" if v else "off"
            cmd += self._flagify(k, v)
        return cmd

    @staticmethod
    def _flagify(k: str, v: Any) -> list[str]:
        # Single-char keys are short options (-c, -t, …); `--c` is invalid.
        flag = k if k.startswith("-") else (f"-{k}" if len(k) == 1 else f"--{k}")
        if isinstance(v, bool):
            return [flag] if v else []
        if isinstance(v, list):
            return [flag] + [str(x) for x in v]
        return [flag, str(v)]


runner = Runner()
