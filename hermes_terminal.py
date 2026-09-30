"""Browser terminal bridge for Spark Studio's embedded Hermes TUI.

The Hermes dashboard uses the same architecture: a real ``hermes --tui``
process runs behind a pseudo-terminal while xterm.js renders its ANSI stream
in the browser.  This module keeps the bridge deliberately small and POSIX
only; DGX Spark runs Linux, and native Windows users should use WSL.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import pty
import select
import signal
import sqlite3
import struct
import sys
import termios
import threading
import time
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import agentlab


class PtyUnavailableError(RuntimeError):
    """Raised when the host cannot provide a POSIX pseudo-terminal."""


class PtyBridge:
    """Byte-safe child-process bridge backed by a POSIX pseudo-terminal."""

    def __init__(self, pid: int, fd: int) -> None:
        self.pid = pid
        self.fd = fd
        self.closed = False
        self._close_lock = threading.Lock()

    @classmethod
    def available(cls) -> bool:
        return os.name == "posix" and not sys.platform.startswith("win")

    @classmethod
    def spawn(
        cls,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        cols: int = 120,
        rows: int = 36,
    ) -> "PtyBridge":
        if not cls.available():
            raise PtyUnavailableError(
                "The embedded Hermes terminal requires Linux, macOS, or WSL."
            )
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover - the child replaces itself
            try:
                os.chdir(cwd)
                os.execvpe(argv[0], list(argv), env)
            except BaseException as exc:  # noqa: BLE001
                os.write(2, f"Unable to start Hermes: {exc}\r\n".encode())
                os._exit(127)
        bridge = cls(pid, fd)
        bridge.resize(cols, rows)
        return bridge

    def read(self, timeout: float = 0.2) -> bytes | None:
        """Return bytes, ``b''`` on timeout, or ``None`` at child EOF."""
        if self.closed:
            return None
        try:
            readable, _, _ = select.select([self.fd], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not readable:
            return b""
        try:
            payload = os.read(self.fd, 65536)
        except OSError as exc:
            if exc.errno in {errno.EIO, errno.EBADF}:
                return None
            raise
        return payload or None

    def write(self, payload: bytes) -> None:
        if self.closed or not payload:
            return
        remaining = memoryview(payload)
        while remaining:
            try:
                written = os.write(self.fd, remaining)
            except OSError as exc:
                if exc.errno in {errno.EIO, errno.EBADF, errno.EPIPE}:
                    return
                raise
            if written <= 0:
                return
            remaining = remaining[written:]

    def resize(self, cols: int, rows: int) -> None:
        if self.closed:
            return
        cols = max(20, min(int(cols), 2000))
        rows = max(5, min(int(rows), 1000))
        try:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self.fd, termios.TIOCSWINSZ, winsize)
        except (OSError, TypeError, ValueError, struct.error):
            pass

    def close(self) -> None:
        """Reap the TUI and every helper in its process group.

        Exactly-once, under a lock. A leased PTY is closed from several
        directions — the drain thread exiting, an explicit Stop, the TTL
        janitor, server shutdown — and a bare ``if self.closed`` check races:
        two callers both pass it, one reaps the child, and the other then
        resolves a *recycled* pid and signals whatever process group now owns
        it. Escalation also stops the moment the child is reaped, so no signal
        is ever sent against a pid this instance no longer owns.
        """
        with self._close_lock:
            if self.closed:
                return
            self.closed = True

            # pty.fork() calls setsid() in the child, so its pgid is its pid.
            # Never re-derive it with getpgid(): that reports the group a pid
            # is in *now*, and once the child is reaped the pid can be recycled
            # into an unrelated group. Refuse to signal our own group outright —
            # a stray killpg here would take down the whole dashboard.
            own_group = os.getpgrp()
            signalable = self.pid > 0 and self.pid not in (own_group, os.getpid())
            for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGKILL):
                if not signalable:
                    break
                try:
                    os.killpg(self.pid, sig)
                except (OSError, ProcessLookupError):
                    break
                if self._reap(deadline=0.4):
                    break
            try:
                os.close(self.fd)
            except OSError:
                pass

    def _reap(self, *, deadline: float) -> bool:
        """Wait briefly for the child; True once it is gone."""
        limit = time.monotonic() + deadline
        while True:
            try:
                if os.waitpid(self.pid, os.WNOHANG)[0] == self.pid:
                    return True
            except ChildProcessError:
                return True
            except OSError:
                return True
            if time.monotonic() >= limit:
                return False
            time.sleep(0.02)


def browser_tui_command(
    binary: str,
    model: str | None,
    provider: str = "custom",
    resume_session: str | None = None,
) -> list[str]:
    """Build the real Hermes Ink TUI command used by the browser terminal.

    ``model`` is omitted when nothing overrides the profile, letting Hermes
    launch on whatever ``model:`` the isolated ``config.yaml`` already carries.
    """
    command = [binary, "--tui"]
    if model:
        command += ["--model", model, "--provider", provider or "custom"]
    command += ["--toolsets", agentlab.hermes_interactive_toolsets()]
    if resume_session:
        command += ["--resume", resume_session]
    return command


def prepare_browser_tui(
    endpoint: dict[str, Any],
    workspace: Path,
    *,
    max_turns: int = 90,
    resume_session: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Refresh the isolated Hermes profile and return command + env.

    A detached endpoint (no engine loaded, or one swapping out mid-session)
    leaves the profile's existing ``model:`` alone, so Chat keeps running on the
    provider the user picked with ``/model``.
    """
    binary = agentlab.find_hermes()
    if not binary:
        raise RuntimeError(f"Hermes Agent is not installed. Run: {agentlab.HERMES_INSTALL}")
    workspace = workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace directory does not exist: {workspace}")

    agentlab._write_hermes_config(
        endpoint.get("base_url"),
        endpoint.get("model"),
        max_turns,
        studio_url=endpoint.get("studio_url") or "http://127.0.0.1:7860",
        enable_search=True,
        respect_user_provider=True,
    )
    binding = agentlab.hermes_model_binding()
    if not binding["model"]:
        raise RuntimeError(
            "No model is loaded and Hermes has no saved model yet. Start an "
            "engine, or pick a provider with /model once Chat is running."
        )
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(agentlab.HERMES_HOME),
            "HERMES_WRITE_SAFE_ROOT": os.pathsep.join(
                (str(workspace), str(agentlab.HERMES_HOME))
            ),
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "HERMES_TUI_DASHBOARD": "1",
            "HERMES_TUI_DISABLE_MOUSE": "1",
            "HERMES_TUI_INLINE": "1",
        }
    )
    command = browser_tui_command(
        binary, binding["model"], binding["provider"], resume_session=resume_session
    )
    return command, env


# ---------------------------------------------------------------------------
# Conversation continuity across lost leases.
#
# A lease lives in the dashboard's memory, so a dashboard restart (or a lease
# reaped after a long sleep) takes its Hermes with it. The conversation itself
# is safe in Hermes' own state.db, though — the ledger below remembers when and
# where each lease's Hermes started, which is enough to find the conversation it
# was having and hand it to ``--resume`` when the browser comes back asking for
# that lease. The Ink TUI doesn't write Hermes' per-terminal breadcrumbs, and a
# bare ``--continue`` would pick up whichever tab spoke last, hence the ledger.

LEASE_LEDGER = agentlab.DATA_DIR / "terminal-leases.json"
_LEASE_LEDGER_MAX_AGE = 7 * 24 * 3600
_LEASE_LEDGER_LOCK = threading.Lock()


def _lease_key(lease_id: str) -> str:
    # The lease id is the reattach credential; only a digest goes to disk.
    return hashlib.sha256(lease_id.encode()).hexdigest()[:32]


def _read_lease_ledger() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(LEASE_LEDGER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def record_lease(lease_id: str, workspace: Path, resumed: str | None = None) -> None:
    """Remember a new lease's start so a later lost-lease reconnect can find
    its conversation. Best-effort: continuity is a nicety, never a blocker."""
    now = time.time()
    entry: dict[str, Any] = {"spawned_at": now, "workspace": str(workspace)}
    if resumed:
        entry["resumed"] = resumed
    with _LEASE_LEDGER_LOCK:
        ledger = {
            key: value
            for key, value in _read_lease_ledger().items()
            if isinstance(value, dict)
            and now - float(value.get("spawned_at") or 0) < _LEASE_LEDGER_MAX_AGE
        }
        ledger[_lease_key(lease_id)] = entry
        try:
            LEASE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            tmp = LEASE_LEDGER.with_suffix(".tmp")
            tmp.write_text(json.dumps(ledger), encoding="utf-8")
            os.replace(tmp, LEASE_LEDGER)
        except OSError:
            pass


def conversation_for_lease(lease_id: str) -> str | None:
    """The Hermes session a lost lease was running, or None to start fresh.

    Picks the newest TUI session its Hermes opened — started after the lease
    and before the next lease in the same workspace — so a ``/new`` mid-lease
    resumes the newer conversation. A lease that itself resumed one and never
    opened another carries that id forward.
    """
    if not lease_id:
        return None
    with _LEASE_LEDGER_LOCK:
        ledger = _read_lease_ledger()
    entry = ledger.get(_lease_key(lease_id))
    if not isinstance(entry, dict):
        return None
    try:
        spawned_at = float(entry.get("spawned_at") or 0)
    except (TypeError, ValueError):
        return None
    workspace = str(entry.get("workspace") or "")
    later = [
        float(other.get("spawned_at") or 0)
        for other in ledger.values()
        if isinstance(other, dict)
        and other.get("workspace") == workspace
        and float(other.get("spawned_at") or 0) > spawned_at
    ]
    until = min(later) if later else float("inf")
    state_db = agentlab.HERMES_HOME / "state.db"
    if not state_db.is_file():
        return None
    try:
        # closing(), not the connection's own context manager — that one only
        # commits, it never closes.
        with closing(sqlite3.connect(f"file:{state_db}?mode=ro", uri=True, timeout=2)) as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE source = 'tui' AND cwd = ?"
                " AND started_at >= ? AND started_at < ?"
                " ORDER BY started_at DESC LIMIT 1",
                (workspace, spawned_at - 1, until),
            ).fetchone()
            if row:
                return str(row[0])
            resumed = str(entry.get("resumed") or "")
            if resumed and conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (resumed,)
            ).fetchone():
                return resumed
    except sqlite3.Error:
        return None
    return None
