"""Browser terminal bridge for Spark Studio's embedded Hermes TUI.

The Hermes dashboard uses the same architecture: a real ``hermes --tui``
process runs behind a pseudo-terminal while xterm.js renders its ANSI stream
in the browser.  This module keeps the bridge deliberately small and POSIX
only; DGX Spark runs Linux, and native Windows users should use WSL.
"""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import time
from collections.abc import Sequence
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
        """Reap the TUI and every helper in its process group."""
        if self.closed:
            return
        self.closed = True
        try:
            pgid = os.getpgid(self.pid)
        except (OSError, ProcessLookupError):
            pgid = None
        for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig) if pgid is not None else os.kill(self.pid, sig)
            except (OSError, ProcessLookupError):
                break
            deadline = time.monotonic() + 0.4
            while time.monotonic() < deadline:
                try:
                    waited, _ = os.waitpid(self.pid, os.WNOHANG)
                except ChildProcessError:
                    waited = self.pid
                if waited == self.pid:
                    break
                time.sleep(0.02)
            else:
                continue
            break
        try:
            os.close(self.fd)
        except OSError:
            pass
        try:
            os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            pass


def browser_tui_command(binary: str, model: str) -> list[str]:
    """Build the real Hermes Ink TUI command used by the browser terminal."""
    return [
        binary,
        "--tui",
        "--model",
        model,
        "--toolsets",
        "file,terminal,mcp-sparkstudio",
    ]


def prepare_browser_tui(
    endpoint: dict[str, Any],
    workspace: Path,
    *,
    max_turns: int = 90,
) -> tuple[list[str], dict[str, str]]:
    """Refresh the isolated local-model profile and return command + env."""
    binary = agentlab.find_hermes()
    if not binary:
        raise RuntimeError(f"Hermes Agent is not installed. Run: {agentlab.HERMES_INSTALL}")
    workspace = workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace directory does not exist: {workspace}")

    agentlab._write_hermes_config(
        endpoint["base_url"],
        endpoint["model"],
        max_turns,
        studio_url=endpoint.get("studio_url") or "http://127.0.0.1:7860",
        enable_search=True,
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
    return browser_tui_command(binary, endpoint["model"]), env
