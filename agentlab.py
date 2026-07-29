"""Hermes-backed Agent Lab for Spark Studio.

The dashboard already measures inference speed and basic tool calling. Agent
Lab answers the next question: can the served model complete real repository
work through an agent harness?

Hermes runs with a Spark Studio-specific ``HERMES_HOME`` so the user's normal
provider, memories, skills, sessions, and gateway are never overwritten.
Benchmark cases are disposable Git repositories. Free-form runs default to a
detached worktree (or a copied workspace for non-Git folders).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

import db
import vitals

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "agent-lab"
HERMES_HOME = DATA_DIR / "hermes"
WORKSPACES_DIR = DATA_DIR / "workspaces"
RESULTS_DIR = DATA_DIR / "results"
_CONFIG_LOCK = threading.Lock()
_INSTALL_LOCK = threading.Lock()
_PENDING_LOCK = threading.Lock()
_PENDING_SUBSYSTEMS = {"memory", "skills"}
_PENDING_ID = re.compile(r"^[0-9a-f]{8}$")
_PENDING_PREVIEW_LIMIT = 16_000

HERMES_INSTALL = (
    "curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/"
    "main/scripts/install.sh | bash"
)

_DENY_COMMANDS = [
    "git push*",
    "git remote set-url*",
    "sudo *",
    "*curl*|*sh*",
    "*wget*|*sh*",
    "docker system prune*",
]

_LEARNING_DEFAULTS: dict[str, bool] = {
    "memory_enabled": True,
    "user_profile_enabled": True,
    "skills_enabled": True,
    "session_search_enabled": True,
    "write_approval": True,
}


SMOKE_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "fix-stats",
        "title": "Repair a numerical edge case",
        "task": (
            "Fix the implementation so every test passes. Preserve the public "
            "summarize(values) API, handle empty input, and use a true arithmetic "
            "mean. Inspect the repository and run the tests before finishing."
        ),
        "files": {
            "stats.py": '''"""Small statistics helper."""


def summarize(values):
    """Return count, total, and arithmetic mean for *values*."""
    total = sum(values)
    return {
        "count": len(values),
        "total": total,
        "mean": total // len(values),
    }
''',
            "test_stats.py": '''import unittest

from stats import summarize


class SummarizeTests(unittest.TestCase):
    def test_fractional_mean_is_not_truncated(self):
        self.assertEqual(
            summarize([1, 2]),
            {"count": 2, "total": 3, "mean": 1.5},
        )

    def test_empty_input(self):
        self.assertEqual(
            summarize([]),
            {"count": 0, "total": 0, "mean": None},
        )

    def test_regular_values(self):
        self.assertEqual(
            summarize([2, 4, 9]),
            {"count": 3, "total": 15, "mean": 5},
        )


if __name__ == "__main__":
    unittest.main()
''',
        },
    },
    {
        "id": "todo-store",
        "title": "Implement a small persistent feature",
        "task": (
            "Implement the missing TodoStore.add and TodoStore.complete behavior "
            "so every test passes. Keep the JSON file human-readable, preserve "
            "existing tasks, reject an unknown task id without modifying the "
            "file, and run the tests before finishing."
        ),
        "files": {
            "todo_store.py": '''import json
from pathlib import Path


class TodoStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, title):
        raise NotImplementedError

    def complete(self, task_id):
        raise NotImplementedError
''',
            "test_todo_store.py": '''import json
import tempfile
import unittest
from pathlib import Path

from todo_store import TodoStore


class TodoStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tasks.json"
        self.store = TodoStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_preserves_tasks_and_allocates_ids(self):
        first = self.store.add("write tests")
        second = self.store.add("ship feature")
        self.assertEqual(first, {"id": 1, "title": "write tests", "done": False})
        self.assertEqual(second["id"], 2)
        self.assertEqual(len(self.store.load()), 2)
        self.assertTrue(self.path.read_text(encoding="utf-8").endswith("\\n"))

    def test_complete_persists(self):
        task = self.store.add("finish me")
        updated = self.store.complete(task["id"])
        self.assertTrue(updated["done"])
        self.assertTrue(self.store.load()[0]["done"])

    def test_unknown_id_does_not_rewrite_file(self):
        self.store.add("keep me")
        before = self.path.read_text(encoding="utf-8")
        with self.assertRaises(KeyError):
            self.store.complete(999)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_file_is_valid_json(self):
        self.store.add("valid")
        self.assertIsInstance(json.loads(self.path.read_text(encoding="utf-8")), list)


if __name__ == "__main__":
    unittest.main()
''',
        },
    },
)

SUITES = {"coding-smoke": SMOKE_CASES}


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def find_hermes() -> str | None:
    override = os.environ.get("SPARK_STUDIO_HERMES_BIN", "").strip()
    if override and Path(override).is_file():
        return override
    found = shutil.which("hermes")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "hermes"
    return str(candidate) if candidate.is_file() else None


def api_base(url: str) -> str:
    """Normalize an engine URL to the OpenAI-compatible ``.../v1`` base."""
    base = url.strip().rstrip("/").replace("://0.0.0.0", "://127.0.0.1")
    return base if base.endswith("/v1") else f"{base}/v1"


def discover_endpoint(
    studio_url: str = "http://127.0.0.1:7860",
    base_url: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Resolve the active engine and model without changing Hermes config."""
    studio = studio_url.rstrip("/")
    active = None
    with httpx.Client(timeout=10) as client:
        if not base_url:
            try:
                response = client.get(f"{studio}/api/active")
                response.raise_for_status()
                active = response.json()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"cannot reach Spark Studio at {studio}: {exc}") from exc
            if not active or not active.get("url"):
                raise RuntimeError("Spark Studio has no active model endpoint")
            if active.get("ready") is False:
                raise RuntimeError("the active model is still loading; wait until it is ready")
            base_url = active["url"]

        base = api_base(base_url)
        try:
            response = client.get(f"{base}/models")
            response.raise_for_status()
            models = response.json().get("data") or []
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cannot reach the model endpoint at {base}: {exc}") from exc

    if not model and models:
        model = models[0].get("id")
    if not model:
        raise RuntimeError("the endpoint did not report a model id; pass --model")
    entry = next((m for m in models if m.get("id") == model), models[0] if models else {})
    context = entry.get("max_model_len") or entry.get("max_context_length")
    return {
        "studio_url": studio,
        "base_url": base,
        "model": model,
        "context_length": context,
        "active_run": active,
        "models": [m.get("id") for m in models if m.get("id")],
    }


def hermes_status(endpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    binary = find_hermes()
    status: dict[str, Any] = {
        "installed": bool(binary),
        "binary": binary,
        "install_command": HERMES_INSTALL,
        "profile": str(HERMES_HOME),
        "endpoint": endpoint,
    }
    if not binary:
        return status
    result = _run([binary, "--version"], timeout=15)
    status["version"] = (result.stdout or result.stderr).strip()
    status["ok"] = result.returncode == 0
    return status


def install_hermes() -> dict[str, Any]:
    """Install the official Hermes Agent CLI for the current user.

    The command is intentionally fixed to Hermes' official installer and the
    caller is expected to enforce Spark Studio's local/private-HTTPS access
    policy. A process-wide lock prevents double-clicks from running two
    installers against the same user profile.
    """
    with _INSTALL_LOCK:
        current = hermes_status()
        if current["installed"] and current.get("ok", True):
            return {**current, "already_installed": True, "install_output": ""}
        curl = shutil.which("curl")
        bash = shutil.which("bash")
        if not curl or not bash:
            missing = " and ".join(
                name for name, path in (("curl", curl), ("bash", bash)) if not path
            )
            return {**current, "error": f"{missing} is required to install Hermes"}
        try:
            result = _run([bash, "-lc", HERMES_INSTALL], timeout=900)
            output = (result.stdout or result.stderr or "").strip()[-6000:]
        except subprocess.TimeoutExpired:
            return {
                **current,
                "error": "Hermes installation timed out after 15 minutes",
            }
        except (OSError, subprocess.SubprocessError) as exc:
            return {**current, "error": f"could not run Hermes installer: {exc}"}
        refreshed = hermes_status()
        response = {
            **refreshed,
            "already_installed": False,
            "install_output": output,
        }
        if result.returncode != 0 or not refreshed["installed"]:
            response["error"] = output or (
                f"Hermes installer exited with code {result.returncode}"
            )
        return response


def _learning_settings_path() -> Path:
    return HERMES_HOME / "learning.json"


def _load_learning_settings_unlocked() -> dict[str, bool]:
    settings = dict(_LEARNING_DEFAULTS)
    try:
        saved = json.loads(_learning_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        saved = None
    if isinstance(saved, dict):
        for key in settings:
            if isinstance(saved.get(key), bool):
                settings[key] = saved[key]
        return settings

    # One-time compatibility path for profiles configured manually before the
    # Learning UI existed. Preserve standard Hermes memory/approval choices.
    try:
        config = yaml.safe_load(
            (HERMES_HOME / "config.yaml").read_text(encoding="utf-8")
        ) or {}
    except (OSError, yaml.YAMLError):
        return settings
    memory = config.get("memory") if isinstance(config, dict) else None
    if isinstance(memory, dict):
        for key in ("memory_enabled", "user_profile_enabled"):
            if isinstance(memory.get(key), bool):
                settings[key] = memory[key]
        if isinstance(memory.get("write_approval"), bool):
            settings["write_approval"] = memory["write_approval"]
    skills = config.get("skills") if isinstance(config, dict) else None
    if isinstance(skills, dict) and isinstance(skills.get("write_approval"), bool):
        settings["write_approval"] = skills["write_approval"]
    return settings


def hermes_learning_settings() -> dict[str, bool]:
    """Return Spark Studio's isolated Hermes learning preferences."""
    with _CONFIG_LOCK:
        return _load_learning_settings_unlocked()


def _apply_learning_config(
    config: dict[str, Any], settings: dict[str, bool]
) -> None:
    memory = config.get("memory") if isinstance(config.get("memory"), dict) else {}
    memory.update(
        {
            "memory_enabled": settings["memory_enabled"],
            "user_profile_enabled": settings["user_profile_enabled"],
            "write_approval": settings["write_approval"],
        }
    )
    config["memory"] = memory
    skills = config.get("skills") if isinstance(config.get("skills"), dict) else {}
    skills["write_approval"] = settings["write_approval"]
    external_dirs = skills.get("external_dirs")
    if not isinstance(external_dirs, list):
        external_dirs = []
    studio_skills = str(APP_DIR / "agent-skills")
    if studio_skills not in external_dirs:
        external_dirs.append(studio_skills)
    skills["external_dirs"] = external_dirs
    config["skills"] = skills


def update_hermes_learning_settings(changes: dict[str, Any]) -> dict[str, bool]:
    """Persist learning preferences without touching the personal Hermes profile."""
    with _CONFIG_LOCK:
        settings = _load_learning_settings_unlocked()
        for key in settings:
            if key in changes:
                if not isinstance(changes[key], bool):
                    raise ValueError(f"{key} must be true or false")
                settings[key] = changes[key]
        HERMES_HOME.mkdir(parents=True, exist_ok=True)
        settings_path = _learning_settings_path()
        temporary = HERMES_HOME / f"learning-{threading.get_ident()}.tmp"
        temporary.write_text(
            json.dumps(settings, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(settings_path)

        config_path = HERMES_HOME / "config.yaml"
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            config = {}
        if isinstance(config, dict) and config:
            _apply_learning_config(config, settings)
            config_tmp = HERMES_HOME / f"config-{threading.get_ident()}.tmp"
            config_tmp.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            config_tmp.replace(config_path)
        return settings


def hermes_interactive_toolsets(
    settings: dict[str, bool] | None = None,
) -> str:
    """Toolsets shared by dashboard Chat and ``sparkstudio hermes``."""
    selected = settings or hermes_learning_settings()
    toolsets = ["file", "terminal", "mcp-sparkstudio"]
    if selected["memory_enabled"] or selected["user_profile_enabled"]:
        toolsets.append("memory")
    if selected["skills_enabled"]:
        toolsets.append("skills")
    if selected["session_search_enabled"]:
        toolsets.append("session_search")
    return ",".join(toolsets)


def hermes_learning_status() -> dict[str, Any]:
    settings = hermes_learning_settings()
    return {
        **settings,
        "profile": str(HERMES_HOME),
        "toolsets": hermes_interactive_toolsets(settings).split(","),
    }


def _validate_pending_target(subsystem: str, pending_id: str) -> None:
    if subsystem not in _PENDING_SUBSYSTEMS:
        raise ValueError("subsystem must be memory or skills")
    if not _PENDING_ID.fullmatch(pending_id):
        raise ValueError("invalid pending write id")


def _pending_preview(record: dict[str, Any]) -> tuple[str, bool]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    content = payload.get("content")
    if not isinstance(content, str):
        content = payload.get("file_content")
    if not isinstance(content, str):
        content = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    old_text = payload.get("old_text")
    if isinstance(old_text, str) and old_text.strip():
        content = f"Replace:\n{old_text}\n\nWith:\n{content}"
    truncated = len(content) > _PENDING_PREVIEW_LIMIT
    return content[:_PENDING_PREVIEW_LIMIT], truncated


def hermes_pending_writes() -> list[dict[str, Any]]:
    """List staged writes in Spark Studio's isolated Hermes profile."""
    records: list[dict[str, Any]] = []
    with _PENDING_LOCK:
        for subsystem in sorted(_PENDING_SUBSYSTEMS):
            directory = HERMES_HOME / "pending" / subsystem
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                pending_id = record.get("id")
                if (
                    not isinstance(pending_id, str)
                    or not _PENDING_ID.fullmatch(pending_id)
                    or path.stem != pending_id
                    or record.get("subsystem") != subsystem
                ):
                    continue
                payload = record.get("payload")
                payload = payload if isinstance(payload, dict) else {}
                preview, truncated = _pending_preview(record)
                try:
                    created_at = float(record.get("created_at") or 0)
                except (TypeError, ValueError):
                    created_at = 0.0
                records.append(
                    {
                        "id": pending_id,
                        "subsystem": subsystem,
                        "action": str(record.get("action") or payload.get("action") or ""),
                        "summary": str(record.get("summary") or ""),
                        "origin": str(record.get("origin") or "foreground"),
                        "created_at": created_at,
                        "target": str(payload.get("target") or ""),
                        "name": str(payload.get("name") or ""),
                        "file_path": str(payload.get("file_path") or ""),
                        "preview": preview,
                        "truncated": truncated,
                    }
                )
    records.sort(key=lambda item: item["created_at"])
    return records


def _hermes_agent_runtime() -> tuple[Path, Path]:
    override = os.environ.get("SPARK_STUDIO_HERMES_AGENT_DIR", "").strip()
    candidates = [Path(override)] if override else []
    candidates.append(Path.home() / ".hermes" / "hermes-agent")
    for source in candidates:
        python = source / "venv" / "bin" / "python"
        if python.is_file() and (source / "hermes_cli" / "write_approval_commands.py").is_file():
            return source, python
    raise RuntimeError(
        "Hermes approval runtime was not found; reinstall Hermes from the Chat tab"
    )


_HERMES_APPROVAL_BRIDGE = """
import json
import sys
from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import write_approval as wa

subsystem, action, pending_id = sys.argv[1:4]
memory_store = None
if subsystem == wa.MEMORY and action == "approve":
    from tools.memory_tool import load_on_disk_store
    memory_store = load_on_disk_store()
message = handle_pending_subcommand(
    subsystem, [action, pending_id], memory_store=memory_store
)
remaining = wa.get_pending(subsystem, pending_id) is not None
print(json.dumps({"ok": not remaining, "message": message or ""}))
""".strip()


def resolve_hermes_pending(
    subsystem: str, pending_id: str, action: str
) -> dict[str, Any]:
    """Approve or reject one staged write through Hermes's official handler."""
    _validate_pending_target(subsystem, pending_id)
    if action not in {"approve", "reject"}:
        raise ValueError("action must be approve or reject")
    source, python = _hermes_agent_runtime()
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["HERMES_HOME"] = str(HERMES_HOME)
    with _PENDING_LOCK:
        pending_path = HERMES_HOME / "pending" / subsystem / f"{pending_id}.json"
        if not pending_path.is_file():
            raise FileNotFoundError(f"pending {subsystem} write {pending_id} was not found")
        try:
            result = _run(
                [
                    str(python),
                    "-c",
                    _HERMES_APPROVAL_BRIDGE,
                    subsystem,
                    action,
                    pending_id,
                ],
                cwd=source,
                env=env,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Hermes could not {action} the pending write: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown Hermes error").strip()
        raise RuntimeError(f"Hermes could not {action} the pending write: {detail[-2000:]}")
    try:
        response = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Hermes returned an invalid {action} response"
        ) from exc
    if not response.get("ok"):
        raise RuntimeError(
            str(response.get("message") or f"Hermes could not {action} the pending write")
        )
    return {
        "ok": True,
        "action": action,
        "id": pending_id,
        "subsystem": subsystem,
        "message": str(response.get("message") or ""),
        "restart_recommended": action == "approve",
    }


def _write_hermes_config(
    base_url: str,
    model: str,
    max_turns: int,
    *,
    studio_url: str = "http://127.0.0.1:7860",
    enable_search: bool = False,
) -> Path:
    HERMES_HOME.mkdir(parents=True, exist_ok=True)
    path = HERMES_HOME / "config.yaml"
    with _CONFIG_LOCK:
        learning = _load_learning_settings_unlocked()
        # Spark Studio owns the model/tool configuration in this isolated
        # profile, while Hermes Mod owns display.skin. Preserve that user-facing
        # selection whenever Chat refreshes the active model endpoint.
        preserved_display = None
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(existing, dict) and isinstance(existing.get("display"), dict):
                preserved_display = existing["display"]
        except (OSError, yaml.YAMLError):
            pass

        config = {
            "model": {
                "provider": "custom",
                "default": model,
                "base_url": api_base(base_url),
                "api_key": "",
            },
            "terminal": {"backend": "local"},
            "approvals": {
                "mode": "smart",
                "timeout": 10,
                "deny": list(_DENY_COMMANDS),
            },
            "agent": {"max_turns": max_turns},
        }
        _apply_learning_config(config, learning)
        if preserved_display:
            config["display"] = preserved_display
        if enable_search:
            config["mcp_servers"] = {
                "sparkstudio": {
                    "command": sys.executable,
                    "args": [
                        str(APP_DIR / "sparkstudio_mcp.py"),
                        "--studio-url",
                        studio_url.rstrip("/"),
                    ],
                    "enabled": True,
                    "timeout": 75,
                    "connect_timeout": 20,
                    "supports_parallel_tool_calls": True,
                    "tools": {
                        "include": ["web_search"],
                        "resources": False,
                        "prompts": False,
                    },
                }
            }
        payload = yaml.safe_dump(config, sort_keys=False)
        temporary = HERMES_HOME / f"config-{threading.get_ident()}.tmp"
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    return path


def active_hermes_skin() -> str:
    """Return the selected skin in Spark Studio's isolated Hermes profile."""
    path = HERMES_HOME / "config.yaml"
    with _CONFIG_LOCK:
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return "default"
    if not isinstance(config, dict):
        return "default"
    display = config.get("display")
    if not isinstance(display, dict):
        return "default"
    skin = display.get("skin")
    return str(skin).strip() if skin else "default"


def build_hermes_command(
    binary: str,
    task: str,
    model: str,
    *,
    max_turns: int = 90,
    unsafe_yolo: bool = False,
) -> list[str]:
    command = [
        binary,
        "chat",
        "--quiet",
        "--query",
        task,
        "--model",
        model,
        "--toolsets",
        "file,terminal",
        "--checkpoints",
        "--source",
        "tool",
        "--max-turns",
        str(max_turns),
    ]
    if unsafe_yolo:
        command.append("--yolo")
    return command


def build_hermes_interactive_command(
    binary: str,
    model: str,
    *,
    max_turns: int = 90,
    unsafe_yolo: bool = False,
) -> list[str]:
    command = [
        binary,
        "chat",
        "--model",
        model,
        "--toolsets",
        hermes_interactive_toolsets(),
        "--checkpoints",
        "--max-turns",
        str(max_turns),
    ]
    if unsafe_yolo:
        command.append("--yolo")
    return command


def launch_hermes(
    endpoint: dict[str, Any],
    repo: Path,
    *,
    max_turns: int = 90,
    unsafe_yolo: bool = False,
) -> int:
    """Launch interactive Hermes against Spark Studio's current model."""
    binary = find_hermes()
    if not binary:
        raise RuntimeError(f"Hermes Agent is not installed. Run: {HERMES_INSTALL}")
    workspace = repo.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"repository directory does not exist: {workspace}")
    _write_hermes_config(
        endpoint["base_url"],
        endpoint["model"],
        max_turns,
        studio_url=endpoint.get("studio_url") or "http://127.0.0.1:7860",
        enable_search=True,
    )
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(HERMES_HOME),
            "HERMES_WRITE_SAFE_ROOT": os.pathsep.join((str(workspace), str(HERMES_HOME))),
        }
    )
    command = build_hermes_interactive_command(
        binary,
        endpoint["model"],
        max_turns=max_turns,
        unsafe_yolo=unsafe_yolo,
    )
    completed = subprocess.run(command, cwd=str(workspace), env=env, check=False)
    return completed.returncode


class _TelemetrySampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def collect() -> None:
            while not self._stop.is_set():
                try:
                    self.samples.append(vitals.snapshot())
                except Exception:  # noqa: BLE001
                    pass
                self._stop.wait(2)

        self._thread = threading.Thread(target=collect, name="agentlab-vitals", daemon=True)
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        numeric = ("gpu_util", "gpu_power", "mem_used_gb", "cpu_pct")
        summary: dict[str, Any] = {"samples": len(self.samples)}
        for key in numeric:
            values = [float(sample[key]) for sample in self.samples if sample.get(key) is not None]
            if values:
                summary[f"{key}_avg"] = round(sum(values) / len(values), 2)
                summary[f"{key}_peak"] = round(max(values), 2)
        return summary


def _agent_prompt(task: str, evaluation: bool = False) -> str:
    preface = (
        "You are running inside Spark Studio Agent Lab. Work only inside the "
        "current repository. Do not use the network, sudo, Docker, Git remotes, "
        "or modify files outside this repository. Inspect the existing code, "
        "make the smallest correct changes, and run the repository's tests. "
        "Do not commit or push. "
    )
    if evaluation:
        preface += (
            "This is an unattended deterministic evaluation. Do not ask questions; "
            "use the task and tests as the complete specification. "
        )
    return f"{preface}\n\nTASK:\n{task}"


def _invoke_hermes(
    endpoint: dict[str, Any],
    workspace: Path,
    task: str,
    *,
    max_turns: int,
    timeout: float,
    unsafe_yolo: bool,
    evaluation: bool,
) -> dict[str, Any]:
    binary = find_hermes()
    if not binary:
        raise RuntimeError(f"Hermes Agent is not installed. Run: {HERMES_INSTALL}")
    _write_hermes_config(endpoint["base_url"], endpoint["model"], max_turns)
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(HERMES_HOME),
            "HERMES_WRITE_SAFE_ROOT": os.pathsep.join((str(workspace), str(HERMES_HOME))),
            "NO_COLOR": "1",
        }
    )
    command = build_hermes_command(
        binary,
        _agent_prompt(task, evaluation=evaluation),
        endpoint["model"],
        max_turns=max_turns,
        unsafe_yolo=unsafe_yolo,
    )
    sampler = _TelemetrySampler()
    started = time.time()
    sampler.start()
    try:
        completed = _run(command, cwd=workspace, env=env, timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(
            command,
            124,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
        )
        timed_out = True
    telemetry = sampler.finish()
    redacted_command = list(command)
    redacted_command[0] = Path(redacted_command[0]).name
    redacted_command[4] = "<task>"
    return {
        "exit_code": completed.returncode,
        "timed_out": timed_out,
        "duration_seconds": round(time.time() - started, 2),
        "response": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
        "command": redacted_command,
        "telemetry": telemetry,
    }


def _git_root(path: Path) -> Path | None:
    result = _run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], timeout=15)
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _prepare_user_workspace(repo: Path, run_id: str, in_place: bool) -> tuple[Path, str]:
    repo = repo.resolve()
    if not repo.is_dir():
        raise ValueError(f"repository directory does not exist: {repo}")
    root = _git_root(repo)
    if in_place:
        return (root or repo), "in-place"
    destination = WORKSPACES_DIR / run_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    if root:
        result = _run(
            ["git", "-C", str(root), "worktree", "add", "--detach", str(destination), "HEAD"],
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"could not create Git worktree: {result.stderr.strip()}")
        return destination, "git-worktree"
    shutil.copytree(
        repo,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "env", "node_modules"),
    )
    return destination, "copy"


def _git_evidence(workspace: Path) -> dict[str, str]:
    root = _git_root(workspace)
    if not root:
        return {"git_status": "", "diff": ""}
    status = _run(["git", "-C", str(root), "status", "--short"], timeout=20)
    diff = _run(["git", "-C", str(root), "diff", "--no-ext-diff", "HEAD"], timeout=30)
    return {"git_status": status.stdout, "diff": diff.stdout}


def _persist(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    saved = dict(row)
    saved["result_json"] = json.dumps(result)
    db.agentlab_upsert(saved)
    return result


def run_agent(
    endpoint: dict[str, Any],
    repo: Path,
    task: str,
    *,
    in_place: bool = False,
    max_turns: int = 90,
    timeout: float = 1800,
    unsafe_yolo: bool = False,
) -> dict[str, Any]:
    run_id = f"agent-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    workspace, isolation = _prepare_user_workspace(repo, run_id, in_place)
    started = int(time.time())
    row = {
        "id": run_id,
        "mode": "run",
        "harness": "hermes",
        "run_id": (endpoint.get("active_run") or {}).get("id"),
        "model": endpoint["model"],
        "base_url": endpoint["base_url"],
        "repo_path": str(workspace),
        "task": task,
        "suite": None,
        "status": "running",
        "created_at": started,
    }
    db.agentlab_upsert({**row, "result_json": "{}"})
    try:
        hermes_result = _invoke_hermes(
            endpoint,
            workspace,
            task,
            max_turns=max_turns,
            timeout=timeout,
            unsafe_yolo=unsafe_yolo,
            evaluation=False,
        )
        result = {
            "id": run_id,
            "mode": "run",
            "harness": "hermes",
            "status": "completed" if hermes_result["exit_code"] == 0 else "failed",
            "model": endpoint["model"],
            "base_url": endpoint["base_url"],
            "task": task,
            "workspace": str(workspace),
            "isolation": isolation,
            "hermes": hermes_result,
            **_git_evidence(workspace),
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "id": run_id,
            "mode": "run",
            "harness": "hermes",
            "status": "failed",
            "model": endpoint["model"],
            "base_url": endpoint["base_url"],
            "task": task,
            "workspace": str(workspace),
            "isolation": isolation,
            "error": str(exc),
        }
    row.update(
        status=result["status"],
        duration_seconds=(result.get("hermes") or {}).get("duration_seconds"),
    )
    report = _write_report(result)
    result["report_path"] = str(report)
    return _persist(row, result)


def _initialize_case(case: dict[str, Any], workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    for relative, content in case["files"].items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _run(["git", "init", "-q"], cwd=workspace)
    _run(["git", "add", "."], cwd=workspace)
    commit = _run(
        [
            "git",
            "-c",
            "user.name=Spark Studio",
            "-c",
            "user.email=agentlab@localhost",
            "commit",
            "-qm",
            "Agent Lab fixture",
        ],
        cwd=workspace,
    )
    if commit.returncode != 0:
        raise RuntimeError(f"could not initialize evaluation fixture: {commit.stderr}")


def verify_workspace(workspace: Path) -> dict[str, Any]:
    started = time.time()
    result = _run([sys.executable, "-m", "unittest", "-q"], cwd=workspace, timeout=120)
    return {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "duration_seconds": round(time.time() - started, 2),
        "output": ((result.stdout or "") + (result.stderr or "")).strip()[-8000:],
    }


def _evaluate_one(
    endpoint: dict[str, Any],
    case: dict[str, Any],
    evaluation_id: str,
    trial: int,
    *,
    max_turns: int,
    timeout: float,
    unsafe_yolo: bool,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    label = f"{case['id']}-t{trial}"
    workspace = WORKSPACES_DIR / evaluation_id / label
    _initialize_case(case, workspace)
    baseline = verify_workspace(workspace)
    if baseline["passed"]:
        raise RuntimeError(f"invalid fixture {case['id']}: tests pass before the agent runs")
    if progress:
        progress(f"[{label}] Hermes started")
    hermes_result = _invoke_hermes(
        endpoint,
        workspace,
        case["task"],
        max_turns=max_turns,
        timeout=timeout,
        unsafe_yolo=unsafe_yolo,
        evaluation=True,
    )
    verification = verify_workspace(workspace)
    result = {
        "case": case["id"],
        "title": case["title"],
        "trial": trial,
        "passed": verification["passed"],
        "workspace": str(workspace),
        "baseline": baseline,
        "verification": verification,
        "hermes": hermes_result,
        **_git_evidence(workspace),
    }
    if progress:
        progress(f"[{label}] {'PASS' if result['passed'] else 'FAIL'}")
    return result


def evaluate(
    endpoint: dict[str, Any],
    *,
    suite: str = "coding-smoke",
    case_ids: list[str] | None = None,
    trials: int = 1,
    jobs: int = 1,
    max_turns: int = 90,
    timeout: float = 1800,
    unsafe_yolo: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choices: {', '.join(SUITES)}")
    cases = list(SUITES[suite])
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown case(s): {', '.join(sorted(missing))}")
    if not cases:
        raise ValueError("no evaluation cases selected")
    if trials < 1 or trials > 10:
        raise ValueError("trials must be between 1 and 10")
    if jobs < 1 or jobs > 8:
        raise ValueError("jobs must be between 1 and 8")

    evaluation_id = f"eval-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    started_at = time.time()
    total = len(cases) * trials
    row = {
        "id": evaluation_id,
        "mode": "eval",
        "harness": "hermes",
        "run_id": (endpoint.get("active_run") or {}).get("id"),
        "model": endpoint["model"],
        "base_url": endpoint["base_url"],
        "repo_path": str(WORKSPACES_DIR / evaluation_id),
        "task": None,
        "suite": suite,
        "status": "running",
        "created_at": int(started_at),
        "total": total,
        "result_json": "{}",
    }
    db.agentlab_upsert(row)
    work = [(case, trial) for trial in range(1, trials + 1) for case in cases]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(jobs, len(work))) as pool:
        futures = {
            pool.submit(
                _evaluate_one,
                endpoint,
                case,
                evaluation_id,
                trial,
                max_turns=max_turns,
                timeout=timeout,
                unsafe_yolo=unsafe_yolo,
                progress=progress,
            ): (case["id"], trial)
            for case, trial in work
        }
        for future in as_completed(futures):
            case_id, trial = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"case": case_id, "trial": trial, "passed": False, "error": str(exc)})
                if progress:
                    progress(f"[{case_id}-t{trial}] ERROR: {exc}")

    results.sort(key=lambda item: (item.get("trial", 0), item.get("case", "")))
    passed = sum(1 for item in results if item.get("passed"))
    duration = round(time.time() - started_at, 2)
    result = {
        "id": evaluation_id,
        "mode": "eval",
        "harness": "hermes",
        "status": "completed",
        "suite": suite,
        "model": endpoint["model"],
        "base_url": endpoint["base_url"],
        "score": round(100 * passed / total, 1),
        "passed": passed,
        "total": total,
        "trials": trials,
        "jobs": jobs,
        "duration_seconds": duration,
        "cases": results,
        "workspace": str(WORKSPACES_DIR / evaluation_id),
    }
    row.update(
        status="completed",
        score=result["score"],
        passed=passed,
        total=total,
        duration_seconds=duration,
    )
    report = _write_report(result)
    result["report_path"] = str(report)
    return _persist(row, result)


def _write_report(result: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_id = result["id"]
    (RESULTS_DIR / f"{result_id}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    md_path = RESULTS_DIR / f"{result_id}.md"
    lines = [
        f"# Spark Studio Agent Lab — {result_id}",
        "",
        "- Harness: Hermes Agent",
        f"- Model: `{result.get('model', 'unknown')}`",
        f"- Endpoint: `{result.get('base_url', 'unknown')}`",
        f"- Status: **{result.get('status', 'unknown')}**",
    ]
    if result.get("mode") == "eval":
        lines += [
            f"- Suite: `{result.get('suite')}`",
            f"- Score: **{result.get('score')} / 100** ({result.get('passed')}/{result.get('total')} passed)",
            f"- Duration: {result.get('duration_seconds')}s",
            "",
            "| Case | Trial | Result | Agent time | Test output |",
            "|---|---:|---|---:|---|",
        ]
        for case in result.get("cases") or []:
            verification = case.get("verification") or {}
            hermes_result = case.get("hermes") or {}
            output = str(verification.get("output") or case.get("error") or "").replace("\n", " ")
            lines.append(
                f"| {case.get('case')} | {case.get('trial')} | "
                f"{'PASS' if case.get('passed') else 'FAIL'} | "
                f"{hermes_result.get('duration_seconds', '—')}s | {output[:180]} |"
            )
    else:
        lines += [
            f"- Workspace: `{result.get('workspace')}`",
            f"- Isolation: `{result.get('isolation')}`",
            "",
            "## Task",
            "",
            result.get("task") or "",
            "",
            "## Hermes response",
            "",
            (result.get("hermes") or {}).get("response") or result.get("error") or "",
            "",
            "## Git status",
            "",
            "```text",
            (result.get("git_status") or "").strip(),
            "```",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def history(limit: int = 50) -> list[dict[str, Any]]:
    rows = db.agentlab_list(limit)
    for row in rows:
        try:
            row["result"] = json.loads(row.pop("result_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            row["result"] = {}
    return rows


def get_result(run_id: str) -> dict[str, Any] | None:
    row = db.agentlab_get(run_id)
    if not row:
        return None
    try:
        row["result"] = json.loads(row.pop("result_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        row["result"] = {}
    return row
