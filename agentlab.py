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


def _write_hermes_config(
    base_url: str,
    model: str,
    max_turns: int,
    *,
    studio_url: str = "http://127.0.0.1:7860",
    enable_search: bool = False,
) -> Path:
    HERMES_HOME.mkdir(parents=True, exist_ok=True)
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
    path = HERMES_HOME / "config.yaml"
    payload = yaml.safe_dump(config, sort_keys=False)
    with _CONFIG_LOCK:
        temporary = HERMES_HOME / f"config-{threading.get_ident()}.tmp"
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    return path


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
        "file,terminal,mcp-sparkstudio",
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
