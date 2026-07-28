import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import agentlab
import sparkstudio_cli


class AgentLabConfigurationTests(unittest.TestCase):
    def test_api_base_normalizes_local_endpoint(self):
        self.assertEqual(
            agentlab.api_base("http://0.0.0.0:8000/"),
            "http://127.0.0.1:8000/v1",
        )
        self.assertEqual(
            agentlab.api_base("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/v1",
        )

    def test_hermes_command_uses_constrained_toolset(self):
        command = agentlab.build_hermes_command(
            "/usr/bin/hermes", "do the task", "local-model", max_turns=42
        )
        self.assertEqual(command[:2], ["/usr/bin/hermes", "chat"])
        self.assertIn("file,terminal", command)
        self.assertIn("--checkpoints", command)
        self.assertEqual(command[command.index("--max-turns") + 1], "42")
        self.assertNotIn("--yolo", command)

    def test_yolo_requires_explicit_opt_in(self):
        command = agentlab.build_hermes_command(
            "hermes", "task", "model", unsafe_yolo=True
        )
        self.assertIn("--yolo", command)

    def test_config_is_written_to_dedicated_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            hermes_home = Path(tmp) / "hermes-profile"
            with mock.patch.object(agentlab, "HERMES_HOME", hermes_home):
                path = agentlab._write_hermes_config(
                    "http://0.0.0.0:8000", "fixture-model", 17
                )
            config = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(path, hermes_home / "config.yaml")
        self.assertEqual(config["model"]["provider"], "custom")
        self.assertEqual(config["model"]["base_url"], "http://127.0.0.1:8000/v1")
        self.assertEqual(config["model"]["default"], "fixture-model")
        self.assertEqual(config["approvals"]["mode"], "smart")
        self.assertTrue(config["approvals"]["deny"])
        self.assertEqual(config["agent"]["max_turns"], 17)


class AgentLabFixtureTests(unittest.TestCase):
    def test_all_smoke_cases_start_with_failing_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case in agentlab.SUITES["coding-smoke"]:
                with self.subTest(case=case["id"]):
                    workspace = root / case["id"]
                    agentlab._initialize_case(case, workspace)
                    verification = agentlab.verify_workspace(workspace)
                    self.assertFalse(verification["passed"])
                    self.assertNotEqual(verification["exit_code"], 0)

    def test_evaluation_scores_verified_workspace_changes(self):
        stats_solution = '''"""Small statistics helper."""


def summarize(values):
    total = sum(values)
    return {
        "count": len(values),
        "total": total,
        "mean": total / len(values) if values else None,
    }
'''
        todo_solution = '''import json
from pathlib import Path


class TodoStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, tasks):
        self.path.write_text(json.dumps(tasks, indent=2) + "\\n", encoding="utf-8")

    def add(self, title):
        tasks = self.load()
        task = {
            "id": max((item["id"] for item in tasks), default=0) + 1,
            "title": title,
            "done": False,
        }
        tasks.append(task)
        self._save(tasks)
        return task

    def complete(self, task_id):
        tasks = self.load()
        task = next((item for item in tasks if item["id"] == task_id), None)
        if task is None:
            raise KeyError(task_id)
        task["done"] = True
        self._save(tasks)
        return task
'''

        def fake_hermes(endpoint, workspace, task, **kwargs):
            if (workspace / "stats.py").exists():
                (workspace / "stats.py").write_text(stats_solution, encoding="utf-8")
            else:
                (workspace / "todo_store.py").write_text(todo_solution, encoding="utf-8")
            return {
                "exit_code": 0,
                "timed_out": False,
                "duration_seconds": 0.01,
                "response": "done",
                "stderr": "",
                "command": ["hermes"],
                "telemetry": {"samples": 0},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch.object(agentlab, "WORKSPACES_DIR", root / "workspaces"),
                mock.patch.object(agentlab, "RESULTS_DIR", root / "results"),
                mock.patch.object(agentlab, "_invoke_hermes", side_effect=fake_hermes),
                mock.patch.object(agentlab.db, "agentlab_upsert", return_value={}),
            ):
                result = agentlab.evaluate(
                    {"base_url": "http://127.0.0.1:8000/v1", "model": "fake"},
                    jobs=2,
                )
                self.assertEqual(result["score"], 100.0)
                self.assertEqual(result["passed"], 2)
                self.assertTrue(Path(result["report_path"]).is_file())


class CliParserTests(unittest.TestCase):
    def test_agent_eval_options_parse(self):
        args = sparkstudio_cli.build_parser().parse_args(
            ["--model", "local", "agent", "eval", "--jobs", "2", "--trials", "3"]
        )
        self.assertEqual(args.command, "agent")
        self.assertEqual(args.agent_command, "eval")
        self.assertEqual(args.jobs, 2)
        self.assertEqual(args.trials, 3)


if __name__ == "__main__":
    unittest.main()
