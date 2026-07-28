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

    def test_interactive_command_uses_current_model_and_no_query(self):
        command = agentlab.build_hermes_interactive_command(
            "hermes", "current-model", max_turns=55
        )
        self.assertEqual(command[:2], ["hermes", "chat"])
        self.assertEqual(command[command.index("--model") + 1], "current-model")
        self.assertEqual(command[command.index("--max-turns") + 1], "55")
        self.assertIn("file,terminal,mcp-sparkstudio", command)
        self.assertIn("--checkpoints", command)
        self.assertNotIn("--query", command)
        self.assertNotIn("--yolo", command)

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
        self.assertNotIn("mcp_servers", config)

    def test_config_refresh_preserves_skin_studio_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            hermes_home = Path(tmp) / "hermes-profile"
            hermes_home.mkdir()
            (hermes_home / "config.yaml").write_text(
                "model:\n  default: old-model\ndisplay:\n  skin: ares\n  compact: true\n",
                encoding="utf-8",
            )
            with mock.patch.object(agentlab, "HERMES_HOME", hermes_home):
                path = agentlab._write_hermes_config(
                    "http://127.0.0.1:9000", "new-model", 19
                )
                active_skin = agentlab.active_hermes_skin()
            config = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(config["model"]["default"], "new-model")
        self.assertEqual(config["display"], {"skin": "ares", "compact": True})
        self.assertEqual(active_skin, "ares")

    def test_interactive_config_exposes_only_sparkstudio_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            hermes_home = Path(tmp) / "hermes-profile"
            with mock.patch.object(agentlab, "HERMES_HOME", hermes_home):
                path = agentlab._write_hermes_config(
                    "http://127.0.0.1:8000/v1",
                    "fixture-model",
                    25,
                    studio_url="http://127.0.0.1:7860/",
                    enable_search=True,
                )
            config = yaml.safe_load(path.read_text(encoding="utf-8"))

        server = config["mcp_servers"]["sparkstudio"]
        self.assertEqual(server["args"][-2:], ["--studio-url", "http://127.0.0.1:7860"])
        self.assertEqual(server["tools"]["include"], ["web_search"])
        self.assertFalse(server["tools"]["resources"])
        self.assertFalse(server["tools"]["prompts"])


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

    def test_interactive_hermes_aliases_parse(self):
        parser = sparkstudio_cli.build_parser()
        direct = parser.parse_args(["hermes", "--repo", "/tmp/project"])
        nested = parser.parse_args(["agent", "chat", "--max-turns", "25"])
        self.assertIs(direct.func, sparkstudio_cli.cmd_agent_chat)
        self.assertEqual(direct.repo, "/tmp/project")
        self.assertIs(nested.func, sparkstudio_cli.cmd_agent_chat)
        self.assertEqual(nested.max_turns, 25)

    def test_search_options_parse(self):
        args = sparkstudio_cli.build_parser().parse_args(
            ["search", "DGX Spark news", "--limit", "7", "--enrich"]
        )
        self.assertIs(args.func, sparkstudio_cli.cmd_search)
        self.assertEqual(args.query, "DGX Spark news")
        self.assertEqual(args.limit, 7)
        self.assertTrue(args.enrich)


if __name__ == "__main__":
    unittest.main()
