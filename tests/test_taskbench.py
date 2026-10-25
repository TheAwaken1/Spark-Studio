import ast
import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import taskbench
import sparkstudio_cli


class TaskBenchFixtureTests(unittest.TestCase):
    def test_file_audit_is_scored_against_exact_seeded_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "file-audit"
            taskbench.initialize_case("file-audit", workspace)
            project = workspace / "sample-project"
            rows = []
            for path in sorted(project.rglob("*.py")):
                source = path.read_text(encoding="utf-8")
                names = [
                    node.name
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                rows.append(
                    {
                        "file": path.relative_to(project).as_posix(),
                        "function_names": ";".join(sorted(names)),
                        "line_count": str(len(source.splitlines())),
                        "first_10_sha256": taskbench.first_ten_sha256(source),
                    }
                )
            with (workspace / "audit.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            passed = taskbench.verify_case("file-audit", workspace)
            rows[0]["line_count"] = "999"
            with (workspace / "audit.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            failed = taskbench.verify_case("file-audit", workspace)

        self.assertTrue(passed["passed"], passed)
        self.assertFalse(failed["passed"])
        self.assertIn("does not match", failed["detail"])

    def test_news_research_requires_three_cited_structured_stories(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "news-research"
            taskbench.initialize_case("news-research", workspace)
            stories = [
                {
                    "headline": f"AI headline {index}",
                    "summary": "A sourced summary of the reported development. " * 3,
                    "source_url": f"https://example.com/ai-news-{index}",
                    "published_date": "2026-08-03",
                }
                for index in range(1, 4)
            ]
            payload = {
                "search_query": "top artificial intelligence news August 3 2026",
                # The verifier requires today's local date, so a hardcoded
                # timestamp would make this test fail on any other day.
                "researched_at": time.strftime("%Y-%m-%dT12:00:00Z"),
                "stories": stories,
            }
            (workspace / "ai_news.json").write_text(json.dumps(payload), encoding="utf-8")
            (workspace / "ai_news.md").write_text(
                "\n".join(f"## {s['headline']}\n{s['summary']}\n{s['source_url']}" for s in stories),
                encoding="utf-8",
            )

            passed = taskbench.verify_case("news-research", workspace)
            payload["stories"][2]["source_url"] = ""
            (workspace / "ai_news.json").write_text(json.dumps(payload), encoding="utf-8")
            failed = taskbench.verify_case("news-research", workspace)

        self.assertTrue(passed["passed"], passed)
        self.assertFalse(failed["passed"])
        self.assertIn("source URL", failed["detail"])

    def test_system_health_is_compared_with_local_probe_evidence(self):
        evidence = {
            "gpu": {
                "available": True,
                "name": "NVIDIA GB10",
                "driver_version": "590.00",
                "memory_total_mib": 122880,
                "memory_used_mib": 24576,
            },
            "cuda": {
                "available": True,
                "nvcc_version": "CUDA 13.0",
                "libraries": ["libcuda.so.1", "libcudart.so.13"],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "system-health"
            taskbench.initialize_case("system-health", workspace)
            report = {
                "summary": "GPU and CUDA are healthy.",
                "commands_run": ["nvidia-smi", "nvcc --version", "ldconfig -p"],
                "gpu": dict(evidence["gpu"]),
                "cuda": dict(evidence["cuda"]),
                "recommendations": ["No action required."],
            }
            (workspace / "system_health.json").write_text(json.dumps(report), encoding="utf-8")
            (workspace / "system_health.md").write_text(
                "# Summary\nNVIDIA GB10 healthy\n122880 MiB total\nCUDA 13.0\nlibcuda.so.1\n",
                encoding="utf-8",
            )

            passed = taskbench.verify_case("system-health", workspace, evidence=evidence)
            report["cuda"]["installed_libraries"] = report["cuda"].pop("libraries")
            (workspace / "system_health.json").write_text(json.dumps(report), encoding="utf-8")
            alias_passed = taskbench.verify_case("system-health", workspace, evidence=evidence)
            report["gpu"]["memory_total_mib"] = 1
            (workspace / "system_health.json").write_text(json.dumps(report), encoding="utf-8")
            failed = taskbench.verify_case("system-health", workspace, evidence=evidence)

        self.assertTrue(passed["passed"], passed)
        self.assertTrue(alias_passed["passed"], alias_passed)
        self.assertFalse(failed["passed"])
        self.assertIn("GPU memory", failed["detail"])

    def test_system_probe_uses_unified_memory_when_gb10_reports_na(self):
        command_result = mock.MagicMock(
            returncode=0,
            stdout="NVIDIA GB10, 595.71.05, [N/A], [N/A]\n",
            stderr="",
        )
        memory = mock.MagicMock(total=128 * 1024**3, used=32 * 1024**3)
        with (
            mock.patch.object(taskbench.shutil, "which", side_effect=lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None),
            mock.patch.object(taskbench.subprocess, "run", return_value=command_result),
            mock.patch.object(taskbench.psutil, "virtual_memory", return_value=memory),
        ):
            evidence = taskbench.collect_system_evidence()

        self.assertTrue(evidence["gpu"]["available"])
        self.assertEqual(evidence["gpu"]["memory_total_mib"], 128 * 1024)
        self.assertEqual(evidence["gpu"]["memory_used_mib"], 32 * 1024)
        self.assertEqual(evidence["gpu"]["memory_kind"], "unified_system")


class TaskBenchRunnerTests(unittest.TestCase):
    def test_evaluate_rejects_remote_model_endpoints(self):
        endpoint = {
            "base_url": "https://api.example.com/v1",
            "model": "remote-model",
            "studio_url": "http://127.0.0.1:7860",
        }
        with self.assertRaisesRegex(ValueError, "loopback"):
            taskbench.evaluate(endpoint, case_ids=["file-audit"])

    def test_news_case_runs_with_only_local_files_terminal_and_managed_search(self):
        endpoint = {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "local-model",
            "studio_url": "http://127.0.0.1:7860",
        }
        hermes_result = {
            "exit_code": 0,
            "timed_out": False,
            "duration_seconds": 1.0,
            "response": "done",
            "stderr": "",
            "telemetry": {},
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(taskbench, "WORKSPACES_DIR", Path(tmp)),
            mock.patch.object(taskbench.agentlab, "_invoke_hermes", return_value=hermes_result) as invoke,
            mock.patch.object(taskbench, "verify_case", return_value={"passed": True, "detail": "ok"}),
        ):
            result = taskbench.run_case(
                endpoint,
                "news-research",
                "eval-1",
                max_turns=25,
                timeout=90,
                unsafe_yolo=False,
            )

        self.assertTrue(result["passed"])
        kwargs = invoke.call_args.kwargs
        self.assertEqual(kwargs["mode"], "task-bench")
        self.assertTrue(kwargs["allow_web_search"])
        self.assertEqual(kwargs["toolsets"], "file,terminal,mcp-sparkstudio")

    def test_evaluate_scores_cases_and_discloses_the_network_case(self):
        endpoint = {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "local-model",
            "studio_url": "http://127.0.0.1:7860",
        }
        case_results = [
            {"case": "file-audit", "passed": True, "detail": "ok", "network_used": False},
            {"case": "news-research", "passed": False, "detail": "bad", "network_used": True},
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(taskbench, "WORKSPACES_DIR", Path(tmp) / "workspaces"),
            mock.patch.object(taskbench, "RESULTS_DIR", Path(tmp) / "results"),
            mock.patch.object(taskbench, "run_case", side_effect=case_results),
            mock.patch.object(taskbench.db, "taskbench_upsert", return_value=1) as persist,
        ):
            result = taskbench.evaluate(
                endpoint,
                case_ids=["file-audit", "news-research"],
                max_turns=30,
                timeout=120,
                unsafe_yolo=False,
            )
            report_exists = Path(result["report_path"]).is_file()

        self.assertEqual(result["score"], 50.0)
        self.assertTrue(result["local_execution"])
        self.assertEqual(result["network_cases"], ["news-research"])
        self.assertTrue(report_exists)
        self.assertEqual(persist.call_count, 1)

    def test_background_runner_exposes_progress_and_final_result(self):
        endpoint = {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "local-model",
            "studio_url": "http://127.0.0.1:7860",
        }
        completed = {
            "id": "taskbench-test",
            "status": "completed",
            "model": "local-model",
            "score": 100.0,
            "passed": 1,
            "total": 1,
            "cases": [{"case": "file-audit", "passed": True}],
            "local_execution": True,
            "network_cases": [],
        }
        with mock.patch.object(taskbench, "evaluate", return_value=completed) as evaluate:
            taskbench.start_eval(
                endpoint,
                case_ids=["file-audit"],
                max_turns=20,
                timeout=60,
                unsafe_yolo=False,
            )
            deadline = time.time() + 2
            state = taskbench.eval_status()
            while state["running"] and time.time() < deadline:
                time.sleep(0.01)
                state = taskbench.eval_status()

        self.assertFalse(state["running"])
        self.assertEqual(state["score"], 100.0)
        self.assertEqual(evaluate.call_args.kwargs["case_ids"], ["file-audit"])

    def test_cli_sends_case_selection_and_limits_to_api(self):
        args = sparkstudio_cli.build_parser().parse_args(
            [
                "--base-url", "http://127.0.0.1:8000/v1",
                "--model", "local-model",
                "bench", "task-bench",
                "--case", "file-audit",
                "--max-turns", "12",
                "--timeout", "60",
            ]
        )
        response = mock.MagicMock()
        response.json.return_value = {
            "running": False,
            "score": 100.0,
            "cases": [{"case": "file-audit", "passed": True, "detail": "ok"}],
        }
        client = mock.MagicMock()
        client.post.return_value = response
        context = mock.MagicMock()
        context.__enter__.return_value = client
        with mock.patch.object(sparkstudio_cli, "_client", return_value=context):
            exit_code = sparkstudio_cli.cmd_bench_task_bench(args)

        self.assertEqual(exit_code, 0)
        client.post.assert_called_once_with(
            "/api/taskbench/run",
            json={
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "local-model",
                "cases": ["file-audit"],
                "max_turns": 12,
                "timeout": 60.0,
                "unsafe_yolo": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
