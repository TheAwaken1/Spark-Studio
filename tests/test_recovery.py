"""The "I broke it" recovery actions.

These are clicked by beginners, so the rules that matter are what each action
refuses to touch: live runs, user-managed containers, jobs sparkrun still
reports, and user data without explicit confirmation. Docker, sparkrun and
the runner are all mocked — nothing here shells out.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import db
import docker_recipe
import recovery
import registry
import sparkrun_service


def _run(status, label=None, containers=None):
    return SimpleNamespace(status=status, label=label, managed_containers=containers or [])


class ClearFinishedRunsTests(unittest.TestCase):
    def setUp(self):
        self.runs = {
            "a": _run("exited", label="qwen"),
            "b": _run("exited"),
            "c": _run("running", label="live"),
        }
        patcher = mock.patch.object(recovery.runner, "runs", self.runs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_drops_only_exited_runs_from_the_live_list(self):
        with mock.patch.object(db, "runs_list_running", return_value=[]):
            out = recovery.clear_finished_runs()
        self.assertEqual(list(self.runs), ["c"])
        # Label when there is one, run id otherwise.
        self.assertEqual(out, {"ok": True, "removed_from_list": ["qwen", "b"],
                               "stale_rows_closed": 0})

    def test_closes_db_rows_left_running_by_a_crashed_session(self):
        rows = [{"id": "c"}, {"id": "ghost"}, {"id": "a"}]
        with mock.patch.object(db, "runs_list_running", return_value=rows), \
                mock.patch.object(db, "runs_update") as upd, \
                mock.patch.object(db, "now", return_value=123):
            out = recovery.clear_finished_runs()
        # "c" is still live; "ghost" is unknown and "a" was just dropped.
        self.assertEqual(out["stale_rows_closed"], 2)
        upd.assert_has_calls([
            mock.call("ghost", status="exited", ended_at=123),
            mock.call("a", status="exited", ended_at=123),
        ])
        self.assertEqual(upd.call_count, 2)

    def test_db_failure_does_not_break_the_declutter(self):
        with mock.patch.object(db, "runs_list_running", side_effect=RuntimeError("locked")):
            out = recovery.clear_finished_runs()
        self.assertTrue(out["ok"])
        self.assertEqual(out["stale_rows_closed"], 0)
        self.assertEqual(list(self.runs), ["c"])


class DockerContainersTests(unittest.TestCase):
    def test_no_docker_means_no_containers(self):
        with mock.patch.object(recovery.shutil, "which", return_value=None), \
                mock.patch.object(recovery.subprocess, "run") as run:
            self.assertEqual(recovery._docker_containers(), [])
        run.assert_not_called()

    def test_parses_name_and_state_and_skips_malformed_lines(self):
        stdout = "sparkrun_abc123_qwen\trunning\n bad-line-no-tab\nspark-vllm-x \texited \n\n"
        res = SimpleNamespace(stdout=stdout, returncode=0)
        with mock.patch.object(recovery.shutil, "which", return_value="/usr/bin/docker"), \
                mock.patch.object(recovery.subprocess, "run", return_value=res) as run:
            out = recovery._docker_containers()
        self.assertEqual(out, [{"name": "sparkrun_abc123_qwen", "state": "running"},
                               {"name": "spark-vllm-x", "state": "exited"}])
        self.assertEqual(run.call_args.args[0][:3], ["/usr/bin/docker", "ps", "-a"])

    def test_docker_errors_yield_an_empty_list(self):
        with mock.patch.object(recovery.shutil, "which", return_value="/usr/bin/docker"), \
                mock.patch.object(recovery.subprocess, "run",
                                  side_effect=subprocess.TimeoutExpired("docker", 20)):
            self.assertEqual(recovery._docker_containers(), [])


class CleanContainersTests(unittest.TestCase):
    def setUp(self):
        self.runs = {}
        self.containers = []
        self.active_jobs = []
        self.sparkrun_bin = None
        self.rm_result = SimpleNamespace(returncode=0, stdout="", stderr="")
        for target, attr, kwargs in (
            (recovery.runner, "runs", {"new": self.runs}),
            (recovery.shutil, "which", {"return_value": "/usr/bin/docker"}),
            (recovery, "_docker_containers", {"side_effect": lambda: self.containers}),
            (sparkrun_service, "parse_status", {"side_effect": lambda: self.active_jobs}),
            (sparkrun_service, "sparkrun_bin", {"side_effect": lambda: self.sparkrun_bin}),
        ):
            patcher = mock.patch.object(target, attr, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(recovery.subprocess, "run",
                                    side_effect=lambda *a, **k: self.rm_result)
        self.rm = patcher.start()
        self.addCleanup(patcher.stop)

    def removed_names(self):
        return [c.args[0][3] for c in self.rm.call_args_list]

    def test_no_docker_is_an_error_not_a_crash(self):
        with mock.patch.object(recovery.shutil, "which", return_value=None):
            self.assertEqual(recovery.clean_containers(),
                             {"ok": False, "error": "docker not found"})
        self.rm.assert_not_called()

    def test_never_touches_containers_it_does_not_own(self):
        self.containers = [{"name": n, "state": "exited"}
                           for n in ("vllm_node", "spark-searxng", "my-stack", "xsparkrun_abc123_q")]
        out = recovery.clean_containers()
        self.assertEqual(out, {"ok": True, "removed": [], "skipped": []})
        self.rm.assert_not_called()

    def test_removes_orphans_with_force_rm(self):
        self.containers = [{"name": "spark-vllm-qwen", "state": "running"},
                           {"name": "sparkrun_abc123_qwen", "state": "exited"}]
        out = recovery.clean_containers()
        self.assertEqual(out["removed"], ["spark-vllm-qwen", "sparkrun_abc123_qwen"])
        self.assertEqual(self.rm.call_args_list[0].args[0],
                         ["/usr/bin/docker", "rm", "-f", "spark-vllm-qwen"])

    def test_containers_of_a_live_run_are_off_limits(self):
        self.runs["r1"] = _run("running", containers=["spark-vllm-live"])
        self.runs["r2"] = _run("exited", containers=["spark-vllm-dead"])
        self.containers = [{"name": "spark-vllm-live", "state": "running"},
                           {"name": "spark-vllm-dead", "state": "exited"}]
        out = recovery.clean_containers()
        self.assertEqual(out["removed"], ["spark-vllm-dead"])
        self.assertEqual(out["skipped"], [{"name": "spark-vllm-live", "why": "owned by a live run"}])

    def test_running_sparkrun_job_still_reported_active_is_kept(self):
        self.active_jobs = [{"jobid": "abc123"}]
        self.containers = [{"name": "sparkrun_abc123_qwen", "state": "Running"},
                           {"name": "sparkrun_def456_llama", "state": "running"}]
        out = recovery.clean_containers()
        self.assertEqual(out["skipped"], [{"name": "sparkrun_abc123_qwen",
                                           "why": "sparkrun reports this job as active"}])
        # sparkrun answered (non-empty) and does not know def456: it is an orphan.
        self.assertEqual(out["removed"], ["sparkrun_def456_llama"])

    def test_stopped_sparkrun_container_is_reaped_even_if_job_is_listed(self):
        self.active_jobs = [{"jobid": "abc123"}]
        self.containers = [{"name": "sparkrun_abc123_qwen", "state": "exited"}]
        self.assertEqual(recovery.clean_containers()["removed"], ["sparkrun_abc123_qwen"])

    def test_empty_sparkrun_status_is_ambiguous_so_running_jobs_are_kept(self):
        self.sparkrun_bin = "/usr/local/bin/sparkrun"
        self.containers = [{"name": "sparkrun_abc123_qwen", "state": "running"},
                           {"name": "sparkrun_def456_llama", "state": "exited"}]
        out = recovery.clean_containers()
        self.assertEqual(out["skipped"], [{"name": "sparkrun_abc123_qwen",
                                           "why": "could not confirm the job is dead"}])
        self.assertEqual(out["removed"], ["sparkrun_def456_llama"])

    def test_sparkrun_status_failure_counts_as_no_active_jobs(self):
        self.sparkrun_bin = "/usr/local/bin/sparkrun"
        self.containers = [{"name": "sparkrun_abc123_qwen", "state": "running"}]
        with mock.patch.object(sparkrun_service, "parse_status", side_effect=RuntimeError):
            out = recovery.clean_containers()
        self.assertEqual(out["skipped"][0]["why"], "could not confirm the job is dead")
        self.rm.assert_not_called()

    def test_without_sparkrun_installed_running_orphans_are_reaped(self):
        self.containers = [{"name": "sparkrun_abc123_qwen", "state": "running"}]
        self.assertEqual(recovery.clean_containers()["removed"], ["sparkrun_abc123_qwen"])

    def test_failed_rm_is_reported_as_skipped(self):
        self.containers = [{"name": "spark-vllm-a", "state": "exited"}]
        self.rm_result = SimpleNamespace(returncode=1, stdout="", stderr="  " + "E" * 300 + "\n")
        out = recovery.clean_containers()
        self.assertEqual(out["removed"], [])
        self.assertEqual(out["skipped"], [{"name": "spark-vllm-a", "why": "E" * 120}])

    def test_rm_exception_is_reported_as_skipped(self):
        self.containers = [{"name": "spark-vllm-a", "state": "exited"},
                           {"name": "spark-vllm-b", "state": "exited"}]
        self.rm.side_effect = [subprocess.TimeoutExpired("docker", 60), self.rm_result]
        out = recovery.clean_containers()
        self.assertEqual(out["removed"], ["spark-vllm-b"])
        self.assertEqual(out["skipped"][0]["name"], "spark-vllm-a")
        self.assertIn("timed out", out["skipped"][0]["why"])


class ResetRegistryTests(unittest.TestCase):
    def test_removes_existing_mirrors_and_reindexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = Path(tmp) / "registry"
            forged = Path(tmp) / "forged"
            (reg / "repo").mkdir(parents=True)
            (reg / "repo" / "recipe.yaml").write_text("x")
            with mock.patch.object(registry, "REGISTRY_ROOT", reg), \
                    mock.patch.object(docker_recipe, "FORGED_DIR", forged), \
                    mock.patch.object(registry, "reindex") as reindex:
                out = recovery.reset_registry()
            self.assertFalse(reg.exists())
            # Only paths that existed are reported.
            self.assertEqual(out, {"ok": True, "removed": [str(reg)]})
            reindex.assert_called_once_with()


class WipeDbTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(db, "DB_PATH", Path(tmp.name) / "wipe.db"):
            conn = db._connect()
        self.addCleanup(conn.close)
        conn.execute("PRAGMA synchronous=OFF")
        patcher = mock.patch.object(db, "_conn", conn)
        patcher.start()
        self.addCleanup(patcher.stop)
        db.init()
        for table, col in (("recipes", "raw_cmd"), ("benchmarks", "engine_version"),
                           ("benchy_runs", "engine_version"), ("runs", "meta_json")):
            db._add_col_if_missing(table, col, "TEXT")
        self.conn = conn

    def count(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def test_requires_confirmation(self):
        db.recipes_upsert({"name": "keep", "engine": "vllm"})
        self.assertEqual(recovery.wipe_db(),
                         {"ok": False, "error": "confirmation required"})
        self.assertEqual(self.count("recipes"), 1)

    def test_deletes_user_history_tables_only(self):
        rec = db.recipes_upsert({"name": "r", "engine": "vllm"})
        db.recipes_upsert({**rec, "notes": "edited"})  # leaves a snapshot
        db.runs_insert({"id": "run1", "recipe_id": rec["id"], "engine": "vllm",
                        "status": "exited", "cmd": "c"})
        db.bench_insert("run1", rec["id"], {"tokens_per_sec": 1.0})
        db.bench_insert("run1", rec["id"], {"tokens_per_sec": 2.0})
        db.benchy_insert("run1", rec["id"], "m", "u", {}, None, 0)
        db.tooleval_insert({"model": "m"})
        db.taskbench_upsert({"model": "m"})
        db.agentlab_upsert({"id": "lab1"})

        out = recovery.wipe_db(confirm=True)

        self.assertEqual(out, {"ok": True, "deleted": {
            "benchmarks": 2, "benchy_runs": 1, "tooleval_runs": 1, "runs": 1, "recipes": 1}})
        for table in ("benchmarks", "benchy_runs", "tooleval_runs", "runs", "recipes",
                      "recipe_snapshots"):
            self.assertEqual(self.count(table), 0, table)
        # Not in the wipe list.
        self.assertEqual(self.count("taskbench_runs"), 1)
        self.assertEqual(self.count("agentlab_runs"), 1)

    def test_does_not_stop_in_memory_runs(self):
        live = {"r": _run("running")}
        with mock.patch.object(recovery.runner, "runs", live):
            recovery.wipe_db(confirm=True)
        self.assertIn("r", live)


if __name__ == "__main__":
    unittest.main()
