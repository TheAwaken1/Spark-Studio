"""Doctor: probes must degrade gracefully, never raise, and report honestly.

The doctor is the single source of truth for "is this box OK?" — the wizard,
Feature Health panel and bug reports all embed its output. The contract in its
docstring ("Checks never raise: a probe that blows up becomes its own 'error'
entry") is exactly what these tests pin down, plus the judgement rules baked
into individual probes (driver 590.x warn, tiny-context free, DGX Spark labels,
docker-recipe engine fallback, Caddyfile URL parsing).
"""

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import doctor


def _fake_host(**over):
    host = {"gpus": [{"name": "NVIDIA GB10", "driver": "580.1", "memory_gb": 128}],
            "gpu_count": 1, "is_dgx_spark": True, "mesh_size": 1,
            "summary": "1× GB10 · 128 GB"}
    host.update(over)
    return host


class CheckWrapperTests(unittest.TestCase):
    def test_probe_crash_becomes_error_entry_not_exception(self):
        def boom():
            raise RuntimeError("nvidia-smi exploded")
        entry = doctor._check("gpu", "NVIDIA GPU", boom)
        self.assertEqual(entry["id"], "gpu")
        self.assertEqual(entry["label"], "NVIDIA GPU")
        self.assertEqual(entry["status"], "error")
        self.assertIn("nvidia-smi exploded", entry["detail"])

    def test_probe_result_merges_onto_base(self):
        entry = doctor._check("x", "X", lambda: {"status": "ok", "detail": "fine"})
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["detail"], "fine")
        self.assertIsNone(entry["fix"])  # base default survives


class WhichVersionTests(unittest.TestCase):
    def test_missing_binary_is_none(self):
        with mock.patch("doctor.shutil.which", return_value=None):
            self.assertIsNone(doctor._which_version("no-such-tool"))

    def test_first_line_truncated_to_80(self):
        long = "x" * 200
        with mock.patch("doctor.shutil.which", return_value="/bin/tool"), \
             mock.patch("doctor.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 0, stdout=long + "\nsecond")):
            self.assertEqual(doctor._which_version("tool"), "x" * 80)

    def test_stderr_used_when_stdout_empty(self):
        with mock.patch("doctor.shutil.which", return_value="/bin/tool"), \
             mock.patch("doctor.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="v1.2.3")):
            self.assertEqual(doctor._which_version("tool"), "v1.2.3")

    def test_timeout_becomes_none(self):
        with mock.patch("doctor.shutil.which", return_value="/bin/tool"), \
             mock.patch("doctor.subprocess.run", side_effect=OSError("hang")):
            self.assertIsNone(doctor._which_version("tool"))


class PlatformPythonTests(unittest.TestCase):
    def test_non_linux_is_error_with_fix(self):
        with mock.patch("doctor.platform.system", return_value="Darwin"):
            res = doctor._probe_platform()
        self.assertEqual(res["status"], "error")
        self.assertIsNotNone(res["fix"])

    def test_python_below_311_warns(self):
        v = mock.Mock(major=3, minor=10, micro=0)
        with mock.patch.object(doctor.sys, "version_info", v):
            self.assertEqual(doctor._probe_python()["status"], "warn")

    def test_python_311_is_ok(self):
        v = mock.Mock(major=3, minor=11, micro=0)
        with mock.patch.object(doctor.sys, "version_info", v):
            self.assertEqual(doctor._probe_python()["status"], "ok")


class ToolProbeTests(unittest.TestCase):
    def test_required_tool_missing_is_error(self):
        fn = doctor._probe_tool("git", "why", "how", required=True)
        with mock.patch("doctor._which_version", return_value=None):
            self.assertEqual(fn()["status"], "error")

    def test_optional_tool_missing_is_warn(self):
        fn = doctor._probe_tool("uv", "why", "how")
        with mock.patch("doctor._which_version", return_value=None):
            res = fn()
        self.assertEqual(res["status"], "warn")
        self.assertEqual(res["fix"], "how")

    def test_found_tool_reports_version(self):
        fn = doctor._probe_tool("git", "why", "how", required=True)
        with mock.patch("doctor._which_version", return_value="git version 2.45"):
            self.assertEqual(fn(), {"status": "ok", "detail": "git version 2.45"})


class GpuDriverMemoryTests(unittest.TestCase):
    def test_no_gpu_is_error(self):
        with mock.patch("hostinfo.probe_host", return_value=_fake_host(gpus=[], gpu_count=0)):
            self.assertEqual(doctor._probe_gpu()["status"], "error")

    def test_dgx_spark_and_mesh_are_labelled(self):
        host = _fake_host(mesh_size=2, summary="1× GB10 · 128 GB · mesh ×2 (~256 GB)")
        with mock.patch("hostinfo.probe_host", return_value=host):
            res = doctor._probe_gpu()
        self.assertEqual(res["status"], "ok")
        self.assertIn("DGX Spark detected", res["detail"])
        self.assertIn("mesh ×2", res["detail"])

    def test_driver_590_warns_about_cudagraph_deadlock(self):
        host = _fake_host(gpus=[{"driver": "590.5"}])
        with mock.patch("hostinfo.probe_host", return_value=host):
            res = doctor._probe_driver()
        self.assertEqual(res["status"], "warn")
        self.assertIn("590.x", res["fix"])

    def test_driver_580_is_ok(self):
        host = _fake_host(gpus=[{"driver": "580.12"}])
        with mock.patch("hostinfo.probe_host", return_value=host):
            self.assertEqual(doctor._probe_driver()["status"], "ok")

    def test_no_gpu_driver_is_warn_not_error(self):
        with mock.patch("hostinfo.probe_host", return_value=_fake_host(gpus=[], gpu_count=0)):
            self.assertEqual(doctor._probe_driver()["status"], "warn")

    def test_low_free_memory_warns(self):
        vm = mock.Mock(total=128 * 1024**3, available=6 * 1024**3)
        with mock.patch("psutil.virtual_memory", return_value=vm):
            self.assertEqual(doctor._probe_memory()["status"], "warn")

    def test_plenty_free_memory_ok(self):
        vm = mock.Mock(total=128 * 1024**3, available=64 * 1024**3)
        with mock.patch("psutil.virtual_memory", return_value=vm):
            res = doctor._probe_memory()
        self.assertEqual(res["status"], "ok")
        self.assertIn("64 GB free / 128 GB unified", res["detail"])


class DockerTests(unittest.TestCase):
    def test_no_docker_binary_warns(self):
        with mock.patch("doctor.shutil.which", return_value=None):
            self.assertEqual(doctor._probe_docker()["status"], "warn")

    def test_daemon_unreachable_warns(self):
        with mock.patch("doctor.shutil.which", return_value="/usr/bin/docker"), \
             mock.patch("doctor.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="down")):
            res = doctor._probe_docker()
        self.assertEqual(res["status"], "warn")
        self.assertIn("daemon not reachable", res["detail"])

    def test_working_daemon_reports_version(self):
        with mock.patch("doctor.shutil.which", return_value="/usr/bin/docker"), \
             mock.patch("doctor.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 0, stdout="27.1.1")):
            self.assertEqual(doctor._probe_docker(), {"status": "ok", "detail": "docker 27.1.1"})


class EngineProbeTests(unittest.TestCase):
    """Docker recipes keep engines usable without a native pip install."""

    def _mirror(self, tmp: Path, engine: str = "vllm"):
        (tmp / "data" / "registry" / "spark-vllm-docker").mkdir(parents=True)

    def test_native_engine_is_ok(self):
        fn = doctor._probe_engine("vllm", "hint")
        with mock.patch("runners.engine_available", return_value=True):
            res = fn()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["detail"], "installed (native)")

    def test_sglang_without_native_and_without_docker_is_warn(self):
        fn = doctor._probe_engine("sglang", "hint")
        with mock.patch("runners.engine_available", return_value=False), \
             mock.patch("doctor.shutil.which", return_value=None):
            res = fn()
        self.assertEqual(res["status"], "warn")
        self.assertEqual(res["fix"], "hint")

    def test_sglang_docker_rescue_does_not_need_the_vllm_mirror(self):
        # sglang containers ship via sparkrun recipes: docker presence is enough.
        fn = doctor._probe_engine("sglang", "hint")
        with mock.patch("runners.engine_available", return_value=False), \
             mock.patch("doctor.shutil.which", return_value="/usr/bin/docker"), \
             mock.patch.object(doctor, "APP_DIR", Path("/nonexistent-spark-studio")):
            res = fn()
        self.assertEqual(res["status"], "ok")
        self.assertIn("sparkrun container recipes", res["detail"])

    def test_vllm_docker_recipe_fallback_is_ok(self):
        fn = doctor._probe_engine("vllm", "hint")
        with TemporaryDirectory() as td, \
             mock.patch("runners.engine_available", return_value=False), \
             mock.patch("doctor.shutil.which", return_value="/usr/bin/docker"), \
             mock.patch.object(doctor, "APP_DIR", Path(td)), \
             mock.patch("doctor._vllm_image_age_suffix", return_value=("", False)):
            self._mirror(Path(td))
            res = fn()
        self.assertEqual(res["status"], "ok")
        self.assertIn("spark-vllm-docker containers", res["detail"])

    def test_stale_runner_image_warns_with_update_fix(self):
        fn = doctor._probe_engine("vllm", "hint")
        with TemporaryDirectory() as td, \
             mock.patch("runners.engine_available", return_value=False), \
             mock.patch("doctor.shutil.which", return_value="/usr/bin/docker"), \
             mock.patch.object(doctor, "APP_DIR", Path(td)), \
             mock.patch("doctor._vllm_image_age_suffix",
                        return_value=(" · vllm-node image is 30 days old", True)):
            self._mirror(Path(td))
            res = fn()
        self.assertEqual(res["status"], "warn")
        self.assertIn("30 days old", res["detail"])
        self.assertIsNotNone(res["fix"])


class AgentIdentityProbeTests(unittest.TestCase):
    def test_claude_missing_warns_with_npm_fix(self):
        with mock.patch("agents.claude_available", return_value=False):
            res = doctor._probe_agent("claude")
        self.assertEqual(res["status"], "warn")
        self.assertIn("claude-code", res["fix"])

    def test_huggingface_signed_in_shows_username(self):
        status = {"installed": True, "logged_in": True, "username": "alice"}
        with mock.patch("agents.huggingface_status", return_value=status):
            res = doctor._probe_huggingface()
        self.assertEqual(res["status"], "ok")
        self.assertIn("as alice", res["detail"])

    def test_hermes_missing_warns(self):
        with mock.patch("agentlab.hermes_status", return_value={"installed": False}):
            self.assertEqual(doctor._probe_hermes()["status"], "warn")


class UrlsTests(unittest.TestCase):
    def test_lan_ips_skip_ipv6_and_loopback(self):
        out = subprocess.CompletedProcess([], 0, stdout="127.0.0.1 fe80::1 192.168.1.10\n")
        with mock.patch("doctor.subprocess.run", return_value=out):
            self.assertEqual(doctor._lan_ips(), ["192.168.1.10"])

    def test_lan_ips_survive_failing_hostname(self):
        with mock.patch("doctor.subprocess.run", side_effect=OSError("no hostname")):
            self.assertEqual(doctor._lan_ips(), [])

    def test_https_urls_parsed_from_caddyfile_via_unit(self):
        with TemporaryDirectory() as td:
            home = Path(td)
            unit = home / ".config" / "systemd" / "user" / "spark-studio-https.service"
            unit.parent.mkdir(parents=True)
            caddyfile = home / "Caddyfile"
            caddyfile.write_text(
                "https://spark.example.com {\n  reverse_proxy 127.0.0.1:7860\n}\n"
                "http://not-this.example.com {\n}\n", encoding="utf-8")
            unit.write_text(f"[Service]\nExecStart=/usr/bin/caddy run --config {caddyfile}\n",
                            encoding="utf-8")
            with mock.patch.object(Path, "home", lambda: home):
                self.assertEqual(doctor._https_urls(), ["https://spark.example.com"])

    def test_https_urls_empty_without_unit(self):
        with TemporaryDirectory() as td:
            with mock.patch.object(Path, "home", lambda: Path(td)):
                self.assertEqual(doctor._https_urls(), [])


class PortProbeTests(unittest.TestCase):
    def test_free_port_warns_with_start_hint(self):
        fn = doctor._probe_port(7860)
        with mock.patch("doctor._port_in_use", return_value=False):
            res = fn()
        self.assertEqual(res["status"], "warn")
        self.assertEqual(res["fix"], "./start.sh")

    def test_serving_port_ok(self):
        fn = doctor._probe_port(7860)
        with mock.patch("doctor._port_in_use", return_value=True):
            self.assertEqual(fn()["status"], "ok")


class RunChecksTests(unittest.TestCase):
    def test_report_shape_and_summary_counts(self):
        # One probe crashes on purpose: the rest of the report must survive it.
        def boom():
            raise RuntimeError("probe exploded")
        with mock.patch("doctor._lan_ips", return_value=["192.168.1.5"]), \
             mock.patch("doctor._probe_gpu", boom), \
             mock.patch("doctor._probe_driver", return_value={"status": "ok"}), \
             mock.patch("doctor._probe_memory", return_value={"status": "ok"}), \
             mock.patch("doctor._probe_docker", return_value={"status": "ok"}), \
             mock.patch("doctor._probe_sparkrun", return_value={"status": "ok"}), \
             mock.patch("doctor._probe_huggingface", return_value={"status": "ok"}), \
             mock.patch("doctor._probe_hermes", return_value={"status": "ok"}), \
             mock.patch("doctor._probe_benchy", return_value={"status": "ok"}), \
             mock.patch("doctor._probe_searxng", return_value={"status": "ok"}), \
             mock.patch("doctor._which_version", return_value="x"), \
             mock.patch("runners.engine_available", return_value=True), \
             mock.patch("doctor._port_in_use", return_value=True), \
             mock.patch("agents.claude_available", return_value=True), \
             mock.patch("agents.codex_available", return_value=True):
            report = doctor.run_checks(port=7860)
        ids = [c["id"] for c in report["checks"]]
        self.assertIn("gpu", ids)
        self.assertIn("port", ids)
        gpu = next(c for c in report["checks"] if c["id"] == "gpu")
        self.assertEqual(gpu["status"], "error")  # crash contained, report survives
        self.assertEqual(sum(report["summary"].values()), len(report["checks"]))
        self.assertEqual(report["urls"]["local"], "http://127.0.0.1:7860")
        self.assertEqual(report["urls"]["lan"], ["http://192.168.1.5:7860"])
        self.assertIsInstance(doctor.app_version(), str)


class FormatCliTests(unittest.TestCase):
    def test_symbols_and_counts_render(self):
        report = {"version": "1.2.3",
                  "urls": {"local": "http://127.0.0.1:7860", "lan": ["http://x:7860"], "https": []},
                  "summary": {"ok": 2, "warn": 1, "error": 1},
                  "checks": [{"id": "a", "label": "A", "status": "ok", "detail": "fine", "fix": None},
                             {"id": "b", "label": "B", "status": "warn", "detail": "meh", "fix": "do X"},
                             {"id": "c", "label": "C", "status": "error", "detail": "bad", "fix": "do Y"}]}
        out = doctor.format_cli(report)
        self.assertIn("A: fine", out)
        self.assertIn("B: meh", out)
        self.assertIn("↳ do X", out)
        self.assertIn("C: bad", out)
        self.assertIn("2 ok · 1 warnings · 1 errors", out)
        self.assertIn("http://x:7860", out)

    def test_ok_status_never_prints_a_fix(self):
        report = {"version": "x", "urls": {"local": "", "lan": [], "https": []},
                  "summary": {"ok": 1, "warn": 0, "error": 0},
                  "checks": [{"id": "a", "label": "A", "status": "ok", "detail": "fine", "fix": "irrelevant"}]}
        self.assertNotIn("irrelevant", doctor.format_cli(report))


class AppVersionTests(unittest.TestCase):
    def test_reads_repo_version_file(self):
        v = doctor.app_version()
        self.assertRegex(v, r"^\d+\.\d+")


if __name__ == "__main__":
    unittest.main()
