import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import agentlab
import hermes_terminal
import runners
import server


class HermesBrowserTerminalTests(unittest.TestCase):
    def test_terminal_access_policy_allows_encrypted_private_lan(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPARK_STUDIO_HERMES_TUI_ALLOW_REMOTE", None)
            self.assertEqual(
                server._terminal_access_policy("192.168.0.158", "wss")[:2],
                (True, "private_https"),
            )
            self.assertEqual(
                server._terminal_access_policy("192.168.0.158", "https")[:2],
                (True, "private_https"),
            )

    def test_terminal_access_policy_rejects_plain_lan_and_public_remote(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPARK_STUDIO_HERMES_TUI_ALLOW_REMOTE", None)
            self.assertEqual(
                server._terminal_access_policy("192.168.0.158", "ws")[:2],
                (False, "private_http_denied"),
            )
            self.assertEqual(
                server._terminal_access_policy("8.8.8.8", "wss")[:2],
                (False, "remote_denied"),
            )

    def test_terminal_access_override_is_explicit(self):
        with mock.patch.dict(
            os.environ, {"SPARK_STUDIO_HERMES_TUI_ALLOW_REMOTE": "1"}
        ):
            self.assertEqual(
                server._terminal_access_policy("8.8.8.8", "ws")[:2],
                (True, "explicit_remote"),
            )

    def test_internal_studio_url_stays_http_behind_wss_proxy(self):
        websocket = SimpleNamespace(scope={"server": ("0.0.0.0", 7860)})
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPARK_STUDIO_INTERNAL_URL", None)
            self.assertEqual(
                server._studio_url_for_websocket(websocket),
                "http://127.0.0.1:7860",
            )

    def test_browser_command_launches_real_tui_with_agent_tools(self):
        command = hermes_terminal.browser_tui_command("/opt/hermes", "fixture-model")
        self.assertEqual(command[:2], ["/opt/hermes", "--tui"])
        self.assertEqual(command[command.index("--model") + 1], "fixture-model")
        self.assertNotIn("--max-turns", command)
        self.assertIn("file,terminal,mcp-sparkstudio", command)

    def test_prepare_uses_isolated_profile_workspace_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile"
            workspace = root / "workspace"
            workspace.mkdir()
            endpoint = {
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "fixture-model",
                "studio_url": "http://127.0.0.1:7860",
            }
            with (
                mock.patch.object(agentlab, "HERMES_HOME", profile),
                mock.patch.object(agentlab, "find_hermes", return_value="/opt/hermes"),
            ):
                command, env = hermes_terminal.prepare_browser_tui(
                    endpoint, workspace, max_turns=33
                )
            config = yaml.safe_load((profile / "config.yaml").read_text())

        self.assertEqual(command[0], "/opt/hermes")
        self.assertEqual(env["HERMES_HOME"], str(profile))
        self.assertIn(str(workspace), env["HERMES_WRITE_SAFE_ROOT"].split(os.pathsep))
        self.assertEqual(env["TERM"], "xterm-256color")
        self.assertEqual(config["model"]["default"], "fixture-model")
        self.assertEqual(
            config["mcp_servers"]["sparkstudio"]["tools"]["include"],
            ["web_search"],
        )

    @unittest.skipUnless(hermes_terminal.PtyBridge.available(), "POSIX PTY required")
    def test_pty_bridge_streams_input_and_output(self):
        bridge = hermes_terminal.PtyBridge.spawn(
            ["/bin/sh", "-c", "printf ready; read line; printf ':got-%s' \"$line\""],
            cwd=Path.cwd(),
            env=os.environ.copy(),
        )
        chunks = []
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                payload = bridge.read(0.1)
                if payload:
                    chunks.append(payload)
                    if b"ready" in b"".join(chunks):
                        break
            bridge.write(b"hello\n")
            while time.monotonic() < deadline:
                payload = bridge.read(0.1)
                if payload is None:
                    break
                if payload:
                    chunks.append(payload)
                    if b"got-hello" in b"".join(chunks):
                        break
        finally:
            bridge.close()

        output = b"".join(chunks)
        self.assertIn(b"ready", output)
        self.assertIn(b"got-hello", output)

    @unittest.skipUnless(hermes_terminal.PtyBridge.available(), "POSIX PTY required")
    def test_dashboard_websocket_bridges_real_terminal_bytes(self):
        active = SimpleNamespace(
            id="fixture-run",
            label="fixture-model",
            url="http://127.0.0.1:8000",
            ready=True,
        )
        endpoint = {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "fixture-model",
        }
        command = [
            "/bin/sh",
            "-c",
            "printf ready; read line; printf ':got-%s' \"$line\"",
        ]
        client = TestClient(server.app)
        with (
            mock.patch.object(server.runner, "active", return_value=active),
            mock.patch.object(server.agentlab, "discover_endpoint", return_value=endpoint),
            mock.patch.object(
                server.hermes_terminal,
                "prepare_browser_tui",
                return_value=(command, os.environ.copy()),
            ),
            client.websocket_connect(
                "/api/agentlab/terminal?workspace=.",
                headers={"origin": "http://testserver"},
            ) as websocket,
        ):
            chunks = [websocket.receive_bytes()]
            websocket.send_bytes(b"hello\n")
            for _ in range(4):
                try:
                    chunks.append(websocket.receive_bytes())
                except WebSocketDisconnect:
                    break
                if b"got-hello" in b"".join(chunks):
                    break

        output = b"".join(chunks)
        self.assertIn(b"ready", output)
        self.assertIn(b"got-hello", output)


class RestartRecoveryTests(unittest.TestCase):
    def test_adopted_run_restores_ownership_metadata_and_pid(self):
        run = runners.Run(
            id="retained",
            engine="vllm",
            recipe_id=7,
            cmd=["vllm", "serve", "repo/model"],
            env={},
            managed_containers=["spark-vllm-test"],
            stop_cmd=["docker", "stop", "spark-vllm-test"],
            adopted_pid=4321,
            label="repo/model",
            meta={"load_secs": 12.3, "pump_cmd": ["tail"], "_reclaimed": True},
        )
        persisted = run.persisted_meta()
        self.assertEqual(persisted["_label"], "repo/model")
        self.assertEqual(persisted["_managed_containers"], ["spark-vllm-test"])
        self.assertEqual(persisted["_stop_cmd"], ["docker", "stop", "spark-vllm-test"])
        self.assertNotIn("pump_cmd", persisted)
        self.assertNotIn("_reclaimed", persisted)
        self.assertEqual(run.summary()["pid"], 4321)

    def test_stop_adopted_run_tears_down_restored_container(self):
        manager = runners.Runner()
        run = runners.Run(
            id="retained", engine="vllm", recipe_id=None, cmd=["adopted"], env={},
            managed_containers=["spark-vllm-test"], adopted_pid=4321, status="running",
        )
        manager.runs[run.id] = run
        with mock.patch.object(manager, "_stop_docker_containers") as stop_containers, \
             mock.patch.object(manager, "finalize"), \
             mock.patch.object(manager, "_reclaim_after_teardown"), \
             mock.patch.object(runners.os, "getpgid", return_value=4321), \
             mock.patch.object(runners.os, "killpg"):
            self.assertTrue(manager.stop(run.id))
        stop_containers.assert_called_once_with(run, force=False)

    def test_keep_runs_uses_child_owned_durable_log(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"SPARK_STUDIO_KEEP_RUNS_ON_EXIT": "1"}
        ), mock.patch.object(runners, "RUN_LOG_DIR", Path(tmp)), mock.patch.object(
            runners.db, "runs_insert"
        ), mock.patch.object(runners.db, "runs_update"), mock.patch.object(
            runners, "_mem_used_gb", return_value=0.0
        ):
            manager = runners.Runner()
            run = manager.start(
                "test",
                {},
                raw_cmd="printf durable-output; sleep 0.1",
                skip_memory_guard=True,
            )
            run.proc.wait(timeout=3)
            deadline = time.time() + 2
            while run.status != "exited" and time.time() < deadline:
                time.sleep(0.02)
            log_path = Path(run.meta["_log_path"])
            self.assertEqual(log_path.parent, Path(tmp))
            self.assertEqual(log_path.read_text(), "durable-output")
            self.assertIn("durable-output", run.ring)

    def test_historical_log_replay_does_not_fake_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "retained.log"
            log_path.write_text("Uvicorn running on http://127.0.0.1:9999\n")
            run = runners.Run(
                id="retained", engine="vllm", recipe_id=None, cmd=["adopted"], env={},
                adopted_pid=os.getpid(), meta={"_log_path": str(log_path)},
            )
            run.status = "running"
            runners.Runner()._attach_log_file(run)
            deadline = time.time() + 1
            while not run.ring and time.time() < deadline:
                time.sleep(0.02)
            self.assertFalse(run.ready)
            run.status = "exited"

    def test_retained_endpoint_is_ready_immediately_after_health_probe(self):
        run = runners.Run(
            id="retained", engine="vllm", recipe_id=None, cmd=["adopted"], env={}
        )
        run.status = "running"
        run.url = "http://127.0.0.1:9999"
        run.ready_at = run.started_at
        response = SimpleNamespace(
            status_code=200, json=lambda: {"data": [{"id": "repo/model"}]}
        )
        client = mock.AsyncMock()
        client.get.return_value = response
        context = mock.MagicMock()
        context.__aenter__ = mock.AsyncMock(return_value=client)
        context.__aexit__ = mock.AsyncMock(return_value=None)
        with mock.patch("httpx.AsyncClient", return_value=context), mock.patch.object(
            server.db, "runs_update"
        ):
            asyncio.run(server._mark_adopted_ready(run))
        self.assertTrue(run.ready)
        self.assertEqual(run.label, "repo/model")


if __name__ == "__main__":
    unittest.main()
