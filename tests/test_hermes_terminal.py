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


if __name__ == "__main__":
    unittest.main()
