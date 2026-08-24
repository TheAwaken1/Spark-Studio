import asyncio
import base64
import itertools
import json
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


def _close_terminal_leases() -> None:
    """Close every leased PTY, as Stop and the TTL janitor do.

    Leases deliberately outlive their socket, so a test that leaves one behind
    leaks a pty child and a pump thread into the rest of the run.
    """
    for session in list(server._TERMINAL_SESSIONS.values()):
        session["closed"] = True
        session["bridge"].close()
    server._TERMINAL_SESSIONS.clear()


class HermesBrowserTerminalTests(unittest.TestCase):
    def tearDown(self):
        _close_terminal_leases()

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

    def test_hermes_install_allows_plain_http_lan_but_not_public_remote(self):
        # Installing the pinned CLI is a fixed action with no shell access, so
        # it follows the dashboard's normal LAN trust boundary, not the HTTPS
        # terminal policy that gates Hermes Chat itself.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPARK_STUDIO_HERMES_TUI_ALLOW_REMOTE", None)
            self.assertTrue(server._lan_peer_allowed("192.168.0.158"))
            self.assertTrue(server._lan_peer_allowed("127.0.0.1"))
            self.assertFalse(server._lan_peer_allowed("8.8.8.8"))
            self.assertFalse(server._lan_peer_allowed(""))

    def test_hermes_install_endpoint_reaches_installer_from_test_peer(self):
        client = TestClient(server.app)
        installed = {"installed": True, "version": "v2026.7.20"}
        with mock.patch.object(
            server.agentlab, "install_hermes", return_value=installed
        ) as install:
            response = client.post("/api/agentlab/install")
        self.assertEqual(response.status_code, 200)
        install.assert_called_once()

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

    def test_websocket_origin_accepts_forwarded_proxy_host(self):
        websocket = SimpleNamespace(
            headers={
                "origin": "https://spark.local:8443",
                "host": "127.0.0.1:7860",
                "x-forwarded-host": "spark.local:8443",
            }
        )
        self.assertTrue(server._websocket_same_origin(websocket))

    def test_websocket_origin_rejection_returns_visible_close_reason(self):
        client = TestClient(server.app)
        with client.websocket_connect(
            "/api/agentlab/terminal",
            headers={"origin": "https://not-spark.example"},
        ) as websocket:
            with self.assertRaises(WebSocketDisconnect) as rejected:
                websocket.receive_bytes()
        self.assertEqual(rejected.exception.code, 4403)
        self.assertEqual(rejected.exception.reason, "origin does not match Spark Studio")

    def test_missing_model_returns_visible_terminal_error(self):
        # No engine AND no saved provider is the only remaining hard stop:
        # there is nothing for Hermes to talk to.
        client = TestClient(server.app)
        with (
            mock.patch.object(server.runner, "active", return_value=None),
            mock.patch.object(
                server.agentlab,
                "hermes_model_binding",
                return_value={"model": "", "provider": ""},
            ),
            client.websocket_connect(
                "/api/agentlab/terminal",
                headers={"origin": "http://testserver"},
            ) as websocket,
        ):
            message = websocket.receive_text()
            self.assertIn("No model is loaded", message)
            with self.assertRaises(WebSocketDisconnect) as failed:
                websocket.receive_bytes()
        self.assertEqual(failed.exception.code, 1011)
        self.assertEqual(failed.exception.reason, "Hermes TUI failed to start")

    def test_detached_session_launches_on_saved_provider_without_engine(self):
        # A recipe swap unloads the engine mid-session. Chat must still start,
        # on whatever provider the user picked with /model.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile"
            workspace = root / "workspace"
            workspace.mkdir()
            profile.mkdir()
            (profile / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "provider": "openai-codex",
                            "default": "gpt-5.6-sol",
                            "base_url": "https://chatgpt.com/backend-api/codex",
                        }
                    }
                )
            )
            with (
                mock.patch.object(agentlab, "HERMES_HOME", profile),
                mock.patch.object(agentlab, "find_hermes", return_value="/opt/hermes"),
            ):
                command, _env = hermes_terminal.prepare_browser_tui(
                    agentlab.detached_endpoint(), workspace, max_turns=33
                )
            config = yaml.safe_load((profile / "config.yaml").read_text())

        self.assertEqual(command[command.index("--model") + 1], "gpt-5.6-sol")
        self.assertEqual(command[command.index("--provider") + 1], "openai-codex")
        self.assertEqual(config["model"]["default"], "gpt-5.6-sol")
        self.assertEqual(config["agent"]["max_turns"], 33)

    def test_config_rewrite_keeps_user_provider_but_rebinds_studio_owned_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            with mock.patch.object(agentlab, "HERMES_HOME", profile):
                # Studio-owned block: a freshly loaded engine re-binds it.
                agentlab._write_hermes_config("http://127.0.0.1:8000", "local-a", 90)
                agentlab._write_hermes_config("http://127.0.0.1:9000", "local-b", 90)
                rebound = yaml.safe_load(
                    (profile / "config.yaml").read_text()
                )["model"]

                # The user switches provider in the TUI; a later engine load
                # must not drag Chat back onto the local endpoint.
                path = profile / "config.yaml"
                config = yaml.safe_load(path.read_text())
                config["model"] = {
                    "provider": "anthropic",
                    "default": "claude-opus-5",
                }
                path.write_text(yaml.safe_dump(config))
                agentlab._write_hermes_config(
                    "http://127.0.0.1:9000", "local-b", 90, respect_user_provider=True
                )
                preserved = yaml.safe_load(path.read_text())["model"]

                # Headless/CLI runs pass `--provider custom` on argv, so they
                # re-bind regardless of what the user picked in Chat.
                agentlab._write_hermes_config("http://127.0.0.1:9000", "local-b", 90)
                headless = yaml.safe_load(path.read_text())["model"]

        self.assertEqual(rebound["default"], "local-b")
        self.assertEqual(rebound["base_url"], "http://127.0.0.1:9000/v1")
        self.assertEqual(preserved["provider"], "anthropic")
        self.assertEqual(preserved["default"], "claude-opus-5")
        self.assertEqual(headless["provider"], "custom")
        self.assertEqual(headless["default"], "local-b")

    def test_http_terminal_creation_requires_browser_header(self):
        client = TestClient(server.app)
        spawn = mock.AsyncMock()
        with mock.patch.object(server, "_spawn_http_terminal", new=spawn):
            response = client.post(
                "/api/agentlab/terminal/sessions",
                json={"workspace": "."},
            )
        self.assertEqual(response.status_code, 403)
        spawn.assert_not_awaited()

    def test_http_terminal_fallback_streams_input_output_and_resize(self):
        bridge = mock.Mock()
        # One burst then silence: the lease pump drains continuously, so a
        # bridge that always returned bytes would never stop producing.
        bridge.read.side_effect = itertools.chain([b"ready"], itertools.repeat(b""))
        spawn = mock.AsyncMock(return_value=bridge)
        client = TestClient(server.app)
        headers = {"X-Spark-Studio-Terminal": "1"}

        with mock.patch.object(server, "_spawn_http_terminal", new=spawn):
            created = client.post(
                "/api/agentlab/terminal/sessions",
                headers=headers,
                json={"workspace": ".", "cols": 132, "rows": 40},
            )
        self.assertEqual(created.status_code, 200)
        session_id = created.json()["session_id"]
        self.assertEqual(created.json()["transport"], "https")

        inspected = client.get(
            f"/api/agentlab/terminal/sessions/{session_id}",
            headers=headers,
        )
        self.assertEqual(inspected.status_code, 200)
        self.assertTrue(inspected.json()["active"])

        output = client.get(
            f"/api/agentlab/terminal/sessions/{session_id}/output",
            headers=headers,
        )
        self.assertEqual(output.status_code, 200)
        self.assertEqual(base64.b64decode(output.json()["data"]), b"ready")
        self.assertFalse(output.json()["closed"])

        typed = client.post(
            f"/api/agentlab/terminal/sessions/{session_id}/input",
            headers=headers,
            json={"type": "input", "data": base64.b64encode(b"hello").decode()},
        )
        resized = client.post(
            f"/api/agentlab/terminal/sessions/{session_id}/input",
            headers=headers,
            json={"type": "resize", "cols": 144, "rows": 48},
        )
        closed = client.delete(
            f"/api/agentlab/terminal/sessions/{session_id}",
            headers=headers,
        )

        self.assertEqual(typed.status_code, 204)
        self.assertEqual(resized.status_code, 204)
        self.assertEqual(closed.status_code, 204)
        bridge.write.assert_called_once_with(b"hello")
        bridge.resize.assert_called_once_with(144, 48)
        bridge.close.assert_called()
        self.assertNotIn(session_id, server._TERMINAL_SESSIONS)

    def test_hermes_install_api_runs_guarded_official_installer(self):
        installed = {
            "installed": True,
            "ok": True,
            "binary": "/home/test/.local/bin/hermes",
            "version": "1.2.3",
        }
        client = TestClient(server.app)
        with mock.patch.object(
            server.agentlab, "install_hermes", return_value=installed
        ) as install:
            response = client.post("/api/agentlab/install")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["installed"])
        install.assert_called_once_with()

    def test_learning_api_defaults_to_profile_on_and_persists_choice(self):
        client = TestClient(server.app)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            agentlab, "HERMES_HOME", Path(tmp) / "hermes-profile"
        ):
            current = client.get("/api/agentlab/learning")
            updated = client.put(
                "/api/agentlab/learning",
                json={"user_profile_enabled": False},
            )
            saved = agentlab.hermes_learning_settings()

        self.assertEqual(current.status_code, 200)
        self.assertTrue(current.json()["user_profile_enabled"])
        self.assertEqual(updated.status_code, 200)
        self.assertFalse(updated.json()["user_profile_enabled"])
        self.assertTrue(updated.json()["restart_required"])
        self.assertFalse(saved["user_profile_enabled"])
        self.assertIn("memory", updated.json()["toolsets"])
        self.assertIn("skills", updated.json()["toolsets"])
        self.assertIn("session_search", updated.json()["toolsets"])

    def test_pending_learning_api_lists_isolated_memory_and_skill_cards(self):
        client = TestClient(server.app)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            agentlab, "HERMES_HOME", Path(tmp) / "hermes-profile"
        ):
            pending = agentlab.HERMES_HOME / "pending"
            (pending / "memory").mkdir(parents=True)
            (pending / "skills").mkdir(parents=True)
            (pending / "memory" / "a1b2c3d4.json").write_text(
                """{
                  "id": "a1b2c3d4", "subsystem": "memory", "action": "add",
                  "summary": "remember recipe preference", "created_at": 1,
                  "payload": {"target": "memory", "content": "Prefer spark-vllm-docker."}
                }"""
            )
            (pending / "skills" / "e5f6a7b8.json").write_text(
                """{
                  "id": "e5f6a7b8", "subsystem": "skills", "action": "write_file",
                  "summary": "update recipe skill", "created_at": 2,
                  "payload": {"name": "recipes", "file_path": "SKILL.md", "content": "Verify the recipe."}
                }"""
            )
            response = client.get("/api/agentlab/pending")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual(body["memory_count"], 1)
        self.assertEqual(body["skills_count"], 1)
        self.assertEqual(body["pending"][0]["preview"], "Prefer spark-vllm-docker.")
        self.assertEqual(body["pending"][1]["file_path"], "SKILL.md")

    def test_edits_to_external_skills_are_approvable_and_flagged(self):
        # Spark Studio registers its bundled agent-skills/ via
        # skills.external_dirs, exactly where Hermes' own _find_skill looks.
        # An edit there must not read as "skill does not exist".
        record = {
            "id": "a1b2c3d4",
            "subsystem": "skills",
            "action": "edit",
            "summary": "rewrite 'sparkrun-recipes'",
            "created_at": 1,
            "payload": {
                "action": "edit",
                "name": "sparkrun-recipes",
                "content": "---\nname: sparkrun-recipes\ndescription: Recipes.\n---\n\n# Recipes\n",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile"
            shared = root / "agent-skills"
            (shared / "sparkrun-recipes").mkdir(parents=True)
            (shared / "sparkrun-recipes" / "SKILL.md").write_text("# sparkrun recipes\n")
            profile.mkdir(parents=True)
            (profile / "config.yaml").write_text(
                yaml.safe_dump({"skills": {"external_dirs": [str(shared)]}})
            )
            pending = profile / "pending" / "skills"
            pending.mkdir(parents=True)
            (pending / "a1b2c3d4.json").write_text(json.dumps(record))
            with mock.patch.object(agentlab, "HERMES_HOME", profile):
                (card,) = agentlab.hermes_pending_writes()
                # An edit to a genuinely unknown skill still blocks.
                record["payload"]["name"] = "never-installed"
                (pending / "a1b2c3d4.json").write_text(json.dumps(record))
                (unknown,) = agentlab.hermes_pending_writes()

        self.assertEqual(card["blocked_reason"], "")
        self.assertIn("outside the Hermes profile", card["skill_scope"])
        self.assertIn("does not exist", unknown["blocked_reason"])
        self.assertEqual(unknown["skill_scope"], "")

    def test_pending_learning_api_resolves_only_explicit_valid_action(self):
        client = TestClient(server.app)
        result = {
            "ok": True,
            "action": "approve",
            "id": "a1b2c3d4",
            "subsystem": "memory",
            "message": "Approved 1 memory write(s).",
            "restart_recommended": True,
        }
        with mock.patch.object(
            server.agentlab, "resolve_hermes_pending", return_value=result
        ) as resolve:
            response = client.post(
                "/api/agentlab/pending/memory/a1b2c3d4/approve"
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["restart_recommended"])
        resolve.assert_called_once_with("memory", "a1b2c3d4", "approve")

        invalid = client.post("/api/agentlab/pending/memory/not-an-id/approve")
        self.assertEqual(invalid.status_code, 400)

    def test_pending_resolution_keeps_spark_studio_hermes_home(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout='{"ok": true, "message": "Approved 1 memory write(s)."}\n',
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            source = Path(tmp) / "hermes-agent"
            python = source / "venv" / "bin" / "python"
            pending = profile / "pending" / "memory"
            pending.mkdir(parents=True)
            (pending / "a1b2c3d4.json").write_text("{}")
            with (
                mock.patch.object(agentlab, "HERMES_HOME", profile),
                mock.patch.object(
                    agentlab, "_hermes_agent_runtime", return_value=(source, python)
                ),
                mock.patch.object(agentlab, "_run", return_value=completed) as run,
            ):
                result = agentlab.resolve_hermes_pending(
                    "memory", "a1b2c3d4", "approve"
                )

        self.assertTrue(result["ok"])
        call = run.call_args
        self.assertEqual(call.args[0][-3:], ["memory", "approve", "a1b2c3d4"])
        self.assertEqual(call.kwargs["env"]["HERMES_HOME"], str(profile))
        self.assertNotIn("PYTHONHOME", call.kwargs["env"])

    def test_pending_cards_repair_legacy_skill_and_block_its_file_until_created(self):
        create_record = {
            "id": "a1b2c3d4",
            "subsystem": "skills",
            "action": "create",
            "summary": "create recipe-helper",
            "created_at": 1,
            "payload": {
                "action": "create",
                "name": "recipe-helper",
                "category": "mlops/inference",
                "content": """---
name: recipe-helper
description: This description is deliberately longer than the current Hermes routing limit and needs compatibility repair.
---

# Recipe helper

Use this skill when repairing recipes.
""",
            },
        }
        file_record = {
            "id": "e5f6a7b8",
            "subsystem": "skills",
            "action": "write_file",
            "summary": "add reference",
            "created_at": 2,
            "payload": {
                "action": "write_file",
                "name": "recipe-helper",
                "file_path": "references/checks.md",
                "content": "Verify the recipe.",
            },
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            agentlab, "HERMES_HOME", Path(tmp) / "profile"
        ):
            pending = agentlab.HERMES_HOME / "pending" / "skills"
            pending.mkdir(parents=True)
            for record in (create_record, file_record):
                (pending / f"{record['id']}.json").write_text(
                    json.dumps(record)
                )
            cards = agentlab.hermes_pending_writes()

        create, supporting_file = cards
        self.assertEqual(create["name"], "recipe-helper")
        self.assertTrue(
            any("mlops-inference" in note for note in create["repair_notes"])
        )
        self.assertIn("Use for recipe helper workflows.", create["preview"])
        self.assertTrue(create["repair_notes"])
        self.assertFalse(create["blocked_reason"])
        self.assertEqual(supporting_file["depends_on"], "a1b2c3d4")
        self.assertIn("create card", supporting_file["blocked_reason"])
        self.assertIn("current Hermes schema", supporting_file["repair_notes"][0])

    def test_pending_skill_repair_is_written_before_official_approval(self):
        record = {
            "id": "a1b2c3d4",
            "subsystem": "skills",
            "action": "create",
            "summary": "create html game skill",
            "created_at": 1,
            "payload": {
                "action": "create",
                "name": "",
                "category": "creative",
                "content": "# single-file-html-game\n\nBuild one-file games.\n",
            },
        }
        completed = SimpleNamespace(
            returncode=0,
            stdout='{"ok": true, "message": "Approved 1 skills write(s)."}\n',
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            source = Path(tmp) / "hermes-agent"
            python = source / "venv" / "bin" / "python"
            pending = profile / "pending" / "skills"
            pending.mkdir(parents=True)
            path = pending / "a1b2c3d4.json"
            path.write_text(json.dumps(record))

            def inspect_repaired(*_args, **_kwargs):
                repaired = json.loads(path.read_text())
                self.assertEqual(repaired["payload"]["name"], "single-file-html-game")
                self.assertTrue(repaired["payload"]["content"].startswith("---\n"))
                return completed

            with (
                mock.patch.object(agentlab, "HERMES_HOME", profile),
                mock.patch.object(
                    agentlab, "_hermes_agent_runtime", return_value=(source, python)
                ),
                mock.patch.object(agentlab, "_run", side_effect=inspect_repaired),
            ):
                result = agentlab.resolve_hermes_pending(
                    "skills", "a1b2c3d4", "approve"
                )

        self.assertTrue(result["ok"])
        self.assertTrue(result["repaired"])
        self.assertTrue(result["repair_notes"])


    def test_browser_command_launches_real_tui_with_agent_tools(self):
        command = hermes_terminal.browser_tui_command("/opt/hermes", "fixture-model")
        self.assertEqual(command[:2], ["/opt/hermes", "--tui"])
        self.assertEqual(command[command.index("--model") + 1], "fixture-model")
        self.assertEqual(command[command.index("--provider") + 1], "custom")
        self.assertNotIn("--max-turns", command)
        self.assertIn("file,terminal,mcp-sparkstudio,memory,skills,session_search", command)

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
            # The lease id arrives first, as a text frame on the control
            # channel; the pty stream itself is always binary.
            lease = json.loads(websocket.receive_text())
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
        self.assertEqual(lease["type"], "session")
        self.assertFalse(lease["resumed"])
        self.assertIn(b"ready", output)
        self.assertIn(b"got-hello", output)
        # End the lease the way Stop does. Popping it alone would leave the pty
        # and its pump thread running for the rest of the process, and a live
        # extra thread makes every later pty.fork() a deadlock risk.
        _close_terminal_leases()


class TerminalLeaseTests(unittest.TestCase):
    """A dropped socket must detach the PTY, never kill it."""

    def setUp(self):
        _close_terminal_leases()
        self.addCleanup(_close_terminal_leases)

    @staticmethod
    def _shell():
        # Echoes each line back, so the test can prove the SAME process is
        # still holding state after the reconnect.
        return [
            "/bin/sh",
            "-c",
            "count=0; while read line; do "
            'count=$((count+1)); printf ":%s-%s" "$count" "$line"; done',
        ]

    @staticmethod
    def _read_until(websocket, needle, limit=40):
        chunks = []
        for _ in range(limit):
            try:
                message = websocket.receive()
            except WebSocketDisconnect:
                break
            payload = message.get("bytes")
            if payload:
                chunks.append(payload)
                if needle in b"".join(chunks):
                    break
        return b"".join(chunks)

    @unittest.skipUnless(hermes_terminal.PtyBridge.available(), "POSIX PTY required")
    def test_reload_reattaches_to_the_same_running_process(self):
        client = TestClient(server.app)
        headers = {"origin": "http://testserver"}
        with (
            mock.patch.object(server, "_resolve_terminal_endpoint", new=mock.AsyncMock(return_value={})),
            mock.patch.object(
                server.hermes_terminal,
                "prepare_browser_tui",
                return_value=(self._shell(), os.environ.copy()),
            ),
        ):
            with client.websocket_connect("/api/agentlab/terminal?workspace=.", headers=headers) as ws:
                lease = json.loads(ws.receive_text())
                ws.send_bytes(b"alpha\n")
                first = self._read_until(ws, b":1-alpha")

            # The socket is gone; the lease and its process are not.
            session = server._TERMINAL_SESSIONS[lease["id"]]
            self.assertFalse(session["attached"])
            self.assertFalse(session["closed"])
            os.kill(session["bridge"].pid, 0)  # raises if the agent was killed

            # Reload: same id, same shell, counter carries on from 1.
            with client.websocket_connect(
                f"/api/agentlab/terminal?workspace=.&session={lease['id']}", headers=headers
            ) as ws:
                resumed = json.loads(ws.receive_text())
                ws.send_bytes(b"beta\n")
                second = self._read_until(ws, b":2-beta")

        self.assertTrue(lease["id"])
        self.assertFalse(lease["resumed"])
        self.assertEqual(resumed["id"], lease["id"])
        self.assertTrue(resumed["resumed"])
        self.assertIn(b":1-alpha", first)
        self.assertIn(b":2-beta", second)

    @unittest.skipUnless(hermes_terminal.PtyBridge.available(), "POSIX PTY required")
    def test_output_produced_while_detached_is_replayed_on_reattach(self):
        client = TestClient(server.app)
        headers = {"origin": "http://testserver"}
        with (
            mock.patch.object(server, "_resolve_terminal_endpoint", new=mock.AsyncMock(return_value={})),
            mock.patch.object(
                server.hermes_terminal,
                "prepare_browser_tui",
                return_value=(self._shell(), os.environ.copy()),
            ),
        ):
            with client.websocket_connect("/api/agentlab/terminal?workspace=.", headers=headers) as ws:
                lease = json.loads(ws.receive_text())

            # Nobody is attached: the pump has to keep draining, or Hermes
            # would block on a full pipe instead of buffering for the reload.
            session = server._TERMINAL_SESSIONS[lease["id"]]
            session["bridge"].write(b"offline\n")
            deadline = time.time() + 5
            while b":1-offline" not in bytes(session["buffer"]) and time.time() < deadline:
                time.sleep(0.05)

            with client.websocket_connect(
                f"/api/agentlab/terminal?workspace=.&session={lease['id']}", headers=headers
            ) as ws:
                json.loads(ws.receive_text())
                replayed = self._read_until(ws, b":1-offline")

        self.assertIn(b":1-offline", replayed)

    @unittest.skipUnless(hermes_terminal.PtyBridge.available(), "POSIX PTY required")
    def test_stop_and_expiry_end_the_lease_for_real(self):
        client = TestClient(server.app)
        headers = {"origin": "http://testserver"}
        with (
            mock.patch.object(server, "_resolve_terminal_endpoint", new=mock.AsyncMock(return_value={})),
            mock.patch.object(
                server.hermes_terminal,
                "prepare_browser_tui",
                return_value=(self._shell(), os.environ.copy()),
            ),
        ):
            with client.websocket_connect("/api/agentlab/terminal?workspace=.", headers=headers) as ws:
                stopped = json.loads(ws.receive_text())
            with client.websocket_connect("/api/agentlab/terminal?workspace=.", headers=headers) as ws:
                expired = json.loads(ws.receive_text())

            # Stop is explicit and immediate.
            deleted = client.delete(
                f"/api/agentlab/terminal/sessions/{stopped['id']}",
                headers={"X-Spark-Studio-Terminal": "1"},
            )
            # An abandoned lease is reaped once its TTL passes.
            server._TERMINAL_SESSIONS[expired["id"]]["last_seen"] -= (
                server._HTTP_TERMINAL_TTL_SECONDS + 1
            )
            server._prune_http_terminal_sessions()

            # A stale id must not silently adopt someone else's session.
            with client.websocket_connect(
                f"/api/agentlab/terminal?workspace=.&session={stopped['id']}", headers=headers
            ) as ws:
                fresh = json.loads(ws.receive_text())

        self.assertEqual(deleted.status_code, 204)
        self.assertNotIn(stopped["id"], server._TERMINAL_SESSIONS)
        self.assertNotIn(expired["id"], server._TERMINAL_SESSIONS)
        self.assertNotEqual(fresh["id"], stopped["id"])
        self.assertFalse(fresh["resumed"])


class _FakeEngine:
    """A minimal OpenAI-compatible engine on a real socket."""

    def __init__(self, model: str):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        served = model

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send({"data": [{"id": served}]})
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                if body.get("model") != served:
                    self._send({"error": f"unknown model {body.get('model')}"}, 400)
                    return
                self._send({"served": served})

            def _send(self, obj, code=200):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.model = model
        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def as_run(self):
        return SimpleNamespace(id="run", label=self.model, url=self.url, ready=True)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class EnginePassthroughTests(unittest.TestCase):
    """`/api/engine/v1` is the stable URL Hermes' /model picker points at."""

    def test_passthrough_follows_the_active_run_across_a_recipe_swap(self):
        first, second = _FakeEngine("recipe-one/model-A"), _FakeEngine("recipe-two/model-B")
        client = TestClient(server.app)
        self.addCleanup(first.close)
        self.addCleanup(second.close)

        # A recipe the agent launched mid-session is discoverable straight away.
        with mock.patch.object(server.runner, "active", return_value=first.as_run()):
            listed = client.get("/api/engine/v1/models").json()
            pinned = client.post(
                "/api/engine/v1/chat/completions",
                json={"model": "recipe-one/model-A", "messages": []},
            )

        # Swapping to another recipe on a different port needs no config change,
        # and a model id pinned before the swap retargets instead of 400ing.
        with mock.patch.object(server.runner, "active", return_value=second.as_run()):
            relisted = client.get("/api/engine/v1/models").json()
            stale = client.post(
                "/api/engine/v1/chat/completions",
                json={"model": "recipe-one/model-A", "messages": []},
            )

        self.assertEqual([m["id"] for m in listed["data"]], ["recipe-one/model-A"])
        self.assertEqual(pinned.json()["served"], "recipe-one/model-A")
        self.assertEqual([m["id"] for m in relisted["data"]], ["recipe-two/model-B"])
        self.assertEqual(stale.status_code, 200)
        self.assertEqual(stale.json()["served"], "recipe-two/model-B")

    def test_passthrough_reports_no_engine_and_refuses_unlisted_paths(self):
        engine = _FakeEngine("recipe/model")
        client = TestClient(server.app)
        self.addCleanup(engine.close)

        with mock.patch.object(server.runner, "active", return_value=None):
            detached = client.get("/api/engine/v1/models")
        with mock.patch.object(server.runner, "active", return_value=engine.as_run()):
            blocked = client.post("/api/engine/v1/load_lora_adapter", json={})

        self.assertEqual(detached.status_code, 503)
        self.assertEqual(blocked.status_code, 404)

    def test_config_registers_passthrough_without_disturbing_user_providers(self):
        mine = {"name": "My LM Studio", "base_url": "http://127.0.0.1:1234/v1"}
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            profile.mkdir(parents=True)
            (profile / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "custom_providers": [
                            mine,
                            # a stale row from an earlier dashboard port
                            {"name": "Spark Studio", "base_url": "http://127.0.0.1:9999/api/engine/v1"},
                        ]
                    }
                )
            )
            with mock.patch.object(agentlab, "HERMES_HOME", profile):
                agentlab._write_hermes_config(
                    None, None, 90, studio_url="http://127.0.0.1:7860"
                )
            providers = yaml.safe_load((profile / "config.yaml").read_text())[
                "custom_providers"
            ]

        studio = [p for p in providers if p["name"] == "Spark Studio"]
        self.assertIn(mine, providers)
        self.assertEqual(len(studio), 1)
        self.assertEqual(studio[0]["base_url"], "http://127.0.0.1:7860/api/engine/v1")
        self.assertNotIn("models", studio[0])
        self.assertTrue(studio[0]["api_key"])

    def test_declared_models_track_the_active_engine_without_probing(self):
        # The TUI's normal /model open renders non-current custom rows from
        # their declared models: list WITHOUT probing (Hermes'
        # build_model_options_payload passes probe_current_custom_provider, not
        # probe_custom_providers). So the declared list is the picker, and the
        # dashboard must keep it truthful across launch, swap, and shutdown.
        mine = {"name": "My LM Studio", "base_url": "http://127.0.0.1:1234/v1"}
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            profile.mkdir(parents=True)
            (profile / "config.yaml").write_text(
                yaml.safe_dump({"custom_providers": [mine]})
            )
            with mock.patch.object(agentlab, "HERMES_HOME", profile):
                # Chat launched while bound to a local engine.
                agentlab._write_hermes_config(
                    "http://127.0.0.1:8000/v1", "recipe-a/model", 90,
                    studio_url="http://127.0.0.1:7860",
                )
                path = profile / "config.yaml"
                launched = yaml.safe_load(path.read_text())["custom_providers"][-1]

                # Watchdog: recipe swapped mid-session.
                swapped_changed = agentlab.update_studio_provider_models(["recipe-b/model"])
                swapped = yaml.safe_load(path.read_text())["custom_providers"][-1]
                noop_changed = agentlab.update_studio_provider_models(["recipe-b/model"])

                # Watchdog: engine unloaded.
                agentlab.update_studio_provider_models([])
                cleared_rows = yaml.safe_load(path.read_text())["custom_providers"]

        self.assertEqual(launched["models"], ["recipe-a/model"])
        self.assertTrue(swapped_changed)
        self.assertEqual(swapped["models"], ["recipe-b/model"])
        self.assertFalse(noop_changed)
        self.assertNotIn("models", cleared_rows[-1])
        self.assertIn(mine, cleared_rows)  # other rows never touched

    def test_provider_row_keeps_live_probing_after_hermes_caches_models(self):
        # Hermes gates live /models discovery on
        #   bool(api_key) or not has_explicit_models
        # (hermes_cli/model_switch.py). It also writes discovered models back
        # into the entry, which turns the second clause False — so without a
        # key the row pins itself to whatever was loaded the first time it was
        # read, and /model keeps offering a recipe that is no longer running.
        def hermes_would_probe(entry):
            api_key = (entry.get("api_key") or "").strip()
            has_explicit_models = bool(entry.get("models"))
            return bool(entry.get("base_url")) and (
                bool(api_key) or not has_explicit_models
            )

        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            with mock.patch.object(agentlab, "HERMES_HOME", profile):
                agentlab._write_hermes_config(
                    None, None, 90, studio_url="http://127.0.0.1:7860"
                )
                path = profile / "config.yaml"
                fresh = yaml.safe_load(path.read_text())["custom_providers"][-1]

                # Hermes persists what it discovered into our row.
                config = yaml.safe_load(path.read_text())
                config["custom_providers"][-1]["models"] = ["recipe-a/model"]
                path.write_text(yaml.safe_dump(config))
                cached = yaml.safe_load(path.read_text())["custom_providers"][-1]

                # A relaunch also drops the cache, so a stale id never persists.
                agentlab._write_hermes_config(
                    None, None, 90, studio_url="http://127.0.0.1:7860"
                )
                relaunched = yaml.safe_load(path.read_text())["custom_providers"][-1]

        self.assertTrue(hermes_would_probe(fresh))
        self.assertTrue(hermes_would_probe(cached))
        self.assertNotIn("models", relaunched)


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


    def test_registered_external_is_marked_for_watchdog(self):
        manager = runners.Runner()
        with mock.patch.object(runners.db, "runs_insert"):
            run = manager.register_external(
                "vllm", "fixture-model", "http://127.0.0.1:9999"
            )
        self.assertTrue(run.meta["_external"])

    def test_dead_external_endpoint_is_retired_after_two_failed_probes(self):
        run = runners.Run(
            id="external", engine="vllm", recipe_id=None, cmd=["external"], env={}
        )
        run.status = "running"
        run.ready = True
        run.url = "http://127.0.0.1:9999"
        run.probe_failures = server._EXTERNAL_PROBE_FAILURES - 1
        run.meta["_external"] = True
        client = mock.AsyncMock()
        client.get.side_effect = RuntimeError("endpoint is down")
        context = mock.MagicMock()
        context.__aenter__ = mock.AsyncMock(return_value=client)
        context.__aexit__ = mock.AsyncMock(return_value=None)
        with (
            mock.patch.object(server.runner, "runs", {run.id: run}),
            mock.patch("httpx.AsyncClient", return_value=context),
            mock.patch.object(runners.db, "runs_update"),
        ):
            asyncio.run(server._watchdog_tick(1))
        self.assertEqual(run.status, "exited")
        self.assertFalse(server.runner.active())
        self.assertIn("external endpoint stopped answering", "\n".join(run.ring))

    def test_sparkrun_start_tracks_launch_target_and_tp_port(self):
        run = runners.Run(
            id="spark", engine="sparkrun", recipe_id=21, cmd=["sparkrun"], env={}
        )
        saved_recipe = {"args": {"_sparkrun": {"port": 8888}}}
        with (
            mock.patch.object(server, "engine_available", return_value=True),
            mock.patch.object(server.sparkrun_service, "canonical_recipe_ref", return_value="@studio/prod"),
            mock.patch.object(server.sparkrun_service, "resolve_recipe_target", return_value="/srv/recipes/prod.yaml"),
            mock.patch.object(server, "_ensure_sparkrun_recipe", return_value=21),
            mock.patch.object(server.db, "recipes_get", return_value=saved_recipe),
            mock.patch.object(server.runner, "start", return_value=run) as start,
        ):
            result = server._start_sparkrun("@studio/prod", 2)

        self.assertIs(result, run)
        self.assertEqual(run.port, 8888)
        self.assertEqual(run.url, "http://127.0.0.1:8888")
        self.assertEqual(start.call_args.kwargs["meta"]["ref"], "@studio/prod")
        self.assertEqual(start.call_args.kwargs["meta"]["launch_target"], "/srv/recipes/prod.yaml")

    def test_sparkrun_teardown_snapshots_container_artifacts(self):
        manager = runners.Runner()
        run = runners.Run(
            id="spark",
            engine="sparkrun",
            recipe_id=None,
            cmd=["sparkrun"],
            env={},
            managed_containers=["sparkrun_job_node_0"],
            meta={"jobid": "job"},
            status="running",
        )

        def fake_run(cmd, stdout=None, **_kwargs):
            if stdout:
                stdout.write("ran: " + " ".join(cmd) + "\n")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(runners, "RUN_LOG_DIR", Path(tmp)), \
             mock.patch.object(runners.sparkrun_service, "sparkrun_bin", return_value="/usr/bin/sparkrun"), \
             mock.patch.object(runners.shutil, "which", return_value="/usr/bin/docker"), \
             mock.patch.object(runners.subprocess, "run", side_effect=fake_run):
            manager._snapshot_before_teardown(run)
            out = Path(tmp) / "spark-artifacts"
            self.assertTrue((out / "sparkrun-status.txt").exists())
            self.assertTrue((out / "sparkrun-logs.txt").exists())
            self.assertTrue((out / "sparkrun_job_node_0-serve-log.txt").exists())
            self.assertIn("preserved teardown logs", "\n".join(run.ring))


if __name__ == "__main__":
    unittest.main()
