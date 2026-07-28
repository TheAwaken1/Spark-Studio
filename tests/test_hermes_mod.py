import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from fastapi.testclient import TestClient

import hermes_mod_service
import server


class HermesModBridgeTests(unittest.TestCase):
    def test_html_assets_are_scoped_to_iframe_bridge(self):
        source = b'<link rel="stylesheet" href="/styles.css"><script src="/app.js"></script>'
        rewritten = hermes_mod_service.rewrite_response("", "text/html", source).decode()
        self.assertIn('href="styles.css"', rewritten)
        self.assertIn('src="app.js"', rewritten)
        self.assertNotIn('href="/styles.css"', rewritten)

    def test_javascript_routes_api_and_uses_browser_image_picker(self):
        source = b"""async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options
  });
}
async function chooseHeroImage() {
  const data = await api('/api/pick-hero-image', { method: 'POST' });
}

function handleHeroStyleChange() {}
setStatus('Opening system image picker...', 'normal');
"""
        rewritten = hermes_mod_service.rewrite_response(
            "app.js", "application/javascript", source
        ).decode()
        self.assertIn("SPARK_STUDIO_BRIDGE", rewritten)
        self.assertIn(hermes_mod_service.BRIDGE_HEADER, rewritten)
        self.assertIn("input.type = 'file'", rewritten)
        self.assertIn("Opening browser image picker", rewritten)
        self.assertNotIn("/api/pick-hero-image", rewritten)

    def test_bridge_api_rejects_requests_without_injected_token(self):
        client = TestClient(server.app)
        response = client.get("/api/hermes-mod/ui/api/status")
        self.assertEqual(response.status_code, 403)
        self.assertIn("bridge token", response.text)

    def test_sandbox_preflight_is_limited_to_bridge(self):
        client = TestClient(server.app)
        response = client.options(
            "/api/hermes-mod/ui/api/status",
            headers={"origin": "null"},
        )
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.headers["access-control-allow-origin"], "null")
        self.assertIn(
            hermes_mod_service.BRIDGE_HEADER.lower(),
            response.headers["access-control-allow-headers"].lower(),
        )

    def test_status_reports_spark_studio_profile_not_personal_profile(self):
        fake_status = {
            "installed": False,
            "healthy": False,
            "profile": str(hermes_mod_service.agentlab.HERMES_HOME),
        }
        client = TestClient(server.app)
        with mock.patch.object(
            server.hermes_mod_service,
            "status",
            new=mock.AsyncMock(return_value=fake_status),
        ):
            response = client.get("/api/hermes-mod/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["profile"], str(hermes_mod_service.agentlab.HERMES_HOME))
        self.assertNotEqual(response.json()["profile"], str(hermes_mod_service.Path.home() / ".hermes"))


class HermesModSkinTests(unittest.TestCase):
    def test_multiline_art_markup_is_repeated_for_each_ink_tui_line(self):
        source = "[bold #c93c24]\nSPARK\nSTUDIO\n[/]"
        expected = "[bold #c93c24]SPARK[/]\n[bold #c93c24]STUDIO[/]"
        self.assertEqual(hermes_mod_service.normalize_tui_art_markup(source), expected)

        hero = "[#e9926d]HELLO\nWORLD[/]"
        self.assertEqual(
            hermes_mod_service.normalize_tui_art_markup(hero),
            "[#e9926d]HELLO[/]\n[#e9926d]WORLD[/]",
        )

    def test_existing_per_line_gradient_markup_is_unchanged(self):
        source = "[#111111]FIRST[/]\n[#222222]SECOND[/]"
        self.assertEqual(hermes_mod_service.normalize_tui_art_markup(source), source)

    def test_skin_save_request_repairs_logo_and_hero_only(self):
        payload = {
            "name": "spark",
            "banner_logo": "[bold #abcdef]\nLOGO\n[/]",
            "banner_hero": "[#123456]HERO\nART[/]",
            "accent": "#fedcba",
        }
        body = hermes_mod_service.normalize_skin_request(
            "PUT",
            "api/skins/spark",
            json.dumps(payload).encode(),
        )
        repaired = json.loads(body)
        self.assertEqual(repaired["banner_logo"], "[bold #abcdef]LOGO[/]")
        self.assertEqual(
            repaired["banner_hero"],
            "[#123456]HERO[/]\n[#123456]ART[/]",
        )
        self.assertEqual(repaired["accent"], "#fedcba")

    def test_repair_saved_skins_migrates_existing_custom_skin(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            skins = home / "skins"
            skins.mkdir()
            skin_path = skins / "spark.yaml"
            skin_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "spark",
                        "banner_logo": "[bold #c93c24]\nONE\nTWO\n[/]",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            with mock.patch.object(hermes_mod_service.agentlab, "HERMES_HOME", home):
                repaired = hermes_mod_service.repair_saved_skins()
            self.assertEqual(repaired, ["spark"])
            payload = yaml.safe_load(skin_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["banner_logo"],
                "[bold #c93c24]ONE[/]\n[bold #c93c24]TWO[/]",
            )

    def test_delete_active_custom_skin_resets_default_and_preserves_others(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            skins = home / "skins"
            skins.mkdir()
            (skins / "spark.yaml").write_text("name: spark\n", encoding="utf-8")
            (skins / "keep.yaml").write_text("name: keep\n", encoding="utf-8")
            (home / "config.yaml").write_text(
                "display:\n  skin: spark\n",
                encoding="utf-8",
            )
            with mock.patch.object(hermes_mod_service.agentlab, "HERMES_HOME", home):
                result = hermes_mod_service.delete_user_skin("spark")
            self.assertEqual(result["active_skin"], "default")
            self.assertFalse((skins / "spark.yaml").exists())
            self.assertTrue((skins / "keep.yaml").is_file())
            config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["display"]["skin"], "default")

    def test_delete_skin_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            hermes_mod_service.delete_user_skin("../personal")

    def test_delete_skin_api_calls_isolated_profile_service(self):
        client = TestClient(server.app)
        result = {"deleted": "spark", "active_skin": "default", "user_skins": []}
        with mock.patch.object(
            server.hermes_mod_service,
            "delete_user_skin",
            return_value=result,
        ) as delete:
            response = client.delete("/api/hermes-mod/skins/spark")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result)
        delete.assert_called_once_with("spark")

    def test_reinstall_request_does_not_reset_running_state(self):
        old_state = hermes_mod_service._state
        try:
            hermes_mod_service._state = "ready"
            fake_status = {"installed": True, "running": True, "state": "ready"}
            with (
                mock.patch.object(hermes_mod_service, "installed", return_value=True),
                mock.patch.object(
                    hermes_mod_service,
                    "status",
                    new=mock.AsyncMock(return_value=fake_status),
                ),
            ):
                result = asyncio.run(hermes_mod_service.install())
            self.assertEqual(result, fake_status)
            self.assertEqual(hermes_mod_service._state, "ready")
        finally:
            hermes_mod_service._state = old_state

    def test_use_original_skin_keeps_custom_skins(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            skins = home / "skins"
            skins.mkdir()
            custom = skins / "spark.yaml"
            custom.write_text("name: spark\n", encoding="utf-8")
            (home / "config.yaml").write_text(
                "display:\n  skin: spark\n", encoding="utf-8"
            )
            with mock.patch.object(hermes_mod_service.agentlab, "HERMES_HOME", home):
                result = hermes_mod_service.use_original_skin()
            self.assertEqual(result["active_skin"], "default")
            self.assertTrue(custom.is_file())


if __name__ == "__main__":
    unittest.main()
