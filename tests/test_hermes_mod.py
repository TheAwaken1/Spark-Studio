import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
