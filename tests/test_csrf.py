"""Cross-site write protection for the dashboard API.

Same-origin CORS stops cross-site reads, but preflight-free writes (forms,
``no-cors`` fetch) from a malicious page could still reach the loopback/LAN
API. The middleware rejects any mutating /api request whose Origin header does
not belong to the dashboard.
"""

import unittest
from unittest import mock

from fastapi.testclient import TestClient

import server


class CrossSiteWriteGuardTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_foreign_origin_write_is_rejected_before_the_handler(self):
        spawn = mock.AsyncMock()
        with mock.patch.object(server, "_spawn_http_terminal", new=spawn):
            response = self.client.post(
                "/api/agentlab/terminal/sessions",
                json={"workspace": "."},
                headers={
                    "Origin": "https://evil.example",
                    "X-Spark-Studio-Terminal": "1",
                },
            )
        self.assertEqual(response.status_code, 403)
        self.assertIn("cross-origin", response.json()["detail"])
        spawn.assert_not_awaited()

    def test_null_origin_write_is_rejected(self):
        response = self.client.post(
            "/api/agentlab/install",
            headers={"Origin": "null"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("cross-origin", response.json()["detail"])

    def test_same_origin_write_reaches_the_handler(self):
        # The handler then rejects for its own reason (missing terminal
        # header), proving the CSRF guard let same-origin traffic through.
        response = self.client.post(
            "/api/agentlab/terminal/sessions",
            json={"workspace": "."},
            headers={"Origin": "http://testserver"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("terminal header", response.json()["detail"])

    def test_forwarded_proxy_host_matches_like_the_websocket_check(self):
        response = self.client.post(
            "/api/agentlab/terminal/sessions",
            json={"workspace": "."},
            headers={
                "Origin": "https://spark.local:8443",
                "X-Forwarded-Host": "spark.local:8443",
            },
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("terminal header", response.json()["detail"])

    def test_headerless_non_browser_clients_are_unaffected(self):
        # The sparkstudio CLI and MCP child use httpx, which sends no Origin.
        response = self.client.post(
            "/api/agentlab/terminal/sessions",
            json={"workspace": "."},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("terminal header", response.json()["detail"])

    def test_reads_are_not_blocked_by_the_guard(self):
        with mock.patch.object(
            server.agentlab, "hermes_status", return_value={"installed": False}
        ):
            response = self.client.get(
                "/api/agentlab/status",
                headers={"Origin": "https://evil.example"},
            )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
