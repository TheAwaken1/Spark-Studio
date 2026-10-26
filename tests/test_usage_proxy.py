"""End to end: a request through /api/engine/v1 is recorded and shows in the usage API.

A small fake OpenAI-style engine runs on localhost and streams a reasoning
phase, an answer, and a final usage chunk. Requests go through the real proxy
route; the database is a throwaway file.
"""

import json
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import db
import usage

try:
    from fastapi.testclient import TestClient

    import server
except Exception:  # noqa: BLE001 - optional deps missing in a minimal env
    TestClient = None
    server = None


def _sse_lines():
    events = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "let me"}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": " think"}}]},
        {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
        {"choices": [{"index": 0, "delta": {"content": " there"}}]},
        {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 4,
                                  "prompt_tokens_details": {"cached_tokens": 16}}},
    ]
    return [f"data: {json.dumps(e)}\n\n" for e in events] + ["data: [DONE]\n\n"]


class _FakeEngine(BaseHTTPRequestHandler):
    received: list = []

    def log_message(self, *args):  # keep test output quiet
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fake-model", "owned_by": "vllm"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        _FakeEngine.received.append(payload)
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for line in _sse_lines():
                self.wfile.write(line.encode())
            return
        body = json.dumps({
            "choices": [{"message": {"content": "Hi", "reasoning_content": "hm"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0}},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@unittest.skipIf(TestClient is None, "fastapi test client or server deps not available")
class EngineProxyUsageTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "spark_studio.db"
        with mock.patch.object(db, "DB_PATH", path):
            conn = db._connect()
        self.addCleanup(conn.close)
        for patcher in (mock.patch.object(db, "_conn", conn),):
            patcher.start()
            self.addCleanup(patcher.stop)
        db.init()

        engine = HTTPServer(("127.0.0.1", 0), _FakeEngine)
        threading.Thread(target=engine.serve_forever, daemon=True).start()
        self.addCleanup(engine.server_close)
        self.addCleanup(engine.shutdown)
        run = SimpleNamespace(id="run-test", url=f"http://127.0.0.1:{engine.server_address[1]}")
        patcher = mock.patch.object(server.runner, "active", return_value=run)
        patcher.start()
        self.addCleanup(patcher.stop)
        usage._live.clear()
        self.client = TestClient(server.app)

    def test_streamed_request_is_recorded_with_usage_and_thinking(self):
        resp = self.client.post(
            "/api/engine/v1/chat/completions",
            json={"model": "fake-model", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(resp.status_code, 200)
        # The client still sees the engine's answer unchanged.
        self.assertIn("Hello", resp.text)

        rows = db.usage_list(since=0)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["endpoint"], "/v1/chat/completions")
        self.assertEqual(row["model"], "fake-model")
        self.assertEqual(row["prompt_tokens"], 40)
        self.assertEqual(row["completion_tokens"], 4)
        self.assertEqual(row["cache_hit_pct"], 40.0)
        self.assertEqual(row["thinking"], 1)
        self.assertEqual(row["stream"], 1)
        self.assertEqual(usage.live_snapshot()["streams_now"], 0)

        summary = self.client.get("/api/usage/summary?period=all").json()
        self.assertEqual(summary["totals"]["requests"], 1)
        self.assertEqual(summary["by_model"][0]["model"], "fake-model")

        detail = self.client.get(f"/api/usage/events/{row['id']}").json()
        self.assertEqual(detail["id"], row["id"])
        self.assertTrue(detail["samples"])

    def test_non_streamed_request_is_recorded(self):
        resp = self.client.post(
            "/api/engine/v1/chat/completions",
            json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(resp.status_code, 200)
        row = db.usage_list(since=0)[0]
        self.assertEqual(row["stream"], 0)
        self.assertEqual(row["completion_tokens"], 2)
        self.assertEqual(row["thinking"], 1)

    def test_thinking_path_turns_thinking_on_and_default_path_does_not(self):
        _FakeEngine.received.clear()
        msg = [{"role": "user", "content": "hi"}]
        self.client.post("/api/engine/think/v1/chat/completions",
                         json={"model": "fake-model", "messages": msg})
        self.client.post("/api/engine/v1/chat/completions",
                         json={"model": "fake-model", "messages": msg})
        forced, default = _FakeEngine.received
        self.assertEqual(forced["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("chat_template_kwargs", default)
        endpoints = sorted(r["endpoint"] for r in db.usage_list(since=0))
        self.assertEqual(endpoints, ["/v1/chat/completions", "/v1/chat/completions · thinking"])

    def test_unknown_period_is_rejected(self):
        self.assertEqual(self.client.get("/api/usage/summary?period=decade").status_code, 400)

    def test_missing_event_is_404(self):
        self.assertEqual(self.client.get("/api/usage/events/nope").status_code, 404)

    def test_live_endpoint_is_empty_when_idle(self):
        body = self.client.get("/api/usage/live").json()
        self.assertEqual(body["streams_now"], 0)
        self.assertEqual(body["streams"], [])


if __name__ == "__main__":
    unittest.main()
