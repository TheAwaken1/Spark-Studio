"""Tests for external endpoint persistence and boot-time adoption.

An engine started outside the app (a `docker run` vLLM container, a systemd
unit) used to vanish from the dashboard on every app restart: external runs
have no pid, so `_reconcile_on_boot()` marked them exited. They now persist
their URL and are re-adopted while `/v1/models` answers, and the opt-in
`SPARK_STUDIO_ADOPT_PORTS` discovers one when nothing else is running.

Every test runs against a temp SQLite file and a throwaway in-process
OpenAI-style stub on 127.0.0.1 — the real `data/spark_studio.db`, sparkrun,
Docker and the network are never touched.
"""
import asyncio
import json
import os
import socket
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import db
import runners
import server


class _StubEngine:
    """Minimal OpenAI-compatible server: GET /v1/models only."""

    def __init__(self, models=("qwen3.8-flash-next",), owned_by="vllm", body=None):
        payload = body if body is not None else json.dumps({
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": owned_by} for m in models],
        })
        self.hits = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                stub.hits += 1
                if self.path != "/v1/models":
                    self.send_response(404)
                    self.end_headers()
                    return
                data = payload.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self._thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _free_port() -> int:
    """A local port with nothing listening on it (connection refused)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _AdoptionTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        conn = sqlite3.connect(str(Path(tmp.name) / "test.db"), check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        for patcher in (
            mock.patch.object(db, "_conn", conn),
            mock.patch.dict(server.runner.runs, clear=True),
            # No sparkrun on the test box: reconciliation must not shell out.
            mock.patch.object(server, "engine_available", return_value=False),
            mock.patch.dict(os.environ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("SPARK_STUDIO_ADOPT_PORTS", None)
        db.init()
        db._add_col_if_missing("runs", "meta_json", "TEXT")

    def stub(self, **kwargs) -> _StubEngine:
        engine = _StubEngine(**kwargs)
        self.addCleanup(engine.stop)
        return engine

    def restart(self):
        """Forget the in-memory runs, as a process restart would."""
        server.runner.runs.clear()

    def running_rows(self):
        return db.runs_list_running()


class PersistenceTests(_AdoptionTestCase):
    def test_register_external_persists_url_and_label(self):
        run = server.runner.register_external("vllm", "my-model", "http://0.0.0.0:8000/")

        row = db.runs_get(run.id)
        meta = json.loads(row["meta_json"])
        self.assertTrue(meta["_external"])
        self.assertEqual(meta["_url"], "http://127.0.0.1:8000")
        self.assertEqual(meta["_label"], "my-model")

    def test_shutdown_leaves_external_rows_running(self):
        run = server.runner.register_external("vllm", "my-model", "http://127.0.0.1:8000")

        server.runner.shutdown_all()

        self.assertEqual(db.runs_get(run.id)["status"], "running")

    def test_shutdown_still_stops_managed_runs(self):
        managed = runners.Run(id="managed1", engine="vllm", recipe_id=None, cmd=["x"], env={}, port=8001)
        managed.status = "running"
        server.runner.runs[managed.id] = managed
        db.runs_insert({"id": managed.id, "engine": "vllm", "status": "running", "cmd": "x"})

        server.runner.shutdown_all()

        self.assertEqual(db.runs_get(managed.id)["status"], "exited")


class ReconcileTests(_AdoptionTestCase):
    def test_answering_external_endpoint_is_readopted_after_restart(self):
        engine = self.stub()
        run = server.runner.register_external("vllm", "my-model", engine.url)
        self.restart()

        asyncio.run(server._reconcile_on_boot())

        adopted = server.runner.runs.get(run.id)
        self.assertIsNotNone(adopted, "the external run should keep its id across the restart")
        self.assertEqual(adopted.status, "running")
        self.assertTrue(adopted.ready)
        self.assertEqual(adopted.url, engine.url)
        self.assertEqual(adopted.label, "my-model")
        self.assertTrue(adopted.meta.get("_external"))
        self.assertEqual(db.runs_get(run.id)["status"], "running")
        # Still external: the watchdog keeps retiring it after 2 failed probes.
        self.assertEqual(json.loads(db.runs_get(run.id)["meta_json"])["_url"], engine.url)

    def test_dead_external_endpoint_is_marked_exited(self):
        run = server.runner.register_external("vllm", "gone", f"http://127.0.0.1:{_free_port()}")
        self.restart()

        asyncio.run(server._reconcile_on_boot())

        self.assertNotIn(run.id, server.runner.runs)
        self.assertEqual(db.runs_get(run.id)["status"], "exited")

    def test_non_openai_answer_counts_as_dead(self):
        for body in ("<html>not an engine</html>", json.dumps({"status": "ok"})):
            with self.subTest(body=body):
                engine = self.stub(body=body)
                run = server.runner.register_external("vllm", "web", engine.url)
                self.restart()

                asyncio.run(server._reconcile_on_boot())

                self.assertEqual(db.runs_get(run.id)["status"], "exited")

    def test_legacy_external_row_without_url_is_still_marked_exited(self):
        # Rows written before this change carry no meta_json at all.
        db.runs_insert({
            "id": "legacy1", "engine": "vllm", "status": "running",
            "pid": None, "port": None, "cmd": "[external] old http://127.0.0.1:8000",
        })

        asyncio.run(server._reconcile_on_boot())

        self.assertEqual(db.runs_get("legacy1")["status"], "exited")


class AdoptPortsConfigTests(unittest.TestCase):
    def test_unset_means_off(self):
        with mock.patch.dict(os.environ, clear=True):
            self.assertEqual(server._adopt_ports_from_env(), [])

    def test_parses_list_and_drops_invalid_entries(self):
        with mock.patch.dict(os.environ, {"SPARK_STUDIO_ADOPT_PORTS": "8000, 8001 x 8000 70000 0"}):
            self.assertEqual(server._adopt_ports_from_env(), [8000, 8001])


class DetectionTests(_AdoptionTestCase):
    def test_detection_is_off_by_default(self):
        engine = self.stub()

        with mock.patch.object(server, "_probe_served_models", new=mock.AsyncMock()) as probe:
            asyncio.run(server._boot_adoption())

        probe.assert_not_awaited()
        self.assertEqual(engine.hits, 0)
        self.assertEqual(server.runner.runs, {})
        self.assertEqual(self.running_rows(), [])

    def test_detects_engine_and_labels_it_with_the_served_model(self):
        engine = self.stub(models=("qwen3.8-flash-next",), owned_by="vllm")
        os.environ["SPARK_STUDIO_ADOPT_PORTS"] = str(engine.port)

        asyncio.run(server._boot_adoption())

        runs = list(server.runner.runs.values())
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].label, "qwen3.8-flash-next")
        self.assertEqual(runs[0].engine, "vllm")
        self.assertEqual(runs[0].url, engine.url)
        self.assertTrue(runs[0].meta.get("_external"))
        self.assertEqual([r["id"] for r in self.running_rows()], [runs[0].id])

    def test_detected_engine_survives_restart_without_duplicate(self):
        engine = self.stub()
        os.environ["SPARK_STUDIO_ADOPT_PORTS"] = str(engine.port)
        asyncio.run(server._boot_adoption())
        first_id = next(iter(server.runner.runs))
        self.restart()

        asyncio.run(server._boot_adoption())

        self.assertEqual(list(server.runner.runs), [first_id])
        self.assertEqual([r["id"] for r in self.running_rows()], [first_id])

    def test_no_duplicate_when_a_sparkrun_run_holds_the_port(self):
        engine = self.stub()
        os.environ["SPARK_STUDIO_ADOPT_PORTS"] = str(engine.port)
        job = runners.Run(id="spark1", engine="sparkrun", recipe_id=None, cmd=["sparkrun run x"], env={}, port=engine.port)
        job.status = "running"
        job.url = engine.url
        server.runner.runs[job.id] = job

        adopted = asyncio.run(server._adopt_listening_ports([engine.port]))

        self.assertEqual(adopted, [])
        self.assertEqual(engine.hits, 0, "nothing to detect while a run is active: no probe at all")
        self.assertEqual(list(server.runner.runs), ["spark1"])
        self.assertEqual(self.running_rows(), [])

    def test_unknown_owner_falls_back_to_vllm_engine(self):
        engine = self.stub(models=("m1",), owned_by="someone-else")

        adopted = asyncio.run(server._adopt_listening_ports([engine.port]))

        self.assertEqual([r.engine for r in adopted], ["vllm"])

    def test_sglang_owner_is_kept(self):
        engine = self.stub(models=("m1",), owned_by="sglang")

        adopted = asyncio.run(server._adopt_listening_ports([engine.port]))

        self.assertEqual([r.engine for r in adopted], ["sglang"])

    def test_nothing_listening_registers_nothing(self):
        adopted = asyncio.run(server._adopt_listening_ports([_free_port()]))

        self.assertEqual(adopted, [])
        self.assertEqual(self.running_rows(), [])

    def test_keeps_looking_while_the_engine_is_still_loading(self):
        os.environ["SPARK_STUDIO_ADOPT_PORTS"] = "8000"
        probe = mock.AsyncMock(side_effect=[None, None, [{"id": "late-model", "owned_by": "vllm"}]])
        with mock.patch.object(server, "_probe_served_models", new=probe), \
                mock.patch.object(server, "_WATCHDOG_INTERVAL", 0), \
                mock.patch.object(server, "_ADOPT_WINDOW", 60):
            asyncio.run(server._adopt_ports_on_boot())

        self.assertEqual(probe.await_count, 3)
        self.assertEqual([r.label for r in server.runner.runs.values()], ["late-model"])

    def test_gives_up_after_the_window(self):
        os.environ["SPARK_STUDIO_ADOPT_PORTS"] = "8000"
        probe = mock.AsyncMock(return_value=None)
        with mock.patch.object(server, "_probe_served_models", new=probe), \
                mock.patch.object(server, "_WATCHDOG_INTERVAL", 0), \
                mock.patch.object(server, "_ADOPT_WINDOW", 0):
            asyncio.run(server._adopt_ports_on_boot())

        self.assertEqual(probe.await_count, 1)
        self.assertEqual(server.runner.runs, {})

    def test_one_pass_adopts_every_listed_port(self):
        first = self.stub(models=("model-a",))
        second = self.stub(models=("model-b",))

        adopted = asyncio.run(server._adopt_listening_ports([first.port, second.port]))

        self.assertEqual([r.label for r in adopted], ["model-a", "model-b"])
        self.assertEqual(len(self.running_rows()), 2)


class ConcurrencyTests(_AdoptionTestCase):
    """The detection pass awaits an HTTP probe; whatever registers an engine
    meanwhile (a managed run, POST /api/external) must win, never be doubled."""

    def _probe_then(self, action):
        """A _probe_served_models that runs ``action`` once its real request
        has completed, i.e. while the caller is still suspended in the await."""
        real_probe = server._probe_served_models

        async def probe(url, timeout=5):
            served = await real_probe(url, timeout=timeout)
            action(url)
            return served

        return mock.patch.object(server, "_probe_served_models", new=probe)

    def test_managed_run_started_during_the_probe_is_not_duplicated(self):
        engine = self.stub(models=("managed-model",))

        def start_managed(_url):
            server.runner.adopt(
                "managed1", engine="sparkrun", jobid="job1",
                url=engine.url, port=engine.port, label="managed-model",
            )
            db.runs_insert({"id": "managed1", "engine": "sparkrun", "status": "running",
                            "port": engine.port, "cmd": "sparkrun run managed-model"})

        with self._probe_then(start_managed):
            adopted = asyncio.run(server._adopt_listening_ports([engine.port]))

        self.assertEqual(adopted, [])
        self.assertEqual(list(server.runner.runs), ["managed1"])
        self.assertEqual([r["id"] for r in self.running_rows()], ["managed1"])
        self.assertEqual(server.runner.active().id, "managed1")

    def test_manual_registration_during_the_probe_is_not_duplicated(self):
        engine = self.stub()
        manual = []

        with self._probe_then(lambda url: manual.append(
                server.runner.register_external("vllm", "manual", url))):
            adopted = asyncio.run(server._adopt_listening_ports([engine.port]))

        self.assertEqual(adopted, [])
        self.assertEqual([r["id"] for r in self.running_rows()], [manual[0].id])

    def test_pass_stops_when_a_run_appears_between_two_ports(self):
        first = self.stub(models=("model-a",))
        second = self.stub(models=("model-b",))

        def start_on_second(url):
            if url.endswith(f":{second.port}"):
                server.runner.adopt("managed1", engine="sparkrun", jobid="job1",
                                    url=second.url, port=second.port)

        with self._probe_then(start_on_second):
            adopted = asyncio.run(server._adopt_listening_ports([first.port, second.port]))

        self.assertEqual([r.label for r in adopted], ["model-a"])
        self.assertEqual(sorted(server.runner.runs), sorted([adopted[0].id, "managed1"]))

    def test_any_run_started_during_the_probe_stops_the_pass(self):
        # Detection only runs while nothing is active, whatever the port.
        engine = self.stub()

        with self._probe_then(lambda _url: server.runner.adopt(
                "managed1", engine="vllm", url="http://127.0.0.1:1", port=1)):
            adopted = asyncio.run(server._adopt_listening_ports([engine.port]))

        self.assertEqual(adopted, [])
        self.assertEqual(list(server.runner.runs), ["managed1"])

    def test_a_port_listed_twice_is_adopted_once(self):
        engine = self.stub()

        adopted = asyncio.run(server._adopt_listening_ports([engine.port, engine.port]))

        self.assertEqual(len(adopted), 1)
        self.assertEqual(len(self.running_rows()), 1)

    def test_local_port_held_by_a_run_with_a_lan_url_is_taken(self):
        # sparkrun can report this box by its LAN address, not 127.0.0.1.
        engine = self.stub()
        server.runner.adopt("spark1", engine="sparkrun", jobid="job1",
                            url=f"http://192.0.2.10:{engine.port}", port=engine.port)

        summary = asyncio.run(server.register_external(
            server.ExternalReq(engine="vllm", name="again", url=engine.url)))

        self.assertEqual(summary["id"], "spark1")
        self.assertEqual(self.running_rows(), [])

    def test_manual_registration_of_an_adopted_endpoint_returns_the_same_run(self):
        engine = self.stub()
        adopted = asyncio.run(server._adopt_listening_ports([engine.port]))

        summary = asyncio.run(server.register_external(
            server.ExternalReq(engine="vllm", name="again", url=engine.url.replace("127.0.0.1", "localhost"))))

        self.assertEqual(summary["id"], adopted[0].id)
        self.assertEqual([r["id"] for r in self.running_rows()], [adopted[0].id])

    def test_manual_registration_of_a_new_endpoint_still_works(self):
        engine = self.stub()

        summary = asyncio.run(server.register_external(
            server.ExternalReq(engine="vllm", name="mine", url=engine.url)))

        self.assertEqual([r["id"] for r in self.running_rows()], [summary["id"]])

    def test_a_stopped_run_does_not_hold_its_endpoint(self):
        engine = self.stub()
        old = server.runner.register_external("vllm", "old", engine.url)
        old.status = "exited"

        summary = asyncio.run(server.register_external(
            server.ExternalReq(engine="vllm", name="new", url=engine.url)))

        self.assertNotEqual(summary["id"], old.id)

    def test_reconcile_prefers_the_surviving_managed_run_on_the_same_port(self):
        engine = self.stub()
        # The managed row is older, so the newer external row comes first in
        # runs_list_running(): it must still not be adopted next to it.
        db.runs_insert({"id": "managed1", "engine": "vllm", "status": "running", "pid": 4242,
                        "port": engine.port, "cmd": "vllm serve m", "started_at": 1.0})
        external = server.runner.register_external("vllm", "dup", engine.url)
        self.restart()

        with mock.patch.object(server, "_cmd_fragment_alive", return_value=True):
            asyncio.run(server._reconcile_on_boot())

        self.assertEqual(list(server.runner.runs), ["managed1"])
        self.assertEqual(db.runs_get(external.id)["status"], "exited")

    def test_reconcile_readopts_one_of_two_rows_for_the_same_endpoint(self):
        engine = self.stub()
        server.runner.register_external("vllm", "one", engine.url)
        server.runner.register_external("vllm", "two", engine.url)
        self.restart()

        asyncio.run(server._reconcile_on_boot())

        self.assertEqual(len(server.runner.runs), 1)
        self.assertEqual(len(self.running_rows()), 1)


class EndpointKeyTests(unittest.TestCase):
    def test_loopback_spellings_are_the_same_endpoint(self):
        keys = {server._endpoint_key(u) for u in (
            "http://127.0.0.1:8000", "http://localhost:8000/", "http://0.0.0.0:8000/v1", "127.0.0.1:8000")}
        self.assertEqual(keys, {("local", 8000)})

    def test_other_host_or_port_is_another_endpoint(self):
        self.assertNotEqual(server._endpoint_key("http://10.0.0.5:8000"), ("local", 8000))
        self.assertNotEqual(server._endpoint_key("http://127.0.0.1:8001"), ("local", 8000))
        self.assertIsNone(server._endpoint_key("http://127.0.0.1:notaport"))


if __name__ == "__main__":
    unittest.main()
