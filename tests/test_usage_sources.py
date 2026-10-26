"""Usage sources beyond live requests: tailnet names, engine metrics, the
engine sampler, live trends, and the history import (Hermes and run logs).

Every database write goes to a throwaway SQLite file. Fake Hermes and run-log
files are built in temp directories, so the real ones are never read.
"""

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import db
import usage
import usage_backfill


def sse(*events) -> bytes:
    out = b""
    for ev in events:
        out += b"data: " + json.dumps(ev).encode() + b"\n\n"
    return out


def delta(content=None, reasoning=None):
    d = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "delta": d}]}


def temp_db_case(testcase):
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    path = Path(tmp.name) / "spark_studio.db"
    with mock.patch.object(db, "DB_PATH", path):
        conn = db._connect()
    testcase.addCleanup(conn.close)
    patcher = mock.patch.object(db, "_conn", conn)
    patcher.start()
    testcase.addCleanup(patcher.stop)
    db.init()
    return conn


class TailscaleNameTests(unittest.TestCase):
    def test_maps_self_and_peers_by_tailnet_address(self):
        status = {
            "Self": {"HostName": "box-a", "TailscaleIPs": ["198.51.100.10", "2001:db8::1"]},
            "Peer": {
                "nodekey:a": {"HostName": "node-a", "TailscaleIPs": ["198.51.100.12"]},
                "nodekey:b": {"DNSName": "node-b.example.ts.net.", "TailscaleIPs": ["198.51.100.13"]},
                "nodekey:c": {"TailscaleIPs": ["198.51.100.14"]},
            },
        }
        names = usage.parse_tailscale_status(status)
        self.assertEqual(names["198.51.100.10"], "box-a")
        self.assertEqual(names["198.51.100.12"], "node-a")
        self.assertEqual(names["198.51.100.13"], "node-b")
        self.assertNotIn("198.51.100.14", names)  # no name, nothing to map

    def test_tolerates_missing_sections(self):
        self.assertEqual(usage.parse_tailscale_status({}), {})
        self.assertEqual(usage.parse_tailscale_status({"Peer": "garbage"}), {})

    def test_node_name_order(self):
        with mock.patch.dict(usage.NODE_NAMES, {"198.51.100.1": "pinned"}, clear=True), \
             mock.patch.dict(usage._ts_names, {"198.51.100.1": "ts-name", "198.51.100.2": "tailnet-box"}, clear=True):
            self.assertEqual(usage.node_name("198.51.100.1"), "pinned")  # env wins
            self.assertEqual(usage.node_name("198.51.100.2"), "tailnet-box")  # then tailnet
            self.assertEqual(usage.node_name("100.3.3.3"), "100.3.3.3")  # then the address
            self.assertEqual(usage.node_name("127.0.0.1"), "this box")


class EngineMetricsTests(unittest.TestCase):
    SAMPLE = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 3.0
vllm:num_requests_waiting{model_name="m"} 1.0
vllm:prompt_tokens_total{model_name="m"} 1000.0
vllm:generation_tokens_total{model_name="m"} 400.0
vllm:gpu_prefix_cache_queries_total{model_name="m"} 200.0
vllm:gpu_prefix_cache_hits_total{model_name="m"} 150.0
vllm:time_to_first_token_seconds_bucket{le="0.1"} 5.0
"""

    def test_picks_the_counters_and_sums_label_sets(self):
        counters = usage.parse_engine_metrics(self.SAMPLE)
        self.assertEqual(counters["running"], 3.0)
        self.assertEqual(counters["waiting"], 1.0)
        self.assertEqual(counters["gen"], 400.0)
        self.assertEqual(counters["cache_q"], 200.0)
        self.assertEqual(counters["cache_h"], 150.0)

    def test_other_engine_prefixes_are_read_too(self):
        # A TensorFold-served engine exposes the same counters under its own prefix.
        text = ("tensorfold:prompt_tokens_total 5988143\n"
                "tensorfold:generation_tokens_total 99189\n"
                "tensorfold:num_requests_running 2\n")
        counters = usage.parse_engine_metrics(text)
        self.assertEqual(counters["prompt"], 5988143.0)
        self.assertEqual(counters["gen"], 99189.0)
        self.assertEqual(counters["running"], 2.0)

    def test_accepts_the_non_gpu_cache_names_too(self):
        text = "vllm:prefix_cache_queries_total 10\nvllm:prefix_cache_hits_total 4\n"
        counters = usage.parse_engine_metrics(text)
        self.assertEqual((counters["cache_q"], counters["cache_h"]), (10.0, 4.0))

    def test_empty_or_unrelated_page_gives_nothing(self):
        self.assertEqual(usage.parse_engine_metrics("# nothing here\nprocess_cpu 1\n"), {})


class EngineSamplerTests(unittest.TestCase):
    def test_turns_counter_deltas_into_rates(self):
        s = usage.EngineSampler()
        first = {"running": 2, "waiting": 0, "gen": 100.0, "prompt": 500.0, "cache_q": 100.0, "cache_h": 50.0}
        second = {"running": 4, "waiting": 1, "gen": 300.0, "prompt": 900.0, "cache_q": 200.0, "cache_h": 150.0}
        self.assertIsNone(s.observe("run1", first, 1000.0))  # nothing to compare yet
        row = s.observe("run1", second, 1010.0)
        self.assertEqual(row["gen_tps"], 20.0)
        self.assertEqual(row["prompt_tps"], 40.0)
        self.assertEqual(row["cache_hit_pct"], 100.0)  # 100 hits out of 100 new queries
        self.assertEqual((row["running"], row["waiting"]), (4, 1))
        self.assertEqual(row["source"], "live")

    def test_engine_restart_counts_nothing_for_that_step(self):
        s = usage.EngineSampler()
        s.observe("run1", {"gen": 5000.0}, 1000.0)
        row = s.observe("run1", {"gen": 10.0}, 1010.0)  # counter went down
        self.assertEqual(row["gen_tps"], 0.0)

    def test_no_cache_queries_means_no_hit_rate(self):
        s = usage.EngineSampler()
        s.observe("run1", {"cache_q": 10.0, "cache_h": 5.0}, 1000.0)
        row = s.observe("run1", {"cache_q": 10.0, "cache_h": 5.0}, 1010.0)
        self.assertIsNone(row["cache_hit_pct"])

    def test_runs_are_tracked_separately(self):
        s = usage.EngineSampler()
        s.observe("a", {"gen": 0.0}, 1000.0)
        self.assertIsNone(s.observe("b", {"gen": 50.0}, 1005.0))


class EnginePersistTests(unittest.TestCase):
    def tick(self, gen, run="run1", ts=0.0):
        return {"ts": ts, "engine_run_id": run, "running": 1, "waiting": 0,
                "gen_tps": gen, "peak_gen_tps": gen, "prompt_tps": 2.0, "cache_hit_pct": 50.0}

    def test_stores_one_row_per_ten_ticks_with_mean_and_busiest_second(self):
        persist = usage.EnginePersist(window=10)
        rows = [persist.add(self.tick(float(i))) for i in range(10)]
        self.assertEqual(rows[:9], [None] * 9)
        stored = rows[9]
        self.assertIsNone(stored["gen_tps"])  # the counter's rate is not reported
        self.assertEqual(stored["cache_hit_pct"], 50.0)
        self.assertEqual(stored["cache_hit_pct"], 50.0)

    def test_a_different_engine_run_starts_a_new_window(self):
        persist = usage.EnginePersist(window=3)
        persist.add(self.tick(1.0, run="a"))
        persist.add(self.tick(1.0, run="a"))
        self.assertIsNone(persist.add(self.tick(5.0, run="b")))

class LiveSeriesTests(unittest.TestCase):
    def test_recent_series_buckets_per_second_oldest_first(self):
        now = 1000.0
        # token_times is appended in arrival order, so oldest first.
        times = [now - 45.0, now - 39.5, now - 2.1, now - 0.2, now - 0.5]
        counts = usage._recent_series(times, now, seconds=40)
        self.assertEqual(len(counts), 40)
        self.assertEqual(counts[-1], 2)  # the two in the last second
        self.assertEqual(counts[-3], 1)  # one about two seconds ago
        self.assertEqual(counts[0], 1)  # one about 39 seconds ago
        self.assertEqual(sum(counts), 4)  # the 45-second-old token is outside the window


class SourceDbTests(unittest.TestCase):
    def setUp(self):
        self.conn = temp_db_case(self)

    def test_engine_samples_are_idempotent_by_id(self):
        row = {"id": "x1", "ts": 100.0, "engine_run_id": "r", "source": "log", "running": 1,
               "waiting": 0, "gen_tps": 5.0, "prompt_tps": 1.0, "cache_hit_pct": None}
        self.assertEqual(db.engine_sample_insert_many([row]), 1)
        self.assertEqual(db.engine_sample_insert_many([row]), 0)
        self.assertEqual(len(db.engine_samples_since(0)), 1)

    def test_prune_removes_only_old_failed_rows(self):
        base = {"node": "n", "client_ip": "", "engine_run_id": "r", "model": "m", "endpoint": "/v1/chat/completions",
                "stream": 1, "error": None, "prompt_tokens": None, "completion_tokens": 1, "cached_tokens": None,
                "cache_hit_pct": None, "thinking": 0, "concurrency": 1, "ttft_ms": None, "decode_secs": None,
                "tok_s": None, "peak_tok_s": None, "samples": None, "ended_at": 0}
        now = time.time()
        db.usage_insert({**base, "id": "old-aborted", "started_at": now - 40 * 86400, "status": "aborted"})
        db.usage_insert({**base, "id": "old-ok", "started_at": now - 40 * 86400, "status": "ok"})
        db.usage_insert({**base, "id": "new-aborted", "started_at": now - 86400, "status": "aborted"})
        self.assertEqual(db.usage_prune_failed(now - 30 * 86400), 1)
        remaining = {r["id"] for r in db.usage_list(since=0, limit=10)}
        self.assertEqual(remaining, {"old-ok", "new-aborted"})

    def test_live_start_is_none_without_live_rows(self):
        self.assertIsNone(db.usage_first_live_start())


class HermesImportTests(unittest.TestCase):
    def setUp(self):
        self.conn = temp_db_case(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name) / "state.db"
        src = sqlite3.connect(self.state)
        src.execute(
            "CREATE TABLE session_model_usage (session_id TEXT, model TEXT, billing_base_url TEXT, "
            "task TEXT, api_call_count INT, input_tokens INT, output_tokens INT, cache_read_tokens INT, "
            "reasoning_tokens INT, first_seen REAL, last_seen REAL)"
        )
        src.execute(
            "INSERT INTO session_model_usage VALUES ('s1','qwen','http://127.0.0.1:8888/v1','',2,"
            "1000,300,4000,50,1700000000.0,1700000100.0)"
        )
        src.execute(
            "INSERT INTO session_model_usage VALUES ('s2','qwen','http://127.0.0.1:8888/v1','',0,0,0,0,0,"
            "1700000200.0,1700000200.0)"
        )
        src.commit()
        src.close()

    def test_imports_summary_rows_with_prompt_and_cache(self):
        self.assertEqual(usage_backfill.import_hermes(self.state, node="this box"), 1)
        row = db.usage_get("hermes-" + usage_backfill.hashlib.sha1(
            "s1|qwen|http://127.0.0.1:8888/v1|".encode()).hexdigest()[:16])
        self.assertEqual(row["prompt_tokens"], 5000)  # uncached input + cache reads
        self.assertEqual(row["cached_tokens"], 4000)
        self.assertEqual(row["cache_hit_pct"], 80.0)
        self.assertEqual(row["completion_tokens"], 300)
        self.assertEqual(row["thinking"], 1)
        self.assertEqual(row["node"], "this box")
        self.assertIsNone(row["tok_s"])  # no timing in Hermes' records

    def test_rerunning_does_not_duplicate(self):
        usage_backfill.import_hermes(self.state, node="this box")
        usage_backfill.import_hermes(self.state, node="this box")
        self.assertEqual(db.usage_summary(0)["totals"]["requests"], 1)

    def test_skips_history_at_or_after_the_first_live_request(self):
        db.usage_insert({
            "id": "live1", "started_at": 1700000050.0, "ended_at": 1700000060.0, "node": "node-a",
            "client_ip": "198.51.100.12", "engine_run_id": "run1", "model": "qwen", "endpoint": "/v1/chat/completions",
            "stream": 1, "status": "ok", "error": None, "prompt_tokens": 1, "completion_tokens": 1,
            "cached_tokens": None, "cache_hit_pct": None, "thinking": 0, "concurrency": 1,
            "ttft_ms": None, "decode_secs": None, "tok_s": None, "peak_tok_s": None, "samples": None,
        })
        # The Hermes session last ran after live capture began, so it's left to the live recorder.
        self.assertEqual(usage_backfill.import_hermes(self.state, node="this box"), 0)

    def test_missing_or_unreadable_source_is_harmless(self):
        self.assertEqual(usage_backfill.import_hermes(Path("/nonexistent/state.db")), 0)
        junk = Path(self.state.parent) / "junk.db"
        junk.write_bytes(b"not a database")
        self.assertEqual(usage_backfill.import_hermes(junk), 0)


class RunLogImportTests(unittest.TestCase):
    def setUp(self):
        self.conn = temp_db_case(self)

    LINE = ("(APIServer pid=110) INFO {stamp} [loggers.py:314] Engine 000: Avg prompt throughput: "
            "9.5 tokens/s, Avg generation throughput: 3.6 tokens/s, Running: 2 reqs, Waiting: 1 reqs, "
            "GPU KV cache usage: 0.0%, Prefix cache hit rate: 42.5%\n")

    def test_parses_timestamps_and_fields(self):
        mtime = time.mktime((2026, 9, 3, 12, 0, 0, 0, 0, -1))
        lines = [self.LINE.format(stamp="09-02 21:19:12"), "unrelated line\n"]
        rows = usage_backfill.parse_throughput_lines(lines, mtime, "abc")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["running"], 2)
        self.assertEqual(row["waiting"], 1)
        self.assertEqual(row["gen_tps"], 3.6)
        self.assertEqual(row["prompt_tps"], 9.5)
        self.assertEqual(row["cache_hit_pct"], 42.5)
        self.assertEqual(time.localtime(row["ts"]).tm_mon, 9)
        self.assertEqual(time.localtime(row["ts"]).tm_hour, 21)

    def test_year_steps_back_when_the_stamp_is_after_the_file(self):
        # A December stamp in a file written in January belongs to the previous year.
        mtime = time.mktime((2027, 1, 5, 9, 0, 0, 0, 0, -1))
        rows = usage_backfill.parse_throughput_lines([self.LINE.format(stamp="12-30 23:59:50")], mtime, "y")
        self.assertEqual(time.localtime(rows[0]["ts"]).tm_year, 2026)

    def test_missing_cache_field_is_none(self):
        line = ("INFO 09-02 21:19:12 [loggers.py:314] Engine 000: Avg prompt throughput: 0.0 tokens/s, "
                "Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs\n")
        rows = usage_backfill.parse_throughput_lines([line], time.time(), "n")
        self.assertIsNone(rows[0]["cache_hit_pct"])

    def test_imports_a_log_directory_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run1.log"
            log.write_text(self.LINE.format(stamp="09-02 21:19:02") + self.LINE.format(stamp="09-02 21:19:12"))
            self.assertEqual(usage_backfill.import_run_logs(Path(tmp)), 2)
            self.assertEqual(usage_backfill.import_run_logs(Path(tmp)), 0)
            self.assertEqual(len(db.engine_samples_since(0)), 2)

    def test_absent_directory_imports_nothing(self):
        self.assertEqual(usage_backfill.import_run_logs(Path("/nonexistent/logs")), 0)

    def test_run_all_survives_missing_sources(self):
        with mock.patch.object(usage_backfill, "hermes_state_path", return_value=Path("/nope/state.db")), \
             mock.patch.object(usage_backfill, "import_run_logs", side_effect=RuntimeError("boom")):
            counts = usage_backfill.run_all()
        self.assertEqual(counts, {"hermes_sessions": 0, "engine_samples": 0, "failed_pruned": 0})


class LiveSnapshotFieldsTests(unittest.TestCase):
    def test_snapshot_includes_engine_block_and_session_peak(self):
        with mock.patch.dict(usage._live, {}, clear=True), \
             mock.patch.object(usage, "_session_peak_gen", 77.0), \
             mock.patch.object(usage, "_engine_history", usage.deque(maxlen=120)):
            usage.note_engine_sample({"ts": time.time(), "running": 1, "waiting": 0,
                                      "gen_tps": 12.5, "prompt_tps": 3.0, "cache_hit_pct": 90.0})
            snap = usage.live_snapshot()
        self.assertEqual(snap["session_peak_tok_s"], 77.0)
        self.assertEqual(snap["engine"]["gen_tps"], 12.5)
        self.assertEqual(len(snap["engine_history"]), 1)


class ToolCallAndLiveTests(unittest.TestCase):
    def setUp(self):
        self.conn = temp_db_case(self)
        patcher = mock.patch.object(usage.db, "usage_insert")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: usage._live.clear())

    def make(self, stream=True):
        return usage.UsageRecorder(client_ip=None, engine_run_id="r", model="qwen",
                                   endpoint="/v1/chat/completions", stream=stream,
                                   engine_url="http://127.0.0.1:1")

    def test_streamed_tool_call_is_named_and_counted(self):
        rec = self.make()
        rec.feed(sse({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"name": "terminal", "arguments": ""}}]}}]}))
        self.assertEqual(rec.phase, "tool call")
        self.assertEqual(rec.tools_in_progress(), ["terminal"])
        rec.feed(sse({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"command": "ls"}'}}]}}]}))
        rec.feed(b"data: [DONE]\n\n")
        self.assertEqual(rec.tools_in_progress(), ["terminal"])  # still in progress until finished
        self.assertEqual(rec.tool_calls_done(), [{"name": "terminal", "args_chars": 17}])

    def test_non_streamed_tool_calls_are_read_from_the_message(self):
        rec = self.make(stream=False)
        body = json.dumps({"choices": [{"message": {"tool_calls": [
            {"function": {"name": "read_file", "arguments": "{}"}},
            {"function": {"name": "web_search", "arguments": '{"q": "x"}'}}]}}]}).encode()
        rec.feed(body)
        record = rec.finish("ok")
        self.assertEqual([c["name"] for c in rec.tool_calls_done()], ["read_file", "web_search"])
        self.assertEqual(record["status"], "ok")

    def test_tool_summary_counts_calls_and_requests(self):
        db.usage_tool_calls_insert("e1", time.time(), "n", "m", [{"name": "terminal", "args_chars": 10}])
        db.usage_tool_calls_insert("e2", time.time(), "n", "m", [{"name": "terminal", "args_chars": 4},
                                                                  {"name": "read_file", "args_chars": 2}])
        out = db.usage_tool_summary(0)
        self.assertEqual(out["by_name"][0], {"name": "terminal", "calls": 2, "requests": 2, "avg_args_chars": 7.0})
        self.assertEqual(len(out["recent"]), 3)

    def test_live_token_count_comes_from_the_engine_tokenizer(self):
        import asyncio
        rec = self.make()
        rec.feed(sse(delta(content="hello there")))
        rec.first_token_at = time.time() - 4.0

        class FakeClient:
            def __init__(self):
                self.calls = []

            async def post(self, url, json=None, timeout=None):
                self.calls.append((url, json["prompt"]))
                return SimpleNamespace(status_code=200, json=lambda: {"count": 80})

        client = FakeClient()
        asyncio.run(usage.refresh_live_tokens(client))
        self.assertEqual(client.calls[0][0], "http://127.0.0.1:1/tokenize")
        self.assertEqual(rec.live_tokens, 80)
        snap = usage.live_snapshot()
        self.assertEqual(snap["streams"][0]["tokens_so_far"], 80)
        self.assertTrue(snap["streams"][0]["tokens_exact"])
        self.assertAlmostEqual(snap["streams"][0]["tok_s"], 20.0, delta=0.5)  # 80 tokens over ~4 s

    def test_no_live_rate_before_the_first_second(self):
        rec = self.make()
        rec.feed(sse(delta(content="hi")))
        rec.live_tokens = 3
        rec.first_token_at = time.time()
        self.assertIsNone(usage._live_rate(rec, time.time()))


if __name__ == "__main__":
    unittest.main()
