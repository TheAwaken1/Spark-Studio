"""Usage tracking: parsing engine streams, recording requests, and summaries.

The recorder is fed the same bytes an engine would send. Every database write
goes to a throwaway SQLite file, so data/spark_studio.db is never touched.
"""

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import db
import usage


def sse(*events) -> bytes:
    """Server-sent events body from dicts, with a closing [DONE]."""
    out = b""
    for ev in events:
        out += b"data: " + json.dumps(ev).encode() + b"\n\n"
    return out + b"data: [DONE]\n\n"


def delta(content=None, reasoning=None):
    d = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "delta": d}]}


class NodeNameTests(unittest.TestCase):
    def test_parses_pairs_and_skips_bad_entries(self):
        self.assertEqual(
            usage.parse_node_names("198.51.100.1=node-a, 198.51.100.2 = node-b,bad,=x,y="),
            {"198.51.100.1": "node-a", "198.51.100.2": "node-b"},
        )

    def test_names_and_loopback(self):
        with mock.patch.dict(usage.NODE_NAMES, {"198.51.100.1": "node-a"}, clear=True):
            self.assertEqual(usage.node_name("198.51.100.1"), "node-a")
            self.assertEqual(usage.node_name("127.0.0.1"), "this box")
            self.assertEqual(usage.node_name("100.9.9.9"), "100.9.9.9")
            self.assertEqual(usage.node_name(None), "unknown")


class PrepareBodyTests(unittest.TestCase):
    def test_adds_include_usage_to_streamed_chat(self):
        body = json.dumps({"model": "m", "stream": True}).encode()
        out = json.loads(usage.prepare_stream_body(body, "chat/completions"))
        self.assertEqual(out["stream_options"], {"include_usage": True})

    def test_leaves_callers_stream_options_alone(self):
        body = json.dumps({"stream": True, "stream_options": {"include_usage": False}}).encode()
        self.assertEqual(usage.prepare_stream_body(body, "chat/completions"), body)

    def test_leaves_non_stream_other_paths_and_junk_alone(self):
        plain = json.dumps({"model": "m"}).encode()
        self.assertEqual(usage.prepare_stream_body(plain, "chat/completions"), plain)
        streamed = json.dumps({"stream": True}).encode()
        self.assertEqual(usage.prepare_stream_body(streamed, "completions"), streamed)
        self.assertEqual(usage.prepare_stream_body(b"not json", "chat/completions"), b"not json")

    def test_request_meta(self):
        self.assertEqual(usage.request_meta(json.dumps({"model": "m", "stream": True}).encode()), ("m", True))
        self.assertEqual(usage.request_meta(b"junk"), ("", False))


class RecorderParsingTests(unittest.TestCase):
    def setUp(self):
        # Keep recorders out of the real database and the module live view.
        patcher = mock.patch.object(usage.db, "usage_insert")
        self.insert = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: usage._live.clear())

    def make(self, stream=True, client_ip="198.51.100.1"):
        with mock.patch.dict(usage.NODE_NAMES, {"198.51.100.1": "node-a"}, clear=True):
            return usage.UsageRecorder(
                client_ip=client_ip, engine_run_id="run1", model="qwen",
                endpoint="/v1/chat/completions", stream=stream,
            )

    def test_stream_split_across_chunks_counts_tokens_and_usage(self):
        rec = self.make()
        body = sse(delta(content="Hel"), delta(content="lo"), delta(content="!"),
                   {"choices": [], "usage": {
                       "prompt_tokens": 100, "completion_tokens": 3,
                       "prompt_tokens_details": {"cached_tokens": 64}}})
        # Cut the body at awkward places, including mid-line.
        for i in range(0, len(body), 7):
            rec.feed(body[i:i + 7])
        record = rec.finish("ok")
        self.assertEqual(record["completion_tokens"], 3)
        self.assertEqual(record["prompt_tokens"], 100)
        self.assertEqual(record["cached_tokens"], 64)
        self.assertEqual(record["cache_hit_pct"], 64.0)
        self.assertEqual(record["node"], "node-a")
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["thinking"], 0)

    def test_reasoning_marks_thinking_then_answering(self):
        rec = self.make()
        rec.feed(sse(delta(reasoning="hmm"), delta(reasoning=" more")))
        self.assertEqual(rec.phase, "thinking")
        self.assertTrue(rec.thinking)
        rec.feed(sse(delta(content="answer")))
        self.assertEqual(rec.phase, "answering")
        record = rec.finish("ok")
        self.assertEqual(record["thinking"], 1)

    def test_think_tags_in_content_count_as_thinking(self):
        rec = self.make()
        rec.feed(sse(delta(content="<think>plan")))
        self.assertEqual(rec.phase, "thinking")
        rec.feed(sse(delta(content="</think>")))
        self.assertEqual(rec.phase, "answering")
        self.assertTrue(rec.thinking)

    def test_missing_usage_falls_back_to_chunk_count(self):
        rec = self.make()
        rec.feed(sse(delta(content="a"), delta(content="b")))
        record = rec.finish("ok")
        self.assertEqual(record["completion_tokens"], 2)
        self.assertIsNone(record["prompt_tokens"])
        self.assertIsNone(record["cache_hit_pct"])

    def test_done_marker_is_recorded(self):
        rec = self.make()
        self.assertFalse(rec.saw_done)
        rec.feed(sse(delta(content="hi")))
        self.assertTrue(rec.saw_done)

    def test_non_stream_reads_the_whole_body(self):
        rec = self.make(stream=False)
        body = json.dumps({
            "choices": [{"message": {"content": "ok", "reasoning_content": "thought"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        }).encode()
        rec.feed(body[:20])
        rec.feed(body[20:])
        record = rec.finish("ok")
        self.assertEqual(record["completion_tokens"], 5)
        self.assertEqual(record["cache_hit_pct"], 0.0)
        self.assertEqual(record["thinking"], 1)
        self.assertEqual(record["stream"], 0)

    def test_error_and_abort_are_recorded_without_tokens(self):
        rec = self.make()
        record = rec.finish("error", "HTTP 500")
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"], "HTTP 500")
        self.assertIsNone(record["tok_s"])
        # A second finish does nothing.
        self.assertIsNone(rec.finish("ok"))

    def test_samples_bucket_tokens_per_second(self):
        rec = self.make()
        base = rec.started_at
        rec.token_times = [base + 0.1, base + 0.2, base + 1.3]
        rec.first_token_at = base + 0.1
        rec.last_token_at = base + 1.3
        record = rec.finish("ok")
        self.assertEqual(record["samples"], [2, 1])
        self.assertIn("peak_tok_s", record)

    def test_fastest_request_is_its_exact_rate(self):
        rec = self.make()
        rec.status = "ok"
        rec.first_token_at = rec.started_at + 1.0
        rec.completion_tokens = 60
        rec.ended_at = rec.started_at + 4.0
        record = rec._record()
        self.assertEqual(record["tok_s"], 20.0)  # 60 tokens over the 3 s after the first token
        self.assertEqual(record["peak_tok_s"], 20.0)
        self.assertEqual(usage._session_peak_gen, 0.0)
        usage._note_finished(record)
        self.assertEqual(usage._session_peak_gen, 20.0)

    def test_registers_and_unregisters_live(self):
        rec = self.make()
        self.assertIn(rec.id, usage._live)
        snap = usage.live_snapshot()
        self.assertEqual(snap["streams_now"], 1)
        self.assertEqual(snap["streams"][0]["node"], "node-a")
        rec.finish("ok")
        self.assertNotIn(rec.id, usage._live)

    def test_concurrency_counts_streams_on_the_same_engine(self):
        first = self.make()
        second = self.make()
        self.assertEqual(first.concurrency, 1)
        self.assertEqual(second.concurrency, 2)
        other = usage.UsageRecorder(client_ip=None, engine_run_id="run2", model="m",
                                    endpoint="/v1/chat/completions", stream=True)
        self.assertEqual(other.concurrency, 1)


class SummaryAndEventTests(unittest.TestCase):
    """Real SQLite file, so the aggregation SQL is exercised end to end."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "spark_studio.db"
        with mock.patch.object(db, "DB_PATH", path):
            conn = db._connect()
        self.addCleanup(conn.close)
        patcher = mock.patch.object(db, "_conn", conn)
        patcher.start()
        self.addCleanup(patcher.stop)
        db.init()

    def row(self, **over):
        base = {
            "id": over.pop("id", "x"),
            "started_at": time.time(),
            "ended_at": time.time(),
            "node": "node-a",
            "client_ip": "198.51.100.1",
            "engine_run_id": "run1",
            "model": "qwen",
            "endpoint": "/v1/chat/completions",
            "stream": 1,
            "status": "ok",
            "error": None,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_tokens": 25,
            "cache_hit_pct": 25.0,
            "thinking": 0,
            "concurrency": 1,
            "ttft_ms": 200.0,
            "decode_secs": 2.0,
            "tok_s": 25.0,
            "peak_tok_s": 30.0,
            "samples": [10, 20, 20],
        }
        base.update(over)
        return base

    def test_summary_groups_by_machine_and_model(self):
        db.usage_insert(self.row(id="a"))
        db.usage_insert(self.row(id="b", node="node-b", model="gemma",
                                 prompt_tokens=None, cached_tokens=None,
                                 cache_hit_pct=None, completion_tokens=10,
                                 tok_s=None, decode_secs=None, thinking=1))
        out = db.usage_summary(0)
        self.assertEqual(out["totals"]["requests"], 2)
        self.assertEqual(out["totals"]["completion_tokens"], 60)
        self.assertEqual(out["totals"]["thinking_requests"], 1)
        # Cache % only counts requests that reported prompt tokens.
        self.assertEqual(out["totals"]["cache_hit_pct"], 25.0)
        self.assertEqual(out["totals"]["avg_tok_s"], 25.0)
        self.assertEqual({n["node"] for n in out["by_node"]}, {"node-a", "node-b"})
        self.assertEqual({m["model"] for m in out["by_model"]}, {"qwen", "gemma"})

    def test_since_filters_old_events(self):
        db.usage_insert(self.row(id="old", started_at=1000.0))
        db.usage_insert(self.row(id="new"))
        self.assertEqual(db.usage_summary(2000.0)["totals"]["requests"], 1)
        self.assertEqual([r["id"] for r in db.usage_list(since=2000.0)], ["new"])

    def test_event_detail_round_trips_samples(self):
        db.usage_insert(self.row(id="detail", samples=[1, 2, 3]))
        got = db.usage_get("detail")
        self.assertEqual(got["samples"], [1, 2, 3])
        self.assertIsNone(db.usage_get("missing"))

    def test_empty_summary_has_no_division_errors(self):
        out = db.usage_summary(0)
        self.assertEqual(out["totals"]["requests"], 0)
        self.assertIsNone(out["totals"]["cache_hit_pct"])
        self.assertIsNone(out["totals"]["avg_tok_s"])


if __name__ == "__main__":
    unittest.main()
