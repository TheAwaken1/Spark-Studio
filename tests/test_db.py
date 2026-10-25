"""The SQLite store behind recipes, safety snapshots, runs and bench history.

Every test runs against a throwaway database file: the module-level
connection is swapped for one opened on a temp path, so the real
data/spark_studio.db is never read or written.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import db

# The connection db.py opened (and migrated) at import, before any test swaps it.
_IMPORT_CONN = db._conn
# Mirrors the module-level migrations at the bottom of db.py.
_MIGRATIONS = (
    ("recipes", "raw_cmd", "TEXT"),
    ("benchmarks", "engine_version", "TEXT"),
    ("benchy_runs", "engine_version", "TEXT"),
    ("runs", "meta_json", "TEXT"),
)


def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


class TempDbTestCase(unittest.TestCase):
    """Point db._conn at a fresh SQLite file for the duration of a test."""

    migrate = True

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = Path(tmp.name) / "spark_studio.db"
        with mock.patch.object(db, "DB_PATH", self.db_path):
            self.conn = db._connect()
        self.addCleanup(self.conn.close)
        # Throwaway file: skip fsyncs so the suite stays fast on slow disks.
        self.conn.execute("PRAGMA synchronous=OFF")
        patcher = mock.patch.object(db, "_conn", self.conn)
        patcher.start()
        self.addCleanup(patcher.stop)
        if self.migrate:
            db.init()
            for table, col, decl in _MIGRATIONS:
                db._add_col_if_missing(table, col, decl)

    def make_recipe(self, **over):
        data = {"name": "r", "engine": "vllm", "model": "m",
                "args": {"max_model_len": 262144}, "env": {}}
        data.update(over)
        return db.recipes_upsert(data)


class SchemaTests(TempDbTestCase):
    migrate = False

    def test_init_creates_every_table(self):
        db.init()
        tables = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("recipes", "runs", "benchmarks", "benchy_runs", "tooleval_runs",
                  "agentlab_runs", "taskbench_runs", "recipe_snapshots"):
            self.assertIn(t, tables)

    def test_init_is_idempotent_and_keeps_data(self):
        db.init()
        rec = db.recipes_upsert({"name": "keep", "engine": "vllm", "args": {}})
        db.init()
        db.init()
        self.assertEqual(db.recipes_get(rec["id"])["name"], "keep")

    def test_add_col_if_missing_upgrades_a_legacy_table_once(self):
        # A pre-raw_cmd install: the recipes table exists without the column.
        self.conn.execute(
            "CREATE TABLE recipes (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,"
            " engine TEXT NOT NULL, model TEXT, args_json TEXT NOT NULL,"
            " env_json TEXT NOT NULL DEFAULT '{}', notes TEXT DEFAULT '', tags TEXT DEFAULT '',"
            " created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)")
        self.conn.execute(
            "INSERT INTO recipes (name, engine, args_json, created_at, updated_at)"
            " VALUES ('old', 'vllm', '{}', 1, 1)")
        self.assertNotIn("raw_cmd", _columns(self.conn, "recipes"))
        db._add_col_if_missing("recipes", "raw_cmd", "TEXT")
        db._add_col_if_missing("recipes", "raw_cmd", "TEXT")  # second call is a no-op
        cols = _columns(self.conn, "recipes")
        self.assertEqual(cols.count("raw_cmd"), 1)
        # init() on the upgraded table leaves the old row readable.
        db.init()
        old = db.recipes_list()[0]
        self.assertEqual(old["name"], "old")
        self.assertIsNone(old["raw_cmd"])

    def test_migrations_bring_a_fresh_db_to_the_current_shape(self):
        db.init()
        for table, col, decl in _MIGRATIONS:
            db._add_col_if_missing(table, col, decl)
        for table, col, _ in _MIGRATIONS:
            self.assertIn(col, _columns(self.conn, table))

    def test_module_import_applied_the_migrations(self):
        # The connection db.py opened at import time already has the columns
        # its queries rely on (runs_insert writes meta_json, etc.).
        # Read-only PRAGMA on the import-time connection.
        for table, col, _ in _MIGRATIONS:
            self.assertIn(col, _columns(_IMPORT_CONN, table))


class RecipeTests(TempDbTestCase):
    def test_upsert_insert_round_trips_args_and_env(self):
        rec = self.make_recipe(args={"tp": 2, "flags": ["--a", "--b"]},
                               env={"HF_HOME": "/models"}, raw_cmd="vllm serve m",
                               notes="n", tags="x")
        self.assertIsInstance(rec["id"], int)
        self.assertEqual(rec["args"], {"tp": 2, "flags": ["--a", "--b"]})
        self.assertEqual(rec["env"], {"HF_HOME": "/models"})
        self.assertEqual(rec["raw_cmd"], "vllm serve m")
        self.assertNotIn("args_json", rec)
        self.assertNotIn("env_json", rec)
        stored = self.conn.execute(
            "SELECT args_json, env_json FROM recipes WHERE id=?", (rec["id"],)).fetchone()
        self.assertEqual(json.loads(stored["args_json"]), rec["args"])
        self.assertEqual(json.loads(stored["env_json"]), rec["env"])

    def test_missing_args_and_env_are_stored_as_empty_objects(self):
        rec = db.recipes_upsert({"name": "bare", "engine": "llamacpp"})
        self.assertEqual(rec["args"], {})
        self.assertEqual(rec["env"], {})
        self.assertEqual(rec["notes"], "")
        self.assertEqual(rec["tags"], "")

    def test_insert_does_not_snapshot(self):
        rec = self.make_recipe()
        self.assertEqual(db.recipe_snapshots_list(rec["id"]), [])

    def test_update_snapshots_the_previous_version_by_default(self):
        rec = self.make_recipe()
        updated = db.recipes_upsert({**rec, "args": {"max_model_len": 32768}})
        self.assertEqual(updated["args"], {"max_model_len": 32768})
        history = db.recipe_snapshots_list(rec["id"])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["source"], "edit")
        self.assertEqual(history[0]["recipe"]["args"], {"max_model_len": 262144})
        self.assertNotIn("id", history[0]["recipe"])

    def test_snapshot_source_is_taken_from_the_payload(self):
        rec = self.make_recipe()
        data = {**rec, "notes": "agent edit", "_snapshot_source": "agent"}
        db.recipes_upsert(data)
        self.assertEqual(db.recipe_snapshots_list(rec["id"])[0]["source"], "agent")

    def test_snapshot_false_skips_history(self):
        rec = self.make_recipe()
        db.recipes_upsert({**rec, "notes": "quiet"}, _snapshot=False)
        self.assertEqual(db.recipes_get(rec["id"])["notes"], "quiet")
        self.assertEqual(db.recipe_snapshots_list(rec["id"]), [])

    def test_list_is_newest_first_and_get_missing_is_none(self):
        with mock.patch.object(db, "now", return_value=100):
            a = self.make_recipe(name="a")
        with mock.patch.object(db, "now", return_value=200):
            b = self.make_recipe(name="b")
        self.assertEqual([r["id"] for r in db.recipes_list()], [b["id"], a["id"]])
        self.assertIsNone(db.recipes_get(999999))

    def test_delete(self):
        rec = self.make_recipe()
        db.recipes_delete(rec["id"])
        self.assertIsNone(db.recipes_get(rec["id"]))

    def test_find_sparkrun_matches_on_the_ref_not_the_name(self):
        self.make_recipe(name="plain")
        hit = self.make_recipe(name="renamed by user",
                               args={"_sparkrun": {"ref": "community/qwen"}})
        self.assertEqual(db.recipes_find_sparkrun("community/qwen")["id"], hit["id"])
        self.assertIsNone(db.recipes_find_sparkrun("community/other"))


class SnapshotTests(TempDbTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(db, "now", return_value=1000)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.rec = self.make_recipe()

    def test_take_returns_none_for_unknown_recipe(self):
        self.assertIsNone(db.recipe_snapshot_take(999999))

    def test_take_dedupes_identical_content(self):
        first = db.recipe_snapshot_take(self.rec["id"])
        again = db.recipe_snapshot_take(self.rec["id"], source="other")
        self.assertEqual(first, again)
        self.assertEqual(len(db.recipe_snapshots_list(self.rec["id"])), 1)

    def test_take_records_a_new_snapshot_after_a_change(self):
        first = db.recipe_snapshot_take(self.rec["id"])
        db.recipes_upsert({**self.rec, "notes": "changed"}, _snapshot=False)
        second = db.recipe_snapshot_take(self.rec["id"], source="manual")
        self.assertNotEqual(first, second)
        history = db.recipe_snapshots_list(self.rec["id"])
        # Newest first.
        self.assertEqual([s["id"] for s in history], [second, first])
        self.assertEqual(history[0]["source"], "manual")
        self.assertEqual(history[0]["recipe"]["notes"], "changed")

    def test_history_is_capped_at_snapshot_keep(self):
        for i in range(db._SNAPSHOT_KEEP + 5):
            db.recipes_upsert({**self.rec, "notes": f"edit {i}"})
        history = db.recipe_snapshots_list(self.rec["id"])
        self.assertEqual(len(history), db._SNAPSHOT_KEEP)
        # The oldest snapshots are the ones dropped.
        self.assertEqual(history[0]["recipe"]["notes"], f"edit {db._SNAPSHOT_KEEP + 3}")

    def test_list_is_scoped_to_one_recipe(self):
        other = self.make_recipe(name="other")
        db.recipe_snapshot_take(self.rec["id"])
        self.assertEqual(db.recipe_snapshots_list(other["id"]), [])

    def test_restore_brings_back_the_old_state_and_snapshots_the_current_one(self):
        rid = self.rec["id"]
        db.recipes_upsert({**self.rec, "args": {"max_model_len": 32768}, "notes": "broken"})
        good = db.recipe_snapshots_list(rid)[0]
        self.assertEqual(good["recipe"]["args"], {"max_model_len": 262144})

        restored = db.recipe_snapshot_restore(rid, good["id"])

        self.assertEqual(restored["id"], rid)
        self.assertEqual(restored["args"], {"max_model_len": 262144})
        self.assertEqual(restored["notes"], "")
        self.assertEqual(db.recipes_get(rid)["args"], {"max_model_len": 262144})
        history = db.recipe_snapshots_list(rid)
        # Exactly one new snapshot: the pre-restore copy of the broken state.
        self.assertEqual(len(history), 2)
        pre = [s for s in history if s["source"] == "pre-restore"]
        self.assertEqual(len(pre), 1)
        self.assertEqual(pre[0]["recipe"]["notes"], "broken")
        self.assertEqual(pre[0]["recipe"]["args"], {"max_model_len": 32768})

    def test_restore_is_undoable(self):
        rid = self.rec["id"]
        db.recipes_upsert({**self.rec, "notes": "v2"})
        db.recipe_snapshot_restore(rid, db.recipe_snapshots_list(rid)[0]["id"])
        pre = [s for s in db.recipe_snapshots_list(rid) if s["source"] == "pre-restore"][0]
        db.recipe_snapshot_restore(rid, pre["id"])
        self.assertEqual(db.recipes_get(rid)["notes"], "v2")

    def test_restore_rejects_unknown_or_foreign_snapshot(self):
        other = self.make_recipe(name="other")
        db.recipes_upsert({**other, "notes": "x"})
        foreign = db.recipe_snapshots_list(other["id"])[0]["id"]
        self.assertIsNone(db.recipe_snapshot_restore(self.rec["id"], foreign))
        self.assertIsNone(db.recipe_snapshot_restore(self.rec["id"], 999999))
        self.assertEqual(db.recipe_snapshots_list(self.rec["id"]), [])

    def test_deleting_a_recipe_cascades_to_its_snapshots(self):
        db.recipes_upsert({**self.rec, "notes": "x"})
        db.recipes_delete(self.rec["id"])
        self.assertEqual(db.recipe_snapshots_list(self.rec["id"]), [])


class StatusTagTests(TempDbTestCase):
    def tags(self, rid):
        return db.recipes_get(rid)["tags"]

    def test_ok_marks_working_and_clears_fix(self):
        rec = self.make_recipe(tags="fix, qwen")
        db.recipes_set_status_tag(rec["id"], True)
        self.assertEqual(self.tags(rec["id"]), "qwen, working")

    def test_not_ok_marks_fix_and_clears_working(self):
        rec = self.make_recipe(tags="working,qwen")
        db.recipes_set_status_tag(rec["id"], False)
        self.assertEqual(self.tags(rec["id"]), "fix, qwen")

    def test_flipping_back_and_forth_never_holds_both(self):
        rec = self.make_recipe(tags="")
        for ok in (True, False, True, False):
            db.recipes_set_status_tag(rec["id"], ok)
            tags = {t.strip() for t in self.tags(rec["id"]).split(",")}
            self.assertEqual(tags, {"working"} if ok else {"fix"})

    def test_unchanged_status_does_not_write(self):
        with mock.patch.object(db, "now", return_value=100):
            rec = self.make_recipe(tags="working")
        with mock.patch.object(db, "now", return_value=200):
            db.recipes_set_status_tag(rec["id"], True)
        after = db.recipes_get(rec["id"])
        self.assertEqual(after["tags"], "working")  # raw string not normalised
        self.assertEqual(after["updated_at"], 100)

    def test_status_change_does_not_snapshot(self):
        rec = self.make_recipe()
        db.recipes_set_status_tag(rec["id"], False)
        self.assertEqual(db.recipe_snapshots_list(rec["id"]), [])

    def test_unknown_recipe_is_ignored(self):
        self.assertIsNone(db.recipes_set_status_tag(999999, True))


class RunTests(TempDbTestCase):
    def run_row(self, rid, status="running", started_at=None, **over):
        row = {"id": rid, "engine": "vllm", "status": status, "cmd": "vllm serve m",
               "pid": 4242, "port": 8000, "started_at": started_at}
        row.update(over)
        db.runs_insert(row)

    def test_insert_and_get(self):
        rec = self.make_recipe()
        self.run_row("r1", recipe_id=rec["id"], meta_json='{"_label": "qwen"}', started_at=50)
        got = db.runs_get("r1")
        self.assertEqual(got["recipe_id"], rec["id"])
        self.assertEqual(got["pid"], 4242)
        self.assertEqual(got["port"], 8000)
        self.assertEqual(got["started_at"], 50)
        self.assertEqual(json.loads(got["meta_json"]), {"_label": "qwen"})
        self.assertIsNone(got["ended_at"])
        self.assertIsNone(db.runs_get("missing"))

    def test_insert_defaults_started_at_to_now(self):
        with mock.patch.object(db, "now", return_value=777):
            self.run_row("r1")
        self.assertEqual(db.runs_get("r1")["started_at"], 777)

    def test_update_sets_only_the_given_fields(self):
        self.run_row("r1")
        db.runs_update("r1", status="exited", ended_at=99, exit_code=137)
        got = db.runs_get("r1")
        self.assertEqual((got["status"], got["ended_at"], got["exit_code"]), ("exited", 99, 137))
        self.assertEqual(got["pid"], 4242)

    def test_update_without_fields_is_a_noop(self):
        self.run_row("r1")
        db.runs_update("r1")
        self.assertEqual(db.runs_get("r1")["status"], "running")

    def test_list_is_newest_first_and_limited(self):
        for i in range(5):
            self.run_row(f"r{i}", started_at=100 + i)
        self.assertEqual([r["id"] for r in db.runs_list()], ["r4", "r3", "r2", "r1", "r0"])
        self.assertEqual([r["id"] for r in db.runs_list(limit=2)], ["r4", "r3"])

    def test_list_running_filters_on_status(self):
        self.run_row("live-old", status="running", started_at=1)
        self.run_row("live-new", status="running", started_at=2)
        self.run_row("done", status="exited", started_at=3)
        self.run_row("boot", status="starting", started_at=4)
        self.run_row("bad", status="error", started_at=5)
        self.assertEqual([r["id"] for r in db.runs_list_running()], ["live-new", "live-old"])
        db.runs_update("live-new", status="exited")
        self.assertEqual([r["id"] for r in db.runs_list_running()], ["live-old"])

    def test_deleting_a_recipe_keeps_its_runs_but_unlinks_them(self):
        rec = self.make_recipe()
        self.run_row("r1", recipe_id=rec["id"])
        db.recipes_delete(rec["id"])
        self.assertIsNone(db.runs_get("r1")["recipe_id"])


class BenchTests(TempDbTestCase):
    def test_bench_insert_and_list_filtered_by_recipe(self):
        a = self.make_recipe(name="a")
        b = self.make_recipe(name="b")
        with mock.patch.object(db, "now", return_value=10):
            db.bench_insert("run-a", a["id"], {"tokens_per_sec": 42.5, "ttft_ms": 120.0,
                                               "prompt_tokens": 10, "completion_tokens": 256,
                                               "memory_mb": 9000.0, "extra": "kept"},
                            engine_version="0.11.0")
        with mock.patch.object(db, "now", return_value=20):
            db.bench_insert("run-b", b["id"], {"tokens_per_sec": 10.0})
        rows = db.bench_list(recipe_id=a["id"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["tokens_per_sec"], 42.5)
        self.assertEqual(row["completion_tokens"], 256)
        self.assertEqual(row["engine_version"], "0.11.0")
        self.assertEqual(json.loads(row["data_json"])["extra"], "kept")
        self.assertEqual([r["run_id"] for r in db.bench_list()], ["run-b", "run-a"])
        self.assertEqual(len(db.bench_list(limit=1)), 1)

    def test_bench_missing_metrics_are_null(self):
        db.bench_insert("r", None, {})
        row = db.bench_list()[0]
        self.assertIsNone(row["tokens_per_sec"])
        self.assertIsNone(row["engine_version"])

    def test_benchy_insert_get_and_list(self):
        bid = db.benchy_insert("run", None, "m", "http://x/v1", {"pp": [512]},
                               {"tg": 30.1}, 0, engine_version="v")
        got = db.benchy_get(bid)
        self.assertEqual(json.loads(got["params_json"]), {"pp": [512]})
        self.assertEqual(json.loads(got["result_json"]), {"tg": 30.1})
        self.assertEqual(got["exit_code"], 0)
        failed = db.benchy_insert("run", None, "m", "u", {}, None, 1)
        self.assertIsNone(db.benchy_get(failed)["result_json"])
        self.assertEqual(len(db.benchy_list()), 2)
        self.assertIsNone(db.benchy_get(999999))

    def test_tooleval_insert_and_list_newest_first(self):
        with mock.patch.object(db, "now", return_value=1):
            first = db.tooleval_insert({"model": "m1", "score": 0.5, "results_json": "[]"})
        with mock.patch.object(db, "now", return_value=2):
            second = db.tooleval_insert({"model": "m2", "score": 0.9})
        self.assertNotEqual(first, second)
        rows = db.tooleval_list()
        self.assertEqual([r["model"] for r in rows], ["m2", "m1"])
        self.assertEqual(rows[1]["score"], 0.5)
        self.assertEqual(rows[1]["results_json"], "[]")
        self.assertIsNone(rows[0]["results_json"])
        self.assertEqual(len(db.tooleval_list(limit=1)), 1)


if __name__ == "__main__":
    unittest.main()
