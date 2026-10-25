"""Guard rails for recipe changes: lint findings, DB snapshots, file snapshots.

The motivating incident: an agent cut a bundled recipe's max_model_len from
262144 to 32768, which silently broke Hermes Agent's 64K context minimum and
left nothing to roll back to.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import db
import recipe_guard


class LintTests(unittest.TestCase):
    def test_context_below_hermes_minimum_is_flagged(self):
        findings = recipe_guard.lint({"defaults": {"max_model_len": 32768}})
        self.assertEqual(findings[0]["code"], "hermes-context")
        self.assertIn("65,536", findings[0]["message"])

    def test_context_at_or_above_minimum_is_clean(self):
        self.assertEqual(recipe_guard.lint({"defaults": {"max_model_len": 65536}}), [])
        self.assertEqual(recipe_guard.lint({"defaults": {"max_model_len": 262144}}), [])

    def test_db_recipe_args_are_checked_too(self):
        findings = recipe_guard.lint({"args": {"max_model_len": 32768}})
        self.assertEqual(findings[0]["code"], "hermes-context")
        findings = recipe_guard.lint({"args": {"_sparkrun": {"max_model_len": 8192}}})
        self.assertEqual(findings[0]["code"], "hermes-context")

    def test_tiny_context_gets_its_own_warning(self):
        findings = recipe_guard.lint({"args": {"max_model_len": 2048}})
        self.assertEqual(findings[0]["code"], "tiny-context")

    def test_recipes_without_the_knob_are_clean(self):
        self.assertEqual(recipe_guard.lint({"args": {"model": "x"}}), [])
        self.assertEqual(recipe_guard.lint(None), [])

    def test_invalid_yaml_is_an_error(self):
        findings = recipe_guard.lint_yaml("defaults: [unclosed")
        self.assertEqual(findings[0]["code"], "invalid-yaml")
        self.assertEqual(findings[0]["level"], "error")

    def test_yaml_lint_matches_the_real_incident(self):
        yaml_text = "name: ds4\ndefaults:\n  port: 8000\n  max_model_len: 32768\n"
        findings = recipe_guard.lint_yaml(yaml_text)
        self.assertEqual([f["code"] for f in findings], ["hermes-context"])


class FileSnapshotTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "recipes").mkdir()
        (root / "snaps").mkdir()
        for name, target in (("BUNDLED_DIR", root / "recipes"), ("SNAPSHOT_DIR", root / "snaps")):
            patcher = mock.patch.object(recipe_guard, name, target)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.live = root / "recipes" / "demo.yaml"
        self.live.write_text("defaults:\n  max_model_len: 262144\n")

    def test_snapshot_restore_roundtrip(self):
        recipe_guard.snapshot_file(self.live, source="launch")
        good = recipe_guard.file_snapshots("demo")[0]["name"]
        self.live.write_text("defaults:\n  max_model_len: 32768\n")  # the bad agent edit
        recipe_guard.snapshot_file(self.live, source="launch")
        restored = recipe_guard.file_snapshot_restore("demo", good)
        self.assertEqual(restored, self.live)
        self.assertIn("262144", self.live.read_text())
        # The restore is undoable: the pre-restore (bad) content is still in
        # history — deduped into its existing launch snapshot rather than
        # copied again under a prerestore label.
        contents = [
            recipe_guard.file_snapshot_read("demo", s["name"])
            for s in recipe_guard.file_snapshots("demo")
        ]
        self.assertTrue(any("32768" in c for c in contents))

    def test_identical_content_is_not_snapshotted_twice(self):
        first = recipe_guard.snapshot_file(self.live)
        second = recipe_guard.snapshot_file(self.live)
        self.assertEqual(first, second)
        self.assertEqual(len(recipe_guard.file_snapshots("demo")), 1)

    def test_files_outside_the_bundled_dir_are_refused(self):
        foreign = Path(self._tmp.name) / "evil.yaml"
        foreign.write_text("x: 1")
        self.assertIsNone(recipe_guard.snapshot_file(foreign))

    def test_snapshot_names_cannot_traverse(self):
        self.assertIsNone(recipe_guard.file_snapshot_read("demo", "../../secrets"))
        self.assertIsNone(recipe_guard.file_snapshot_read("../demo", "x.yaml"))


class DbSnapshotTests(unittest.TestCase):
    def setUp(self):
        db.init()
        self.recipe = db.recipes_upsert(
            {"name": "snap-test", "engine": "vllm", "model": "m",
             "args": {"max_model_len": 262144}, "env": {}}
        )
        self.addCleanup(db.recipes_delete, self.recipe["id"])

    def test_updates_snapshot_the_previous_version_and_restore_undoes_them(self):
        rid = self.recipe["id"]
        db.recipes_upsert({**self.recipe, "args": {"max_model_len": 32768},
                           "_snapshot_source": "agent"})
        history = db.recipe_snapshots_list(rid)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["source"], "agent")
        self.assertEqual(history[0]["recipe"]["args"]["max_model_len"], 262144)
        restored = db.recipe_snapshot_restore(rid, history[0]["id"])
        self.assertEqual(restored["args"]["max_model_len"], 262144)
        # Restore first snapshots the pre-restore (broken) state.
        sources = [s["source"] for s in db.recipe_snapshots_list(rid)]
        self.assertIn("pre-restore", sources)

    def test_identical_saves_do_not_duplicate_history(self):
        rid = self.recipe["id"]
        db.recipes_upsert(dict(self.recipe))
        db.recipes_upsert(dict(self.recipe))
        self.assertEqual(len(db.recipe_snapshots_list(rid)), 1)

    def test_history_is_capped(self):
        rid = self.recipe["id"]
        for i in range(db._SNAPSHOT_KEEP + 5):
            db.recipes_upsert({**self.recipe, "notes": f"edit {i}"})
        self.assertLessEqual(len(db.recipe_snapshots_list(rid)), db._SNAPSHOT_KEEP)

    def test_deleting_the_recipe_removes_its_history(self):
        rid = self.recipe["id"]
        db.recipes_upsert({**self.recipe, "notes": "changed"})
        db.recipes_delete(rid)
        self.assertEqual(db.recipe_snapshots_list(rid), [])
        # Re-create so addCleanup's delete has a harmless target.
        self.recipe = db.recipes_upsert(
            {"name": "snap-test", "engine": "vllm", "args": {}, "env": {}}
        )


if __name__ == "__main__":
    unittest.main()
