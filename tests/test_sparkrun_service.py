from pathlib import Path
import unittest
from unittest import mock

import sparkrun_service
import yaml


class BundledRecipeTests(unittest.TestCase):
    def test_bundled_recipe_is_listed_when_registry_listing_succeeds(self):
        with mock.patch.object(
            sparkrun_service.subprocess,
            "run",
            return_value=mock.Mock(returncode=0, stdout="[]"),
        ):
            recipes = sparkrun_service.list_recipes()

        recipe = next(
            r for r in recipes
            if r["ref"] == "@studio/qwen3.5-122b-a10b-int4-fp8-hybrid"
        )
        self.assertEqual(recipe["model"], "bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid")
        self.assertEqual(recipe["min_nodes"], 1)
        self.assertEqual(recipe["max_nodes"], 1)

    def test_bundled_recipe_ref_resolves_only_to_bundled_directory(self):
        target = sparkrun_service.resolve_recipe_target(
            "@studio/qwen3.5-122b-a10b-int4-fp8-hybrid"
        )
        self.assertEqual(Path(target).name, "qwen3.5-122b-a10b-int4-fp8-hybrid.yaml")
        self.assertEqual(Path(target).parent, sparkrun_service.BUNDLED_RECIPES_DIR)

        self.assertEqual(
            sparkrun_service.resolve_recipe_target("@official/example"),
            "@official/example",
        )
        self.assertEqual(
            sparkrun_service.resolve_recipe_target("@studio/../../etc/passwd"),
            "@studio/../../etc/passwd",
        )

    def test_bundled_recipe_path_is_canonicalized_to_studio_ref(self):
        path = (
            sparkrun_service.BUNDLED_RECIPES_DIR
            / "deepseek-v4-flash-0731-ds4-fast-agent.yaml"
        )

        self.assertEqual(
            sparkrun_service.canonical_recipe_ref(str(path)),
            "@studio/deepseek-v4-flash-0731-ds4-fast-agent",
        )
        self.assertEqual(
            sparkrun_service.canonical_recipe_ref("@official/example"),
            "@official/example",
        )
        self.assertEqual(
            sparkrun_service.canonical_recipe_ref("/tmp/untrusted.yaml"),
            "/tmp/untrusted.yaml",
        )

    def test_hybrid_recipe_preserves_runtime_requirements(self):
        path = Path(
            sparkrun_service.resolve_recipe_target(
                "@studio/qwen3.5-122b-a10b-int4-fp8-hybrid"
            )
        )
        recipe = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertNotIn("model_revision", recipe)
        self.assertEqual(recipe["executor_config"], {"entrypoint": "", "user": "root"})
        self.assertEqual(recipe["defaults"]["max_model_len"], 20000)
        self.assertEqual(recipe["mods"], ["mods/qwen35-122b-hybrid-dflash"])


class DockerFallbackTests(unittest.TestCase):
    """sparkrun's state store can lose a job whose container is still serving.

    `cluster status --json` then reports zero jobs and an idle host while
    `sparkrun_<jobid>_solo` answers /v1/models. The containers are the ground
    truth, so status parsing falls back to them — otherwise the dashboard
    adopts nothing and every downstream surface says no model is loaded.
    """

    def setUp(self):
        sparkrun_service._ORPHAN_REF_CACHE.clear()

    def test_orphaned_container_is_recovered_with_ref_from_its_model(self):
        empty_status = mock.Mock(
            returncode=0,
            stdout='{"groups": {}, "solo_entries": [], "idle_hosts": ["192.168.0.132"]}',
        )
        docker_ps = mock.Mock(
            returncode=0,
            stdout="sparkrun_1fc4097aff27_solo\nspark-searxng\n",
        )

        def fake_run(cmd, **_kwargs):
            if cmd[1:3] == ["cluster", "status"]:
                return empty_status
            if cmd[-2:] == ["--format", "{{.Names}}"]:
                return docker_ps
            raise AssertionError(f"unexpected command: {cmd}")

        with (
            mock.patch.object(sparkrun_service, "sparkrun_bin", return_value="/usr/bin/sparkrun"),
            mock.patch.object(sparkrun_service.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(sparkrun_service.subprocess, "run", side_effect=fake_run),
            mock.patch.object(
                sparkrun_service, "export_running_recipe",
                return_value={"model": "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"},
            ),
            mock.patch.object(
                sparkrun_service, "find_recipes_by_model",
                return_value=[{"ref": "@studio/qwen3.6-35b-a3b-unsloth-nvfp4-fast"}],
            ) as find,
        ):
            jobs = sparkrun_service.parse_status()
            # Second sweep: the ref recovery is cached, not re-paid.
            sparkrun_service.parse_status()

        (job,) = jobs
        self.assertEqual(job["jobid"], "1fc4097aff27")
        self.assertEqual(job["ref"], "@studio/qwen3.6-35b-a3b-unsloth-nvfp4-fast")
        self.assertEqual(job["containers"], ["sparkrun_1fc4097aff27_solo"])
        self.assertEqual(job["hosts"], [{"role": "solo", "ip": "127.0.0.1", "status": "Up"}])
        find.assert_called_once()

    def test_reported_jobs_win_and_no_containers_means_no_jobs(self):
        real_status = mock.Mock(
            returncode=0,
            stdout='{"groups": {}, "solo_entries": [{"cluster_id": "sparkrun_abc123", '
                   '"meta": {"recipe": "@official/x"}, "host": "10.0.0.1", "status": "Up"}]}',
        )
        with (
            mock.patch.object(sparkrun_service, "sparkrun_bin", return_value="/usr/bin/sparkrun"),
            mock.patch.object(sparkrun_service.subprocess, "run", return_value=real_status),
            mock.patch.object(sparkrun_service, "_jobs_from_docker") as fallback,
        ):
            jobs = sparkrun_service.parse_status()
        self.assertEqual(jobs[0]["jobid"], "abc123")
        fallback.assert_not_called()

        empty = mock.Mock(returncode=0, stdout='{"groups": {}, "solo_entries": []}')
        no_containers = mock.Mock(returncode=0, stdout="spark-searxng\n")

        def fake_run(cmd, **_kwargs):
            return empty if cmd[1:3] == ["cluster", "status"] else no_containers

        with (
            mock.patch.object(sparkrun_service, "sparkrun_bin", return_value="/usr/bin/sparkrun"),
            mock.patch.object(sparkrun_service.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(sparkrun_service.subprocess, "run", side_effect=fake_run),
        ):
            self.assertEqual(sparkrun_service.parse_status(), [])


if __name__ == "__main__":
    unittest.main()
