"""Recommend: ranking rules that decide what the first-run wizard proposes.

The module's promise is in its docstring: nothing is hardcoded to a model
name — ranking comes from live signals (✓ working recipes, cached weights,
bench history, MoE active params, quant-aware memory fit). These tests pin
the parsing of those signals (name → params/active/quant/weight), the scoring
levers (measured tok/s beats estimates; proven ×1.5; cached ×1.3; NVFP4 ×0.4)
and the memory-fit gate, so a future tweak can't silently stop recommending
the models users already proved on their own box.
"""

import unittest
from unittest import mock

import recommend


class SizeParsingTests(unittest.TestCase):
    def test_total_and_active_moe_params(self):
        self.assertEqual(recommend._parse_size("Qwen3.6-35B-A3B"), (35.0, 3.0))

    def test_dense_model_has_no_active(self):
        self.assertEqual(recommend._parse_size("Qwen3.8-27B"), (27.0, None))

    def test_decimal_sizes(self):
        self.assertEqual(recommend._parse_size("LFM2.5-2.6B"), (2.6, None))

    def test_active_token_not_double_counted_as_total(self):
        # "-A10B" alone must not read as total=10 — active is skipped for total.
        total, active = recommend._parse_size("Big-MoE-A10B")
        self.assertEqual(active, 10.0)
        self.assertIsNone(total)

    def test_no_size_tokens(self):
        self.assertEqual(recommend._parse_size("Mistral-Small"), (None, None))

    def test_lowercase_b_accepted(self):
        self.assertEqual(recommend._parse_size("qwen3-14b-a3b"), (14.0, 3.0))


class QuantTests(unittest.TestCase):
    def test_nvfp4_detected(self):
        self.assertEqual(recommend._quant_of("Model-NVFP4"), "nvfp4")

    def test_int4_variants(self):
        for name in ("Model-AWQ-4bit", "model-GPTQ", "model-autoround"):
            self.assertIsNotNone(recommend._quant_of(name), name)

    def test_bf16(self):
        self.assertEqual(recommend._quant_of("Model-BF16"), "bf16")

    def test_unquantized_is_none(self):
        self.assertIsNone(recommend._quant_of("Qwen3.8-27B"))

    def test_est_weight_gb_quant_aware(self):
        self.assertAlmostEqual(recommend._est_weight_gb("X-35B-NVFP4", 35.0), 19.2)
        self.assertAlmostEqual(recommend._est_weight_gb("X-27B-FP8", 27.0), 28.4)
        self.assertAlmostEqual(recommend._est_weight_gb("X-27B", 27.0), 54.0)

    def test_est_weight_gb_without_total(self):
        self.assertIsNone(recommend._est_weight_gb("X-NVFP4", None))


class SpeedScoreTests(unittest.TestCase):
    def _entry(self, **over):
        e = {"model": "m", "tokens_per_sec": None, "active_params_b": None,
             "params_b": None, "quant": None, "proven": False, "cached": False}
        e.update(over)
        return e

    def test_measured_tps_beats_any_estimate(self):
        slow_box = self._entry(tokens_per_sec=50.0)
        fast_guess = self._entry(active_params_b=1.0)
        self.assertGreater(recommend._speed_score(slow_box),
                           recommend._speed_score(self._entry(active_params_b=3.0)))
        self.assertEqual(recommend._speed_score(slow_box), 50.0 * 1.0)

    def test_moe_active_params_lever(self):
        moe = self._entry(params_b=122.0, active_params_b=10.0)
        dense = self._entry(params_b=27.0)
        self.assertGreater(recommend._speed_score(moe), recommend._speed_score(dense))

    def test_nvfp4_penalty_only_on_estimates(self):
        q = self._entry(active_params_b=8.0, quant="nvfp4")
        plain = self._entry(active_params_b=8.0)
        self.assertAlmostEqual(recommend._speed_score(q),
                               recommend._speed_score(plain) * 0.4)
        # measured truth is NOT penalised — it already includes kernel reality
        m_q = self._entry(tokens_per_sec=20.0, quant="nvfp4")
        self.assertAlmostEqual(recommend._speed_score(m_q), 20.0)

    def test_proven_and_cached_multipliers(self):
        base = recommend._speed_score(self._entry(active_params_b=10.0))
        self.assertAlmostEqual(recommend._speed_score(
            self._entry(active_params_b=10.0, proven=True)), base * 1.5)
        self.assertAlmostEqual(recommend._speed_score(
            self._entry(active_params_b=10.0, cached=True)), base * 1.3)
        self.assertAlmostEqual(recommend._speed_score(
            self._entry(active_params_b=10.0, proven=True, cached=True)), base * 1.95)


class QualityScoreTests(unittest.TestCase):
    def test_bigger_params_win(self):
        self.assertGreater(recommend._quality_score({"params_b": 35.0}),
                           recommend._quality_score({"params_b": 8.0}))

    def test_bf16_beats_lower_quant_same_size(self):
        bf16 = recommend._quality_score({"params_b": 27.0, "quant": "bf16"})
        int4 = recommend._quality_score({"params_b": 27.0, "quant": "nvfp4"})
        self.assertGreater(bf16, int4)

    def test_proven_boost(self):
        self.assertGreater(
            recommend._quality_score({"params_b": 27.0, "proven": True}),
            recommend._quality_score({"params_b": 27.0}))


class WhyTests(unittest.TestCase):
    def test_reason_mentions_the_decisive_signals(self):
        e = {"proven": True, "tokens_per_sec": 41.7, "cached": True,
             "active_params_b": 3.0, "params_b": 35.0, "est_weight_gb": 19.3,
             "caps": {"tool_call_parser": "qwen3_coder"}, "source": "your recipe"}
        self.assertIn("ran successfully", recommend._why(e, "fastest"))
        self.assertIn("measured 42 tok/s", recommend._why(e, "fastest"))
        self.assertIn("already downloaded", recommend._why(e, "fastest"))
        self.assertIn("3B active", recommend._why(e, "fastest"))
        self.assertIn("qwen3_coder", recommend._why(e, "tool_calling"))
        self.assertIn("19.3 GB", recommend._why(e, "low_memory"))

    def test_community_recipe_provenance(self):
        e = {"source": "community recipe", "params_b": None, "caps": {}}
        self.assertIn("Spark-validated", recommend._why(e, "fastest"))

    def test_no_signals_still_gives_a_reason(self):
        self.assertEqual(recommend._why({"source": "cached model"}, "fastest"),
                         "fits this Spark")


def _wire(**over):
    """Patch every external signal recommend reads; return the mocks."""
    host = {"total_memory_gb": 128, "summary": "1× GB10 · 128 GB"}
    host.update(over.pop("host", {}))
    patches = {
        "hostinfo.probe_host": mock.Mock(return_value=host),
        "db.recipes_list": mock.Mock(return_value=over.get("saved", [])),
        "db.bench_list": mock.Mock(return_value=over.get("bench", [])),
        "models.scan": mock.Mock(return_value=over.get("cached", [])),
        "registry.all_recipes": mock.Mock(return_value=over.get("registry", [])),
        "recipe_brain.capabilities_for": mock.Mock(
            side_effect=lambda m: over.get("caps", {}).get(m, {})),
    }
    return patches


def _run_collect(**over):
    mocks = _wire(**over)
    with mock.patch("hostinfo.probe_host", mocks["hostinfo.probe_host"]), \
         mock.patch("db.recipes_list", mocks["db.recipes_list"]), \
         mock.patch("db.bench_list", mocks["db.bench_list"]), \
         mock.patch("models.scan", mocks["models.scan"]), \
         mock.patch("registry.all_recipes", mocks["registry.all_recipes"]), \
         mock.patch("recipe_brain.capabilities_for", mocks["recipe_brain.capabilities_for"]):
        return recommend._collect()


class CollectTests(unittest.TestCase):
    def test_saved_working_recipe_is_marked_proven(self):
        saved = [{"id": 1, "name": "my-qwen", "model": "Qwen/Qwen3.6-35B-A3B-NVFP4",
                  "tags": "working", "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""}]
        entries = _run_collect(saved=saved)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["proven"])
        self.assertEqual(entries[0]["source"], "your recipe")

    def test_fix_tagged_recipe_is_dropped(self):
        saved = [{"id": 1, "name": "broken", "model": "Qwen/Qwen3.6-35B-A3B-NVFP4",
                  "tags": "fix", "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""}]
        self.assertEqual(_run_collect(saved=saved), [])

    def test_working_and_fix_is_kept(self):
        # a recipe repaired by an agent carries both tags — proven signal wins
        saved = [{"id": 1, "name": "healed", "model": "Qwen/Qwen3.6-35B-A3B-NVFP4",
                  "tags": "fix, working", "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""}]
        self.assertEqual(len(_run_collect(saved=saved)), 1)

    def test_model_without_org_slash_ignored(self):
        saved = [{"id": 1, "name": "local", "model": "not-a-repo",
                  "tags": "working", "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""}]
        self.assertEqual(_run_collect(saved=saved), [])

    def test_non_chat_models_never_recommended(self):
        saved = [{"id": i, "name": n, "model": m, "tags": "working",
                  "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""}
                 for i, (n, m) in enumerate([
                     ("emb", "Qwen/Qwen3-Embedding-0.6B"),
                     ("rerank", "BAAI/bge-reranker-v2-m3"),
                     ("guard", "Meta-Llama/Llama-Guard-4-12B")])]
        self.assertEqual(_run_collect(saved=saved), [])

    def test_oversized_model_dropped_by_memory_gate(self):
        # bf16 405B → est 810 GB >> 128*0.75 budget
        saved = [{"id": 1, "name": "huge", "model": "Meta/Llama-405B",
                  "tags": "working", "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""}]
        self.assertEqual(_run_collect(saved=saved), [])

    def test_cached_model_gets_a_flat_vllm_recipe(self):
        cached = [{"repo": "unsloth/Qwen3.6-35B-A3B-NVFP4", "size_gb": 19.0}]
        entries = _run_collect(cached=cached)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertTrue(e["cached"])
        self.assertEqual(e["source"], "cached model")
        self.assertEqual(e["recipe"]["engine"], "vllm")
        # real on-disk size beats the name-based estimate
        self.assertEqual(e["est_weight_gb"], 19.0)

    def test_gguf_cache_skipped(self):
        cached = [{"repo": "bartowski/model-GGUF", "size_gb": 12.0}]
        self.assertEqual(_run_collect(cached=cached), [])

    def test_bench_history_lifts_the_right_model(self):
        saved = [
            {"id": 1, "name": "a", "model": "Org/Alpha-8B", "tags": "working",
             "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""},
            {"id": 2, "name": "b", "model": "Org/Beta-8B", "tags": "working",
             "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""}]
        bench = [{"recipe_id": 2, "tokens_per_sec": 90.0}]
        entries = _run_collect(saved=saved, bench=bench)
        by = {e["model"]: e for e in entries}
        self.assertEqual(by["Org/Beta-8B"]["tokens_per_sec"], 90.0)
        self.assertIsNone(by["Org/Alpha-8B"]["tokens_per_sec"])

    def test_duplicate_model_merges_and_proven_wins(self):
        rec = mock.Mock(model="Qwen/Qwen3.6-35B-A3B", min_nodes=1,
                       name="community-qwen", raw_yaml="")
        saved = [{"id": 1, "name": "mine", "model": "Qwen/Qwen3.6-35B-A3B",
                  "tags": "working", "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""}]
        with mock.patch("forge._from_registry_recipe",
                        return_value={"name": "community-qwen"}):
            entries = _run_collect(saved=saved, registry=[rec])
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["proven"])
        self.assertEqual(entries[0]["source"], "your recipe")

    def test_multi_node_registry_recipe_excluded_from_starter(self):
        rec = mock.Mock(model="DeepSeek/DS-V3", min_nodes=2, name="cluster-only",
                        raw_yaml="")
        with mock.patch("forge._from_registry_recipe", return_value={}):
            entries = _run_collect(registry=[rec])
        self.assertEqual(entries, [])


class RecommendShapeTests(unittest.TestCase):
    def test_five_categories_k_limit_and_public_fields(self):
        one = _run_collect(saved=[{"id": 1, "name": "mine", "model": "Org/Model-8B",
                                   "tags": "working", "engine": "vllm",
                                   "args": {}, "env": {}, "raw_cmd": ""}])
        self.assertTrue(one)  # sanity
        mocks = _wire(saved=[{"id": 1, "name": "mine", "model": "Org/Model-8B",
                              "tags": "working", "engine": "vllm",
                              "args": {}, "env": {}, "raw_cmd": ""}])
        with mock.patch("hostinfo.probe_host", mocks["hostinfo.probe_host"]), \
             mock.patch("db.recipes_list", mocks["db.recipes_list"]), \
             mock.patch("db.bench_list", mocks["db.bench_list"]), \
             mock.patch("models.scan", mocks["models.scan"]), \
             mock.patch("registry.all_recipes", mocks["registry.all_recipes"]), \
             mock.patch("recipe_brain.capabilities_for", mocks["recipe_brain.capabilities_for"]):
            out = recommend.recommend(k=2)
        self.assertEqual(sorted(out["categories"]),
                         ["best_quality", "coding", "fastest", "low_memory", "tool_calling"])
        self.assertEqual(out["candidates"], 1)
        for items in out["categories"].values():
            self.assertLessEqual(len(items), 2)
            for item in items:
                for field in ("category", "model", "name", "source", "reason",
                              "proven", "recipe"):
                    self.assertIn(field, item)

    def test_coding_category_only_houses_coders(self):
        saved = [{"id": 1, "name": "gen", "model": "Org/General-8B", "tags": "working",
                  "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""},
                 {"id": 2, "name": "coder", "model": "Qwen/Qwen3-Coder-30B-A3B",
                  "tags": "working", "engine": "vllm", "args": {}, "env": {}, "raw_cmd": ""}]
        mocks = _wire(saved=saved)
        with mock.patch("hostinfo.probe_host", mocks["hostinfo.probe_host"]), \
             mock.patch("db.recipes_list", mocks["db.recipes_list"]), \
             mock.patch("db.bench_list", mocks["db.bench_list"]), \
             mock.patch("models.scan", mocks["models.scan"]), \
             mock.patch("registry.all_recipes", mocks["registry.all_recipes"]), \
             mock.patch("recipe_brain.capabilities_for", mocks["recipe_brain.capabilities_for"]):
            out = recommend.recommend(k=3)
        coders = [i["model"] for i in out["categories"]["coding"]]
        self.assertEqual(coders, ["Qwen/Qwen3-Coder-30B-A3B"])


if __name__ == "__main__":
    unittest.main()
