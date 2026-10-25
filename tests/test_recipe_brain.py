import unittest
from unittest import mock

import recipe_brain


class Lfm25RecipeTests(unittest.TestCase):
    def test_detects_lfm25_from_repo_or_architecture(self):
        self.assertEqual(
            recipe_brain.detect_family({"repo": "LiquidAI/LFM2.5-2.6B"}),
            "lfm2.5",
        )
        self.assertEqual(
            recipe_brain.detect_family({"architecture": "Lfm2ForCausalLM"}),
            "lfm2.5",
        )

    def test_exposes_lfm25_tool_and_reasoning_capabilities(self):
        caps = recipe_brain.capabilities_for("LiquidAI/LFM2.5-2.6B")

        self.assertEqual(caps["tool_call_parser"], "lfm2")
        self.assertEqual(caps["reasoning_parser"], "qwen3")
        self.assertTrue(caps["supports_tools"])
        self.assertTrue(caps["supports_reasoning"])

    @mock.patch.object(recipe_brain, "_match_mods", return_value=[])
    def test_synthesized_recipe_enables_openai_tool_calls(self, _match_mods):
        result = recipe_brain.synthesize_recipe(
            {
                "repo": "LiquidAI/LFM2.5-2.6B",
                "architecture": "Lfm2ForCausalLM",
                "context": 131072,
                "context_known": True,
                "weight_gb": 5.1,
            },
            host={"effective_gpu_count": 1, "effective_memory_gb": 119},
        )

        self.assertIsNotNone(result)
        command = result["command"]
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertIn("--tool-call-parser lfm2", command)
        self.assertIn("--reasoning-parser qwen3", command)
        self.assertNotIn("--trust-remote-code", command)
        self.assertEqual(result["profile"]["family"], "lfm2.5")


if __name__ == "__main__":
    unittest.main()