"""Tests for the Qwen3 thinking-default policy (server._apply_thinking_default).

Policy: default Qwen3 reasoning models to medium thinking, applied at the
/api/engine/v1 proxy and /api/chat handlers. Explicit caller choices —
chat_template_kwargs.enable_thinking, or an inline /think|/no_think — always win.
"""
import unittest

import server


def apply(payload, path="chat/completions", served=("qwen3.8-27b-nvfp4",)):
    """Run the policy and report (changed, thinking, reasoning effort)."""
    changed = server._apply_thinking_default(payload, path, list(served))
    ctk = payload.get("chat_template_kwargs") or {}
    return changed, ctk.get("enable_thinking", "UNSET"), payload.get("reasoning_effort", "UNSET")


def apply_cache_safety(payload, path="chat/completions", served=("qwen3.8-flash-next",)):
    return server._apply_prompt_cache_safety(payload, path, list(served))


def apply_output_budget(payload, path="chat/completions", served=("qwen3.8-flash-next",)):
    return server._apply_output_token_default(payload, path, list(served))


class ThinkingDefaultTests(unittest.TestCase):
    def test_glm53_defaults_to_thinking_off(self):
        p = {"model": "local", "messages": [{"role": "user", "content": "hello"}]}
        changed, value, _ = apply(p, served=("GLM-5.3-Flash-EXL3-2.05",))
        self.assertTrue(changed)
        self.assertIs(value, False)

    def test_glm53_honors_hermes_desktop_reasoning_levels(self):
        for level, native in (("low", "low"), ("medium", "high"), ("high", "max")):
            with self.subTest(level=level):
                p = {"model": "local", "messages": [],
                     "reasoning": {"enabled": True, "effort": level}}
                changed, value, _ = apply(p, served=("GLM-5.3-Flash-EXL3-2.05",))
                self.assertTrue(changed)
                self.assertIs(value, True)
                self.assertEqual(p["chat_template_kwargs"]["reasoning_effort"], native)

    def test_glm53_top_level_effort_and_explicit_off(self):
        p = {"model": "local", "messages": [], "reasoning_effort": "high"}
        changed, value, _ = apply(p, served=("GLM-5.3-Flash-EXL3-2.05",))
        self.assertTrue(changed)
        self.assertIs(value, True)
        self.assertEqual(p["chat_template_kwargs"]["reasoning_effort"], "max")

        off = {"model": "local", "messages": [],
               "reasoning": {"enabled": False, "effort": "high"}}
        changed, value, _ = apply(off, served=("GLM-5.3-Flash-EXL3-2.05",))
        self.assertTrue(changed)
        self.assertIs(value, False)
        self.assertNotIn("reasoning_effort", off["chat_template_kwargs"])

    def test_glm53_native_effort_is_not_overwritten(self):
        p = {"model": "local", "messages": [],
             "reasoning": {"enabled": True, "effort": "medium"},
             "chat_template_kwargs": {"reasoning_effort": "low"}}
        changed, value, _ = apply(p, served=("GLM-5.3-Flash-EXL3-2.05",))
        self.assertTrue(changed)
        self.assertIs(value, True)
        self.assertEqual(p["chat_template_kwargs"]["reasoning_effort"], "low")

        native_only = {"model": "local", "messages": [],
                       "chat_template_kwargs": {"reasoning_effort": "low"}}
        changed, value, _ = apply(native_only, served=("GLM-5.3-Flash-EXL3-2.05",))
        self.assertTrue(changed)
        self.assertIs(value, True)
        self.assertEqual(native_only["chat_template_kwargs"]["reasoning_effort"], "low")

    def test_glm53_explicit_thinking_is_preserved(self):
        for kwargs in ({"enable_thinking": True}, {"thinking": True}):
            with self.subTest(kwargs=kwargs):
                p = {"model": "GLM-5.3-Flash-EXL3-2.05", "messages": [],
                     "chat_template_kwargs": kwargs.copy()}
                changed, _, _ = apply(p, served=("GLM-5.3-Flash-EXL3-2.05",))
                self.assertFalse(changed)
                self.assertEqual(p["chat_template_kwargs"], kwargs)

    def test_glm53_inline_think_is_preserved(self):
        p = {"model": "local", "messages": [{"role": "user", "content": "solve this /think"}]}
        changed, value, _ = apply(p, served=("GLM-5.3-Flash-EXL3-2.05",))
        self.assertFalse(changed)
        self.assertEqual(value, "UNSET")

    def test_qwen3_defaults_to_medium_thinking(self):
        p = {"model": "qwen3.8-27b-nvfp4", "messages": [{"role": "user", "content": "hi"}]}
        changed, val, effort = apply(p)
        self.assertTrue(changed)
        self.assertIs(val, True)
        self.assertEqual(effort, "medium")

    def test_flash_next_defaults_to_thinking_off_for_agent_loops(self):
        p = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "use the tools"}]}
        changed, val, effort = apply(p, served=("qwen3.8-flash-next",))
        self.assertTrue(changed)
        self.assertIs(val, False)
        self.assertEqual(effort, "UNSET")

    def test_flash_next_generic_medium_default_stays_thinking_off(self):
        p = {
            "model": "qwen3.8-flash-next",
            "reasoning_effort": "medium",
            "reasoning": {"enabled": True, "effort": "medium"},
            "messages": [{"role": "user", "content": "fix the bug"}],
        }
        changed, val, effort = apply(p, served=("qwen3.8-flash-next",))
        self.assertTrue(changed)
        self.assertIs(val, False)
        self.assertEqual(effort, "UNSET")
        self.assertNotIn("reasoning", p)

    def test_flash_next_inline_think_preserves_reasoning_controls(self):
        p = {
            "model": "qwen3.8-flash-next",
            "reasoning": {"enabled": True, "effort": "high"},
            "messages": [{"role": "user", "content": "solve this /think"}],
        }
        changed, val, _ = apply(p, served=("qwen3.8-flash-next",))
        self.assertFalse(changed)
        self.assertEqual(val, "UNSET")
        self.assertEqual(p["reasoning"]["effort"], "high")

    def test_flash_next_explicit_thinking_true_is_respected(self):
        p = {
            "model": "qwen3.8-flash-next",
            "messages": [{"role": "user", "content": "hard problem"}],
            "chat_template_kwargs": {"enable_thinking": True},
        }
        changed, val, effort = apply(p, served=("qwen3.8-flash-next",))
        self.assertFalse(changed)
        self.assertIs(val, True)
        self.assertEqual(effort, "UNSET")

    def test_detects_qwen3_via_served_when_model_is_local(self):
        # Hermes often sends model="local"; detection must fall back to served ids.
        p = {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        changed, val, effort = apply(p, served=("qwen3.8-27b-nvfp4",))
        self.assertTrue(changed)
        self.assertIs(val, True)
        self.assertEqual(effort, "medium")

    def test_non_qwen_model_untouched(self):
        p = {"model": "glm-4.7-flash", "messages": [{"role": "user", "content": "hi"}]}
        changed, val, effort = apply(p, served=("glm-4.7-flash",))
        self.assertFalse(changed)
        self.assertEqual(val, "UNSET")
        self.assertEqual(effort, "UNSET")

    def test_explicit_enable_thinking_true_respected(self):
        p = {
            "model": "qwen3.8-27b-nvfp4",
            "messages": [{"role": "user", "content": "hard problem"}],
            "chat_template_kwargs": {"enable_thinking": True},
        }
        changed, val, effort = apply(p)
        self.assertFalse(changed)
        self.assertIs(val, True)  # caller's choice preserved
        self.assertEqual(effort, "UNSET")

    def test_explicit_enable_thinking_false_respected(self):
        p = {
            "model": "qwen3.8-27b-nvfp4",
            "messages": [{"role": "user", "content": "x"}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        changed, val, effort = apply(p)
        self.assertFalse(changed)  # caller's choice preserved
        self.assertIs(val, False)
        self.assertEqual(effort, "UNSET")

    def test_inline_think_directive_respected(self):
        p = {"model": "qwen3.8-27b-nvfp4", "messages": [{"role": "user", "content": "solve this /think"}]}
        changed, val, effort = apply(p)
        self.assertFalse(changed)
        self.assertEqual(val, "UNSET")
        self.assertEqual(effort, "UNSET")

    def test_inline_no_think_directive_respected(self):
        p = {"model": "qwen3.8-27b-nvfp4", "messages": [{"role": "user", "content": "quick /no_think"}]}
        changed, val, effort = apply(p)
        self.assertFalse(changed)
        self.assertEqual(val, "UNSET")
        self.assertEqual(effort, "UNSET")

    def test_preserves_other_chat_template_kwargs(self):
        p = {
            "model": "qwen3.8-27b-nvfp4",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"add_generation_prompt": True},
        }
        changed, _, effort = apply(p)
        self.assertTrue(changed)
        self.assertIs(p["chat_template_kwargs"]["enable_thinking"], True)
        self.assertIs(p["chat_template_kwargs"]["add_generation_prompt"], True)
        self.assertEqual(effort, "medium")

    def test_only_chat_completions_path(self):
        p = {"model": "qwen3.8-27b-nvfp4", "prompt": "hi"}
        changed, val, effort = apply(p, path="completions")
        self.assertFalse(changed)
        self.assertEqual(val, "UNSET")
        self.assertEqual(effort, "UNSET")

    def test_multimodal_content_list_does_not_crash(self):
        p = {
            "model": "qwen3.8-27b-nvfp4",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
        changed, val, effort = apply(p)
        self.assertTrue(changed)
        self.assertIs(val, True)
        self.assertEqual(effort, "medium")


class PromptCacheSafetyTests(unittest.TestCase):
    def test_disables_cache_for_flash_next(self):
        payload = {"model": "qwen3.8-flash-next", "messages": []}
        self.assertTrue(apply_cache_safety(payload))
        self.assertIs(payload["cache_prompt"], False)

    def test_detects_flash_next_via_served_model(self):
        payload = {"model": "local", "messages": []}
        self.assertTrue(apply_cache_safety(payload))
        self.assertIs(payload["cache_prompt"], False)

    def test_other_models_and_paths_are_untouched(self):
        other = {"model": "deepseek-v4-flash-0731", "messages": []}
        self.assertFalse(apply_cache_safety(other, served=("deepseek-v4-flash-0731",)))
        self.assertNotIn("cache_prompt", other)

        embeddings = {"model": "qwen3.8-flash-next", "input": "hello"}
        self.assertFalse(apply_cache_safety(embeddings, path="embeddings"))
        self.assertNotIn("cache_prompt", embeddings)

    def test_already_disabled_is_idempotent(self):
        payload = {"model": "qwen3.8-flash-next", "cache_prompt": False}
        self.assertFalse(apply_cache_safety(payload))
        self.assertIs(payload["cache_prompt"], False)


class OutputTokenDefaultTests(unittest.TestCase):
    def test_flash_next_gets_large_agent_budget_when_omitted(self):
        payload = {"model": "qwen3.8-flash-next", "messages": []}
        self.assertTrue(apply_output_budget(payload))
        self.assertEqual(payload["max_tokens"], 32768)

    def test_detects_flash_next_via_served_model(self):
        payload = {"model": "local", "messages": []}
        self.assertTrue(apply_output_budget(payload))
        self.assertEqual(payload["max_tokens"], 32768)

    def test_explicit_max_tokens_is_respected(self):
        payload = {"model": "qwen3.8-flash-next", "messages": [], "max_tokens": 8192}
        self.assertFalse(apply_output_budget(payload))
        self.assertEqual(payload["max_tokens"], 8192)

    def test_explicit_max_completion_tokens_is_respected(self):
        payload = {
            "model": "qwen3.8-flash-next",
            "messages": [],
            "max_completion_tokens": 12000,
        }
        self.assertFalse(apply_output_budget(payload))
        self.assertNotIn("max_tokens", payload)

    def test_non_flash_models_and_non_generation_paths_untouched(self):
        other = {"model": "deepseek-v4-flash-0731", "messages": []}
        self.assertFalse(apply_output_budget(other, served=("deepseek-v4-flash-0731",)))
        self.assertNotIn("max_tokens", other)

        embeddings = {"model": "qwen3.8-flash-next", "input": "hello"}
        self.assertFalse(apply_output_budget(embeddings, path="embeddings"))
        self.assertNotIn("max_tokens", embeddings)


if __name__ == "__main__":
    unittest.main()
