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


class ThinkingDefaultTests(unittest.TestCase):
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

    def test_flash_next_medium_effort_enables_thinking(self):
        p = {
            "model": "qwen3.8-flash-next",
            "reasoning_effort": "medium",
            "messages": [{"role": "user", "content": "fix the bug"}],
        }
        changed, val, effort = apply(p, served=("qwen3.8-flash-next",))
        self.assertTrue(changed)
        self.assertIs(val, True)
        self.assertEqual(effort, "UNSET")

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


if __name__ == "__main__":
    unittest.main()
