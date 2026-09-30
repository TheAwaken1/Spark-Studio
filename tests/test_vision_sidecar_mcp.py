from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import vision_sidecar_mcp


class VisionSidecarMcpTests(unittest.TestCase):
    def test_prepare_image_resizes_long_edge_and_returns_png_data_uri(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            Image.new("RGB", (2400, 1200)).save(path)

            uri = vision_sidecar_mcp.prepare_image(str(path), max_edge=1024)

        prefix, encoded = uri.split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as resized:
            self.assertEqual(resized.size, (1024, 512))

    def test_normalize_result_enforces_contract_and_purchase_guard(self):
        raw = json.dumps(
            {
                "summary": "Checkout page",
                "ocr": ["Buy now", 42],
                "ui": [
                    {"label": "Buy now", "role": "button", "where": "bottom right"},
                    {"label": "Mystery", "role": "invalid", "where": "top"},
                ],
                "price": "$19.99",
                "cta": "Buy now",
                "captcha": 0,
                "login_wall": 1,
                "unsafe_to_purchase": False,
                "claimed_purchase": True,
            }
        )

        result = vision_sidecar_mcp.normalize_result(raw)

        self.assertEqual(
            set(result),
            {"summary", "ocr", "ui", "price", "cta", "captcha", "login_wall", "unsafe_to_purchase"},
        )
        self.assertEqual(result["ocr"], ["Buy now", "42"])
        self.assertEqual(result["ui"][1]["role"], "other")
        self.assertFalse(result["captcha"])
        self.assertTrue(result["login_wall"])
        self.assertTrue(result["unsafe_to_purchase"])

    def test_normalize_result_accepts_fenced_json(self):
        result = vision_sidecar_mcp.normalize_result(
            "```json\n{\"summary\":\"Settings\",\"ocr\":[],\"ui\":[],"
            "\"price\":null,\"cta\":null,\"captcha\":false,\"login_wall\":false}\n```"
        )

        self.assertEqual(result["summary"], "Settings")
        self.assertIsNone(result["price"])
        self.assertTrue(result["unsafe_to_purchase"])

    def test_prepare_image_rejects_non_image_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not-image.txt"
            path.write_text("not an image")
            with self.assertRaisesRegex(ValueError, "valid image"):
                vision_sidecar_mcp.prepare_image(str(path))


if __name__ == "__main__":
    unittest.main()
