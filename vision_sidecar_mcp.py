"""MCP bridge from Hermes to the isolated Qwen3-VL screen sidecar."""

from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from PIL import Image, UnidentifiedImageError

MAX_INPUT_BYTES = 25 * 1024 * 1024
MAX_INPUT_PIXELS = 40_000_000
DEFAULT_ENDPOINT = "http://100.66.100.22:8081/v1"
DEFAULT_MODEL = "qwen3-vl-8b-instruct"
_ALLOWED_ROLES = {"button", "link", "input", "price", "other"}

_SYSTEM_PROMPT = """You are a read-only screen and product-image describer. Return only one JSON object with exactly these keys:
{"summary":"...","ocr":["..."],"ui":[{"label":"...","role":"button|link|input|price|other","where":"..."}],"price":null,"cta":null,"captcha":false,"login_wall":false,"unsafe_to_purchase":true}
Report only visible evidence. Never claim that a click, checkout, order, payment, or purchase happened. Do not decide what action to take. Keep the response concise. unsafe_to_purchase must always be true."""


def _read_image_bytes(image: str) -> bytes:
    if image.startswith("data:"):
        header, separator, encoded = image.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("image data URI must be base64 encoded")
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("image data URI contains invalid base64") from exc
    elif image.startswith(("http://", "https://")):
        request = urllib.request.Request(image, headers={"User-Agent": "Hermes-Vision-Sidecar/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(MAX_INPUT_BYTES + 1)
    else:
        path = Path(image).expanduser()
        if not path.is_file():
            raise ValueError(f"image path does not exist: {path}")
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("image exceeds the 25 MiB input limit")
        data = path.read_bytes()

    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("image exceeds the 25 MiB input limit")
    return data


def prepare_image(image: str, max_edge: int = 1024) -> str:
    """Load and resize an image, returning a PNG data URI."""
    if not 256 <= max_edge <= 1024:
        raise ValueError("max_edge must be between 256 and 1024")
    try:
        with Image.open(io.BytesIO(_read_image_bytes(image))) as opened:
            opened.load()
            if opened.width * opened.height > MAX_INPUT_PIXELS:
                raise ValueError("image exceeds the 40 megapixel input limit")
            rendered = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("input is not a valid image") from exc

    rendered.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    rendered.save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalize_result(content: str) -> dict[str, Any]:
    """Normalize model output to the fixed read-only tool contract."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("vision sidecar did not return a JSON object")
        try:
            raw = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("vision sidecar returned invalid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("vision sidecar JSON must be an object")

    ocr_value = raw.get("ocr")
    ocr = ocr_value if isinstance(ocr_value, list) else []
    ui_value = raw.get("ui")
    ui_items = ui_value if isinstance(ui_value, list) else []
    ui: list[dict[str, str]] = []
    for item in ui_items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "other").lower()
        ui.append(
            {
                "label": str(item.get("label") or ""),
                "role": role if role in _ALLOWED_ROLES else "other",
                "where": str(item.get("where") or ""),
            }
        )

    def optional_text(value: Any) -> str | None:
        return None if value is None or value == "" else str(value)

    return {
        "summary": str(raw.get("summary") or ""),
        "ocr": [str(item) for item in ocr],
        "ui": ui,
        "price": optional_text(raw.get("price")),
        "cta": optional_text(raw.get("cta")),
        "captcha": bool(raw.get("captcha", False)),
        "login_wall": bool(raw.get("login_wall", False)),
        "unsafe_to_purchase": True,
    }


def describe_image(
    endpoint: str,
    model: str,
    image: str,
    question: str | None = None,
    max_edge: int = 1024,
) -> dict[str, Any]:
    """Send one resized image to the isolated OpenAI-compatible sidecar."""
    data_uri = prepare_image(image, max_edge=max_edge)
    prompt = question or "Describe this screen using the required JSON contract."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.load(response)
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("vision sidecar returned an unexpected response") from exc
    return normalize_result(content)


def create_server(endpoint: str, model: str) -> FastMCP:
    mcp = FastMCP(
        "vision-sidecar",
        instructions=(
            "Read-only image description through the 3060pc vision sidecar. "
            "Use for attached images and computer-use screenshots, then reason over "
            "the returned JSON with the primary model. Never delegate clicks or purchases."
        ),
        log_level="ERROR",
    )

    @mcp.tool(
        name="describe_screen",
        description=(
            "Describe a screenshot or product image with the isolated 3060pc VLM. "
            "Accepts a local path, base64 data URI, or HTTP URL; resizes to 1024px. "
            "Returns read-only OCR/UI JSON and never authorizes purchases."
        ),
        structured_output=True,
    )
    def describe_screen(
        image: str,
        question: str | None = None,
        max_edge: int = 1024,
    ) -> dict[str, Any]:
        return describe_image(endpoint, model, image, question, max_edge)

    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes 3060pc vision-sidecar MCP bridge")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    create_server(args.endpoint, args.model).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
