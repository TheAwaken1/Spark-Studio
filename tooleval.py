"""Tool Eval Bench — how *useful* is the served model, beyond raw speed?

Runs a built-in suite of deterministic tool-calling and structured-output
cases against any OpenAI-compatible endpoint and scores them pass/fail:

  selection   pick the right tool among several
  arguments   extract correct argument values from the request
  restraint   answer directly when no tool is needed (no spurious calls)
  multi_turn  actually use a tool result in the final answer
  json_output emit strict JSON matching a requested shape

Scores are 0-100 (percent of cases passed), overall and per category.
Mirrors the sparkrun_service update pattern: start_eval() kicks off a
background thread, eval_status() reports progress, results persist to db.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx

import db

# One human-readable report (.md) + raw data (.json) per eval run.
RESULTS_DIR = Path(__file__).parent / "tooleval-results"

# ---- tool definitions shared by the cases ----------------------------------

_TOOLS = {
    "get_weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    "search_web": {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for up-to-date information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "calculate": {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression exactly",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    "book_meeting": {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": "Book a meeting in the calendar",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                    "time": {"type": "string", "description": "24h time HH:MM"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["date", "time", "attendees"],
            },
        },
    },
    "convert_currency": {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount between currencies",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
}

_ALL_TOOLS = list(_TOOLS.values())


def _norm_args(call: dict | None) -> dict:
    """Parsed arguments of a tool call ({} when absent/malformed)."""
    if not call:
        return {}
    raw = (call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}") or {}
    except Exception:  # noqa: BLE001
        return {}


def _call_name(call: dict | None) -> str | None:
    return ((call or {}).get("function") or {}).get("name")


def _num_close(v: Any, want: float, tol: float = 0.01) -> bool:
    try:
        return abs(float(v) - want) <= tol
    except (TypeError, ValueError):
        return False


# ---- case definitions -------------------------------------------------------
# Each case: id, category, request-builder fields, and a check(msg) -> str|None
# returning None on pass or a human-readable failure detail.

def _expect_tool(name: str, arg_check=None):
    def check(msg: dict) -> str | None:
        calls = msg.get("tool_calls") or []
        if not calls:
            body = (msg.get("content") or "").strip()
            return f"no tool call (answered with text: {body[:80]!r})" if body else "no tool call in response"
        got = _call_name(calls[0])
        if got != name:
            return f"called {got!r}, expected {name!r}"
        if arg_check:
            return arg_check(_norm_args(calls[0]))
        return None
    return check


def _expect_no_tool(msg: dict) -> str | None:
    calls = msg.get("tool_calls") or []
    if calls:
        return f"spurious tool call to {_call_name(calls[0])!r}"
    if not (msg.get("content") or "").strip():
        return "empty response"
    return None


def _expect_json(check_obj):
    def check(msg: dict) -> str | None:
        text = (msg.get("content") or "").strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0:
            start, end = text.find("["), text.rfind("]")
        if start < 0:
            return f"no JSON found in response: {text[:80]!r}"
        try:
            obj = json.loads(text[start:end + 1])
        except Exception as e:  # noqa: BLE001
            return f"invalid JSON ({e})"
        return check_obj(obj)
    return check


CASES: list[dict[str, Any]] = [
    # --- selection: pick the right tool among all five -----------------------
    {
        "id": "select-weather", "category": "selection",
        "prompt": "What's the weather like in Paris right now?",
        "tools": _ALL_TOOLS,
        "check": _expect_tool("get_weather"),
    },
    {
        "id": "select-calc", "category": "selection",
        "prompt": "Compute 847 * 293 exactly. Use a tool to be sure.",
        "tools": _ALL_TOOLS,
        "check": _expect_tool("calculate"),
    },
    {
        "id": "select-search", "category": "selection",
        "prompt": "What are today's top technology news headlines?",
        "tools": _ALL_TOOLS,
        "check": _expect_tool("search_web"),
    },
    # --- arguments: extract exact values -------------------------------------
    {
        "id": "args-weather", "category": "arguments",
        "prompt": "What's the temperature in Tokyo, in celsius?",
        "tools": [_TOOLS["get_weather"]],
        "check": _expect_tool("get_weather", lambda a: (
            None if "tokyo" in str(a.get("city", "")).lower()
            and str(a.get("units", "celsius")).lower() == "celsius"
            else f"bad args: {a}"
        )),
    },
    {
        "id": "args-meeting", "category": "arguments",
        "prompt": "Book a meeting on 2026-03-14 at 15:30 with Alice and Bob.",
        "tools": [_TOOLS["book_meeting"]],
        "check": _expect_tool("book_meeting", lambda a: (
            None if str(a.get("date", "")).startswith("2026-03-14")
            and "15:30" in str(a.get("time", ""))
            and {"alice", "bob"} <= {str(x).lower() for x in (a.get("attendees") or [])}
            else f"bad args: {a}"
        )),
    },
    {
        "id": "args-currency", "category": "arguments",
        "prompt": "How much is 250 US dollars in euros?",
        "tools": [_TOOLS["convert_currency"]],
        "check": _expect_tool("convert_currency", lambda a: (
            None if _num_close(a.get("amount"), 250)
            and str(a.get("from_currency", "")).upper().startswith("USD")
            and str(a.get("to_currency", "")).upper().startswith("EUR")
            else f"bad args: {a}"
        )),
    },
    # --- restraint: tools offered but not needed ------------------------------
    {
        "id": "restraint-haiku", "category": "restraint",
        "prompt": "Write a haiku about the moon.",
        "tools": _ALL_TOOLS,
        "check": _expect_no_tool,
    },
    {
        "id": "restraint-definition", "category": "restraint",
        "prompt": "What does the acronym HTTP stand for?",
        "tools": _ALL_TOOLS,
        "check": _expect_no_tool,
    },
    # --- multi_turn: use the tool result in the final answer ------------------
    {
        "id": "turn-weather", "category": "multi_turn",
        "messages": [
            {"role": "user", "content": "What's the weather in Berlin?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Berlin"}'},
            }]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": '{"temp_c": 7, "condition": "light rain"}'},
        ],
        "tools": [_TOOLS["get_weather"]],
        "check": lambda msg: (
            None if "7" in (msg.get("content") or "") and "rain" in (msg.get("content") or "").lower()
            else f"answer ignores tool result: {(msg.get('content') or '')[:100]!r}"
        ),
    },
    {
        "id": "turn-currency", "category": "multi_turn",
        "messages": [
            {"role": "user", "content": "Convert 250 USD to EUR."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "convert_currency",
                             "arguments": '{"amount": 250, "from_currency": "USD", "to_currency": "EUR"}'},
            }]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": '{"converted": 231.50, "currency": "EUR"}'},
        ],
        "tools": [_TOOLS["convert_currency"]],
        "check": lambda msg: (
            None if re.search(r"231[.,]5", msg.get("content") or "")
            else f"answer ignores tool result: {(msg.get('content') or '')[:100]!r}"
        ),
    },
    # --- json_output: strict structured output (no tools) ---------------------
    {
        "id": "json-extract", "category": "json_output",
        "prompt": ('Extract the person into strict JSON with keys "name" (string), '
                   '"age" (number), "city" (string). Reply with ONLY the JSON object.\n'
                   "Text: Maria, 34, lives in Lisbon."),
        "check": _expect_json(lambda o: (
            None if isinstance(o, dict)
            and str(o.get("name", "")).lower().startswith("maria")
            and _num_close(o.get("age"), 34)
            and "lisbon" in str(o.get("city", "")).lower()
            else f"wrong fields: {o}"
        )),
    },
    {
        "id": "json-list", "category": "json_output",
        "prompt": ('List the chemical symbols for gold, silver and iron as a strict JSON '
                   'array of 3 strings, e.g. ["X","Y","Z"]. Reply with ONLY the JSON array.'),
        "check": _expect_json(lambda o: (
            None if isinstance(o, list)
            and {str(x).strip().lower() for x in o} == {"au", "ag", "fe"}
            else f"wrong list: {o}"
        )),
    },
]

CATEGORIES = ["selection", "arguments", "restraint", "multi_turn", "json_output"]

# ---- runner ------------------------------------------------------------------

_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "model": None,
    "base_url": None,
    "done": 0,
    "total": len(CASES),
    "cases": [],            # [{id, category, ok, detail}]
    "score": None,          # 0-100 overall, set when finished
    "category_scores": {},  # {category: 0-100}
    "tools_unsupported": False,
    "error": None,
    "started": None,
    "finished": None,
    "report_path": None,   # markdown report of the last finished eval
}


def _prompt_of(case: dict) -> str:
    if case.get("prompt"):
        return case["prompt"]
    msgs = case.get("messages") or []
    user = next((m.get("content") for m in msgs if m.get("role") == "user"), "")
    return f"{user} (+ simulated tool result)"


def _write_report(model: str, base: str, score: float, cat_scores: dict,
                  results: list[dict], unsupported: bool) -> str | None:
    """Write <timestamp>_<model>.md (+ .json) into tooleval-results/.
    Returns the markdown path, or None if writing failed."""
    try:
        RESULTS_DIR.mkdir(exist_ok=True)
        slug = re.sub(r"[^\w.-]+", "_", model).strip("_")[:60] or "model"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = RESULTS_DIR / f"{stamp}_{slug}.md"
        by_id = {c["id"]: c for c in CASES}
        lines = [
            f"# Tool Eval Bench — {model}",
            "",
            f"- **Endpoint:** {base}",
            f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Overall score:** {score}%",
        ]
        if unsupported:
            lines.append("- **Note:** the engine rejected the `tools` parameter — "
                         "tool calling is likely not enabled on this server.")
        lines += ["", "| Category | Score |", "|---|---|"]
        lines += [f"| {cat} | {v}% |" for cat, v in cat_scores.items()]
        lines += ["", "| Result | Case | Category | Prompt | Model did | Failure detail |",
                  "|---|---|---|---|---|---|"]
        esc = lambda s: str(s or "").replace("|", "\\|").replace("\n", " ")  # noqa: E731
        for r in results:
            lines.append(
                f"| {'✅ pass' if r['ok'] else '❌ fail'} | {r['id']} | {r['category']} "
                f"| {esc(_prompt_of(by_id.get(r['id'], {})))[:120]} "
                f"| {esc(r.get('observed'))} | {esc(r.get('detail'))} |"
            )
        path.write_text("\n".join(lines) + "\n")
        path.with_suffix(".json").write_text(json.dumps({
            "model": model, "base_url": base, "score": score,
            "category_scores": cat_scores, "tools_unsupported": unsupported,
            "cases": results,
        }, indent=1))
        return str(path)
    except Exception:  # noqa: BLE001
        return None


def eval_status() -> dict[str, Any]:
    with _lock:
        return json.loads(json.dumps(_state))  # deep copy, JSON-safe


def _observed(msg: dict) -> str:
    """One line describing what the model actually did — shown in the Detail
    column for passing cases and written into the report file."""
    calls = msg.get("tool_calls") or []
    if calls:
        fn = (calls[0].get("function") or {})
        args = fn.get("arguments")
        args_str = args if isinstance(args, str) else json.dumps(args or {})
        return f"called {fn.get('name')}({args_str[:160]})"
    content = (msg.get("content") or "").strip().replace("\n", " ")
    return f"answered: {content[:160]}" if content else "(empty response)"


def _max_tokens_for(client: httpx.Client, base: str) -> int:
    """Answer budget clamped to the engine's context window. Thinking models
    need thousands of tokens of headroom, but asking a 4096-context engine for
    max_tokens=4096 gets every request rejected with a context-length 400."""
    try:
        r = client.get(f"{base}/v1/models", timeout=10)
        data = (r.json().get("data") or [{}])[0]
        ctx = int(data.get("max_model_len") or 0)
    except Exception:  # noqa: BLE001
        ctx = 0
    if not ctx:
        return 4096
    # Prompts in this suite are tiny (<500 tokens incl. tool schemas).
    return max(256, min(4096, ctx - 1024))


def _run_case(client: httpx.Client, base: str, model: str, case: dict,
              max_tokens: int) -> tuple[bool, str | None, bool, str]:
    """Returns (ok, detail, engine_rejected_tools, observed)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": case.get("messages") or [{"role": "user", "content": case["prompt"]}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if case.get("tools"):
        body["tools"] = case["tools"]
    r = None
    for attempt in (1, 2):
        try:
            r = client.post(f"{base}/v1/chat/completions", json=body, timeout=300)
            break
        except httpx.TimeoutException:
            # A slow/hung engine is an infrastructure problem, not a model
            # capability — retry once before letting it cost a case.
            if attempt == 2:
                return False, "timed out twice (300s each) — engine too slow or hung, not a capability verdict", False, "(no response)"
        except Exception as e:  # noqa: BLE001
            return False, f"request failed: {e}", False, "(no response)"
    if r.status_code >= 400:
        if "context length" in r.text.lower() and body["max_tokens"] > 512:
            # Engine context is smaller than advertised — shrink and retry.
            body["max_tokens"] = 512
            try:
                r = client.post(f"{base}/v1/chat/completions", json=body, timeout=300)
            except Exception as e:  # noqa: BLE001
                return False, f"request failed: {e}", False, "(no response)"
    if r.status_code >= 400:
        rejected = bool(case.get("tools")) and r.status_code in (400, 422)
        return False, f"HTTP {r.status_code}: {r.text[:160]}", rejected, "(request rejected)"
    try:
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
    except Exception as e:  # noqa: BLE001
        return False, f"bad response JSON: {e}", False, "(unparseable response)"
    if isinstance(msg.get("content"), str):
        # Inline chain-of-thought must not leak into the checks: a <think>
        # block mentioning "rain" or containing braces would skew results.
        msg = {**msg, "content": re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", msg["content"]).strip()}
    observed = _observed(msg)
    if not (msg.get("content") or "").strip() and not msg.get("tool_calls"):
        # Empty visible answer: distinguish "spent the budget thinking" from
        # a genuinely empty reply, otherwise the detail is a useless ''.
        if (msg.get("reasoning_content") or "").strip() or choice.get("finish_reason") == "length":
            return False, "model spent its whole token budget thinking — no visible answer", False, observed
    detail = case["check"](msg)
    return detail is None, detail, False, observed


def start_eval(base_url: str, model: str, run_id: str | None = None,
               recipe_id: int | None = None) -> dict[str, Any]:
    """Kick off the suite in a background thread. Raises ValueError when an
    eval is already in flight."""
    base = base_url.rstrip("/")
    with _lock:
        if _state["running"]:
            raise ValueError("a tool eval is already running")
        _state.update(
            running=True, model=model, base_url=base, done=0, total=len(CASES),
            cases=[], score=None, category_scores={}, tools_unsupported=False,
            error=None, started=time.time(), finished=None, report_path=None,
        )

    def _worker() -> None:
        results: list[dict[str, Any]] = []
        tool_rejections = 0
        tool_cases = sum(1 for c in CASES if c.get("tools"))
        try:
            with httpx.Client() as client:
                max_tokens = _max_tokens_for(client, base)
                for case in CASES:
                    ok, detail, rejected, observed = _run_case(client, base, model, case, max_tokens)
                    tool_rejections += 1 if rejected else 0
                    results.append({
                        "id": case["id"], "category": case["category"],
                        "ok": ok, "detail": detail, "observed": observed,
                    })
                    with _lock:
                        _state["done"] = len(results)
                        _state["cases"] = list(results)
        except Exception as e:  # noqa: BLE001
            with _lock:
                _state.update(running=False, error=str(e), finished=time.time())
            return
        score = round(100 * sum(r["ok"] for r in results) / len(results), 1)
        cat_scores = {}
        for cat in CATEGORIES:
            rs = [r for r in results if r["category"] == cat]
            if rs:
                cat_scores[cat] = round(100 * sum(r["ok"] for r in rs) / len(rs), 1)
        unsupported = tool_cases > 0 and tool_rejections == tool_cases
        report_path = _write_report(model, base, score, cat_scores, results, unsupported)
        with _lock:
            _state.update(
                running=False, score=score, category_scores=cat_scores,
                tools_unsupported=unsupported, finished=time.time(),
                report_path=report_path,
            )
        try:
            db.tooleval_insert({
                "run_id": run_id,
                "recipe_id": recipe_id,
                "model": model,
                "base_url": base,
                "score": score,
                "results_json": json.dumps({
                    "cases": results,
                    "category_scores": cat_scores,
                    "tools_unsupported": unsupported,
                    "report_path": report_path,
                }),
            })
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_worker, name="tooleval", daemon=True).start()
    return eval_status()
