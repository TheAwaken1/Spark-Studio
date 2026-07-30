"""Command-line client for Spark Studio and its Hermes Agent Lab."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

import agentlab
import studio_search

DEFAULT_STUDIO_URL = os.environ.get("SPARK_STUDIO_URL", "http://127.0.0.1:7860")


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _client(args: argparse.Namespace, timeout: float = 30) -> httpx.Client:
    return httpx.Client(base_url=args.studio_url.rstrip("/"), timeout=timeout)


def _endpoint(args: argparse.Namespace) -> dict[str, Any]:
    return agentlab.discover_endpoint(args.studio_url, args.base_url, args.model)


def _request_error(exc: Exception) -> RuntimeError:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = exc.response.text
        return RuntimeError(f"Spark Studio returned {exc.response.status_code}: {detail}")
    return RuntimeError(str(exc))


def cmd_status(args: argparse.Namespace) -> int:
    try:
        with _client(args) as client:
            system_response = client.get("/api/system")
            system_response.raise_for_status()
            active_response = client.get("/api/active")
            active_response.raise_for_status()
            payload = {"system": system_response.json(), "active": active_response.json()}
    except Exception as exc:  # noqa: BLE001
        raise _request_error(exc) from exc
    if args.json:
        _json(payload)
        return 0
    system = payload["system"]
    active = payload["active"]
    print(f"Spark Studio {system.get('version', '?')} · {args.studio_url}")
    if active:
        state = "ready" if active.get("ready") else active.get("status", "loading")
        print(f"Active: {active.get('label') or active.get('engine')} · {state}")
        print(f"Endpoint: {active.get('url') or 'not available yet'}")
    else:
        print("Active: no model loaded")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    try:
        with _client(args, timeout=120) as client:
            response = client.get("/api/doctor")
            response.raise_for_status()
            report = response.json()
    except Exception as exc:  # noqa: BLE001
        raise _request_error(exc) from exc
    if args.json:
        _json(report)
    else:
        print(f"Spark Studio Doctor · v{report.get('version', '?')}")
        for check in report.get("checks") or []:
            icon = {"ok": "OK", "warn": "WARN", "error": "ERROR"}.get(check.get("status"), "-")
            print(f"{icon:5} {check.get('label')}: {check.get('detail')}")
            if check.get("fix") and check.get("status") != "ok":
                print(f"      fix: {check['fix']}")
    return 1 if (report.get("summary") or {}).get("error") else 0


def cmd_models(args: argparse.Namespace) -> int:
    endpoint = _endpoint(args)
    if args.json:
        _json(endpoint)
    else:
        print(f"Endpoint: {endpoint['base_url']}")
        for model in endpoint["models"]:
            marker = "*" if model == endpoint["model"] else " "
            print(f"{marker} {model}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    endpoint = _endpoint(args)
    prompt = args.prompt
    if prompt == "-":
        prompt = sys.stdin.read()
    payload = {
        "model": endpoint["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    with httpx.Client(timeout=args.timeout) as client:
        response = client.post(f"{endpoint['base_url']}/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()
    if args.json:
        _json(result)
    else:
        message = ((result.get("choices") or [{}])[0].get("message") or {})
        print(message.get("content") or "")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    try:
        payload = studio_search.search(
            args.studio_url,
            args.query,
            limit=args.limit,
            enrich=args.enrich,
            timeout=args.timeout,
        )
    except Exception as exc:  # noqa: BLE001
        raise _request_error(exc) from exc
    if args.json:
        _json(payload)
    else:
        print(studio_search.format_results(payload))
    return 0


def cmd_bench_speed(args: argparse.Namespace) -> int:
    payload = {"model": args.model or "local", "max_tokens": args.max_tokens, "runs": args.runs}
    if args.base_url:
        payload["url"] = args.base_url.rstrip("/").removesuffix("/v1")
    with _client(args, timeout=max(args.timeout, 600)) as client:
        response = client.post("/api/bench", json=payload)
        response.raise_for_status()
        result = response.json()
    if args.json:
        _json(result)
    else:
        tokens_per_sec = result.get("tokens_per_sec")
        ttft_ms = result.get("ttft_ms")
        print(f"Model: {result.get('resolved_model', args.model or 'local')}")
        if tokens_per_sec is None or ttft_ms is None:
            print("Benchmark failed: the endpoint returned no generated tokens", file=sys.stderr)
            for error in result.get("errors") or []:
                print(f"  {error}", file=sys.stderr)
            return 1
        print(f"Generation: {tokens_per_sec:.2f} tokens/s")
        print(f"TTFT: {ttft_ms:.1f} ms")
        if result.get("engine_version"):
            print(f"Engine: {result['engine_version']}")
    return 0


def cmd_bench_tools(args: argparse.Namespace) -> int:
    payload = {"base_url": args.base_url, "model": args.model}
    with _client(args, timeout=30) as client:
        response = client.post("/api/tooleval/run", json=payload)
        response.raise_for_status()
        state = response.json()
        while state.get("running"):
            if not args.json:
                print(f"Tool Eval: {state.get('done', 0)}/{state.get('total', '?')}", end="\r", flush=True)
            time.sleep(2)
            response = client.get("/api/tooleval/status")
            response.raise_for_status()
            state = response.json()
    if not args.json:
        print(" " * 50, end="\r")
    if args.json:
        _json(state)
    else:
        if state.get("error"):
            print(f"Tool Eval failed: {state['error']}")
            return 1
        print(f"Tool Eval: {state.get('score')} / 100")
        for category, score in (state.get("category_scores") or {}).items():
            print(f"  {category}: {score}")
        if state.get("report_path"):
            print(f"Report: {state['report_path']}")
    return 0


def cmd_agent_doctor(args: argparse.Namespace) -> int:
    endpoint = None
    endpoint_error = None
    try:
        endpoint = _endpoint(args)
    except Exception as exc:  # noqa: BLE001
        endpoint_error = str(exc)
    status = agentlab.hermes_status(endpoint)
    status["endpoint_error"] = endpoint_error
    context = (endpoint or {}).get("context_length")
    if context is not None:
        status["context_ok"] = int(context) >= 64000
    if args.json:
        _json(status)
    else:
        if status["installed"]:
            print(f"Hermes: OK · {status.get('version') or status.get('binary')}")
        else:
            print("Hermes: NOT INSTALLED")
            print(f"Install: {status['install_command']}")
        print(f"Isolated profile: {status['profile']}")
        if endpoint:
            print(f"Model: {endpoint['model']}")
            print(f"Endpoint: {endpoint['base_url']}")
            if context:
                suffix = "OK" if status.get("context_ok") else "LOW (Hermes recommends 64K+)"
                print(f"Context: {context:,} · {suffix}")
        else:
            print(f"Endpoint: ERROR · {endpoint_error}")
    return 0 if status["installed"] and endpoint else 1


def cmd_agent_chat(args: argparse.Namespace) -> int:
    if args.json:
        raise RuntimeError("--json cannot be used with interactive Hermes")
    endpoint = _endpoint(args)
    print(f"Hermes model: {endpoint['model']}")
    print(f"Hermes endpoint: {endpoint['base_url']}")
    print(f"Isolated profile: {agentlab.HERMES_HOME}")
    return agentlab.launch_hermes(
        endpoint,
        Path(args.repo),
        max_turns=args.max_turns,
        unsafe_yolo=args.unsafe_yolo,
    )


def cmd_agent_auth(args: argparse.Namespace) -> int:
    """Run ``hermes auth`` against Spark Studio's isolated profile.

    Bare ``hermes auth add <provider>`` writes to the personal ``~/.hermes``,
    which the dashboard deliberately never reads. This wrapper targets the
    profile that dashboard Chat and ``sparkstudio hermes`` actually use, so
    ``/model`` can switch to the authenticated provider.
    """
    binary = agentlab.find_hermes()
    if not binary:
        raise RuntimeError(f"Hermes Agent is not installed. Run: {agentlab.HERMES_INSTALL}")
    agentlab.HERMES_HOME.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HERMES_HOME"] = str(agentlab.HERMES_HOME)
    print(f"Isolated profile: {agentlab.HERMES_HOME}", flush=True)
    completed = subprocess.run([binary, "auth", *args.auth_args], env=env, check=False)
    if completed.returncode == 0 and args.auth_args[:1] == ["add"]:
        print("Credential saved. Restart Hermes Chat (or /model in a new session) to use it.")
    return completed.returncode


def _print_agent_run(result: dict[str, Any]) -> None:
    print(f"Run: {result['id']} · {result['status']}")
    print(f"Model: {result['model']}")
    print(f"Workspace: {result.get('workspace')}")
    print(f"Isolation: {result.get('isolation')}")
    hermes_result = result.get("hermes") or {}
    if hermes_result.get("response"):
        print("\nHermes response:\n")
        print(hermes_result["response"])
    if result.get("git_status"):
        print("\nChanged files:")
        print(result["git_status"].rstrip())
    if result.get("error") or hermes_result.get("stderr"):
        print(f"\nError: {result.get('error') or hermes_result.get('stderr')}", file=sys.stderr)
    if result.get("report_path"):
        print(f"\nReport: {result['report_path']}")


def cmd_agent_run(args: argparse.Namespace) -> int:
    endpoint = _endpoint(args)
    if not args.json:
        mode = "in place" if args.in_place else "in an isolated workspace"
        print(f"Running Hermes with {endpoint['model']} {mode}…")
    result = agentlab.run_agent(
        endpoint,
        Path(args.repo),
        args.task,
        in_place=args.in_place,
        max_turns=args.max_turns,
        timeout=args.timeout,
        unsafe_yolo=args.unsafe_yolo,
    )
    if args.json:
        _json(result)
    else:
        _print_agent_run(result)
    return 0 if result["status"] == "completed" else 1


def cmd_agent_cases(args: argparse.Namespace) -> int:
    cases = [
        {"suite": suite, "id": case["id"], "title": case["title"], "task": case["task"]}
        for suite, members in agentlab.SUITES.items()
        for case in members
    ]
    if args.json:
        _json(cases)
    else:
        for case in cases:
            print(f"{case['suite']}/{case['id']}: {case['title']}")
    return 0


def cmd_agent_eval(args: argparse.Namespace) -> int:
    endpoint = _endpoint(args)
    if not args.json:
        count = len(args.case or agentlab.SUITES[args.suite]) * args.trials
        print(f"Evaluating {endpoint['model']} with Hermes · {count} run(s), jobs={args.jobs}")
    result = agentlab.evaluate(
        endpoint,
        suite=args.suite,
        case_ids=args.case,
        trials=args.trials,
        jobs=args.jobs,
        max_turns=args.max_turns,
        timeout=args.timeout,
        unsafe_yolo=args.unsafe_yolo,
        progress=None if args.json else print,
    )
    if args.json:
        _json(result)
    else:
        print(f"\nAgent score: {result['score']} / 100 ({result['passed']}/{result['total']} passed)")
        for case in result["cases"]:
            print(f"  {'PASS' if case.get('passed') else 'FAIL'} {case.get('case')} trial {case.get('trial')}")
        print(f"Report: {result['report_path']}")
        print(f"Workspaces: {result['workspace']}")
    return 1 if args.fail_below is not None and result["score"] < args.fail_below else 0


def cmd_agent_history(args: argparse.Namespace) -> int:
    rows = agentlab.history(args.limit)
    if args.json:
        _json(rows)
    else:
        if not rows:
            print("No Agent Lab runs yet.")
        for row in rows:
            score = "—" if row.get("score") is None else f"{row['score']:.0f}"
            print(f"{row['id']}  {row['mode']:4}  {row['status']:9}  score={score:>3}  {row.get('model') or '?'}")
    return 0


def cmd_agent_show(args: argparse.Namespace) -> int:
    row = agentlab.get_result(args.run_id)
    if not row:
        raise RuntimeError(f"Agent Lab run not found: {args.run_id}")
    if args.json:
        _json(row)
    else:
        result = row.get("result") or {}
        if result.get("mode") == "run":
            _print_agent_run(result)
        else:
            print(f"Run: {row['id']} · {row['status']}")
            print(f"Model: {row.get('model')}")
            print(f"Score: {row.get('score')} / 100 ({row.get('passed')}/{row.get('total')})")
            if result.get("report_path"):
                print(f"Report: {result['report_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparkstudio", description="DGX Spark model CLI and Hermes Agent Lab")
    parser.add_argument("--studio-url", default=DEFAULT_STUDIO_URL, help="Spark Studio URL")
    parser.add_argument("--base-url", help="override inference endpoint (with or without /v1)")
    parser.add_argument("--model", help="override the served model id")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="show Spark Studio and active model status")
    status.set_defaults(func=cmd_status)
    doctor = commands.add_parser("doctor", help="run the Spark Studio system doctor")
    doctor.set_defaults(func=cmd_doctor)
    models = commands.add_parser("models", help="list models reported by the active endpoint")
    models.set_defaults(func=cmd_models)

    search = commands.add_parser("search", help="search via Spark Studio's SearXNG/DDG pipeline")
    search.add_argument("query", help="web search query")
    search.add_argument("--limit", type=int, default=5, help="number of results (1-10)")
    search.add_argument("--enrich", action="store_true", help="fetch readable text from top result pages")
    search.add_argument("--timeout", type=float, default=60)
    search.set_defaults(func=cmd_search)

    hermes = commands.add_parser("hermes", help="launch Hermes with the active Spark Studio model")
    hermes.add_argument("--repo", default=".", help="workspace path (default: current directory)")
    hermes.add_argument("--max-turns", type=int, default=90)
    hermes.add_argument("--unsafe-yolo", action="store_true", help="disable Hermes command approvals (not recommended)")
    hermes.set_defaults(func=cmd_agent_chat)

    chat = commands.add_parser("chat", help="send one prompt directly to the loaded model")
    chat.add_argument("prompt", help="prompt text, or - to read stdin")
    chat.add_argument("--max-tokens", type=int, default=1024)
    chat.add_argument("--temperature", type=float, default=0.2)
    chat.add_argument("--timeout", type=float, default=600)
    chat.set_defaults(func=cmd_chat)

    bench = commands.add_parser("bench", help="run model benchmarks")
    bench_commands = bench.add_subparsers(dest="bench_command", required=True)
    speed = bench_commands.add_parser("speed", help="quick tokens/s and TTFT benchmark")
    speed.add_argument("--max-tokens", type=int, default=256)
    speed.add_argument("--runs", type=int, default=3)
    speed.add_argument("--timeout", type=float, default=600)
    speed.set_defaults(func=cmd_bench_speed)
    tools = bench_commands.add_parser("tools", help="built-in deterministic tool-use benchmark")
    tools.set_defaults(func=cmd_bench_tools)

    agent = commands.add_parser("agent", help="Hermes Agent Lab")
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    agent_doctor = agent_commands.add_parser("doctor", help="check Hermes and the active model")
    agent_doctor.set_defaults(func=cmd_agent_doctor)
    agent_chat = agent_commands.add_parser("chat", help="launch interactive Hermes with the active model")
    agent_chat.add_argument("--repo", default=".", help="workspace path (default: current directory)")
    agent_chat.add_argument("--max-turns", type=int, default=90)
    agent_chat.add_argument("--unsafe-yolo", action="store_true", help="disable Hermes command approvals (not recommended)")
    agent_chat.set_defaults(func=cmd_agent_chat)
    cases = agent_commands.add_parser("cases", help="list deterministic evaluation cases")
    cases.set_defaults(func=cmd_agent_cases)

    auth = agent_commands.add_parser(
        "auth",
        help="manage provider credentials for the dashboard's isolated Hermes profile",
    )
    auth.add_argument(
        "auth_args",
        nargs=argparse.REMAINDER,
        metavar="args",
        help="passed through to `hermes auth` (e.g. `add openai-codex`, `list`)",
    )
    auth.set_defaults(func=cmd_agent_auth)

    run = agent_commands.add_parser("run", help="run Hermes on a repository task")
    run.add_argument("task")
    run.add_argument("--repo", default=".", help="repository path (default: current directory)")
    run.add_argument("--in-place", action="store_true", help="modify the repository directly instead of an isolated worktree")
    run.add_argument("--max-turns", type=int, default=90)
    run.add_argument("--timeout", type=float, default=1800)
    run.add_argument("--unsafe-yolo", action="store_true", help="disable Hermes command approvals (not recommended)")
    run.set_defaults(func=cmd_agent_run)

    evaluate = agent_commands.add_parser("eval", help="run deterministic coding tasks through Hermes")
    evaluate.add_argument("--suite", default="coding-smoke", choices=sorted(agentlab.SUITES))
    evaluate.add_argument("--case", action="append", help="case id; repeat to select several")
    evaluate.add_argument("--trials", type=int, default=1)
    evaluate.add_argument("--jobs", type=int, default=1, help="parallel Hermes agents (1-8)")
    evaluate.add_argument("--max-turns", type=int, default=90)
    evaluate.add_argument("--timeout", type=float, default=1800)
    evaluate.add_argument("--fail-below", type=float, help="exit non-zero when score is below this value")
    evaluate.add_argument("--unsafe-yolo", action="store_true", help="disable Hermes command approvals (not recommended)")
    evaluate.set_defaults(func=cmd_agent_eval)

    history = agent_commands.add_parser("history", help="list saved Agent Lab runs")
    history.add_argument("--limit", type=int, default=20)
    history.set_defaults(func=cmd_agent_history)
    show = agent_commands.add_parser("show", help="show one saved Agent Lab run")
    show.add_argument("run_id")
    show.set_defaults(func=cmd_agent_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "json", False):
            _json({"error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
