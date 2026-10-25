"""Guard rails for recipe changes: lint checks and pre-launch file snapshots.

Motivation (from a real incident): an agent asked to "make this recipe
faster" cut ``max_model_len`` from 262144 to 32768 in a bundled sparkrun
YAML. The engine then advertised a 32K window, which silently broke Hermes
Agent's 64K minimum — and because the edit was a plain file write, there was
nothing to roll back to. This module gives every recipe change two safety
nets:

- **Lint**: a small set of rules that flag configurations known to break
  other parts of the stack before they are saved or launched.
- **File snapshots**: bundled ``recipes/*.yaml`` files are copied into
  ``data/recipe_file_snapshots/<stem>/`` right before every launch, so any
  out-of-band edit (agent shell, manual vim) has a restorable "last known
  launched" copy. DB-backed recipes get the same protection inside db.py.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import yaml

APP_DIR = Path(__file__).parent
SNAPSHOT_DIR = APP_DIR / "data" / "recipe_file_snapshots"
BUNDLED_DIR = APP_DIR / "recipes"

_KEEP = 20

# Hermes Agent refuses endpoints below a 64K context window; the recipe knob
# that shrinks the advertised window most often is max_model_len.
HERMES_MIN_CONTEXT = 65536


# ----- lint -----------------------------------------------------------------

def _context_length(doc: dict[str, Any]) -> int | None:
    """Find the effective context-window knob wherever recipes keep it."""
    candidates = [
        (doc.get("defaults") or {}).get("max_model_len"),  # sparkrun YAML
        (doc.get("args") or {}).get("max_model_len"),      # DB recipe args
        (doc.get("args") or {}).get("max-model-len"),
        ((doc.get("args") or {}).get("_sparkrun") or {}).get("max_model_len"),
        doc.get("max_model_len"),
    ]
    for value in candidates:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def lint(doc: dict[str, Any] | None) -> list[dict[str, str]]:
    """Return findings for a parsed recipe (sparkrun YAML or DB recipe dict)."""
    findings: list[dict[str, str]] = []
    if not isinstance(doc, dict):
        return findings
    context = _context_length(doc)
    if context is not None:
        if context < 4096:
            findings.append({
                "level": "warn",
                "code": "tiny-context",
                "message": (
                    f"max_model_len is only {context:,} tokens — most chat and "
                    "agent workloads will truncate almost immediately."
                ),
            })
        elif context < HERMES_MIN_CONTEXT:
            findings.append({
                "level": "warn",
                "code": "hermes-context",
                "message": (
                    f"max_model_len {context:,} is below Hermes Agent's minimum of "
                    f"{HERMES_MIN_CONTEXT:,} — Hermes Chat and the Agent Lab will "
                    "refuse this endpoint. Use 65536 or higher (or remove the cap)."
                ),
            })
    port = (doc.get("defaults") or {}).get("port", doc.get("port"))
    if port is not None:
        try:
            port_num = int(port)
            if not (1024 <= port_num <= 65535):
                findings.append({
                    "level": "warn",
                    "code": "bad-port",
                    "message": f"port {port_num} is outside the usable range 1024–65535.",
                })
        except (TypeError, ValueError):
            findings.append({
                "level": "warn",
                "code": "bad-port",
                "message": f"port {port!r} is not a number.",
            })
    return findings


def lint_yaml(text: str) -> list[dict[str, str]]:
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [{
            "level": "error",
            "code": "invalid-yaml",
            "message": f"recipe YAML does not parse: {exc}",
        }]
    return lint(doc if isinstance(doc, dict) else None)


# ----- bundled-file snapshots ----------------------------------------------

def _safe_stem(stem: str) -> str | None:
    return stem if re.fullmatch(r"[\w][\w.-]*", stem or "") else None


def snapshot_file(path: Path, source: str = "launch") -> Path | None:
    """Copy a bundled recipe YAML into its snapshot dir if its content is new.

    Only files inside the bundled recipes directory are snapshotted — that is
    the surface agents edit through the shell with no other undo path.
    """
    try:
        path = path.resolve()
        path.relative_to(BUNDLED_DIR.resolve())
    except (OSError, ValueError):
        return None
    stem = _safe_stem(path.stem)
    if not stem or not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    target_dir = SNAPSHOT_DIR / stem
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(target_dir.glob("*.yaml"))
    if existing:
        try:
            if existing[-1].read_text(encoding="utf-8") == content:
                return existing[-1]
        except OSError:
            pass
    # Millisecond stamp: within-second edits (agent loops) must never collide
    # into the same filename and silently overwrite an older snapshot.
    stamp = int(time.time() * 1000)
    label = re.sub(r"[^A-Za-z0-9_]", "", source) or "edit"
    target = target_dir / f"{stamp}-{label}.yaml"
    while target.exists():
        stamp += 1
        target = target_dir / f"{stamp}-{label}.yaml"
    target.write_text(content, encoding="utf-8")
    for stale in sorted(target_dir.glob("*.yaml"))[:-_KEEP]:
        try:
            stale.unlink()
        except OSError:
            pass
    return target


def file_snapshots(stem: str) -> list[dict[str, Any]]:
    stem = _safe_stem(stem) or ""
    target_dir = SNAPSHOT_DIR / stem
    if not stem or not target_dir.is_dir():
        return []
    rows = []
    for f in sorted(target_dir.glob("*.yaml"), reverse=True):
        stat = f.stat()
        rows.append({
            "name": f.name,
            "taken_at": int(stat.st_mtime),
            "size": stat.st_size,
            "source": f.stem.split("-")[-1],
        })
    return rows


def file_snapshot_read(stem: str, name: str) -> str | None:
    stem = _safe_stem(stem) or ""
    if not stem or "/" in name or name != Path(name).name:
        return None
    f = SNAPSHOT_DIR / stem / name
    try:
        return f.read_text(encoding="utf-8") if f.is_file() else None
    except OSError:
        return None


def file_snapshot_restore(stem: str, name: str) -> Path | None:
    """Write a snapshot back over ``recipes/<stem>.yaml`` (snapshotting the
    current content first so the restore itself is undoable)."""
    content = file_snapshot_read(stem, name)
    if content is None:
        return None
    live = BUNDLED_DIR / f"{stem}.yaml"
    if live.exists():
        snapshot_file(live, source="prerestore")
    live.write_text(content, encoding="utf-8")
    return live
