"""Local HuggingFace cache scanner + deleter.

Models can live in several places on a Spark box: the default
``~/.cache/huggingface/hub``, an ``HF_HOME``/``HF_HUB_CACHE`` override in the
server's environment, or — most commonly here — a cache directory that only
the *recipes* know about, delivered to engine containers via
``-e HF_HUB_CACHE=...`` flags in their raw commands. We scan all of them.
"""

from __future__ import annotations

import os
import re
import shutil
import stat as stat_mod
from pathlib import Path

_RAW_CMD_CACHE_RE = re.compile(
    r"(?:-e\s+|\b)(?:HF_HUB_CACHE|HF_HOME)=(\"[^\"]*\"|'[^']*'|\S+)"
)


def _recipe_cache_hints() -> list[Path]:
    """Cache dirs referenced by saved recipes (env json or raw_cmd -e flags)."""
    import db

    hints: list[Path] = []
    try:
        for r in db.recipes_list():
            for k in ("HF_HUB_CACHE", "HF_HOME"):
                v = (r.get("env") or {}).get(k)
                if v:
                    hints.append(Path(str(v)))
            for m in _RAW_CMD_CACHE_RE.finditer(r.get("raw_cmd") or ""):
                hints.append(Path(m.group(1).strip("\"'")))
    except Exception:  # noqa: BLE001
        pass
    return hints


def _hub_dirs() -> list[Path]:
    """Every directory that holds ``models--*`` entries directly."""
    candidates: list[Path] = []
    for var in ("HF_HUB_CACHE", "HF_HOME"):
        v = os.environ.get(var)
        if v:
            candidates.append(Path(v))
    candidates.append(Path.home() / ".cache" / "huggingface")
    candidates += _recipe_cache_hints()

    hubs: list[Path] = []
    seen: set[Path] = set()
    for c in candidates:
        # A candidate is either the hub itself (models--* inside) or an
        # HF_HOME-style root with a hub/ subdirectory.
        for p in (c, c / "hub"):
            try:
                p = p.resolve()
                if p in seen or not p.is_dir():
                    continue
                if any(d.name.startswith("models--") for d in p.iterdir()):
                    seen.add(p)
                    hubs.append(p)
            except OSError:
                continue
    return hubs


def scan() -> list[dict]:
    out: list[dict] = []
    seen_paths: set[str] = set()
    for hub in _hub_dirs():
        for d in sorted(hub.iterdir()):
            if not d.name.startswith("models--") or not d.is_dir():
                continue
            if str(d) in seen_paths:
                continue
            seen_paths.add(str(d))
            repo = d.name.removeprefix("models--").replace("--", "/")
            size = _dir_size(d)
            out.append({
                "repo": repo,
                "size_gb": round(size / 1e9, 2),
                "path": str(d),
                "cache": str(hub),
            })
    return sorted(out, key=lambda m: m["repo"])


def delete(path: str) -> dict:
    """Remove a cached model directory. Only accepts a ``models--*`` dir that
    sits directly inside one of the known hub dirs — never an arbitrary path."""
    target = Path(path).resolve()
    if not target.name.startswith("models--"):
        raise ValueError("not a HuggingFace model cache directory")
    if target.parent not in _hub_dirs():
        raise ValueError("path is outside every known HF cache")
    if not target.is_dir():
        raise ValueError("model directory no longer exists")
    freed = _dir_size(target)
    shutil.rmtree(target)
    return {
        "deleted": target.name.removeprefix("models--").replace("--", "/"),
        "freed_gb": round(freed / 1e9, 2),
    }


def _dir_size(p: Path) -> int:
    """Bytes actually on disk. lstat so the snapshots/ symlinks into blobs/
    don't double-count every model file."""
    total = 0
    for root, _, files in os.walk(p, followlinks=False):
        for f in files:
            try:
                st = os.lstat(os.path.join(root, f))
                if not stat_mod.S_ISLNK(st.st_mode):
                    total += st.st_size
            except OSError:
                pass
    return total
