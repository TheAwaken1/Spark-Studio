"""Local mirror of the curated DGX Spark recipe sources.

We shallow-clone three upstream repos into ``app/data/registry/`` and parse
them into a single in-memory index that Forge and Fix can query offline:

  - spark-arena/recipe-registry  (official + experimental curated recipes)
  - eugr/spark-vllm-docker        (reference recipes + per-model mods)
  - spark-arena/sparkrun          (pinned image tags via versions.yaml)

Forge looks up an HF repo here before falling back to heuristics. Fix
inlines the matching recipe + relevant mods in the prompt instead of
asking the agent to fetch URLs.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REGISTRY_ROOT = Path(__file__).parent / "data" / "registry"

REPOS = {
    "recipe-registry": "https://github.com/spark-arena/recipe-registry.git",
    "spark-vllm-docker": "https://github.com/eugr/spark-vllm-docker.git",
    "sparkrun": "https://github.com/spark-arena/sparkrun.git",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUANT_TOKENS = {
    "fp8", "fp16", "bf16", "int4", "int8",
    "awq", "gptq", "nvfp4", "mxfp4", "autoround", "marlin",
    "gguf",
}


@dataclass
class Recipe:
    name: str
    model: str | None
    engine: str
    container: str | None
    mods: list[str]
    defaults: dict[str, Any]
    env: dict[str, str]
    command: str
    description: str
    source_repo: str          # e.g. "spark-vllm-docker"
    source_path: str          # e.g. "recipes/qwen3.5-122b-fp8.yaml"
    raw_yaml: str
    min_nodes: int = 1        # Sparks required (1 GPU per Spark)
    max_nodes: int | None = None
    tokens: set[str] = field(default_factory=set)

    def to_public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "engine": self.engine,
            "container": self.container,
            "mods": self.mods,
            "defaults": self.defaults,
            "env": self.env,
            "command": self.command,
            "description": self.description,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "source": {
                "repo": self.source_repo,
                "path": self.source_path,
            },
            "raw_yaml": self.raw_yaml,
        }


@dataclass
class Mod:
    name: str            # folder name (e.g. "fix-qwen3.5-autoround")
    source_repo: str
    source_path: str     # "mods/fix-qwen3.5-autoround"
    files: dict[str, str]  # filename -> content
    tokens: set[str]

    def to_public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": {"repo": self.source_repo, "path": self.source_path},
            "files": self.files,
        }


_recipes: list[Recipe] = []
_mods: list[Mod] = []
_image_pins: dict[str, str] = {}    # versions.yaml from sparkrun
_last_indexed_at: float = 0.0
# What the most recent sync() pulled in, for the "new recipes" UI badge.
_last_sync: dict[str, Any] = {"at": None, "new_recipes": [], "updated_files": 0}


# ----------------------------- helpers ------------------------------------

def _tokenize(*parts: str | None) -> set[str]:
    out: set[str] = set()
    for p in parts:
        if not p:
            continue
        for tok in _TOKEN_RE.findall(p.lower()):
            if len(tok) >= 2:
                out.add(tok)
    return out


def _git(args: list[str], cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    try:
        res = subprocess.run(
            ["git"] + args,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return res.returncode, res.stdout or ""
    except FileNotFoundError:
        return 127, "git not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "git timed out"


def _clone_or_pull(name: str, url: str) -> tuple[bool, str]:
    target = REGISTRY_ROOT / name
    if not target.exists():
        REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
        rc, out = _git(["clone", "--depth", "1", url, str(target)])
        return rc == 0, out
    rc, out = _git(["fetch", "--depth", "1", "origin"], cwd=target)
    if rc != 0:
        return False, out
    rc, out2 = _git(["reset", "--hard", "origin/HEAD"], cwd=target)
    return rc == 0, (out + out2)


def _commit_short(target: Path) -> str | None:
    rc, out = _git(["rev-parse", "--short", "HEAD"], cwd=target, timeout=10)
    if rc != 0:
        return None
    return out.strip() or None


def _commit_full(target: Path) -> str | None:
    rc, out = _git(["rev-parse", "HEAD"], cwd=target, timeout=10)
    if rc != 0:
        return None
    return out.strip() or None


# ----------------------------- parsing ------------------------------------

def _walk_yaml(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.y*ml") if p.is_file())


def _parse_recipe_yaml(text: str) -> dict[str, Any] | None:
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(doc, dict):
        return None
    return doc


def _build_recipe(doc: dict[str, Any], source_repo: str, source_path: str, raw: str) -> Recipe | None:
    # The spark-vllm-docker schema. recipe-registry follows the same shape.
    name = doc.get("name") or Path(source_path).stem
    model = doc.get("model")
    engine = "vllm"  # all curated recipes today are vLLM
    container = doc.get("container")
    mods = list(doc.get("mods") or [])
    defaults = dict(doc.get("defaults") or {})
    env = {str(k): str(v) for k, v in (doc.get("env") or {}).items()}
    command = str(doc.get("command") or "").strip()
    description = str(doc.get("description") or "").strip()
    if not command and not container:
        # Not a runnable recipe (could be a fragment); skip.
        return None

    def _as_int(v: Any) -> int | None:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    # Node requirement: min_nodes is authoritative; a tensor_parallel default
    # also implies node count on Spark (1 GPU per box), so take the larger.
    min_nodes = max(
        _as_int(doc.get("min_nodes")) or 1,
        _as_int(defaults.get("tensor_parallel")) or 1,
    )
    max_nodes = _as_int(doc.get("max_nodes"))
    tokens = _tokenize(name, model, source_path) | _tokenize(*mods)
    return Recipe(
        name=str(name),
        model=str(model) if model else None,
        engine=engine,
        container=str(container) if container else None,
        mods=mods,
        defaults=defaults,
        env=env,
        command=command,
        description=description,
        source_repo=source_repo,
        source_path=source_path,
        raw_yaml=raw,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        tokens=tokens,
    )


def _scan_recipes_in(repo_dir: Path, source_repo: str, subdirs: list[str]) -> list[Recipe]:
    out: list[Recipe] = []
    for sub in subdirs:
        root = repo_dir / sub
        for yml in _walk_yaml(root):
            try:
                raw = yml.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            doc = _parse_recipe_yaml(raw)
            if not doc:
                continue
            rel = yml.relative_to(repo_dir).as_posix()
            r = _build_recipe(doc, source_repo, rel, raw)
            if r:
                out.append(r)
    return out


def _scan_mods_in(repo_dir: Path, source_repo: str) -> list[Mod]:
    mods_root = repo_dir / "mods"
    if not mods_root.exists():
        return []
    out: list[Mod] = []
    for child in sorted(mods_root.iterdir()):
        if not child.is_dir():
            continue
        files: dict[str, str] = {}
        for f in sorted(child.iterdir()):
            if not f.is_file():
                continue
            if f.stat().st_size > 64 * 1024:
                continue  # skip oversized blobs
            try:
                files[f.name] = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        if not files:
            continue
        rel = child.relative_to(repo_dir).as_posix()
        out.append(
            Mod(
                name=child.name,
                source_repo=source_repo,
                source_path=rel,
                files=files,
                tokens=_tokenize(child.name),
            )
        )
    return out


def _scan_image_pins(sparkrun_dir: Path) -> dict[str, str]:
    f = sparkrun_dir / "versions.yaml"
    if not f.exists():
        return {}
    try:
        doc = yaml.safe_load(f.read_text(encoding="utf-8", errors="replace"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(doc, dict):
        return {}
    return {str(k): str(v) for k, v in doc.items() if isinstance(v, (str, int))}


# ----------------------------- index --------------------------------------

def reindex() -> dict[str, Any]:
    global _recipes, _mods, _image_pins, _last_indexed_at
    recipes: list[Recipe] = []
    mods: list[Mod] = []

    svd = REGISTRY_ROOT / "spark-vllm-docker"
    if svd.exists():
        recipes += _scan_recipes_in(svd, "spark-vllm-docker", ["recipes"])
        mods += _scan_mods_in(svd, "spark-vllm-docker")

    rr = REGISTRY_ROOT / "recipe-registry"
    if rr.exists():
        recipes += _scan_recipes_in(rr, "recipe-registry", ["official-recipes", "experimental-recipes", "tuning"])

    spr = REGISTRY_ROOT / "sparkrun"
    pins = _scan_image_pins(spr) if spr.exists() else {}

    _recipes = recipes
    _mods = mods
    _image_pins = pins
    _last_indexed_at = time.time()
    return status()


def status() -> dict[str, Any]:
    repos = []
    for name in REPOS:
        target = REGISTRY_ROOT / name
        if not target.exists():
            repos.append({"name": name, "present": False})
            continue
        st = target.stat()
        repos.append({
            "name": name,
            "present": True,
            "commit": _commit_short(target),
            "mtime": st.st_mtime,
        })
    return {
        "root": str(REGISTRY_ROOT),
        "repos": repos,
        "recipe_count": len(_recipes),
        "mod_count": len(_mods),
        "image_pins": _image_pins,
        "indexed_at": _last_indexed_at,
        "last_sync": _last_sync,
    }


async def sync(reindex_after: bool = True) -> dict[str, Any]:
    """Run clone/pull for each repo, then rebuild the index. Returns status."""
    global _last_sync
    results: list[dict[str, Any]] = []

    def _do_one(name: str, url: str) -> dict[str, Any]:
        target = REGISTRY_ROOT / name
        old = _commit_full(target) if target.exists() else None
        ok, out = _clone_or_pull(name, url)
        new = _commit_full(target) if target.exists() else None
        return {"name": name, "ok": ok, "log": out[-2000:], "old": old, "new": new}

    loop = asyncio.get_event_loop()
    for name, url in REPOS.items():
        results.append(await loop.run_in_executor(None, _do_one, name, url))

    # Diff each moved repo so the UI can surface what actually arrived.
    # The old HEAD object survives the shallow fetch+reset, so a plain
    # two-commit diff works; if it doesn't (pruned), we just skip the detail.
    new_recipes: list[dict[str, str]] = []
    updated_files = 0
    for res in results:
        old, new = res.pop("old", None), res.pop("new", None)
        if not (res["ok"] and old and new and old != new):
            continue
        rc, out = _git(["diff", "--name-status", old, new],
                       cwd=REGISTRY_ROOT / res["name"], timeout=30)
        if rc != 0:
            continue
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) < 2 or not parts[-1].endswith((".yaml", ".yml")):
                continue
            updated_files += 1
            if parts[0][:1] == "A":
                new_recipes.append({
                    "repo": res["name"],
                    "path": parts[-1],
                    "name": Path(parts[-1]).stem,
                })
    if any(r["ok"] for r in results):
        _last_sync = {
            "at": time.time(),
            "new_recipes": new_recipes,
            "updated_files": updated_files,
        }

    if reindex_after:
        reindex()
    return {"status": status(), "results": results}


# ----------------------------- lookup -------------------------------------

def _norm_repo(repo: str) -> str:
    return repo.strip().lower()


def by_exact_repo(repo: str) -> list[Recipe]:
    target = _norm_repo(repo)
    return [r for r in _recipes if r.model and _norm_repo(r.model) == target]


def _model_family_token(repo: str) -> str | None:
    """First non-trivial token from the repo basename, e.g.
    'Qwen/Qwen3.5-122B-A10B-FP8' -> 'qwen3.5'."""
    base = repo.split("/", 1)[-1]
    toks = _TOKEN_RE.findall(base.lower())
    for tok in toks:
        if len(tok) >= 4 and not tok.isdigit() and tok not in _QUANT_TOKENS:
            return tok
    return toks[0] if toks else None


def _quant_tokens_in(repo: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(repo.lower()) if t in _QUANT_TOKENS}


def by_similar(repo: str, k: int = 3) -> list[Recipe]:
    """Return up to k most similar recipes by family + quant overlap."""
    family = _model_family_token(repo)
    quants = _quant_tokens_in(repo)
    repo_toks = _tokenize(repo)
    scored: list[tuple[float, Recipe]] = []
    for r in _recipes:
        score = 0.0
        if family and family in r.tokens:
            score += 3.0
        score += 1.5 * len(quants & r.tokens)
        score += 0.5 * len(repo_toks & r.tokens) / max(1, len(repo_toks))
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored[:k]]


def relevant_mods(repo: str | None, log_signature: str | None = None, k: int = 3) -> list[Mod]:
    """Pick the most likely mods for a given repo + log substring. Used by Fix
    to inline mod content (run.sh + patches) so the agent sees real fixes."""
    family = _model_family_token(repo) if repo else None
    repo_toks = _tokenize(repo) if repo else set()
    log_toks = _tokenize(log_signature) if log_signature else set()
    scored: list[tuple[float, Mod]] = []
    for m in _mods:
        score = 0.0
        if family and family in m.tokens:
            score += 3.0
        score += 1.0 * len(repo_toks & m.tokens)
        score += 1.0 * len(log_toks & m.tokens)
        if score > 0:
            scored.append((score, m))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored[:k]]


def all_recipes() -> list[Recipe]:
    return list(_recipes)


def all_mods() -> list[Mod]:
    return list(_mods)


def image_pins() -> dict[str, str]:
    return dict(_image_pins)


# Build the index at import time so the first request is warm.
try:
    reindex()
except Exception:  # noqa: BLE001
    # Sync hasn't run yet — that's fine; status() will report repo_count=0.
    pass
