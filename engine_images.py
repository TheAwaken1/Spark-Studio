"""Engine container image management for the spark-vllm-docker pipeline.

The recipes always run whatever is tagged ``vllm-node`` (plus the -tf5 /
-mxfp4 variants). "Some new model doesn't run" is almost always a stale
runner image — new architectures land in vLLM nightlies, and eugr's
``build-and-copy.sh`` (no flags) re-pulls the tested nightly and retags.

This module provides the read side: which images exist, which one
``vllm-node`` actually is, and — on demand — the vLLM/FlashInfer versions
inside an image (probed by running python inside it; cached by image id).
The build/pull actions stream through the server's SSE endpoint.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

SPARK_VLLM_DOCKER = Path(__file__).parent / "data" / "registry" / "spark-vllm-docker"

# Image repos that belong to the Spark vLLM ecosystem.
_RELEVANT_RE = re.compile(
    r"^(vllm-node|eugr/spark-vllm|ghcr\.io/spark-arena/|sparkarena/|vllm/vllm-openai)", re.I
)

# image id → {"vllm": …, "flashinfer": …} (or {"error": …}); probing spins up
# a container (~2-5 s), so never do it implicitly for every image.
_probe_cache: dict[str, dict[str, Any]] = {}

# Flags build-and-copy.sh accepts that we allow from the UI. Value-taking
# flags consume the next token. Everything else is rejected — the command is
# exec'd as an array, but there's no reason to pass through arbitrary input.
BUILD_FLAGS_BARE = {
    "--use-wheels", "--rebuild-vllm", "--rebuild-flashinfer",
    "--apply-preset-vllm-prs", "--pre-tf", "--exp-mxfp4",
    "--force-vllm-download", "--force-flashinfer-download", "--cleanup",
}
BUILD_FLAGS_VALUED = {
    "--vllm-ref", "--flashinfer-ref", "--apply-flashinfer-pr",
    "--apply-vllm-pr", "--tag", "--build-jobs",
}


def validate_build_flags(raw: str) -> tuple[list[str], str | None]:
    """Parse a user-supplied flag string against the allowlist.
    Returns (args, error). Values may not themselves look like flags."""
    import shlex
    try:
        tokens = shlex.split(raw or "")
    except ValueError as e:
        return [], f"unparseable flags: {e}"
    args: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in BUILD_FLAGS_BARE:
            args.append(t)
            i += 1
        elif t in BUILD_FLAGS_VALUED:
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("-"):
                return [], f"{t} needs a value"
            if not re.fullmatch(r"[\w./:@#-]+", tokens[i + 1]):
                return [], f"invalid value for {t}: {tokens[i + 1]!r}"
            args += [t, tokens[i + 1]]
            i += 2
        else:
            return [], f"flag not allowed: {t!r} (see build-and-copy.sh --help)"
    return args, None


def _docker(args: list[str], timeout: int = 30) -> tuple[int, str]:
    docker = shutil.which("docker")
    if not docker:
        return 127, "docker not found"
    try:
        res = subprocess.run([docker] + args, capture_output=True, text=True, timeout=timeout)
        return res.returncode, (res.stdout or res.stderr or "").strip()
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def list_images() -> dict[str, Any]:
    """Relevant local images + which image id vllm-node points at."""
    rc, out = _docker(["images", "--format",
                       "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"])
    if rc != 0:
        return {"error": out, "images": [], "pipeline_ready": SPARK_VLLM_DOCKER.exists()}
    rc2, node_id = _docker(["inspect", "--format", "{{.Id}}", "vllm-node"])
    node_id = node_id[7:19] if rc2 == 0 and node_id.startswith("sha256:") else (node_id[:12] if rc2 == 0 else None)

    images = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 4 or "<none>" in parts[0]:
            continue
        ref, iid, created, size = parts
        if not _RELEVANT_RE.match(ref):
            continue
        images.append({
            "ref": ref,
            "id": iid,
            "created": created.split(" +")[0],
            "size": size,
            "is_vllm_node": node_id is not None and iid == node_id,
            "versions": _probe_cache.get(iid),
        })
    images.sort(key=lambda x: (not x["is_vllm_node"], x["ref"]))
    return {"images": images, "vllm_node_id": node_id,
            "pipeline_ready": (SPARK_VLLM_DOCKER / "build-and-copy.sh").exists()}


def probe_image(ref: str) -> dict[str, Any]:
    """vLLM + FlashInfer versions inside an image (cached by image id)."""
    if not re.fullmatch(r"[\w./:@-]+", ref or ""):
        return {"error": "invalid image reference"}
    rc, iid = _docker(["inspect", "--format", "{{.Id}}", ref])
    if rc != 0:
        return {"error": f"image not found: {ref}"}
    iid = iid[7:19] if iid.startswith("sha256:") else iid[:12]
    if iid in _probe_cache:
        return {"ref": ref, "id": iid, **_probe_cache[iid]}
    rc, out = _docker([
        "run", "--rm", "--network", "none", ref, "python3", "-c",
        "import vllm, flashinfer; print(vllm.__version__ + '|' + flashinfer.__version__)",
    ], timeout=120)
    last = (out.splitlines() or [""])[-1]
    if rc == 0 and "|" in last:
        v, fi = last.split("|", 1)
        result = {"vllm": v.strip(), "flashinfer": fi.strip()}
    else:
        result = {"error": last[:200] or "probe failed"}
    _probe_cache[iid] = result
    return {"ref": ref, "id": iid, **result}


def vllm_node_age_days() -> float | None:
    """Days since the vllm-node image was created (cheap; for doctor)."""
    rc, out = _docker(["inspect", "--format", "{{.Created}}", "vllm-node"])
    if rc != 0:
        return None
    try:
        from datetime import datetime, timezone
        created = datetime.fromisoformat(out.replace("Z", "+00:00"))
        return round((datetime.now(timezone.utc) - created).total_seconds() / 86400, 1)
    except Exception:  # noqa: BLE001
        return None
