#!/usr/bin/env bash
# Apply the pinned vLLM 0.23 compatibility/optimization patches from
# Entrpi/qwen3.5-122B-A10B-on-spark before sparkrun starts vLLM.
set -euo pipefail

readonly UPSTREAM_COMMIT="a77cbdab26956ef6ac9cdca544e5fb9ec1f3bb2a"
readonly BASE_URL="https://raw.githubusercontent.com/Entrpi/qwen3.5-122B-A10B-on-spark/${UPSTREAM_COMMIT}/runtime"
readonly PATCH_DIR="$(mktemp -d /tmp/qwen35-hybrid-patches.XXXXXX)"
trap 'rm -rf "$PATCH_DIR"' EXIT

python3 - "$BASE_URL" "$PATCH_DIR" <<'PY'
from pathlib import Path
from urllib.request import urlopen
import sys

base_url, output_dir = sys.argv[1], Path(sys.argv[2])
files = (
    "patch_fla_shmem.py",
    "patch_inc_hybrid.py",
    "patch_int8_lmhead_v3.py",
    "patch_unify2.py",
    "patch_prefix_align.py",
)
for name in files:
    with urlopen(f"{base_url}/{name}", timeout=60) as response:
        payload = response.read()
    (output_dir / name).write_bytes(payload)
    print(f"[qwen35-hybrid] fetched {name} ({len(payload)} bytes)", flush=True)
PY

(
  cd "$PATCH_DIR"
  sha256sum -c <<'SUMS'
842e1fbe12aa5c7bdc2cbb98465921ad2a3cee7bba61626af2732863ed67e903  patch_fla_shmem.py
9a1328d8f037cf9a7be6ce898b5ca3f615c5ae061f39f65bb1a0a45fc45be6c6  patch_inc_hybrid.py
a4af0df014a4777acc28cd758c052a7838bdff121e2f83e925cb61273a7a0744  patch_int8_lmhead_v3.py
a42609fdacc04ae7ed6cdf443f72ddc954bf6b9226b3969dea5825c9e2a2322f  patch_unify2.py
d8ab004730195d6372faad3d5d7e2209399f6b1f1cf0935266b97aabb2d966ee  patch_prefix_align.py
SUMS
)

# FLA is a prefill-only optimization and is deliberately non-fatal upstream.
python3 "$PATCH_DIR/patch_fla_shmem.py" || \
  echo "[qwen35-hybrid] warning: optional FLA shared-memory patch did not apply"

# These four are required for this exact hybrid + DFlash profile. Fail closed if
# the pinned image no longer matches their expected vLLM 0.23 source anchors.
python3 "$PATCH_DIR/patch_inc_hybrid.py"
python3 "$PATCH_DIR/patch_int8_lmhead_v3.py"
python3 "$PATCH_DIR/patch_unify2.py"
python3 "$PATCH_DIR/patch_prefix_align.py"

echo "[qwen35-hybrid] required vLLM 0.23 patches applied"
