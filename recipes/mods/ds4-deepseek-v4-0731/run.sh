#!/usr/bin/env bash
# Build Entrpi/ds4 v0.5.3 for GB10 and cache the matching 0731 DSpark
# drafter in sparkrun's persistent Hugging Face cache mount.
set -euo pipefail

src=/cache/huggingface/ds4-engine
assets=/cache/huggingface/ds4-assets
ref="${DS4_REF:-v0.5.3}"
drafter=DSpark-drafter-Q2K-Q8-0731.gguf
mkdir -p "$assets"

if [[ ! -d "$src/.git" ]]; then
  git clone --depth 1 --branch "$ref" https://github.com/Entrpi/ds4.git "$src"
else
  git -C "$src" fetch --depth 1 origin "$ref"
  git -C "$src" reset --hard FETCH_HEAD
fi

if [[ ! -x "$src/ds4-server" ]]; then
  make -C "$src" cuda -j"$(nproc)" CUDA_ARCH=sm_121
fi

test -x "$src/ds4-server"

if [[ ! -s "$assets/$drafter" ]]; then
  python3 - "$assets/$drafter" <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil, sys

dest = Path(sys.argv[1])
src = Path(hf_hub_download(
    repo_id="bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF",
    filename="DSpark-drafter-Q2K-Q8-0731.gguf",
))
tmp = dest.with_suffix(dest.suffix + ".partial")
shutil.copyfile(src, tmp)
tmp.replace(dest)
PY
fi

test -s "$assets/$drafter"
