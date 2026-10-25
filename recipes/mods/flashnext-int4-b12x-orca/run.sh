#!/bin/bash
# Mod: flashnext-int4-b12x
# Lets eugr's vllm-node-b12x image serve azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound across two GB10 nodes with
# the b12x RoCE all-reduce (recipe: recipes/qwen3.8-flash-next-int4-b12x.yaml).
#
# The launcher copies this folder into EVERY node's container and runs run.sh before the launch script; a non-zero exit
# aborts the launch, so a node with a missing or incomplete model never starts serving. Nothing outside the container is
# written, the Hugging Face cache is read-only here, and nothing is copied: the two folders below are symlinks + small JSON.
#
#   1. check the model in the HF cache (what ~/Qwen3.8-Flash-Next-Int4-FAST/setup.sh downloads)
#   2. build /workspace/flashnext/model     (served model view: PLE table indexed, config fields b12x needs)   build_views.py
#            /workspace/flashnext/draft-k10 (slim MTP draft folder, top-k 10, the head's own shared-expert width)
#   3. patch this container's vLLM: fp8 hybrid, int4 lm_head, loader index filter (required); draft x2, GDN fixes (optional)
#                                                                                                                 patch_b12x.py
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FLASHNEXT_REPO:-/root/.cache/huggingface/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound}"
OUT=/workspace/flashnext
die(){ echo "FATAL $(hostname): $*" >&2; exit 1; }
# A variant copy of this mod (eugr-setup.sh --orca installs mods/flashnext-int4-b12x-orca) carries variant.env, which points
# REPO at the variant's local HF-cache repo and sets ORCA=1 (the Orca-abliterated build, see orca/ in the repo). Without the
# file this is the base model, unchanged.
ORCA=0
# Spark Studio reuses this installed mod for both entries. The normal recipe
# sets FLASHNEXT_VARIANT=base; otherwise the adjacent variant.env preserves the
# existing Orca behavior.
if [ "${FLASHNEXT_VARIANT:-orca}" != base ] && [ -f "$MOD_DIR/variant.env" ]; then
  . "$MOD_DIR/variant.env"
fi

[ -f "$REPO/refs/main" ] || die "model not in the HF cache ($(basename "$REPO")) -- run ./setup.sh$([ "$ORCA" = 1 ] && echo " --orca") in ~/Qwen3.8-Flash-Next-Int4-FAST on this node"
REV=$(cat "$REPO/refs/main"); SNAP="$REPO/snapshots/$REV"
[ -d "$SNAP/ple-table" ] || die "snapshot ${REV:0:8} has no ple-table -- run ./setup.sh in ~/Qwen3.8-Flash-Next-Int4-FAST on this node"
for f in config.json model.safetensors.index.json model-healed-shared-expert.safetensors model_extra_tensors.safetensors \
         model-lmhead-int4.safetensors medium_chat_template.jinja; do
  [ -e "$SNAP/$f" ] || die "snapshot ${REV:0:8} is incomplete: $f is missing (download still running?)"
done
python3 -c "import b12x" 2>/dev/null || die "this is not the b12x image (no b12x package) -- the recipe needs vllm-node-b12x"
if [ "$ORCA" = 1 ]; then
  [ -f "$SNAP/orca_direction.safetensors" ] || die "Orca variant ${REV} has no orca_direction.safetensors -- re-run ./setup.sh --orca in ~/Qwen3.8-Flash-Next-Int4-FAST on this node"
  export ORCA_DIRECTION="$SNAP/orca_direction.safetensors"    # patch_b12x.py applies orca-proj only when this is set
fi

mkdir -p "$OUT"
VIEWS=$(python3 "$MOD_DIR/build_views.py" "$SNAP" "$OUT") || die "could not build the model folders"
PATCH=$(python3 "$MOD_DIR/patch_b12x.py" "$MOD_DIR") || die "could not patch vLLM: $PATCH"

echo "$(hostname): snapshot ${REV:0:8} | $VIEWS"
echo "$(hostname): $PATCH"
