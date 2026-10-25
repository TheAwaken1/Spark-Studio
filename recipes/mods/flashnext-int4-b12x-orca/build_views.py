#!/usr/bin/env python3
"""Build, inside the container, the two model folders the b12x vLLM needs -- symlinks + a few small JSON files, nothing
copied, nothing written outside the container:

  <out>/model      the served model: every snapshot file linked, the 33 PLE-table files (shipped in an unindexed
                   `ple-table/` subfolder) linked at top level, and
                     * model.safetensors.index.json = the shipped index + the 128 `ngram_embedding.shard_N.weight` keys
                       and the table's `ngram_embedding.weight_scale`. ONLY those keys: two of the table files also carry
                       1,057 tensors of the original checkpoint (layer 1's original experts, shared expert and router
                       gate). A loader that reads by index key never touches them; one that reads whole files would
                       silently overwrite V8's healed layer 1 -- see the loader filter in patch_b12x.py.
                     * config.json = shipped config + `text_config.ple_embedding_dtype` (b12x's PLE storage needs it;
                       absent -> bf16 assumed) + the `Qwen3_8FlashNextForConditionalGeneration` architecture name this
                       vLLM registers (it does not know `Qwen4ExpForConditionalGeneration`).
  <out>/draft-k10  the MTP draft: only the files the speculative head needs (its own tensors, the int4 lm_head, the
                   embedding shard) + an index of just those keys + a config routing top-k 10 with the head's own
                   shared-expert width (640; the model's is 1280). Slim on purpose: a draft folder that links every
                   shard makes the draft load walk the whole model again (2026-09-11 and 2026-09-21 out-of-memory).

  usage: build_views.py <snapshot dir> <output root>          prints one summary line; exit != 0 on any inconsistency
"""
import glob, json, os, shutil, struct, sys

snap, root = (os.path.abspath(a) for a in sys.argv[1:3])
view, draft = os.path.join(root, "model"), os.path.join(root, "draft-k10")
B12X_ARCH = "Qwen3_8FlashNextForConditionalGeneration"


def fail(msg):
    sys.exit(f"build_views: {msg}")


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return h


def fresh(d):
    if os.path.lexists(d):
        shutil.rmtree(d)
    os.makedirs(d)


index = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
wm = dict(index["weight_map"])
cfg = json.load(open(os.path.join(snap, "config.json")))
text = cfg.get("text_config", cfg)
parts = int(text.get("split_ngram_parts", 0)) or fail("config has no split_ngram_parts")

# ---------------------------------------------------------------- served model view
fresh(view)
for name in sorted(os.listdir(snap)):
    if name in ("config.json", "model.safetensors.index.json", "ple-table"):
        continue
    os.symlink(os.path.join(snap, name), os.path.join(view, name))
ple_files = sorted(glob.glob(os.path.join(snap, "ple-table", "*.safetensors")))
ple_files or fail(f"no ple-table/*.safetensors in {snap}")
shards, scale_key, table_dtype = set(), None, None
for f in ple_files:
    base = os.path.basename(f)
    if os.path.lexists(os.path.join(view, base)):
        fail(f"table file name {base} collides with a model file")
    os.symlink(f, os.path.join(view, base))
    for key, meta in header(f).items():
        if ".ple.ple_embedding.ngram_embedding." not in key:
            continue                                           # foreign tensors stay out of the index
        if key in wm:
            fail(f"{key} is already indexed -- unexpected checkpoint layout")
        wm[key] = base
        leaf = key.split(".ngram_embedding.", 1)[1]
        if leaf == "weight_scale":
            scale_key = key
        elif leaf.startswith("shard_") and leaf.endswith(".weight"):
            shards.add(int(leaf[len("shard_"):-len(".weight")]))
            table_dtype = table_dtype or meta["dtype"]
if shards != set(range(parts)):
    fail(f"PLE table incomplete: {len(shards)} of {parts} shards found")
if table_dtype != "F8_E4M3" or scale_key is None:
    fail(f"expected an fp8 table with a weight_scale, got {table_dtype} / scale {scale_key}")
json.dump({"metadata": index.get("metadata", {}), "weight_map": wm},
          open(os.path.join(view, "model.safetensors.index.json"), "w"))
text["ple_embedding_dtype"] = "float8_e4m3fn"
archs = cfg.setdefault("architectures", [])
if B12X_ARCH not in archs:
    archs.append(B12X_ARCH)
json.dump(cfg, open(os.path.join(view, "config.json"), "w"), indent=2)

# ---------------------------------------------------------------- slim MTP draft folder
fresh(draft)
keep = {k: v for k, v in wm.items()
        if k.startswith("mtp.") or k.startswith("lm_head.") or k.endswith("embed_tokens.weight")}
if not any(k.startswith("mtp.") for k in keep):
    fail("no mtp.* tensors in the index -- this snapshot has no speculative head")
for name in sorted(os.listdir(view)):
    if name.endswith(".safetensors") and name not in set(keep.values()):
        continue
    if name in ("config.json", "model.safetensors.index.json"):
        continue
    os.symlink(os.path.join(view, name), os.path.join(draft, name))
json.dump({"metadata": {}, "weight_map": keep}, open(os.path.join(draft, "model.safetensors.index.json"), "w"))
dcfg = json.loads(json.dumps(cfg))
dtext = dcfg.get("text_config", dcfg)
dtext["num_experts_per_tok"] = 10
head = header(os.path.join(snap, "model_extra_tensors.safetensors"))
widths = {v["shape"][0] for k, v in head.items() if k.endswith("mlp.shared_expert.gate_proj.weight")}
if len(widths) == 1:
    dtext["shared_expert_intermediate_size"] = widths.pop()
json.dump(dcfg, open(os.path.join(draft, "config.json"), "w"), indent=2)

print(f"model view: {len(os.listdir(view))} entries, +{len(shards)} table shards +1 scale indexed "
      f"({len(ple_files)} table files) | draft: {len(keep)} keys in {len(set(keep.values()))} files, top-k 10, "
      f"shared expert {dtext.get('shared_expert_intermediate_size')}")
