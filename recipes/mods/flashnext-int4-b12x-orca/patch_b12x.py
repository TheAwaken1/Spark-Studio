#!/usr/bin/env python3
"""Patch the vllm-node-b12x image IN THE RUNNING CONTAINER so it can serve the INT4-AutoRound checkpoint. Every edit is
idempotent (marker comment) and checked (anchor must match exactly once, file must still parse). The image on disk is
never modified -- a new container starts clean and the mod re-applies this.

  REQUIRED (exit 1 if they cannot be applied -- the launch is aborted rather than serving a broken model):
    fp8-hybrid   300 side layers of the checkpoint are blockwise fp8 inside a GPTQ checkpoint; stock AutoGPTQConfig would
                 build them as bf16 and fail to load. Saren's vllm_fp8_hybrid.py (verbatim from the qwen38-flash-dgx
                 image) + its hook, gated by VLLM_FP8_HYBRID=1.
    int4 head    the model's ParallelLMHead is built without quant_config, so the int4 AutoRound lm_head cannot load
                 (the MTP head in this vLLM already passes it).
    index filter safety net for --load-format safetensors: reads tensors only where the folder's index says they live.
                 The b12x loader already does this; the stock safetensors loader reads whole files and would load the
                 original layer-1 weights sitting in two PLE-table files over V8's healed ones.
  OPTIONAL (a warning, the launch goes on):
    draft x2     VLLM_MTP_DRAFT_SCALE scales the draft's logits (exact: the verifier decides) -- +2.5pp acceptance.
    mtp-cap      lets b12x's QSA take up to 7 speculative tokens (stock cap 4); the solo recipe runs 5.
    GDN fixes    spark-fla-shmem (GB10 gets the big GDN tiles) and spark-fla-warps (num_warps=2: fla#953 Blackwell race
                 that corrupts GDN state on the prefix-cache path), same one-liners as the qwen38-flash-dgx image.
    penalties    server-default presence/frequency penalty from --override-generation-config (stock vLLM drops both
                 keys, and requests default them to 0.0); a request that sets the field keeps its own value.
    orca-proj    ONLY for the Orca variant (ORCA_DIRECTION set by run.sh from variant.env): projects OrcaRouter's refusal
                 direction out of every decoder layer's attention and MLP block output (REQUIRED there). See orca/README.md.
    pad rows     dead MTP drafter under concurrency: the QSA metadata builder clears the whole unused tail of its request-id
                 buffer every step, so the drafter's padded graph rows (request counts not in {1,2,4,8,16}) never carry a
                 stale request index (b12x QSA error 128 -> ring commit skipped, NaN-poisoned, 1 token/step for the rest of
                 the request). Prunning/qwen38_flash_next/mtp_sweep/ROOTCAUSE.md.

  usage: patch_b12x.py <mod dir>
"""
import ast, os, re, shutil, sys

MOD = sys.argv[1]
SP = "/usr/local/lib/python3.12/dist-packages"
V = f"{SP}/vllm"
MARK = "flashnext-int4-b12x"
done, warn, fatal, unneeded = [], [], [], []


def edit(path, fn, label, required):
    try:
        src = open(path).read()
        if MARK + ":" + label in src:
            done.append(f"{label} (already)")
            return
        new = fn(src)
        if new is None or new == src:
            raise ValueError("anchor not found")
        ast.parse(new)
        open(path, "w").write(new)
        done.append(label)
    except Exception as e:  # noqa: BLE001
        (fatal if required else warn).append(f"{label}: {e} [{path}]")


def once(src, old, new):
    if src.count(old) != 1:
        return None
    return src.replace(old, new)


# 1. fp8 hybrid ----------------------------------------------------------------------------------------------------
shutil.copy(os.path.join(MOD, "vllm_fp8_hybrid.py"), f"{SP}/vllm_fp8_hybrid.py")
edit(f"{V}/model_executor/layers/quantization/auto_gptq.py",
     lambda s: s + f"\n\n# --- {MARK}:fp8-hybrid (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid import apply as _fp8_hybrid_apply\n"
                   "_fp8_hybrid_apply()\n",
     "fp8-hybrid", True)

# The model code moved: images up to 2026-09-13 ship it as vllm/models/qwen3_8_flash_next/{model,mtp}.py; from the
# 2026-09-21 image (vLLM 0.1.dev21473) it is vllm/models/qwen4_exp/nvidia/{model,mtp}.py (qwen3_8_flash_next is only a
# compatibility alias there). Patch whichever this image has.
MODEL_DIR = next((d for d in (f"{V}/models/qwen4_exp/nvidia", f"{V}/models/qwen3_8_flash_next")
                  if os.path.isfile(f"{d}/model.py")), f"{V}/models/qwen3_8_flash_next")

# 2. int4 lm_head ----------------------------------------------------------------------------------------------------
edit(f"{MODEL_DIR}/model.py",
     lambda s: once(s, '            prefix=maybe_prefix(prefix, "lm_head"),\n',
                    f'            quant_config=vllm_config.quant_config,  # {MARK}:int4-head\n'
                    '            prefix=maybe_prefix(prefix, "lm_head"),\n'),
     "int4-head", True)

# 3. index filter for the stock safetensors loader ----------------------------------------------------------------------
FILTER = f'''

# --- {MARK}:index-filter ---
# Only for folders under /workspace/flashnext: yield a tensor only from the file the folder's index names for it.
def _flashnext_index_filter():
    import json as _json
    from pathlib import Path as _Path
    _orig = DefaultModelLoader._safetensors_weights_iterator
    _maps = {{}}

    def _iterate(self, hf_weights_files, source, **kwargs):  # newer loaders pass indexed_tensor_files=...
        files = list(hf_weights_files)
        folders = {{_Path(f).parent for f in files}}
        if len(folders) != 1 or not str(next(iter(folders))).startswith("/workspace/flashnext/"):
            yield from _orig(self, files, source, **kwargs)
            return
        folder = next(iter(folders))
        if folder not in _maps:
            _maps[folder] = _json.load(open(folder / "model.safetensors.index.json"))["weight_map"]
        weight_map = _maps[folder]
        for f in files:
            here = _Path(f).name
            for name, tensor in _orig(self, [f], source, **kwargs):
                if weight_map.get(name) == here:
                    yield name, tensor

    DefaultModelLoader._safetensors_weights_iterator = _iterate


_flashnext_index_filter()
'''
edit(f"{V}/model_executor/model_loader/default_loader.py", lambda s: s + FILTER, "index-filter", True)

# 3b. the same filter for "mixed" files ------------------------------------------------------------------------------------
# A file that holds any PLE-table tensor does NOT go through _safetensors_weights_iterator: b12x's vLLM reads it directly in
# file_backed_safetensors_weights_iterator (every tensor in it, table rows as on-disk references). Two of the table files also
# carry layer 1's ORIGINAL experts / shared expert / gate -> boot 2 died on "experts has no parameter 'w2_weight'". Wrap that
# iterator too (default_loader calls it by its module-global name, so rebinding the name there is enough).
MIXED = f'''

# --- {MARK}:index-filter-mixed ---
def _flashnext_index_filter_mixed():
    import json as _json
    from pathlib import Path as _Path
    global file_backed_safetensors_weights_iterator
    _orig = file_backed_safetensors_weights_iterator
    _maps = {{}}

    def _filtered(hf_weights_files, ordinary_iterator, file_weight_filter, **kwargs):
        files = list(hf_weights_files)
        folders = {{_Path(f).parent for f in files}}
        if len(folders) != 1 or not str(next(iter(folders))).startswith("/workspace/flashnext/"):
            yield from _orig(files, ordinary_iterator, file_weight_filter, **kwargs)
            return
        folder = next(iter(folders))
        if folder not in _maps:
            _maps[folder] = _json.load(open(folder / "model.safetensors.index.json"))["weight_map"]
        weight_map = _maps[folder]
        for f in files:
            here = _Path(f).name
            for name, tensor in _orig([f], ordinary_iterator, file_weight_filter, **kwargs):
                if weight_map.get(name) == here:
                    yield name, tensor

    file_backed_safetensors_weights_iterator = _filtered


_flashnext_index_filter_mixed()
'''
edit(f"{V}/model_executor/model_loader/default_loader.py",
     lambda s: s + MIXED if "file_backed_safetensors_weights_iterator" in s else None, "index-filter-mixed", True)

# 4. draft x2 --------------------------------------------------------------------------------------------------------
def _scale(s):
    pat = re.compile(r"(self\.logits_processor = LogitsProcessor\(\n(\s+)config\.vocab_size,\n)")
    if len(pat.findall(s)) != 1:
        return None
    return pat.sub(lambda m: m.group(1) + m.group(2) +
                   f'scale=float(__import__("os").environ.get("VLLM_MTP_DRAFT_SCALE", "1.0")),  # {MARK}:draft-scale\n', s)
edit(f"{MODEL_DIR}/mtp.py", _scale, "draft-scale", False)

# 6. MTP depth up to 7 --------------------------------------------------------------------------------------------------
# b12x's QSA refuses more than 4 speculative tokens (_QSA_MAX_SPECULATIVE_TOKENS = 4). At depth 5-7 its raw ring is 12 rows
# (4 * ceil((4 + K) / 4)) and the b12x kernels take the ring size as a parameter; the only extra requirement is a page size
# that 12 divides, which the solo recipe's block_size 1632 gives (1632 = 12 * 136 = 8 * 204). Depth 8 is refused further
# down by b12x's GDN plan (state_index_columns = K + 1 <= 8), hence 7. Only the cap moves: nothing changes at depth <= 4.
# Measured 2026-09-27 on the pinned 09-13 image: depth 5 = +4 % run.py tok/s over depth 4, same quality.
def _mtp_cap(s):
    m = re.search(r"^_QSA_MAX_SPECULATIVE_TOKENS = (\d+)$", s, re.M)
    if not m:
        return None
    if int(m.group(1)) >= 7:
        return s + f"\n# {MARK}:mtp-cap (this image already allows {m.group(1)})\n"
    return s.replace(m.group(0), f"_QSA_MAX_SPECULATIVE_TOKENS = 7  # {MARK}:mtp-cap (was {m.group(1)})")
# Images whose QSA lives in b12x_qsa.py (eugr's 2026-10-15 build, vLLM 0.1.dev21830) have no such cap: nothing to patch.
B12X_QSA = os.path.isfile(f"{MODEL_DIR}/b12x_qsa.py")
_qsa = next((p for p in (f"{MODEL_DIR}/nvidia/qsa.py", f"{MODEL_DIR}/qsa.py")
             if os.path.isfile(p) and "_QSA_MAX_SPECULATIVE_TOKENS" in open(p).read()), None)
if _qsa:
    edit(_qsa, _mtp_cap, "mtp-cap", False)
elif B12X_QSA:
    unneeded.append("mtp-cap")
else:
    warn.append(f"mtp-cap: no QSA file with _QSA_MAX_SPECULATIVE_TOKENS under {MODEL_DIR}")

# 5. GDN fixes ---------------------------------------------------------------------------------------------------------
fla = f"{V}/third_party/flash_linear_attention/ops"
edit(f"{fla}/utils.py", lambda s: once(s, "DEFAULT = 102400", f"DEFAULT = 101376  # {MARK}:fla-shmem GB10 99KiB"),
     "fla-shmem", False)
edit(f"{fla}/chunk_delta_h.py",
     lambda s: once(s, "for num_warps in [2, 4]", f"for num_warps in [2]  # {MARK}:fla-warps fla#953"),
     "fla-warps", False)

# 7. server-default presence/frequency penalty ------------------------------------------------------------------------
# vLLM takes server-wide sampling defaults from --override-generation-config only for temperature/top_p/top_k/min_p/
# repetition_penalty, and the OpenAI requests default presence_penalty to 0.0, so a server default never applies. Recipes
# set --override-generation-config '{"presence_penalty":0.5}' against looping thinking blocks: let that key through and use
# it when the request does not set the field itself (a request that sends presence_penalty, even 0, keeps its own value).
edit(f"{V}/config/model.py",
     lambda s: once(s, '            "min_p",\n            "max_new_tokens",\n',
                    '            "min_p",\n            "max_new_tokens",\n'
                    f'            "presence_penalty",  # {MARK}:default-penalties\n            "frequency_penalty",\n'),
     "default-penalties", False)
for _proto in ("chat_completion", "completion"):
    edit(f"{V}/entrypoints/openai/{_proto}/protocol.py",
         lambda s: once(s, "            presence_penalty=self.presence_penalty,\n"
                           "            frequency_penalty=self.frequency_penalty,\n",
                        f"            presence_penalty=(self.presence_penalty if 'presence_penalty' in self.model_fields_set  # {MARK}:request-penalty-default\n"
                        "                else default_sampling_params.get('presence_penalty', self.presence_penalty)),\n"
                        "            frequency_penalty=(self.frequency_penalty if 'frequency_penalty' in self.model_fields_set\n"
                        "                else default_sampling_params.get('frequency_penalty', self.frequency_penalty)),\n"),
         f"request-penalty-default", False)

# 8. drafter pad rows ----------------------------------------------------------------------------------------------------
# The MTP drafter's step-0 graphs exist only for {1,2,4,8,16} requests; a decode step with another count replays a padded graph whose
# extra rows read the QSA builder's request-id buffer past the step's tokens. Stock code cleared that tail only when the step itself
# was padded, so the rows kept the previous prefill step's request index; b12x's QSA validation then charged that request (error 128),
# skipped its ring commit and NaN-poisoned it: 1 token/step until the request ended (parallel agents hit it constantly).
# The 2026-10-15 build (b12x_qsa.py layout) maps tokens to requests in a new kernel; the deterministic repro (prefill_probe
# phases 7/8/11/12) and tools/mtp_drafter_check.sh find no dead drafter on it, so it is left as shipped there.
_qsa = next((p for p in (f"{MODEL_DIR}/nvidia/qsa.py", f"{MODEL_DIR}/qsa.py") if os.path.isfile(p)), f"{MODEL_DIR}/nvidia/qsa.py")
if B12X_QSA:
    unneeded.append("drafter-pad-rows")
else:
    edit(_qsa,
         lambda s: once(s, "        if num_mapped_tokens < cm.num_actual_tokens:\n            request_ids[num_mapped_tokens:].fill_(-1)\n",
                        f"        self._request_ids[num_mapped_tokens:].fill_(-1)  # {MARK}:drafter-pad-rows (clear the whole unused tail)\n"),
         "drafter-pad-rows", False)

# 9. orca-proj (Orca variant only) ------------------------------------------------------------------------------------
# OrcaRouter's abliteration (orcarouter/Qwen3.8-Flash-Next-Uncensored, lambda = 1) is W' = W - r (r^T W) on every residual
# writer. The variant's bf16 writers (embeddings, ple.value_proj, the MTP head) carry it in their weights; for the trunk it is
# applied here at runtime to the two block outputs of every decoder layer -- attn_out (o_proj / GDN out_proj) and mlp_out
# (routed experts + shared expert, already summed) -- which is the same map and keeps the int4 / fp8 weights byte-identical
# (re-rounding an int4 weight that already sits on the grid would erase a 2 % edit). P is idempotent, so a layer that also
# carries the fold in its weights is unaffected by the second pass.
ORCA_DIRECTION = os.environ.get("ORCA_DIRECTION", "")
if ORCA_DIRECTION:
    ORCA_FN = f'''

# --- {MARK}:orca-proj ---
def _orca_direction():
    from safetensors.torch import load_file as _load
    _r = _load({ORCA_DIRECTION!r})["orca_refusal_direction"].to(torch.float32)
    return (_r / _r.norm()).to(torch.get_default_device())


def _orca_project(x, r):
    c = x.float() @ r
    return x - (c.unsqueeze(-1) * r).to(x.dtype)
'''

    def _orca(s):
        s = once(s, '''        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            workspace=workspace,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )
''', '''        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            workspace=workspace,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )
        self.register_buffer("_orca_r", _orca_direction(), persistent=False)  # %s:orca-proj
''' % MARK)
        if s is None:
            return None
        s = once(s, '''        if output_indices is not None:
            hidden_states = hidden_states[output_indices]
            attn_out = attn_out[output_indices]
''', '''        attn_out = _orca_project(attn_out, self._orca_r)  # %s:orca-proj
        if output_indices is not None:
            hidden_states = hidden_states[output_indices]
            attn_out = attn_out[output_indices]
''' % MARK)
        if s is None:
            return None
        s = once(s, '''        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection
''', '''        mlp_out = _orca_project(self.mlp(block_input), self._orca_r)  # %s:orca-proj
        return hidden_states, mlp_out, injection
''' % MARK)
        return None if s is None else s + ORCA_FN
    edit(f"{MODEL_DIR}/model.py", _orca, "orca-proj", True)

print("patched: " + ", ".join(done) + ("" if not unneeded else " | not needed on this image: " + ", ".join(unneeded))
      + ("" if not warn else " | WARNING (optional, skipped): " + "; ".join(warn)))
if fatal:
    sys.exit("FATAL required patch failed: " + "; ".join(fatal))
