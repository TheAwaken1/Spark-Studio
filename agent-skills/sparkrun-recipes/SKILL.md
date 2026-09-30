---
name: sparkrun-recipes
description: Create, repair, review, and optimize sparkrun inference recipe YAML for DGX Spark.
---

# sparkrun recipes

Use this skill whenever a task creates, fixes, tunes, or evaluates a sparkrun
recipe or translates a model launch command into recipe YAML.

## Reference docs

- `${HERMES_SKILL_DIR}/../../Title Recipe Format.md` (repo root)

## Quick start

Before changing a recipe, read the canonical local reference at:

    `${HERMES_SKILL_DIR}/../../Title Recipe Format.md`

Then:

1. Inspect the existing recipe, registry conventions, and the active model or
   failure evidence before proposing changes.
2. Prefer explicit `runtime`, `min_nodes`, and `max_nodes`; preserve valid
   registry-specific conventions when they intentionally differ.
3. Keep overridable values in `defaults` and reference them with command
   placeholders instead of duplicating literals.
4. Never hardcode credentials. Use environment expansion where required.
5. Validate the YAML and run the narrowest relevant recipe, launch, doctor,
   or benchmark checks available in the workspace.

## vLLM / DGX Spark pitfalls

- Unsloth NVFP4 checkpoints on GB10 (SM 12.1) need `env: CUTE_DSL_ARCH: sm_121a`.
  `moe_backend: flashinfer_b12x` is ONLY for actual MoE checkpoints (Marlin is
  ~2x slower there). Do NOT copy it onto a dense model.
- CHECK THE MODEL CLASS BEFORE COPYING A RECIPE. `unsloth/Qwen3.8-27B-NVFP4`
  (Qwen3_5ForConditionalGeneration) is a DENSE HYBRID model, NOT MoE: 64 layers =
  48 Gated DeltaNet (linear_attention) + 16 full_attention, SwiGLU MLP, zero
  experts. Its `config.json` puts the real transformer under `text_config` and
  has no `num_experts`. Cloning the qwen3.6-35b-A3B MoE recipe onto it (moe
  backend, no --quantization, default FA attn) serves at ~5 tok/s. Correct dense
  profile: `--quantization compressed-tensors` (mixed-precision checkpoint),
  `--attention-backend triton_attn` (FA2 CANNOT serve FP8 KV on GB10/SM121 —
  needs FA3 on SM90 / FA4 on SM100; only the 16 full-attn layers use it), MTP
  self-spec-decode ON (`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
  — the checkpoint ships model_mtp.safetensors; this is the ~4x decode lever,
  5->20 tok/s), and NO --moe-backend. MiaAI-Lab measures ~20 tok/s with this on
  `vllm/vllm-openai:nightly-aarch64` (vLLM >=0.25). Native context 262144; 1M via
  static YaRN factor 4.0 through --hf-overrides (slightly hurts short-context).
- CORRECTED 08-15: the eugr b12x nightly does NOT wedge on MTP for qwen3.8-27b
  on the d20260810 build (eugr/spark-vllm@sha256:2858d85c..., vLLM 0.26.1rc1,
  registers Qwen3_5MTP). It serves cleanly with MTP-3 on the NATIVE NVFP4 path:
  the engine log shows "Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM"
  + "CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8" (the checkpoint
  is MIXED-PRECISION: group_0 float-quantized FP8 + group_1 nvfp4-pack-quantized,
  quant_method=compressed-tensors — vLLM auto-detects it, do NOT pass
  --quantization or --attention-backend, let it pick). This IS the drowzeys
  config. First boot is LONG (~25 min): weight load ~300s, torch.compile ~135s,
  then `--enable-flashinfer-autotune` runs "[AutoTuner] Tuning fp4_gemm" over
  every captured graph shape (many passes) before the API server binds — the log
  sitting at KV-allocation / autotune for 10+ min is NORMAL, not wedged (results
  cache for later boots). "Not enough SMs to use max_autotune_gemm" is a benign
  GB10 warning. The old "b12x wedges, use upstream aarch64 instead" advice was a
  DIFFERENT/older build; the current champion recipe is on eugr b12x + native FP4.
 - The old "n=3 is WORSE / OOMs" note was ALSO the old path. On the native FP4
 eugr build, num_speculative_tokens=3 works (n>=4 crashes: 1 MTP layer). BUT the
 speedup is CONTENT-DEPENDENT because MTP acceptance depends on predictability.
 MEASURED single-stream on GB10 (thinking off): counting 21.5, code+tests 20.7,
 reasoning prose 13.0, creative prose 11.6 tok/s. So ~20-21 tok/s on CODE/agentic
 output (repetitive tokens draft well) but ~12-13 on novel prose (memory-bw floor,
 no config escapes it on one Spark). drowzeys' headline "31.7 tok/s" is a
 best-case MTP-friendly gen, NOT reproducible on general prose — don't quote it
 as the expected number.
 - Qwen3.8-27b BURNS TOKENS THINKING by default (reasoning_parser qwen3). "17*23"
 spends 43 completion tokens (thinking) vs 4 with thinking off, and long-context
 gets eaten "just thinking". For agentic/chat responsiveness the BIGGEST win is
 request-side "chat_template_kwargs":{"enable_thinking":false} — bigger than any
 kernel tweak. Leave thinking on only when you actually want chain-of-thought.
 - 27B DENSE ~16 tok/s IS THE GB10 CEILING — do not burn launch cycles chasing
 >20 tok/s single-stream on qwen3.8-27b (or any ~27-35B DENSE model) on ONE
 Spark. Decode is memory-bandwidth-bound: a 27B NVFP4 model reads ~14 GB/token,
 so raw decode is ~5 tok/s and MTP(2) triples it to ~16 (measured 15.5-16.5,
 peak 19). Levers that DON'T clear 20 (all empirically tried on 08-14):
 * num_speculative_tokens 2->3: WORSE. This checkpoint has ONE MTP layer;
   vLLM warns n>1 reuses it per draft position with degrading acceptance
   (pos1 already ~0.5), and n=3 also inflates spec KV scratch -> engine-init
   OOM at 262144 context.
 * --async-scheduling: INCOMPATIBLE with speculative decoding -> engine core
   init crash. Never combine the two.
 * max_num_batched_tokens 16384->8192, fastsafetensors, 64K vs 262144 context:
   all fine/beneficial for ROBUSTNESS and BOOT TIME but ZERO effect on tok/s.
 If the goal is agentic responsiveness (>20 tok/s), switch MODEL, not config:
 use an A3B MoE (~3B active/token) e.g. qwen3.6-35b-a3b-unsloth-nvfp4-fast
 (measured ~72 tok/s single-stream on this same GB10) or Nemotron-3.5-Lightning
 -30B-A3B, or a small dense (LFM2.5-2.6B). Keep 27B as the max-quality option.
 - GB10 UNIFIED-MEMORY gpu_memory_utilization is measured against FREE memory at
 launch, and ~12 GiB of the 121.63 GiB is held by the OS/other containers, so
 only ~109 GiB is free. util 0.9 -> 109.46 GiB requested -> "Free memory less
 than desired utilization" abort. Use 0.85 (~103 GiB). Separately, a 262144
 context + MTP scratch needs ~9 GiB minimum KV and intermittently fails to fit
 ("estimated maximum model length is 64000" in the error) depending on residual
 memory — set max_model_len 65536 for a deterministic boot; 64K is ample for
 chat/agentic and override to 262144 only when long context is actually needed.
- DeepSeek V4 EXL3 can hit two independent startup failures on the pinned NVIDIA 26.02 image. If TileLang imports before `flashinfer.comm`, TileLang maps its incomplete `libcudart_stub.so`; FlashInfer's `find_loaded_library("libcudart")` then reuses the stub and crashes on missing `cudaDeviceReset`. Reproduce cheaply in the pristine image with `import tilelang; import flashinfer.comm.cuda_ipc`. `LD_PRELOAD` does not fix this because FlashInfer selects the already mapped pathname. Patch `flashinfer/comm/cuda_ipc.py` in a pre-launch hook so a path ending in `libcudart_stub.so` is redirected to `/usr/local/cuda/targets/sbsa-linux/lib/libcudart.so.13`, and verify the same import order before a 17-minute model load. Separately, on driver 595.84 the old 384K + DSpark profile is kernel-OOM-killed even with 16 GiB swap. Keep the fast path at 256K with `MODE=dspark`, a 7 GiB KV pool (`7516192768` bytes), one request slot, `MAX_CUDAGRAPH_CAPTURE_SIZE=6`, `CUDAGRAPH_CAPTURE_SIZES=6`, and graph-profiler estimation disabled. Widths 12/24 consumed 4.19 GiB and caused a first-request OOM; width 6 preserves K5 single-stream speculation. The verified warmed structured result is 42.14 tok/s, with live engine telemetry at 40.7 tok/s. Do not use `MODE=mtp0` or `--enforce-eager` as a final performance profile. A 5.5 GiB pool is insufficient: vLLM reports 6.69 GiB required for 256K.
- DeepSeek V4 EXL3 384K leaves only ~1–2 GiB physical UMA free when loaded. A
  live Hermes request can produce NVIDIA `NV_ERR_NO_MEMORY`, then global OOM on
  a no-swap host. `oom_score_adj=500` correctly makes EngineCore the first OOM
  victim and protects the desktop, but does not preserve the model. The stable
  fix is persistent host swap: a 16-GiB `/swap.img` on local NVMe, mode 0600,
  enabled in `/etc/fstab`, with swappiness 60. This lets ordinary anonymous
  pages spill before CUDA needs additional physical UMA. Keep the positive OOM
  score as the last-resort host safeguard and make the 384K recipe refuse to
  launch when `/proc/swaps` has no active entry. Verified with a real 240,015-
  token prefill plus completion: 220.6s, exact response, service remained
  healthy, and no new NVIDIA/kernel OOM errors. Do not lower
  `MAX_NUM_BATCHED_TOKENS=8224`; upstream documents that it sizes the locked
  b12x MLA workspace and smaller values crash later under wide attention.
- DeepSeek V4 EXL3 384K can trigger a GLOBAL host OOM despite a fixed KV pool:
  no-swap pressure caused health checks to hang, then the kernel killed Sunshine,
  Spark Studio, Node, and the desktop portal while EngineCore survived at
  `oom_score_adj=0`. Do not assume the largest UMA process is selected first.
  Raise the model launch shell to a positive score (verified `echo 500 >
  /proc/self/oom_score_adj`, inherited by EngineCore) so a true collision kills
  the model before the desktop. Keep EarlyOOM disabled; this is kernel OOM victim
  ordering, not memory reservation. Also keep explicit KV headroom: 8.2 GiB
  (`8804682957`) yielded 422,945 tokens / 1.10x at 384K and returned ~0.6 GiB
  versus the previous 8.8-GiB pool.
 - fastsafetensors (`--load-format fastsafetensors`) cut qwen3.8-27b weight load
 from ~219s to ~83s on GB10 (GDS unsupported -> nogds mmap fallback, still much
 faster). Pure win, keep it. torch.compile still ~88s; recompiles when
 max_model_len changes (cache key includes compile range).
- BOOT CRASH on multimodal Qwen3.8-27B (Qwen3-VL vision path) with the eugr
  b12x native-FP4 build: the engine inits FULLY (weights, KV, drafter, CUDA
  graphs, "Supported tasks: ['generate']") then dies ~2s before binding the API
  in vLLM's online-renderer warmup. `renderers/base.py::_warmup_mm_processor`
  builds a dummy multimodal input at seq_len=max_model_len (262144); the
  Qwen3-VL image processor blows up in `torch.cat(processed_images)` and the
  startup watchdog fires `_interrupt_init -> KeyboardInterrupt("terminated")`,
  killing the server. FIX for a TEXT/agentic profile: set every modality limit
  to 0 (`limit_mm_per_prompt: '{"image":0,"video":0}'`) — base.py filters
  `mm_limits = {k:v if v>0}`, so an all-zero map removes the mm_processor from
  warmup entirely. Also add `--skip-mm-profiling` (VllmConfig.skip_mm_profiling
  exists in this build) as a belt-and-suspenders guard. To actually serve
  images later, restore nonzero limits AND lower max_model_len / drop
  skip-mm-profiling so the dummy mm warmup can't OOM at 262144.
- MEASURED AGAIN 08-15 (mm disabled, thinking off, MTP-3, warm caches):
  structured/counting 21.0, code+tests 18.8, reasoning prose 13.5, creative
  prose 12.0 tok/s. Reproduces the 08-14 numbers (21.5/20.7/13.0/11.6) — recipe
  is stable. Floor is ~12 (not the old broken ~5); agentic/code ~19-21. Second
  boot still ~13 min (weight load 205s, torch.compile 127s, FP4 autotune
  re-runs even with a cache file present, graph capture). Don't expect a warm
  cache to make boot instant — autotune re-profiles.
- MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark at upstream commit
  `09d4424` (2026-09-05) validates every indexed checkpoint shard before launch;
  the earlier `config.json`-only false-complete bug is fixed. Still verify the
  local checkpoint with `hf cache verify Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`
  after updating (51 files in the known-good snapshot).
- Upstream `7d0712d` (2026-09-29) fixes the packed-PLE staging width at 1,440
  bytes (the old 2,560-byte stride sent stale rows on multi-token forwards),
  hardens poisoned MTP ring-cache handling, and ships the vllm#53388 EAGLE
  block-drop backport. For long Hermes sessions set `MTP_DISABLE_BLOCK_DROP=1`:
  it preserves the last complete prefix-cache block between turns and cuts warm
  TTFT without changing target-verified output. Preserve an existing intentional
  BF16-KV/two-stream fidelity profile unless the user explicitly asks to adopt
  upstream's FP8/four-stream throughput defaults. Host unittest discovery will
  report import errors for torch/vLLM-only tests; live container startup is the
  relevant patch/engine validation, followed by `/health`, `/v1/models`, and a
  two-step typed tool-call probe through the actual client proxy.
- Its 2026-09-05 performance/safety update adds host-capped GPU budgeting
  (`HOST_RESERVE_GIB=26`, shipped `KV_TARGET_GIB=16`), complete decode graph
  widths (`CUDAGRAPH_CAPTURE_SIZES=auto`), batched PLE page prefetch, richer
  memwatch/archived logs, and optional reduced-vocabulary MTP drafting. The
  reported ~26% decode gain mostly requires a locally generated 65K token-id
  artifact via `MTP_DRAFT_VOCAB`; upstream does NOT distribute that corpus
  artifact, so an empty value honestly keeps full-vocabulary drafting. Never
  claim the full headline gain merely from pulling the repo.
- Upstream keeps `MAX_NUM_BATCHED_TOKENS=2048` as its soaked default. `8192`
  measured ~11% better prefill / ~10% lower TTFT at a ~2.6% KV-pool cost, but
  was only observed for minutes; expose it as an opt-in until locally soaked.
  Do not enable `MTP_K_SCHEDULE`: it forced PIECEWISE graphs and regressed
  single-stream step time by 24% in upstream testing.
- Flash Next can burn reasoning tokens on every Hermes tool iteration when a
  proxy defaults all Qwen3 models to thinking-on. For agent responsiveness,
  make the request-side default `chat_template_kwargs.enable_thinking=false`
  specifically for the `qwen3.8-flash-next` served alias while preserving
  explicit `/think` or caller settings; this is separate from serving speed.
- That MiaAI launcher defaults `REQUIRE_IDLE_GPU=true` and rejects even Sunshine's
  ~195 MiB desktop-encoder allocation. On a verified otherwise-idle Spark, set
  `REQUIRE_IDLE_GPU=false` only after its own live memory-budget check fits; keep
  the cgroup cap, host reserve, and memwatch safeguards enabled.
- Never put `#` comments INSIDE a `command: |` block — sparkrun renders them into
  the actual `bash -c` string, and a `\`-continued line before a comment mangles
  the command. Put rationale in YAML comments above `command:`.
- `sparkrun search <name>` / bare-name resolve only scan the CWD; run from the
  `recipes/` dir (or pass the path) — the loose file won't show from repo root.

## SGLang / DGX Spark pitfalls

- Qwen3.8 multimodal SGLang containers may auto-select CUDA IPC for multimodal
  feature transport, then crash immediately after Uvicorn starts with
  `RuntimeError: pidfd_getfd: Operation not permitted` under Docker's security
  boundary. Pass `--mm-feature-transport cpu`; this keeps multimodal processing
  available while avoiding the forbidden pooled CUDA IPC handle path.
- On GB10 unified memory, `--mem-fraction-static 0.95` can leave only ~4.5 GB
  after Qwen3.8 base + MTP + Mamba + FP8 KV allocation and terminate rank 0
  during initialization. `0.85` leaves ~14 GB through graph capture and has
  been verified with the 262K context SGLang profile.

## Making a local recipe visible to Spark Studio / Hermes TUI

Drop the YAML in `<repo>/recipes/*.yaml`. `sparkrun_service._bundled_recipes()`
globs that dir live and exposes each as `@studio/<stem>` via the running server's
`GET /api/sparkrun/recipes` (server on :7860). No registration/DB step needed —
verify with `curl -s localhost:7860/api/sparkrun/recipes | grep @studio/<stem>`.

## Custom-engine pitfalls

- A custom command under `runtime: llama-cpp` is rendered correctly, including
  the pre-synced GGUF path, but setting `served_model_name` makes sparkrun append
  llama.cpp's `--alias`. Omit that default when the custom binary does not
  support `--alias`.
- Entrpi/ds4 with the DeepSeek V4 Flash 0731 base must use the matching DSpark
  drafter explicitly. Do not pass `--preset spark`: that preset also requires
  the legacy MTP GGUF, which is incompatible with the 0731 base.
- A cached Entrpi/ds4 checkout may have a different UID from the current
  pre-exec container and make `git fetch` fail with `detected dubious ownership`.
  Before fetching, narrowly trust only the cache checkout with
  `git config --global --add safe.directory /cache/huggingface/ds4-engine`;
  do not use a wildcard safe-directory exception.
- ds4 v0.5.3 does not expose `/health` or `/healthz` (both return 404). Use
  `/v1/models` plus a real chat completion as readiness gates. Its declared
  `model_vram` is model weight size, not total UMA: 262K context buffers,
  drafter, CUDA artifacts, and filesystem cache can raise live host usage well
  above the ~88 GB model estimate.
- A Hugging Face `.locks/models--antirez--deepseek-v4-gguf` directory does not
  prove the large GGUF is cached. Inspect the actual model repository size or
  resolved snapshot file; sparkrun may need to resume a large partial download.