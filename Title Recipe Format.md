Title: Recipe Format

URL Source: https://sparkrun.dev/recipes/format/

Markdown Content:
A sparkrun recipe is a YAML file that describes everything needed to launch an inference workload: the model, the container image, the runtime engine, default parameters, and the serve command. Recipes are the central abstraction in sparkrun — they let you capture a known-good configuration once and replay it with a single command.

`sparkrun run my-recipe --solo          # use defaultssparkrun run my-recipe -H host1,host2  # override hostssparkrun run my-recipe -o port=9000    # override any default`

The smallest useful recipe needs only `model`, `runtime`, `container`, and `command`:

`model: Qwen/Qwen3-1.7Bruntime: vllmcontainer: scitrera/dgx-spark-vllm:0.16.0-t5defaults:  port: 8000  host: 0.0.0.0command: |  vllm serve {model} --host {host} --port {port}`

## Full recipe example

[Section titled “Full recipe example”](https://sparkrun.dev/recipes/format/#full-recipe-example)

`model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4model_revision: abc123defruntime: vllmmin_nodes: 1container: scitrera/dgx-spark-vllm:0.16.0-t5metadata:  description: NVIDIA Nemotron 3 Nano 30B (upstream NVFP4) -- cluster or solo  maintainer: scitrera.ai <open-source-team@scitrera.com>  model_params: 30B  model_dtype: nvfp4defaults:  port: 8000  host: 0.0.0.0  tensor_parallel: 1  gpu_memory_utilization: 0.8  max_model_len: 200000  served_model_name: nemotron3-30b-a3b  tool_call_parser: qwen3_coderenv:  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"command: |  vllm serve \      {model} \      --served-model-name {served_model_name} \      --max-model-len {max_model_len} \      --gpu-memory-utilization {gpu_memory_utilization} \      -tp {tensor_parallel} \      --host {host} \      --port {port} \      --enable-auto-tool-choice \      --tool-call-parser {tool_call_parser} \      --trust-remote-code`

The HuggingFace model identifier to serve.

`# Standard HuggingFace modelmodel: Qwen/Qwen3-1.7B# GGUF model with quantization variant (llama-cpp runtime)model: Qwen/Qwen3-1.7B-GGUF:Q8_0`

For GGUF models, the colon syntax (`repo:quant`) selects a specific quantization variant. sparkrun uses this to download only the matching quant files rather than the entire repository.

The model value is injected into the command template as `{model}` and is also used for model pre-sync (downloading and distributing weights to target hosts before launch).

Pin the model to a specific HuggingFace revision (branch, tag, or commit hash).

`model: QuantTrio/MiniMax-M2.5-AWQmodel_revision: bbe738792c`

This affects model download, cache checking, VRAM estimation, and model sync. Use it for reproducible deployments — without it, a model’s `main` branch may change between downloads.

Which inference engine to use:

| Value | Engine | Clustering | Notes |
| --- | --- | --- | --- |
| `vllm` | vLLM | varies | Virtual alias — resolves to `vllm-distributed` (default) or `vllm-ray`. See [vLLM runtime](https://sparkrun.dev/runtimes/vllm/). |
| `vllm-distributed` | vLLM | Native | Default vLLM variant. Uses vLLM’s built-in distributed backend. |
| `vllm-ray` | vLLM | Ray | vLLM with Ray head/worker orchestration. |
| `sglang` | SGLang | Native | First-class. Solo and multi-node. |
| `llama-cpp` | llama.cpp | N/A | Solo mode. GGUF models. |
| `trtllm` | TensorRT-LLM | MPI | NVIDIA TensorRT-LLM with MPI-based multi-node orchestration. |

Explicitly setting `runtime` is **recommended** for clarity and forward-compatibility. However, this field is technically optional — when omitted, sparkrun uses [automatic runtime detection](https://sparkrun.dev/recipes/format/#automatic-runtime-detection) to infer the runtime from the `command` field. If neither `runtime` nor a recognizable command is present, the recipe defaults to `vllm` behavior.

### `container` (required)

[Section titled “container (required)”](https://sparkrun.dev/recipes/format/#container-required)

The Docker/OCI container image to run.

`container: scitrera/dgx-spark-vllm:0.16.0-t5container: scitrera/dgx-spark-sglang:0.5.8-t5container: scitrera/dgx-spark-llama-cpp:b8076-cu131`

### `command` (recommended)

[Section titled “command (recommended)”](https://sparkrun.dev/recipes/format/#command-recommended)

The shell command to execute inside the container. Uses `{placeholder}` syntax — any key from `defaults` (or CLI overrides) can be referenced.

`command: |  vllm serve \      {model} \      --served-model-name {served_model_name} \      --host {host} \      --port {port}`

Substitution is iterative: if a placeholder resolves to a string containing another `{placeholder}`, it will be expanded in a second pass.

New recipes should set `min_nodes` and `max_nodes` directly — they’re explicit, compose cleanly with CLI overrides, and cover every case the older flags do.

Minimum number of nodes required. Defaults to `1`.

On DGX Spark, each node has one GPU, so `min_nodes` is effectively the minimum GPU count.

Maximum number of nodes supported. Defaults to unlimited.

Set `max_nodes: 1` for models or runtimes that do not support multi-node inference.

### Deprecated: `mode`, `solo_only`, `cluster_only`

[Section titled “Deprecated: mode, solo_only, cluster_only”](https://sparkrun.dev/recipes/format/#deprecated-mode-solo_only-cluster_only)

`mode` — explicit topology mode (`auto`, `solo`, or `cluster`). Inferred from `min_nodes` / `max_nodes` when omitted.

| Value | Meaning |
| --- | --- |
| `auto` | (default) sparkrun decides based on node counts and CLI flags |
| `solo` | Forces single-node. Sets `min_nodes = max_nodes = 1`. |
| `cluster` | Forces multi-node. Requires 2+ hosts. |

`solo_only` / `cluster_only` — boolean shorthand for the above:

`solo_only: true     # equivalent to max_nodes: 1, mode: solocluster_only: true  # equivalent to min_nodes: 2, mode: cluster`

## Configuration fields

[Section titled “Configuration fields”](https://sparkrun.dev/recipes/format/#configuration-fields)

A flat dictionary of default parameter values. Every key is available as a `{placeholder}` in the command template and can be overridden at launch via CLI flags or `-o key=value`.

`defaults:  port: 8000  host: 0.0.0.0  tensor_parallel: 1  gpu_memory_utilization: 0.8  max_model_len: 200000  served_model_name: nemotron3-30b-a3b`

#### Standardized defaults

[Section titled “Standardized defaults”](https://sparkrun.dev/recipes/format/#standardized-defaults)

The following keys are recognized by all runtimes and are automatically mapped to the correct runtime-specific flags. New runtimes are expected to support these keys for cross-runtime compatibility.

| Default key | CLI flag | Description |
| --- | --- | --- |
| `port` | `--port` | Serve port |
| `host` | `-o host=X` | Bind address |
| `tensor_parallel` | `--tp` | Tensor parallelism degree |
| `gpu_memory_utilization` | `--gpu-mem` | GPU memory fraction (0.0–1.0) |
| `max_model_len` | `--max-model-len` | Maximum sequence length |
| `served_model_name` | `--served-model-name` | Model name exposed by the API (works for all runtimes) |

Any other key can appear in defaults — there is no fixed schema. Runtime-specific parameters (like `tool_call_parser`, `attention_backend`, etc.) are passed through as `{placeholder}` values for use in the command template.

**Config chain precedence** (highest wins):

1.   CLI overrides (`--port 9000`, `--served-model-name my-model`, `-o key=value`)
2.   Recipe defaults

Environment variables injected into the container.

`env:  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"  HF_TOKEN: "${HF_TOKEN}"`

Shell variable references (`${VAR}`) are expanded from the **control machine’s** environment when the recipe is loaded — forward secrets without hardcoding them.

Descriptive and VRAM-estimation fields. Does not affect how the workload runs.

`metadata:  description: NVIDIA Nemotron 3 Nano 30B (upstream NVFP4)  maintainer: scitrera.ai <open-source-team@scitrera.com>  model_params: 30B  model_dtype: nvfp4`

#### VRAM estimation fields (optional)

[Section titled “VRAM estimation fields (optional)”](https://sparkrun.dev/recipes/format/#vram-estimation-fields-optional)

**Prefer auto-detection.** sparkrun pulls parameter counts, layer counts, head dimensions, and dtypes from each model’s HuggingFace config — for the overwhelming majority of recipes you don’t need to set any of these fields. They exist as **manual overrides** for the rare cases where auto-detect fails or returns misleading numbers, e.g., heavily-quantized variants, custom MoE topologies, or repos with missing/atypical `config.json` metadata.

When you do need to override, the values in `metadata` take precedence over anything sparkrun infers from the model itself.

| Field | Type | Description |
| --- | --- | --- |
| `model_params` | string | Parameter count: `"1.7B"`, `"30B"`, `"397B"` |
| `model_dtype` | string | Weight dtype: `bf16`, `fp16`, `fp8`, `nvfp4`, `int4`, `q8_0`, etc. |
| `kv_dtype` | string | KV cache dtype (defaults to `bfloat16`) |
| `num_layers` | int | Number of transformer layers |
| `num_kv_heads` | int | Number of key-value attention heads |
| `head_dim` | int | Dimension per attention head |
| `model_vram` | float | Override: total model weight VRAM in GB |
| `kv_vram_per_token` | float | Override: KV cache bytes per token |

Use `sparkrun show <recipe>` or `sparkrun recipe vram <recipe>` to see what auto-detection currently reports — start there and only fill in metadata fields when the displayed estimate is off.

A **mod** is a named directory containing a `run.sh` script (plus any supporting files — patches, templates, env-setup snippets, etc.) that gets copied into the container and executed _before_ the serve command. Mods are a builder/runtime-agnostic mechanism for small, named container tweaks (e.g., applying a Nemotron parser plugin, patching a Python module, dropping in custom config).

Originally introduced for `eugr/spark-vllm-docker`, mods are now first-class in sparkrun: any recipe can declare them, any registry can publish them, and any runtime sees them as plain `pre_exec` entries after Phase 1 resolution. See [Execution Flow › Phase 1](https://sparkrun.dev/developer-reference/execution-flow/#phase-1-preparing).

`mods` is a **top-level** list field on the recipe:

`model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4runtime: vllmcontainer: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latestmods:  - mods/nemotron-nano          # resolved against the recipe's location / registry  - @eugr/mods/qwen3-tools      # explicitly scoped to the @eugr registry  - my-private-tweak            # short form — leading "mods/" is optional`

The leading `mods/` prefix is purely a naming convention and is stripped during resolution, so `nemotron-nano` and `mods/nemotron-nano` are equivalent.

For each entry in `mods:`, sparkrun walks four stages in order and uses the first hit:

| # | Stage | What it looks for |
| --- | --- | --- |
| 1 | **Explicit scoped reference** (`@registry/<rel>`) | Resolves `<rel>` under that registry’s configured `mods_subpath`. Bypasses every other stage when the user opts in explicitly. |
| 2 | **Adjacent to the recipe file** | Tries `<rel>` and `mods/<rel>` relative to the directory of the recipe YAML — works for local recipes and recipes shipped inside a registry repo. |
| 3 | **Same registry as the recipe** | If the recipe came from a registry, tries `<mods_subpath>/<rel>` inside that registry (using the registry’s configured `mods_subpath`). |
| 4 | **`@eugr` fallback** | Falls back to the `eugr` registry — the original source of truth for mods, kept as a safety net for legacy recipes and shared community mods. |

A mod source is valid when the target directory contains a `run.sh`. If none of the four stages find a match, `sparkrun run` fails loudly with the list of paths it tried.

**Local mod adjacent to a local recipe** (stage 2):

`./my-recipe.yaml./mods/fix-config/run.sh`

`mods:  - fix-config`

Resolved via stage 2 — `dirname(my-recipe.yaml)/mods/fix-config/run.sh`. No registry needed.

**Mod from the same registry as the recipe** (stage 3):

A registry repo at `github.com/myteam/spark-recipes` ships:

`recipes/qwen-tools.yamlmods/qwen3-tool-parser/run.sh`

…and its `.sparkrun/registry.yaml` declares `mods: mods`. Then `qwen-tools.yaml` can reference:

`mods:  - qwen3-tool-parser`

…and sparkrun finds it inside the same registry’s `mods_subpath` (stage 3 — assuming it’s not adjacent at stage 2).

**Cross-registry reference with `@registry/`** (stage 1):

`mods:  - @official/nemotron-nano        # pulled from the @official registry's mods_subpath  - @eugr/mods/qwen3-coder         # pulled from @eugr, with explicit "mods/" prefix`

Useful for mixing-and-matching mods from registries that don’t own the recipe.

**Legacy fallback to `@eugr`** (stage 4):

A recipe authored against the original eugr workflow that just lists `mods: [nemotron-nano]` (no registry scope, not adjacent, not in its own registry) still resolves — sparkrun reaches stage 4 and looks up `nemotron-nano` inside the `@eugr` registry’s `mods_subpath`. This keeps older v1 recipes working without modification.

## Runtime-specific fields

[Section titled “Runtime-specific fields”](https://sparkrun.dev/recipes/format/#runtime-specific-fields)

A dictionary for runtime-specific configuration:

`runtime_config:  build_args: [--some-flag]`

Unknown top-level keys are automatically swept into `runtime_config`.

## Automatic runtime detection

[Section titled “Automatic runtime detection”](https://sparkrun.dev/recipes/format/#automatic-runtime-detection)

While explicitly setting `runtime` is recommended, sparkrun can infer the runtime from the `command` field when it is omitted. This is useful for quick experimentation or when adapting command snippets from documentation without worrying about which runtime value to set.

| Command prefix | Detected runtime |
| --- | --- |
| `vllm serve ...` | `vllm` (then resolved to `vllm-distributed` or `vllm-ray`) |
| `sglang serve ...` | `sglang` |
| `python -m sglang.launch_server ...` | `sglang` |
| `python3 -m sglang.launch_server ...` | `sglang` |
| `llama-server ...` | `llama-cpp` |

If the command doesn’t match any of these patterns, the recipe falls through to `vllm` behavior by default.

An explicit `runtime` field **always wins** — command-hint detection only fires when `runtime` is omitted. Existing recipes that set `runtime` continue to work identically.

## Benchmark configuration

[Section titled “Benchmark configuration”](https://sparkrun.dev/recipes/format/#benchmark-configuration)

Recipes can include a `benchmark:` block to define default benchmark settings used by `sparkrun benchmark`:

`benchmark:  framework: llama-benchy  pp: [2048]  depth: [0, 4096]  prefix_caching: true`

See the [benchmark command](https://sparkrun.dev/cli/benchmark/) for details on how benchmark configuration is resolved.

sparkrun searches for recipes in this order:

1.   **`@spark-arena/` shortcut** — `@spark-arena/UUID` expands to a Spark Arena URL.
2.   **URL** — if the argument is an HTTP/HTTPS URL, the recipe is fetched directly and cached.
3.   **`@registry/recipe-name` scoped lookup** — disambiguate recipes across registries using scoped syntax.
4.   **Exact/relative file path** — if the argument is a path to an existing file.
5.   **Current working directory** — sparkrun scans `.yaml`/`.yml` files in the CWD that are valid recipes.
6.   **Registry search** — flat name lookup in configured custom registries, then recursive glob.

Filenames are matched with or without `.yaml`/`.yml` extensions.

## Command template substitution

[Section titled “Command template substitution”](https://sparkrun.dev/recipes/format/#command-template-substitution)

The `command` field supports `{placeholder}` substitution from the config chain:

`defaults:  port: 8000  served_model_name: my-modelcommand: |  vllm serve {model} --port {port} --served-model-name {served_model_name}`

With `sparkrun run recipe -o port=9000`, this renders as:

`vllm serve Qwen/Qwen3-1.7B --port 9000 --served-model-name my-model`

The special placeholder `{model}` is always available from the top-level `model` field.
