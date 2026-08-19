Title: Format de Recipe

URL Source: https://sparkrun.dev/recipes/format/

Markdown Content:
Une recipe sparkrun est un fichier YAML qui décrit tout ce qu'il faut pour lancer un workload d'inférence : le modèle, l'image de container, l'engine de runtime, les paramètres par défaut, et la commande de service. Les recipes sont l'abstraction centrale dans sparkrun — elles vous permettent de capturer une configuration connue-bonne une fois et de la rejouer avec une seule commande.

`sparkrun run my-recipe --solo          # use defaultssparkrun run my-recipe -H host1,host2  # override hostssparkrun run my-recipe -o port=9000    # override any default`

La plus petite recipe utile n'a besoin que de `model`, `runtime`, `container`, et `command` :

`model: Qwen/Qwen3-1.7Bruntime: vllmcontainer: scitrera/dgx-spark-vllm:0.16.0-t5defaults:  port: 8000  host: 0.0.0.0command: |  vllm serve {model} --host {host} --port {port}`

## Exemple de recipe complet

[Section titled “Full recipe example”](https://sparkrun.dev/recipes/format/#full-recipe-example)

`model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4model_revision: abc123defruntime: vllmmin_nodes: 1container: scitrera/dgx-spark-vllm:0.16.0-t5metadata:  description: NVIDIA Nemotron 3 Nano 30B (upstream NVFP4) -- cluster or solo  maintainer: scitrera.ai <open-source-team@scitrera.com>  model_params: 30B  model_dtype: nvfp4defaults:  port: 8000  host: 0.0.0.0  tensor_parallel: 1  gpu_memory_utilization: 0.8  max_model_len: 200000  served_model_name: nemotron3-30b-a3b  tool_call_parser: qwen3_coderenv:  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"command: |  vllm serve \      {model} \      --served-model-name {served_model_name} \      --max-model-len {max_model_len} \      --gpu-memory-utilization {gpu_memory_utilization} \      -tp {tensor_parallel} \      --host {host} \      --port {port} \      --enable-auto-tool-choice \      --tool-call-parser {tool_call_parser} \      --trust-remote-code`

L'identifiant de modèle HuggingFace à servir.

`# Standard HuggingFace modelmodel: Qwen/Qwen3-1.7B# GGUF model with quantization variant (llama-cpp runtime)model: Qwen/Qwen3-1.7B-GGUF:Q8_0`

Pour les modèles GGUF, la syntaxe à deux-points (`repo:quant`) sélectionne une variante de quantification spécifique. sparkrun l'utilise pour ne télécharger que les fichiers quant correspondants plutôt que tout le dépôt.

La valeur de `model` est injectée dans le template de commande comme `{model}` et est aussi utilisée pour le pre-sync du modèle (télécharger et distribuer les poids vers les hôtes cibles avant le lancement).

Épinglez le modèle sur une révision HuggingFace spécifique (branche, tag, ou hash de commit).

`model: QuantTrio/MiniMax-M2.5-AWQmodel_revision: bbe738792c`

Cela affecte le téléchargement du modèle, la vérification de cache, l'estimation VRAM, et le sync du modèle. Utilisez-le pour des déploiements reproductibles — sans cela, la branche `main` d'un modèle peut changer entre les téléchargements.

Quel engine d'inférence utiliser :

| Valeur | Engine | Clustering | Notes |
| --- | --- | --- | --- |
| `vllm` | vLLM | varies | Virtual alias — resolves to `vllm-distributed` (default) or `vllm-ray`. See [vLLM runtime](https://sparkrun.dev/runtimes/vllm/). |
| `vllm-distributed` | vLLM | Native | Default vLLM variant. Uses vLLM's built-in distributed backend. |
| `vllm-ray` | vLLM | Ray | vLLM with Ray head/worker orchestration. |
| `sglang` | SGLang | Native | First-class. Solo and multi-node. |
| `llama-cpp` | llama.cpp | N/A | Solo mode. GGUF models. |
| `trtllm` | TensorRT-LLM | MPI | NVIDIA TensorRT-LLM with MPI-based multi-node orchestration. |

Définir `runtime` explicitement est **recommandé** pour la clarté et la
forward-compatibility. Cependant, ce champ est techniquement optionnel —
lorsqu'il est omis, sparkrun utilise la
[détection automatique de runtime](https://sparkrun.dev/recipes/format/#automatic-runtime-detection)
pour inférer le runtime depuis le champ `command`. Si ni `runtime` ni
une commande reconnaissable n'est présent, la recipe se rabat sur le
comportement `vllm`.

### `container` (requis)

[Section titled “container (required)”](https://sparkrun.dev/recipes/format/#container-required)

L'image de container Docker/OCI à exécuter.

`container: scitrera/dgx-spark-vllm:0.16.0-t5container: scitrera/dgx-spark-sglang:0.5.8-t5container: scitrera/dgx-spark-llama-cpp:b8076-cu131`

### `command` (recommandé)

[Section titled “command (recommended)”](https://sparkrun.dev/recipes/format/#command-recommended)

La commande shell à exécuter dans le container. Utilise la syntaxe `{placeholder}` — toute clé depuis `defaults` (ou des overrides CLI) peut être référencée.

`command: |  vllm serve \      {model} \      --served-model-name {served_model_name} \      --host {host} \      --port {port}`

La substitution est itérative : si un placeholder se résout vers une chaîne contenant un autre `{placeholder}`, il sera étendu dans une seconde passe.

Les nouvelles recipes devraient définir `min_nodes` et `max_nodes` directement — ils sont explicites, se composent proprement avec les overrides CLI, et couvrent tous les cas que les anciens flags faisaient.

Nombre minimum de nœuds requis. Défaut à `1`.

Sur DGX Spark, chaque nœud a un GPU, donc `min_nodes` est en pratique le nombre minimum de GPUs.

Nombre maximum de nœuds supportés. Défaut à illimité.

Mettez `max_nodes: 1` pour les modèles ou runtimes qui ne supportent pas l'inférence multi-nœud.

### Déprécié : `mode`, `solo_only`, `cluster_only`

[Section titled “Deprecated: mode, solo_only, cluster_only”](https://sparkrun.dev/recipes/format/#deprecated-mode-solo_only-cluster_only)

`mode` — mode de topologie explicite (`auto`, `solo`, ou `cluster`). Inféré depuis `min_nodes` / `max_nodes` quand omis.

| Valeur | Signification |
| --- | --- |
| `auto` | (défaut) sparkrun décide selon les counts de nœuds et les flags CLI |
| `solo` | Force single-node. Définit `min_nodes = max_nodes = 1`. |
| `cluster` | Force multi-node. Requiert 2+ hôtes. |

`solo_only` / `cluster_only` — raccourcis booléens pour ce qui précède :

`solo_only: true     # equivalent to max_nodes: 1, mode: solocluster_only: true  # equivalent to min_nodes: 2, mode: cluster`

## Champs de configuration

[Section titled “Configuration fields”](https://sparkrun.dev/recipes/format/#configuration-fields)

Un dictionnaire plat de valeurs de paramètres par défaut. Chaque clé est disponible comme `{placeholder}` dans le template de commande et peut être overridée au lancement via des flags CLI ou `-o key=value`.

`defaults:  port: 8000  host: 0.0.0.0  tensor_parallel: 1  gpu_memory_utilization: 0.8  max_model_len: 200000  served_model_name: nemotron3-30b-a3b`

#### Défauts standardisés

[Section titled “Standardized defaults”](https://sparkrun.dev/recipes/format/#standardized-defaults)

Les clés suivantes sont reconnues par tous les runtimes et automatiquement mappées vers les bons flags runtime-spécifiques. Les nouveaux runtimes sont censés supporter ces clés pour la compatibilité cross-runtime.

| Clé de défaut | Flag CLI | Description |
| --- | --- | --- |
| `port` | `--port` | Serve port |
| `host` | `-o host=X` | Bind address |
| `tensor_parallel` | `--tp` | Tensor parallelism degree |
| `gpu_memory_utilization` | `--gpu-mem` | GPU memory fraction (0.0–1.0) |
| `max_model_len` | `--max-model-len` | Maximum sequence length |
| `served_model_name` | `--served-model-name` | Model name exposed by the API (works for all runtimes) |

Toute autre clé peut apparaître dans defaults — il n'y a pas de schéma fixe. Les paramètres runtime-spécifiques (comme `tool_call_parser`, `attention_backend`, etc.) sont passés comme valeurs `{placeholder}` pour utilisation dans le template de commande.

**Précédence de la chaîne de config** (le plus haut gagne) :

1.   Overrides CLI (`--port 9000`, `--served-model-name my-model`, `-o key=value`)
2.   Défauts de la recipe

Variables d'environnement injectées dans le container.

`env:  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"  HF_TOKEN: "${HF_TOKEN}"`

Les références de variables shell (`${VAR}`) sont étendues depuis l'environnement de la **machine de contrôle** quand la recipe est chargée — transférez les secrets sans les hardcoder.

Champs descriptifs et d'estimation VRAM. N'affecte pas comment le workload tourne.

`metadata:  description: NVIDIA Nemotron 3 Nano 30B (upstream NVFP4)  maintainer: scitrera.ai <open-source-team@scitrera.com>  model_params: 30B  model_dtype: nvfp4`

#### Champs d'estimation VRAM (optionnel)

[Section titled “VRAM estimation fields (optional)”](https://sparkrun.dev/recipes/format/#vram-estimation-fields-optional)

**Préférez l'auto-détection.** sparkrun tire les counts de paramètres, les
counts de layers, les dimensions de head, et les dtypes depuis la config
HuggingFace de chaque modèle — pour l'écrasante majorité des recipes vous
n'avez pas besoin de définir ces champs. Ils existent comme **overrides
manuels** pour les rares cas où l'auto-detect échoue ou renvoie des
chiffres trompeurs, par ex. des variantes lourdement quantifiées, des
topologies MoE custom, ou des dépôts avec une metadata `config.json`
manquante/atypique.

Quand vous devez override, les valeurs dans `metadata` prennent la
précédence sur tout ce que sparkrun infère depuis le modèle lui-même.

| Champ | Type | Description |
| --- | --- | --- |
| `model_params` | string | Parameter count: `"1.7B"`, `"30B"`, `"397B"` |
| `model_dtype` | string | Weight dtype: `bf16`, `fp16`, `fp8`, `nvfp4`, `int4`, `q8_0`, etc. |
| `kv_dtype` | string | KV cache dtype (defaults to `bfloat16`) |
| `num_layers` | int | Number of transformer layers |
| `num_kv_heads` | int | Number of key-value attention heads |
| `head_dim` | int | Dimension per attention head |
| `model_vram` | float | Override: total model weight VRAM in GB |
| `kv_vram_per_token` | float | Override: KV cache bytes per token |

Utilisez `sparkrun show <recipe>` ou `sparkrun recipe vram <recipe>`
pour voir ce que l'auto-détection rapporte actuellement — commencez là
et ne remplissez les champs metadata que quand l'estimation affichée
est fausse.

Un **mod** est un répertoire nommé contenant un script `run.sh` (plus
n'importe quels fichiers de support — patches, templates, snippets de
setup env, etc.) qui est copié dans le container et exécuté *avant* la
commande de service. Les mods sont un mécanisme builder/runtime-agnostic
pour des tweaks de container petits et nommés (par ex. appliquer un
plugin parser Nemotron, patcher un module Python, déposer une config
custom).

Introduits à l'origine pour `eugr/spark-vllm-docker`, les mods sont
maintenant first-class dans sparkrun : n'importe quelle recipe peut les
déclarer, n'importe quel registry peut les publier, et n'importe quel
runtime les voit comme de simples entrées `pre_exec` après la
résolution Phase 1. Voir
[Execution Flow › Phase 1](https://sparkrun.dev/developer-reference/execution-flow/#phase-1-preparing).

`mods` est un champ liste **top-level** sur la recipe :

`model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4runtime: vllmcontainer: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latestmods:  - mods/nemotron-nano          # resolved against the recipe's location / registry  - @eugr/mods/qwen3-tools      # explicitly scoped to the @eugr registry  - my-private-tweak            # short form — leading "mods/" is optional`

Le préfixe `mods/` de tête est purement une convention de nommage et
est strippé pendant la résolution, donc `nemotron-nano` et
`mods/nemotron-nano` sont équivalents.

Pour chaque entrée dans `mods:`, sparkrun parcourt quatre étapes dans
l'ordre et utilise le premier hit :

| # | Étape | Ce qu'elle cherche |
| --- | --- | --- |
| 1 | **Explicit scoped reference** (`@registry/<rel>`) | Resolves `<rel>` under that registry's configured `mods_subpath`. Bypasses every other stage when the user opts in explicitly. |
| 2 | **Adjacent to the recipe file** | Tries `<rel>` and `mods/<rel>` relative to the directory of the recipe YAML — works for local recipes and recipes shipped inside a registry repo. |
| 3 | **Same registry as the recipe** | If the recipe came from a registry, tries `<mods_subpath>/<rel>` inside that registry (using the registry's configured `mods_subpath`). |
| 4 | **`@eugr` fallback** | Falls back to the `eugr` registry — the original source of truth for mods, kept as a safety net for legacy recipes and shared community mods. |

Une source de mod est valide quand le répertoire cible contient un
`run.sh`. Si aucune des quatre étapes ne trouve de match, `sparkrun
run` échoue bruyamment avec la liste des paths qu'il a essayés.

**Mod local adjacent à une recipe locale** (étape 2) :

`./my-recipe.yaml./mods/fix-config/run.sh`

`mods:  - fix-config`

Résolu via étape 2 — `dirname(my-recipe.yaml)/mods/fix-config/run.sh`.
Pas de registry nécessaire.

**Mod depuis le même registry que la recipe** (étape 3) :

Un repo de registry à `github.com/myteam/spark-recipes` ship :

`recipes/qwen-tools.yamlmods/qwen3-tool-parser/run.sh`

…et son `.sparkrun/registry.yaml` déclare `mods: mods`. Alors
`qwen-tools.yaml` peut référencer :

`mods:  - qwen3-tool-parser`

…et sparkrun le trouve dans le `mods_subpath` du même registry
(étape 3 — en supposant qu'il n'est pas adjacent à l'étape 2).

**Référence cross-registry avec `@registry/`** (étape 1) :

`mods:  - @official/nemotron-nano        # pulled from the @official registry's mods_subpath  - @eugr/mods/qwen3-coder         # pulled from @eugr, with explicit "mods/" prefix`

Utile pour mixer-et-assortir des mods de registries qui ne possèdent
pas la recipe.

**Fallback legacy vers `@eugr`** (étape 4) :

Une recipe écrite contre le workflow eugr original qui liste juste
`mods: [nemotron-nano]` (pas de scope de registry, pas adjacent, pas
dans son propre registry) se résout toujours — sparkrun atteint
l'étape 4 et cherche `nemotron-nano` dans le `mods_subpath` du
registry `@eugr`. Cela garde les anciennes recipes v1 fonctionnant
sans modification.

## Champs runtime-spécifiques

[Section titled “Runtime-specific fields”](https://sparkrun.dev/recipes/format/#runtime-specific-fields)

Un dictionnaire pour la configuration runtime-spécifique :

`runtime_config:  build_args: [--some-flag]`

Les clés top-level inconnues sont automatiquement swept dans
`runtime_config`.

## Détection automatique de runtime

[Section titled “Automatic runtime detection”](https://sparkrun.dev/recipes/format/#automatic-runtime-detection)

Bien que définir `runtime` explicitement soit recommandé, sparkrun peut
inférer le runtime depuis le champ `command` quand il est omis. C'est
utile pour de l'expérimentation rapide ou quand on adapte des snippets
de commande depuis la documentation sans se soucier de quelle valeur de
runtime définir.

| Préfixe de commande | Runtime détecté |
| --- | --- |
| `vllm serve ...` | `vllm` (then resolved to `vllm-distributed` or `vllm-ray`) |
| `sglang serve ...` | `sglang` |
| `python -m sglang.launch_server ...` | `sglang` |
| `python3 -m sglang.launch_server ...` | `sglang` |
| `llama-server ...` | `llama-cpp` |

Si la commande ne matche aucun de ces patterns, la recipe retombe sur
le comportement `vllm` par défaut.

Un champ `runtime` explicite **gagne toujours** — la détection par
hint de commande ne fire que quand `runtime` est omis. Les recipes
existantes qui définissent `runtime` continuent de fonctionner de
manière identique.

## Configuration de benchmark

[Section titled “Benchmark configuration”](https://sparkrun.dev/recipes/format/#benchmark-configuration)

Les recipes peuvent inclure un bloc `benchmark:` pour définir les
paramètres de benchmark par défaut utilisés par `sparkrun benchmark` :

`benchmark:  framework: llama-benchy  pp: [2048]  depth: [0, 4096]  prefix_caching: true`

Voir la [commande benchmark](https://sparkrun.dev/cli/benchmark/) pour
les détails sur comment la configuration de benchmark est résolue.

sparkrun cherche les recipes dans cet ordre :

1.   **Shortcut `@spark-arena/`** — `@spark-arena/UUID` s'étend vers une URL Spark Arena.
2.   **URL** — si l'argument est une URL HTTP/HTTPS, la recipe est fetchée directement et cachée.
3.   **Lookup scoped `@registry/recipe-name`** — disambiguate les recipes à travers les registries en utilisant la syntaxe scoped.
4.   **Chemin de fichier exact/relatif** — si l'argument est un chemin vers un fichier existant.
5.   **Répertoire de travail courant** — sparkrun scanne les fichiers `.yaml`/`.yml` dans le CWD qui sont des recipes valides.
6.   **Recherche de registry** — lookup de nom flat dans les registries custom configurés, puis recursive glob.

Les noms de fichiers sont matchés avec ou sans extensions `.yaml`/`.yml`.

## Substitution de template de commande

[Section titled “Command template substitution”](https://sparkrun.dev/recipes/format/#command-template-substitution)

Le champ `command` supporte la substitution `{placeholder}` depuis la chaîne de config :

`defaults:  port: 8000  served_model_name: my-modelcommand: |  vllm serve {model} --port {port} --served-model-name {served_model_name}`

Avec `sparkrun run recipe -o port=9000`, cela rend :

`vllm serve Qwen/Qwen3-1.7B --port 9000 --served-model-name my-model`

Le placeholder spécial `{model}` est toujours disponible depuis le champ top-level `model`.
