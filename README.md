# Spark Studio

**Your DGX Spark, one friendly dashboard.** Launch local models with one click, watch memory and logs live, chat, benchmark — and when a recipe breaks or runs slow, let Claude Code or Codex diagnose, patch, and relaunch it for you (your own Pro/Max/Plus subscription, no API keys).

Runs **vLLM**, **SGLang**, **llama.cpp**, **WebGPU (WebLLM)**, and **sparkrun** community recipes.

## Quick Start

```bash
git clone https://github.com/TheAwaken1/Spark-Studio.git
cd Spark-Studio
./start.sh
```

Open **http://127.0.0.1:7860** (or `http://<spark-ip>:7860` from any machine on your LAN). First run sets up the Python environment automatically, then a **setup wizard** checks your system, recommends a model that fits your Spark, and launches it.

```bash
./start.sh --doctor   # full system health report at any time
```

## What you can do in 60 seconds

1. Open the dashboard — the wizard checks your Spark and picks a starter model
2. Click **Launch** — watch load progress and unified-memory use live
3. **Chat** with the model the moment it's serving
4. Click **Benchmark** for tok/s + TTFT, or **Optimize Speed** to let an agent tune it
5. If anything fails, **Auto-Fix & Retry** reads the logs and patches the recipe

## Demo

<!-- TODO: capture docs/demo.gif — paste a HF model id → Forge → Run → chat -->
*Screenshots and demo GIF coming soon.*

## Features

- **First-run setup wizard** — on a fresh install the browser opens to a guided flow: full system check (`/api/doctor`), pick a goal (fast chat / best quality / coding / agents & tools / low memory), and launch a model **recommended for your hardware** — ranked from recipes proven on your Spark, models already on disk, community-validated registry recipes, and your own benchmark history. Reopen any time via **Setup Wizard** on the Overview tab
- **Beginner / Advanced mode** — a sidebar toggle. Beginner shows just the essentials (Overview, Recipes, Models, Chat, Runs & Logs); Advanced shows every engine tab, Forge, Benchmarks, and Agents. Fresh installs start in Beginner; existing setups stay Advanced
- **Dashboard UI** with dedicated tabs per engine (vLLM, SGLang, llama.cpp, WebGPU). Run lists are named by model/recipe (not hex ids), recipe editors are full Monaco editors with YAML/JSON/shell highlighting, and on phones/tablets the sidebar collapses into a slide-out menu — handy for checking vitals or logs from the couch
- **LAN-ready** — `./start.sh` binds to all interfaces so every computer on your network can use the app (no login; don't expose it to the internet)
- **Zero-CDN, offline-ready UI** — fonts (Inter / JetBrains Mono), Font Awesome icons, Monaco, Chart.js, and highlight.js are bundled under `web/vendor/` and served locally; the dashboard is fully functional on a firewalled or offline box. The browser tab shows the running engine (`▶ vllm · Spark Studio`) and the app is installable as a PWA
- **Drop-zone recipe runner** — paste or drop a JSON recipe, click Run, stream logs live
- **Run anything** — one box on the Recipes tab accepts a [spark-arena.com](https://spark-arena.com/leaderboard) benchmark link (or the whole share blurb), a recipe YAML/JSON, a HuggingFace model id, or a `@community/ref`, and just runs it. Arena imports are saved to My Recipes automatically
- **Ask Claude / Ask Codex** button on every failing run — reads the recipe + last 300 log lines, inlines the sparkrun recipe-schema reference (RECIPES.md), matching curated recipes, and fix patches from the local registry mirror, and returns a patched recipe with diagnosis
- **Auto-Fix & Retry** — one click on a failed run starts a hands-free loop: the agent diagnoses, patches the recipe, relaunches, watches the new logs, and retries with fresh context — up to 3 attempts — until the engine actually serves. No more clicking Fix over and over
- **Optimize Speed** — the same hands-free loop, but for *slow* runs instead of broken ones: one click on a healthy run benchmarks it (tok/s + TTFT), hands the agent the measured numbers, live GPU/memory vitals, and DGX Spark tuning knowledge (FlashInfer vs Marlin backends, `sparkrun tune` kernel configs, KV-cache quantization, per-family env vars from the eugr/sparkrun registries), relaunches the tuned recipe, and re-benchmarks. Whichever configuration *measured* fastest is the one left serving and saved on the recipe — a patch that benches slower is rolled back automatically. Declares victory at ≥10% improvement (`SPARK_STUDIO_OPTIMIZE_MARGIN`)
- **Recipe Forge** — paste any HuggingFace repo id; Spark Studio checks the synced recipe registry for a Spark-validated YAML, falls back to adapted registry recipes, then heuristic presets. Each result is badged so you know ground truth vs. a guess. One-click starter chips surface your recent forges and Spark-validated models from the registry
- **Hardware-aware fit check** — probes the local box via `nvidia-smi`, badges every recipe with **Fits this Spark** / **Needs N GPUs** / **Too big**
- **Compatibility check** — verdict ranges from `excellent` to `too-large` with reasons
- **Recipe library** (SQLite) — save, edit, tag, share (copy/paste as JSON between users), import/export. New recipes are created in Recipe Forge
- **sparkrun kept up to date** — `start.sh` runs `sparkrun update` on every launch (skip with `--no-sparkrun-update` or `SPARK_STUDIO_NO_SPARKRUN_UPDATE=1`), and the Community recipes toolbar has an **Update sparkrun** button with a channel picker: Stable (PyPI), Beta (develop), Alpha (bleeding edge), or YOLO (`--yolo`, alias for alpha). The chosen channel is remembered by sparkrun for future updates — including the automatic one at launch — and the toolbar shows the installed version
- **Community recipes via sparkrun** — browse the mirrored `@official`/`@experimental` registry and launch on your Spark mesh with one click. The Nodes (TP) selector filters to recipes that actually fit your node count (multi-Spark recipes are badged and hidden at 1 node); Stop wires through `sparkrun stop`, and if that fails (stale job id, sparkrun error) Spark Studio force-removes the job's containers itself — Stop always means stopped. Every launch auto-saves a recipe in My Recipes (deduped per ref) so it's always one click away. Saved sparkrun recipes can carry launch options (`args._sparkrun.max_model_len` and arbitrary `-o key=value` `overrides`) that apply on every relaunch — handy for pinning workarounds to a model
- **Server-side ✓ working / ✗ failed badges** — a watchdog probes each engine's `/v1/models` and tags the recipe the moment it starts serving (or fails), even with every browser tab closed. It also catches sparkrun's nastiest failure mode: the engine crashing inside a container that stays "Up" — the run is marked failed, the real traceback is pulled from the in-container serve log for Ask Claude, and the zombie container is torn down. Slow multi-node loads can extend the never-ready deadline via `SPARK_STUDIO_SPARKRUN_GRACE` (seconds, default 1200)
- **Restart-proof runs** — on boot Spark Studio reconciles its run database with reality: still-live sparkrun workloads (even ones launched from a terminal, or left serving via `SPARK_STUDIO_KEEP_RUNS_ON_EXIT=1`) are re-adopted with logs, Stop, and chat intact; orphaned rows are marked exited. On normal exit (Ctrl+C) all launched workloads are unloaded automatically
- **Registry auto-sync** — the three upstream repos are refreshed on every start; a ✨ badge shows recipes that arrived since last sync
- **Local models** — scans every HF cache (env vars *and* caches referenced by your recipes), shows true on-disk sizes, one-click "Serve with vLLM", "Forge", or **Delete** to free disk space
- **Chat & Canvas / Engine Chat** — Monaco editor + chat, auto-targets whichever engine is running; renders Chart.js charts, Word/Excel export cards, and web-grounded answers. Auto-fits `max_tokens` to the engine's real context window every turn
- **Benchmarks** — quick tokens/s + TTFT sanity bench, plus full [llama-benchy](https://github.com/eugr/llama-benchy) sweeps (pp/tg at depth, concurrency, prefix caching). Every result records the engine version; compare any two runs side-by-side, and copy a shareable markdown report (hardware + engine + recipe + results) for the community
- **Tool Eval Bench** — answers "how *useful* is this model?", not just how fast. 12 deterministic cases score five skills 0–100: **tool selection** (pick the right tool among five), **argument extraction** (exact dates/amounts/names into tool args), **restraint** (answer directly instead of spurious tool calls), **using tool results** in the final answer, and **strict JSON** output. Each case shows what the model actually did (`called get_weather({"city": "Tokyo"})`); thinking models get a fair token budget and `<think>` blocks are stripped before checking. Every eval saves a markdown + JSON report to `tooleval-results/` and scores are kept in history per model. If the engine was launched without tool calling, the bench says so instead of silently scoring zero
- **Load telemetry on every run** — run cards show how long the model took to become ready and how much unified RAM it claimed (`loaded in 3m42s · +38.2 GB RAM`), stamped the moment the engine first answers. Stats persist in run history and survive app restarts; adopted/external endpoints (already loaded) honestly show nothing instead of a bogus number
- **Pre-launch memory guard** — on DGX Spark's 128 GB unified pool each model fills most of the pool, so only one fits at a time. Before launching, Spark Studio stops any other resident model, waits for its memory to actually free, and blocks a launch that still won't fit (with a one-click "launch anyway") — so swapping models doesn't OOM the box or take the dashboard down with it. See [Memory / OOM protection](#memory--oom-protection)
- **WebGPU tab** — in-browser inference via MLC WebLLM, with PDF/CSV/XLSX attachment extraction and built-in web search (bundled SearXNG, auto-started)
- **Crash-loop patching** — accept Claude / Codex's patched recipe with one click; re-runs immediately
- **Honest run states** — badges distinguish **failed** (crash, red) from **stopped** (you hit Stop) and clean exits, so a page of finished runs doesn't look like a page of errors
- **OpenAI-compatible gateway** — point any client (Continue, Cursor, etc.) at the active run's `:<port>/v1`

## Prerequisites

- **Linux** (NVIDIA DGX Spark / aarch64 recommended; x86_64 also works)
- **Python 3.11**
- **Git**
- **Node.js + npm** (only needed for the Claude Code and Codex agent features)
- **uv** (recommended) — `pip install uv` — or plain `pip`
- **nvidia-smi** available on PATH for GPU telemetry

Your inference engine(s) installed separately:
- [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [SGLang](https://docs.sglang.ai/start/install.html)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

Optional extras:
- [llama-benchy](https://github.com/eugr/llama-benchy) for full benchmark sweeps — `uv pip install --python env/bin/python llama-benchy`
- [sparkrun](https://github.com/spark-arena/sparkrun) for multi-node community recipes — `uvx sparkrun setup` (guided cluster wizard)
- **Docker** for spark-vllm-docker recipes and the bundled SearXNG web search

## Installation

### 1. Clone and start

```bash
git clone https://github.com/TheAwaken1/Spark-Studio.git
cd Spark-Studio
./start.sh
```

That's it — on first run `start.sh` creates the `./env` virtualenv and
installs `requirements.txt` automatically (via **uv** if installed, which can
also fetch Python 3.11 for you; otherwise plain `python3 -m venv` + pip).

<details>
<summary>Manual setup (if you prefer to run the steps yourself)</summary>

```bash
# venv — uv (recommended) or plain Python
uv venv env --python 3.11        # or: python3.11 -m venv env

# dependencies
uv pip install --python env/bin/python -r requirements.txt   # or: env/bin/pip install -r requirements.txt
```

</details>

### 2. (Optional) Pre-download the recipe registries

Spark Studio mirrors three upstream repos locally for offline use. The app
clones and refreshes them automatically on every start, so this step is only
needed if you want the mirrors in place before first boot (e.g. offline install):

```bash
mkdir -p data/registry
git clone --depth 1 https://github.com/spark-arena/recipe-registry.git data/registry/recipe-registry
git clone --depth 1 https://github.com/eugr/spark-vllm-docker.git    data/registry/spark-vllm-docker
git clone --depth 1 https://github.com/spark-arena/sparkrun.git        data/registry/sparkrun
```

### 3. (Optional) Install agent CLIs

For the **Ask Claude** and **Ask Codex** buttons:

```bash
npm install -g @anthropic-ai/claude-code @openai/codex

```

After installing, log in from the **Agents** tab inside Spark Studio — no API keys needed, just your browser OAuth flow.

## Running

```bash
./start.sh
```

(First run also sets up the Python environment — see Installation.)

### Health check (doctor)

```bash
./start.sh --doctor
```

Prints a full system report — OS/GPU/driver, unified memory, Docker, every
engine (vLLM/SGLang/llama.cpp), sparkrun, Claude/Codex agents, llama-benchy,
SearXNG, and your dashboard URLs — with a one-line fix for anything missing.
Exit code is non-zero when a core check fails, so it's scriptable. The same
report is served at `GET /api/doctor` for the UI and bug reports.

or manually:

```bash
env/bin/python -m uvicorn server:app --host 0.0.0.0 --port 7860
```

Then open **http://127.0.0.1:7860** in your browser, or reach it from any
machine on your network at `http://<this-machine's-LAN-IP>:7860`.

You can use any available port — just change `7860` to whatever you prefer
(`./start.sh --port 8000` works too; extra args are passed through to uvicorn).
If the port is already taken, `start.sh` tells you which process holds it and
suggests an alternative instead of failing mid-boot.
To restrict access to this machine only, use `--host 127.0.0.1`.

> **Note:** the app has no built-in authentication — anyone on your network
> can use it. Don't expose the port to the internet.

Stopping the app with **Ctrl+C also unloads everything it launched** — engine
processes, docker containers, and sparkrun workloads — so models don't linger
on the GPU after the dashboard is gone. If you *want* a model to keep serving
across app restarts, start with `SPARK_STUDIO_KEEP_RUNS_ON_EXIT=1 ./start.sh`;
the next boot re-adopts it automatically.

If `ufw` is enabled, allow the port for your LAN:

```bash
sudo ufw allow from 192.168.0.0/24 to any port 7860 proto tcp
```

### Optional environment variables

| Variable | Purpose | Default |
|---|---|---|
| `HF_HOME` | Override HuggingFace cache root (hub lives under `$HF_HOME/hub`) | `~/.cache/huggingface` |
| `HF_HUB_CACHE` | Point directly at a hub directory (`models--*` folders) | `$HF_HOME/hub` |
| `HF_HUB_ENABLE_HF_TRANSFER` | Faster HF downloads via `hf_transfer` | unset |
| `SEARXNG_URL` | Point web search at a specific SearXNG instance (overrides the bundled container) | auto-detected |
| `SPARK_STUDIO_NO_SPARKRUN_UPDATE` | Set to `1` to skip the automatic `sparkrun update` that `start.sh` runs on launch (same as `./start.sh --no-sparkrun-update`) | unset (auto-update) |
| `SPARK_STUDIO_SPARKRUN_GRACE` | Seconds a sparkrun run may stay not-ready before the watchdog fails it (only applies when no local container signal is available, e.g. remote multi-node heads) | `1200` |
| `SPARK_STUDIO_KEEP_RUNS_ON_EXIT` | Set to `1` to leave models serving when the app exits (Ctrl+C); the next boot re-adopts them | unset (models unload) |
| `SPARK_STUDIO_NO_MEMORY_GUARD` | Set to `1` to disable the pre-launch unified-memory guard (stop-and-wait + fit check before launching a model) | unset (guard on) |
| `SPARK_STUDIO_MEM_GUARD_TIMEOUT` | Max seconds the guard waits for a stopped model's memory to be reclaimed before proceeding | `120` |
| `SPARK_STUDIO_AGENT_TIMEOUT` | Seconds to wait for a Claude/Codex answer before giving up | `420` |
| `SPARK_STUDIO_AUTOFIX_WAIT` | Seconds Auto-Fix & Retry (and Optimize Speed) waits for a relaunched engine before judging the attempt | `1800` |
| `SPARK_STUDIO_OPTIMIZE_MARGIN` | Percent tok/s improvement over baseline at which Optimize Speed declares success and stops early | `10` |
| `SPARK_STUDIO_CORS_ORIGINS` | Comma-separated origins allowed to call the API cross-origin (off by default — the UI is same-origin and the app has no auth) | unset (no CORS) |

The Models tab scans all of the above **plus** any cache your saved recipes
hand to engine containers via `-e HF_HUB_CACHE=…`, so models show up even when
only the recipes know where they live.

Example:
```bash
HF_HOME=/mnt/models/.cache/huggingface \
HF_HUB_ENABLE_HF_TRANSFER=1 \
env/bin/python -m uvicorn server:app --host 0.0.0.0 --port 7860
```

### Web search

The chat's globe toggle grounds answers in live web results. On boot the app
auto-starts a bundled **SearXNG** container (`spark-searxng`, official
`searxng/searxng` image, bound to `127.0.0.1`). Config lives in
`data/searxng/settings.yml` (only reliable, key-free engines are enabled, JSON
output on). Backend priority is: `SEARXNG_URL` env override → bundled container →
any SearXNG on a well-known local port → **DuckDuckGo** (`ddgs`) fallback. Requires
Docker; if Docker is absent, search transparently falls back to DuckDuckGo.

Search is more than links: news/trending queries are routed to dedicated news
indexes with freshness windows, results are de-duplicated per domain, and the
top pages are **fetched and their article text extracted** (lxml) so the model
answers from real content with inline source citations — not from homepage
snippets. Reasoning models' chain-of-thought renders as a collapsible
"Thinking" section instead of polluting the answer.

## Memory / OOM protection

DGX Spark shares one 128 GB pool between GPU and system RAM, so a model that
overcommits can drive the box to true out-of-memory. The `sparkrun setup`
wizard installs **earlyoom** to kill a runaway workload before the kernel
locks up — good — but its default `--prefer` list includes `python`, and Spark
Studio's dashboard *is* a `python` process. Under memory pressure earlyoom can
then SIGKILL the ~100 MB dashboard right alongside the multi-GB model (you'll
see a bare `Killed` in the terminal where you ran `./start.sh`).

Spark Studio mitigates this from its side automatically:

- **Pre-launch memory guard.** Because each model fills most of the pool, only
  one fits at a time. Before starting a model the app stops any other resident
  model, **waits for its unified memory to actually be reclaimed** (the
  teardown/reclaim lag is exactly what causes back-to-back launches to OOM),
  and refuses a launch that still won't fit — the UI offers a one-click "launch
  anyway". This prevents the OOM at the source instead of cleaning up after it.
  Estimated footprint comes from the recipe's `gpu-memory-utilization` /
  `mem-fraction-static` (× the 128 GB pool); llama.cpp isn't pool-filling so
  it's exempt from the hard fit-check. Override with the `force` flag on a
  launch, or globally with `SPARK_STUDIO_NO_MEMORY_GUARD=1`.
- **OOM priority.** At startup the app tries to lower its own OOM priority (only
  possible with privilege, e.g. a systemd unit with `OOMScoreAdjust=-500`), and
  — the part that always works — it **raises the OOM priority of every engine
  subprocess it launches**, so if memory pressure does hit, it kills the
  relaunchable model before the control plane.

For the docker / sparkrun path (where the model runs in a container the app
doesn't own), apply the box-wide fix once — drop `python` from earlyoom's
`--prefer` list:

```bash
sudo sed -i.bak 's/|python3|python)/)/' /etc/default/earlyoom
sudo systemctl restart earlyoom
pgrep -a earlyoom   # verify: the --prefer group no longer contains python
```

Inference engines are still matched by their real names (`vllm`, `sglang`,
`llama-server`, …), so earlyoom keeps protecting the box — it just stops
treating the dashboard as a preferred sacrifice.

## Updating

Pull the latest source and refresh dependencies:

```bash
git pull

uv pip install --python env/bin/python -r requirements.txt --upgrade

# Update agent CLIs
npm install -g @anthropic-ai/claude-code @openai/codex
```

Registry mirrors refresh themselves on every app start (or click **Refresh
now** on the Forge tab) — no manual git commands needed.

## Resetting

To start completely fresh (wipes the venv and the local database):

```bash
rm -rf env data/spark_studio.db
```

Then just `./start.sh` again — it rebuilds the environment automatically.

## Recipe Schema

```json
{
  "name": "Llama 3.1 8B · vLLM",
  "engine": "vllm",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "args": {
    "max-model-len": 131072,
    "max-num-batched-tokens": 16384,
    "gpu-memory-utilization": 0.9,
    "enable-chunked-prefill": true,
    "trust-remote-code": true
  },
  "env": { "VLLM_WORKER_MULTIPROC_METHOD": "spawn" },
  "notes": "",
  "tags": "throughput,vllm"
}
```

### Context, batch size, and capabilities (vLLM)

Every vLLM recipe — whether **forged**, **created**, or **edited** — is normalized so it serves the
**full context the model supports, capped at 262144**, with `max-num-batched-tokens: 16384`. The cap
uses the model's native context (from its HF config), so a 262144-native model like Qwen3 gets the
full 256K while a 131072 model like Llama-3.1 gets 131072 — vLLM never has to be asked for more
context than the model allows, so recipes always launch. (vLLM sizes the KV cache to
`gpu-memory-utilization`, not to `max-model-len`, so a high ceiling doesn't OOM — it only trades some
max concurrency.)

**Reasoning and tool calling** are added automatically for recognized model families (Qwen3, GLM-4.7,
gpt-oss, MiniMax-M2, Nemotron, Gemma-4, …) — Forge wires in the correct `--tool-call-parser` /
`--reasoning-parser` and `--enable-auto-tool-choice`. Unrecognized families get nothing, because a
wrong parser breaks serving. The **recipe editor** shows a **Capabilities** row with Tool calling /
Reasoning toggles (each enabled only when a parser is known for that model) so you can override per
recipe; `GET /api/recipes/capabilities?model=<repo>` returns what a given model supports and the
context that will be applied.

Engine values: `vllm` | `sglang` | `llamacpp`. The runner maps `args` to CLI flags automatically — kebab-case keys become `--kebab-case`, booleans become bare flags.

## API

Once running at `http://127.0.0.1:7860`:

### JavaScript

```js
// List recipes
const recipes = await fetch('/api/recipes').then(r => r.json());

// Start a run
const run = await fetch('/api/runs', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    engine: 'vllm',
    args: { model: 'meta-llama/Llama-3.1-8B-Instruct', 'max-model-len': 16384 },
  }),
}).then(r => r.json());

// Stream logs
const es = new EventSource(`/api/runs/${run.id}/stream`);
es.addEventListener('log', ev => console.log(ev.data));

// Chat against the active engine
const reply = await fetch('/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ model: 'local', messages: [{ role: 'user', content: 'Hi!' }] }),
}).then(r => r.json());
```

### Python

```python
import httpx

BASE = "http://127.0.0.1:7860"

# Forge recipes for a HF model
resp = httpx.get(f"{BASE}/api/hf/forge", params={"repo": "meta-llama/Llama-3.1-8B-Instruct"}).json()
print(resp["report"]["verdict"], resp["recipes"][0])

# Launch a run
run = httpx.post(f"{BASE}/api/runs", json={
    "engine": "vllm",
    "args": {"model": "meta-llama/Llama-3.1-8B-Instruct", "max-model-len": 16384},
}).json()

# Ask Claude to fix a broken run
tail = httpx.get(f"{BASE}/api/runs/{run['id']}/tail", params={"n": 200}).json()
fix = httpx.post(f"{BASE}/api/agents/fix", json={
    "agent": "claude",
    "recipe": {"engine": "vllm", "args": {"model": "…"}},
    "logs": "\n".join(tail["lines"]),
}).json()
print(fix["diagnosis"])
```

### cURL

```bash
# Compatibility check
curl "http://127.0.0.1:7860/api/hf/check?repo=meta-llama/Llama-3.1-70B-Instruct"

# Start a vLLM run
curl -X POST http://127.0.0.1:7860/api/runs \
  -H "Content-Type: application/json" \
  -d '{"engine":"vllm","args":{"model":"meta-llama/Llama-3.1-8B-Instruct","max-model-len":16384}}'

# Stream logs
curl -N http://127.0.0.1:7860/api/runs/<run_id>/stream

# Benchmark the active run
curl -X POST http://127.0.0.1:7860/api/bench \
  -H "Content-Type: application/json" \
  -d '{"runs":3,"max_tokens":256}'

# Generate a Word doc
curl -X POST http://127.0.0.1:7860/api/export/docx \
  -H "Content-Type: application/json" \
  -d '{"title":"Test","sections":[{"heading":"H1","level":1},{"paragraph":"hello"}]}' \
  -o test.docx
```

### Endpoint Map

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/system` | GPU / engine / platform info |
| `GET` | `/api/doctor` | Full system health report (same as `./start.sh --doctor`) |
| `GET` | `/api/recommend?k=` | Starter-model recommendations per goal, ranked from local signals |
| `GET` | `/api/host?refresh=` | Structured GPU + Spark-mesh probe |
| `GET` | `/api/active` | Engine currently serving chat |
| `GET` `POST` `DELETE` | `/api/recipes[...]` | Recipe CRUD (POST normalizes vLLM context/batch) |
| `GET` | `/api/recipes/capabilities?model=` | Tool/reasoning support + suggested max_model_len for a model |
| `GET` `POST` | `/api/runs[...]` | List / start engine runs |
| `GET` | `/api/runs/{id}/stream` | SSE log stream |
| `GET` | `/api/runs/{id}/tail?n=` | Ring-buffer snapshot |
| `POST` | `/api/runs/{id}/stop?force=` | SIGTERM / SIGKILL |
| `GET` | `/api/hf/check?repo=` | DGX Spark compatibility report |
| `GET` | `/api/hf/forge?repo=` | Generated starter recipes |
| `GET` `DELETE` | `/api/models/local` | Scan HF caches / delete a cached model |
| `POST` `DELETE` | `/api/external[...]` | Register / unregister an already-running endpoint as a run |
| `POST` | `/api/arena/import` | Turn a spark-arena.com benchmark link into a runnable recipe |
| `GET` | `/api/engines/install/{engine}` | SSE stream of `uv pip install` for vLLM / SGLang |
| `GET` | `/api/models/served` | Model id + context length of the active run |
| `GET` `POST` | `/api/registry/status` `/api/registry/sync` | Mirror status (incl. new-recipe diff) / manual sync |
| `GET` | `/api/registry/recipes` `/api/registry/mods` | Indexed curated recipes and fix mods |
| `GET` | `/api/sparkrun/status` `/api/sparkrun/recipes` | sparkrun install state + version / launchable community recipes |
| `POST` | `/api/sparkrun/run` | Launch `@official/…` or `@experimental/…` on the mesh |
| `POST` | `/api/sparkrun/update` | Run `sparkrun update` in the background (`{"channel": "stable"\|"beta"\|"alpha"\|"yolo"}`, omit to stay on the current channel) |
| `GET` | `/api/sparkrun/update/status` | Progress/result of the last sparkrun update (running, ok, log, version before/after) |
| `POST` | `/api/attachments/extract` | Extract text from PDF / CSV / XLSX uploads |
| `POST` | `/api/export/docx` | Build a `.docx` from a JSON spec |
| `POST` | `/api/export/xlsx` | Build a `.xlsx` from a JSON spec |
| `GET` | `/api/search/status` | Active search backend (bundled SearXNG / DuckDuckGo) |
| `GET` | `/api/search?q=` | Web search (SearXNG, DuckDuckGo fallback) |
| `POST` | `/api/searxng/start` | (Re)start the bundled SearXNG container |
| `POST` | `/api/searxng/stop` | Stop the bundled SearXNG container |
| `GET` | `/api/agents/status` | Claude/Codex install + login state |
| `GET` | `/api/agents/login/{claude\|codex}` | SSE OAuth flow |
| `POST` | `/api/agents/fix` | JSON-structured recipe patch |
| `GET` | `/api/agents/autofix/{rid}` | SSE hands-free fix loop (diagnose → patch → relaunch → retry) |
| `GET` | `/api/agents/optimize/{rid}` | SSE speed-optimization loop (bench → patch → relaunch → re-bench; fastest config wins) |
| `POST` | `/api/chat` | Proxy to active engine (OpenAI-compatible, auto-fits `max_tokens`) |
| `GET` `POST` | `/api/bench[...]` | Run and list quick benchmarks |
| `GET` `POST` | `/api/benchy/...` | llama-benchy: status, run (SSE), history |
| `GET` | `/api/benchy/{id}/export` | Shareable markdown benchmark report |
| `POST` | `/api/tooleval/run` | Start the Tool Eval Bench against a run (defaults to the active engine) |
| `GET` | `/api/tooleval/status` | Live progress, per-case results, and scores of the current/last eval |
| `GET` | `/api/tooleval/history` | Past Tool Eval scores per model (reports live in `tooleval-results/`) |
| `GET` | `/api/spark/vitals` | Live GPU / unified-memory telemetry (SSE) |

## Platform

Linux + NVIDIA. Tested on DGX Spark (Grace Blackwell, aarch64). vLLM and SGLang are Linux-first; llama.cpp is cross-platform but wired for GPU offload here.

## Community & credits

Spark Studio is built on the work of the DGX Spark / GB10 community:

- [spark-arena/recipe-registry](https://github.com/spark-arena/recipe-registry) — the curated recipe registry this app mirrors and forges from
- [spark-arena/sparkrun](https://github.com/spark-arena/sparkrun) — multi-node workload launcher, integrated as a runner
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — canonical docker orchestration for Spark recipes (used verbatim for docker runs)
- [eugr/llama-benchy](https://github.com/eugr/llama-benchy) — the benchmark engine behind the Benchmarks tab
- [Spark Arena leaderboard](https://spark-arena.com/leaderboard) — community benchmark hub; paste its recipe YAMLs straight into the engine tabs, and share your own results with the ⧉ report button
