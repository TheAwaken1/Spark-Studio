# Spark Studio User-Friendly Install & Onboarding Roadmap

## Goal

Make Spark Studio feel as close to a **one-click install** and **beginner-friendly local AI workstation** as possible.

The ideal experience should be:

```bash
curl -fsSL https://sparkstudio.dev/install | bash
```

Then the browser opens to a guided setup flow:

```text
Welcome to Spark Studio
✅ DGX Spark detected
✅ GPU ready
✅ sparkrun found
⚠️ Claude Code not installed

Recommended model:
Qwen 2.5 7B Instruct — fits your Spark

[Launch Model]
```

After launch:

```text
Model loaded in 2m14s · +38 GB RAM
[Chat Now] [Benchmark] [Optimize Speed] [Stop]
```

The user should not need to understand Python environments, model flags, YAML, registry details, LAN binding, or dependency problems before getting their first model running.

---

# 1. Create a True One-Command Installer

## Recommendation

Add a beginner-friendly install command:

```bash
curl -fsSL https://raw.githubusercontent.com/YOURNAME/Spark-Studio/main/install.sh | bash
```

Eventually, upgrade to a branded domain:

```bash
curl -fsSL https://sparkstudio.dev/install | bash
```

Optional future packaging:

```bash
uvx spark-studio
```

or:

```bash
pipx run spark-studio
```

## What the installer should do

The installer should:

- Check for Linux
- Check for Python 3.11
- Install or suggest `uv`
- Check for Git
- Check for `nvidia-smi`
- Detect DGX Spark / NVIDIA GPU
- Check for Node/npm only if agent features are requested
- Check for Docker only if Web Search / SearXNG is requested
- Check whether `sparkrun` exists
- Offer to run `uvx sparkrun setup`
- Clone Spark Studio
- Create the virtual environment
- Install Python dependencies
- Start Spark Studio
- Print the local and LAN URLs

## Example install output

```text
Spark Studio Installer

✅ Linux detected
✅ NVIDIA GPU detected
✅ nvidia-smi found
✅ Python 3.11 found
✅ uv found
⚠️ Docker not found — Web Search container will be disabled
⚠️ sparkrun not found — Community Recipes will be disabled

Install sparkrun now? [Y/n]

Cloning Spark Studio...
Creating Python environment...
Installing dependencies...
Starting Spark Studio...

Local:   http://127.0.0.1:7860
LAN:     http://192.168.1.50:7860
```

## Why this matters

The current clone-and-run flow is good for developers, but beginners still have to understand Git, folders, Python errors, optional dependencies, and networking.

The goal is to make setup feel like installing an app, not configuring a development environment.

---

# 2. Add a First-Run Setup Wizard

## Recommendation

On the first browser launch, do not drop users directly into the full dashboard.

Show a guided onboarding wizard.

## First-run wizard flow

### Step 1: System Check

Show clear system status:

```text
✅ DGX Spark detected
✅ NVIDIA GPU detected
✅ 128 GB unified memory detected
✅ nvidia-smi available
✅ Python environment ready
⚠️ Docker missing — Web Search disabled
⚠️ Claude Code missing — Ask Claude disabled
⚠️ Codex missing — Ask Codex disabled
```

### Step 2: Choose User Goal

Ask what the user wants to do:

- I just want to run one local model
- I want to use sparkrun community recipes
- I want to run multi-node Spark workloads
- I want Claude/Codex auto-fix
- I want WebGPU/browser models
- I want benchmarking and model comparison

### Step 3: Pick a Starter Model

Show only options that fit the current hardware.

Recommended categories:

- Fastest starter
- Best quality that fits
- Lowest memory
- Best for coding
- Best for tool calling
- Best for chat

### Step 4: Launch

Use one big button:

```text
[Launch Recommended Model]
```

### Step 5: Confirm Success

After the model loads:

```text
Model loaded successfully

Loaded in: 2m14s
Memory used: +38 GB RAM
Endpoint: http://127.0.0.1:8000/v1

[Chat Now] [Run Benchmark] [Optimize Speed] [Stop]
```

## Why this matters

Spark Studio has a lot of powerful features. The first-run wizard protects beginners from feeling overwhelmed and gets them to a successful first run faster.

---

# 3. Add Beginner Mode and Advanced Mode

## Recommendation

Add a UI mode toggle:

```text
Mode: Beginner | Advanced
```

The default should be **Beginner Mode**.

## Beginner Mode

Show only the essentials:

- Home
- Models
- Recipes
- Running
- Chat
- Logs
- Stop

Hide advanced features behind simple language.

Example labels:

- “Run a Model”
- “Chat”
- “See Logs”
- “Stop Model”
- “Fix Problem”

## Advanced Mode

Show the full power-user interface:

- vLLM
- SGLang
- llama.cpp
- WebGPU
- sparkrun
- Agents
- Auto-Fix
- Optimize Speed
- Benchmarks
- Tool Eval
- Registry
- API
- Memory/OOM controls
- Exports
- Search backend
- External endpoints

## Why this matters

The project is powerful, but the number of tabs/features can intimidate new users.

Beginner Mode lets Spark Studio feel simple without removing power from advanced users.

---

# 4. Add a Recommended Starter Model Flow

## Recommendation

Ship Spark Studio with a beginner-safe “Start Here” flow.

The user should not need to know YAML, Hugging Face IDs, engine flags, or registry names to launch their first model.

## Suggested starter buttons

```text
[Run Fast Starter Model]
[Run Best Quality Model That Fits]
[Run Coding Model]
[Run Tool-Calling Model]
[Run Low-Memory Model]
```

## Hardware-aware recommendation logic

Spark Studio should inspect:

- Available unified memory
- GPU count / Spark node count
- Local models already downloaded
- Installed engines
- Registry recipes
- Compatibility score
- Expected memory usage
- Known working recipes

Then recommend one safe default.

## Example UI

```text
Recommended for your Spark:

Qwen 2.5 7B Instruct
Reason: Fast, reliable, fits easily, good chat quality.

Engine: vLLM
Estimated memory: 24–36 GB
Expected startup: 2–4 minutes

[Launch]
```

## Why this matters

A beginner should be able to get a model running without touching a recipe editor.

---

# 5. Add Plain-English Error Messages

## Recommendation

Translate common technical failures into helpful explanations.

## Example: missing vLLM

Instead of only showing:

```text
ModuleNotFoundError: No module named vllm
```

Show:

```text
vLLM is not installed yet.

Spark Studio can still run sparkrun recipes, llama.cpp, and WebGPU.

To enable vLLM, click:

[Install vLLM]

Or run:

uv pip install --python env/bin/python vllm
```

## Example: not enough memory

Instead of:

```text
CUDA out of memory
```

Show:

```text
This model is too large for the available memory right now.

Spark Studio can:
1. Stop the currently running model
2. Wait for memory to clear
3. Try a smaller context length
4. Launch a smaller recommended model

[Stop Other Model and Retry]
[Reduce Context and Retry]
[Pick Smaller Model]
[Launch Anyway]
```

## Example: Docker missing

```text
Docker is not installed.

Web Search with bundled SearXNG needs Docker.
Spark Studio will continue using the DuckDuckGo fallback when possible.

[Install Docker Guide]
[Continue Without Docker]
```

## Why this matters

Friendly error messages make the app feel stable even when something fails.

Users do not mind problems as much when the app explains what happened and gives the next step.

---

# 6. Make Dependencies Optional by Feature

## Recommendation

Do not make users feel like they need every dependency before starting.

Organize dependencies by feature.

| User wants | Required |
|---|---|
| Basic dashboard | Python + Git |
| GPU telemetry | `nvidia-smi` |
| sparkrun community recipes | `sparkrun` |
| Claude/Codex auto-fix | Node + Claude Code / Codex CLI |
| Web search | Docker for bundled SearXNG, or fallback search |
| Benchmarks | llama-benchy |
| vLLM engine | vLLM |
| SGLang engine | SGLang |
| llama.cpp engine | llama.cpp |
| WebGPU inference | Compatible browser / WebLLM assets |

## UI behavior

Each feature should show one of these states:

```text
✅ Available
⚠️ Not installed
❌ Error
⬇️ Install
⏭️ Skip
```

## Example

```text
Claude Auto-Fix
Status: Not installed

This feature lets Spark Studio ask Claude Code to diagnose and patch broken recipes.

[Install Claude Code]
[Skip]
```

## Why this matters

A 1-click install should not fail just because optional features are missing.

The app should install the core first and let users add advanced features later.

---

# 7. Package Spark Studio Three Ways

## Recommendation

Support three installation paths.

---

## Option A: Recommended Script Installer

Best for most DGX Spark users.

```bash
curl -fsSL https://sparkstudio.dev/install | bash
```

Use this as the primary README quick start.

---

## Option B: Developer Install

Best for GitHub users who want to modify the code.

```bash
git clone https://github.com/YOURNAME/Spark-Studio.git
cd Spark-Studio
./start.sh
```

---

## Option C: Docker Compose Install

Best for users who want isolation.

```bash
docker compose up
```

Even if GPU inference engines are host-managed, Docker can still be useful for the dashboard/control plane, SearXNG, and supporting services.

## Why this matters

Different users trust different installation styles.

A script installer feels easy, Git feels transparent, and Docker feels clean.

---

# 8. Improve the README Landing Section

## Recommendation

Make the top of the README shorter, clearer, and more emotional.

Do not lead with every feature. Lead with the outcome.

## Suggested README opening

```markdown
# Spark Studio

One-click DGX Spark inference dashboard.

Run vLLM, SGLang, llama.cpp, WebGPU, and sparkrun recipes from one friendly UI.
Launch models, monitor memory, chat, benchmark, and auto-fix broken recipes with Claude or Codex.

## Quick Start

curl -fsSL https://sparkstudio.dev/install | bash

Open:

http://<your-spark-ip>:7860

## What you can do in 60 seconds

1. Pick a model
2. Click Launch
3. Watch logs and memory
4. Chat with the model
5. Click Auto-Fix if it fails
```

## Recommended README structure

```markdown
# Spark Studio

## Quick Start

## What You Can Do in 60 Seconds

## Screenshots / GIF Demo

## Why Spark Studio?

## Beginner Setup

## Advanced Setup

## Feature Overview

## Requirements

## Installation Options

## First Run Wizard

## Troubleshooting

## FAQ

## API

## Community & Credits
```

## Move the giant feature list lower

Keep the full feature list, but move it below:

- Quick Start
- Screenshots
- First-run experience
- Why Spark Studio?

## Why this matters

The README should sell the project before it documents every detail.

Most users decide whether to keep reading in the first few seconds.

---

# 9. Add Screenshots and a Short GIF Demo

## Recommendation

Add visuals near the top of the README.

## Must-have screenshots

1. Home dashboard
2. First-run wizard
3. One-click recipe launch
4. Logs and GPU memory
5. Chat working
6. Auto-Fix button
7. Benchmark results
8. Local models tab

## Best demo GIF

Create a 20-second GIF showing:

```text
Paste model ID
→ Click Forge
→ Click Run
→ Watch model load
→ Chat with model
```

## Suggested README section

```markdown
## Demo

![Spark Studio demo](docs/demo.gif)

Paste a Hugging Face model ID, click Forge, launch the recommended recipe, and chat with the model.
```

## Why this matters

A GIF communicates the product faster than a large README.

For this kind of project, visuals are not optional — they are part of the onboarding.

---

# 10. Add a Health Check / Doctor Command

## Recommendation

Add:

```bash
./spark-studio doctor
```

or:

```bash
spark-studio doctor
```

## Example output

```text
Spark Studio Doctor

✅ NVIDIA GPU detected
✅ nvidia-smi found
✅ Python 3.11 found
✅ uv found
✅ Spark Studio environment ready
⚠️ Docker not found — Web Search container disabled
⚠️ sparkrun not found — Community Recipes disabled
✅ vLLM installed
❌ SGLang missing — click Install in UI
✅ LAN URL: http://192.168.1.50:7860
```

## Doctor should check

- OS
- Architecture
- Python version
- uv
- Git
- Node/npm
- Docker
- NVIDIA driver
- `nvidia-smi`
- Available memory
- DGX Spark detection
- vLLM installed
- SGLang installed
- llama.cpp installed
- sparkrun installed
- Claude Code installed
- Codex installed
- llama-benchy installed
- SearXNG container status
- LAN IP
- Port availability

## Why this matters

A doctor command gives users confidence and gives maintainers better bug reports.

---

# 11. Add Desktop Launcher and Systemd Service Options

## Recommendation

After install, offer:

```text
Create desktop launcher? [Y/n]
Install as systemd service? [y/N]
Start on boot? [y/N]
```

## Desktop launcher

Create a `.desktop` file so users can open Spark Studio like a normal app.

Example app name:

```text
Spark Studio
```

Action:

```text
Open Spark Studio Dashboard
```

## Systemd service

Allow:

```bash
sudo systemctl enable --now spark-studio
```

This would make Spark Studio always available at:

```text
http://spark.local:7860
```

or:

```text
http://<spark-lan-ip>:7860
```

## Why this matters

A real app should not require opening a terminal every time.

This is especially helpful for a dedicated DGX Spark box.

---

# 12. Add Local Network Discovery

## Recommendation

Every launch should clearly print all access URLs.

## Example

```text
Spark Studio is running:

Local:     http://127.0.0.1:7860
LAN:       http://192.168.1.50:7860
Hostname:  http://dgx-spark.local:7860
```

## Add UI display

Show the same URLs inside the app:

```text
Access Spark Studio from another computer:
http://192.168.1.50:7860
```

## Optional QR code

Show a QR code for phones/tablets on the same network.

```text
Scan to open Spark Studio on your phone
```

## Why this matters

LAN access is one of Spark Studio’s strengths, but users should not have to find their IP address manually.

---

# 13. Add an “I Broke It” Recovery Section

## Recommendation

Add both UI recovery buttons and README recovery commands.

## UI recovery buttons

Create a “Recovery” or “Troubleshooting” page with:

- Clear failed runs
- Remove orphan containers
- Restart Spark Studio
- Reset app database
- Reset registry cache
- Rebuild Python environment
- Full safe reset

## Safe descriptions

Before each action, explain what will happen.

Example:

```text
Reset app database

This removes saved recipes, run history, benchmark history, and settings.
It does not delete downloaded models.

[Reset Database]
```

Example:

```text
Remove orphan containers

This stops containers that Spark Studio launched but can no longer control.
It does not delete model files.

[Clean Containers]
```

## README section

```markdown
## I Broke It — Safe Reset

Clear app database only:

rm -f data/spark_studio.db

Rebuild Python environment:

rm -rf env
./start.sh

Full reset:

rm -rf env data/spark_studio.db
./start.sh
```

## Why this matters

Beginners are more willing to experiment when they know there is a safe reset path.

---

# 14. Add Better Install-Time Choices

## Recommendation

During install, ask what type of setup the user wants.

## Example

```text
Choose setup type:

1. Basic — dashboard, recipes, local runs
2. Recommended — Basic + sparkrun + model manager
3. Full — Recommended + Claude/Codex + Web Search + Benchmarks
4. Custom

Select [2]:
```

## Suggested defaults

Use **Recommended** as the default.

## Setup profiles

### Basic

- Spark Studio dashboard
- Python environment
- Recipe library
- Local model scan
- Logs
- Chat

### Recommended

- Everything in Basic
- sparkrun integration
- Registry sync
- GPU telemetry
- Memory guard
- Starter model flow

### Full

- Everything in Recommended
- Claude Code
- Codex
- llama-benchy
- Docker/SearXNG
- WebGPU assets
- Advanced benchmarking

## Why this matters

A single install command can still feel user-friendly while avoiding a giant all-or-nothing dependency install.

---

# 15. Add a Better Home Dashboard

## Recommendation

The home screen should answer five questions immediately:

1. Is my Spark healthy?
2. Is a model running?
3. What can I launch?
4. How much memory is free?
5. What should I do next?

## Suggested dashboard cards

### System Status

```text
DGX Spark detected
GPU ready
Unified memory: 72 GB free / 128 GB total
```

### Active Model

```text
No model running

[Launch Recommended Model]
```

or:

```text
Qwen 2.5 7B Instruct
Loaded in 2m14s · +38 GB RAM

[Chat] [Benchmark] [Optimize] [Stop]
```

### Recommended Next Action

Examples:

```text
Start by launching a beginner-friendly model.
```

```text
Your model is running. Try chatting or run a quick benchmark.
```

```text
This run failed. Auto-Fix can diagnose and patch the recipe.
```

### Feature Health

```text
✅ sparkrun
✅ vLLM
⚠️ Docker
⚠️ Claude Code
```

## Why this matters

The home screen should act like a control center, not just a list of tabs.

---

# 16. Add “One-Click Fix” Everywhere

## Recommendation

Wherever Spark Studio detects a problem, show one next action.

Examples:

```text
vLLM missing
[Install vLLM]
```

```text
sparkrun missing
[Install sparkrun]
```

```text
Run failed
[Auto-Fix and Retry]
```

```text
Model too large
[Try Smaller Recommended Model]
```

```text
Port already in use
[Use Port 7861]
```

```text
Memory not freed yet
[Wait and Retry]
```

## Why this matters

The user should not have to read logs to know what to do next.

---

# 17. Add Better Bug Report Export

## Recommendation

Add a “Copy Bug Report” button.

## Report should include

- Spark Studio version
- OS
- Architecture
- Python version
- GPU info
- Unified memory info
- Installed engines
- sparkrun version
- Docker status
- Node/npm status
- Active recipe
- Last 300 log lines
- Error summary
- Recent doctor output

## Example button

```text
[Copy Bug Report]
```

## Why this matters

This makes GitHub issues cleaner and easier to debug.

It also helps users ask for help from Claude, Codex, or the community.

---

# 18. Add Versioned Releases

## Recommendation

Create GitHub releases:

```text
v0.1.0
v0.2.0
v0.3.0
```

## Add an update command

```bash
spark-studio update
```

or:

```bash
./start.sh --update
```

## Add UI update check

```text
Spark Studio v0.2.0
Update available: v0.2.1

[Update Now]
```

## Why this matters

A project feels more trustworthy when users can install a known release instead of always pulling from `main`.

---

# 19. Add a Smaller “Lite” Mode

## Recommendation

Create a lightweight path for users who only want sparkrun and chat.

Possible command:

```bash
spark-studio --lite
```

Lite Mode could disable:

- Agents
- Tool Eval
- WebGPU
- Exports
- Advanced benchmarks
- Search backend
- Registry diff badges
- Advanced API pages

## Why this matters

Some users may want Spark Studio’s UI without the full workstation experience.

Lite Mode can also help troubleshoot performance or dependency issues.

---

# 20. Suggested Priority Order

If building this in stages, use this order:

## Phase 1 — First Impression

1. Shorten the README top section
2. Add screenshots/GIF
3. Add first-run setup wizard
4. Add recommended starter model
5. Add Beginner Mode

## Phase 2 — Install Experience

6. Add one-command installer
7. Add doctor command
8. Add local network URL display
9. Add plain-English dependency checks
10. Add install profiles: Basic / Recommended / Full

## Phase 3 — Recovery and Trust

11. Add “I Broke It” recovery page
12. Add one-click fixes for common issues
13. Add bug report export
14. Add versioned releases
15. Add update command

## Phase 4 — App-Like Polish

16. Add desktop launcher
17. Add systemd service option
18. Add QR code for LAN access
19. Add Lite Mode
20. Add in-app update check

## Phase 5 — Community Cluster Mode

21. Add Cluster page
22. Add node health cards
23. Add multi-node launch selector
24. Add fits-this-cluster badges
25. Add multi-node readiness checks
26. Add per-node logs
27. Add retry with fewer nodes
28. Add cluster benchmark comparison
29. Add export cluster report


---


---

# 22. Multi-Node / Cluster Mode

## Recommendation

Yes, multi-node support should be included in the initial Spark Studio roadmap.

However, it should not be treated as a required beginner feature.

It should be positioned as:

```text
Advanced / Community Feature
Powered by sparkrun
```

The beginner experience should remain:

```text
Install → Launch one model → Chat → Monitor → Fix basic issues
```

The advanced/community experience can become:

```text
Discover cluster → Choose nodes / TP → Launch recipe through sparkrun → Monitor all nodes → Benchmark → Export report
```

## Important Design Rule

Spark Studio should not try to become the distributed runtime.

Instead:

```text
sparkrun handles multi-node orchestration.
Spark Studio provides the friendly UI, health checks, logs, fit checks, recipes, benchmarks, and recovery tools.
```

This keeps the architecture clean and avoids rebuilding what sparkrun already does well.

---

## Why This Belongs in the Roadmap

Many DGX Spark community users run more than one node.

Some users may have:

- 1 DGX Spark
- 2 DGX Spark nodes
- 3 DGX Spark nodes
- 4+ DGX Spark nodes

Spark Studio should work great for one-node users, but it should also feel valuable to community users with larger Spark clusters.

The opportunity is to make multi-node inference easier to understand.

Most users do not want to manually think through:

- Which nodes are online
- Whether Docker is running everywhere
- Whether sparkrun sees the cluster
- Which TP value to use
- Which recipe fits the cluster
- Which node failed
- Where the logs are
- Whether the endpoint is actually serving
- Whether TP 2 or TP 4 is faster

Spark Studio can make that friendly.

---

## Suggested UI: Cluster Page

Add a dedicated page:

```text
Cluster
```

The Cluster page should show all known Spark nodes and their health.

## Example

```text
Cluster Status

Node 1: online · 128 GB · ready
Node 2: online · 128 GB · ready
Node 3: offline
Node 4: online · 128 GB · ready

Available tensor parallel sizes:
TP 1 ✅
TP 2 ✅
TP 3 ⚠️ one node offline
TP 4 ❌ not enough healthy nodes
```

## Node Card Details

Each node card should show:

- Node name / hostname
- IP address
- Online/offline status
- GPU status
- Unified memory total
- Unified memory free
- Docker status
- sparkrun status
- Current workload
- Last heartbeat
- Error state if any

## Why This Matters

Multi-node users need confidence that their cluster is healthy before launching a large model.

---

## Multi-Node Launch Mode

When launching a recipe, Spark Studio should show a simple run target selector.

## Example

```text
Run on:

( ) This Spark only
( ) 2-node cluster
( ) 3-node cluster
( ) 4-node cluster
```

Behind the scenes, Spark Studio can translate that into the correct sparkrun launch command.

Example:

```bash
sparkrun run <recipe> --tp 2
```

or:

```bash
sparkrun run <recipe> --tp 4
```

The user should not need to remember command-line flags.

---

## Fits This Cluster Badges

Spark Studio already has the idea of hardware-aware fit checks.

For multi-node, expand this from:

```text
Fits this Spark
Needs N GPUs
Too big
```

to:

```text
Fits this Spark
Fits 2-node cluster
Fits 4-node cluster
Too big for current cluster
```

## Example

```text
Llama 70B

Single Spark: ❌ Too big
2-node cluster: ✅ Fits
4-node cluster: ✅ Fits better
```

## Why This Matters

A recipe can be too large for one Spark but reasonable for two or four nodes.

The UI should make that obvious.

---

## Multi-Node Readiness Checks

Distributed runs fail differently from single-node runs.

Before launching a multi-node recipe, Spark Studio should check:

- Can sparkrun see all nodes?
- Are all selected nodes online?
- Is Docker running on each node?
- Is the model available or downloadable?
- Is the network path healthy?
- Is the head node reachable?
- Is the expected OpenAI-compatible endpoint available?
- Is there enough memory per node?
- Is another workload already running?
- Are stale containers or zombie jobs present?

## Example UI

```text
Cluster Readiness

✅ Node 1 reachable
✅ Node 2 reachable
✅ Docker running on Node 1
❌ Docker not running on Node 2
✅ sparkrun installed
⚠️ Model not cached on all nodes

[Fix Docker on Node 2]
[Continue Anyway]
[Cancel]
```

## Why This Matters

Multi-node errors are intimidating.

Plain-English readiness checks would make Spark Studio feel much more polished than raw CLI usage.

---

## Per-Node Logs

Add per-node log views for distributed runs.

## Example

```text
Logs

[All Nodes] [Head Node] [Node 1] [Node 2] [Node 3] [Node 4]
```

If a node fails, Spark Studio should call it out clearly.

## Example

```text
Node 3 failed during model load.

Likely cause:
The container exited before joining the distributed runtime group.

Suggested actions:
[Ask Agent to Diagnose]
[Retry Without Node 3]
[Stop Cluster Run]
```

## Why This Matters

A single combined log stream is hard to read.

Multi-node users need to know which node failed.

---

## Cluster Benchmark Comparison

Add benchmarks that compare different node counts.

## Example

```text
Benchmark this model across:

☑ TP 1
☑ TP 2
☑ TP 4
```

## Example Results

| TP | Nodes | TTFT | Tok/s | Memory / Node | Status |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 2.1s | 44 | 109 GB | Stable |
| 2 | 2 | 1.6s | 71 | 78 GB | Stable |
| 4 | 4 | 1.8s | 92 | 52 GB | Stable |

## Summary Output

```text
Best value: TP 2
Fastest: TP 4
Most stable: TP 1
```

## Why This Matters

Community users will want to know whether using more nodes is actually worth it.

Spark Studio can turn those experiments into shareable results.

---

## Retry With Fewer Nodes

When a multi-node run fails, Spark Studio should offer practical recovery options.

## Example

```text
4-node launch failed.

Suggested actions:
[Retry TP 2]
[Retry TP 1]
[Ask Agent to Diagnose]
[Show Logs]
[Stop All Nodes]
```

## Why This Matters

A failed 4-node run should not feel like a dead end.

Spark Studio can guide the user toward a smaller working configuration.

---

## Export Cluster Report

Add a shareable cluster benchmark report.

## Report Should Include

- Model ID
- Recipe
- Engine
- sparkrun version
- Spark Studio version
- Node count
- Node hardware summary
- TP setting
- Launch command
- Engine version
- Load time
- TTFT
- Tokens/sec
- Memory per node
- Logs summary
- Failure summary if any
- Final recommendation

## Example Button

```text
[Export Cluster Report]
```

## Why This Matters

Multi-node users are often community users.

They will want to share what worked.

---

## Single-Node User Experience

Since many users only have one DGX Spark, including the original developer setup, multi-node features should not make Spark Studio feel limited.

If only one node is detected, show:

```text
Cluster Mode: Single Node
```

Then hide or soften advanced multi-node options.

Example:

```text
Cluster features become available when sparkrun detects 2 or more Spark nodes.
```

Optional CTA:

```text
[Learn About Multi-Node Setup]
```

Do not make single-node users feel like they are missing the main experience.

---

## Suggested Roadmap Placement

Multi-node support should be added to the roadmap, but not as Phase 1.

Recommended placement:

## Phase 1 — First Impression

Beginner install, first-run wizard, starter model, simple mode.

## Phase 2 — Install Experience

Installer, doctor command, dependency checks, LAN display.

## Phase 3 — Recovery and Trust

Recovery tools, one-click fixes, bug reports, versioned releases.

## Phase 4 — App-Like Polish

Desktop launcher, systemd service, QR code, Lite Mode.

## Phase 5 — Community Cluster Mode

- Cluster page
- Node health cards
- Multi-node launch selector
- Fits-this-cluster badges
- Readiness checks
- Per-node logs
- Retry with fewer nodes
- Cluster benchmark comparison
- Export cluster report

---

## Final Take

Multi-node support should absolutely be part of the Spark Studio roadmap.

It gives Spark Studio credibility with the DGX Spark community.

But it should be presented carefully:

```text
Beginner users get a clean single-node experience.
Advanced users get Cluster Mode powered by sparkrun.
```

The product message becomes:

```text
Spark Studio works beautifully on one DGX Spark,
and scales into a friendly control center for multi-node Spark clusters.
```

# 23. Final Product Vision

The best version of Spark Studio should feel like this:

1. User runs one command.
2. Installer checks the system.
3. Browser opens automatically.
4. Wizard recommends a model.
5. User clicks Launch.
6. Model loads.
7. User chats.
8. If anything fails, Spark Studio explains the problem and offers one fix button.

The goal is not just “more features.”

The goal is:

```text
Power-user capability with beginner-friendly onboarding.
```

Spark Studio already has the powerful part.

The next big improvement is making that power feel simple, safe, and guided.
