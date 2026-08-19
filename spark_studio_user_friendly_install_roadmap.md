# Feuille de route d'install et d'onboarding conviviale pour Spark Studio

## Objectif

Faire en sorte que Spark Studio se rapproche autant que possible d'un
**install one-click** et d'une **station de travail AI locale
beginner-friendly**.

L'expérience idéale devrait être :

```bash
curl -fsSL https://sparkstudio.dev/install | bash
```

Puis le navigateur s'ouvre sur un flux de setup guidé :

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

Après le lancement :

```text
Model loaded in 2m14s · +38 GB RAM
[Chat Now] [Benchmark] [Optimize Speed] [Stop]
```

L'utilisateur ne devrait pas avoir besoin de comprendre les
environnements Python, les flags de modèle, le YAML, les détails de
registry, le binding LAN, ou les problèmes de dépendances avant de
faire tourner son premier modèle.

---

# 1. Créer un vrai installateur one-command

## Recommandation

Ajoutez une commande d'install beginner-friendly :

```bash
curl -fsSL https://raw.githubusercontent.com/YOURNAME/Spark-Studio/main/install.sh | bash
```

À terme, upgradez vers un domaine brandé :

```bash
curl -fsSL https://sparkstudio.dev/install | bash
```

Packaging futur optionnel :

```bash
uvx spark-studio
```

ou :

```bash
pipx run spark-studio
```

## Ce que l'installateur devrait faire

L'installateur devrait :

- Vérifier Linux
- Vérifier Python 3.11
- Installer ou suggérer `uv`
- Vérifier Git
- Vérifier `nvidia-smi`
- Détecter DGX Spark / GPU NVIDIA
- Vérifier Node/npm seulement si les features d'agent sont demandées
- Vérifier Docker seulement si Web Search / SearXNG est demandé
- Vérifier si `sparkrun` existe
- Proposer de lancer `uvx sparkrun setup`
- Cloner Spark Studio
- Créer le virtualenv
- Installer les dépendances Python
- Démarrer Spark Studio
- Afficher les URLs locale et LAN

## Exemple de sortie d'install

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

## Pourquoi c'est important

Le flow clone-and-run actuel est bon pour les développeurs, mais les
débutants doivent encore comprendre Git, les dossiers, les erreurs
Python, les dépendances optionnelles, et le réseau.

Le but est de faire ressembler le setup à l'install d'une app, pas à
la configuration d'un environnement de développement.

---

# 2. Ajouter un assistant de setup au premier lancement

## Recommandation

Au premier lancement navigateur, ne lâchez pas les utilisateurs directement dans le dashboard complet.

Affichez un assistant d'onboarding guidé.

## Flow de l'assistant au premier lancement

### Étape 1 : Vérification système

Affichez un statut système clair :

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

### Étape 2 : Choisir un objectif utilisateur

Demandez ce que l'utilisateur veut faire :

- I just want to run one local model
- I want to use sparkrun community recipes
- I want to run multi-node Spark workloads
- I want Claude/Codex auto-fix
- I want WebGPU/browser models
- I want benchmarking and model comparison

### Étape 3 : Choisir un modèle de démarrage

Affichez seulement les options qui tiennent dans le hardware actuel.

Catégories recommandées :

- Fastest starter
- Best quality that fits
- Lowest memory
- Best for coding
- Best for tool calling
- Best for chat

### Étape 4 : Lancement

Utilisez un gros bouton :

```text
[Launch Recommended Model]
```

### Étape 5 : Confirmer le succès

Après que le modèle charge :

```text
Model loaded successfully

Loaded in: 2m14s
Memory used: +38 GB RAM
Endpoint: http://127.0.0.1:8000/v1

[Chat Now] [Run Benchmark] [Optimize Speed] [Stop]
```

## Pourquoi c'est important

Spark Studio a beaucoup de features puissantes. L'assistant au premier
lancement protège les débutants du sentiment d'être submergés et les
amène plus vite à un premier run réussi.

---

# 3. Ajouter Mode Débutant et Mode Avancé

## Recommandation

Ajoutez un toggle de mode d'UI :

```text
Mode: Beginner | Advanced
```

Le défaut devrait être **Mode Débutant**.

## Mode Débutant

Affichez seulement l'essentiel :

- Home
- Models
- Recipes
- Running
- Chat
- Logs
- Stop

Cachez les features avancées derrière un langage simple.

Exemple de labels :

- « Run a Model »
- « Chat »
- « See Logs »
- « Stop Model »
- « Fix Problem »

## Mode Avancé

Affichez l'interface complète pour power users :

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

## Pourquoi c'est important

Le projet est puissant, mais le nombre d'onglets/features peut intimider les nouveaux utilisateurs.

Le Mode Débutant permet à Spark Studio de rester simple sans retirer la puissance aux utilisateurs avancés.

---

# 4. Ajouter un flow de modèle de démarrage recommandé

## Recommandation

Livrez Spark Studio avec un flow « Start Here » safe pour les débutants.

L'utilisateur ne devrait pas avoir besoin de connaître YAML, les IDs
Hugging Face, les flags d'engine, ou les noms de registry pour lancer
son premier modèle.

## Boutons de démarrage suggérés

```text
[Run Fast Starter Model]
[Run Best Quality Model That Fits]
[Run Coding Model]
[Run Tool-Calling Model]
[Run Low-Memory Model]
```

## Logique de recommandation hardware-aware

Spark Studio devrait inspecter :

- Mémoire unifiée disponible
- Compte de GPU / compte de nœuds Spark
- Modèles locaux déjà téléchargés
- Engines installés
- Recipes du registry
- Score de compatibilité
- Usage mémoire attendu
- Recipes connues qui marchent

Puis recommander un défaut safe.

## Exemple d'UI

```text
Recommended for your Spark:

Qwen 2.5 7B Instruct
Reason: Fast, reliable, fits easily, good chat quality.

Engine: vLLM
Estimated memory: 24–36 GB
Expected startup: 2–4 minutes

[Launch]
```

## Pourquoi c'est important

Un débutant devrait pouvoir faire tourner un modèle sans toucher à un éditeur de recipe.

---

# 5. Ajouter des messages d'erreur en langage clair

## Recommandation

Traduisez les échecs techniques courants en explications utiles.

## Exemple : vLLM manquant

Au lieu d'afficher seulement :

```text
ModuleNotFoundError: No module named vllm
```

Affichez :

```text
vLLM is not installed yet.

Spark Studio can still run sparkrun recipes, llama.cpp, and WebGPU.

To enable vLLM, click:

[Install vLLM]

Or run:

uv pip install --python env/bin/python vllm
```

## Exemple : mémoire insuffisante

Au lieu de :

```text
CUDA out of memory
```

Affichez :

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

## Exemple : Docker manquant

```text
Docker is not installed.

Web Search with bundled SearXNG needs Docker.
Spark Studio will continue using the DuckDuckGo fallback when possible.

[Install Docker Guide]
[Continue Without Docker]
```

## Pourquoi c'est important

Des messages d'erreur conviviaux font paraître l'app stable même quand quelque chose échoue.

Les utilisateurs acceptent mieux les problèmes quand l'app explique ce qui s'est passé et donne la prochaine étape.

---

# 6. Rendre les dépendances optionnelles par feature

## Recommandation

Ne faites pas sentir aux utilisateurs qu'ils ont besoin de chaque
dépendance avant de commencer.

Organisez les dépendances par feature.

| L'utilisateur veut | Requis |
|---|---|
| Dashboard basique | Python + Git |
| Télémétrie GPU | `nvidia-smi` |
| Recipes communautaires sparkrun | `sparkrun` |
| Auto-fix Claude/Codex | Node + Claude Code / Codex CLI |
| Recherche web | Docker pour SearXNG bundled, ou recherche fallback |
| Benchmarks | llama-benchy |
| Engine vLLM | vLLM |
| Engine SGLang | SGLang |
| Engine llama.cpp | llama.cpp |
| Inférence WebGPU | Browser compatible / assets WebLLM |

## Comportement de l'UI

Chaque feature devrait afficher un de ces états :

```text
✅ Available
⚠️ Not installed
❌ Error
⬇️ Install
⏭️ Skip
```

## Exemple

```text
Claude Auto-Fix
Status: Not installed

This feature lets Spark Studio ask Claude Code to diagnose and patch broken recipes.

[Install Claude Code]
[Skip]
```

## Pourquoi c'est important

Un install 1-click ne devrait pas échouer juste parce que des features
optionnelles manquent.

L'app devrait installer le core d'abord et laisser les utilisateurs
ajouter les features avancées plus tard.

---

# 7. Packager Spark Studio de trois façons

## Recommandation

Supportez trois chemins d'install.

---

## Option A : Installateur script recommandé

Le mieux pour la plupart des utilisateurs DGX Spark.

```bash
curl -fsSL https://sparkstudio.dev/install | bash
```

Utilisez cela comme quick start principal du README.

---

## Option B : Install développeur

Le mieux pour les utilisateurs GitHub qui veulent modifier le code.

```bash
git clone https://github.com/YOURNAME/Spark-Studio.git
cd Spark-Studio
./start.sh
```

---

## Option C : Install Docker Compose

Le mieux pour les utilisateurs qui veulent de l'isolation.

```bash
docker compose up
```

Même si les engines d'inférence GPU sont gérés sur l'hôte, Docker peut
quand même être utile pour le dashboard/control plane, SearXNG, et les
services de support.

## Pourquoi c'est important

Différents utilisateurs font confiance à différents styles d'install.

Un installateur script est perçu comme facile, Git comme transparent,
et Docker comme propre.

---

# 8. Améliorer la section d'atterrissage du README

## Recommandation

Rendez le haut du README plus court, plus clair, et plus émotionnel.

Ne commencez pas par chaque feature. Commencez par le résultat.

## Ouverture README suggérée

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

## Structure README recommandée

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

## Déplacer la grande liste de features plus bas

Gardez la liste complète de features, mais déplacez-la en-dessous de :

- Quick Start
- Screenshots
- First-run experience
- Why Spark Studio?

## Pourquoi c'est important

Le README devrait vendre le projet avant de documenter chaque détail.

La plupart des utilisateurs décident s'ils continuent à lire dans les
premières secondes.

---

# 9. Ajouter des screenshots et un court GIF de démo

## Recommandation

Ajoutez des visuels près du haut du README.

## Screenshots indispensables

1. Home dashboard
2. First-run wizard
3. One-click recipe launch
4. Logs and GPU memory
5. Chat working
6. Auto-Fix button
7. Benchmark results
8. Local models tab

## Meilleur GIF de démo

Créez un GIF de 20 secondes montrant :

```text
Paste model ID
→ Click Forge
→ Click Run
→ Watch model load
→ Chat with model
```

## Section README suggérée

```markdown
## Demo

![Spark Studio demo](docs/demo.gif)

Paste a Hugging Face model ID, click Forge, launch the recommended recipe, and chat with the model.
```

## Pourquoi c'est important

Un GIF communique le produit plus vite qu'un grand README.

Pour ce genre de projet, les visuels ne sont pas optionnels — ils font
partie de l'onboarding.

---

# 10. Ajouter une commande Health Check / Doctor

## Recommandation

Ajoutez :

```bash
./spark-studio doctor
```

ou :

```bash
spark-studio doctor
```

## Exemple de sortie

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

## Le Doctor devrait vérifier

- OS
- Architecture
- Version Python
- uv
- Git
- Node/npm
- Docker
- Driver NVIDIA
- `nvidia-smi`
- Mémoire disponible
- Détection DGX Spark
- vLLM installé
- SGLang installé
- llama.cpp installé
- sparkrun installé
- Claude Code installé
- Codex installé
- llama-benchy installé
- Statut container SearXNG
- IP LAN
- Disponibilité de port

## Pourquoi c'est important

Une commande doctor donne confiance aux utilisateurs et donne aux
mainteners de meilleurs rapports de bug.

---

# 11. Ajouter les options Desktop Launcher et service Systemd

## Recommandation

Après l'install, proposez :

```text
Create desktop launcher? [Y/n]
Install as systemd service? [y/N]
Start on boot? [y/N]
```

## Desktop launcher

Créez un fichier `.desktop` pour que les utilisateurs puissent ouvrir
Spark Studio comme une app normale.

Exemple de nom d'app :

```text
Spark Studio
```

Action :

```text
Open Spark Studio Dashboard
```

## Service Systemd

Permettez :

```bash
sudo systemctl enable --now spark-studio
```

Cela rendrait Spark Studio toujours disponible à :

```text
http://spark.local:7860
```

ou :

```text
http://<spark-lan-ip>:7860
```

## Pourquoi c'est important

Une vraie app ne devrait pas nécessiter d'ouvrir un terminal à chaque fois.

C'est particulièrement utile pour une box DGX Spark dédiée.

---

# 12. Ajouter la découverte de réseau local

## Recommandation

Chaque lancement devrait clairement afficher toutes les URLs d'accès.

## Exemple

```text
Spark Studio is running:

Local:     http://127.0.0.1:7860
LAN:       http://192.168.1.50:7860
Hostname:  http://dgx-spark.local:7860
```

## Affichage dans l'UI

Affichez les mêmes URLs dans l'app :

```text
Access Spark Studio from another computer:
http://192.168.1.50:7860
```

## QR code optionnel

Affichez un QR code pour les phones/tablettes sur le même réseau.

```text
Scan to open Spark Studio on your phone
```

## Pourquoi c'est important

L'accès LAN est une des forces de Spark Studio, mais les utilisateurs
ne devraient pas avoir à trouver leur adresse IP manuellement.

---

# 13. Ajouter une section de recovery « J'ai cassé »

## Recommandation

Ajoutez à la fois des boutons de recovery dans l'UI et des commandes de
recovery dans le README.

## Boutons de recovery dans l'UI

Créez une page « Recovery » ou « Troubleshooting » avec :

- Clear failed runs
- Remove orphan containers
- Restart Spark Studio
- Reset app database
- Reset registry cache
- Rebuild Python environment
- Full safe reset

## Descriptions safe

Avant chaque action, expliquez ce qui va se passer.

Exemple :

```text
Reset app database

This removes saved recipes, run history, benchmark history, and settings.
It does not delete downloaded models.

[Reset Database]
```

Exemple :

```text
Remove orphan containers

This stops containers that Spark Studio launched but can no longer control.
It does not delete model files.

[Clean Containers]
```

## Section README

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

## Pourquoi c'est important

Les débutants sont plus enclins à expérimenter quand ils savent qu'il y
a un chemin de reset safe.

---

# 14. Ajouter de meilleurs choix à l'install

## Recommandation

Pendant l'install, demandez quel type de setup l'utilisateur veut.

## Exemple

```text
Choose setup type:

1. Basic — dashboard, recipes, local runs
2. Recommended — Basic + sparkrun + model manager
3. Full — Recommended + Claude/Codex + Web Search + Benchmarks
4. Custom

Select [2]:
```

## Défauts suggérés

Utilisez **Recommended** comme défaut.

## Profils de setup

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

## Pourquoi c'est important

Une seule commande d'install peut quand même rester user-friendly tout
en évitant un install de dépendance géant tout-ou-rien.

---

# 15. Ajouter un meilleur dashboard d'accueil

## Recommandation

L'écran d'accueil devrait répondre à cinq questions immédiatement :

1. Is my Spark healthy?
2. Is a model running?
3. What can I launch?
4. How much memory is free?
5. What should I do next?

## Cartes de dashboard suggérées

### Statut système

```text
DGX Spark detected
GPU ready
Unified memory: 72 GB free / 128 GB total
```

### Modèle actif

```text
No model running

[Launch Recommended Model]
```

ou :

```text
Qwen 2.5 7B Instruct
Loaded in 2m14s · +38 GB RAM

[Chat] [Benchmark] [Optimize] [Stop]
```

### Action suivante recommandée

Exemples :

```text
Start by launching a beginner-friendly model.
```

```text
Your model is running. Try chatting or run a quick benchmark.
```

```text
This run failed. Auto-Fix can diagnose and patch the recipe.
```

### Santé des features

```text
✅ sparkrun
✅ vLLM
⚠️ Docker
⚠️ Claude Code
```

## Pourquoi c'est important

L'écran d'accueil devrait agir comme un centre de contrôle, pas juste
une liste d'onglets.

---

# 16. Ajouter « One-Click Fix » partout

## Recommandation

Partout où Spark Studio détecte un problème, affichez une action suivante.

Exemples :

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

## Pourquoi c'est important

L'utilisateur ne devrait pas avoir à lire les logs pour savoir quoi
faire ensuite.

---

# 17. Ajouter un meilleur export de rapport de bug

## Recommandation

Ajoutez un bouton « Copy Bug Report ».

## Le rapport devrait inclure

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

## Exemple de bouton

```text
[Copy Bug Report]
```

## Pourquoi c'est important

Cela rend les issues GitHub plus propres et plus faciles à débugger.

Cela aide aussi les utilisateurs à demander de l'aide à Claude, Codex,
ou à la communauté.

---

# 18. Ajouter des releases versionnées

## Recommandation

Créez des releases GitHub :

```text
v0.1.0
v0.2.0
v0.3.0
```

## Ajouter une commande d'update

```bash
spark-studio update
```

ou :

```bash
./start.sh --update
```

## Ajouter un check d'update dans l'UI

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
