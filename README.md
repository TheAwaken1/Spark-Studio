# Spark Studio

**Votre DGX Spark, un dashboard convivial.** Lancez des modèles locaux en un clic, surveillez la mémoire et les logs en direct, chattez, benchmarkez — et quand une recipe casse ou tourne lentement, laissez Claude Code ou Codex diagnostiquer, patcher, et la relancer pour vous (votre propre abonnement Pro/Max/Plus, sans clés API).

Exécute des recipes communautaires **vLLM**, **SGLang**, **llama.cpp**, **WebGPU (WebLLM)**, et **sparkrun**.

## Démarrage rapide

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/TheAwaken1/Spark-Studio/main/install.sh)
```

ou clonez vous-même :

```bash
git clone https://github.com/TheAwaken1/Spark-Studio.git
cd Spark-Studio
./start.sh
```

Ouvrez **http://127.0.0.1:7860** (ou `http://<ip-spark>:7860` depuis toute machine de votre LAN). Au premier lancement, l'environnement Python est mis en place automatiquement, puis un **assistant d'installation** vérifie votre système, recommande un modèle adapté à votre Spark, et le lance.

> **Note de sécurité :** le dashboard n'a pas d'authentification et se lie
> à toutes les interfaces pour que votre LAN puisse l'utiliser. Quiconque
> sur votre réseau peut le contrôler, et — via la configuration HTTPS/Caddy
> — ouvrir **Hermes Chat, qui est un véritable shell interactif tournant
> sous votre utilisateur**. Ne l'exécutez que sur un réseau où vous faites
> confiance à tous les appareils (pas de Wi-Fi invité/partagé), utilisez
> `./start.sh --host 127.0.0.1` pour un usage local uniquement, et n'exposez
> jamais le port sur internet.

```bash
./start.sh --doctor   # rapport de santé complet du système, à tout moment
```

## Ce que vous pouvez faire en 60 secondes

1. Ouvrez le dashboard — l'assistant vérifie votre Spark et choisit un modèle de démarrage
2. Cliquez sur **Lancer** — suivez la progression du chargement et l'utilisation de la mémoire unifiée en direct
3. **Chattez** avec le modèle dès qu'il sert
4. Cliquez sur **Benchmark** pour tok/s + TTFT, ou **Optimiser la vitesse** pour laisser un agent le régler
5. En cas d'échec, **Correction auto & Reprise** lit les logs et patche la recipe

## Démo

<!-- TODO: capture docs/demo.gif — paste a HF model id → Forge → Run → chat -->
*Captures d'écran et GIF de démo à venir.*

## Fonctionnalités

- **Assistant d'installation au premier lancement** — sur une installation fraîche, le navigateur s'ouvre sur un flux guidé : vérification système complète (`/api/doctor`), choix d'un objectif (chat rapide / meilleure qualité / code / agents & outils / mémoire réduite), et lancement d'un modèle **recommandé pour votre matériel** — classé à partir de recipes éprouvées sur votre Spark, de modèles déjà sur disque, de recipes validées par la communauté, et de votre propre historique de benchmarks. Rouvrable à tout moment via **Assistant d'installation** sur l'onglet Vue d'ensemble
- **Mode Débutant / Avancé** — un toggle dans la barre latérale. Débutant affiche uniquement l'essentiel (Vue d'ensemble, Recipes, Modèles, Chat, Runs & Logs) ; Avancé affiche tous les onglets d'engine, Forge, Benchmarks, et Agents. Les nouvelles installations démarrent en Débutant ; les installations existantes restent en Avancé
- **UI Dashboard** avec des onglets dédiés par engine (vLLM, SGLang, llama.cpp, WebGPU). Les listes de runs sont nommées par modèle/recipe (pas par ids hex), les éditeurs de recipe sont des éditeurs Monaco complets avec coloration YAML/JSON/shell, et sur téléphone/tablette la barre latérale se replie en menu coulissant — pratique pour vérifier les signes vitaux ou les logs depuis le canapé
- **Prêt pour le LAN** — `./start.sh` se lie à toutes les interfaces pour que chaque machine de votre réseau puisse utiliser l'app (pas de login ; ne l'exposez pas sur internet)
- **UI Zero-CDN, prête hors ligne** — les fontes (Inter / JetBrains Mono), icônes Font Awesome, Monaco, Chart.js, et highlight.js sont bundlées sous `web/vendor/` et servies localement ; le dashboard est pleinement fonctionnel sur une box en pare-feu ou hors ligne. L'onglet du navigateur montre l'engine en cours d'exécution (`▶ vllm · Spark Studio`) et l'app est installable en tant que PWA
- **Runner de recipe par drop-zone** — collez ou déposez une recipe JSON, cliquez sur Run, streamez les logs en direct
- **Lancez n'importe quoi** — une zone sur l'onglet Recipes accepte un lien de benchmark [spark-arena.com](https://spark-arena.com/leaderboard) (ou le blurb de partage complet), un YAML/JSON de recipe, un id de modèle HuggingFace, ou un `@community/ref`, et l'exécute. Les imports Arena sont sauvegardés dans Mes Recipes automatiquement
- **Bouton Ask Claude / Ask Codex** sur chaque run en échec — lit la recipe + les 300 dernières lignes de log, inline la référence du schéma de recipe sparkrun (RECIPES.md), les recipes curatoriales correspondantes, et les patches de fix depuis le miroir local du registry, et renvoie une recipe patchée avec diagnostic
- **Login navigateur Hugging Face** — connectez la CLI officielle `hf` depuis **Agents & Identités** via son flow OAuth device-code. Spark Studio affiche le nom d'utilisateur connecté et permet les téléchargements de modèles privés/gated sans lire ni stocker le token
- **Correction auto & Reprise** — un clic sur un run échoué démarre une boucle mains libres : l'agent diagnostique, patche la recipe, relance, surveille les nouveaux logs, et réessaie avec un contexte frais — jusqu'à 3 tentatives — jusqu'à ce que l'engine serve réellement. Fini de cliquer sur Fix en boucle
- **Optimiser la vitesse** — la même boucle mains libres, mais pour les runs *lents* au lieu des cassés : un clic sur un run sain le benchmarke (tok/s + TTFT), passe à l'agent les chiffres mesurés, les signes vitaux GPU/mémoire en direct, et la connaissance de tuning DGX Spark (backends FlashInfer vs Marlin, configs kernel `sparkrun tune`, quantification du cache KV, variables d'environnement par famille depuis les registries eugr/sparkrun), relance la recipe tunée, et re-benchmarke. La configuration qui a *mesuré* la plus rapide est celle qui reste en service et est sauvegardée sur la recipe — un patch qui bench plus lentement est annulé automatiquement. Déclare la victoire à ≥10% d'amélioration (`SPARK_STUDIO_OPTIMIZE_MARGIN`)
- **Recipe Forge** — collez n'importe quel id de repo Hugging Face ; Spark Studio consulte le registry de recipes synchronisé pour un YAML validé pour Spark, se rabat sur les recipes du registry adaptées, puis sur des presets heuristiques. Chaque résultat est badgé pour distinguer la vérité terrain d'un guess. Des chips starter en un clic font remonter vos forges récentes et les modèles validés Spark depuis le registry
- **Vérification de compatibilité matérielle** — sonde la box locale via `nvidia-smi`, badge chaque recipe avec **Tient sur ce Spark** / **Nécessite N GPUs** / **Trop gros**
- **Vérification de compatibilité** — verdict allant de `excellent` à `too-large` avec raisons
- **Bibliothèque de recipes** (SQLite) — sauvegardez, éditez, taguez, partagez (copier/coller en JSON entre utilisateurs), importez/exportez. Les nouvelles recipes sont créées dans Recipe Forge — et **chaque lancement direct depuis un onglet d'engine est aussi auto-sauvegardé** (tout engine, taggé `auto-saved`, dédupliqué par engine+model), donc un run qui a fonctionné est toujours à un clic dans Mes Recipes avec son badge engine et ✓ working
- **sparkrun maintenu à jour** — `start.sh` lance `sparkrun update` à chaque démarrage (à passer avec `--no-sparkrun-update` ou `SPARK_STUDIO_NO_SPARKRUN_UPDATE=1`), et la barre d'outils Community recipes a un bouton **Update sparkrun** avec un sélecteur de canal : Stable (PyPI), Beta (develop), Alpha (bleeding edge), ou YOLO (`--yolo`, alias pour alpha). Le canal choisi est mémorisé par sparkrun pour les futures mises à jour — y compris la mise à jour automatique au lancement — et la barre d'outils affiche la version installée
- **Recipes communautaires via sparkrun** — parcourez le registry mirroir `@official`/`@experimental` et lancez sur votre mesh Spark en un clic. Le sélecteur Nodes (TP) filtre vers les recipes qui tiennent réellement dans votre nombre de nœuds (les recipes multi-Spark sont badgées et masquées à 1 nœud) ; Stop passe par `sparkrun stop`, et si cela échoue (id de job obsolète, erreur sparkrun) Spark Studio force-supprime les containers du job lui-même — Stop signifie toujours stoppé. Chaque lancement auto-sauvegarde une recipe dans Mes Recipes (dédupliquée par ref) pour qu'elle soit toujours à un clic. Les recipes sparkrun sauvegardées peuvent porter des options de lancement (`args._sparkrun.max_model_len` et `-o key=value` `overrides` arbitraires) qui s'appliquent à chaque relancement — pratique pour épingler des workarounds à un modèle
- **Badges serveur ✓ working / ✗ failed** — un watchdog sonde `/v1/models` de chaque engine et tague la recipe dès qu'elle commence à servir (ou échoue), même avec tous les onglets navigateur fermés. Il rattrape aussi le plus méchant mode d'échec de sparkrun : l'engine qui crash dans un container qui reste "Up" — le run est marqué échoué, la vraie traceback est tirée du log de service in-container pour Ask Claude, et le container zombie est démantelé. Les chargements lents multi-nœuds peuvent étendre le délai never-ready via `SPARK_STUDIO_SPARKRUN_GRACE` (secondes, défaut 1200)
- **Runs résistants au redémarrage** — le service systemd préserve les groupes de processus d'engine et les logs appartenant aux enfants pendant que seul le dashboard redémarre. Au boot Spark Studio réconcilie sa base de runs avec la réalité, vérifie la santé de chaque endpoint `/v1/models` retenu, et restaure automatiquement **Ready**, les logs, Stop, le chat, le label du modèle, et le lien de recipe ; les lignes orphelines sont marquées exited. Les sessions `./start.sh` directes déchargent toujours leurs engines au Ctrl+C normal sauf si `SPARK_STUDIO_KEEP_RUNS_ON_EXIT=1` est défini
- **Auto-sync des registries** — les trois repos upstream sont rafraîchis à chaque démarrage ; un badge ✨ montre les recipes arrivées depuis le dernier sync
- **Modèles locaux** — scanne chaque cache HF (variables d'environnement *et* caches référencés par vos recipes), affiche les tailles réelles sur disque, « Servir avec vLLM » / « Forge » / **Supprimer** en un clic pour libérer de l'espace disque
- **Chat & Canvas / Engine Chat** — éditeur Monaco + chat, cible automatiquement l'engine en cours d'exécution ; rend des graphiques Chart.js, des cartes d'export Word/Excel, et des réponses grounded sur le web. Ajuste automatiquement `max_tokens` à la fenêtre de contexte réelle de l'engine à chaque tour
- **Benchmarks** — bench rapide tok/s + TTFT, plus sweeps complets [llama-benchy](https://github.com/eugr/llama-benchy) (pp/tg à profondeur, concurrence, prefix caching). Chaque résultat enregistre la version de l'engine ; comparez deux runs côte à côte, et copiez un rapport markdown partageable (hardware + engine + recipe + résultats) pour la communauté
- **Tool Eval Bench** — répond à « à quel point ce modèle est-il *utile* ? », pas seulement à sa vitesse. 12 cas déterministes notent cinq compétences de 0 à 100 : **sélection d'outil** (choisir le bon outil parmi cinq), **extraction d'arguments** (dates/montants/noms exacts dans les args d'outil), **retenue** (répondre directement au lieu d'appels d'outils spurieux), **utilisation des résultats d'outil** dans la réponse finale, et sortie **strict JSON**. Chaque cas montre ce que le modèle a réellement fait (`called get_weather({"city": "Tokyo"})`) ; les modèles de raisonnement reçoivent un budget de token juste et les blocs `<think>` sont strippés avant vérification. Chaque eval sauvegarde un rapport markdown + JSON dans `tooleval-results/` et les scores sont conservés dans l'historique par modèle. Si l'engine a été lancé sans tool calling, le bench le dit au lieu de noter zéro silencieusement
- **CLI `sparkstudio` + Hermes Agent Lab** — chattez avec le modèle chargé, lancez des benches de vitesse/outils, confiez une vraie tâche de dépôt à [Hermes Agent](https://github.com/NousResearch/hermes-agent), ou notez les modèles sur des fixtures de code jetables. Elle suit le [NVIDIA DGX Spark Hermes playbook](https://build.nvidia.com/spark/hermes-agent/instructions) : l'endpoint OpenAI-compatible local actif est configuré comme custom provider de Hermes, tandis que les rapports, diffs, sortie de tests, et télémétrie Spark sont conservés pour comparaison
- **Hermes Chat dans le dashboard** — le TUI Hermes Ink complet tourne dans un onglet xterm.js à travers un vrai bridge POSIX PTY/WebSocket. Les slash commands, `/model`, les approbations, l'activité d'outils, le switching local/Claude/Codex, et la recherche web Spark Studio fonctionnent exactement comme dans le terminal. La session survit aux swaps de recipe, switches de provider, et reloads de page : Chat continue de tourner pendant qu'un engine charge (ou sans aucun), un reload se rattache au même agent live, et le provider **Spark Studio** de `/model` liste toujours la recipe qui sert actuellement — via un passthrough stable `/api/engine/v1` qui suit l'engine actif à travers les ports, donc switcher local ↔ Claude/Codex en pleine session marche tout simplement
- **Télémétrie de chargement sur chaque run** — les cartes de run affichent combien de temps le modèle a mis à devenir ready et combien de RAM unifiée il a réclamé (`chargé en 3m42s · +38.2 GB RAM`), estampillé au moment où l'engine répond pour la première fois. Les stats persistent dans l'historique des runs et survivent aux redémarrages de l'app ; les endpoints adoptés/externes (déjà chargés) affichent honnêtement rien au lieu d'un chiffre bidon
- **Garde mémoire pré-lancement** — sur le pool unifié de 128 GB du DGX Spark, chaque modèle remplit la plupart du pool, donc un seul tient à la fois. Avant de lancer, Spark Studio arrête tout autre modèle résident, attend que sa mémoire soit réellement libérée, et bloque un lancement qui ne tiendrait toujours pas (avec un « lancer quand même » en un clic) — pour que swapper des modèles n'OOM pas la box ou ne fasse pas tomber le dashboard avec. Voir [Protection mémoire / OOM](#protection-mémoire--oom)
- **Onglet WebGPU** — inférence in-browser via MLC WebLLM, avec extraction d'attachements PDF/CSV/XLSX et recherche web intégrée (SearXNG bundlé, démarré automatiquement)
- **Patching en crash-loop** — acceptez la recipe patchée de Claude / Codex en un clic ; le run reprend immédiatement
- **États de run honnêtes** — les badges distinguent **failed** (crash, rouge) de **stopped** (vous avez cliqué Stop) et les sorties propres, pour qu'une page de runs terminés ne ressemble pas à une page d'erreurs
- **Page Cluster (multi-nœud via sparkrun)** — cartes de santé de nœuds avec **télémétrie live par nœud** (CPU, mémoire unifiée, util/temp/power GPU — streamée depuis `sparkrun cluster monitor --json`, pour que les Sparks distants rapportent de vrais chiffres sans que Spark Studio ne SSH nulle part), dispo TP d'un coup d'œil, **vérifications de readiness de lancement** en clair avant un run multi-nœud, viewer de log de service par nœud pour les containers de job locaux, et **Réessayer avec moins de nœuds** sur les lancements multi-nœud échoués. Les boxes mono-Spark voient une vue « Single Node » propre avec un pointeur vers comment faire un mesh — rien ne semble manquer
- **Gateway OpenAI-compatible** — pointez n'importe quel client (Continue, Cursor, etc.) vers le `:<port>/v1` du run actif

## Prérequis

- **Linux** (NVIDIA DGX Spark / aarch64 recommandé ; x86_64 marche aussi)
- **Python 3.11**
- **Git**
- **Node.js + npm** (seulement nécessaire pour les features d'agents Claude Code et Codex)
- **uv** (recommandé) — `pip install uv` — ou `pip` standard
- **nvidia-smi** disponible dans le PATH pour la télémétrie GPU

Vos engine(s) d'inférence installés séparément :
- [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [SGLang](https://docs.sglang.ai/start/install.html)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

Extras optionnels :
- [llama-benchy](https://github.com/eugr/llama-benchy) pour les sweeps de benchmark complets — `uv pip install --python env/bin/python llama-benchy`
- [sparkrun](https://github.com/spark-arena/sparkrun) pour les recipes communautaires multi-nœud — `uvx sparkrun setup` (assistant cluster guidé)
- [Hermes Agent](https://hermes-agent.nousresearch.com/) pour l'Agent Lab local — installé en un clic depuis **Hermes → Chat**, pinné sur la release `v2026.7.20` (commande manuelle ci-dessous)
- **Docker** pour les recipes spark-vllm-docker et la recherche web SearXNG bundled

## Installation

### Option A — Installateur one-command

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/TheAwaken1/Spark-Studio/main/install.sh)
```

Vérifie votre système (git, Python/uv, GPU, Docker, npm), clone dans
`~/spark-studio`, bootstrap l'environnement, lance le doctor, et propose
d'installer les pièces optionnelles par profil :

| Profil | Ajoute |
|---|---|
| `--basic` | dashboard + recipes + runs locaux |
| `--recommended` *(défaut)* | + sparkrun CLI (recipes communautaires, kernel tuning) |
| `--full` | + llama-benchy + CLIs d'agent Claude/Codex + Hermes Agent + Hermes Mod pinné |

Les installs pipées (`curl … \| bash`) ne peuvent pas prompter, donc elles
tournent en non-interactif complet avec les défauts ci-dessus ; utilisez la
forme `bash <(curl …)` pour avoir les prompts. `--dir <path>` choisit
l'emplacement d'install.

### Option B — Clone et start

```bash
git clone https://github.com/TheAwaken1/Spark-Studio.git
cd Spark-Studio
./start.sh
```

C'est tout — au premier lancement `start.sh` crée le virtualenv `./env` et
installe `requirements.txt` automatiquement (via **uv** s'il est installé,
qui peut aussi fetch Python 3.11 pour vous ; sinon `python3 -m venv` + pip
standard).

<details>
<summary>Setup manuel (si vous préférez faire les étapes vous-même)</summary>

```bash
# venv — uv (recommandé) ou Python standard
uv venv env --python 3.11        # ou : python3.11 -m venv env

# dépendances
uv pip install --python env/bin/python -r requirements.txt   # ou : env/bin/pip install -r requirements.txt
```

</details>

### 2. (Optionnel) Pré-télécharger les registries de recipes

Spark Studio mirror trois repos upstream localement pour usage hors ligne.
L'app clone et rafraîchit automatiquement à chaque démarrage, donc cette
étape n'est nécessaire que si vous voulez les mirrors en place avant le
premier boot (par ex. install offline) :

```bash
mkdir -p data/registry
git clone --depth 1 https://github.com/spark-arena/recipe-registry.git data/registry/recipe-registry
git clone --depth 1 https://github.com/eugr/spark-vllm-docker.git    data/registry/spark-vllm-docker
git clone --depth 1 https://github.com/spark-arena/sparkrun.git        data/registry/sparkrun
```

### 3. (Optionnel) Connecter les CLIs d'agent, Hugging Face, et Hermes

Pour les boutons **Ask Claude** et **Ask Codex** :

```bash
npm install -g @anthropic-ai/claude-code @openai/codex
```

Après installation, loguez-vous depuis **Agents & Identités** dans Spark Studio — pas de clés API nécessaires, juste votre flow OAuth navigateur.

Spark Studio installe déjà la CLI Hugging Face officielle. Dans **Agents & Identités**, cliquez sur **Log in** sous Hugging Face, ouvrez l'URL navigateur affichée, et confirmez le code one-time. La CLI `hf` possède le credential sauvegardé ; Spark Studio ne fait que vérifier `hf auth whoami` pour afficher votre nom d'utilisateur public. Cela active les téléchargements de repos privés et modèles gated que votre compte a accepté.

Pour l'Agent Lab, ouvrez **Hermes → Chat** et cliquez sur **Install Hermes** — c'est
tout le setup. Spark Studio lance l'installateur per-user officiel (pinné
sur la release `v2026.7.20`), détecte la CLI automatiquement, et connecte
Hermes à quelque modèle que ce soit qui soit actuellement chargé. Il n'y a
pas d'assistant auquel répondre : le profil isolé est généré depuis
l'endpoint d'engine live (URL, id de modèle, longueur de contexte) et
rafraîchi automatiquement avant chaque run, donc un port d'engine changé
n'a jamais besoin d'édition manuelle. La commande manuelle équivalente est :

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/v2026.7.20/scripts/install.sh \
    | bash -s -- --commit 3ef6bbd201263d354fd83ec55b3c306ded2eb72a --non-interactive
./start.sh --install-cli
```

#### Optionnel : `hermes` standalone hors de Spark Studio

**Sautez cette section sauf si vous voulez aussi utiliser `hermes` brut
avec votre profil personnel `~/.hermes`.** Rien dans Spark Studio
(dashboard Chat, `sparkstudio hermes`, Agent Lab benchmarks) ne le
requiert. Si vous lancez bien l'assistant de setup interactif de Hermes
pour votre profil personnel, chargez d'abord le modèle dans Spark Studio,
copiez l'endpoint montré dans **Engine Chat** (ajoutez `/v1`, par ex.
`http://127.0.0.1:41293/v1` — le port change à chaque run, donc utilisez
toujours l'URL courante), et répondez :

| Prompt de l'installateur | Réponse |
|---|---|
| Install ripgrep and ffmpeg? | **Yes** |
| Import or migrate OpenClaw configuration? | **No** |
| How would you like to set up Hermes? | **Blank Slate** |
| Select provider | **Custom endpoint (enter URL manually)** |
| API base URL | **`http://127.0.0.1:41293/v1`** (ou l'URL Engine Chat courante plus `/v1`) |
| API key | **Laisser vide** |
| Model | Sélectionnez le modèle rapporté par l'endpoint chargé |
| Context length | **Laisser vide pour auto-détection** |
| Display name | Acceptez le défaut, ou utilisez **Spark Studio Local** |
| Terminal backend | **Local** |
| What next? | **Start with everything disabled — finish now** |

Rechargez le shell et vérifiez l'intégration :

```bash
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
which hermes
sparkstudio agent doctor
sparkstudio hermes
sparkstudio agent cases
sparkstudio agent eval --suite coding-smoke
```

Spark Studio réécrit son profil Hermes isolé avec l'URL d'engine
actuellement active avant chaque run Agent Lab et avant
`sparkstudio hermes`, donc un changement de port ultérieur ne nécessite pas
d'éditer votre configuration Hermes personnelle.

## Lancement

```bash
./start.sh
```

(Le premier lancement configure aussi l'environnement Python — voir Installation.)

### Lancez-le comme une app

```bash
./start.sh --install-launcher   # "Spark Studio" dans votre menu d'applications
./start.sh --install-service    # service utilisateur systemd (auto-start, self-updating)
./start.sh --install-cli        # ~/.local/bin/sparkstudio
```

Le **launcher desktop** démarre le serveur si besoin et ouvre le dashboard —
pas de terminal. Le **service** survit aux reboots (`loginctl enable-linger $USER`
pour tourner sans être logué), garde les modèles en service à travers les
redémarrages (`KEEP_RUNS_ON_EXIT` baked into the unit), et active les updates
in-app en un clic : quand une nouvelle version est sur GitHub, le badge de
version dans la sidebar s'allume — cliquez et le service pull, rafraîchit
les dépendances, et se redémarre. Si votre service a été installé avant
que ce comportement soit ajouté, lancez `./start.sh --install-service` une
fois pour installer la politique de redémarrage qui préserve les modèles.
Sans le service, le badge apparaît toujours ; l'update s'applique et vous
redémarrez `./start.sh` vous-même.

Cliquez sur la ligne de version dans la sidebar à tout moment pour un
**QR code** — scannez-le pour ouvrir Spark Studio sur un téléphone ou une
tablette sur le même réseau.

### Health check (doctor)

```bash
./start.sh --doctor
```

Affiche un rapport système complet — OS/GPU/driver, mémoire unifiée, Docker,
chaque engine (vLLM/SGLang/llama.cpp), sparkrun, agents Claude/Codex,
llama-benchy, SearXNG, et vos URLs de dashboard — avec un fix en une
ligne pour tout ce qui manque. Le code de sortie est non-zéro quand un
check critique échoue, donc c'est scriptable. Le même rapport est servi
sur `GET /api/doctor` pour l'UI et les rapports de bug.

## CLI `sparkstudio` + Hermes Agent Lab

Démarrez Spark Studio, lancez un modèle tool-capable, et vérifiez le path complet :

```bash
sparkstudio status
sparkstudio models
sparkstudio search "latest NVIDIA DGX Spark documentation"
sparkstudio search "latest NVIDIA DGX Spark documentation" --enrich
sparkstudio chat "Write a Python function that validates a recipe"
sparkstudio bench speed
sparkstudio bench tools
sparkstudio agent doctor
sparkstudio hermes
```

`sparkstudio search` utilise le pipeline `/api/search` existant du dashboard :
SearXNG configuré en premier, SearXNG bundled en second, puis fallback
DuckDuckGo. Ajoutez `--enrich` pour fetcher le texte lisible des top pages,
`--limit 1..10` pour dimensionner le set de résultats, ou placez le
`--json` global avant `search` pour l'automatisation.

Utilisez `sparkstudio hermes` (alias : `sparkstudio agent chat`) pour une
session Hermes interactive qui suit le modèle actuellement chargé dans le
dashboard. Le `hermes` brut lit intentionnellement votre `~/.hermes/config.yaml`
personnel ; si ce profil contient un port d'engine plus ancien, il
continuera d'essayer l'ancien modèle. Les sessions interactives Spark
Studio reçoivent aussi un outil MCP read-only, `mcp__sparkstudio__web_search`.
Il route chaque query à travers le pipeline SearXNG/DDG de Spark Studio.
Le serveur MCP n'expose aucun browser, shell, credential, ou API d'écriture
génériques. Les résultats de recherche sont traités comme du matériel
source non trusted et incluent leurs URLs pour citation.

La vue **Hermes → Chat** du dashboard est le même vrai Hermes TUI, pas
une seconde implémentation de chat. Ouvrez l'onglet après qu'un modèle
soit ready et il démarre contre ce modèle chargé automatiquement. Choisissez
un workspace avant de démarrer ; le changer nécessite d'arrêter et
redémarrer le TUI. Utilisez **Stop** ou **Restart** pour le lifecycle du
PTY, et tapez `/model` dans Hermes pour bouger entre le modèle local et
n'importe quel provider Claude ou Codex authentifié. Le profil isolé
`data/agent-lab/hermes` et le serveur MCP de recherche Spark Studio sont
partagés avec `sparkstudio hermes`, donc les deux points d'entrée se
comportent de manière cohérente.

Pour rendre Claude, Codex, ou Copilot disponibles à `/model` dans Hermes
Chat, authentifiez le **profil isolé** — pas votre profil personnel :

```bash
sparkstudio agent auth add openai-codex   # ou anthropic-claude, copilot, …
```

`hermes auth add <provider>` brut écrit dans votre `~/.hermes` personnel,
que le dashboard ne lit jamais délibérément, donc les credentials ajoutés
de cette manière afficheront toujours « needs setup » dans `/model`.
`sparkstudio agent auth` wrappe `hermes auth` avec le `HERMES_HOME` du
dashboard ; utilisez-le aussi pour `list` et les autres sous-commandes
d'auth. Le modèle local n'a besoin d'aucun credential. Spark Studio
préfère le bridge WebSocket et retombe automatiquement sur un transport
terminal HTTPS same-origin quand un navigateur, une politique de
certificat, ou un proxy réseau rejette WSS. Le fallback porte les mêmes
octets PTY, l'input clavier, les events de resize, et le comportement de
cleanup ; aucun setting utilisateur n'est requis.

Le widget hardware de la sidebar utilise un stream de télémétrie séparé,
read-only. Un navigateur ou un proxy peut brièvement reconnecter ce stream
sans interrompre Spark Studio, le modèle chargé, ou Hermes. L'UI garde la
dernière bonne lecture pendant les retries courts et étiquette un retry
plus long comme **Télémétrie en reconnexion**.

### Hermes Learning

La vue **Hermes → Learning** contrôle la connaissance durable pour le
profil isolé de Spark Studio. Les sessions interactives dashboard et
`sparkstudio hermes` activent trois capacités complémentaires par défaut :

- **Memory** garde un ensemble borné de faits d'environnement, d'engine, de modèle, et de recipe.
- **Skills** préservent des procédures réutilisables pour fixer, finetuner, benchmarker, et optimiser des recipes ou modèles.
- **Session search** retrouve des détails pertinents depuis des conversations antérieures.

Le **User profile** est aussi activé par défaut pour qu'Hermes puisse se
rappeler des préférences personnelles optionnelles, mais il a son propre
toggle et peut être désactivé sans éteindre la mémoire technique. Aucune
de ces features ne retrain ou n'altère les poids du modèle ; elles
fournissent un contexte durable qui suit le profil Hermes isolé quand
`/model` switch entre un modèle local, Claude, ou Codex.

**Review learning before saving** est activé par défaut. Les écritures de
mémoire et de skill proposées apparaissent comme cartes d'approbation
au-dessus du terminal Chat embarqué, montrant le contenu avec des
boutons explicites **Approuver** et **Rejeter**. Vous n'avez pas besoin
de taper `approved` dans le prompt. Les commandes `/memory pending` et
`/skills pending` restent disponibles pour la revue terminal-first.
L'apprentissage approuvé est disponible aux nouvelles sessions Chat,
donc redémarrez un Chat déjà en cours pour le charger en contexte. Les
anciennes skills staged sont normalisées au schéma Hermes courant et
clairement étiquetées **Repair & Approve**. Les cartes de fichiers de
support restent désactivées tant que leur carte create-skill précédente
n'est pas approuvée, prévenant les échecs d'ordre de dépendance. Les
évaluations de modèle déterministes de l'Agent Lab gardent
intentionnellement le toolset constrained stateless, empêchant les
réponses apprises de contaminer les scores de comparaison.

Pour le travail de recipe, Spark Studio enregistre la skill curatoriale
`sparkrun-recipes`. Elle dirige les agents vers la référence
`Title Recipe Format.md` du dépôt capturée depuis la documentation
officielle de format de recipe sparkrun, sans injecter toute la
référence dans des prompts non liés.

La même zone **Hermes** inclut aussi **Skin Studio**, alimenté par
l'add-on pinné [`hermes-mod@0.2.0`](https://github.com/cocktailpeanut/hermes-mod)
— un grand merci à [@cocktailpeanut](https://github.com/cocktailpeanut)
pour l'avoir construit et open-sourcé. À la première utilisation,
cliquez sur **Install add-on** ; si Hermes manque, le bouton installe
Hermes d'abord puis continue automatiquement. Le profil one-command
`--full` préinstalle les deux. Spark Studio stocke l'add-on sous
`data/addons/hermes-mod` git-ignored, le démarre sur loopback, et
l'embarque à travers un bridge same-origin sandboxé. Node.js et npm
sont les seuls prérequis de l'add-on. Il cible toujours
`data/agent-lab/hermes`, jamais votre `~/.hermes` personnel. Créez ou
chargez un skin, cliquez sur **Activate** dans Skin Studio, puis
choisissez **Restart Chat to apply**. Spark Studio convertit
automatiquement le markup multi-ligne de logo et de couleur de hero en
la forme orientée ligne requise par le TUI Ink. Utilisez le picker
**Saved skin** et **Delete skin** pour supprimer un skin custom ;
supprimer le skin actif ramène Hermes en toute sécurité au skin par
défaut. **Use original Hermes** sélectionne le design par défaut
intégré sans supprimer de skins sauvegardés ; choisissez **Restart Chat
to apply** après avoir switché de design. Seul le PTY Hermes redémarre
— l'engine d'inférence chargé continue de tourner. La sélection
d'image utilise le picker du navigateur, donc elle marche aussi via le
dashboard privé HTTPS/LAN de Spark Studio.

Parce qu'Hermes peut lancer des commandes terminal et éditer des
fichiers, le terminal WebSocket accepte les navigateurs loopback et les
navigateurs LAN privé same-origin via HTTPS/WSS. Les connexions HTTP
plain depuis un autre device et les adresses remote publiques sont
rejetées. Le setup Caddy bundled (`https://<Spark-IP>:8443`) satisfait
automatiquement l'exigence LAN privé chiffré. Pour outrepasser
délibérément cette frontière quand un autre transport de confiance
protège déjà Spark Studio, démarrez avec :

```bash
SPARK_STUDIO_HERMES_TUI_ALLOW_REMOTE=1 ./start.sh
```

N'activez pas l'accès terminal remote non restreint sur un réseau non
de confiance. La validation WebSocket Origin reste enforced dans tous
les modes. Le fallback HTTPS requiert un custom request header
same-origin-only et un session ID non-devinable. Le fallback HTTPS
garde un lease de session chiffré de dix minutes pour que revenir d'un
preview ouvert ou reload le dashboard se rattache au même Hermes TUI.
Appuyer sur **Stop** le termine toujours immédiatement ; les sessions
abandonnées et leurs helpers sont reaped automatiquement quand le lease
expire.

**Réglages de retry Caddy recommandés.** Par défaut, Caddy renvoie un
502 instantané quand le dashboard est en plein redémarrage
(`./start.sh --update`, l'updater in-app, ou
`systemctl --user restart spark-studio`). Avec un onglet dashboard
ouvert, ça peut laisser des panneaux à moitié chargés — le plus visible
étant un iframe Skin Studio dont les assets n'ont pas chargé. Dites à
Caddy de tenir pendant les redémarrages au lieu de ça en donnant à la
directive `reverse_proxy` une fenêtre de retry :

```caddyfile
https://<Spark-IP>:8443 {
    tls internal
    reverse_proxy 127.0.0.1:7860 {
        # Tient pendant les redémarrages du dashboard au lieu de renvoyer 502 :
        # maintient chaque requête jusqu'à 15s, en retry toutes les 500ms.
        lb_try_duration 15s
        lb_try_interval 500ms
    }
}
```

Les requêtes faites pendant un redémarrage se mettent alors en pause
brièvement et complètent une fois le dashboard de retour, donc les
updates sont invisibles aux onglets navigateur ouverts. Notez qu'avec
`admin off` dans le Caddyfile, `caddy reload` ne peut pas marcher —
redémarrez le proxy pour appliquer les changements :
`systemctl --user restart spark-studio-https`.

Lancez la suite coding smoke déterministe via Hermes :

```bash
sparkstudio agent cases
sparkstudio agent eval --suite coding-smoke

# Plus de confiance, et workers parallèles si le modèle chargé peut les gérer
sparkstudio agent eval --trials 3 --jobs 2 --fail-below 70
```

Chaque cas démarre comme un petit dépôt Git avec des tests qui échouent
connu. Hermes reçoit le toolset `file,terminal`, édite le dépôt, lance
ses tests, et est noté seulement par une commande de test post-run
propre. Les rapports markdown et JSON atterrissent sous
`data/agent-lab/results/` ; `sparkstudio agent history` et
`sparkstudio agent show <run-id>` retrouvent l'historique backed SQLite.

Utilisez le même harness sur votre propre dépôt :

```bash
sparkstudio agent run \
  "Find the cause of the failing tests, implement the smallest fix, and rerun them" \
  --repo /path/to/project
```

Par défaut, cela crée un worktree détaché depuis le `HEAD` courant du
dépôt, préservant le checkout original et tout travail non committé. Le
résultat imprime le workspace retenu et le diff. Passez `--in-place`
seulement quand vous voulez qu'Hermes édite le checkout original.

L'Agent Lab utilise un profil dédié à `data/agent-lab/hermes` pour qu'il
ne réécrive jamais vos settings Hermes personnels. Les sessions
`sparkstudio hermes` interactives ajoutent l'unique outil MCP de
recherche à `file,terminal` ; les `agent run` et `agent eval` non
attentifs restent restreints à `file,terminal` et interdisent
explicitement l'usage réseau. Les smart approvals, denials pour les
commandes communes à haut risque, et Hermes checkpoints restent
activés. C'est un guide de process, pas un sandbox OS : relisez les
changements générés avant de les utiliser et évitez `--unsafe-yolo`
sauf si le workspace est jetable.

Pour un serveur externe OpenAI-compatible, placez les overrides globaux
avant la commande :
`sparkstudio --base-url http://127.0.0.1:41293/v1 --model MODEL agent eval`.
Ajoutez `--json` à la même position pour l'automatisation.

ou manuellement :

```bash
env/bin/python -m uvicorn server:app --host 0.0.0.0 --port 7860
```

Puis ouvrez **http://127.0.0.1:7860** dans votre navigateur, ou joignez-le
depuis toute machine de votre réseau à `http://<IP-LAN-de-cette-machine>:7860`.

Vous pouvez utiliser n'importe quel port disponible — changez juste
`7860` pour ce que vous préférez (`./start.sh --port 8000` marche aussi ;
les args supplémentaires sont passés à uvicorn). Si le port est déjà
pris, `start.sh` vous dit quel process le tient et suggère une
alternative au lieu d'échouer en plein boot. Pour restreindre l'accès
à cette machine uniquement, utilisez `--host 127.0.0.1`.

> **Note :** l'app n'a pas d'authentification intégrée — quiconque sur
> votre réseau peut l'utiliser. N'exposez pas le port sur internet.

Arrêter l'app avec **Ctrl+C décharge aussi tout ce qu'elle a lancé** —
processus d'engine, containers docker, et workloads sparkrun — donc les
modèles ne restent pas sur le GPU après que le dashboard soit parti.
Si vous *voulez* qu'un modèle continue de servir à travers les
redémarrages de l'app, démarrez avec
`SPARK_STUDIO_KEEP_RUNS_ON_EXIT=1 ./start.sh` ; le boot suivant le
réadopte automatiquement.

Si `ufw` est activé, autorisez le port pour votre LAN :

```bash
sudo ufw allow from 192.168.0.0/24 to any port 7860 proto tcp
```

### Variables d'environnement optionnelles

| Variable | Rôle | Défaut |
|---|---|---|
| `HF_HOME` | Override la racine du cache HuggingFace (hub sous `$HF_HOME/hub`) | `~/.cache/huggingface` |
| `HF_HUB_CACHE` | Pointe directement sur un répertoire hub (dossiers `models--*`) | `$HF_HOME/hub` |
| `HF_HUB_ENABLE_HF_TRANSFER` | Téléchargements HF plus rapides via `hf_transfer` | non défini |
| `SEARXNG_URL` | Pointe la recherche web sur une instance SearXNG spécifique (override le container bundled) | auto-détecté |
| `SPARK_STUDIO_NO_SPARKRUN_UPDATE` | Mettez à `1` pour skip le `sparkrun update` automatique que `start.sh` lance au boot (équivalent à `./start.sh --no-sparkrun-update`) | non défini (auto-update) |
| `SPARK_STUDIO_SPARKRUN_GRACE` | Secondes pendant lesquelles un run sparkrun peut rester not-ready avant que le watchdog ne l'échoue (s'applique seulement quand aucun signal de container local n'est disponible, par ex. heads multi-nœud remote) | `1200` |
| `SPARK_STUDIO_KEEP_RUNS_ON_EXIT` | Mettez à `1` pour donner à la sortie d'engine un log durable, laisser les modèles en service quand l'app exit, et réadopter les endpoints sains comme Ready au boot suivant | non défini (modèles déchargés) |
| `SPARK_STUDIO_NO_MEMORY_GUARD` | Mettez à `1` pour désactiver la garde mémoire unifiée pré-lancement (stop-and-wait + fit check avant de lancer un modèle) | non défini (garde active) |
| `SPARK_STUDIO_MEM_GUARD_TIMEOUT` | Secondes max que la garde attend que la mémoire d'un modèle arrêté soit réclamée avant de continuer | `120` |
| `SPARK_STUDIO_AGENT_TIMEOUT` | Secondes à attendre une réponse Claude/Codex avant d'abandonner | `420` |
| `SPARK_STUDIO_AUTOFIX_WAIT` | Secondes que Auto-Fix & Retry (et Optimiser la vitesse) attend un engine relancé avant de juger la tentative | `1800` |
| `SPARK_STUDIO_OPTIMIZE_MARGIN` | Pourcentage d'amélioration tok/s sur la baseline auquel Optimiser la vitesse déclare le succès et s'arrête tôt | `10` |
| `SPARK_STUDIO_CORS_ORIGINS` | Origines séparées par des virgules autorisées à appeler l'API cross-origin (off par défaut — l'UI est same-origin et l'app n'a pas d'auth) | non défini (pas de CORS) |

L'onglet Models scanne tout ce qui précède **plus** tout cache que vos
recipes sauvegardées passent aux containers d'engine via
`-e HF_HUB_CACHE=…`, donc les modèles apparaissent même quand seules
les recipes savent où ils vivent.

Exemple :
```bash
HF_HOME=/mnt/models/.cache/huggingface \
HF_HUB_ENABLE_HF_TRANSFER=1 \
env/bin/python -m uvicorn server:app --host 0.0.0.0 --port 7860
```

### Recherche web

Le toggle globe du chat ancre les réponses dans des résultats web live.
Au boot, l'app auto-démarre un container **SearXNG** bundled
(`spark-searxng`, image officielle `searxng/searxng`, lié à
`127.0.0.1`). La config vit dans `data/searxng/settings.yml`
(seuls les engines fiables et sans clé sont activés, sortie JSON on).
Priorité du backend : override env `SEARXNG_URL` → container bundled →
tout SearXNG sur un port local well-known → fallback **DuckDuckGo**
(`ddgs`). Requiert Docker ; si Docker est absent, la recherche retombe
transparrement sur DuckDuckGo.

La recherche est plus que des liens : les requêtes news/trending sont
routées vers des index de news dédiés avec des fenêtres de fraîcheur,
les résultats sont dédupliqués par domaine, et les top pages sont
**fetchées et leur texte d'article extrait** (lxml) pour que le modèle
réponde depuis du vrai contenu avec citations de source inline — pas
depuis des snippets de homepage. Le chain-of-thought des modèles de
raisonnement se rend comme une section repliable « Thinking » au lieu
de polluer la réponse.

## Prometheus / Grafana

Chaque instance Spark Studio expose des métriques [Prometheus](https://prometheus.io)
standard sur `/metrics` — utilisation/température/power/clock GPU, mémoire
unifiée, CPU, et gauges de readiness par run. Pour des dashboards
historiques à travers un mesh, ajoutez un scrape job par Spark et
pointez Grafana sur Prometheus :

```yaml
scrape_configs:
  - job_name: spark-studio
    static_configs:
      - targets: ["192.168.0.132:7860", "192.168.0.133:7860"]
```

Les métriques engine au niveau token (throughput, usage KV-cache,
profondeur de queue) viennent des engines eux-mêmes — le run vLLM actif
sert son propre `/metrics` sur son port ; scrape ça en parallèle. Pas
besoin de stack Kubernetes/DCGM, mais si vous en avez une, ces
endpoints se scrapent de la même façon.

## Protection mémoire / OOM

DGX Spark partage un pool de 128 GB entre GPU et RAM système, donc un
modèle qui overcommit peut drive la box jusqu'à un vrai OOM. Le wizard
`sparkrun setup` installe **earlyoom** pour tuer un workload en fuite
avant que le kernel ne se lock — bien — mais sa liste `--prefer` par
défaut inclut `python`, et le dashboard de Spark Studio *est* un
processus `python`. Sous pression mémoire, earlyoom peut alors SIGKILL
le dashboard de ~100 MB aux côtés du modèle de plusieurs GB (vous
verrez un `Killed` brut dans le terminal où vous avez lancé
`./start.sh`).

Spark Studio mitige cela depuis son côté automatiquement :

- **Garde mémoire pré-lancement.** Parce que chaque modèle remplit la
  plupart du pool, un seul tient à la fois. Avant de démarrer un
  modèle, l'app arrête tout autre modèle résident, **attend que sa
  mémoire unifiée soit réellement réclamée** (le lag teardown/reclaim
  est exactement ce qui cause des OOM en back-to-back), et refuse un
  lancement qui ne tiendrait toujours pas — l'UI offre un « lancer
  quand même » en un clic. Cela prévient l'OOM à la source au lieu de
  nettoyer après. L'empreinte estimée vient du `gpu-memory-utilization`
  / `mem-fraction-static` de la recipe (× le pool de 128 GB) ;
  llama.cpp ne remplit pas le pool donc il est exempt du fit-check
  dur. Override avec le flag `force` sur un lancement, ou globalement
  avec `SPARK_STUDIO_NO_MEMORY_GUARD=1`.
- **Reclaim mémoire post-stop.** Sur le GB10, la mémoire unifiée d'un
  modèle arrêté reste pinnée après que le process CUDA exit — `free`
  continue d'afficher ~100 GB utilisés sans rien qui tourne (un quirk
  DGX Spark connu ; le fix manuel est
  `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`). Après chaque
  stop/exit de modèle, l'app attend que les containers/process du run
  soient vraiment partis et flush les caches automatiquement ; la garde
  pré-lancement flush aussi avant de refuser un lancement pour manque
  de mémoire. Le flush nécessite root, donc accordez cette unique
  commande narrow passwordless (écrire dans `drop_caches` n'évince que
  des caches réclamables — ça ne peut pas nuire aux workloads en cours) :

  ```bash
  echo "$USER ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" \
    | sudo tee /etc/sudoers.d/spark-studio-reclaim
  sudo chmod 0440 /etc/sudoers.d/spark-studio-reclaim
  ```

  Sans la règle, le log de run affiche une ligne `[reclaim]` avec ce hint exact.
- **Priorité OOM.** Au démarrage, l'app essaie de baisser sa propre
  priorité OOM (seulement possible avec privilège, par ex. une unité
  systemd avec `OOMScoreAdjust=-500`), et — la partie qui marche
  toujours — elle **élève la priorité OOM de chaque sous-processus
  d'engine qu'elle lance**, donc si la pression mémoire arrive, ça tue
  le modèle relaunchable avant le control plane.

Pour le path docker / sparkrun (où le modèle tourne dans un container
que l'app ne possède pas), appliquez le fix box-wide une fois — retirez
`python` de la liste `--prefer` d'earlyoom :

```bash
sudo sed -i.bak 's/|python3|python)/)/' /etc/default/earlyoom
sudo systemctl restart earlyoom
pgrep -a earlyoom   # vérifiez : le groupe --prefer ne contient plus python
```

Les engines d'inférence sont toujours matchés par leurs vrais noms
(`vllm`, `sglang`, `llama-server`, …), donc earlyoom continue de
protéger la box — il arrête juste de traiter le dashboard comme un
sacrifice préféré.

## Mise à jour

```bash
./start.sh --update
```

Pull le dernier code, rafraîchit les dépendances Python, rapporte le
changement de version, et démarre l'app — une seule commande pour être
à jour *et* en train de tourner.

Équivalent manuel :

```bash
git pull

uv pip install --python env/bin/python -r requirements.txt --upgrade

# Update agent CLIs
npm install -g @anthropic-ai/claude-code @openai/codex
```

Les mirrors de registry se rafraîchissent à chaque démarrage d'app (ou
cliquez sur **Refresh now** sur l'onglet Forge) — pas de commandes git
manuelles nécessaires.

## J'ai cassé — Safe Reset

L'onglet **Recovery** (visible aussi en Mode Débutant) a des actions
en un clic, clairement étiquetées : clear les runs terminés, retirer
les containers orphelins (ne touche jamais aux containers que vous
gérez vous-même ou aux jobs sparkrun actifs), reset le cache du
registry, reset la base de l'app (confirm-gated ; les modèles sur
disque sont intouchés), et **Copier le rapport de bug** — un bundle
markdown de la santé système + la recipe du run en échec et les 300
dernières lignes de log (secrets redactés), prêt pour une issue
GitHub ou un agent.

Depuis le terminal, quand l'app elle-même ne démarre pas :

```bash
rm -f data/spark_studio.db     # clear les recipes/historique sauvegardés uniquement
rm -rf env && ./start.sh       # rebuild l'environnement Python
rm -rf env data/spark_studio.db && ./start.sh   # full reset
```

Les modèles téléchargés vivent dans le cache HF et survivent à
chaque reset ci-dessus.

## Schéma de recipe

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

### Contexte, batch size, et capabilities (vLLM)

Chaque recipe vLLM — qu'elle soit **forgée**, **créée**, ou **éditee**
— est normalisée pour qu'elle serve le **contexte complet que le
modèle supporte, plafonné à 262144**, avec
`max-num-batched-tokens: 16384`. Le plafond utilise le contexte natif
du modèle (depuis sa config HF), donc un modèle 262144-native comme
Qwen3 obtient le 256K complet tandis qu'un modèle 131072 comme
Llama-3.1 obtient 131072 — vLLM n'a jamais à être demandé pour plus
de contexte que le modèle n'en autorise, donc les recipes lancent
toujours. (vLLM dimensionne le cache KV sur `gpu-memory-utilization`,
pas sur `max-model-len`, donc un plafond élevé n'OOM pas — ça trade
juste un peu de concurrence max.)

**Reasoning et tool calling** sont ajoutés automatiquement pour les
familles de modèles reconnues (Qwen3, GLM-4.7, gpt-oss, MiniMax-M2,
Nemotron, Gemma-4, …) — Forge wire le bon `--tool-call-parser` /
`--reasoning-parser` et `--enable-auto-tool-choice`. Les familles non
reconnues n'obtiennent rien, parce qu'un mauvais parser casse le
service. L'**éditeur de recipe** affiche une ligne **Capabilities**
avec des toggles Tool calling / Reasoning (chacun activé seulement
quand un parser est connu pour ce modèle) pour que vous puissiez
overrider par recipe ; `GET /api/recipes/capabilities?model=<repo>`
renvoie ce qu'un modèle donné supporte et le contexte qui sera
appliqué.

Valeurs d'engine : `vllm` | `sglang` | `llamacpp`. Le runner mappe
`args` vers des flags CLI automatiquement — les clés en kebab-case
deviennent `--kebab-case`, les booléens deviennent des flags bare.

## API

Une fois lancé sur `http://127.0.0.1:7860` :

### JavaScript

```js
// Liste les recipes
const recipes = await fetch('/api/recipes').then(r => r.json());

// Démarre un run
const run = await fetch('/api/runs', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    engine: 'vllm',
    args: { model: 'meta-llama/Llama-3.1-8B-Instruct', 'max-model-len': 16384 },
  }),
}).then(r => r.json());

// Stream les logs
const es = new EventSource(`/api/runs/${run.id}/stream`);
es.addEventListener('log', ev => console.log(ev.data));

// Chat contre l'engine actif
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

# Forge des recipes pour un modèle HF
resp = httpx.get(f"{BASE}/api/hf/forge", params={"repo": "meta-llama/Llama-3.1-8B-Instruct"}).json()
print(resp["report"]["verdict"], resp["recipes"][0])

# Lance un run
run = httpx.post(f"{BASE}/api/runs", json={
    "engine": "vllm",
    "args": {"model": "meta-llama/Llama-3.1-8B-Instruct", "max-model-len": 16384},
}).json()

# Demande à Claude de fixer un run cassé
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

### Map d'endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/system` | GPU / engine / platform info |
| `GET` | `/api/doctor` | Full system health report (same as `./start.sh --doctor`) |
| `GET` | `/api/recommend?k=` | Starter-model recommendations per goal, ranked from local signals |
| `POST` | `/api/recovery/{clear-runs\|clean-containers\|reset-registry\|reset-db}` | One-click recovery actions (reset-db needs `{"confirm":true}`) |
| `GET` | `/api/bugreport?run_id=` | Markdown bug report: doctor + run + recipe (redacted) + logs |
| `GET` `POST` | `/api/update/check` `/api/update/apply` | Update check against origin/main / pull + deps (+ self-restart under systemd) |
| `GET` | `/api/cluster` `/api/cluster/readiness?tp=` | Spark-mesh node health + TP availability / pre-launch checks |
| `GET` | `/api/cluster/monitor` | SSE: live per-node CPU/mem/GPU/power via `sparkrun cluster monitor --json` |
| `GET` | `/api/sparkrun/nodelog?container=` | Per-node serve-log tail for local sparkrun job containers |
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
| `GET` | `/api/agents/status` | Claude/Codex/Hugging Face install + login state |
| `GET` | `/api/agents/login/{claude\|codex\|huggingface}` | SSE browser OAuth/device flow |
| `POST` | `/api/agents/fix` | JSON-structured recipe patch |
| `GET` | `/api/agents/autofix/{rid}` | SSE hands-free fix loop (diagnose → patch → relaunch → retry) |
| `GET` | `/api/agents/optimize/{rid}` | SSE speed-optimization loop (bench → patch → relaunch → re-bench; fastest config wins) |
| `POST` | `/api/chat` | Proxy to active engine (OpenAI-compatible, auto-fits `max_tokens`) |
| `GET` `POST` | `/api/engine/v1/{models\|chat/completions\|completions\|embeddings\|rerank}` | Stable OpenAI-compatible base URL for whichever engine is live — recipes swap ports, this URL doesn't (used by Hermes Chat's `/model` picker; stale model ids retarget to the served model) |
| `GET` `POST` | `/api/bench[...]` | Run and list quick benchmarks |
| `GET` `POST` | `/api/benchy/...` | llama-benchy: status, run (SSE), history |
| `GET` | `/api/benchy/{id}/export` | Shareable markdown benchmark report |
| `POST` | `/api/tooleval/run` | Start the Tool Eval Bench against a run (defaults to the active engine) |
| `GET` | `/api/tooleval/status` | Live progress, per-case results, and scores of the current/last eval |
| `GET` | `/api/tooleval/history` | Past Tool Eval scores per model (reports live in `tooleval-results/`) |
| `GET` | `/api/hermes-mod/status` | Pinned Skin Studio install/runtime state, active skin, and isolated profile |
| `POST` | `/api/hermes-mod/{install\|start\|stop}` | Manage the optional loopback Hermes Mod sidecar |
| `POST` | `/api/hermes-mod/skins/default` | Select the built-in original Hermes skin without deleting custom skins |
| `DELETE` | `/api/hermes-mod/skins/{name}` | Delete one custom skin from the isolated Spark Studio Hermes profile |
| `GET` `POST` `PUT` `DELETE` | `/api/hermes-mod/ui/...` | Sandboxed, token-protected iframe bridge to Hermes Mod |
| `GET` | `/api/agentlab/status` | Hermes install state and its isolated Spark Studio profile |
| `POST` | `/api/agentlab/install` | Run the fixed official Hermes per-user installer from the private dashboard |
| `GET` | `/api/agentlab/terminal/status` | Embedded Hermes TUI readiness, active model, workspace, and access mode |
| `GET` `PUT` | `/api/agentlab/learning` | Isolated Hermes memory, user-profile, skills, session-search, and write-approval preferences |
| `WS` | `/api/agentlab/terminal` | Byte-safe PTY bridge for the dashboard Hermes Chat tab (same-origin; local-only by default) |
| `POST` `GET` `DELETE` | `/api/agentlab/terminal/sessions[...]` | Same-origin HTTPS PTY fallback: create, poll output, send input/resize, and close |
| `GET` | `/api/agentlab/history` | Saved free-form and deterministic Agent Lab runs |
| `GET` | `/api/agentlab/{run-id}` | One Agent Lab result, including report and workspace metadata |
| `GET` | `/api/spark/vitals` | Live GPU / unified-memory telemetry (SSE) |
| `GET` | `/metrics` | Prometheus exposition: GPU/CPU/memory + run gauges (scrape each Spark for Grafana history) |
| `GET` | `/api/images` `/api/images/probe?ref=` | Spark-vLLM runner images / vLLM+FlashInfer versions inside one |
| `GET` | `/api/images/build?mode=` | SSE: run build-and-copy.sh (nightly / wheels / allowlisted advanced flags) |

## Plateforme

Linux + NVIDIA. Testé sur DGX Spark (Grace Blackwell, aarch64). vLLM et
SGLang sont Linux-first ; llama.cpp est cross-platform mais wired ici
pour GPU offload.

## Communauté & crédits

Spark Studio est construit sur le travail de la communauté DGX Spark / GB10 :

- [spark-arena/recipe-registry](https://github.com/spark-arena/recipe-registry) — le registry de recipes curatoriales que cette app mirror et forgé depuis
- [spark-arena/sparkrun](https://github.com/spark-arena/sparkrun) — launcher de workload multi-nœud, intégré en tant que runner
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — orchestration docker canonique pour les recipes Spark (utilisé verbatim pour les runs docker)
- [eugr/llama-benchy](https://github.com/eugr/llama-benchy) — le moteur de benchmark derrière l'onglet Benchmarks
- [Spark Arena leaderboard](https://spark-arena.com/leaderboard) — hub de benchmark communautaire ; collez ses YAMLs de recipe directement dans les onglets d'engine, et partagez vos propres résultats avec le bouton ⧉ report

## Licence

MIT — voir [LICENSE](LICENSE). Les assets vendored sous `web/vendor/`
(Monaco Editor, Chart.js, marked, DOMPurify, highlight.js,
qrcode-generator, Font Awesome Free) conservent leurs propres licences ;
voir les headers dans chaque fichier.
