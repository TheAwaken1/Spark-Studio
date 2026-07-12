// Spark Studio dashboard

// ---------- utilities -----------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const h = (tag, props = {}, ...children) => {
  const el = document.createElement(tag);
  Object.entries(props).forEach(([k, v]) => {
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) {
      if (typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (typeof v === 'string') el.setAttribute(k, v);
    }
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (v !== false && v != null) el.setAttribute(k, v);
  });
  children.flat().forEach((c) => {
    if (c == null) return;
    el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  });
  return el;
};
const api = async (path, opts = {}) => {
  const r = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? (typeof opts.body === 'string' ? opts.body : JSON.stringify(opts.body)) : undefined,
  });
  // 507 = pre-launch memory guard blocked the launch (won't fit in unified
  // memory). Offer a one-click "launch anyway" that retries with force:true —
  // centralized here so every launch call site (engine tabs, recipes,
  // sparkrun, forge) gets it without duplicating the flow.
  if (r.status === 507 && opts.method === 'POST' && !opts._forced) {
    let detail = await r.text();
    try { detail = JSON.parse(detail).detail || detail; } catch { /* keep raw */ }
    if (confirm(`${detail}\n\nLaunch anyway?`)) {
      const body = typeof opts.body === 'string' ? JSON.parse(opts.body || '{}') : (opts.body || {});
      return api(path, { ...opts, _forced: true, body: { ...body, force: true } });
    }
    throw new Error('Launch cancelled — not enough unified memory for another model.');
  }
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  if (r.status === 204) return null;
  return r.json();
};
const fmtTime = (s) => (s ? new Date(s * 1000).toLocaleTimeString() : '—');

// Monaco loads once, up front, so every consumer (canvas editor, recipe
// editors) awaits the same promise and shares the 'spark' theme.
window.monacoReady = new Promise((resolve) => {
  window.require(['vs/editor/editor.main'], () => {
    monaco.editor.defineTheme('spark', {
      base: 'vs-dark', inherit: true,
      rules: [{ background: '121826' }],
      colors: { 'editor.background': '#121826' },
    });
    resolve(window.monaco);
  });
});

// ---------- tab switching -------------------------------------------------
$$('.tab').forEach((t) =>
  t.addEventListener('click', () => {
    $$('.tab').forEach((x) => x.classList.remove('active'));
    $$('.panel').forEach((x) => x.classList.remove('active'));
    t.classList.add('active');
    const panel = $(`.panel[data-panel="${t.dataset.tab}"]`);
    if (panel) panel.classList.add('active');
    if (t.dataset.tab === 'overview') refreshOverview();
    if (['vllm', 'sglang', 'llamacpp'].includes(t.dataset.tab)) refreshEnginePanel(t.dataset.tab);
    if (t.dataset.tab === 'vllm') refreshEngineImages();
    if (t.dataset.tab === 'recipes') { refreshRecipes(); refreshSparkrun(); }
    if (t.dataset.tab === 'models') refreshLocalModels();
    if (t.dataset.tab === 'logs') refreshRuns();
    if (t.dataset.tab === 'bench') refreshBenchTab();
    if (t.dataset.tab === 'agents') refreshAgents();
    if (t.dataset.tab === 'cluster') refreshCluster();
    if (t.dataset.tab !== 'cluster') stopClusterMonitor(); // one ssh-fanout stream only while watching
    if (t.dataset.tab === 'forge') refreshForgeSuggest();
    if (t.dataset.tab === 'chat') syncChatTarget();
    if (t.dataset.tab === 'webgpu') { refreshWgServerStatus(); refreshWgSearchStatus(); startAmbient(); }
    else { stopAmbient(); }
  })
);
$$('[data-goto]').forEach((b) =>
  b.addEventListener('click', () => $(`.tab[data-tab="${b.dataset.goto}"]`)?.click())
);

// Mobile off-canvas sidebar
const navToggle = $('#navToggle');
const navScrim = $('#navScrim');
function closeSidebar() {
  $('.sidebar')?.classList.remove('open');
  if (navScrim) navScrim.hidden = true;
}
if (navToggle && navScrim) {
  navToggle.addEventListener('click', () => {
    const open = $('.sidebar').classList.toggle('open');
    navScrim.hidden = !open;
  });
  navScrim.addEventListener('click', closeSidebar);
  $$('.tab').forEach((t) => t.addEventListener('click', closeSidebar));
}

// ---------- system panel --------------------------------------------------
window._hostInfo = null;
async function refreshSystem() {
  try {
    const [sys, host] = await Promise.all([
      api('/system'),
      api('/host').catch(() => null),
    ]);
    window._hostInfo = host;
    $('#sysVllm').innerHTML = sys.engines.vllm ? '<span class="badge ok">ok</span>' : '<span class="badge no">missing</span>';
    $('#sysSglang').innerHTML = sys.engines.sglang ? '<span class="badge ok">ok</span>' : '<span class="badge no">missing</span>';
    $('#sysLlama').innerHTML = sys.engines.llamacpp ? '<span class="badge ok">ok</span>' : '<span class="badge no">missing</span>';
    const hostBlock = host && host.gpu_count ? `
      <div><strong>${escapeHtml(host.summary)}</strong></div>
      ${host.mesh_size > 1 ? `<div class="muted">Mesh: ${host.cluster_nodes.length} Sparks (${host.cluster_nodes.map(escapeHtml).join(', ')})</div>` : '<div class="muted">Solo (no Spark mesh configured)</div>'}
      ${host.gpus.map((g) => `<div class="muted">GPU ${g.index}: ${escapeHtml(g.name)} · ${g.memory_gb} GB · driver ${escapeHtml(g.driver)}</div>`).join('')}
    ` : '<div class="muted">nvidia-smi unavailable</div>';
    $('#systemDetail').innerHTML =
      `<div>${escapeHtml(sys.platform)}</div>` +
      `<div>Python ${escapeHtml(sys.python)}</div>` +
      hostBlock;
    // Sidebar footer: version + how to reach this dashboard from the LAN.
    // Clicking it opens the QR modal for phones/tablets.
    const meta = $('#sidebarMeta');
    if (meta && sys.version) {
      const lan = (sys.urls?.lan || [])[0];
      window._lanUrls = sys.urls || {};
      const upd = window._updateAvailable
        ? ` · <span class="update-badge" title="Click for update options">⬆ v${escapeHtml(window._updateAvailable)}</span>` : '';
      meta.innerHTML = `v${escapeHtml(sys.version)}${lan ? ` · <span class="mono">${escapeHtml(lan)}</span>` : ''}${upd}`;
      meta.style.cursor = 'pointer';
    }
  } catch (e) { console.error(e); }
  try {
    const a = await api('/agents/status');
    const agentBadge = (s) => s.logged_in
      ? '<span class="badge ok" title="Logged in and ready">ready</span>'
      : (s.installed
        ? '<span class="badge warn" title="Installed — log in on the Agents tab">log in</span>'
        : '<span class="badge no" title="CLI not installed — see Agents tab">missing</span>');
    $('#sysClaude').innerHTML = agentBadge(a.claude);
    $('#sysCodex').innerHTML = agentBadge(a.codex);
  } catch (e) { /* ignore */ }
  updateTitle();
}

// Reflect the active engine in the browser tab so it's findable at a glance.
async function updateTitle() {
  try {
    const act = await api('/active');
    document.title = act?.engine ? `▶ ${act.engine} · Spark Studio` : 'Spark Studio';
  } catch { document.title = 'Spark Studio'; }
}

// ---------- overview ------------------------------------------------------
async function refreshOverview() {
  const active = await api('/active').catch(() => null);
  $('#activeRunCard').innerHTML = active
    ? `<div style="font-weight:600;overflow-wrap:anywhere">${escapeHtml(runLabel(active))}</div>
       <div style="margin-top:4px"><span class="badge ok">${active.engine}</span> <code title="${escapeHtml(active.cmd)}">${active.id.slice(0, 8)}</code>${active.ready ? '' : ' <span class="muted">starting…</span>'}</div>
       ${loadStats(active) ? `<div class="muted" style="margin-top:4px"><i class="fa-solid fa-gauge"></i> ${loadStats(active)}</div>` : ''}
       <div style="margin-top:8px"><a class="btn" href="${active.url}/v1/models" target="_blank">/v1/models</a>
       <button class="btn danger" onclick="window.sparkStop('${active.id}')">Stop</button></div>`
    : 'No engine running.';
  const runs = await api('/runs').catch(() => []);
  $('#overviewRuns').innerHTML = runs.slice(0, 6).map(renderRunRow).join('') || '<div class="muted">No runs yet.</div>';
}

function runOutcome(r) {
  return r.outcome || r.status;
}

function runLabel(r) {
  // Model id / recipe ref beats a hex run id for humans scanning the list.
  return r.label || r.ref || r.engine;
}

function fmtDur(secs) {
  if (secs == null) return '';
  if (secs < 60) return `${Math.round(secs)}s`;
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return m >= 60 ? `${Math.floor(m / 60)}h${m % 60}m` : `${m}m${String(s).padStart(2, '0')}s`;
}

// "loaded in 3m42s · +38.2 GB RAM" once the engine first answered.
function loadStats(r) {
  if (r.load_secs == null) return '';
  const ram = r.ram_delta_gb != null ? ` · ${r.ram_delta_gb >= 0 ? '+' : ''}${r.ram_delta_gb} GB RAM` : '';
  return `loaded in ${fmtDur(r.load_secs)}${ram}`;
}

function renderRunRow(r) {
  const outcome = runOutcome(r);
  const stats = loadStats(r);
  return `<div class="run-item" title="run ${r.id}" onclick="document.querySelector('.tab[data-tab=logs]').click();setTimeout(()=>window.selectRun('${r.id}'),50)">
    <div class="ri-top"><span class="ri-engine">${r.engine}</span><span class="ri-status ${outcome}">${outcome}</span></div>
    <div class="ri-name">${escapeHtml(runLabel(r))}</div>
    <div class="ri-id">${r.port ? ':' + r.port + ' · ' : ''}${r.id.slice(0, 8)} · ${fmtTime(r.started_at)}${stats ? ' · ' + stats : ''}</div>
  </div>`;
}

window.sparkStop = async (rid) => { await api(`/runs/${rid}/stop`, { method: 'POST' }); refreshOverview(); updateTitle(); };

// ---------- engine panels (generated) ------------------------------------
const ENGINE_DEFAULTS = {
  // DGX Spark has 128 GB unified memory — use full native context lengths.
  vllm:    { model: 'meta-llama/Llama-3.1-8B-Instruct', 'max-model-len': 131072, 'max-num-batched-tokens': 16384, 'gpu-memory-utilization': 0.9, 'trust-remote-code': true },
  sglang:  { model: 'meta-llama/Llama-3.1-8B-Instruct', 'context-length': 131072, 'mem-fraction-static': 0.88, 'trust-remote-code': true },
  llamacpp: { model: 'bartowski/Llama-3.1-8B-Instruct-GGUF', 'ctx-size': 32768, 'n-gpu-layers': 999, 'flash-attn': true },
};

// Spark-vllm-docker recipe templates. Paste any spark-arena recipe in this format
// and Spark Studio expands {placeholders}, drops empty flags, and wraps in docker.
const SPARK_TEMPLATES = {
  vllm: `model: cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit
container: sparkrun-eugr-vllm-tf5
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 1
  gpu_memory_utilization: 0.8
  max_model_len: 262144
  max_num_batched_tokens: 16384
  tool_call_parser: gemma4
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
command: |
  vllm serve {model} \\
    --max-model-len {max_model_len} \\
    --kv-cache-dtype {kv_cache_dtype} \\
    --gpu-memory-utilization {gpu_memory_utilization} \\
    --host {host} \\
    --port {port} \\
    --load-format {load_format} \\
    --enable-prefix-caching \\
    --tool-call-parser {tool_call_parser} \\
    --enable-auto-tool-choice \\
    --reasoning-parser {reasoning_parser} \\
    --async-scheduling \\
    --max-num-batched-tokens {max_num_batched_tokens} \\
    --trust-remote-code \\
    -tp {tensor_parallel} \\
    -pp {pipeline_parallel}
runtime: vllm
metadata:
  description: Gemma4-26B-A4B-INT4-AWQ — AWQ quantized Gemma 4 26B MoE
  kv_dtype: fp8
recipe_version: '2'
`,
  sglang: `model: meta-llama/Llama-3.1-8B-Instruct
container: lmsysorg/sglang:latest
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 1
command: |
  python -m sglang.launch_server \\
    --model-path {model} \\
    --host {host} \\
    --port {port} \\
    --tp {tensor_parallel}
runtime: sglang
`,
  llamacpp: `model: bartowski/Llama-3.1-8B-Instruct-GGUF
container: ghcr.io/ggerganov/llama.cpp:server-cuda
defaults:
  port: 8000
  host: 0.0.0.0
  ctx_size: 8192
  n_gpu_layers: 999
command: |
  llama-server \\
    -hf {model} \\
    --host {host} \\
    --port {port} \\
    --ctx-size {ctx_size} \\
    --n-gpu-layers {n_gpu_layers} \\
    --flash-attn
runtime: llamacpp
`,
};

// ---- Spark recipe expander -----------------------------------------------
function expandSparkRecipe(yamlText) {
  const r = jsyaml.load(yamlText);
  if (!r || !r.command) throw new Error('Recipe missing `command` field');
  const defaults = {
    pipeline_parallel: 1,
    load_format: 'auto',
    reasoning_parser: '',
    kv_cache_dtype: r.metadata?.kv_dtype || 'auto',
    ...(r.defaults || {}),
  };
  const lookup = (k) => (k === 'model' ? r.model : defaults[k]);

  // Drop entire "  --flag {placeholder} \" lines when the placeholder is empty
  // (e.g. --reasoning-parser when no reasoning model).
  let cmd = r.command.replace(
    /^([ \t]*)(-[\w-]+|--[\w-]+)([ \t]+)\{(\w+)\}([ \t]*\\?[ \t]*\r?\n?)/gm,
    (match, indent, flag, sp, key, tail) => {
      const v = lookup(key);
      if (v === undefined || v === null || v === '') return '';
      return match;
    }
  );
  // Substitute remaining placeholders
  cmd = cmd.replace(/\{(\w+)\}/g, (_, key) => {
    const v = lookup(key);
    return v !== undefined && v !== null ? String(v) : '';
  });
  return { recipe: r, defaults, command: cmd.trim() };
}

function buildSparkDockerCmd({ recipe, command }) {
  const env = Object.entries(recipe.env || {})
    .map(([k, v]) => `-e ${k}=${JSON.stringify(String(v))}`)
    .join(' ');
  const containerName = recipe.container_name || 'vllm_node';
  // Respect HF_HOME if set (e.g. custom cache path) — same logic as launch-cluster.sh
  const hfCacheMount = '${HF_HOME:-$HOME/.cache/huggingface}';
  return [
    `set -e`,
    `docker rm -f ${containerName} 2>/dev/null || true`,
    `mkdir -p "${hfCacheMount}" $HOME/.cache/vllm $HOME/.cache/flashinfer $HOME/.triton`,
    `docker run -d --rm --gpus all --privileged --network host --name ${containerName} ${env} \\`,
    `  -v "${hfCacheMount}":/root/.cache/huggingface \\`,
    `  -v $HOME/.cache/vllm:/root/.cache/vllm \\`,
    `  -v $HOME/.cache/flashinfer:/root/.cache/flashinfer \\`,
    `  -v $HOME/.triton:/root/.triton \\`,
    `  ${recipe.container} sleep infinity`,
    `docker exec ${containerName} ${command}`,
  ].join('\n');
}

// ---- Spark recipe exporter -------------------------------------------------
// Rebuild the community spark-arena YAML from a saved recipe so Share emits
// the format people actually trade. Recipes imported from spark YAML keep the
// original text verbatim in args._spark_yaml (same stash pattern as _registry),
// so those round-trip exactly; everything else is synthesized best-effort.
function buildSparkYaml(r) {
  const args = r.args || {};
  if (args._spark_yaml) return String(args._spark_yaml);
  if (args._registry || args._sparkrun) return null; // pipeline recipes — share as JSON/@ref
  if (r.raw_cmd) return sparkYamlFromRawCmd(r);
  return sparkYamlFromArgs(r);
}

function sparkYamlDoc({ name, notes, model, container, container_name, env, defaults, command }) {
  const doc = { recipe_version: '1', name: name || model || 'recipe' };
  if (notes) doc.description = notes;
  if (model) doc.model = model;
  doc.container = container;
  if (container_name && container_name !== 'vllm_node') doc.container_name = container_name;
  if (env && Object.keys(env).length) {
    doc.env = Object.fromEntries(Object.entries(env).map(([k, v]) => [k, String(v)]));
  }
  if (defaults && Object.keys(defaults).length) doc.defaults = defaults;
  doc.command = command.trim() + '\n';
  return jsyaml.dump(doc, { lineWidth: -1 });
}

// Parse a raw_cmd with the buildSparkDockerCmd shape (docker run … <image>
// sleep infinity && docker exec <name> <command>) back into YAML parts.
// Returns null for raw commands that don't match — those share as JSON.
function sparkYamlFromRawCmd(r) {
  const raw = r.raw_cmd || '';
  const img = raw.match(/^\s*(\S+)\s+sleep infinity\b.*$/m);
  const exec = raw.match(/docker exec\s+(\S+)\s+([\s\S]+)$/);
  if (!img || !exec) return null;
  const env = {};
  // env flags only appear in the docker run section, before `<image> sleep infinity`
  const head = raw.slice(0, img.index);
  for (const m of head.matchAll(/(?:^|\s)-e\s+([A-Za-z_][A-Za-z0-9_]*)=(?:"([^"]*)"|'([^']*)'|(\S+))/g)) {
    env[m[1]] = m[2] ?? m[3] ?? m[4];
  }
  return sparkYamlDoc({
    name: r.name, notes: r.notes, model: r.model,
    container: img[1], container_name: exec[1], env,
    command: exec[2],
  });
}

// Synthesize spark YAML for args-based (natively launched) recipes, mirroring
// the flag mapping in runners.py so the command matches what actually ran.
function sparkYamlFromArgs(r) {
  const model = r.model || r.args?.model;
  if (!model) return null;
  const bases = {
    vllm: { container: 'vllm-node', head: 'vllm serve {model} --host {host} --port {port}' },
    sglang: { container: 'lmsysorg/sglang:latest', head: 'python3 -m sglang.launch_server --model-path {model} --host {host} --port {port}' },
    llamacpp: {
      container: 'ghcr.io/ggerganov/llama.cpp:server-cuda',
      head: `llama-server ${(model.match(/\//g) || []).length === 1 ? '-hf' : '-m'} {model} --host {host} --port {port}`,
    },
  };
  const base = bases[r.engine];
  if (!base) return null;
  const flags = Object.entries(r.args || {})
    .filter(([k]) => !k.startsWith('_') && !['model', 'model-path', 'm', 'port', 'host'].includes(k))
    .map(([k, v]) => {
      const flag = k.startsWith('-') ? k : `--${k}`;
      if (v === false || v === null || v === undefined) return null;
      if (v === true) return flag;
      if (Array.isArray(v)) return `${flag} ${v.join(' ')}`;
      return `${flag} ${v}`;
    })
    .filter(Boolean);
  return sparkYamlDoc({
    name: r.name, notes: r.notes, model,
    container: base.container, env: r.env,
    defaults: { port: 8000, host: '0.0.0.0' },
    command: [base.head, ...flags].join(' \\\n  '),
  });
}

// Starter templates for raw-command mode. Edit the MODEL/PORT/IMAGE placeholders.
const RAW_TEMPLATES = {
  vllm: `# DGX Spark — paste a recipe from spark-arena.com/leaderboard or spark-vllm-docker
# Switch to "Spark recipe (YAML)" mode above and paste YAML, or use raw commands here.
mkdir -p "\${HF_HOME:-$HOME/.cache/huggingface}" $HOME/.cache/vllm $HOME/.cache/flashinfer $HOME/.triton && \\
docker rm -f vllm_node 2>/dev/null; \\
docker run -d --rm --gpus all --privileged --network host --name vllm_node \\
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \\
  -v "\${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \\
  -v $HOME/.cache/vllm:/root/.cache/vllm \\
  sparkrun-eugr-vllm-tf5 sleep infinity && \\
docker exec vllm_node vllm serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \\
  --host 0.0.0.0 --port 8000 \\
  --max-model-len 262144 --gpu-memory-utilization 0.80 \\
  --kv-cache-dtype fp8 --load-format fastsafetensors \\
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser gemma4 \\
  --async-scheduling --trust-remote-code --tensor-parallel-size 1
`,
  sglang: `# SGLang via Docker (adjust image + model)
docker run -d --rm --gpus all --network host --name sglang_node \\
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \\
  lmsysorg/sglang:latest \\
  python -m sglang.launch_server --model-path MODEL --host 0.0.0.0 --port 8000
`,
  llamacpp: `# llama.cpp server (native binary or via docker)
llama-server -hf bartowski/Llama-3.1-8B-Instruct-GGUF \\
  --host 0.0.0.0 --port 8000 --n-gpu-layers 999 --ctx-size 8192 --flash-attn
`,
};

function buildEnginePanel(panel) {
  const engine = panel.dataset.engine;
  const ui = panel.querySelector('.engine-ui');
  ui.innerHTML = `
    <div data-slot="missing-banner"></div>
    <div class="card">
      <div class="toolbar" style="margin-bottom:14px">
        <button class="btn" data-mode="spark"><i class="fa-solid fa-bolt"></i> Spark recipe (YAML)</button>
        <button class="btn" data-mode="args">Args mode (JSON)</button>
        <button class="btn" data-mode="raw">Raw command</button>
      </div>
      <div class="engine-launch">
        <div>
          <div data-slot="spark-mode">
            <div class="muted" style="margin-bottom:8px">Paste a recipe YAML from <a href="https://spark-arena.com/leaderboard" target="_blank">spark-arena.com/leaderboard</a>, <a href="https://github.com/eugr/spark-vllm-docker/tree/main/recipes" target="_blank">spark-vllm-docker</a>, or <a href="https://github.com/spark-arena/recipe-registry" target="_blank">recipe-registry</a>. Spark Studio expands <code>{placeholders}</code>, drops empty flags, and wraps it in <code>docker run</code> + <code>docker exec</code> against the recipe's <code>container</code>.</div>
            <div class="dropzone" data-engine="${engine}">
              <i class="fa-solid fa-file-arrow-down"></i> Drop a recipe YAML here, or paste below
            </div>
            <textarea class="recipe-editor spark-editor" data-engine="${engine}" style="min-height:340px">${(SPARK_TEMPLATES[engine] || '').replace(/</g, '&lt;')}</textarea>
            <div class="toolbar" style="margin-top:8px">
              <button class="btn" data-action="preview"><i class="fa-solid fa-eye"></i> Preview command</button>
            </div>
            <pre class="login-log" data-slot="preview" hidden></pre>
          </div>
          <div data-slot="args-mode" hidden>
            <textarea class="recipe-editor" data-engine="${engine}" placeholder='{\n  "model": "…",\n  "max-model-len": 16384\n}'>${JSON.stringify(ENGINE_DEFAULTS[engine], null, 2)}</textarea>
          </div>
          <div data-slot="raw-mode" hidden>
            <div class="muted" style="margin-bottom:8px">Full shell command. Run via <code>bash -lc</code>; ready when stdout shows <code>Application startup complete</code> or <code>Uvicorn running on</code>.</div>
            <textarea class="recipe-editor raw-editor" data-engine="${engine}" style="min-height:260px">${RAW_TEMPLATES[engine] || ''}</textarea>
          </div>
          <div class="toolbar" style="margin-top:10px">
            <button class="btn primary" data-action="run">▶ Run</button>
            <button class="btn" data-action="save">Save as recipe</button>
            <select data-action="loadRecipe"><option value="">— load saved recipe —</option></select>
          </div>
        </div>
        <div>
          <h3 style="margin-top:0">Recent ${engine} runs</h3>
          <div data-slot="recent"></div>
        </div>
      </div>
    </div>
    <div class="card">
      <h3><i class="fa-solid fa-link"></i> Connect existing ${engine} endpoint</h3>
      <div class="muted" style="margin-bottom:10px">Already running ${engine} (e.g. <code>spark-vllm-docker</code> or another host)? Register it so chat, benchmarks, and Ask-Agent target it.</div>
      <div class="form-row">
        <input data-ext="name" type="text" placeholder="Label (e.g. docker-vllm-8B)" style="flex:1" />
        <input data-ext="url" type="text" placeholder="http://127.0.0.1:8000" style="flex:2" />
        <button class="btn primary" data-ext="connect"><i class="fa-solid fa-plug"></i> Connect</button>
      </div>
    </div>
  `;
  ui.querySelector('[data-ext="connect"]').addEventListener('click', async () => {
    const name = ui.querySelector('[data-ext="name"]').value.trim() || `external-${engine}`;
    const url = ui.querySelector('[data-ext="url"]').value.trim();
    if (!url) { toast('Enter a URL', 'danger'); return; }
    try {
      const r = await api('/external', { method: 'POST', body: { engine, name, url } });
      toast(`Connected ${name} (${r.id})`);
      refreshEnginePanel(engine);
      refreshOverview();
    } catch (e) { toast(e.message, 'danger'); }
  });

  const editor = ui.querySelector('.recipe-editor:not(.spark-editor):not(.raw-editor)');
  const dz = ui.querySelector('.dropzone');
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', async (e) => {
    e.preventDefault(); dz.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (!f) return;
    const text = await f.text();
    const mode = ui.dataset.mode || 'spark';
    if (mode === 'spark' || /\.ya?ml$/i.test(f.name)) { sparkEditor.value = text; setMode('spark'); }
    else if (mode === 'raw') { rawEditor.value = text; }
    else { editor.value = tryReformat(text); }
  });

  // Spark / Args / Raw mode toggle
  const sparkMode = ui.querySelector('[data-slot="spark-mode"]');
  const argsMode = ui.querySelector('[data-slot="args-mode"]');
  const rawMode = ui.querySelector('[data-slot="raw-mode"]');
  const sparkEditor = ui.querySelector('.spark-editor');
  const rawEditor = ui.querySelector('.raw-editor');
  const previewBox = ui.querySelector('[data-slot="preview"]');
  enhanceEditor(sparkEditor, 'yaml');
  enhanceEditor(editor, 'json');
  enhanceEditor(rawEditor, 'shell');
  const setMode = (mode) => {
    ui.dataset.mode = mode;
    sparkMode.hidden = mode !== 'spark';
    argsMode.hidden = mode !== 'args';
    rawMode.hidden = mode !== 'raw';
    ui.querySelectorAll('[data-mode]').forEach((b) => {
      const active = b.dataset.mode === mode;
      b.classList.toggle('primary', active);
      b.style.background = active ? 'linear-gradient(135deg,var(--accent),var(--accent-2))' : '';
      b.style.color = active ? '#0b0f17' : '';
    });
  };
  ui.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  setMode('spark');

  function getRawCmd(mode) {
    if (mode === 'raw') return rawEditor.value.trim();
    if (mode === 'spark') {
      const expanded = expandSparkRecipe(sparkEditor.value);
      return buildSparkDockerCmd(expanded);
    }
    return null;
  }

  ui.querySelector('[data-action="preview"]').addEventListener('click', () => {
    try {
      previewBox.hidden = false;
      previewBox.textContent = getRawCmd('spark');
    } catch (e) {
      previewBox.hidden = false;
      previewBox.textContent = `Parse error: ${e.message}`;
    }
  });

  ui.querySelector('[data-action="run"]').addEventListener('click', async () => {
    try {
      const mode = ui.dataset.mode || 'spark';
      let body;
      if (mode === 'args') {
        body = { engine, args: JSON.parse(editor.value) };
      } else {
        const raw = getRawCmd(mode);
        if (!raw) throw new Error('empty command');
        body = { engine, raw_cmd: raw, args: {} };
      }
      const validation = validateRecipePayload(body);
      if (!validation.ok) throw new Error(validation.error);
      const run = await api('/runs', { method: 'POST', body });
      toast(`Started ${engine} run ${run.id}${run.port ? ' on :' + run.port : ''}`);
      updateTitle();
      $('.tab[data-tab="logs"]').click();
      setTimeout(() => window.selectRun(run.id), 50);
      setTimeout(async () => {
        try {
          const r = await api(`/runs/${run.id}`);
          if (runOutcome(r) === 'failed') toast(`${engine} failed (code ${r.exit_code}). Click Ask Claude on Logs tab.`, 'danger');
          else if (r.status === 'exited') toast(`${engine} exited.`);
        } catch {}
      }, 4000);
    } catch (e) { toast(`Run failed: ${e.message}`, 'danger'); }
  });
  ui.querySelector('[data-action="save"]').addEventListener('click', () => {
    const mode = ui.dataset.mode || 'spark';
    try {
      if (mode === 'spark') {
        const { recipe, command } = expandSparkRecipe(sparkEditor.value);
        const raw_cmd = buildSparkDockerCmd({ recipe, command });
        const validation = validateRecipePayload({ raw_cmd });
        if (!validation.ok) throw new Error(validation.error);
        openRecipeModal({
          engine,
          model: recipe.model,
          raw_cmd,
          args: { _spark_yaml: sparkEditor.value },
          notes: recipe.metadata?.description || '',
          tags: [recipe.metadata?.quantization, recipe.metadata?.kv_dtype, 'spark-recipe'].filter(Boolean).join(','),
          name: `${engine} · ${recipe.model || 'spark-recipe'}`,
          env: recipe.env || {},
        });
      } else if (mode === 'raw') {
        const validation = validateRecipePayload({ raw_cmd: rawEditor.value });
        if (!validation.ok) throw new Error(validation.error);
        openRecipeModal({ engine, raw_cmd: rawEditor.value, name: `${engine} · docker recipe` });
      } else {
        let args; try { args = JSON.parse(editor.value); } catch (e) { toast('Invalid JSON', 'danger'); return; }
        openRecipeModal({ engine, model: args.model, args, name: `${engine} · ${args.model || 'untitled'}` });
      }
    } catch (e) { toast(e.message, 'danger'); }
  });
  ui.querySelector('[data-action="loadRecipe"]').addEventListener('change', async (e) => {
    if (!e.target.value) return;
    const r = await api(`/recipes/${e.target.value}`);
    editor.value = JSON.stringify({ model: r.model, ...r.args }, null, 2);
  });

  refreshEnginePanel(engine);
}

async function refreshEnginePanel(engine) {
  const ui = $(`.panel[data-panel="${engine}"] .engine-ui`);
  if (!ui) return;
  const [recipes, sys] = await Promise.all([
    api('/recipes').catch(() => []),
    api('/system').catch(() => ({ engines: {} })),
  ]);
  const banner = ui.querySelector('[data-slot="missing-banner"]');
  if (!sys.engines[engine]) {
    // A pip install can take many minutes (torch is GBs) — its state must
    // survive panel rebuilds (tab switches re-render this banner), otherwise
    // a running install silently "resets" to the Install button.
    const st = _installs[engine];
    const running = !!st?.active;
    const failed = !!st && !st.active && st.code !== 0 && st.code !== null;
    banner.innerHTML = `<div class="card" style="border-color:var(--warn);background:rgba(255,183,74,0.08)">
      <h3 style="color:var(--warn)"><i class="fa-solid fa-triangle-exclamation"></i> ${engine} is not installed</h3>
      <div class="muted" style="margin-bottom:10px">Spark Studio can't find <code>${engine === 'llamacpp' ? 'llama-server' : engine}</code> in this launcher's environment.</div>
      ${engine === 'llamacpp'
        ? `<div class="muted">Install via conda or build from source:</div>
           <pre class="mono" style="background:var(--bg);padding:10px;border-radius:6px">conda install -c conda-forge llama.cpp</pre>`
        : `${running
              ? `<div><span class="badge">⏳ installing… this can take several minutes (large downloads)</span></div>`
              : `<button class="btn primary" data-install="${engine}"><i class="fa-solid fa-download"></i> ${failed ? `Retry install (last attempt exited ${st.code})` : `Install ${engine}`}</button>`}
           <pre class="login-log" data-slot="install-log" ${st?.lines.length ? '' : 'hidden'}></pre>`
      }
    </div>`;
    const log = banner.querySelector('[data-slot="install-log"]');
    if (log && st?.lines.length) { log.textContent = st.lines.join('\n'); log.scrollTop = log.scrollHeight; }
    const btn = banner.querySelector('[data-install]');
    if (btn) btn.addEventListener('click', () => streamInstall(engine, banner));
  } else {
    banner.innerHTML = '';
  }
  const sel = ui.querySelector('[data-action="loadRecipe"]');
  sel.innerHTML = '<option value="">— load saved recipe —</option>' +
    recipes.filter((r) => r.engine === engine).map((r) => `<option value="${r.id}">${r.name}</option>`).join('');
  const runs = (await api('/runs').catch(() => [])).filter((r) => r.engine === engine).slice(0, 5);
  ui.querySelector('[data-slot="recent"]').innerHTML = runs.map(renderRunRow).join('') || '<div class="muted">None yet.</div>';
}

// Engine installs in flight — keyed by engine so state survives re-renders.
const _installs = {};
function streamInstall(engine, bannerEl) {
  if (_installs[engine]?.active) return; // one install at a time per engine
  const st = (_installs[engine] = { active: true, code: null, lines: [] });
  refreshEnginePanel(engine); // swap the button for the "installing…" badge
  const liveLog = () => document.querySelector(`.panel[data-panel="${engine}"] [data-slot="install-log"]`);
  const es = new EventSource(`/api/engines/install/${engine}`);
  es.addEventListener('log', (ev) => {
    st.lines.push(ev.data);
    if (st.lines.length > 500) st.lines.shift();
    const log = liveLog(); // re-query: the panel may have been rebuilt since
    if (log) { log.hidden = false; log.textContent = st.lines.join('\n'); log.scrollTop = log.scrollHeight; }
  });
  es.addEventListener('done', (ev) => {
    es.close();
    st.active = false;
    st.code = Number(ev.data);
    if (st.code === 0) {
      toast(`${engine} installed`);
      delete _installs[engine];
    } else {
      st.lines.push(`[install failed — exit ${st.code}]`);
      toast(`${engine} install failed (exit ${st.code}) — the log stays on the ${engine} tab`, 'danger');
    }
    refreshEnginePanel(engine); // success clears the banner; failure keeps log + Retry
    refreshSystem();
  });
  es.addEventListener('error', () => {
    es.close();
    st.active = false;
    st.code = st.code ?? -1;
    st.lines.push('[install stream disconnected — the pip process may still be running; re-check in a minute]');
    toast('Install stream disconnected', 'danger');
    refreshEnginePanel(engine);
  });
}

$$('.panel[data-engine]').forEach(buildEnginePanel);

function tryReformat(text) {
  try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return text; }
}

function validateRawCommand(raw) {
  const cmd = (raw || '').trim();
  if (!cmd) return { ok: false, error: 'Command is empty.' };

  const hf = cmd.match(/--hf-overrides\s+(['"])(.*?)\1/s);
  if (!hf) return { ok: true };

  const payload = hf[2];
  if (payload.includes('{{') || payload.includes('}}')) {
    return {
      ok: false,
      error: 'Invalid --hf-overrides JSON: found doubled braces like `{{` / `}}`. Use normal JSON braces only.',
    };
  }
  try {
    JSON.parse(payload);
  } catch (e) {
    return {
      ok: false,
      error: `Invalid --hf-overrides JSON: ${e.message}`,
    };
  }
  return { ok: true };
}

function validateRecipePayload(recipe) {
  if (recipe?.args?._registry) return { ok: true };
  if (recipe.raw_cmd) return validateRawCommand(recipe.raw_cmd);
  return { ok: true };
}

function extractQuotedValue(text) {
  if (!text) return null;
  const s = String(text).trim();
  if (!s) return null;
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
    return s.slice(1, -1);
  }
  return s;
}

function extractShellDefault(raw, key) {
  const text = String(raw || '');
  const patterns = [
    new RegExp(`${key}="\\$\\{${key}:-([^"}]+)\\}"`),
    new RegExp(`${key}='\\$\\{${key}:-([^'}]+)\\}'`),
    new RegExp(`${key}=\\$\\{${key}:-([^}\\n]+)\\}`),
  ];
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match?.[1]) return extractQuotedValue(match[1]);
  }
  return null;
}

function firstUseful(values) {
  for (const value of values) {
    const s = extractQuotedValue(value);
    if (!s) continue;
    if (s.includes('${')) continue;
    if (/^<.*>$/.test(s)) continue;
    return s;
  }
  return null;
}

function inferRecipeModel(recipe = {}) {
  const args = recipe.args || {};
  const env = recipe.env || {};
  const raw = recipe.raw_cmd || '';

  const direct = firstUseful([
    recipe.model,
    args.model,
    args['model-path'],
    args.m,
    env.MODEL,
    env.HF_MODEL_ID,
    env.RECIPE_MODEL,
  ]);
  if (direct) return direct;

  const defaults = firstUseful([
    extractShellDefault(raw, 'MODEL'),
    extractShellDefault(raw, 'HF_MODEL_ID'),
    extractShellDefault(raw, 'RECIPE_MODEL'),
  ]);
  if (defaults) return defaults;

  const patterns = [
    /\bvllm\s+serve\s+("[^"]+"|'[^']+'|\S+)/i,
    /--model-path\s+("[^"]+"|'[^']+'|\S+)/i,
    /\s-hf\s+("[^"]+"|'[^']+'|\S+)/i,
    /\s-m\s+("[^"]+"|'[^']+'|\S+)/i,
  ];
  for (const pattern of patterns) {
    const match = raw.match(pattern);
    const candidate = firstUseful([match?.[1]]);
    if (candidate && !candidate.startsWith('$')) return candidate;
  }

  return null;
}

function inferRecipeDetails(recipe = {}) {
  const args = recipe.args || {};
  const env = recipe.env || {};
  const raw = recipe.raw_cmd || '';
  const details = [];

  const push = (label, value) => {
    const v = firstUseful([value]);
    if (v) details.push(`${label} ${v}`);
  };

  push('image', env.VLLM_IMAGE || env.IMAGE || extractShellDefault(raw, 'VLLM_IMAGE') || extractShellDefault(raw, 'IMAGE'));
  push('len', args['max-model-len'] || env.MAX_MODEL_LEN || extractShellDefault(raw, 'MAX_MODEL_LEN'));
  push('mem', args['gpu-memory-utilization'] || env.GPU_MEMORY_UTILIZATION || env.GPU_MEM_UTIL || extractShellDefault(raw, 'GPU_MEMORY_UTILIZATION'));
  push('dtype', args.dtype);
  push('kv', args['kv-cache-dtype'] || env.KV_CACHE_DTYPE || extractShellDefault(raw, 'KV_CACHE_DTYPE'));
  push('tp', args['tensor-parallel-size'] || args.tp || args.tensor_parallel || args.tensor_parallel_size);

  return details.slice(0, 4);
}

function enrichRecipe(recipe = {}) {
  const model = recipe.model || inferRecipeModel(recipe);
  const details = inferRecipeDetails({ ...recipe, model });
  return { ...recipe, model, _details: details };
}

// ---------- recipes -------------------------------------------------------
async function refreshRecipes() {
  const [recipes, runs] = await Promise.all([
    api('/recipes').catch(() => []),
    api('/runs').catch(() => []),
  ]);
  const activeByRecipe = new Map(
    runs
      .filter((run) => run.status === 'running' && run.recipe_id)
      .map((run) => [Number(run.recipe_id), run])
  );
  const search = $('#recSearch').value.toLowerCase();
  const enriched = recipes.map((r) => enrichRecipe(r));
  const filtered = enriched.filter((r) => !search || (r.name + ' ' + (r.model || '') + ' ' + r.tags + ' ' + (r._details || []).join(' ')).toLowerCase().includes(search));
  const html = filtered.map((r) => {
    const isDocker = !!(r.args && r.args._registry);
    const tagList = (r.tags || '').split(',').map((t) => t.trim()).filter(Boolean);
    const isWorking = tagList.includes('working');
    const isFailed  = tagList.includes('fix') && !isWorking;
    const displayTags = tagList.filter((t) => t !== 'working' && t !== 'fix');
    const statusBadge = isWorking
      ? '<span class="rc-status-badge working" title="Last run succeeded">✓ working</span>'
      : isFailed
        ? '<span class="rc-status-badge failed" title="Last run failed">✗ failed</span>'
        : '';
    return `
    <div class="recipe-card${activeByRecipe.has(Number(r.id)) ? ' is-running' : ''}">
      <div class="rc-head">
        <div class="rc-name">${escapeHtml(r.name)}</div>
        <div style="display:flex;gap:4px;align-items:center">
          ${statusBadge}
          ${isDocker ? '<span class="rc-source registry" title="Runs via spark-vllm-docker pipeline">Docker</span>' : ''}
          <div class="rc-engine">${r.engine}</div>
        </div>
      </div>
      ${activeByRecipe.has(Number(r.id)) ? `<div class="rc-running"><span class="badge ok">Running</span><span class="mono">${escapeHtml(activeByRecipe.get(Number(r.id)).id)}</span><button class="btn danger" data-stop-run="${activeByRecipe.get(Number(r.id)).id}" title="Stop run">■ Stop</button></div>` : ''}
      <div class="rc-model">${escapeHtml(r.model || '—')}</div>
      ${r._details?.length ? `<div class="rc-notes">${escapeHtml(r._details.join(' · '))}</div>` : ''}
      ${r.notes ? `<div class="rc-notes">${escapeHtml(r.notes)}</div>` : ''}
      ${displayTags.length ? `<div class="rc-tags">${displayTags.map((t) => `<span>${escapeHtml(t)}</span>`).join('')}</div>` : ''}
      <div class="rc-actions">
        <button class="btn primary" data-run="${r.id}">▶ Run</button>
        <button class="btn" data-edit="${r.id}">Edit</button>
        <button class="btn" data-share="${r.id}" title="Copy recipe in the community spark-arena YAML format (falls back to JSON for pipeline recipes)">⧉ Share</button>
        <button class="btn danger" data-del="${r.id}">Del</button>
      </div>
    </div>
  `}).join('');
  $('#recipesList').innerHTML = html || '<div class="muted">No recipes yet. Create one or Forge from an HF model.</div>';
  $$('#recipesList [data-run]').forEach((b) => b.addEventListener('click', async () => {
    try {
      const r = await api(`/recipes/${b.dataset.run}`);
      const hasRegistry = !!(r.args && r.args._registry);
      if (hasRegistry) {
        openLaunchModal(r);
      } else {
        const body = r.raw_cmd
          ? { engine: r.engine, raw_cmd: r.raw_cmd, env: r.env || {}, recipe_id: r.id }
          : { engine: r.engine, args: { model: r.model, ...r.args }, env: r.env || {}, recipe_id: r.id };
        const run = await api('/runs', { method: 'POST', body });
        toast(`Started run ${run.id}`);
        $('.tab[data-tab="logs"]').click();
        setTimeout(() => window.selectRun(run.id), 50);
      }
    } catch (e) { toast(e.message, 'danger'); }
  }));
  $$('#recipesList [data-edit]').forEach((b) => b.addEventListener('click', async () => {
    openRecipeModal(await api(`/recipes/${b.dataset.edit}`));
  }));
  $$('#recipesList [data-share]').forEach((b) => b.addEventListener('click', async () => {
    try {
      const r = await api(`/recipes/${b.dataset.share}`);
      // Community sparkrun recipes: the @ref itself is the share format.
      if (r.args?._sparkrun?.ref) {
        await copyText(r.args._sparkrun.ref);
        toast(`Copied ${r.args._sparkrun.ref} — paste it into any Spark Studio or run it with sparkrun`);
        return;
      }
      // Prefer the community spark-arena YAML format when the recipe maps to it.
      const yaml = buildSparkYaml(r);
      if (yaml) {
        await copyText(yaml);
        toast('Recipe copied as spark-arena YAML — share it anywhere; Paste on the Recipes tab imports it');
        return;
      }
      // Strip local-only state: the DB id and the working/fix status tags.
      const shareable = {
        name: r.name,
        engine: r.engine,
        model: r.model || null,
        args: r.args || {},
        env: r.env || {},
        notes: r.notes || '',
        tags: (r.tags || '').split(',').map((t) => t.trim())
          .filter((t) => t && t !== 'working' && t !== 'fix').join(','),
        raw_cmd: r.raw_cmd || null,
      };
      await copyText(JSON.stringify(shareable, null, 2));
      toast('Recipe copied as JSON — others can add it with Import or Paste on their Recipes tab');
    } catch (e) { toast(e.message, 'danger'); }
  }));
  $$('#recipesList [data-del]').forEach((b) => b.addEventListener('click', async () => {
    if (!confirm('Delete recipe?')) return;
    await api(`/recipes/${b.dataset.del}`, { method: 'DELETE' });
    refreshRecipes();
  }));
  $$('#recipesList [data-stop-run]').forEach((b) => b.addEventListener('click', async (ev) => {
    ev.stopPropagation();
    try {
      await api(`/runs/${b.dataset.stopRun}/stop?force=true`, { method: 'POST' });
      toast(`Stopping run ${b.dataset.stopRun}…`);
      setTimeout(refreshRecipes, 2500);
    } catch (e) { toast(e.message, 'danger'); }
  }));
}
// ---- Community recipes via sparkrun ---------------------------------------
// Community-recipe cache: avoids a ~1s `sparkrun list` subprocess on every
// tab visit, but with a TTL — a session-long cache meant an app update (or
// registry sync) never showed new recipes until a full page reload.
let _sparkrunCache = null;
let _sparkrunCacheAt = 0;
const SPARKRUN_CACHE_TTL_MS = 120000;
async function refreshSparkrun() {
  const list = $('#sparkrunList');
  const state = $('#sparkrunState');
  try {
    const [status, recipes] = await Promise.all([
      api('/sparkrun/status'),
      (_sparkrunCache && Date.now() - _sparkrunCacheAt < SPARKRUN_CACHE_TTL_MS)
        ? Promise.resolve(_sparkrunCache) : api('/sparkrun/recipes'),
    ]);
    _sparkrunCache = recipes;
    _sparkrunCacheAt = Date.now();
    const tp = Math.max(1, Number($('#sparkrunTp').value) || 1);
    const q = ($('#sparkrunSearch').value || '').toLowerCase();
    // Only recipes that can actually run on the selected node count.
    const runnable = recipes.filter((r) => (r.min_nodes || 1) <= tp && (!r.max_nodes || tp <= r.max_nodes));
    const filtered = runnable.filter((r) => !q || `${r.ref} ${r.model || ''} ${r.engine} ${r.description || ''}`.toLowerCase().includes(q));
    state.textContent = status.installed
      ? `sparkrun ${status.version || '?'} · ${runnable.length} of ${recipes.length} recipes fit ${tp} node${tp > 1 ? 's' : ''}`
      : `sparkrun not installed — ${status.hint}`;
    $('#sparkrunUpdate').disabled = !status.installed || status.update?.running;
    if (status.update?.running && !_sparkrunUpdatePoll) watchSparkrunUpdate();
    list.innerHTML = filtered.map((r) => `
      <div class="recipe-card">
        <div class="rc-head">
          <div class="rc-name mono">${escapeHtml(r.ref)}</div>
          <div style="display:flex;gap:4px;align-items:center">
            ${(r.min_nodes || 1) > 1 ? `<span class="badge" title="Requires ${r.min_nodes} DGX Sparks (tensor parallelism)">${r.min_nodes}× Spark</span>` : ''}
            <span class="rc-source registry" title="Runs on your Spark mesh via sparkrun">${escapeHtml(r.namespace)}</span>
            <div class="rc-engine">${escapeHtml(r.engine || '')}</div>
          </div>
        </div>
        <div class="rc-model">${escapeHtml(r.model || '—')}</div>
        ${r.description ? `<div class="rc-notes">${escapeHtml(r.description)}</div>` : ''}
        <div class="rc-actions">
          <button class="btn primary" data-sparkrun="${escapeHtml(r.ref)}" ${status.installed ? '' : 'disabled title="Install sparkrun first"'}>▶ Run via sparkrun</button>
        </div>
      </div>`).join('') || `<div class="muted">No recipes fit ${tp} node${tp > 1 ? 's' : ''}${q ? ' matching your search' : ''}. Raise Nodes (TP) to see multi-Spark recipes.</div>`;
    $$('#sparkrunList [data-sparkrun]').forEach((btn) => btn.addEventListener('click', async () => {
      try {
        const tp = Number($('#sparkrunTp').value) || 1;
        const run = await api('/sparkrun/run', { method: 'POST', body: { ref: btn.dataset.sparkrun, tp } });
        toast(`Started ${run.id} — ${btn.dataset.sparkrun}${tp > 1 ? ` on ${tp} nodes` : ''}${run.recipe_id ? ' · saved to My Recipes' : ''}`);
        refreshRecipes();
        $('.tab[data-tab="logs"]').click();
        setTimeout(() => window.selectRun(run.id), 50);
      } catch (e) { toast(e.message, 'danger'); }
    }));
  } catch (e) {
    state.textContent = `sparkrun status unavailable: ${e.message}`;
  }
}
$('#sparkrunSearch').addEventListener('input', refreshSparkrun);
$('#sparkrunTp').addEventListener('input', refreshSparkrun);

// ---- sparkrun self-update ---------------------------------------------------
// POST /sparkrun/update kicks the update off in the background; we poll its
// status until it finishes, then re-fetch status so the version line updates.
let _sparkrunUpdatePoll = null;
function watchSparkrunUpdate() {
  const btn = $('#sparkrunUpdate');
  const state = $('#sparkrunState');
  btn.disabled = true;
  _sparkrunUpdatePoll = setInterval(async () => {
    try {
      const st = await api('/sparkrun/update/status');
      if (st.running) {
        state.textContent = `updating sparkrun${st.channel ? ` (--${st.channel})` : ''}…`;
        return;
      }
      clearInterval(_sparkrunUpdatePoll);
      _sparkrunUpdatePoll = null;
      btn.disabled = false;
      if (st.ok) {
        const moved = st.version_after && st.version_after !== st.version_before;
        toast(moved
          ? `sparkrun updated: ${st.version_before || '?'} → ${st.version_after}`
          : `sparkrun is up to date (${st.version_after || st.version_before || '?'})`);
      } else if (st.ok === false) {
        toast(`sparkrun update failed — ${(st.log || []).slice(-3).join(' · ') || 'see server logs'}`, 'danger');
      }
      // The Community list is served from the app's own registry mirror, not
      // sparkrun's registries — re-sync it too so new recipes actually appear.
      try { await api('/registry/sync', { method: 'POST' }); } catch { /* non-fatal */ }
      _sparkrunCache = null;
      refreshSparkrun();
    } catch { /* transient poll error — keep polling */ }
  }, 2000);
}
$('#sparkrunUpdate').addEventListener('click', async () => {
  const channel = $('#sparkrunChannel').value || null;
  if ((channel === 'alpha' || channel === 'yolo' || channel === 'beta')
      && !confirm(`Switch sparkrun to the ${channel === 'beta' ? 'beta (develop)' : 'alpha (bleeding edge)'} channel?\n\nPreview builds install from git and the channel is remembered for future updates. Going back to Stable later may downgrade.`)) return;
  try {
    await api('/sparkrun/update', { method: 'POST', body: { channel } });
    toast(`sparkrun update started${channel ? ` on --${channel}` : ''}…`);
    watchSparkrunUpdate();
  } catch (e) { toast(e.message, 'danger'); }
});

$('#recSearch').addEventListener('input', refreshRecipes);
$('#recPaste').addEventListener('click', () => {
  const text = prompt('Paste a shared recipe (JSON/YAML), a spark-arena link, an HF id, or a @ref:');
  if (!text) return;
  try {
    openRecipeModal(JSON.parse(text));
  } catch {
    runAnythingDispatch(text);
  }
});

// ---------- run anything ---------------------------------------------------
// One box, zero ceremony: spark-arena links, recipe YAML/JSON, HF ids, @refs.
async function runSparkYamlText(yamlText, meta = {}) {
  const { recipe, command } = expandSparkRecipe(yamlText);
  const raw_cmd = buildSparkDockerCmd({ recipe, command });
  const saved = await api('/recipes', { method: 'POST', body: {
    name: meta.name || recipe.name || recipe.model || 'imported recipe',
    engine: meta.engine || 'vllm',
    model: meta.model || recipe.model || null,
    raw_cmd,
    // Keep the pristine spark YAML so Share can re-emit the community format.
    args: { _spark_yaml: yamlText },
    env: {},
    notes: meta.notes || '',
    tags: meta.tags || 'imported',
  }});
  const run = await api('/runs', { method: 'POST', body: { engine: saved.engine, raw_cmd, env: {}, recipe_id: saved.id } });
  toast(`Started ${saved.model || saved.name} · saved to My Recipes`);
  refreshRecipes();
  $('.tab[data-tab="logs"]').click();
  setTimeout(() => window.selectRun(run.id), 50);
  return run;
}

async function runAnythingDispatch(text) {
  const status = $('#runAnythingStatus');
  const setStatus = (t) => { if (status) status.textContent = t; };
  text = (text || '').trim();
  if (!text) return;
  try {
    if (/spark-arena\.com\/benchmark\//i.test(text)) {
      setStatus('Fetching the recipe from Spark Arena…');
      const imp = await api('/arena/import', { method: 'POST', body: { text } });
      setStatus('');
      await runSparkYamlText(imp.yaml, {
        name: imp.name || imp.model,
        model: imp.model,
        engine: (imp.runtime || 'vllm').includes('sglang') ? 'sglang' : 'vllm',
        notes: `Imported from ${imp.url}${imp.description ? ' — ' + imp.description : ''}`,
        tags: 'arena',
      });
      return;
    }
    if (/^@[\w.-]+\/[\w.-]+$/.test(text)) {
      const tp = Number($('#sparkrunTp')?.value) || 1;
      const run = await api('/sparkrun/run', { method: 'POST', body: { ref: text, tp } });
      toast(`Started ${text} · saved to My Recipes`);
      refreshRecipes();
      $('.tab[data-tab="logs"]').click();
      setTimeout(() => window.selectRun(run.id), 50);
      return;
    }
    if (text.startsWith('{')) { openRecipeModal(JSON.parse(text)); return; }
    if (/(^|\n)\s*(model|container|command|recipe_version)\s*:/.test(text)) {
      await runSparkYamlText(text, { tags: 'imported' });
      return;
    }
    if (/^[\w][\w.-]*\/[\w][\w.-]*$/.test(text)) {
      $('#forgeRepo').value = text;
      $('.tab[data-tab="forge"]').click();
      $('#forgeGenerate').click();
      return;
    }
    setStatus('Not recognized — paste a spark-arena benchmark link, recipe YAML/JSON, a HuggingFace id, or a @community/ref.');
  } catch (e) {
    setStatus('');
    toast(e.message, 'danger');
  }
}
$('#runAnythingGo')?.addEventListener('click', () => runAnythingDispatch($('#runAnything').value));
$('#runAnything')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runAnythingDispatch($('#runAnything').value);
});
$('#recImport').addEventListener('click', () => $('#recImportFile').click());
$('#recImportFile').addEventListener('change', async (e) => {
  const file = e.target.files[0]; if (!file) return;
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    openRecipeModal(parsed);
  } catch (err) { toast(`Import failed: ${err.message}`, 'danger'); }
});

// Recipe modal -----------------------------------------------------------
// Detected tool/reasoning capabilities for the model currently in the editor.
let _rmCaps = null;
// Reflect a recipe's tool/reasoning parsers as toggles, enabled only when the
// model's family has a known parser (a wrong parser breaks serving), and show
// the context the save will apply. Flat vLLM recipes only — docker/raw recipes
// carry their own command.
async function refreshRecipeCaps() {
  const box = $('#rmCaps');
  const engine = $('#rmEngine').value;
  const model = $('#rmModel').value.trim();
  const raw = $('#rmRawCmd').value.trim();
  if (engine !== 'vllm' || !model || raw) { box.hidden = true; _rmCaps = null; return; }
  let args = {};
  try { args = JSON.parse($('#rmArgs').value || '{}'); } catch {}
  box.hidden = false;
  // Reflect current args immediately; strict presence, no auto-check on open.
  $('#rmCapTool').checked = !!args['tool-call-parser'];
  $('#rmCapReason').checked = !!args['reasoning-parser'];
  $('#rmCapTool').disabled = $('#rmCapReason').disabled = true;
  $('#rmCapToolParser').textContent = args['tool-call-parser'] || '';
  $('#rmCapReasonParser').textContent = args['reasoning-parser'] || '';
  $('#rmCapHint').textContent = 'detecting model…';
  const token = (refreshRecipeCaps._t = (refreshRecipeCaps._t || 0) + 1);
  try {
    const caps = await api(`/recipes/capabilities?model=${encodeURIComponent(model)}`);
    if (token !== refreshRecipeCaps._t) return; // a newer request superseded us
    _rmCaps = caps;
    $('#rmCapTool').disabled = !caps.supports_tools;
    $('#rmCapReason').disabled = !caps.supports_reasoning;
    $('#rmCapToolParser').textContent = caps.tool_call_parser || '';
    $('#rmCapReasonParser').textContent = caps.reasoning_parser || '';
    const fam = caps.family && caps.family !== 'generic' ? caps.family : 'unrecognized family';
    const ctx = caps.suggested_max_model_len
      ? ` · will serve max_model_len ${caps.suggested_max_model_len.toLocaleString()}` : '';
    const none = (!caps.supports_tools && !caps.supports_reasoning) ? ' — no known parser for this family' : '';
    $('#rmCapHint').textContent = `${fam}${none}${ctx}`;
  } catch {
    if (token !== refreshRecipeCaps._t) return;
    _rmCaps = null;
    $('#rmCapHint').textContent = 'could not detect capabilities';
  }
}

function openRecipeModal(r) {
  const recipe = enrichRecipe(r);
  const m = $('#recipeModal');
  $('#rmName').value = recipe.name || '';
  $('#rmEngine').value = recipe.engine || 'vllm';
  $('#rmModel').value = recipe.model || (recipe.args?.model ?? '');
  $('#rmTags').value = recipe.tags || '';
  $('#rmNotes').value = recipe.notes || '';
  $('#rmRawCmd').value = recipe.raw_cmd || '';
  const args = { ...(recipe.args || {}) };
  delete args.model;
  $('#rmArgs').value = JSON.stringify(args, null, 2);
  $('#rmEnv').value = JSON.stringify(recipe.env || {}, null, 2);
  m.dataset.id = recipe.id || '';
  m.dataset.rawCmd0 = recipe.raw_cmd || '';
  m.hidden = false;
  refreshRecipeCaps();
}
$$('#recipeModal [data-close]').forEach((b) => b.addEventListener('click', () => ($('#recipeModal').hidden = true)));
// Re-detect capabilities when the model, engine, or raw-command fields change.
['rmModel', 'rmEngine', 'rmRawCmd'].forEach((id) => {
  const el = $(`#${id}`);
  if (el) el.addEventListener('change', refreshRecipeCaps);
});
$('#rmSave').addEventListener('click', () => saveRecipeFromModal(false));
$('#rmSaveRun').addEventListener('click', () => saveRecipeFromModal(true));
async function saveRecipeFromModal(thenRun) {
  try {
    const id = $('#recipeModal').dataset.id ? Number($('#recipeModal').dataset.id) : null;
    const recipe = enrichRecipe({
      id,
      name: $('#rmName').value.trim() || 'untitled',
      engine: $('#rmEngine').value,
      model: $('#rmModel').value.trim() || null,
      args: JSON.parse($('#rmArgs').value || '{}'),
      env: JSON.parse($('#rmEnv').value || '{}'),
      notes: $('#rmNotes').value,
      tags: $('#rmTags').value,
      raw_cmd: $('#rmRawCmd').value.trim() || null,
    });
    // The stored spark YAML only describes the raw_cmd it was expanded from —
    // if the command was edited here, drop it so Share doesn't emit stale YAML.
    if (recipe.args?._spark_yaml && (recipe.raw_cmd || '') !== $('#recipeModal').dataset.rawCmd0) {
      delete recipe.args._spark_yaml;
    }
    // Apply the capability toggles (flat vLLM only). The editor is authoritative
    // here — the backend won't re-add a parser we turned off.
    if (recipe.engine === 'vllm' && !recipe.raw_cmd && _rmCaps && !$('#rmCaps').hidden) {
      const a = recipe.args || (recipe.args = {});
      if ($('#rmCapTool').checked && _rmCaps.tool_call_parser) {
        a['enable-auto-tool-choice'] = true;
        a['tool-call-parser'] = _rmCaps.tool_call_parser;
      } else {
        delete a['enable-auto-tool-choice'];
        delete a['tool-call-parser'];
      }
      if ($('#rmCapReason').checked && _rmCaps.reasoning_parser) {
        a['reasoning-parser'] = _rmCaps.reasoning_parser;
      } else {
        delete a['reasoning-parser'];
      }
    }
    const validation = validateRecipePayload(recipe);
    if (!validation.ok) throw new Error(validation.error);
    const saved = await api('/recipes', { method: 'POST', body: recipe });
    $('#recipeModal').hidden = true;
    refreshRecipes();
    if (thenRun) {
      const hasRegistry = !!(saved.args && saved.args._registry);
      const runBody = (!hasRegistry && saved.raw_cmd)
        ? { engine: saved.engine, raw_cmd: saved.raw_cmd, env: saved.env, recipe_id: saved.id }
        : { engine: saved.engine, args: { model: saved.model, ...saved.args }, env: saved.env, recipe_id: saved.id };
      const run = await api('/runs', { method: 'POST', body: runBody });
      toast(`Started ${run.id}`);
      $('.tab[data-tab="logs"]').click();
      setTimeout(() => window.selectRun(run.id), 50);
    } else {
      toast('Recipe saved');
    }
  } catch (e) { toast(e.message, 'danger'); }
}

// ---------- forge ---------------------------------------------------------
function forgeRecent() {
  try { return JSON.parse(localStorage.getItem('forgeRecent') || '[]'); } catch { return []; }
}
function pushForgeRecent(repo) {
  try {
    const list = [repo, ...forgeRecent().filter((r) => r !== repo)].slice(0, 6);
    localStorage.setItem('forgeRecent', JSON.stringify(list));
  } catch {}
}
// Fill the Forge tab's empty space with something actionable: your recent
// forges plus Spark-validated models from the synced registry, one click away.
async function refreshForgeSuggest() {
  const box = $('#forgeSuggest');
  if (!box) return;
  const recent = forgeRecent();
  let popular = [];
  try {
    const recipes = await api('/registry/recipes');
    popular = [...new Set(recipes.map((r) => r.model).filter(Boolean))]
      .filter((m) => !recent.includes(m)).slice(0, 10);
  } catch {}
  const chip = (m) => `<button class="forge-chip" data-chip="${escapeHtml(m)}">${escapeHtml(m)}</button>`;
  box.innerHTML =
    (recent.length ? `<div class="forge-suggest-row"><span class="forge-suggest-label">Recent</span>${recent.map(chip).join('')}</div>` : '') +
    (popular.length ? `<div class="forge-suggest-row"><span class="forge-suggest-label">Spark-validated</span>${popular.map(chip).join('')}</div>` : '');
  $$('#forgeSuggest [data-chip]').forEach((b) => b.addEventListener('click', () => {
    $('#forgeRepo').value = b.dataset.chip;
    $('#forgeGenerate').click();
  }));
}

$('#forgeCheck').addEventListener('click', async () => {
  const repo = $('#forgeRepo').value.trim();
  if (!repo) return;
  pushForgeRecent(repo);
  $('#forgeReport').innerHTML = '<div class="muted">Checking…</div>';
  try {
    const rep = await api(`/hf/check?repo=${encodeURIComponent(repo)}`);
    $('#forgeReport').innerHTML = renderVerdict(rep);
  } catch (e) { $('#forgeReport').innerHTML = `<div class="verdict too-large">${e.message}</div>`; }
});
$('#forgeGenerate').addEventListener('click', async () => {
  const repo = $('#forgeRepo').value.trim();
  if (!repo) return;
  pushForgeRecent(repo);
  $('#forgeRecipes').innerHTML = '<div class="muted">Forging…</div>';
  try {
    const { report, recipes } = await api(`/hf/forge?repo=${encodeURIComponent(repo)}`);
    $('#forgeReport').innerHTML = renderVerdict(report);
    $('#forgeRecipes').innerHTML = recipes.map((r, i) => renderForgeCard(r, i)).join('');
    window._forgeRecipes = recipes;
    $$('#forgeRecipes [data-forge-run]').forEach((b) => b.addEventListener('click', async () => {
      try {
        const rec = recipes[Number(b.dataset.forgeRun)];
        // Save the recipe first so fix-modal can retrieve _registry context via recipe_id.
        const saved = await api('/recipes', { method: 'POST', body: rec });
        const isDocker = rec.source === 'registry' || rec.source === 'similar' || rec.source === 'synth';
        const runBody = { engine: saved.engine, args: { model: saved.model, ...(saved.args || {}) }, env: saved.env || {}, recipe_id: saved.id };
        const run = await api('/runs', { method: 'POST', body: runBody });
        toast(isDocker
          ? `Started ${run.id} · first run builds the docker image and downloads the model — this can take 20+ min, watch the log stream`
          : `Started ${run.id}`);
        refreshRecipes();
        $('.tab[data-tab="logs"]').click();
        setTimeout(() => window.selectRun(run.id), 50);
      } catch (e) {
        toast(`Failed to start run: ${e.message}`, 'danger');
      }
    }));
    $$('#forgeRecipes [data-forge-save]').forEach((b) => b.addEventListener('click', () => openRecipeModal(recipes[Number(b.dataset.forgeSave)])));
    $$('#forgeRecipes [data-forge-yaml]').forEach((b) => b.addEventListener('click', () => {
      const block = b.parentElement.querySelector('.yaml-block');
      const open = block.style.display !== 'none';
      block.style.display = open ? 'none' : 'block';
      b.textContent = open ? 'View YAML ▾' : 'Hide YAML ▴';
    }));
  } catch (e) { $('#forgeRecipes').innerHTML = `<div class="verdict too-large">${e.message}</div>`; }
});

function renderVerdict(r) {
  return `<div class="verdict ${r.verdict}">
    <h4>${r.verdict.toUpperCase()}: ${escapeHtml(r.repo || '')}</h4>
    <div>${r.reasons.map(escapeHtml).join(' · ')}</div>
    <div class="specs">
      <span class="spec">params: ${r.params_human || '?'}</span>
      <span class="spec">dtype: ${r.dtype || 'unknown'}</span>
      <span class="spec">weights: ${r.weight_gb ? r.weight_gb.toFixed(1) + ' GB' : '?'}</span>
      <span class="spec">context: ${r.context}</span>
      <span class="spec">arch: ${r.architecture || '?'}</span>
      <span class="spec">max tokens @ full ctx: ${r.max_tokens_at_full_ctx ?? '?'}</span>
    </div>
  </div>`;
}

function renderFitBadge(fit) {
  if (!fit || !fit.verdict) return '';
  const verdict = fit.verdict;
  const labels = {
    fits: 'Fits this Spark',
    needs_cluster: `Needs ${fit.required_gpus || '?'} GPUs · have ${fit.available_gpus ?? 0}`,
    too_big: 'Too big for GPU memory',
    unknown: 'Hardware unknown',
  };
  const tip = fit.reason ? ` title="${escapeHtml(fit.reason)}"` : '';
  return `<span class="rc-fit ${verdict}"${tip}>${labels[verdict] || verdict}</span>`;
}

function renderForgeCard(r, i) {
  const source = r.source || 'heuristic';
  const reg = r.registry;
  const sourceLabel = source === 'sparkrun' ? `Community-validated (${escapeHtml(r.sparkrun?.registry || 'sparkrun')})`
    : source === 'registry' ? 'Spark-validated'
    : source === 'similar' ? 'Adapted from registry'
    : source === 'synth' ? 'Synthesized for your hardware'
    : 'Heuristic guess';
  const meta = reg && reg.origin ? `${escapeHtml(reg.origin.repo)}/${escapeHtml(reg.origin.path)}` : '';
  const synth = reg && reg.synth_profile;
  const synthBlock = synth ? `
    <div class="rc-synth">
      <div class="rc-synth-row">
        <span class="rc-synth-pill">family <code>${escapeHtml(synth.family || '?')}</code></span>
        <span class="rc-synth-pill">quant <code>${escapeHtml(synth.quant || '?')}</code></span>
        <span class="rc-synth-pill">container <code>${escapeHtml(synth.container || '?')}</code></span>
        ${synth.tool_call_parser ? `<span class="rc-synth-pill">tool <code>${escapeHtml(synth.tool_call_parser)}</code></span>` : ''}
        ${synth.reasoning_parser ? `<span class="rc-synth-pill">reasoning <code>${escapeHtml(synth.reasoning_parser)}</code></span>` : ''}
        ${synth.chat_template ? `<span class="rc-synth-pill">template <code>${escapeHtml(synth.chat_template)}</code></span>` : ''}
        ${(synth.build_args || []).length ? `<span class="rc-synth-pill">build <code>${escapeHtml(synth.build_args.join(' '))}</code></span>` : ''}
      </div>
      ${(synth.mods || []).length ? `<div class="rc-synth-mods">mods: ${synth.mods.map((m) => `<code>${escapeHtml(m.split('/').pop())}</code>`).join(', ')}</div>` : ''}
    </div>` : '';
  const yamlToggle = reg && reg.raw_yaml ? `
    <button class="yaml-toggle" data-forge-yaml="${i}">View YAML ▾</button>
    <pre class="yaml-block" style="display:none">${escapeHtml(reg.raw_yaml)}</pre>` : '';
  const previewArgs = { ...r.args };
  delete previewArgs._registry;
  const fitBadge = renderFitBadge(r.fit);
  const fitNote = r.fit && r.fit.reason && r.fit.verdict !== 'fits'
    ? `<div class="rc-fit-note">${escapeHtml(r.fit.reason)}</div>` : '';
  const blocked = r.fit && (r.fit.verdict === 'needs_cluster' || r.fit.verdict === 'too_big');
  return `
    <div class="recipe-card${blocked ? ' is-blocked' : ''}">
      <div class="rc-head"><div class="rc-name">${escapeHtml(r.name)}</div><div class="rc-engine">${r.engine}</div></div>
      <div class="rc-source-row">
        <span class="rc-source ${source}">${sourceLabel}</span>
        ${fitBadge}
        ${meta ? `<span class="rc-source-meta">${meta}</span>` : ''}
      </div>
      <div class="rc-model">${escapeHtml(r.model || '')}</div>
      <div class="rc-notes">${escapeHtml(r.notes)}</div>
      ${fitNote}
      ${synthBlock}
      ${reg && reg.container && !synth ? `<div class="rc-source-meta">image: <code>${escapeHtml(reg.container)}</code>${(reg.mods||[]).length ? ' · mods: ' + reg.mods.map(m => `<code>${escapeHtml(m.split('/').pop())}</code>`).join(', ') : ''}</div>` : ''}
      ${Object.keys(previewArgs).length ? `<pre class="mono" style="font-size:11px;background:var(--bg);padding:8px;border-radius:6px;overflow:auto">${escapeHtml(JSON.stringify(previewArgs, null, 2))}</pre>` : ''}
      ${yamlToggle}
      <div class="rc-actions">
        <button class="btn primary" data-forge-run="${i}">▶ Run</button>
        <button class="btn" data-forge-save="${i}">Save</button>
      </div>
    </div>`;
}

async function refreshRegistryStatus() {
  const el = $('#registryStatus');
  if (!el) return;
  try {
    const s = await api('/registry/status');
    const present = (s.repos || []).filter((r) => r.present).length;
    const total = (s.repos || []).length;
    const cls = s.recipe_count > 0 ? 'ok' : present > 0 ? 'stale' : 'empty';
    el.className = `registry-status ${cls}`;
    const repos = (s.repos || []).map((r) => `<span class="repo-pill">${escapeHtml(r.name)}${r.present ? ' @ ' + (r.commit || '?') : ' (missing)'}</span>`).join(' ');
    const indexed = s.indexed_at ? new Date(s.indexed_at * 1000).toLocaleTimeString() : null;
    const ageHint = indexed ? ` · synced ${indexed} (auto on every start)` : '';
    const fresh = s.last_sync?.new_recipes || [];
    const freshBadge = fresh.length
      ? `<span class="badge ok" title="${escapeHtml(fresh.map((n) => n.name).join(', '))}">✨ ${fresh.length} new recipe${fresh.length > 1 ? 's' : ''}</span>`
      : '';
    el.innerHTML = `
      <span class="status-dot"></span>
      <span><strong>${s.recipe_count}</strong> recipes · <strong>${s.mod_count}</strong> mods · ${present}/${total} repos${escapeHtml(ageHint)}</span>
      ${freshBadge}
      ${repos}
      <button id="registrySync" class="btn" style="margin-left:auto"><i class="fa-solid fa-arrows-rotate"></i> Refresh now</button>`;
    $('#registrySync').addEventListener('click', syncRegistry);
  } catch (e) {
    el.className = 'registry-status empty';
    el.innerHTML = `<span class="status-dot"></span><span>Registry status unavailable: ${escapeHtml(e.message)}</span>`;
  }
}

async function syncRegistry() {
  const btn = $('#registrySync');
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-arrows-rotate fa-spin"></i> Syncing…'; }
  try {
    const res = await api('/registry/sync', { method: 'POST' });
    const failed = (res.results || []).filter((r) => !r.ok);
    if (failed.length) {
      toast(`Synced with ${failed.length} failure(s): ${failed.map((f) => f.name).join(', ')}`, 'danger');
    } else {
      const fresh = res.status?.last_sync?.new_recipes || [];
      toast(fresh.length
        ? `Registry synced — ${fresh.length} new recipe${fresh.length > 1 ? 's' : ''}: ${fresh.map((n) => n.name).slice(0, 4).join(', ')}${fresh.length > 4 ? '…' : ''}`
        : 'Registry synced — no new recipes');
    }
  } catch (e) {
    toast(`Sync failed: ${e.message}`, 'danger');
  } finally {
    refreshRegistryStatus();
  }
}

document.addEventListener('DOMContentLoaded', refreshRegistryStatus);
// Also kick once now in case the listener missed the event.
setTimeout(refreshRegistryStatus, 100);

// ---------- models --------------------------------------------------------
async function refreshLocalModels() {
  const list = await api('/models/local').catch(() => []);
  const totalGb = list.reduce((s, m) => s + (m.size_gb || 0), 0);
  $('#modelsList').innerHTML = list.length ? `
    <div class="muted" style="margin:8px 0">${list.length} model${list.length > 1 ? 's' : ''} · ${totalGb.toFixed(1)} GB on disk</div>
    <table><tr><th>Repo</th><th>Size</th><th>Cache</th><th></th></tr>
    ${list.map((m) => `<tr><td class="mono">${escapeHtml(m.repo)}</td><td>${m.size_gb} GB</td>
    <td class="mono muted" style="font-size:11px">${escapeHtml(m.cache || '')}</td>
    <td><button class="btn" data-model="${escapeHtml(m.repo)}">Serve with vLLM</button>
        <button class="btn" data-forge="${escapeHtml(m.repo)}">Forge</button>
        <button class="btn danger" data-del-model="${escapeHtml(m.path)}" data-del-repo="${escapeHtml(m.repo)}" data-del-size="${m.size_gb}">Del</button></td></tr>`).join('')}</table>`
    : '<div class="muted" style="margin:8px 0">No cached models found in any known HF cache.</div>';
  $$('#modelsList [data-model]').forEach((b) => b.addEventListener('click', () => {
    $(`.panel[data-panel="vllm"] .recipe-editor`).value = JSON.stringify({ model: b.dataset.model, 'max-model-len': 16384 }, null, 2);
    $('.tab[data-tab="vllm"]').click();
  }));
  $$('#modelsList [data-forge]').forEach((b) => b.addEventListener('click', () => {
    $('#forgeRepo').value = b.dataset.forge;
    $('.tab[data-tab="forge"]').click();
    $('#forgeGenerate').click();
  }));
  $$('#modelsList [data-del-model]').forEach((b) => b.addEventListener('click', async () => {
    const repo = b.dataset.delRepo;
    if (!confirm(`Delete ${repo} (${b.dataset.delSize} GB) from the HF cache?\n\nStop any run serving it first — a running engine keeps the files alive until it exits. You can re-download it later from HuggingFace.`)) return;
    b.disabled = true;
    try {
      const res = await api(`/models/local?path=${encodeURIComponent(b.dataset.delModel)}`, { method: 'DELETE' });
      toast(`Deleted ${res.deleted} — freed ${res.freed_gb} GB`);
    } catch (e) { toast(e.message, 'danger'); }
    refreshLocalModels();
  }));
}
$('#modelsRefresh').addEventListener('click', refreshLocalModels);

// ---------- runs & logs ---------------------------------------------------
let currentStream = null;
let currentRunId = null;
let currentRunRecipe = null;

function decodeShellBody(run) {
  if (run.raw_cmd) return run.raw_cmd;
  const cmd = String(run.cmd || '').trim();
  const m = cmd.match(/^bash -lc ('([\s\S]*)'|"([\s\S]*)")$/);
  if (!m) return null;
  return (m[2] ?? m[3] ?? '').replace(/'\\''/g, "'");
}

async function recipeFromRun(run) {
  if (run.recipe_id) {
    try { return await api(`/recipes/${run.recipe_id}`); } catch {}
  }
  const raw = decodeShellBody(run);
  const partial = {
    id: null,
    engine: run.engine,
    model: null,
    args: {},
    env: {},
    notes: '',
    tags: runOutcome(run) === 'failed' ? 'failed-run' : '',
    raw_cmd: raw,
  };
  const enriched = enrichRecipe(partial);
  const model = enriched.model;
  return { ...enriched, name: model ? `${run.engine} ${model}` : `${run.engine} recipe` };
}

function summarizeRunFailure(run, lines) {
  const text = lines.join('\n');
  if (/invalid reference format/i.test(text)) {
    return 'Docker command is malformed. A mount path or image name was split incorrectly, often by spaces or uppercase path fragments.';
  }
  if (/includes invalid characters for a local volume name/i.test(text)) {
    return 'Docker received a literal shell variable in a volume mount. The recipe needs an absolute expanded host path before `docker run`.';
  }
  if (/No such container/i.test(text)) {
    return 'The wrapper expected a container that never started. Fix the first `docker run` failure before applying container patch steps.';
  }
  if (/argument --hf-overrides/i.test(text)) {
    return 'The recipe passed invalid JSON to `--hf-overrides`. The payload needs valid JSON with normal braces and correct quoting.';
  }
  if (/Engine core initialization failed/i.test(text) && /Failed core proc\(s\): \{\}/i.test(text)) {
    return 'vLLM reached API server startup, but the engine subprocess died before surfacing a structured cause. Spark Studio now appends recent container logs after failure; check the `[container:...]` lines for the real error, which is usually an incompatible model/image/flag combination or a CUDA library mismatch inside the container.';
  }
  if (/tool-call-parser/i.test(text) && /qwen/i.test(text)) {
    return 'This run is using a Qwen tool-call parser path. That can be unstable for plain chat recipes and may need a safer serving configuration.';
  }
  if (run.status === 'exited' && run.exit_code) {
    const last = [...lines].reverse().find((line) => line.trim() && !line.startsWith('[cleanup]'));
    return last || `Run exited with code ${run.exit_code}.`;
  }
  return '';
}

function setRecoveryButtons(disabled) {
  ['logsSaveRecipe', 'logsEditRecipe', 'logsFixInline', 'logsFixInlineCodex'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.disabled = disabled;
  });
}

function updateLogsRecovery(run, lines) {
  if (window._autofixActive) return; // autofix owns the recovery box while looping
  const box = $('#logsRecovery');
  const title = $('#logsRecoveryTitle');
  const text = $('#logsRecoveryText');
  box.classList.remove('is-error', 'is-running');

  if (!run) {
    box.hidden = true;
    setRecoveryButtons(true);
    return;
  }

  box.hidden = false;
  if (run.status === 'running') {
    box.classList.add('is-running');
    title.textContent = 'Recipe is running';
    text.textContent = currentRunRecipe?.id
      ? `Saved recipe ${currentRunRecipe.name || currentRunRecipe.id} is active. You can still open it or ask an agent to optimize it.`
      : 'This run came from an ad hoc command. Save it as a recipe if you want to keep, edit, or optimize it.';
  } else if (runOutcome(run) === 'failed') {
    box.classList.add('is-error');
    title.textContent = 'Run failed. Fix it in-app and save the repaired recipe.';
    text.textContent = summarizeRunFailure(run, lines) || `Run exited with code ${run.exit_code}. Ask Claude or Codex to patch and save the recipe.`;
  } else if (runOutcome(run) === 'stopped') {
    title.textContent = 'Run stopped';
    text.textContent = currentRunRecipe?.id
      ? 'You stopped this run. Relaunch its recipe any time, or ask an agent to optimize it first.'
      : 'You stopped this ad hoc run. Save it as a recipe if you want to launch it again later.';
  } else {
    title.textContent = 'Run completed';
    text.textContent = currentRunRecipe?.id
      ? 'This recipe completed cleanly. You can reopen it, duplicate it, or ask an agent to optimize it.'
      : 'This ad hoc run completed cleanly. Save it as a recipe if you want to keep it.';
  }

  const canEdit = !!(currentRunRecipe && (currentRunRecipe.raw_cmd || Object.keys(currentRunRecipe.args || {}).length || currentRunRecipe.model));
  setRecoveryButtons(!canEdit);
  // Optimize needs a healthy serving endpoint (Auto-Fix covers broken runs).
  const canOptimize = !!(run && run.status === 'running' && run.ready);
  ['logsOptimize', 'logsOptimizeCodex'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.disabled = !canOptimize;
  });
  // Failed multi-node sparkrun run → offer the classic recovery: fewer nodes.
  const retry = $('#logsRetryLowerTp');
  if (retry) {
    const lowerTp = (run && run.engine === 'sparkrun' && runOutcome(run) === 'failed'
      && run.ref && Number(run.tp) > 1) ? Math.floor(Number(run.tp) / 2) : 0;
    retry.hidden = !lowerTp;
    if (lowerTp) {
      retry.textContent = `⬇ Retry with TP ${lowerTp}`;
      retry.dataset.ref = run.ref;
      retry.dataset.tp = String(lowerTp);
    }
  }
}
$('#logsRetryLowerTp')?.addEventListener('click', async (ev) => {
  const b = ev.currentTarget;
  try {
    const run = await api('/sparkrun/run', { method: 'POST', body: { ref: b.dataset.ref, tp: Number(b.dataset.tp) } });
    toast(`Relaunched ${b.dataset.ref} with TP ${b.dataset.tp}`);
    window.selectRun(run.id);
  } catch (e) { toast(e.message, 'danger'); }
});

async function refreshRuns() {
  const runs = await api('/runs').catch(() => []);
  $('#runsList').innerHTML = runs.map((r) =>
    `<div class="run-item" data-id="${r.id}" title="run ${r.id}">
      <div class="ri-top"><span class="ri-engine">${r.engine}</span><span class="ri-status ${runOutcome(r)}">${runOutcome(r)}</span></div>
      <div class="ri-name">${escapeHtml(runLabel(r))}</div>
      <div class="ri-id">${r.port ? ':' + r.port + ' · ' : ''}${r.id.slice(0, 8)} · ${fmtTime(r.started_at)}${loadStats(r) ? ' · ' + loadStats(r) : ''}</div>
    </div>`).join('') || '<div class="muted">No runs yet.</div>';
  $$('#runsList .run-item').forEach((el) => el.addEventListener('click', () => window.selectRun(el.dataset.id)));
}

window.selectRun = async (rid) => {
  currentRunId = rid;
  $$('#runsList .run-item').forEach((el) => el.classList.toggle('active', el.dataset.id === rid));
  const r = await api(`/runs/${rid}`);
  currentRunRecipe = await recipeFromRun(r);
  $('#logsTitle').textContent = `${runLabel(r)} · ${r.engine} · ${runOutcome(r)}`;
  $('#logsOutput').textContent = '';
  ['logsStop', 'logsKill', 'logsFix', 'logsFixCodex'].forEach((id) => ($(`#${id}`).disabled = false));
  if (currentStream) currentStream.close();
  const initialTail = await api(`/runs/${rid}/tail?n=300`).catch(() => ({ lines: [] }));
  updateLogsRecovery(r, initialTail.lines || []);
  currentStream = new EventSource(`/api/runs/${rid}/stream`);
  currentStream.addEventListener('log', (ev) => {
    const pre = $('#logsOutput');
    const stickToBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 24;
    pre.textContent += ev.data + '\n';
    if (stickToBottom) pre.scrollTop = pre.scrollHeight;
  });
  currentStream.addEventListener('eof', async () => {
    currentStream.close();
    try {
      const latest = await api(`/runs/${rid}`);
      const tail = await api(`/runs/${rid}/tail?n=300`).catch(() => ({ lines: [] }));
      updateLogsRecovery(latest, tail.lines || []);
      $('#logsTitle').textContent = `${runLabel(latest)} · ${latest.engine} · ${runOutcome(latest)}`;
      // working/fix tags are written server-side now (watchdog + runner) —
      // just repaint the recipe cards so fresh badges show up.
      if (latest.recipe_id) refreshRecipes();
    } catch {}
  });
};

$('#logsStop').addEventListener('click', async () => currentRunId && api(`/runs/${currentRunId}/stop`, { method: 'POST' }));
$('#logsKill').addEventListener('click', async () => currentRunId && api(`/runs/${currentRunId}/stop?force=true`, { method: 'POST' }));
$('#logsFix').addEventListener('click', () => openAgentModal('claude'));
$('#logsFixCodex').addEventListener('click', () => openAgentModal('codex'));
$('#logsFixInline').addEventListener('click', () => openAgentModal('claude'));
$('#logsFixInlineCodex').addEventListener('click', () => openAgentModal('codex'));

// ---------- hands-free fix / optimize loops --------------------------------
let autofixES = null;
function setAutofixButtons(disabled) {
  ['logsAutofix', 'logsAutofixCodex', 'logsOptimize', 'logsOptimizeCodex'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.disabled = disabled;
  });
}
// Shared SSE driver for Auto-Fix and Optimize — both stream `af` events.
function startAgentLoop(agentName, kind) {
  if (!currentRunId) return;
  if (autofixES) { autofixES.close(); autofixES = null; }
  const label = kind === 'optimize' ? 'Optimize Speed' : 'Auto-Fix & Retry';
  const endpoint = kind === 'optimize'
    ? `/api/agents/optimize/${currentRunId}?agent=${agentName}&attempts=2`
    : `/api/agents/autofix/${currentRunId}?agent=${agentName}&attempts=3`;
  window._autofixActive = true;
  setAutofixButtons(true);
  const box = $('#logsRecovery');
  const title = $('#logsRecoveryTitle');
  const text = $('#logsRecoveryText');
  box.hidden = false;
  box.classList.remove('is-error');
  box.classList.add('is-running');
  title.textContent = `${label} (${agentName})`;
  text.textContent = 'Starting…';
  const finish = () => {
    window._autofixActive = false;
    setAutofixButtons(false);
    if (autofixES) { autofixES.close(); autofixES = null; }
  };
  autofixES = new EventSource(endpoint);
  autofixES.addEventListener('af', (ev) => {
    let d;
    try { d = JSON.parse(ev.data); } catch { return; }
    if (d.type === 'status') {
      text.textContent = d.text;
    } else if (d.type === 'bench') {
      text.textContent = `📊 ${d.text}`;
    } else if (d.type === 'diagnosis') {
      text.textContent = `Diagnosis: ${d.text}`;
    } else if (d.type === 'launched') {
      // Follow the new run's log stream; the _autofixActive flag keeps
      // selectRun's recovery-box update from clobbering our status text.
      title.textContent = `${label} (${agentName}) — attempt ${d.attempt}`;
      window.selectRun(d.run_id);
    } else if (d.type === 'done') {
      finish();
      title.textContent = d.ok ? `${label} finished` : `${label} gave up`;
      text.textContent = d.text;
      box.classList.toggle('is-error', !d.ok);
      box.classList.toggle('is-running', !!d.ok);
      toast(d.text, d.ok ? undefined : 'danger');
      refreshRecipes();
      refreshRuns();
    }
  });
  autofixES.addEventListener('error', () => {
    finish();
    text.textContent = `${label} stream disconnected.`;
  });
}
const startAutofix = (agentName) => startAgentLoop(agentName, 'autofix');
$('#logsAutofix').addEventListener('click', () => startAutofix('claude'));
$('#logsAutofixCodex').addEventListener('click', () => startAutofix('codex'));
$('#logsOptimize').addEventListener('click', () => startAgentLoop('claude', 'optimize'));
$('#logsOptimizeCodex').addEventListener('click', () => startAgentLoop('codex', 'optimize'));
$('#logsSaveRecipe').addEventListener('click', () => {
  if (!currentRunRecipe) return;
  openRecipeModal(currentRunRecipe);
});
$('#logsEditRecipe').addEventListener('click', async () => {
  if (!currentRunRecipe) return;
  if (currentRunRecipe.id) {
    openRecipeModal(await api(`/recipes/${currentRunRecipe.id}`));
  } else {
    openRecipeModal(currentRunRecipe);
  }
});

function defaultAgentGoal(mode, recipe = {}) {
  if (mode === 'optimize') {
    return `Maximize tokens/second (tok/s) throughput for this ${recipe.engine || 'engine'} recipe on NVIDIA DGX Spark (Grace-Blackwell GB10, 128 GB unified memory, CUDA 13, aarch64). Push gpu-memory-utilization higher (0.92+), enable chunked prefill, use FlashInfer attention backend, enable fp8 KV cache, increase batch sizes, and consider MXFP4 quantization (vllm-node-mxfp4 container) if the model supports it. Only use --enforce-eager if absolutely necessary — it kills throughput. Return the patched recipe as JSON.`;
  }
  return `Fix this ${recipe.engine || 'engine'} recipe so it launches successfully on NVIDIA DGX Spark (Grace-Blackwell GB10, 128 GB unified memory, CUDA 13, aarch64). Preserve the user's intent and model unless the logs show the current path is incompatible.`;
}

function setFixBusy(isBusy, which = 'agent') {
  ['fixAnalyze', 'fixOptimize', 'fixSend', 'fixApply'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.disabled = isBusy;
  });
  $('#fixTitle').textContent = isBusy
    ? `${which === 'claude' ? 'Claude' : 'Codex'} is analyzing…`
    : `${which === 'claude' ? 'Claude' : 'Codex'} suggestion`;
}

async function openAgentModal(which) {
  if (!currentRunId) return;
  const r = await api(`/runs/${currentRunId}`);
  const recipe = currentRunRecipe || await recipeFromRun(r);
  $('#fixTitle').textContent = `${which === 'claude' ? 'Claude' : 'Codex'} assistant`;
  $('#fixGoal').value = '';
  $('#fixDiagnosis').innerHTML = '<div class="muted">Choose Fix Recipe, Optimize, or write your own instruction and send it.</div>';
  $('#fixPatched').value = '';
  $('#fixNotes').innerHTML = '';
  $('#fixModal').dataset.baseRecipe = JSON.stringify(recipe);
  $('#fixModal').dataset.agent = which;
  $('#fixModal').hidden = false;
}

async function askAgent(goal = '') {
  try {
    const which = $('#fixModal').dataset.agent || 'claude';
    const tail = await api(`/runs/${currentRunId}/tail?n=300`);
    const recipe = JSON.parse($('#fixModal').dataset.baseRecipe || '{}');
    const finalGoal = goal || $('#fixGoal').value.trim() || defaultAgentGoal('fix', recipe);
    $('#fixGoal').value = finalGoal;
    setFixBusy(true, which);
    $('#fixDiagnosis').innerHTML = '<div class="muted">Thinking…</div>';
    $('#fixPatched').value = '';
    $('#fixNotes').innerHTML = '';
    const res = await api('/agents/fix', {
      method: 'POST',
      body: { agent: which, recipe, logs: tail.lines.join('\n'), goal: finalGoal },
    });
    setFixBusy(false, which);
    $('#fixDiagnosis').innerHTML = `<p>${escapeHtml(res.diagnosis || '')}</p>`;
    $('#fixPatched').value = JSON.stringify(res.patched_recipe || {}, null, 2);
    $('#fixNotes').innerHTML = (res.diff_notes || []).map((n) => `<div class="muted">• ${escapeHtml(n)}</div>`).join('');
  } catch (e) {
    setFixBusy(false, $('#fixModal').dataset.agent || 'claude');
    $('#fixDiagnosis').innerHTML = `<div class="verdict too-large">${escapeHtml(e.message)}</div>`;
  }
}

$('#fixAnalyze').addEventListener('click', () => {
  try {
    const recipe = JSON.parse($('#fixModal').dataset.baseRecipe || '{}');
    askAgent(defaultAgentGoal('fix', recipe));
  } catch (e) {
    toast(e.message, 'danger');
  }
});
$('#fixOptimize').addEventListener('click', () => {
  try {
    const recipe = JSON.parse($('#fixModal').dataset.baseRecipe || '{}');
    askAgent(defaultAgentGoal('optimize', recipe));
  } catch (e) {
    toast(e.message, 'danger');
  }
});
$('#fixSend').addEventListener('click', () => askAgent());
$$('#fixModal [data-close]').forEach((b) => b.addEventListener('click', () => ($('#fixModal').hidden = true)));
$('#fixApply').addEventListener('click', async () => {
  try {
    const base = enrichRecipe(JSON.parse($('#fixModal').dataset.baseRecipe || '{}'));
    const patched = JSON.parse($('#fixPatched').value);
    const mergedArgs = { ...(base.args || {}), ...(patched.args || {}) };

    // If the agent's patch doesn't return a _registry block, strip the base
    // recipe's _registry so the agent's raw_cmd or flat args actually take
    // effect instead of the original registry pipeline silently winning.
    const patchProvidesRegistry = !!(patched.args && patched.args._registry);
    if (!patchProvidesRegistry) {
      delete mergedArgs._registry;
    }

    const rawChanged = Object.prototype.hasOwnProperty.call(patched, 'raw_cmd')
      || Object.prototype.hasOwnProperty.call(patched, 'cmd')
      || Object.prototype.hasOwnProperty.call(patched, 'command');
    const raw = rawChanged
      ? (patched.raw_cmd || patched.cmd || patched.command || null)
      : (base.raw_cmd || null);
    const model = patched.model ?? base.model ?? mergedArgs.model ?? inferRecipeModel({ ...base, ...patched, args: mergedArgs, raw_cmd: raw }) ?? null;
    delete mergedArgs.model;
    const _eng = patched.engine || base.engine || 'vllm';
    const _recipeName = (() => {
      if (model) return `${_eng} ${model}`;
      if (patched.name && !/recovered|patched/i.test(patched.name)) return patched.name;
      if (base.name && !/recovered|patched/i.test(base.name)) return base.name;
      return `${_eng} recipe`;
    })();
    const savedRecipe = enrichRecipe({
      id: base.id ?? patched.id ?? null,
      name: _recipeName,
      engine: _eng,
      model,
      args: mergedArgs,
      env: { ...(base.env || {}), ...(patched.env || {}) },
      notes: patched.notes ?? base.notes ?? '',
      tags: patched.tags ?? base.tags ?? '',
      raw_cmd: raw,
    });
    const saved = await api('/recipes', { method: 'POST', body: savedRecipe });
    currentRunRecipe = saved;
    const hasRegistry = !!(saved.args && saved.args._registry);
    const runBody = (!hasRegistry && saved.raw_cmd)
      ? { engine: saved.engine, raw_cmd: saved.raw_cmd, env: saved.env || {}, recipe_id: saved.id }
      : { engine: saved.engine, args: { model: saved.model, ...(saved.args || {}) }, env: saved.env || {}, recipe_id: saved.id };
    // Stop any currently running engine before launching the patched recipe
    // so the GPU is free (DGX Spark can't run two engines simultaneously).
    try {
      const active = await api('/active').catch(() => null);
      if (active && active.id) {
        toast('Stopping current engine…');
        await api(`/runs/${active.id}/stop`, { method: 'POST' });
        // Give it a moment to release GPU memory before starting the new run.
        await new Promise((r) => setTimeout(r, 3000));
      }
    } catch (_) { /* best effort */ }

    const run = await api('/runs', { method: 'POST', body: runBody });
    $('#fixModal').hidden = true;
    refreshRecipes();
    toast(`Started patched run ${run.id}`);
    setTimeout(() => window.selectRun(run.id), 50);
  } catch (e) { toast(e.message, 'danger'); }
});

// ---------- Launch Settings modal ----------------------------------------
let _launchRecipe = null;
function openLaunchModal(r) {
  _launchRecipe = r;
  const reg = r.args._registry;
  const d = reg.defaults || {};
  $('#launchModalTitle').textContent = `Launch: ${r.name || r.model || 'Recipe'}`;
  $('#launchModalDesc').textContent = r.model || '';
  $('#lsMaxModelLen').value = d.max_model_len ?? 131072;
  $('#lsGpuMem').value = d.gpu_memory_utilization ?? 0.90;
  $('#lsMaxBatched').value = d.max_num_batched_tokens ?? d.max_model_len ?? 131072;
  $('#lsMaxSeqs').value = d.max_num_seqs ?? 16;
  // Try to read native max from raw_yaml; fall back to descriptive text
  let nativeMax = 'see model card';
  try {
    const raw = reg.raw_yaml || '';
    const m = raw.match(/max_position_embeddings[:\s]+(\d+)/);
    if (m) nativeMax = parseInt(m[1], 10).toLocaleString();
  } catch (_) {}
  $('#lsNativeMax').textContent = nativeMax;
  $('#launchModal').hidden = false;
}
$$('#launchModal [data-close]').forEach((b) => b.addEventListener('click', () => { $('#launchModal').hidden = true; }));
$('#lsRun').addEventListener('click', async () => {
  if (!_launchRecipe) return;
  try {
    const r = _launchRecipe;
    const reg = JSON.parse(JSON.stringify(r.args._registry));
    const maxLen = parseInt($('#lsMaxModelLen').value, 10);
    reg.defaults = {
      ...(reg.defaults || {}),
      max_model_len: maxLen,
      max_num_batched_tokens: parseInt($('#lsMaxBatched').value, 10) || maxLen,
      gpu_memory_utilization: parseFloat($('#lsGpuMem').value),
      max_num_seqs: parseInt($('#lsMaxSeqs').value, 10) || 16,
    };
    const body = {
      engine: r.engine,
      args: { model: r.model, ...r.args, _registry: reg },
      env: r.env || {},
      recipe_id: r.id,
    };
    const run = await api('/runs', { method: 'POST', body });
    $('#launchModal').hidden = true;
    toast(`Started ${run.id} · first run builds the docker image and downloads the model — this can take 20+ min`);
    $('.tab[data-tab="logs"]').click();
    setTimeout(() => window.selectRun(run.id), 50);
  } catch (e) { toast(e.message, 'danger'); }
});

// ---------- chat & canvas ------------------------------------------------
let monacoEditor = null;
let canvasPreviewDoc = '';
let canvasPopup = null;
const CANVAS_STOP_DOC = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Canvas Stopped</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #05080d; color: #e6edf7; font-family: system-ui, sans-serif; }
    body { display: grid; place-items: center; }
    .box { padding: 18px 22px; border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; background: rgba(255,255,255,0.03); }
  </style>
</head>
<body>
  <div class="box">Preview stopped</div>
</body>
</html>`;

function applyCanvasSplit(previewPercent) {
  const preview = Math.max(25, Math.min(80, Number(previewPercent) || 62));
  const editor = 100 - preview;
  $('#canvasEditor').style.flex = `0 0 ${editor}%`;
  $('.canvas-preview-shell').style.flex = `1 1 ${preview}%`;
  $('#canvasStatus').textContent = `Running ${$('#canvasLang').value} preview · ${preview}% canvas`;
  if (monacoEditor) monacoEditor.layout();
}

function escapeScriptTag(s) {
  return String(s).replace(/<\/script/gi, '<\\/script');
}

// null = auto-detect, true = force page/doc mode, false = force game mode
let canvasPageModeForced = null;

// Returns true when the HTML looks like a canvas game (has drawingContext + animation loop).
// Both signals must be present — a decorative canvas on a page won't trigger this.
function _isGameContent(src) {
  const hasCanvasCtx = /\.getContext\s*\(\s*['"](?:2d|webgl2?|bitmaprenderer)/i.test(src);
  const hasAnimLoop  = /requestAnimationFrame|setInterval/i.test(src);
  return hasCanvasCtx && hasAnimLoop;
}

// CSS injected into document-type previews to un-clip tall content.
// Uses !important to override the common `html, body { height: 100%; overflow: hidden; }`
// that models copy from game templates.
const _PAGE_NORMALIZE_CSS = `<style id="_sp-page">
html { height: auto !important; min-height: 100%; }
body { height: auto !important; min-height: 100%; overflow-y: auto !important; overflow-x: hidden; }
</style>`;

function withCanvasGuards(doc, opts = {}) {
  const normalize = opts.page ? _PAGE_NORMALIZE_CSS : '';
  const guard = `<script>
(() => {
  const emit = (kind, msg) => {
    try { parent?.postMessage?.({ source: 'spark-canvas-preview', kind, message: String(msg ?? '') }, '*'); } catch {}
    const node = document.getElementById('overlay');
    if (node) node.textContent = String(msg ?? kind);
  };
  window.alert = (msg) => emit('alert', msg);
  window.confirm = (msg) => { emit('confirm', msg); return false; };
  window.prompt = (msg, def = '') => { emit('prompt', msg); return def ?? ''; };
  window.onerror = (msg) => emit('error', msg);
})();
</script>`;
  const inject = normalize + guard;
  if (/<head[\s>]/i.test(doc)) {
    return doc.replace(/<head([^>]*)>/i, `<head$1>${inject}`);
  }
  if (/<body[\s>]/i.test(doc)) {
    return doc.replace(/<body([^>]*)>/i, `<body$1>${inject}`);
  }
  return `${inject}${doc}`;
}

function _resolvePageMode(source) {
  if (canvasPageModeForced !== null) return canvasPageModeForced;
  return !_isGameContent(source);
}

function buildCanvasDoc(code, lang) {
  const source = code || '';
  if (/^\s*<!doctype html/i.test(source) || /^\s*<html[\s>]/i.test(source)) {
    return withCanvasGuards(source, { page: _resolvePageMode(source) });
  }
  if (lang === 'html') {
    return withCanvasGuards(source, { page: _resolvePageMode(source) });
  }
  if (lang === 'javascript' || lang === 'typescript') {
    return withCanvasGuards(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Canvas Preview</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #05080d; color: #e6edf7; font-family: system-ui, sans-serif; overflow: hidden; }
    #game { width: 100vw; height: 100vh; display: block; }
    #overlay { position: fixed; top: 12px; left: 12px; padding: 6px 10px; border-radius: 999px; background: rgba(0,0,0,0.45); font-size: 12px; z-index: 2; }
  </style>
</head>
<body>
  <div id="overlay">Canvas preview</div>
  <canvas id="game"></canvas>
  <script>
${escapeScriptTag(source)}
  </script>
</body>
</html>`);
  }
  return withCanvasGuards(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Canvas Preview</title>
  <style>
    body { margin: 0; padding: 24px; background: #05080d; color: #e6edf7; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    pre { white-space: pre-wrap; word-break: break-word; }
  </style>
</head>
<body>
  <h3>Preview is best with HTML or JavaScript</h3>
  <pre>${escapeHtml(source)}</pre>
</body>
</html>`);
}

function runCanvasPreview() {
  if (!monacoEditor) return;
  const lang = $('#canvasLang').value;
  const code = monacoEditor.getValue();
  const doc = buildCanvasDoc(code, lang);
  canvasPreviewDoc = doc;
  $('#canvasPreview').srcdoc = doc;
  const previewSize = $('#canvasSplit')?.value || '62';
  $('#canvasStatus').textContent = `Running ${lang} preview · ${previewSize}% canvas`;
  if (canvasPopup && !canvasPopup.closed) {
    canvasPopup.document.open();
    canvasPopup.document.write(doc);
    canvasPopup.document.close();
  }
}

function stopCanvasPreview() {
  canvasPreviewDoc = CANVAS_STOP_DOC;
  $('#canvasPreview').srcdoc = CANVAS_STOP_DOC;
  $('#canvasStatus').textContent = 'Preview stopped';
  if (canvasPopup && !canvasPopup.closed) {
    canvasPopup.document.open();
    canvasPopup.document.write(CANVAS_STOP_DOC);
    canvasPopup.document.close();
  }
}

function popoutCanvasPreview() {
  const doc = canvasPreviewDoc || ($('#canvasPreview').srcdoc || '');
  if (!doc) {
    toast('Nothing to preview yet', 'danger');
    return;
  }
  canvasPopup = (canvasPopup && !canvasPopup.closed)
    ? canvasPopup
    : window.open('about:blank', 'spark-studio-canvas-preview');
  if (!canvasPopup) {
    toast('Popup blocked', 'danger');
    return;
  }
  canvasPopup.document.open();
  canvasPopup.document.write(doc);
  canvasPopup.document.close();
  canvasPopup.focus();
}

// Upgrade a plain <textarea> to a Monaco editor. The textarea stays in the
// DOM (hidden) and its `value` property is redirected at the instance level,
// so every existing read/write keeps working untouched.
function enhanceEditor(textarea, language) {
  if (!textarea || textarea.dataset.monaco) return;
  textarea.dataset.monaco = '1';
  window.monacoReady.then((monaco) => {
    const height = Math.max(
      textarea.offsetHeight || 0,
      parseInt(textarea.style.minHeight, 10) || 0,
      320,
    );
    const host = document.createElement('div');
    host.className = 'monaco-recipe-editor';
    host.style.height = `${height}px`;
    textarea.insertAdjacentElement('afterend', host);
    textarea.style.display = 'none';
    const ed = monaco.editor.create(host, {
      value: textarea.value,
      language,
      theme: 'spark',
      minimap: { enabled: false },
      fontSize: 12,
      scrollBeyondLastLine: false,
      automaticLayout: true,
      wordWrap: 'on',
      tabSize: 2,
      padding: { top: 8 },
    });
    Object.defineProperty(textarea, 'value', {
      configurable: true,
      get: () => ed.getValue(),
      set: (v) => { if ((v ?? '') !== ed.getValue()) ed.setValue(v ?? ''); },
    });
  });
}

window.monacoReady.then(() => {
  monacoEditor = monaco.editor.create($('#canvasEditor'), {
    value: `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spark Studio Infographic</title>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-secondary: #818cf8;
            --text: #f1f5f9;
            --text-dim: #94a3b8;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }

        body {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container { max-width: 1000px; width: 100%; }

        header { text-align: center; margin-bottom: 40px; }

        h1 {
            font-size: 3rem;
            margin: 0;
            background: linear-gradient(to right, var(--accent), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle { color: var(--text-dim); font-size: 1.2rem; }

        #canvas-container {
            position: relative;
            width: 100%;
            height: 400px;
            background: #020617;
            border-radius: 16px;
            border: 1px solid #334155;
            margin-bottom: 40px;
            overflow: hidden;
        }

        canvas { display: block; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .card {
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #334155;
            transition: transform 0.2s;
        }

        .card:hover { transform: translateY(-5px); border-color: var(--accent); }

        .card h3 { margin-top: 0; color: var(--accent); display: flex; align-items: center; gap: 10px; }

        .card ul { list-style: none; padding: 0; margin: 0; }

        .card li {
            margin-bottom: 10px;
            color: var(--text-dim);
            font-size: 0.95rem;
            display: flex;
            align-items: flex-start;
        }

        .card li::before { content: "▹"; color: var(--accent-secondary); margin-right: 8px; }

        .steps-container {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-top: 20px;
        }

        .step {
            flex: 1;
            background: #0f172a;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 0.85rem;
            border-top: 4px solid var(--accent-secondary);
        }

        .step-num { display: block; font-weight: bold; color: var(--accent); margin-bottom: 5px; }

        footer {
            text-align: center;
            color: var(--text-dim);
            font-size: 0.8rem;
            margin-top: 40px;
            padding-bottom: 40px;
        }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Spark Studio</h1>
        <p class="subtitle">The Ultimate LLM Orchestration Dashboard for NVIDIA DGX Spark</p>
    </header>

    <div id="canvas-container">
        <canvas id="infographicCanvas"></canvas>
    </div>

    <div class="grid">
        <div class="card">
            <h3>🚀 Core Engines</h3>
            <ul>
                <li>vLLM (High Throughput)</li>
                <li>SGLang (Optimized Serving)</li>
                <li>llama.cpp (Local/Quantized)</li>
            </ul>
        </div>
        <div class="card">
            <h3>🤖 AI Assistance</h3>
            <ul>
                <li><strong>Ask Claude:</strong> Auto-diagnose logs</li>
                <li><strong>Ask Codex:</strong> Patch broken recipes</li>
                <li><strong>Recipe Forge:</strong> HF ID to YAML</li>
                <li><strong>One-click Agent:</strong> No API keys needed</li>
            </ul>
        </div>
        <div class="card">
            <h3>🛠️ Management</h3>
            <ul>
                <li>Hardware-aware fit checks</li>
                <li>Local HF Cache scanning</li>
                <li>SQLite Recipe Library</li>
                <li>OpenAI-compatible Gateway</li>
            </ul>
        </div>
    </div>

    <h2 style="text-align: center; color: var(--accent-secondary);">Deployment Pipeline</h2>
    <div class="steps-container">
        <div class="step"><span class="step-num">01</span>Install <code>uv</code></div>
        <div class="step"><span class="step-num">02</span>Clone Repo</div>
        <div class="step"><span class="step-num">03</span>Create Venv</div>
        <div class="step"><span class="step-num">04</span>Install Deps</div>
        <div class="step"><span class="step-num">05</span>Sync Registry</div>
    </div>

    <footer>Spark Studio | Optimized for Linux + NVIDIA DGX Spark (Grace Blackwell)</footer>
</div>

<script>
    const canvas = document.getElementById('infographicCanvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('canvas-container');

    function resize() {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        draw();
    }

    function draw() {
        const w = canvas.width;
        const h = canvas.height;
        ctx.fillStyle = '#020617';
        ctx.fillRect(0, 0, w, h);

        const centerX = w / 2;
        const centerY = h / 2;

        const nodes = [
            { x: centerX - 200, y: centerY - 100, label: 'vLLM',     color: '#38bdf8' },
            { x: centerX + 200, y: centerY - 100, label: 'SGLang',   color: '#818cf8' },
            { x: centerX - 200, y: centerY + 100, label: 'llama.cpp',color: '#fb7185' },
            { x: centerX + 200, y: centerY + 100, label: 'WebGPU',   color: '#34d399' }
        ];

        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        nodes.forEach(node => {
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(node.x, node.y);
            ctx.stroke();
        });
        ctx.setLineDash([]);

        const gradient = ctx.createRadialGradient(centerX, centerY, 10, centerX, centerY, 80);
        gradient.addColorStop(0, '#38bdf8');
        gradient.addColorStop(1, '#1e293b');
        ctx.beginPath();
        ctx.arc(centerX, centerY, 60, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.shadowBlur = 30;
        ctx.shadowColor = '#38bdf8';
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 18px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('SPARK', centerX, centerY - 5);
        ctx.font = '12px sans-serif';
        ctx.fillText('STUDIO', centerX, centerY + 15);

        nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 45, 0, Math.PI * 2);
            ctx.fillStyle = '#1e293b';
            ctx.strokeStyle = node.color;
            ctx.lineWidth = 3;
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 14px sans-serif';
            ctx.fillText(node.label, node.x, node.y + 5);
        });

        ctx.beginPath();
        ctx.arc(centerX, centerY - 130, 35, 0, Math.PI * 2);
        ctx.strokeStyle = '#f59e0b';
        ctx.setLineDash([2, 2]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#f59e0b';
        ctx.font = 'italic 12px sans-serif';
        ctx.fillText('AI Agents', centerX, centerY - 125);
    }

    window.addEventListener('resize', resize);
    resize();
<\/script>
</body>
</html>`,
    language: 'html',
    theme: 'spark',
    automaticLayout: true,
    fontSize: 13,
    minimap: { enabled: false },
  });
  applyCanvasSplit($('#canvasSplit').value);
  runCanvasPreview();
});
$('#canvasLang').addEventListener('change', (e) => monacoEditor && monaco.editor.setModelLanguage(monacoEditor.getModel(), e.target.value));
$('#canvasCopy').addEventListener('click', () => monacoEditor && navigator.clipboard.writeText(monacoEditor.getValue()));
$('#canvasRun').addEventListener('click', runCanvasPreview);
$('#canvasStop').addEventListener('click', stopCanvasPreview);
$('#canvasPopout').addEventListener('click', popoutCanvasPreview);
$('#canvasClear').addEventListener('click', () => monacoEditor && monacoEditor.setValue(''));
$('#canvasSplit').addEventListener('input', (e) => applyCanvasSplit(e.target.value));
$('#canvasPageMode').addEventListener('click', () => {
  const btn = $('#canvasPageMode');
  // Cycle: auto → forced-page → forced-game → auto
  if (canvasPageModeForced === null) {
    canvasPageModeForced = true;
    btn.classList.add('active');
    btn.title = 'Page mode ON (click to switch to game mode)';
  } else if (canvasPageModeForced === true) {
    canvasPageModeForced = false;
    btn.classList.add('active');
    btn.style.opacity = '0.5';
    btn.title = 'Game mode forced (click to reset to auto-detect)';
  } else {
    canvasPageModeForced = null;
    btn.classList.remove('active');
    btn.style.opacity = '';
    btn.title = 'Auto-detect page vs game mode (click to force page mode)';
  }
  if (canvasPreviewDoc) runCanvasPreview();
});
window.addEventListener('message', (ev) => {
  if (ev.data?.source !== 'spark-canvas-preview') return;
  $('#canvasStatus').textContent = ev.data.message || ev.data.kind || 'Preview event';
});

function updateChatControls() {
  $('#chatTempValue').textContent = Number($('#chatTemp').value).toFixed(2);
  $('#chatMaxTokensValue').textContent = String(Number($('#chatMaxTokens').value));
  const target = $('#chatTarget').textContent || 'No engine active';
  $('#chatSettingsStatus').textContent = `${target} · temperature ${Number($('#chatTemp').value).toFixed(2)} · max tokens ${Number($('#chatMaxTokens').value)}`;
}
$('#chatTemp').addEventListener('input', updateChatControls);
$('#chatMaxTokens').addEventListener('input', updateChatControls);
updateChatControls();

async function syncChatTarget() {
  const active = await api('/active').catch(() => null);
  if (active) {
    const served = await api('/models/served').catch(() => ({}));
    // Reflect the engine's real context in the max-tokens slider immediately —
    // not only at first send — so the default is never the 4096 markup value.
    if (served.max_model_len) _applyMaxModelLen('chatMaxTokens', 'chatMaxTokensValue', served.max_model_len);
    const ready = active.ready && served.model;
    const badgeClass = ready ? 'ok' : 'no';
    const badgeLabel = ready ? active.engine : `${active.engine} · loading`;
    const modelBit = served.model ? ` · model <code>${escapeHtml(served.model)}</code>` : '';
    const note = ready ? '' : ' <span class="muted">(engine still starting — wait for the model to load)</span>';
    $('#chatTarget').innerHTML = `Target: <span class="badge ${badgeClass}">${badgeLabel}</span> <code>${escapeHtml(active.url)}</code>${modelBit}${note}`;
  } else {
    $('#chatTarget').textContent = 'No engine active — start one from the vLLM / SGLang / llama.cpp tab.';
  }
  updateChatControls();
}

// ---------- Attachments (shared by Chat & WebGPU tabs) ----------
async function fileToContent(file) {
  if (file.type.startsWith('image/')) {
    const dataUrl = await readAsDataURL(file);
    return { type: 'image_url', image_url: { url: dataUrl }, _label: file.name, _kind: 'image' };
  }
  if (file.type.startsWith('video/')) {
    // Models that don't speak video natively still get a usable thumbnail.
    const dataUrl = await readAsDataURL(file);
    const frameUrl = await videoFirstFrame(dataUrl).catch(() => null);
    return { type: 'image_url', image_url: { url: frameUrl || dataUrl }, _label: file.name, _kind: 'video', _videoUrl: dataUrl };
  }
  if (/^text\//.test(file.type) || /\.(json|ya?ml|md|txt|py|js|ts|csv)$/i.test(file.name)) {
    const text = await file.text();
    return { type: 'text', text: `\n\n---\nAttached file: ${file.name}\n\`\`\`\n${text.slice(0, 50000)}\n\`\`\`\n`, _label: file.name, _kind: 'file' };
  }
  if (/\.(pdf|xlsx|tsv)$/i.test(file.name)) {
    return extractStructuredAttachment(file);
  }
  return { type: 'text', text: `\n\n[attached: ${file.name} (${(file.size/1024).toFixed(1)} KB) — content not extracted]`, _label: file.name, _kind: 'file' };
}
async function extractStructuredAttachment(file) {
  const body = new FormData();
  body.append('file', file);
  const r = await fetch('/api/attachments/extract', { method: 'POST', body });
  if (!r.ok) throw new Error(await r.text());
  const payload = await r.json();
  return {
    type: 'text',
    text: `\n\n---\nAttached file: ${file.name}\n\`\`\`\n${payload.text}\n\`\`\`\n`,
    _label: file.name,
    _kind: 'file',
  };
}
function readAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}
function videoFirstFrame(dataUrl) {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video');
    v.muted = true; v.playsInline = true; v.crossOrigin = 'anonymous';
    v.src = dataUrl;
    v.addEventListener('loadeddata', () => {
      try {
        const c = document.createElement('canvas');
        c.width = v.videoWidth; c.height = v.videoHeight;
        c.getContext('2d').drawImage(v, 0, 0);
        resolve(c.toDataURL('image/jpeg', 0.85));
      } catch (e) { reject(e); }
    });
    v.addEventListener('error', () => reject(new Error('video decode failed')));
  });
}
function bindAttachmentUI({ btnId, inputId, listId, store }) {
  const btn = document.getElementById(btnId);
  const input = document.getElementById(inputId);
  const list = document.getElementById(listId);
  btn.addEventListener('click', () => input.click());
  input.addEventListener('change', async () => {
    for (const f of input.files) {
      try {
        const part = await fileToContent(f);
        store.push(part);
      } catch (e) { toast(`Attach failed: ${e.message}`, 'danger'); }
    }
    input.value = '';
    renderAttachments(list, store);
  });
}
function renderAttachments(list, store) {
  list.innerHTML = '';
  store.forEach((p, i) => {
    const tile = document.createElement('div');
    tile.className = 'attachment';
    if (p._kind === 'image') tile.innerHTML = `<img src="${p.image_url.url}"/><span>${escapeHtml(p._label)}</span>`;
    else if (p._kind === 'video') tile.innerHTML = `<img src="${p.image_url.url}"/><span>${escapeHtml(p._label)} <i class="fa-solid fa-film"></i></span>`;
    else tile.innerHTML = `<i class="fa-solid fa-file-lines"></i><span>${escapeHtml(p._label)}</span>`;
    const x = document.createElement('span');
    x.className = 'remove'; x.textContent = '×';
    x.addEventListener('click', () => { store.splice(i, 1); renderAttachments(list, store); });
    tile.appendChild(x);
    list.appendChild(tile);
  });
}
function buildUserMessage(text, attachments) {
  if (!attachments.length) return { role: 'user', content: text };
  const parts = [];
  if (text) parts.push({ type: 'text', text });
  for (const a of attachments) {
    const { _label, _kind, _videoUrl, ...rest } = a;
    parts.push(rest);
  }
  return { role: 'user', content: parts };
}
function createAvatar(role) {
  return h('div', { class: `msg-avatar ${role}` },
    h('i', { class: role === 'user' ? 'fa-solid fa-user' : 'fa-solid fa-bolt' }));
}

function wrapMessageRow(role, bubble) {
  const row = h('div', { class: `msg-row ${role}` });
  if (role === 'user') { row.appendChild(bubble); row.appendChild(createAvatar(role)); }
  else { row.appendChild(createAvatar(role)); row.appendChild(bubble); }
  return row;
}

function appendAssistantMessage(msgs) {
  const bubble = h('div', { class: 'chat-msg assistant' }, '');
  msgs.appendChild(wrapMessageRow('assistant', bubble));
  return bubble;
}

// Coalesces rapid streaming deltas into one render + scroll per animation frame.
function makeStreamRenderer(node, msgs) {
  let pending = null;
  let rafId = null;
  return (acc, reasoning) => {
    pending = { acc, reasoning };
    if (rafId != null) return;
    rafId = requestAnimationFrame(() => {
      rafId = null;
      // The final (non-streaming) render usually lands between the last delta
      // and this frame — the last token and [DONE] arrive within the same
      // ~16ms window. Painting now would overwrite the finished message
      // (charts, doc cards) with its streaming code-block form + cursor.
      if (node._streamFinalized) return;
      renderChatContent(node, pending.acc, { streaming: true, reasoning: pending.reasoning });
      msgs.scrollTop = msgs.scrollHeight;
    });
  };
}

function renderUserBubble(text, attachments) {
  const node = h('div', { class: 'chat-msg user' });
  if (text) node.appendChild(document.createTextNode(text));
  for (const a of attachments) {
    if (a._kind === 'image') node.appendChild(h('img', { src: a.image_url.url, alt: a._label }));
    else if (a._kind === 'video') {
      const v = document.createElement('video');
      v.src = a._videoUrl; v.controls = true; node.appendChild(v);
    } else node.appendChild(h('div', { class: 'muted', style: 'font-size:11px;margin-top:6px' }, `📎 ${a._label}`));
  }
  return wrapMessageRow('user', node);
}

function estimateTokens(text) {
  const s = typeof text === 'string' ? text : JSON.stringify(text || '');
  return Math.max(1, Math.round(s.length / 4));
}

function estimateInputTokens(messages) {
  return messages.reduce((sum, msg) => {
    const content = msg?.content;
    if (typeof content === 'string') return sum + estimateTokens(content);
    if (Array.isArray(content)) {
      return sum + content.reduce((inner, part) => {
        if (part?.type === 'text') return inner + estimateTokens(part.text || '');
        if (part?.type === 'image_url') return inner + 85;
        return inner;
      }, 0);
    }
    return sum;
  }, 0);
}

function formatMetrics(m) {
  if (!m) return '';
  const tokps = m.tokps != null ? `${m.tokps.toFixed(1)} tok/s` : 'tok/s —';
  const out = `out ${m.tokensOut ?? '—'}`;
  const inn = `in ${m.tokensIn ?? '—'}`;
  const ttft = `ttft ${m.ttftMs != null ? `${Math.round(m.ttftMs)} ms` : '—'}`;
  const total = `total ${m.totalMs != null ? `${(m.totalMs / 1000).toFixed(2)} s` : '—'}`;
  return `${tokps} · ${out} · ${inn} · ${ttft} · ${total}`;
}

function setAssistantMetrics(node, metrics) {
  let meta = node.querySelector('.metrics');
  if (!meta) {
    meta = h('div', { class: 'metrics' });
    node.appendChild(meta);
  }
  meta.textContent = formatMetrics(metrics);
}

// ---------- chart + markdown rendering ------------------------------------

// Local models rarely emit strict JSON — they produce JS object literals with
// single quotes, trailing commas, comments, unquoted keys, or a
// `const config = {...}` wrapper. Repair instead of eval'ing model output.
function _repairJson(src) {
  let out = '';
  let i = 0;
  let inStr = false;
  let quote = '';
  const n = src.length;
  while (i < n) {
    const c = src[i];
    if (inStr) {
      if (c === '\\') { out += c + (src[i + 1] || ''); i += 2; continue; }
      if (c === quote) { inStr = false; out += '"'; i++; continue; }
      if (c === '"') { out += '\\"'; i++; continue; }
      out += c; i++; continue;
    }
    if (c === '"' || c === "'") { inStr = true; quote = c; out += '"'; i++; continue; }
    if (c === '/' && src[i + 1] === '/') { while (i < n && src[i] !== '\n') i++; continue; }
    if (c === '/' && src[i + 1] === '*') {
      i += 2;
      while (i < n && !(src[i] === '*' && src[i + 1] === '/')) i++;
      i += 2; continue;
    }
    out += c; i++;
  }
  return out
    .replace(/,\s*([}\]])/g, '$1')                        // trailing commas
    .replace(/([{,]\s*)([A-Za-z_$][\w$]*)(\s*:)/g, '$1"$2"$3'); // unquoted keys
}

function _parseChartConfig(code) {
  const trimmed = (code || '').trim();
  const candidates = [trimmed];
  // Model wrapped the object in prose or `const config = {...};` — take the
  // outermost braces.
  const start = trimmed.indexOf('{');
  const end = trimmed.lastIndexOf('}');
  if (start >= 0 && end > start && (start > 0 || end < trimmed.length - 1)) {
    candidates.push(trimmed.slice(start, end + 1));
  }
  for (const cand of candidates) {
    for (const txt of [cand, _repairJson(cand)]) {
      try {
        const obj = JSON.parse(txt);
        if (obj && typeof obj === 'object' && !Array.isArray(obj)) return obj;
      } catch { /* try next */ }
    }
  }
  return null;
}

function _looksLikeChartConfig(code) {
  const obj = _parseChartConfig(code);
  return !!(obj && 'type' in obj && 'data' in obj);
}

function _appendChartBlock(parent, code) {
  const wrapper = document.createElement('div');
  wrapper.className = 'chat-chart-wrapper';
  const canvas = document.createElement('canvas');
  wrapper.appendChild(canvas);
  parent.appendChild(wrapper);
  try {
    const config = _parseChartConfig(code);
    if (!config) throw new Error('config is not valid JSON (even after repair)');
    if (!config.type || !config.data) throw new Error('config needs both "type" and "data"');
    // merge in readable defaults
    config.options = Object.assign({
      responsive: true,
      plugins: { legend: { labels: { color: '#c9d6e8' } } },
      scales: config.type !== 'pie' && config.type !== 'doughnut' ? {
        x: { ticks: { color: '#8fa3bf' }, grid: { color: 'rgba(255,255,255,0.07)' } },
        y: { ticks: { color: '#8fa3bf' }, grid: { color: 'rgba(255,255,255,0.07)' } },
      } : undefined,
    }, config.options || {});
    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
    new Chart(canvas, config);
  } catch (e) {
    const err = document.createElement('pre');
    err.className = 'chat-chart-error';
    err.textContent = `Chart error: ${e.message}\n\n${code}`;
    wrapper.appendChild(err);
  }
}

const _DOC_EXPORT_META = {
  docx: {
    icon: 'fa-file-word', label: 'Word document', endpoint: '/api/export/docx',
    summary: (spec) => `${(spec.sections || []).length} section${(spec.sections || []).length === 1 ? '' : 's'}`,
  },
  xlsx: {
    icon: 'fa-file-excel', label: 'Excel workbook', endpoint: '/api/export/xlsx',
    summary: (spec) => {
      const sheets = spec.sheets || [];
      const rows = sheets.reduce((sum, s) => sum + (s.rows || []).length, 0);
      return `${sheets.length} sheet${sheets.length === 1 ? '' : 's'} · ${rows} row${rows === 1 ? '' : 's'}`;
    },
  },
};

function _appendDocBlock(parent, kind, code) {
  const meta = _DOC_EXPORT_META[kind];
  const wrapper = h('div', { class: 'chat-doc-card' });
  try {
    const spec = JSON.parse(code.trim());
    const title = spec.title || (spec.sheets && spec.sheets[0]?.name) || meta.label;
    wrapper.appendChild(h('div', { class: 'chat-doc-icon' }, h('i', { class: `fa-solid ${meta.icon}` })));
    const info = h('div', { class: 'chat-doc-info' },
      h('div', { class: 'chat-doc-title' }, title),
      h('div', { class: 'chat-doc-meta' }, meta.summary(spec)));
    wrapper.appendChild(info);
    const btn = h('button', { class: 'btn primary chat-doc-download' },
      h('i', { class: 'fa-solid fa-download' }), ' Download');
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      const original = btn.innerHTML;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating…';
      try {
        const r = await fetch(meta.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(spec),
        });
        if (!r.ok) throw new Error(await r.text());
        const blob = await r.blob();
        const disposition = r.headers.get('Content-Disposition') || '';
        const match = /filename="?([^"]+)"?/.exec(disposition);
        const filename = match ? match[1] : `export.${kind}`;
        const url = URL.createObjectURL(blob);
        const a = h('a', { href: url, download: filename });
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      } catch (e) {
        toast(`Export failed: ${e.message}`, 'danger');
      } finally {
        btn.disabled = false;
        btn.innerHTML = original;
      }
    });
    wrapper.appendChild(btn);
  } catch (e) {
    wrapper.appendChild(h('pre', { class: 'chat-chart-error' }, `${meta.label} error: ${e.message}\n\n${code}`));
  }
  parent.appendChild(wrapper);
}

function _appendCodeBlock(parent, lang, code) {
  const wrapper = h('div', { class: 'chat-code-wrapper' });
  const copyBtn = h('button', { class: 'chat-code-copy', title: 'Copy code' }, h('i', { class: 'fa-solid fa-copy' }));
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(code);
      copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
      setTimeout(() => { copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>'; }, 1500);
    } catch { /* clipboard unavailable */ }
  });
  wrapper.appendChild(h('div', { class: 'chat-code-header' },
    h('span', { class: 'chat-code-lang' }, lang || 'text'), copyBtn));
  const pre = document.createElement('pre');
  pre.className = 'chat-code-block';
  const codeEl = document.createElement('code');
  if (lang) codeEl.className = `language-${lang}`;
  codeEl.textContent = code;
  pre.appendChild(codeEl);
  wrapper.appendChild(pre);
  parent.appendChild(wrapper);
  if (window.hljs) {
    try { window.hljs.highlightElement(codeEl); } catch { /* unknown language, leave plain */ }
  }
}

// Some models emit chain-of-thought as inline <think>…</think> tags instead
// of the reasoning_content field — route those to the thinking block too.
// An unclosed trailing <think> (mid-stream) counts as all-reasoning.
function _splitThinkTags(text) {
  let reasoning = '';
  let content = '';
  let last = 0;
  const re = /<think>([\s\S]*?)(?:<\/think>|$)/gi;
  let m;
  while ((m = re.exec(text)) !== null) {
    content += text.slice(last, m.index);
    reasoning += (reasoning ? '\n' : '') + m[1];
    last = m.index + m[0].length;
  }
  content += text.slice(last);
  return { content, reasoning };
}

function _appendThinkingBlock(parent, text, open) {
  const details = h('details', { class: 'chat-thinking' });
  if (open) details.open = true;
  details.appendChild(h('summary', {}, open ? '💭 Thinking…' : '💭 Thought process'));
  const pre = document.createElement('pre');
  pre.textContent = text.trim();
  details.appendChild(pre);
  parent.appendChild(details);
}

function renderChatContent(node, text, { streaming = false, reasoning = '' } = {}) {
  // A final render must win over any still-queued streaming re-render (see
  // makeStreamRenderer): flag the node so a stale rAF callback skips painting.
  node._streamFinalized = !streaming;
  const metrics = node.querySelector('.metrics');
  const split = _splitThinkTags(text || '');
  text = split.content;
  const thinking = [reasoning, split.reasoning].filter((s) => s && s.trim()).join('\n');
  node.innerHTML = '';
  if (thinking) {
    // Expanded while the model is still thinking with no answer yet;
    // collapsed once real content lands so the answer takes the stage.
    _appendThinkingBlock(node, thinking, streaming && !text.trim());
  }

  const codeBlockRe = /```([\w-]*)\r?\n?([\s\S]*?)```/g;
  let lastIdx = 0;
  let m;

  const appendText = (str) => {
    if (!str) return;
    const div = document.createElement('div');
    div.className = 'chat-text-block';
    if (window.marked && window.DOMPurify) {
      div.innerHTML = window.DOMPurify.sanitize(window.marked.parse(str, { gfm: true, breaks: true }));
    } else {
      div.textContent = str;
    }
    node.appendChild(div);
  };

  while ((m = codeBlockRe.exec(text)) !== null) {
    if (m.index > lastIdx) appendText(text.slice(lastIdx, m.index));
    const lang = (m[1] || '').toLowerCase();
    const code = m[2];
    if (lang === 'chartjs' || lang === 'chart' || (lang === 'json' && _looksLikeChartConfig(code))) {
      // While streaming, later deltas re-render everything — drawing the
      // chart each time thrashes Chart.js, so show it as code until final.
      if (streaming) _appendCodeBlock(node, 'chartjs', code);
      else _appendChartBlock(node, code);
    } else if (lang === 'docx' || lang === 'xlsx') {
      _appendDocBlock(node, lang, code);
    } else {
      _appendCodeBlock(node, lang, code);
    }
    lastIdx = m.index + m[0].length;
  }

  if (lastIdx < text.length) appendText(text.slice(lastIdx));
  if (streaming) node.appendChild(h('span', { class: 'stream-cursor' }, '▋'));
  if (metrics) node.appendChild(metrics);
}

async function currentServerModelInfo() {
  const served = await api('/models/served').catch(() => ({}));
  return {
    model: served.model || 'local',
    url: served.url || '',
    maxModelLen: served.max_model_len || null,
  };
}

// Update a max_tokens range slider to reflect the engine's actual context window.
function _applyMaxModelLen(sliderId, labelId, maxModelLen) {
  const slider = $('#' + sliderId);
  if (!slider || !maxModelLen) return;
  // Track the engine context we've applied in a dataset flag — the old
  // "did slider.max change?" heuristic silently failed whenever the engine's
  // context happened to equal the HTML max attribute (131072 — very common),
  // leaving the value stuck at the 4096 markup default.
  const seen = slider.dataset.engineCtx;
  slider.max = maxModelLen;
  if (seen !== String(maxModelLen)) {
    // New engine (or first detection): default to the FULL context. The
    // server auto-fits to actual prompt size, so full is always safe.
    slider.dataset.engineCtx = String(maxModelLen);
    slider.value = maxModelLen;
    if (labelId) $('#' + labelId).textContent = String(maxModelLen);
  } else if (Number(slider.value) > maxModelLen) {
    slider.value = maxModelLen;
    if (labelId) $('#' + labelId).textContent = String(maxModelLen);
  }
}

function withModelIdentity(messages, label) {
  const system = {
    role: 'system',
    content: `You are the model currently running as ${label}. If asked which model you are, answer with exactly that running model identity unless the user explicitly asks you to roleplay otherwise.

When the user's message contains a block starting with "[Web search results — use these for an accurate, up-to-date answer:]", real-time web search has already been done for you and the sources include full page excerpts, not just links. These are current, live results — treat them as authoritative and more up to date than your training data. Write a polished, magazine-quality answer from them:
- Lead with the direct answer or the top takeaways — never with commentary about the search results themselves.
- Pull concrete facts, names, numbers, and dates out of the page excerpts; synthesize across sources instead of summarizing them one by one.
- Structure for scanning: short bold headers or a numbered/bulleted list when the content is enumerable (headlines, rankings, steps), flowing prose when it isn't.
- Cite as you go with inline markdown links on the source name, e.g. ([NBC News](https://www.nbcnews.com/...)); every distinct claim should be attributable to one of the provided sources.
- Never mention the mechanics: do NOT say "based on the provided search results", do NOT say you cannot browse the web, do NOT recommend the user visit sites or search elsewhere, and do NOT pad with disclaimers about recency. If a specific detail genuinely isn't in the excerpts, note that in one short sentence and answer with everything they do support.

When the user asks for a chart, graph, or data visualization, output a Chart.js v4 configuration as JSON inside a \`\`\`chartjs\`\`\` code block. The client renders it automatically. Example:
\`\`\`chartjs
{"type":"bar","data":{"labels":["Jan","Feb","Mar"],"datasets":[{"label":"Sales","data":[42,67,53],"backgroundColor":"rgba(118,199,255,0.6)"}]}}
\`\`\`
Supported types: bar, line, pie, doughnut, radar, polarArea, scatter, bubble.
The block must be strict JSON: double-quoted keys and strings, no comments, no trailing commas, and no JavaScript functions/callbacks — the client parses it with JSON.parse, it does not execute code.

When the user asks you to create, generate, or draft a Word document, output a JSON spec inside a \`\`\`docx\`\`\` code block. The client renders a download card and builds a real .docx file from it. Schema:
\`\`\`docx
{"title":"Q1 Report","sections":[
  {"heading":"Summary","level":1},
  {"paragraph":"Revenue grew 12% quarter over quarter."},
  {"bullets":["Point one","Point two"]},
  {"table":{"headers":["Metric","Value"],"rows":[["Revenue","$1.2M"],["Growth","12%"]]}}
]}
\`\`\`
Section types: heading (level 1-3), paragraph, bullets (array of strings), table (headers + rows).

When the user asks you to create, generate, or draft an Excel spreadsheet, output a JSON spec inside a \`\`\`xlsx\`\`\` code block. The client renders a download card and builds a real .xlsx file from it. Schema:
\`\`\`xlsx
{"sheets":[{"name":"Budget","headers":["Item","Cost"],"rows":[["Rent",1200],["Utilities",150]]}]}
\`\`\`
A workbook can have multiple sheets. Use numbers (not strings) for numeric cells where appropriate.`,
  };
  return [system, ...messages];
}

function contentPartsToText(value) {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  return value.map((part) => {
    if (typeof part === 'string') return part;
    if (!part || typeof part !== 'object') return '';
    if (typeof part.text === 'string') return part.text;
    if (typeof part.content === 'string') return part.content;
    if (typeof part.reasoning_content === 'string') return part.reasoning_content;
    return '';
  }).join('');
}

// Split a streamed choice into user-facing content vs the model's private
// chain-of-thought (vLLM reasoning parsers emit `reasoning_content` deltas).
// Mixing them into one string is what used to dump raw "I will formulate the
// response…" thinking into the chat bubble.
function extractChoiceParts(choice) {
  if (!choice || typeof choice !== 'object') return { content: '', reasoning: '' };
  const delta = choice.delta || {};
  const message = choice.message || {};
  const content =
    contentPartsToText(delta.content) ||
    contentPartsToText(message.content) ||
    (typeof delta.content === 'string' ? delta.content : '') ||
    (typeof message.content === 'string' ? message.content : '');
  const reasoning =
    contentPartsToText(delta.reasoning_content) ||
    contentPartsToText(message.reasoning_content) ||
    (typeof delta.reasoning_content === 'string' ? delta.reasoning_content : '') ||
    (typeof message.reasoning_content === 'string' ? message.reasoning_content : '') ||
    (typeof delta.reasoning === 'string' ? delta.reasoning : '') ||
    (typeof message.reasoning === 'string' ? message.reasoning : '');
  return { content, reasoning };
}

// Stream OpenAI-format SSE from /api/chat into `onDelta(text)` callbacks.
// Returns the full accumulated text and timing/usage metrics once the stream closes.
async function streamChat(messages, onDelta, opts = {}) {
  const startedAt = performance.now();
  let firstDeltaAt = null;
  let usage = null;
  const finalize = () => {
    const finishedAt = performance.now();
    const tokensOut = usage?.completion_tokens ?? estimateTokens(accReasoning + acc);
    const tokensIn = usage?.prompt_tokens ?? estimateInputTokens(messages);
    const totalMs = finishedAt - startedAt;
    const genMs = Math.max(1, finishedAt - (firstDeltaAt ?? finishedAt));
    return {
      text: acc,
      reasoning: accReasoning,
      metrics: {
        tokensOut,
        tokensIn,
        ttftMs: firstDeltaAt != null ? firstDeltaAt - startedAt : null,
        totalMs,
        tokps: tokensOut / (genMs / 1000),
      },
    };
  };
  const r = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: opts.signal,
    body: JSON.stringify({
      messages,
      stream: true,
      model: opts.model,
      temperature: opts.temperature,
      max_tokens: opts.maxTokens,
      stream_options: { include_usage: true },
    }),
  });
  if (!r.ok) {
    const txt = await r.text();
    let msg = txt;
    try { msg = JSON.parse(txt).detail || txt; } catch { /* keep raw */ }
    throw new Error(msg);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let acc = '';
  let accReasoning = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line || !line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') return finalize();
      try {
        const obj = JSON.parse(payload);
        if (obj.error) {
          const detail = obj.error.message || JSON.stringify(obj.error);
          throw new Error(`engine error${obj.error.status ? ' ' + obj.error.status : ''}: ${detail}`);
        }
        if (obj.usage) usage = obj.usage;
        const parts = extractChoiceParts(obj.choices?.[0]);
        if (parts.content || parts.reasoning) {
          acc += parts.content;
          accReasoning += parts.reasoning;
          if (firstDeltaAt == null) firstDeltaAt = performance.now();
          onDelta(acc, accReasoning);
        }
      } catch (err) {
        // Re-throw upstream errors; ignore partial-frame parse failures.
        if (err instanceof Error && err.message.startsWith('engine error')) throw err;
      }
    }
  }
  return finalize();
}

// ---------- Chat (against active engine) ----------
$('#chatSend').addEventListener('click', sendChat);
$('#chatStop').addEventListener('click', stopChat);
$('#chatInput').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) sendChat(); });

const chatAttachments = [];
bindAttachmentUI({ btnId: 'chatAttachBtn', inputId: 'chatAttachInput', listId: 'chatAttachments', store: chatAttachments });

const chatHistory = [];
let chatStream = null;
function setChatBusy(busy) {
  $('#chatSend').disabled = busy;
  $('#chatStop').disabled = !busy;
}
function stopChat() {
  if (chatStream) chatStream.abort();
}
$('#chatClear').addEventListener('click', () => {
  if (chatStream) return;
  chatHistory.length = 0;
  $('#chatMessages').innerHTML = '';
  $('#chatSettingsStatus').textContent = 'Chat history cleared.';
});
async function sendChat() {
  if (chatStream) return;
  const input = $('#chatInput');
  const text = input.value.trim();
  if (!text && !chatAttachments.length) return;
  input.value = '';
  const userMsg = buildUserMessage(text, chatAttachments);
  chatHistory.push(userMsg);
  const msgs = $('#chatMessages');
  msgs.appendChild(renderUserBubble(text, chatAttachments));
  const assistant = appendAssistantMessage(msgs);
  msgs.scrollTop = msgs.scrollHeight;
  chatAttachments.length = 0;
  renderAttachments($('#chatAttachments'), chatAttachments);

  try {
    chatStream = new AbortController();
    setChatBusy(true);
    const modelInfo = await currentServerModelInfo();
    if (modelInfo.maxModelLen) _applyMaxModelLen('chatMaxTokens', 'chatMaxTokensValue', modelInfo.maxModelLen);
    const messages = withModelIdentity(chatHistory, modelInfo.model);
    const renderLive = makeStreamRenderer(assistant, msgs);
    const res = await streamChat(messages, renderLive, {
      temperature: Number($('#chatTemp').value),
      maxTokens: Number($('#chatMaxTokens').value),
      signal: chatStream.signal,
    });
    if (!res.text && !res.reasoning) { assistant.textContent = '(empty response)'; } else { renderChatContent(assistant, res.text || '*(the model spent its whole budget thinking — raise max tokens or ask again)*', { reasoning: res.reasoning }); }
    chatHistory.push({ role: 'assistant', content: res.text });
    if (res.metrics?.tokps) recordTps(res.metrics.tokps);
    setAssistantMetrics(assistant, res.metrics);
    $('#chatSettingsStatus').textContent = `${$('#chatTarget').textContent || 'Chat'} · ${formatMetrics(res.metrics)} · temp ${Number($('#chatTemp').value).toFixed(2)} · max ${Number($('#chatMaxTokens').value)}`;
    maybeSendToCanvas(res.text);
  } catch (e) {
    if (e.name === 'AbortError') {
      if (assistant.textContent) renderChatContent(assistant, assistant.textContent);
      else assistant.textContent = '(stopped)';
      $('#chatSettingsStatus').textContent = `${$('#chatTarget').textContent || 'Chat'} · stopped`;
    } else {
      assistant.textContent = `Error: ${e.message}`;
    }
  } finally {
    chatStream = null;
    setChatBusy(false);
  }
}

function maybeSendToCanvas(text) {
  const blocks = [...text.matchAll(/```(\w+)?\n([\s\S]*?)```/g)];
  if (!blocks.length || !monacoEditor) return;
  const preferred = [...blocks].reverse().find((m) => ['html', 'javascript', 'typescript'].includes((m[1] || '').toLowerCase())) || blocks[0];
  const lang = (preferred[1] || $('#canvasLang').value).toLowerCase();
  if (['python','javascript','typescript','json','yaml','markdown','html','cpp','rust','shell'].includes(lang)) {
    $('#canvasLang').value = lang;
    monaco.editor.setModelLanguage(monacoEditor.getModel(), lang);
  }
  monacoEditor.setValue(preferred[2]);
  if (['html', 'javascript', 'typescript'].includes(lang)) runCanvasPreview();
}

// ---------- benchmarks ---------------------------------------------------
// ---------- Tool Eval Bench ------------------------------------------------
const TOOLEVAL_CATS = {
  selection: 'Tool selection', arguments: 'Arguments', restraint: 'Restraint',
  multi_turn: 'Uses results', json_output: 'Strict JSON',
};
let toolevalPoll = null;

function toolevalBadge(score) {
  const cls = score >= 80 ? 'ok' : score >= 50 ? 'warn' : 'no';
  return `<span class="badge ${cls}">${score}%</span>`;
}

function renderToolEvalStatus(s) {
  const busy = s.running;
  $('#toolevalGo').disabled = busy;
  $('#toolevalProgress').textContent = busy ? `running ${s.done}/${s.total}…` : '';
  if (s.error) {
    $('#toolevalScore').innerHTML = `<div class="muted" style="color:var(--danger)">Eval failed: ${escapeHtml(s.error)}</div>`;
    return;
  }
  if (s.score == null && !busy) return;
  const cats = Object.entries(s.category_scores || {})
    .map(([c, v]) => `<span style="margin-right:10px">${TOOLEVAL_CATS[c] || c} ${toolevalBadge(v)}</span>`).join('');
  $('#toolevalScore').innerHTML = s.score == null ? '' :
    `<div style="margin:10px 0;font-size:15px"><b>${escapeHtml(s.model || '')}</b> — overall ${toolevalBadge(s.score)} ${cats}
     ${s.tools_unsupported ? '<div class="muted" style="color:var(--warn);margin-top:4px"><i class="fa-solid fa-triangle-exclamation"></i> The engine rejected the <code>tools</code> parameter — this server/model likely wasn\'t launched with tool-calling enabled (e.g. vLLM needs <code>--enable-auto-tool-choice --tool-call-parser …</code>).</div>' : ''}</div>`;
  const rows = (s.cases || []).map((c) =>
    `<tr><td>${c.ok ? '<span class="badge ok">✓</span>' : '<span class="badge no">✗</span>'}</td>
     <td class="mono">${escapeHtml(c.id)}</td><td>${TOOLEVAL_CATS[c.category] || c.category}</td>
     <td class="muted" style="font-size:11px;overflow-wrap:anywhere">${escapeHtml(c.ok ? (c.observed || '') : [c.detail, c.observed].filter(Boolean).join(' — '))}</td></tr>`).join('');
  $('#toolevalCases').innerHTML = rows ? `<table><tr><th></th><th>Case</th><th>Category</th><th>Detail</th></tr>${rows}</table>` : '';
  if (s.report_path && !busy) {
    $('#toolevalCases').insertAdjacentHTML('beforeend',
      `<div class="muted" style="margin-top:6px;font-size:11px"><i class="fa-solid fa-file-lines"></i> Full report saved to <code>${escapeHtml(s.report_path)}</code> (+ .json)</div>`);
  }
}

async function refreshToolEval() {
  try {
    renderToolEvalStatus(await api('/tooleval/status'));
  } catch { /* endpoint unavailable */ }
  const hist = await api('/tooleval/history?limit=10').catch(() => []);
  $('#toolevalHistory').innerHTML = hist.length
    ? `<h4 style="margin:14px 0 6px"><i class="fa-solid fa-clock-rotate-left"></i> History</h4>
       <table><tr><th>When</th><th>Model</th><th>Score</th></tr>${hist.map((h) =>
         `<tr><td>${new Date(h.created_at * 1000).toLocaleString()}</td>
          <td class="mono">${escapeHtml(h.model || '')}</td><td>${toolevalBadge(h.score ?? 0)}</td></tr>`).join('')}</table>`
    : '';
}

$('#toolevalGo').addEventListener('click', async () => {
  $('#toolevalGo').disabled = true;
  $('#toolevalScore').innerHTML = '';
  $('#toolevalCases').innerHTML = '';
  try {
    await api('/tooleval/run', { method: 'POST', body: { run_id: $('#toolevalTarget').value || undefined } });
  } catch (e) {
    $('#toolevalGo').disabled = false;
    toast(`Tool eval failed to start: ${e.message}`, 'danger');
    return;
  }
  if (toolevalPoll) clearInterval(toolevalPoll);
  toolevalPoll = setInterval(async () => {
    const s = await api('/tooleval/status').catch(() => null);
    if (!s) return;
    renderToolEvalStatus(s);
    if (!s.running) {
      clearInterval(toolevalPoll);
      toolevalPoll = null;
      refreshToolEval();
    }
  }, 1000);
});

let benchChart = null;
let benchyStream = null;
async function refreshBenchTab() {
  const runs = await api('/runs').catch(() => []);
  const running = runs.filter((r) => r.status === 'running');
  const opts = running.map((r) => `<option value="${r.id}">${runLabel(r)} · ${r.engine} ${r.url || ''}</option>`).join('') || '<option value="">No running engines</option>';
  $('#benchRun').innerHTML = opts;
  $('#benchyTarget').innerHTML = opts;
  $('#toolevalTarget').innerHTML = opts;
  refreshToolEval();

  // llama-benchy install state
  try {
    const s = await api('/benchy/status');
    $('#benchyMissing').hidden = !!s.installed;
  } catch {}

  // Quick bench prev results
  const prev = await api('/bench').catch(() => []);
  if (!benchChart) {
    benchChart = new Chart($('#benchChart'), {
      type: 'bar',
      data: { labels: [], datasets: [
        { label: 'tokens/s', data: [], backgroundColor: '#76c7ff' },
        { label: 'TTFT ms', data: [], backgroundColor: '#b084ff' },
      ]},
      options: { scales: { y: { ticks: { color: '#8592a6' } }, x: { ticks: { color: '#8592a6' } } }, plugins: { legend: { labels: { color: '#e6edf7' } } } },
    });
  }
  benchChart.data.labels = prev.map((b) => `${(b.run_id || '?').slice(0, 6)}`);
  benchChart.data.datasets[0].data = prev.map((b) => b.tokens_per_sec ?? 0);
  benchChart.data.datasets[1].data = prev.map((b) => b.ttft_ms ?? 0);
  benchChart.update();
  $('#benchTable').innerHTML = `<table><tr><th>When</th><th>Model</th><th>Run</th><th>tok/s</th><th>TTFT ms</th></tr>
    ${prev.map((b) => { const d = b.data_json ? JSON.parse(b.data_json) : {}; return `<tr><td>${new Date(b.created_at * 1000).toLocaleString()}</td><td class="mono">${escapeHtml(d.model || '—')}</td><td class="mono">${b.run_id || ''}</td><td>${(b.tokens_per_sec || 0).toFixed(1)}</td><td>${(b.ttft_ms || 0).toFixed(0)}</td></tr>`; }).join('')}</table>`;

  // llama-benchy history
  const hist = await api('/benchy/list?limit=20').catch(() => []);
  $('#benchyHistory').innerHTML = hist.length ? hist.map(renderBenchyHistoryRow).join('') : '<div class="muted">No llama-benchy runs yet.</div>';
  $$('#benchyHistory [data-bench-id]').forEach((el) => el.addEventListener('click', async () => {
    const row = await api(`/benchy/${el.dataset.benchId}`);
    renderBenchyResult(row.result_json ? JSON.parse(row.result_json) : null);
  }));
  $$('#benchyHistory [data-bench-share]').forEach((el) => el.addEventListener('click', async (ev) => {
    ev.stopPropagation();
    try {
      const { markdown } = await api(`/benchy/${el.dataset.benchShare}/export`);
      await copyText(markdown);
      toast('Benchmark report copied as markdown — paste it anywhere to share');
    } catch (e) { toast(e.message, 'danger'); }
  }));
  const cmpBoxes = $$('#benchyHistory [data-bench-cmp]');
  const updateCmp = () => {
    const checked = cmpBoxes.filter((b) => b.checked);
    $('#benchyCompare').disabled = checked.length !== 2;
  };
  cmpBoxes.forEach((b) => b.addEventListener('click', (ev) => { ev.stopPropagation(); updateCmp(); }));
  updateCmp();
}
function renderBenchyHistoryRow(r) {
  const params = JSON.parse(r.params_json || '{}');
  const hasResult = !!r.result_json;
  return `<div class="run-item" data-bench-id="${r.id}">
    <div class="ri-top">
      <span class="ri-engine">${hasResult ? `<input type="checkbox" data-bench-cmp="${r.id}" title="Select for comparison" style="margin-right:6px" />` : ''}#${r.id} · ${escapeHtml(r.model || '')}</span>
      <span class="ri-id">${r.engine_version ? escapeHtml(r.engine_version) + ' · ' : ''}${new Date(r.created_at * 1000).toLocaleString()}
        ${hasResult ? `<button class="btn" data-bench-share="${r.id}" title="Copy shareable markdown report" style="margin-left:8px;padding:2px 8px">⧉</button>` : ''}</span>
    </div>
    <div class="muted" style="font-size:11px">pp=${(params.pp || []).join(',')} · tg=${(params.tg || []).join(',')} · depth=${(params.depth || []).join(',')} · conc=${(params.concurrency || [1]).join(',')} · runs=${params.runs}</div>
  </div>`;
}
$('#benchyCompare').addEventListener('click', async () => {
  const ids = $$('#benchyHistory [data-bench-cmp]').filter((b) => b.checked).map((b) => b.dataset.benchCmp);
  if (ids.length !== 2) return;
  try {
    const [a, b] = await Promise.all(ids.map((id) => api(`/benchy/${id}`)));
    renderBenchyCompare(a, b);
  } catch (e) { toast(e.message, 'danger'); }
});

// Side-by-side view of two stored benchy runs, matched on test shape
// (prompt/response/context/concurrency). Δ is B relative to A.
function renderBenchyCompare(a, b) {
  const ra = JSON.parse(a.result_json || '{}');
  const rb = JSON.parse(b.result_json || '{}');
  const key = (t) => `${t.prompt_size}|${t.response_size}|${t.context_size}|${t.concurrency}|${t.is_context_prefill_phase ? 1 : 0}`;
  const label = (t) => {
    let s = `pp${t.prompt_size}+tg${t.response_size}`;
    if (t.context_size) s += ` @ d${t.context_size}`;
    if (t.is_context_prefill_phase) s += ' (prefill)';
    return `${s} ×${t.concurrency || 1}`;
  };
  const bByKey = new Map((rb.benchmarks || []).map((t) => [key(t), t]));
  const mean = (t, k) => t?.[k]?.mean ?? null;
  const fmt = (v) => (v == null ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: 1 }));
  const delta = (va, vb) => {
    if (va == null || vb == null || !va) return '';
    const pct = ((vb - va) / va) * 100;
    const cls = pct >= 0 ? 'ok' : 'danger';
    return ` <span class="badge ${cls}" style="font-size:10px">${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%</span>`;
  };
  const head = (r, row) => `#${row.id} ${escapeHtml(r.model || row.model || '')}${row.engine_version ? ` · ${escapeHtml(row.engine_version)}` : ''}`;
  const rows = (ra.benchmarks || []).map((ta) => {
    const tb = bByKey.get(key(ta));
    const cells = ['pp_throughput', 'tg_throughput', 'e2e_ttft'].map((k) => {
      const va = mean(ta, k);
      const vb = mean(tb, k);
      return `<td>${fmt(va)}</td><td>${fmt(vb)}${k === 'e2e_ttft' ? '' : delta(va, vb)}</td>`;
    }).join('');
    return `<tr><td class="mono">${label(ta)}</td>${cells}</tr>`;
  }).join('');
  $('#benchyCharts').hidden = true;
  $('#benchyResult').innerHTML = `
    <h4 style="margin:18px 0 6px"><i class="fa-solid fa-scale-balanced"></i> Compare — A: ${head(ra, a)} vs B: ${head(rb, b)}</h4>
    <table>
      <tr><th>test</th><th>pp t/s A</th><th>pp t/s B</th><th>tg t/s A</th><th>tg t/s B</th><th>ttft ms A</th><th>ttft ms B</th></tr>
      ${rows || '<tr><td colspan="7" class="muted">no comparable tests</td></tr>'}
    </table>`;
  $('#benchyResult').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
function parseInts(s) {
  return s.split(',').map((x) => x.trim()).filter(Boolean).map(Number).filter((n) => !Number.isNaN(n));
}

$('#benchyGo').addEventListener('click', () => {
  const log = $('#benchyLog'); log.textContent = '';
  $('#benchyResult').innerHTML = '';
  $('#benchyStop').disabled = false;
  $('#benchyGo').disabled = true;

  // EventSource is GET-only, so POST→SSE manually via fetch + reader.
  const body = {
    run_id: $('#benchyTarget').value || undefined,
    model: $('#benchyModel').value.trim() || undefined,
    served_model_name: $('#benchyServed').value.trim() || undefined,
    tokenizer: $('#benchyTokenizer').value.trim() || undefined,
    pp: parseInts($('#benchyPp').value),
    tg: parseInts($('#benchyTg').value),
    depth: parseInts($('#benchyDepth').value),
    concurrency: parseInts($('#benchyConcurrency').value),
    runs: Number($('#benchyRuns').value),
    latency_mode: $('#benchyLatency').value,
    enable_prefix_caching: $('#benchyPrefix').checked,
    no_cache: $('#benchyNoCache').checked,
    skip_coherence: $('#benchySkipCoh').checked,
  };
  benchyStream = new AbortController();
  fetch('/api/benchy/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal: benchyStream.signal,
  }).then(async (r) => {
    if (!r.ok) throw new Error(await r.text());
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let evt = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx);
        buf = buf.slice(idx + 1);
        if (line.startsWith('event:')) evt = line.slice(6).trim();
        else if (line.startsWith('data:')) {
          const data = line.slice(5).trim();
          if (evt === 'log') {
            log.textContent += data + '\n';
            log.scrollTop = log.scrollHeight;
          } else if (evt === 'done') {
            try {
              const obj = JSON.parse(data);
              renderBenchyResult(obj.result);
              toast('llama-benchy completed');
            } catch (e) { toast(`Result parse: ${e.message}`, 'danger'); }
          } else if (evt === 'error') {
            toast(`benchy error: ${data}`, 'danger');
          }
        }
      }
    }
  }).catch((e) => {
    if (e.name !== 'AbortError') toast(e.message, 'danger');
  }).finally(() => {
    $('#benchyGo').disabled = false;
    $('#benchyStop').disabled = true;
    benchyStream = null;
    refreshBenchTab();
  });
});
$('#benchyStop').addEventListener('click', () => { if (benchyStream) benchyStream.abort(); });

// Chart instances we re-use across runs.
let _benchyPpChart = null;
let _benchyTgChart = null;

const SERIES_COLORS = [
  '#76c7ff', '#b084ff', '#4ad08a', '#ffb74a', '#ff6272',
  '#ff9aa2', '#ffd166', '#06d6a0', '#118ab2', '#ef476f',
  '#83c5be', '#e29578', '#9bf6ff', '#bdb2ff', '#fdffb6', '#ffadad',
];

function renderBenchyResult(result) {
  const box = $('#benchyResult');
  const charts = $('#benchyCharts');
  if (!result) { box.innerHTML = '<div class="muted">No result file.</div>'; charts.hidden = true; return; }

  // The real llama-benchy JSON schema is
  // { version, timestamp, model, latency_ms, max_concurrency, benchmarks: [...] }
  // where each benchmark row has: concurrency, context_size, prompt_size,
  // response_size, is_context_prefill_phase, pp_throughput / tg_throughput /
  // peak_throughput / ttfr / est_ppt / e2e_ttft (each {mean, std, values}).
  const rows = result.benchmarks || result.results || result.data || (Array.isArray(result) ? result : []);
  if (!rows.length) {
    charts.hidden = true;
    box.innerHTML = `<pre class="mono" style="background:var(--bg);padding:10px;border-radius:6px;max-height:320px;overflow:auto">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
    return;
  }

  // ---- Charts: tokens/s vs concurrency, grouped by test+depth ----
  const series = groupBenchyForCharts(rows);
  const hasPP = series.pp.datasets.length > 0;
  const hasTG = series.tg.datasets.length > 0;
  charts.hidden = !(hasPP || hasTG);
  if (hasPP) renderBenchyChart('pp', series.pp);
  if (hasTG) renderBenchyChart('tg', series.tg);

  // ---- Table: render the canonical llama-benchy columns ----
  box.innerHTML = renderBenchyTable(result, rows);

  // ---- Download button ----
  const dl = document.createElement('button');
  dl.className = 'btn';
  dl.style.marginTop = '10px';
  dl.innerHTML = '<i class="fa-solid fa-download"></i> Download .txt';
  dl.addEventListener('click', () => downloadBenchyResult(result, rows));
  box.appendChild(dl);
}

function downloadBenchyResult(result, rows) {
  const lines = [];
  lines.push('llama-benchy Results');
  lines.push('='.repeat(60));
  lines.push(`Model    : ${result.model || '—'}`);
  lines.push(`Date     : ${result.timestamp || new Date().toISOString()}`);
  lines.push(`Latency  : ${(result.latency_ms ?? 0).toFixed(2)} ms`);
  lines.push('');

  const cols = ['Test', 't/s', 'peak t/s', 'ttfr (ms)', 'est_ppt (ms)', 'e2e_ttft (ms)'];
  const widths = [32, 18, 14, 12, 14, 14];
  const pad = (s, w) => String(s).padEnd(w);
  lines.push(cols.map((c, i) => pad(c, widths[i])).join(' | '));
  lines.push(widths.map((w) => '-'.repeat(w)).join('-+-'));

  for (const r of rows) {
    const depth = Number(r.context_size || 0);
    const ppTok = Number(r.prompt_size || 0);
    const tgTok = Number(r.response_size || 0);
    const fmt = (v) => {
      if (v == null) return '';
      if (typeof v === 'object' && 'mean' in v) return `${(+v.mean).toFixed(2)} ± ${(+v.std).toFixed(2)}`;
      return typeof v === 'number' ? v.toFixed(2) : String(v);
    };
    if (r.is_context_prefill_phase) {
      const test = `ctx_pp${ppTok || ''} @ d${depth} c${r.concurrency}`;
      lines.push([test, fmt(r.pp_throughput), '', fmt(r.ttfr), fmt(r.est_ppt), fmt(r.e2e_ttft)].map((v, i) => pad(v, widths[i])).join(' | '));
    } else {
      if (r.pp_throughput && (r.pp_throughput.mean ?? null) !== null) {
        const test = `pp${ppTok}${depth ? ` @ d${depth}` : ''}${r.concurrency > 1 ? ` c${r.concurrency}` : ''}`;
        lines.push([test, fmt(r.pp_throughput), '', fmt(r.ttfr), fmt(r.est_ppt), fmt(r.e2e_ttft)].map((v, i) => pad(v, widths[i])).join(' | '));
      }
      if (r.tg_throughput && (r.tg_throughput.mean ?? null) !== null) {
        const test = `tg${tgTok}${depth ? ` @ d${depth}` : ''}${r.concurrency > 1 ? ` c${r.concurrency}` : ''}`;
        lines.push([test, fmt(r.tg_throughput), fmt(r.peak_throughput), '', '', ''].map((v, i) => pad(v, widths[i])).join(' | '));
      }
    }
  }

  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const modelSlug = (result.model || 'benchy').replace(/[^a-z0-9._-]/gi, '_').slice(0, 40);
  const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
  a.href = url;
  a.download = `llama-benchy_${modelSlug}_${ts}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

function groupBenchyForCharts(rows) {
  const pp = new Map();
  const tg = new Map();
  const concSet = new Set();

  for (const r of rows) {
    const conc = Number(r.concurrency || 1);
    concSet.add(conc);
    const depth = Number(r.context_size || 0);
    const ppTok = Number(r.prompt_size || 0);
    const tgTok = Number(r.response_size || 0);

    const ppV = numeric(r.pp_throughput);
    const tgV = numeric(r.tg_throughput);

    if (r.is_context_prefill_phase) {
      // Context-prefill phase: only the ctx_pp series has a meaningful PP value.
      if (ppV != null) {
        const name = `ctx_pp @ d${depth}`;
        if (!pp.has(name)) pp.set(name, new Map());
        pp.get(name).set(conc, ppV);
      }
    } else {
      if (ppV != null && ppTok > 0) {
        const name = `pp${ppTok} @ d${depth}`;
        if (!pp.has(name)) pp.set(name, new Map());
        pp.get(name).set(conc, ppV);
      }
      if (tgV != null && tgTok > 0) {
        const name = `tg${tgTok} @ d${depth}`;
        if (!tg.has(name)) tg.set(name, new Map());
        tg.get(name).set(conc, tgV);
      }
    }
  }

  const concs = [...concSet].sort((a, b) => a - b);
  return { pp: shapeForChart(pp, concs), tg: shapeForChart(tg, concs) };
}

function renderBenchyTable(result, rows) {
  const head = `<div class="muted" style="margin:6px 0">model: <code>${escapeHtml(result.model || '')}</code> · latency: ${(result.latency_ms ?? 0).toFixed(2)} ms · ${result.timestamp || ''}</div>`;
  const cols = ['test', 't/s', 'peak t/s', 'ttfr (ms)', 'est_ppt (ms)', 'e2e_ttft (ms)'];
  const trs = rows.map((r) => {
    const depth = Number(r.context_size || 0);
    const ppTok = Number(r.prompt_size || 0);
    const tgTok = Number(r.response_size || 0);
    // llama-benchy renders each row as either pp or tg in markdown; here we
    // render *both* sub-rows when both throughputs are present, mirroring its
    // README output.
    const out = [];
    if (r.is_context_prefill_phase) {
      out.push({
        test: `ctx_pp${ppTok ? ppTok : ''} @ d${depth} c${r.concurrency}`,
        tps: r.pp_throughput, peak: null, ttfr: r.ttfr, ppt: r.est_ppt, e2e: r.e2e_ttft,
      });
    } else {
      if (r.pp_throughput && (r.pp_throughput.mean ?? null) !== null) {
        out.push({
          test: `pp${ppTok}${depth ? ` @ d${depth}` : ''}${r.concurrency > 1 ? ` c${r.concurrency}` : ''}`,
          tps: r.pp_throughput, peak: null, ttfr: r.ttfr, ppt: r.est_ppt, e2e: r.e2e_ttft,
        });
      }
      if (r.tg_throughput && (r.tg_throughput.mean ?? null) !== null) {
        out.push({
          test: `tg${tgTok}${depth ? ` @ d${depth}` : ''}${r.concurrency > 1 ? ` c${r.concurrency}` : ''}`,
          tps: r.tg_throughput, peak: r.peak_throughput, ttfr: null, ppt: null, e2e: null,
        });
      }
    }
    return out.map((o) => `<tr>
      <td class="mono">${escapeHtml(o.test)}</td>
      <td class="mono">${formatBenchyCell(o.tps)}</td>
      <td class="mono">${formatBenchyCell(o.peak)}</td>
      <td class="mono">${formatBenchyCell(o.ttfr)}</td>
      <td class="mono">${formatBenchyCell(o.ppt)}</td>
      <td class="mono">${formatBenchyCell(o.e2e)}</td>
    </tr>`).join('');
  }).join('');
  return head + `<table><tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr>${trs}</table>`;
}

function shapeForChart(family, concs) {
  const labels = concs.map((c) => `c${c}`);
  const names = [...family.keys()].sort();
  const datasets = names.map((name, i) => ({
    label: name,
    data: concs.map((c) => family.get(name).get(c) ?? null),
    spanGaps: true,
    borderColor: SERIES_COLORS[i % SERIES_COLORS.length],
    backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length],
    borderWidth: 2,
    pointRadius: 4,
    tension: 0.2,
  }));
  return { labels, datasets };
}

function isPpSeries(test) {
  const t = test.toLowerCase();
  return t.startsWith('pp') || t.startsWith('ctx_pp') || t.startsWith('ctx pp');
}

function canonicalSeries(test) {
  // Strip trailing "@ cN" so multiple concurrency rows collapse into one series.
  return test.replace(/\s*@\s*c\d+\s*$/i, '').trim();
}

function extractTagged(s, prefix) {
  const m = s.match(new RegExp(`@\\s*${prefix}(\\d+)`, 'i'));
  return m ? Number(m[1]) : null;
}

function numeric(v) {
  if (v == null) return null;
  if (typeof v === 'number') return v;
  if (typeof v === 'string') { const n = Number(v); return Number.isFinite(n) ? n : null; }
  if (typeof v === 'object' && 'mean' in v) return Number(v.mean);
  return null;
}

function renderBenchyChart(kind, data) {
  const ref = kind === 'pp' ? '_benchyPpChart' : '_benchyTgChart';
  const canvasId = kind === 'pp' ? 'benchyPpChart' : 'benchyTgChart';
  if (window[ref]) { window[ref].destroy(); }
  window[ref] = new Chart($(`#${canvasId}`), {
    type: 'line',
    data,
    options: {
      responsive: true,
      animation: false,
      plugins: {
        legend: { labels: { color: '#e6edf7', boxWidth: 12, font: { size: 11 } }, position: 'bottom' },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${(+ctx.parsed.y).toFixed(2)} tok/s`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#8592a6' },
          grid: { color: 'rgba(133,146,166,0.15)' },
          title: { display: true, text: 'Concurrency', color: '#8592a6' },
        },
        y: {
          ticks: { color: '#8592a6' },
          grid: { color: 'rgba(133,146,166,0.15)' },
          title: { display: true, text: 'Tokens/sec', color: '#8592a6' },
          beginAtZero: true,
        },
      },
    },
  });
  _benchyPpChart = ref === '_benchyPpChart' ? window[ref] : _benchyPpChart;
  _benchyTgChart = ref === '_benchyTgChart' ? window[ref] : _benchyTgChart;
}
function formatBenchyCell(v) {
  if (v == null) return '';
  if (typeof v === 'object') {
    if ('mean' in v && 'std' in v) return `${(+v.mean).toFixed(2)} ± ${(+v.std).toFixed(2)}`;
    return escapeHtml(JSON.stringify(v));
  }
  if (typeof v === 'number') return v.toFixed(2);
  return escapeHtml(String(v));
}

$('#benchGo').addEventListener('click', async () => {
  const body = {
    run_id: $('#benchRun').value || undefined,
    prompt: $('#benchPrompt').value || undefined,
    max_tokens: Number($('#benchTokens').value),
    runs: Number($('#benchRuns').value),
  };
  try {
    const res = await api('/bench', { method: 'POST', body });
    toast(`tokens/s ${(res.tokens_per_sec || 0).toFixed(1)} · TTFT ${(res.ttft_ms || 0).toFixed(0)}ms`);
    refreshBenchTab();
  } catch (e) { toast(e.message, 'danger'); }
});

// ---------- agents login -------------------------------------------------
async function refreshAgents() {
  const a = await api('/agents/status');
  const pkgs = { claude: '@anthropic-ai/claude-code', codex: '@openai/codex' };
  for (const which of ['claude', 'codex']) {
    const s = a[which] || {};
    const status = $(`#${which}Status`);
    const btn = $(`#${which}Login`);
    if (!s.installed) {
      status.innerHTML = `<span class="badge no">not installed</span> — install with <code>npm install -g ${pkgs[which]}</code>, then reopen this tab.`;
      btn.disabled = true;
      btn.classList.remove('primary');
      btn.innerHTML = '<i class="fa-solid fa-right-to-bracket"></i> Log in';
    } else if (s.logged_in) {
      status.innerHTML = '<span class="badge ok">✓ logged in</span> — ready to fix and optimize recipes.';
      btn.disabled = false;
      btn.classList.remove('primary');
      btn.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Re-authenticate';
    } else {
      status.innerHTML = '<span class="badge warn">not logged in</span> — uses your existing subscription, no API key needed.';
      btn.disabled = false;
      btn.classList.add('primary');
      btn.innerHTML = '<i class="fa-solid fa-right-to-bracket"></i> Log in';
    }
  }
}
$('#claudeLogin').addEventListener('click', () => streamLogin('claude'));
$('#codexLogin').addEventListener('click', () => streamLogin('codex'));
function streamLogin(which) {
  const pre = $(`#${which}Log`);
  pre.textContent = '';
  const es = new EventSource(`/api/agents/login/${which}`);
  es.addEventListener('log', (ev) => {
    pre.textContent += ev.data + '\n';
    pre.scrollTop = pre.scrollHeight;
    const url = ev.data.match(/https?:\/\/\S+/);
    if (url) {
      window.open(url[0], '_blank', 'noopener');
    }
  });
  es.addEventListener('done', () => { es.close(); refreshAgents(); });
  es.addEventListener('error', () => { es.close(); refreshAgents(); });
}

// ---------- Engine chat tab ----------------------------------------------
const wgAttachments = [];
const wgHistory = [];
const wgSearch = { enabled: false, available: false, url: '', backend: '', state: '', error: '', initialized: false };
let _wgSearchStartingPoll = null;

function updateWgControls() {
  $('#wgTempValue').textContent = Number($('#wgTemp').value).toFixed(2);
  $('#wgMaxTokensValue').textContent = String(Number($('#wgMaxTokens').value));
}
$('#wgTemp').addEventListener('input', updateWgControls);
$('#wgMaxTokens').addEventListener('input', updateWgControls);
updateWgControls();

bindAttachmentUI({ btnId: 'wgAttachBtn', inputId: 'wgAttachInput', listId: 'wgAttachments', store: wgAttachments });

// ---------- Globe search pill toggle ---------------------------------------
function _applySearchPillState() {
  const pill = $('#wgSearchToggle');
  if (!pill) return;
  pill.classList.toggle('active', wgSearch.enabled && wgSearch.available);
  pill.classList.toggle('unavailable', !wgSearch.available);
  pill.title = wgSearch.available
    ? (wgSearch.enabled ? 'Web search ON — click to disable' : 'Web search OFF — click to enable')
    : `Web search unavailable${wgSearch.error ? ': ' + wgSearch.error : ''}`;
}

$('#wgSearchToggle').addEventListener('click', () => {
  if (!wgSearch.available) {
    toast(wgSearch.error || 'Web search not available — start Docker for bundled SearXNG', 'danger');
    return;
  }
  wgSearch.enabled = !wgSearch.enabled;
  _applySearchPillState();
  const ta = $('#wgInput');
  ta.placeholder = wgSearch.enabled ? 'Ask with web search… (Ctrl+Enter)' : 'Ask…';
});

async function refreshWgSearchStatus() {
  try {
    const status = await api('/search/status');
    wgSearch.available = !!status.enabled;
    wgSearch.url = status.url || '';
    wgSearch.backend = status.backend || '';
    wgSearch.state = status.state || '';
    wgSearch.error = status.error || '';
  } catch (e) {
    wgSearch.available = false;
    wgSearch.url = '';
    wgSearch.backend = '';
    wgSearch.state = '';
    wgSearch.error = e.message;
  }
  // Auto-enable on first load if search is available
  if (!wgSearch.initialized) {
    wgSearch.initialized = true;
    wgSearch.enabled = wgSearch.available;
  }
  _applySearchPillState();

  const statusEl = $('#wgSearxStatus');
  if (!statusEl) return;

  // Bundled SearXNG container still coming up (first-run image pull can take a
  // while) — show a spinner and keep polling until it's ready.
  if (wgSearch.state === 'starting') {
    statusEl.innerHTML = '<span class="search-status-dot warn pulse"></span>Starting bundled SearXNG search engine…';
    if (_wgSearchStartingPoll) clearTimeout(_wgSearchStartingPoll);
    _wgSearchStartingPoll = setTimeout(refreshWgSearchStatus, 4000);
    return;
  }
  if (_wgSearchStartingPoll) { clearTimeout(_wgSearchStartingPoll); _wgSearchStartingPoll = null; }

  if (!wgSearch.available) {
    const st = await api('/searxng/status').catch(() => ({ managed: {} }));
    const docker = st.managed && st.managed.docker;
    statusEl.innerHTML = docker
      ? '<span class="search-status-dot warn"></span>Bundled SearXNG not running — retry from the search controls'
      : '<span class="search-status-dot off"></span>No search · install Docker so the bundled SearXNG can start · DuckDuckGo fallback via Update';
    return;
  }
  let backendLabel;
  if (wgSearch.backend === 'duckduckgo') {
    backendLabel = '🦆 DuckDuckGo (fallback)';
  } else if (wgSearch.backend === 'searxng (bundled)') {
    backendLabel = `🔍 SearXNG (bundled) · <code>${wgSearch.url}</code>`;
  } else {
    backendLabel = `🔍 SearXNG · <code>${wgSearch.url}</code>`;
  }
  statusEl.innerHTML = `<span class="search-status-dot on"></span>Search ready · ${backendLabel}`;
}

// Holds the search results from the most recent web-grounded message
// so we can render source cards after the assistant responds.
let _lastSearchResults = [];

async function buildWgUserMessage(text, attachments) {
  _lastSearchResults = [];
  const base = buildUserMessage(text, attachments);
  if (!wgSearch.enabled || !text) return base;

  // Show searching state on the pill
  const pill = $('#wgSearchToggle');
  if (pill) pill.classList.add('searching');
  try {
    const qs = new URLSearchParams({ q: text, limit: '5', enrich: '1' });
    const resp = await fetch(`/api/search?${qs.toString()}`);
    if (!resp.ok) throw new Error(await resp.text());
    const payload = await resp.json();
    if (!payload.results?.length) return base;

    _lastSearchResults = payload.results;

    // Full page excerpts (server-fetched) beat snippets by an order of
    // magnitude — the model gets actual article text to synthesize from.
    const resultsText = payload.results
      .map((item, idx) => {
        const date = item.published ? `\nPublished: ${item.published}` : '';
        const body = item.content
          ? `Page content:\n${item.content}`
          : (item.snippet ? `Snippet: ${item.snippet}` : '(no excerpt available)');
        return `[Source ${idx + 1}] ${item.title}\nURL: ${item.url}${date}\n${body}`;
      })
      .join('\n\n=====\n\n');
    const block = `[Web search results — use these for an accurate, up-to-date answer:]\n\n${resultsText}\n\n[End of search results]`;
    if (typeof base.content === 'string') {
      return { role: 'user', content: `${block}\n\n---\n${base.content}` };
    }
    return {
      role: 'user',
      content: [{ type: 'text', text: `${block}\n\n---\n` }, ...base.content],
    };
  } catch (e) {
    toast(`Web search failed — sending without context: ${e.message}`, 'danger');
    return base;
  } finally {
    if (pill) pill.classList.remove('searching');
  }
}

// Renders glassmorphism source citation cards below an assistant bubble.
function renderSearchSources(results, msgContainer) {
  if (!results || !results.length) return;
  const wrapper = h('div', { class: 'search-sources' });
  const label = h('div', { class: 'search-sources-label' }, `Sources · ${results.length}`);
  wrapper.appendChild(label);
  results.forEach((item, idx) => {
    let domain = '';
    try { domain = new URL(item.url).hostname.replace(/^www\./, ''); } catch { domain = item.url; }
    const faviconSrc = `https://www.google.com/s2/favicons?domain=${domain}&sz=16`;
    const card = h('a', {
      class: 'search-source-card',
      href: item.url,
      target: '_blank',
      rel: 'noopener noreferrer',
      style: `animation-delay: ${idx * 80}ms`,
    },
      h('img', { class: 'search-source-favicon', src: faviconSrc, alt: '', onerror: "this.style.display='none'" }),
      h('div', { class: 'search-source-body' },
        h('div', { class: 'search-source-title' }, item.title || domain),
        item.snippet ? h('div', { class: 'search-source-snippet' }, item.snippet) : null,
        h('div', { class: 'search-source-domain' }, domain),
      ),
      h('span', { class: 'search-source-link' }, '↗'),
    );
    wrapper.appendChild(card);
  });
  msgContainer.appendChild(wrapper);
}

async function refreshWgServerStatus() {
  const active = await api('/active').catch(() => null);
  if (!active) {
    $('#wgServerStatus').innerHTML = '⚠ No engine running. Start one from the vLLM / SGLang / llama.cpp tab.';
    $('#wgSettingsStatus').textContent = 'No running engine. Temperature and max tokens will apply once an engine is active.';
    return;
  }
  const served = await api('/models/served').catch(() => ({}));
  const ctxLen = served.max_model_len;
  _applyMaxModelLen('wgMaxTokens', 'wgMaxTokensValue', ctxLen);
  const ctxLabel = ctxLen ? ` · ctx ${(ctxLen / 1024).toFixed(0)}k` : '';
  $('#wgServerStatus').innerHTML = `Talking to <span class="badge ok">${active.engine}</span> at <code>${active.url}</code>${served.model ? ` · model <code>${served.model}</code>` : ''}${ctxLabel}. Image inputs are supported when the model is multimodal.`;
  $('#wgSettingsStatus').textContent = `Server model: ${served.model || 'auto'} · temperature ${Number($('#wgTemp').value).toFixed(2)} · max tokens ${Number($('#wgMaxTokens').value)}${ctxLabel}`;
}


$('#wgSend').addEventListener('click', sendWg);
$('#wgStop').addEventListener('click', stopWg);
$('#wgClear').addEventListener('click', () => {
  wgHistory.length = 0;
  wgAttachments.length = 0;
  renderAttachments($('#wgAttachments'), wgAttachments);
  $('#wgMessages').innerHTML = '';
});
$('#wgInput').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) sendWg(); });

let wgStream = null;
function setWgBusy(busy) {
  $('#wgSend').disabled = busy;
  $('#wgStop').disabled = !busy;
  $('#wgChat')?.classList.toggle('generating', busy);
  _ambient.targetIntensity = busy ? 1.05 : 0.82;
}
function stopWg() {
  if (wgStream) wgStream.abort();
}

async function sendWg() {
  if (wgStream) return;
  const text = $('#wgInput').value.trim();
  if (!text && !wgAttachments.length) return;
  $('#wgInput').value = '';
  const currentAttachments = [...wgAttachments];

  const msgs = $('#wgMessages');
  msgs.appendChild(renderUserBubble(text, currentAttachments));
  const reply = appendAssistantMessage(msgs);
  msgs.scrollTop = msgs.scrollHeight;
  wgAttachments.length = 0;
  renderAttachments($('#wgAttachments'), wgAttachments);

  try {
    wgStream = new AbortController();
    setWgBusy(true);
    reply.classList.add('streaming');
    const msg = await buildWgUserMessage(text, currentAttachments);
    wgHistory.push(msg);
    const modelInfo = await currentServerModelInfo();
    if (modelInfo.maxModelLen) _applyMaxModelLen('wgMaxTokens', 'wgMaxTokensValue', modelInfo.maxModelLen);
    const messages = withModelIdentity(wgHistory, modelInfo.model);
    const renderLive = makeStreamRenderer(reply, msgs);
    const res = await streamChat(messages, renderLive, {
      model: modelInfo.model,
      temperature: Number($('#wgTemp').value),
      maxTokens: Number($('#wgMaxTokens').value),
      signal: wgStream.signal,
    });
    if (!res.text && !res.reasoning) { reply.textContent = '(empty response)'; } else { renderChatContent(reply, res.text || '*(the model spent its whole budget thinking — raise max tokens or ask again)*', { reasoning: res.reasoning }); }
    wgHistory.push({ role: 'assistant', content: res.text });
    setAssistantMetrics(reply, res.metrics);
    // Attach source cards if this was a web-grounded reply
    if (_lastSearchResults.length) renderSearchSources(_lastSearchResults, reply.closest('.chat-msg') || reply);
    if (res.metrics?.tokps) recordTps(res.metrics.tokps);
    const searchLabel = wgSearch.enabled && _lastSearchResults.length
      ? ` · ${wgSearch.backend === 'duckduckgo' ? 'DDG' : 'SearXNG'} ↗${_lastSearchResults.length}`
      : '';
    $('#wgSettingsStatus').textContent = `Server model: ${modelInfo.model} · ${formatMetrics(res.metrics)}${searchLabel}`;
  } catch (e) {
    if (e.name === 'AbortError') {
      if (reply.textContent) renderChatContent(reply, reply.textContent);
      else reply.textContent = '(stopped)';
      $('#wgSettingsStatus').textContent = 'Server chat stopped';
    } else {
      reply.textContent = `Error: ${e.message}`;
    }
  } finally {
    reply.classList.remove('streaming');
    wgStream = null;
    setWgBusy(false);
  }
}

refreshWgSearchStatus();

// ---------- toast --------------------------------------------------------
function toast(msg, kind = 'ok') {
  const t = h('div', { class: `toast ${kind}` }, msg);
  Object.assign(t.style, {
    position: 'fixed', bottom: '24px', right: '24px', padding: '10px 14px',
    background: kind === 'danger' ? 'var(--danger)' : 'var(--bg-elev)',
    color: kind === 'danger' ? '#fff' : 'var(--text)',
    border: '1px solid var(--border)', borderRadius: '10px',
    boxShadow: '0 10px 40px rgba(0,0,0,0.4)', zIndex: 200, fontSize: '13px',
  });
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

function escapeHtml(s) { return String(s ?? '').replace(/[&<>"']/g, (m) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

// Copy text to the clipboard. navigator.clipboard only exists in secure
// contexts (https / localhost); when the app is served over plain http on the
// LAN, fall back to the legacy hidden-textarea path.
async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(text); return; } catch { /* fall through */ }
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try {
    if (!document.execCommand('copy')) throw new Error('copy rejected');
  } finally { ta.remove(); }
}

// ---------- WebGPU ambient aurora background -------------------------------

const _AURORA_WGSL = /* wgsl */`
struct Uniforms { time: f32, width: f32, height: f32, intensity: f32 }
@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>,3>(
    vec2<f32>(-1.0,-3.0), vec2<f32>(-1.0,1.0), vec2<f32>(3.0,1.0));
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

fn h2(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}
fn vn(p: vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f*f*(3.0-2.0*f);
  return mix(mix(h2(i),h2(i+vec2(1.,0.)),u.x),
             mix(h2(i+vec2(0.,1.)),h2(i+vec2(1.,1.)),u.x),u.y);
}
fn fbm(p: vec2<f32>) -> f32 {
  var v=0.0; var a=0.5; var pp=p;
  for(var i=0;i<4;i++){v+=a*vn(pp);pp=pp*2.1+vec2(1.3,0.7);a*=0.5;}
  return v;
}

@fragment
fn fs_main(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = fc.xy / vec2<f32>(u.width, u.height);
  let t  = u.time * 0.14;
  var col = vec3<f32>(0.0);

  // Band 1: electric blue
  let n1 = fbm(vec2(uv.x*2.0 + t*0.5, t*0.22));
  let w1 = sin(uv.x*3.3 + t*0.70 + n1*1.9)*0.115;
  col += exp(-abs(uv.y-0.36-w1)*18.0) * vec3(0.05,0.50,1.00) * 0.64;

  // Band 2: violet-purple
  let n2 = fbm(vec2(uv.x*1.6 + 4.1 - t*0.35, t*0.19+2.3));
  let w2 = sin(uv.x*2.2 - t*0.58 + n2*2.1+1.9)*0.10;
  col += exp(-abs(uv.y-0.57-w2)*23.0) * vec3(0.54,0.09,0.96) * 0.52;

  // Band 3: teal
  let n3 = fbm(vec2(uv.x*3.3 + 8.9 + t*0.55, t*0.27+5.5));
  let w3 = sin(uv.x*5.3 + t*0.88 + n3*1.6+3.6)*0.08;
  col += exp(-abs(uv.y-0.25-w3)*27.0) * vec3(0.00,0.88,0.76) * 0.40;

  // Band 4: pink-magenta (faint, slow)
  let w4 = sin(uv.x*1.8 + t*0.36+5.8)*0.09;
  col += exp(-abs(uv.y-0.71-w4)*30.0) * vec3(0.95,0.22,0.64) * 0.30;

  // Ambient nebula blobs
  col += exp(-length((uv-vec2(0.25,0.40))*vec2(1.6,2.1))*4.2)*vec3(0.08,0.40,0.90)*0.14;
  col += exp(-length((uv-vec2(0.75,0.56))*vec2(1.9,2.3))*5.0)*vec3(0.52,0.09,0.82)*0.12;

  // Fine grain
  col = max(col + (h2(uv*420.0 + t*19.0)-0.5)*0.018, vec3(0.0));

  // Vignette
  let vgn = 1.0-smoothstep(0.28,0.88,length((uv-0.5)*vec2(1.35,1.05)));
  col *= vgn;

  // Subtle tonemap + gamma
  col = col/(col+vec3(0.75));
  col = pow(col, vec3(0.88));

  return vec4<f32>(col, u.intensity * 0.80);
}
`;

const _ambient = {
  device: null, pipeline: null, uniformBuf: null, bindGroup: null,
  ctx: null, running: false, raf: null,
  intensity: 0.82, targetIntensity: 0.82,
  startTime: performance.now() / 1000,
};

async function _initAmbient() {
  if (_ambient.device) return true;
  if (!navigator.gpu) return false;
  const canvas = $('#wgAmbientCanvas');
  if (!canvas) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return false;
    const device = await adapter.requestDevice();
    const ctx = canvas.getContext('webgpu');
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format: fmt, alphaMode: 'premultiplied' });

    const uniformBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ code: _AURORA_WGSL });
    const bgl = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: {} }],
    });
    const pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex:   { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: fmt,
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one',       dstFactor: 'zero',               operation: 'add' },
        },
      }]},
      primitive: { topology: 'triangle-list' },
    });
    const bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
    });
    Object.assign(_ambient, { device, pipeline, uniformBuf, bindGroup, ctx });
    return true;
  } catch (e) {
    console.warn('WebGPU ambient init failed:', e.message);
    return false;
  }
}

function _renderAmbientFrame() {
  if (!_ambient.running || !_ambient.device) return;
  const canvas = $('#wgAmbientCanvas');
  if (!canvas) return;

  // Sync canvas pixel size to its CSS size
  const dpr = window.devicePixelRatio || 1;
  const parent = canvas.parentElement;
  const cw = Math.round(parent.clientWidth  * dpr);
  const ch = Math.round(parent.clientHeight * dpr);
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw; canvas.height = ch;
    canvas.style.width  = parent.clientWidth  + 'px';
    canvas.style.height = parent.clientHeight + 'px';
  }

  // Lerp intensity
  _ambient.intensity += (_ambient.targetIntensity - _ambient.intensity) * 0.04;

  const t = (performance.now() / 1000) - _ambient.startTime;
  _ambient.device.queue.writeBuffer(
    _ambient.uniformBuf, 0,
    new Float32Array([t, cw, ch, _ambient.intensity]),
  );

  const encoder = _ambient.device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: _ambient.ctx.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: 'clear', storeOp: 'store',
    }],
  });
  pass.setPipeline(_ambient.pipeline);
  pass.setBindGroup(0, _ambient.bindGroup);
  pass.draw(3);
  pass.end();
  _ambient.device.queue.submit([encoder.finish()]);
  _ambient.raf = requestAnimationFrame(_renderAmbientFrame);
}

async function startAmbient() {
  const ok = await _initAmbient();
  if (!ok) return;
  _ambient.running = true;
  if (!_ambient.raf) _renderAmbientFrame();
}

function stopAmbient() {
  _ambient.running = false;
  if (_ambient.raf) { cancelAnimationFrame(_ambient.raf); _ambient.raf = null; }
}

// ---------- live DGX Spark vitals ----------------------------------------

const _vitals = { last: null, expanded: false };
const _tpsHistory = [];   // rolling last-30 tok/s values from completed chats

// Draw the unified-memory bar: the fill grows left→right over a fixed
// green→amber→red gradient laid across the whole track, so the leading edge
// shifts from green toward red as usage climbs. Deliberately no glow and no
// rounded stroke caps — the old arc's halo made a steady reading look jittery.
function _drawMemBar(canvas, pct, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const logW = canvas.clientWidth || canvas.width;
  const logH = canvas.clientHeight || canvas.height;
  if (!logW) return; // hidden (collapsed body) — nothing to draw into
  canvas.width  = logW * dpr;
  canvas.height = logH * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, logW, logH);

  const radius = logH / 2;
  const rounded = (w) => {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(0, 0, w, logH, Math.min(radius, w / 2));
    else ctx.rect(0, 0, w, logH);
  };

  // Track
  rounded(logW);
  ctx.fillStyle = 'rgba(255,255,255,0.08)';
  ctx.fill();

  const frac = Math.max(0, Math.min(1, (pct || 0) / 100));
  if (frac > 0) {
    const grad = ctx.createLinearGradient(0, 0, logW, 0);
    grad.addColorStop(0, '#4ad08a');    // green — plenty free
    grad.addColorStop(0.7, '#ffb74a');  // amber — getting full
    grad.addColorStop(1, '#ff6272');    // red — nearly out
    rounded(Math.max(logH, logW * frac)); // min = height so the pill stays round
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // Centered label on the large bar only
  if (opts.label && logH >= 14) {
    ctx.font = `bold ${Math.round(logH * 0.6)}px var(--mono, monospace)`;
    ctx.fillStyle = '#e8eaf0';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(0,0,0,0.65)';
    ctx.shadowBlur = 3;
    ctx.fillText(`${Math.round(pct)}% MEM`, logW / 2, logH / 2 + 0.5);
    ctx.shadowBlur = 0;
  }
}

// Draw a mini TPS sparkline
function _drawTpsSparkline() {
  const canvas = $('#vitalsTps');
  if (!canvas || !_tpsHistory.length) return;
  const dpr = window.devicePixelRatio || 1;
  const lw = canvas.width, lh = canvas.height;
  canvas.width  = lw * dpr;
  canvas.height = lh * dpr;
  canvas.style.width  = lw + 'px';
  canvas.style.height = lh + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, lw, lh);

  const max = Math.max(..._tpsHistory, 1);
  const step = lw / Math.max(_tpsHistory.length - 1, 1);
  const pad = 4;

  // Gradient fill under curve
  const grad = ctx.createLinearGradient(0, 0, 0, lh);
  grad.addColorStop(0, 'rgba(118,199,255,0.3)');
  grad.addColorStop(1, 'rgba(118,199,255,0.02)');

  ctx.beginPath();
  _tpsHistory.forEach((v, i) => {
    const x = i * step;
    const y = lh - pad - (v / max) * (lh - pad * 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo((_tpsHistory.length - 1) * step, lh);
  ctx.lineTo(0, lh);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  _tpsHistory.forEach((v, i) => {
    const x = i * step;
    const y = lh - pad - (v / max) * (lh - pad * 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#76c7ff';
  ctx.lineWidth = 1.5;
  ctx.shadowBlur = 6;
  ctx.shadowColor = '#76c7ff';
  ctx.stroke();
  ctx.shadowBlur = 0;

  const latest = _tpsHistory[_tpsHistory.length - 1];
  $('#vitalsTpsLabel').textContent = `${latest.toFixed(1)} tok/s`;
}

// Called after every completed assistant message to feed the TPS sparkline
function recordTps(tokps) {
  if (!tokps || !isFinite(tokps)) return;
  _tpsHistory.push(tokps);
  if (_tpsHistory.length > 40) _tpsHistory.shift();
  if (_vitals.expanded) _drawTpsSparkline();
}

function _updateVitalsUI(d) {
  const memPct = d.mem_pct ?? 0;
  const memUsed = d.mem_used_gb ?? 0;
  const memTotal = d.mem_total_gb ?? 0;
  const gpuUtil  = d.gpu_util  != null ? `${Math.round(d.gpu_util)}%` : '—';
  const gpuTemp  = d.gpu_temp  != null ? `${Math.round(d.gpu_temp)}°C` : '—';
  const gpuPower = d.gpu_power != null ? `${d.gpu_power.toFixed(1)} W` : '—';
  const gpuClock = d.gpu_clock != null ? `${Math.round(d.gpu_clock)} MHz` : '—';
  const cpu      = d.cpu_pct   != null ? `${d.cpu_pct.toFixed(1)}%` : '—';

  // Collapsed header
  $('#vitalsMemText').textContent = `${memUsed} / ${memTotal} GB`;
  const gpuBlurb = d.gpu_util != null
    ? `GPU ${Math.round(d.gpu_util)}% · CPU ${d.cpu_pct?.toFixed(0)}% · ${gpuTemp}`
    : `CPU ${d.cpu_pct?.toFixed(0)}% · Mem ${memPct}%`;
  $('#vitalsStatsText').textContent = gpuBlurb;

  // Slim bar (header)
  _drawMemBar($('#vitalsBar'), memPct);

  if (!_vitals.expanded) return;

  // Large labeled bar
  _drawMemBar($('#vitalsBarLg'), memPct, { label: true });

  // Stat rows
  $('#vGpuUtil').textContent  = gpuUtil;
  $('#vGpuTemp').textContent  = gpuTemp;
  $('#vGpuPower').textContent = gpuPower;
  $('#vGpuClock').textContent = gpuClock;
  $('#vCpu').textContent      = cpu;

  _drawTpsSparkline();
}

// Toggle expand / collapse
$('#vitalsToggle').addEventListener('click', () => {
  _vitals.expanded = !_vitals.expanded;
  $('#vitalsWidget').classList.toggle('expanded', _vitals.expanded);
  if (_vitals.expanded && _vitals.last) {
    _updateVitalsUI(_vitals.last);
    _drawTpsSparkline();
  }
});

// Start the SSE stream
(function _startVitalsStream() {
  const es = new EventSource('/api/spark/vitals');
  es.addEventListener('vitals', (ev) => {
    try {
      const d = JSON.parse(ev.data);
      _vitals.last = d;
      _updateVitalsUI(d);
    } catch { /* ignore parse errors */ }
  });
  es.addEventListener('error', () => {
    // Back-off is handled by the browser; just update header text
    $('#vitalsMemText').textContent = 'reconnecting…';
  });
})();

// ---------- UI mode (Beginner / Advanced) ---------------------------------
// Beginner hides power-user tabs behind a simple five-tab layout. Stored per
// browser; fresh installs (no recipes, no runs) default to Beginner, existing
// setups stay Advanced so nobody gets demoted on their own box.
const BEGINNER_TABS = new Set(['overview', 'recipes', 'models', 'chat', 'logs', 'recovery']);
function applyUiMode(mode) {
  document.body.dataset.uimode = mode;
  localStorage.setItem('uiMode', mode);
  $$('.ui-mode-btn').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  // If the active tab just got hidden, land somewhere visible.
  const active = $('.tab.active');
  if (mode === 'beginner' && active && !BEGINNER_TABS.has(active.dataset.tab)) {
    $('.tab[data-tab="overview"]')?.click();
  }
}
$$('.ui-mode-btn').forEach((b) => b.addEventListener('click', () => applyUiMode(b.dataset.mode)));

// ---------- cluster page ------------------------------------------------------
// Live node telemetry: sparkrun cluster monitor --json relayed over SSE.
let clusterMonES = null;
function stopClusterMonitor() {
  if (clusterMonES) { clusterMonES.close(); clusterMonES = null; }
}
function startClusterMonitor() {
  if (clusterMonES) return;
  clusterMonES = new EventSource('/api/cluster/monitor');
  clusterMonES.addEventListener('nodes', (ev) => {
    let d;
    try { d = JSON.parse(ev.data); } catch { return; }
    for (const [ip, m] of Object.entries(d.hosts || {})) {
      const el = document.querySelector(`.cluster-node[data-ip="${CSS.escape(ip)}"] .node-vitals`);
      if (!el) continue;
      const n = (v) => (v === '' || v == null ? null : Number(v));
      const memUsed = n(m.mem_used_mb), memTot = n(m.mem_total_mb);
      const bits = [
        n(m.cpu_usage_pct) != null ? `CPU ${m.cpu_usage_pct}%` : null,
        memUsed != null && memTot ? `mem ${(memUsed / 1024).toFixed(1)}/${(memTot / 1024).toFixed(0)} GB` : null,
        n(m.gpu_util_pct) != null ? `GPU ${m.gpu_util_pct}%` : null,
        n(m.gpu_temp_c) != null ? `${m.gpu_temp_c}°C` : null,
        n(m.gpu_power_w) != null ? `${Number(m.gpu_power_w).toFixed(0)} W` : null,
        n(m.sparkrun_jobs) ? `▶ ${m.sparkrun_jobs} job${Number(m.sparkrun_jobs) > 1 ? 's' : ''}` : null,
      ].filter(Boolean);
      el.textContent = bits.join(' · ');
      el.title = `${m.hostname || ip} · load ${m.cpu_load_1m ?? '?'} · GPU clock ${m.gpu_clock_mhz || '?'} MHz`;
    }
  });
  clusterMonES.addEventListener('error', () => { /* browser retries; sparkrun may be busy */ });
}

async function refreshCluster() {
  try {
    const c = await api('/cluster');
    $('#clusterTitle').textContent = c.mesh
      ? `${c.cluster_name || 'Cluster'} — ${c.online_nodes}/${c.nodes.length} nodes online`
      : `${c.cluster_name || 'Cluster'} — Single Node`;
    $('#clusterSoloCta').hidden = c.mesh;
    $('#clusterNodes').innerHTML = c.nodes.map((n) => {
      const dot = n.online ? '🟢' : '🔴';
      const mem = n.memory_free_gb != null ? ` · ${n.memory_free_gb} GB free / ${n.memory_total_gb} GB` : '';
      const wl = n.workload
        ? `<div class="muted">▶ ${escapeHtml(n.workload.job)} · ${escapeHtml(n.workload.role)} (${escapeHtml(n.workload.state)})</div>` : '';
      return `<div class="cluster-node" data-ip="${escapeHtml(n.ip)}">
        <div>${dot} <b>${escapeHtml(n.ip)}</b>${n.local ? ' <span class="badge">this Spark</span>' : ''}
          ${n.summary ? `<span class="muted"> · ${escapeHtml(n.summary)}${mem}</span>` : (n.online ? '' : ' <span class="muted">— unreachable</span>')}</div>
        <div class="node-vitals muted mono"></div>
        ${wl}</div>`;
    }).join('');
    startClusterMonitor(); // live per-node CPU/mem/GPU/power via sparkrun
    $('#clusterTp').innerHTML = 'Tensor parallel: ' + c.tp_options.map((t) =>
      `<span class="badge ${t.ok ? 'ok' : 'no'}" title="${escapeHtml(t.why || '')}">TP ${t.tp} ${t.ok ? '✓' : '✗'}</span>`
    ).join(' ');
    const sel = $('#clusterTpSel');
    sel.innerHTML = c.tp_options.map((t) => `<option value="${t.tp}" ${!t.ok ? 'disabled' : ''}>${t.tp}</option>`).join('');
    $('#clusterJobs').innerHTML = c.jobs.length ? c.jobs.map((j) => `
      <div class="cluster-job">
        <b>${escapeHtml(j.ref)}</b> · TP ${j.tp} · <span class="mono">${escapeHtml(j.jobid)}</span>
        <div>${(j.containers || []).map((ct) =>
          `<button class="btn" data-nodelog="${escapeHtml(ct)}">📄 ${escapeHtml(ct.split('_').pop())} log</button>`).join(' ')}</div>
      </div>`).join('') : '<span class="muted">None.</span>';
    $$('#clusterJobs [data-nodelog]').forEach((b) => b.addEventListener('click', async () => {
      const pre = $('#clusterNodeLog');
      pre.style.display = 'block';
      pre.textContent = 'Loading…';
      try {
        const r = await api(`/sparkrun/nodelog?container=${encodeURIComponent(b.dataset.nodelog)}`);
        pre.textContent = r.lines.length ? r.lines.join('\n')
          : (r.note || 'no log') + '\nFor remote nodes use: sparkrun logs <jobid>';
        pre.scrollTop = pre.scrollHeight;
      } catch (e) { pre.textContent = e.message; }
    }));
  } catch (e) {
    $('#clusterNodes').innerHTML = `<span class="muted">${escapeHtml(e.message)}</span>`;
  }
}
$('#clusterCheck')?.addEventListener('click', async () => {
  const box = $('#clusterReadiness');
  box.innerHTML = '<div class="muted">Checking…</div>';
  try {
    const r = await api(`/cluster/readiness?tp=${$('#clusterTpSel').value}`);
    const icon = { ok: '✅', warn: '⚠️', error: '❌' };
    box.innerHTML = r.checks.map((c) =>
      `<div class="wiz-check ${c.status}"><span>${icon[c.status]}</span><span>${escapeHtml(c.detail)}</span></div>`
      + (c.fix && c.status !== 'ok' ? `<div class="muted" style="margin-left:24px">↳ ${escapeHtml(c.fix)}</div>` : '')
    ).join('')
    + `<div class="wiz-note ${r.ok ? 'ok' : 'error'}" style="margin-top:8px">${r.ok
        ? `Ready for a TP ${r.tp} launch — pick a recipe on the Recipes tab.`
        : `Not ready for TP ${r.tp} — fix the ❌ items above.`}</div>`;
  } catch (e) { box.innerHTML = `<div class="wiz-note error">${escapeHtml(e.message)}</div>`; }
});

// ---------- LAN QR + update check -------------------------------------------
function openQrModal() {
  const urls = window._lanUrls || {};
  const target = (urls.lan || [])[0] || urls.local;
  if (!target || typeof qrcode === 'undefined') return;
  const qr = qrcode(0, 'M');
  qr.addData(target);
  qr.make();
  $('#qrCode').innerHTML = qr.createSvgTag({ cellSize: 5, margin: 2 });
  $('#qrUrls').innerHTML = [urls.local, ...(urls.lan || [])].filter(Boolean)
    .map((u) => `<div>${escapeHtml(u)}</div>`).join('');
  $('#qrModal').hidden = false;
}
$('#sidebarMeta')?.addEventListener('click', async () => {
  // With an update pending, the badge click offers the one-click update first.
  if (window._updateAvailable) {
    if (confirm(`Update to v${window._updateAvailable}?\n\nPulls the latest code and refreshes dependencies.`
        + (window._updateRestarts ? '\nThe service restarts itself; models keep serving.' : '\nRestart ./start.sh afterwards to finish.'))) {
      try {
        const r = await api('/update/apply', { method: 'POST', body: {} });
        toast(r.restarting ? 'Updated — restarting… (reload this page in ~10s)' : 'Updated — restart ./start.sh to finish', undefined);
        if (!r.restarting) window._updateAvailable = null;
      } catch (e) { toast(e.message, 'danger'); }
      return;
    }
  }
  openQrModal();
});
$$('#qrModal [data-close]').forEach((b) => b.addEventListener('click', () => ($('#qrModal').hidden = true)));

// One update check per page load, a few seconds after boot (git fetch is slow).
setTimeout(async () => {
  try {
    const u = await api('/update/check');
    if (u.update_available) {
      window._updateAvailable = u.latest_version || 'latest';
      window._updateRestarts = !!u.can_self_restart;
      refreshSystem(); // repaint the sidebar meta with the badge
      toast(`Update available: v${u.latest_version} (${u.behind_commits} commit${u.behind_commits === 1 ? '' : 's'} behind) — click the version in the sidebar`, undefined);
    }
  } catch { /* offline or not a git checkout — fine */ }
}, 6000);

// ---------- engine images (spark-vllm-docker runner) -------------------------
const _imgBuild = { active: false, lines: [] };
async function refreshEngineImages() {
  const box = $('#imagesList');
  if (!box) return;
  try {
    const d = await api('/images');
    if (d.error) { box.textContent = d.error; return; }
    box.innerHTML = d.images.map((im) => `
      <div class="img-row ${im.is_vllm_node ? 'is-node' : ''}" data-imgref="${escapeHtml(im.ref)}">
        <span class="mono">${escapeHtml(im.ref)}</span>
        ${im.is_vllm_node ? '<span class="badge ok" title="This is what docker recipes launch">← vllm-node</span>' : ''}
        <span class="muted">${escapeHtml(im.created)} · ${escapeHtml(im.size)}</span>
        <span class="img-vers muted mono">${im.versions
          ? (im.versions.error ? escapeHtml(im.versions.error)
             : `vLLM ${escapeHtml(im.versions.vllm)} · FlashInfer ${escapeHtml(im.versions.flashinfer)}`)
          : '<button class="btn" data-imgprobe>🔍 versions</button>'}</span>
      </div>`).join('') || '<span class="muted">No Spark vLLM images yet — run "Update to tested nightly".</span>';
    // Auto-probe the image recipes actually run.
    const node = d.images.find((im) => im.is_vllm_node);
    if (node && !node.versions) probeImage(node.ref);
    $$('#imagesList [data-imgprobe]').forEach((b) =>
      b.addEventListener('click', () => probeImage(b.closest('.img-row').dataset.imgref)));
  } catch (e) { box.textContent = e.message; }
  // Restore an in-flight/finished build log across tab switches.
  const log = $('#imgBuildLog');
  if (log && _imgBuild.lines.length) { log.hidden = false; log.textContent = _imgBuild.lines.join('\n'); log.scrollTop = log.scrollHeight; }
  $$('#engineImagesCard [data-imgbuild]').forEach((b) => (b.disabled = _imgBuild.active));
}
async function probeImage(ref) {
  const row = document.querySelector(`.img-row[data-imgref="${CSS.escape(ref)}"] .img-vers`);
  if (row) row.textContent = 'probing…';
  try {
    const v = await api(`/images/probe?ref=${encodeURIComponent(ref)}`);
    if (row) row.textContent = v.error ? v.error : `vLLM ${v.vllm} · FlashInfer ${v.flashinfer}`;
  } catch (e) { if (row) row.textContent = e.message; }
}
$$('#engineImagesCard [data-imgbuild]').forEach((b) => b.addEventListener('click', () => {
  if (_imgBuild.active) return;
  const mode = b.dataset.imgbuild;
  const flags = mode === 'advanced' ? ($('#imgFlags').value || '').trim() : '';
  if (mode === 'advanced' && !flags) { toast('Enter build flags first (e.g. --vllm-ref v0.24.0)', 'danger'); return; }
  if (!confirm(mode === 'nightly'
    ? 'Pull the tested nightly and retag vllm-node?\n\nUsually a few minutes. Running models keep serving their old image until relaunched.'
    : 'Start an image build?\n\nSource builds can take 30–60+ minutes. It keeps running even if you close this page.')) return;
  _imgBuild.active = true; _imgBuild.lines = [];
  refreshEngineImages();
  const log = $('#imgBuildLog'); log.hidden = false; log.textContent = '';
  const es = new EventSource(`/api/images/build?mode=${mode}&flags=${encodeURIComponent(flags)}`);
  es.addEventListener('log', (ev) => {
    _imgBuild.lines.push(ev.data);
    if (_imgBuild.lines.length > 800) _imgBuild.lines.shift();
    const l = $('#imgBuildLog');
    if (l) { l.hidden = false; l.textContent = _imgBuild.lines.join('\n'); l.scrollTop = l.scrollHeight; }
  });
  es.addEventListener('done', (ev) => {
    es.close();
    _imgBuild.active = false;
    const code = Number(ev.data);
    toast(code === 0 ? 'Image updated — relaunch recipes to use it' : `Image build exited ${code} — see the log`, code === 0 ? undefined : 'danger');
    refreshEngineImages();
  });
  es.addEventListener('error', () => {
    es.close();
    _imgBuild.active = false;
    _imgBuild.lines.push('[stream disconnected — the build continues server-side; revisit this tab to check]');
    refreshEngineImages();
  });
}));

// ---------- recovery page ---------------------------------------------------
function recoveryReport(html) { $('#recoveryResult').innerHTML = html; }
$$('[data-recover]').forEach((b) => b.addEventListener('click', async () => {
  const action = b.dataset.recover;
  b.disabled = true;
  recoveryReport('Working…');
  try {
    const r = await api(`/recovery/${action}`, { method: 'POST', body: {} });
    if (action === 'clear-runs') {
      recoveryReport(`Removed ${r.removed_from_list.length} finished run(s) from the list`
        + (r.stale_rows_closed ? ` · closed ${r.stale_rows_closed} stale database row(s)` : '') + '.');
      refreshRuns();
    } else if (action === 'clean-containers') {
      const rm = r.removed.length ? `Removed: ${r.removed.map(escapeHtml).join(', ')}` : 'Nothing to remove';
      const sk = (r.skipped || []).length
        ? `<br>Kept: ${r.skipped.map((s) => `${escapeHtml(s.name)} <span class="muted">(${escapeHtml(s.why)})</span>`).join(', ')}` : '';
      recoveryReport(rm + sk);
    } else if (action === 'reset-registry') {
      recoveryReport('Registry cache cleared — re-downloading community recipes in the background.');
    }
    toast('Done');
  } catch (e) { recoveryReport(''); toast(e.message, 'danger'); }
  b.disabled = false;
}));
$('#recoverResetDb')?.addEventListener('click', async () => {
  if (!confirm('Delete ALL saved recipes, run history, and benchmark history?\n\nDownloaded models are NOT deleted. A running model keeps serving.\n\nThere is no undo.')) return;
  try {
    const r = await api('/recovery/reset-db', { method: 'POST', body: { confirm: true } });
    recoveryReport('Database reset: ' + Object.entries(r.deleted).map(([t, n]) => `${t} ${n}`).join(' · '));
    toast('Database reset');
    refreshRecipes(); refreshRuns(); refreshOverview();
  } catch (e) { toast(e.message, 'danger'); }
});
async function copyBugReport(runId) {
  try {
    toast('Building bug report…');
    const r = await api(`/bugreport${runId ? `?run_id=${runId}` : ''}`);
    await copyText(r.markdown);
    toast('Bug report copied to clipboard 📋');
  } catch (e) { toast(`Bug report failed: ${e.message}`, 'danger'); }
}
$('#recoverBugReport')?.addEventListener('click', () => copyBugReport(null));
$('#logsBugReport')?.addEventListener('click', () => copyBugReport(currentRunId));

// ---------- first-run setup wizard ----------------------------------------
const wiz = {
  modal: null, goal: 'fastest', recs: null, pick: null, runId: null,
  watch: null, startedAt: 0,
};
function wizStep(n) {
  $$('#wizardModal .wiz-step').forEach((s) => (s.hidden = s.dataset.step !== String(n)));
}
function wizClose(markDone = true) {
  if (markDone) localStorage.setItem('wizardDone', '1');
  if (wiz.watch) { clearInterval(wiz.watch); wiz.watch = null; }
  $('#wizardModal').hidden = true;
}
async function openWizard() {
  $('#wizardModal').hidden = false;
  wizStep(1);
  $('#wizNext1').disabled = true;
  $('#wizChecks').innerHTML = '<div class="muted">Running checks…</div>';
  try {
    const rep = await api('/doctor');
    const icon = { ok: '✅', warn: '⚠️', error: '❌' };
    $('#wizChecks').innerHTML = rep.checks.map((c) => `
      <div class="wiz-check ${c.status}">
        <span>${icon[c.status] || '•'}</span>
        <span class="wiz-check-label">${escapeHtml(c.label)}</span>
        <span class="muted">${escapeHtml(c.detail)}</span>
      </div>`).join('');
    const s = rep.summary;
    const note = s.error
      ? `<div class="wiz-note error">⚠ ${s.error} core issue(s) — you can continue, but fixes above are recommended.</div>`
      : (s.warn ? `<div class="wiz-note">Optional features missing (${s.warn}) — everything else is ready.</div>`
                : '<div class="wiz-note ok">Everything looks great — your Spark is ready.</div>');
    $('#wizChecks').insertAdjacentHTML('beforeend', note);
  } catch (e) {
    $('#wizChecks').innerHTML = `<div class="wiz-note error">System check failed: ${escapeHtml(e.message)}</div>`;
  }
  $('#wizNext1').disabled = false;
}
function wizRecCard(e, hero) {
  const badges = [
    e.proven ? '<span class="badge ok">✓ ran on this Spark</span>' : '',
    e.cached ? '<span class="badge">💾 already downloaded</span>' : '',
    e.tokens_per_sec ? `<span class="badge">${Math.round(e.tokens_per_sec)} tok/s measured</span>` : '',
    e.est_weight_gb ? `<span class="badge">~${e.est_weight_gb} GB</span>` : '',
  ].filter(Boolean).join(' ');
  return `
    <div class="wiz-rec ${hero ? 'hero' : ''}" data-model="${escapeHtml(e.model)}">
      <div class="wiz-rec-name">${escapeHtml(e.model)}</div>
      <div class="wiz-rec-badges">${badges}</div>
      <div class="muted">${escapeHtml(e.reason)}</div>
    </div>`;
}
async function wizShowRecs() {
  wizStep(3);
  $('#wizRec').innerHTML = '<div class="muted">Finding the best model for your Spark…</div>';
  $('#wizAlts').innerHTML = '';
  try {
    wiz.recs = wiz.recs || await api('/recommend');
    const list = (wiz.recs.categories[wiz.goal] || []);
    if (!list.length) {
      $('#wizRec').innerHTML = '<div class="wiz-note">No ready-made match for that goal yet — the Recipe Forge can build one from any Hugging Face model id.</div>';
      $('#wizLaunch').disabled = true;
      return;
    }
    $('#wizLaunch').disabled = false;
    wiz.pick = list[0];
    $('#wizRec').innerHTML = '<div class="muted">Recommended for your Spark:</div>' + wizRecCard(list[0], true);
    if (list.length > 1) {
      $('#wizAlts').innerHTML = '<div class="muted">Alternatives:</div>'
        + list.slice(1).map((e) => wizRecCard(e, false)).join('');
    }
    $$('#wizardModal .wiz-rec').forEach((el) => el.addEventListener('click', () => {
      const m = el.dataset.model;
      wiz.pick = list.find((x) => x.model === m) || wiz.pick;
      $$('#wizardModal .wiz-rec').forEach((x) => x.classList.toggle('hero', x.dataset.model === m));
    }));
  } catch (e) {
    $('#wizRec').innerHTML = `<div class="wiz-note error">${escapeHtml(e.message)}</div>`;
  }
}
async function wizLaunch() {
  if (!wiz.pick) return;
  localStorage.setItem('wizardDone', '1'); // launching counts as onboarded
  wizStep(4);
  wiz.startedAt = Date.now();
  const status = $('#wizLaunchStatus');
  status.textContent = 'Saving recipe…';
  try {
    let recipe = wiz.pick.recipe;
    if (!recipe.id) recipe = await api('/recipes', { method: 'POST', body: recipe });
    status.textContent = 'Launching engine…';
    const body = (!recipe.args?._registry && recipe.raw_cmd)
      ? { engine: recipe.engine, raw_cmd: recipe.raw_cmd, env: recipe.env || {}, recipe_id: recipe.id }
      : { engine: recipe.engine, args: { model: recipe.model, ...(recipe.args || {}) }, env: recipe.env || {}, recipe_id: recipe.id };
    const run = await api('/runs', { method: 'POST', body });
    wiz.runId = run.id;
    refreshRecipes(); refreshRuns();
    wiz.watch = setInterval(async () => {
      try {
        const r = await api(`/runs/${wiz.runId}`);
        const secs = Math.round((Date.now() - wiz.startedAt) / 1000);
        if (r.ready) {
          clearInterval(wiz.watch); wiz.watch = null;
          const stats = [
            r.load_secs ? `Loaded in ${fmtDur(r.load_secs)}` : `Ready after ${secs}s`,
            r.ram_delta_gb != null ? `${r.ram_delta_gb >= 0 ? '+' : ''}${r.ram_delta_gb} GB RAM` : '',
            r.url ? `Endpoint: ${r.url}/v1` : '',
          ].filter(Boolean).join(' · ');
          $('#wizSuccess').innerHTML = `
            <div class="wiz-note ok">🎉 <b>${escapeHtml(wiz.pick.model)}</b> is serving.</div>
            <div class="muted">${escapeHtml(stats)}</div>`;
          wizStep(5);
        } else if (r.status === 'exited') {
          clearInterval(wiz.watch); wiz.watch = null;
          status.innerHTML = `<div class="wiz-note error">The launch failed (exit ${r.exit_code ?? '?'}).
            Auto-Fix can read the logs and patch the recipe for you.</div>`;
          $('#wizWatchLogs').textContent = 'Open Logs & Auto-Fix';
        } else {
          status.textContent = `Loading model… ${secs}s elapsed (status: ${r.status})`;
        }
      } catch { /* transient — keep polling */ }
    }, 3000);
  } catch (e) {
    status.innerHTML = `<div class="wiz-note error">${escapeHtml(e.message)}</div>`;
  }
}
function wizGotoRun(tab) {
  wizClose();
  $(`.tab[data-tab="${tab}"]`)?.click();
  if (wiz.runId && tab === 'logs') setTimeout(() => window.selectRun(wiz.runId), 80);
}
$('#openWizard')?.addEventListener('click', () => { wiz.recs = null; openWizard(); });
$('#wizSkip')?.addEventListener('click', () => wizClose(true));
$('#wizNext1')?.addEventListener('click', () => wizStep(2));
$$('#wizardModal .wiz-goal').forEach((b) => b.addEventListener('click', () => {
  wiz.goal = b.dataset.goal; wizShowRecs();
}));
$('#wizBack3')?.addEventListener('click', () => wizStep(2));
$('#wizLaunch')?.addEventListener('click', wizLaunch);
$('#wizWatchLogs')?.addEventListener('click', () => wizGotoRun('logs'));
$('#wizChat')?.addEventListener('click', () => wizGotoRun('chat'));
$('#wizBench')?.addEventListener('click', () => wizGotoRun('bench'));
$('#wizDone')?.addEventListener('click', () => wizClose(true));

// Fresh-install detection: default Beginner Mode + auto-open the wizard only
// when there is truly nothing here yet (no recipes, no runs, never dismissed).
(async function bootOnboarding() {
  const savedMode = localStorage.getItem('uiMode');
  try {
    const [recipes, runs] = await Promise.all([
      api('/recipes').catch(() => []),
      api('/runs').catch(() => []),
    ]);
    const fresh = !recipes.length && !runs.length;
    applyUiMode(savedMode || (fresh ? 'beginner' : 'advanced'));
    if (fresh && !localStorage.getItem('wizardDone')) openWizard();
  } catch {
    applyUiMode(savedMode || 'advanced');
  }
})();

// ---------- boot ---------------------------------------------------------
refreshSystem();
refreshOverview();
setInterval(refreshSystem, 15000);
setInterval(() => { if ($('.tab.active').dataset.tab === 'logs') refreshRuns(); }, 3000);
setInterval(() => { if ($('.tab.active').dataset.tab === 'overview') refreshOverview(); }, 5000);
