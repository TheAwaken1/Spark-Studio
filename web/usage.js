// Spark Usage tab: what the engines are serving right now and what they have
// served. Polls the /api/usage/* endpoints only while the tab is open.

const $u = (sel, root = document) => root.querySelector(sel);

function usageEsc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}

function usageFmt(n, digits = 0) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function usagePct(n) {
  return n === null || n === undefined ? '—' : `${usageFmt(n, 1)}%`;
}

function usageTime(epochSecs) {
  return new Date(epochSecs * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function usageDuration(secs) {
  if (secs === null || secs === undefined) return '—';
  if (secs < 60) return `${usageFmt(secs, 1)} s`;
  return `${Math.floor(secs / 60)} m ${usageFmt(secs % 60, 0)} s`;
}

function usageMiniSpark(values, opts = {}) {
  const vals = (values || []).map((v) => (v == null ? 0 : Number(v)));
  if (!vals.length) return '';
  const w = opts.w || 160;
  const h = opts.h || 36;
  const max = Math.max(1, ...vals);
  const step = vals.length > 1 ? w / (vals.length - 1) : w;
  const pts = vals.map((v, i) => `${(i * step).toFixed(1)},${(h - (v / max) * (h - 4) - 2).toFixed(1)}`).join(' ');
  return `<svg class="usage-mini" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="${opts.label || 'trend'}">
      <polyline points="${pts}" class="usage-spark-line"></polyline></svg>`;
}

// Real compute, derived from the totals above: a cached prompt token was not
// prefilled again, so only the uncached part and the generated tokens cost GPU work.
function usageActualCards(t) {
  const prompt = Number(t.prompt_tokens) || 0;
  const cached = Number(t.cached_tokens) || 0;
  const completion = Number(t.completion_tokens) || 0;
  const uncached = Math.max(prompt - cached, 0);
  const cachePct = prompt ? (100 * cached) / prompt : null;
  const uncachedPct = prompt ? (100 * uncached) / prompt : null;
  const card = (label, value, hint) =>
    `<div class="usage-actual-card"><div class="usage-tile-label">${label}</div>
       <div class="usage-actual-value">${value}</div><div class="muted usage-tile-hint">${hint}</div></div>`;
  return `<div class="usage-actual-head"><h3><i class="fa-solid fa-microchip"></i> Actual compute</h3>
      <span class="muted">Real GPU work for this period, not the raw prompt total</span></div>
    <div class="usage-actual-grid">
      ${card('Actual prefill', `${usageFmt(uncached)} <span class="muted">tokens</span>`,
        `uncached prompt tokens · ${usagePct(uncachedPct)} of ${usageFmt(prompt)} prompt`)}
      ${card('Actual decode', `${usageFmt(completion)} <span class="muted">tokens</span>`, 'generated tokens')}
      ${card('Total compute', `${usageFmt(uncached + completion)} <span class="muted">tokens</span>`,
        'prefill + decode')}
      ${card('Cache savings', `${usageFmt(cached)} <span class="muted">tokens</span>`,
        cachePct == null ? 'no prompt tokens yet' : `${usagePct(cachePct)} of prompt tokens avoided by cache`)}
    </div>`;
}

let usagePeriod = 'session';
let usageTimer = null;
let usageTimer2 = null;
let usageEvents = [];
let usageMoreCursor = null;
let usageWired = false;

function usageWire() {
  if (usageWired) return;
  usageWired = true;
  $u('#usagePeriods').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-period]');
    if (!btn) return;
    usagePeriod = btn.dataset.period;
    for (const b of $u('#usagePeriods').querySelectorAll('[data-period]')) {
      b.classList.toggle('active', b === btn);
    }
    usageLoadAll(true);
  });
  $u('#usageMore').addEventListener('click', () => usageLoadEvents(false));
  $u('#usageImport').addEventListener('click', async () => {
    const btn = $u('#usageImport');
    btn.disabled = true;
    try {
      const r = await fetch('/api/usage/backfill', { method: 'POST' });
      const counts = await r.json();
      $u('#usageImportNote').textContent = `Imported ${usageFmt(counts.hermes_sessions)} Hermes rows and ${usageFmt(counts.engine_samples)} engine samples.`;
      usageLoadAll(true);
    } catch {
      $u('#usageImportNote').textContent = 'Import failed. Check the Studio log.';
    } finally {
      btn.disabled = false;
    }
  });
  $u('#usageModal').addEventListener('click', (e) => {
    if (e.target.id === 'usageModal' || e.target.closest('[data-close]')) $u('#usageModal').hidden = true;
  });
}

export function startUsageTab() {
  usageWire();
  usageLoadAll(true);
  stopUsageTab();
  usageTimer = setInterval(() => usageLive(), 1500);
  // Totals and the log change less often; refresh them on a slower beat.
  usageTimer2 = setInterval(() => usageLoadAll(false), 6000);
}

export function stopUsageTab() {
  if (usageTimer) clearInterval(usageTimer);
  if (usageTimer2) clearInterval(usageTimer2);
  usageTimer = null;
  usageTimer2 = null;
}

async function usageJson(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

async function usageLoadAll(reset) {
  await Promise.all([usageLive(), usageSummary(), usageLoadEvents(reset), usageTools()]);
}

async function usageTools() {
  let data;
  try {
    data = await usageJson(`/api/usage/tools?period=${usagePeriod}`);
  } catch {
    return;
  }
  const byName = data.by_name || [];
  $u('#usageToolsByName').innerHTML = byName.length
    ? `<table class="usage-table"><thead><tr><th>Tool</th><th>Calls</th><th>Requests</th><th>Avg args</th></tr></thead><tbody>${
        byName.map((r) => `<tr><td class="mono">${usageEsc(r.name)}</td><td>${usageFmt(r.calls)}</td><td>${usageFmt(r.requests)}</td><td>${usageFmt(r.avg_args_chars)} chars</td></tr>`).join('')
      }</tbody></table>`
    : '<p class="muted">No tool calls in this period.</p>';
  const recent = data.recent || [];
  $u('#usageToolsRecent').innerHTML = recent.length
    ? `<table class="usage-table"><thead><tr><th>When</th><th>Machine</th><th>Tool</th></tr></thead><tbody>${
        recent.slice(0, 20).map((r) => `<tr><td>${usageTime(r.ts)}</td><td>${usageEsc(r.node)}</td><td class="mono">${usageEsc(r.name)}</td></tr>`).join('')
      }</tbody></table>`
    : '<p class="muted">No recent tool calls.</p>';
}

async function usageLive() {
  let live;
  try {
    live = await usageJson('/api/usage/live');
  } catch {
    return;
  }
  $u('#usageStreamsNow').textContent = usageFmt(live.streams_now);
  $u('#usagePeakStreams').textContent = usageFmt(live.session_peak_streams);
  $u('#usageTokNow').textContent = live.tok_s_now == null ? '—' : usageFmt(live.tok_s_now, 1);
  $u('#usagePeakTok').textContent = usageFmt(live.session_peak_tok_s, 1);
  const eng = live.engine;
  $u('#usageEngine').innerHTML = eng
    ? `<span class="muted">Engine</span>
       <span class="mono">${usageFmt(eng.running)}</span> running ·
       <span class="mono">${usageFmt(eng.waiting)}</span> waiting ·
       cache <span class="mono">${usagePct(eng.cache_hit_pct)}</span>`
    : '<span class="muted">No engine metrics yet.</span>';
  const list = $u('#usageLiveList');
  if (!live.streams.length) {
    list.innerHTML = '<p class="muted">Nothing generating right now. Requests from this box and your other machines appear here while they run.</p>';
    return;
  }
  list.innerHTML = live.streams.map((s) => {
    const phase = s.phase === 'thinking'
      ? '<span class="usage-badge think"><i class="fa-solid fa-brain"></i> thinking</span>'
      : s.phase === 'answering'
        ? '<span class="usage-badge answer"><i class="fa-solid fa-pen"></i> answering</span>'
        : '<span class="usage-badge"><i class="fa-solid fa-hourglass-half"></i> waiting</span>';
    return `<div class="usage-live-row">
      <div class="usage-live-who"><strong>${usageEsc(s.node)}</strong> <span class="muted mono">${usageEsc(s.model || 'model')}</span></div>
      ${phase}
      <div class="usage-live-rate">${s.tok_s == null ? '<span class="muted">counting…</span>' : `<span class="mono">${usageFmt(s.tok_s, 1)}</span> tok/s`}</div>
      <div class="usage-live-tools">${(s.tools || []).map((n) => `<span class="usage-badge"><i class="fa-solid fa-wrench"></i> ${usageEsc(n)}</span>`).join('')}</div>
      <div class="usage-live-trend">${usageMiniSpark(s.recent, { w: 200, h: 34, label: 'stream activity, last 40 seconds' })}</div>
      <div class="muted">${s.tokens_exact ? '' : '≈ '}${usageFmt(s.tokens_so_far)} tokens · ${usageDuration(s.elapsed)}</div>
    </div>`;
  }).join('');
}

async function usageSummary() {
  let data;
  try {
    data = await usageJson(`/api/usage/summary?period=${usagePeriod}`);
  } catch {
    return;
  }
  const t = data.totals;
  $u('#usageActual').innerHTML = usageActualCards(t);
  const tiles = [
    ['Requests', usageFmt(t.requests), `${usageFmt(t.failed_requests)} failed`],
    ['Prompt tokens', usageFmt(t.prompt_tokens), 'in'],
    ['Completion tokens', usageFmt(t.completion_tokens), 'out'],
    ['Cache hit', usagePct(t.cache_hit_pct), `${usageFmt(t.cached_tokens)} cached tokens`],
    ['Avg tok/s', usageFmt(t.avg_tok_s, 1), 'exact tokens ÷ time to last token'],
    ['Peak tok/s', usageFmt(t.peak_tok_s, 1), 'fastest single request'],
    ['Thinking', t.requests ? usagePct((100 * t.thinking_requests) / t.requests) : '—', `${usageFmt(t.thinking_requests)} requests`],
    ['Peak streams', usageFmt(t.peak_streams), 'at once'],
  ];
  $u('#usageTiles').innerHTML = tiles.map(([label, value, hint]) =>
    `<div class="usage-tile"><div class="usage-tile-label">${label}</div><div class="usage-tile-value">${value}</div><div class="muted usage-tile-hint">${usageEsc(hint)}</div></div>`
  ).join('');

  const table = (rows, key, head) => {
    if (!rows.length) return '<p class="muted">No requests in this period.</p>';
    return `<table class="usage-table"><thead><tr><th>${head}</th><th>Requests</th><th>Completion</th><th>Avg tok/s</th><th>Peak tok/s</th><th>Cache</th></tr></thead><tbody>${
      rows.map((r) => `<tr>
        <td>${usageEsc(r[key] || '—')}</td>
        <td>${usageFmt(r.requests)}</td>
        <td>${usageFmt(r.completion_tokens)}</td>
        <td>${usageFmt(r.avg_tok_s, 1)}</td>
        <td>${usageFmt(r.peak_tok_s, 1)}</td>
        <td>${usagePct(r.cache_hit_pct)}</td>
      </tr>`).join('')
    }</tbody></table>`;
  };
  $u('#usageByNode').innerHTML = table(data.by_node, 'node', 'Machine');
  $u('#usageByModel').innerHTML = table(data.by_model, 'model', 'Model');
}

async function usageLoadEvents(reset) {
  if (reset) {
    usageEvents = [];
    usageMoreCursor = null;
  }
  let rows;
  try {
    const before = usageMoreCursor ? `&before=${usageMoreCursor}` : '';
    rows = await usageJson(`/api/usage/events?period=${usagePeriod}&limit=50${before}`);
  } catch {
    return;
  }
  if (reset) {
    usageEvents = rows;
  } else {
    const seen = new Set(usageEvents.map((e) => e.id));
    usageEvents = usageEvents.concat(rows.filter((r) => !seen.has(r.id)));
  }
  usageMoreCursor = rows.length ? rows[rows.length - 1].started_at : null;
  $u('#usageMore').hidden = rows.length < 50;
  const log = $u('#usageLog');
  if (!usageEvents.length) {
    log.innerHTML = '<p class="muted">No requests yet in this period.</p>';
    return;
  }
  log.innerHTML = `<table class="usage-table usage-log"><thead><tr>
      <th>When</th><th>Machine</th><th>Model</th><th>Endpoint</th><th>Prompt</th><th>Completion</th><th>Cache</th><th>Tok/s</th><th></th>
    </tr></thead><tbody>${
      usageEvents.map((e) => `<tr class="usage-row ${e.status !== 'ok' ? 'failed' : ''}" data-id="${usageEsc(e.id)}">
        <td>${usageTime(e.started_at)}</td>
        <td>${usageEsc(e.node)}</td>
        <td class="mono">${usageEsc(e.model || '—')}${String(e.engine_run_id || '').startsWith('backfill') ? ' <span class="usage-badge">imported</span>' : ''}</td>
        <td class="muted mono">${usageEsc(e.endpoint)}</td>
        <td>${usageFmt(e.prompt_tokens)}</td>
        <td>${usageFmt(e.completion_tokens)}</td>
        <td>${usagePct(e.cache_hit_pct)}</td>
        <td>${usageFmt(e.tok_s, 1)}</td>
        <td>${e.thinking ? '<span class="usage-badge think"><i class="fa-solid fa-brain"></i></span>' : ''}${e.status !== 'ok' ? '<span class="usage-badge fail">' + usageEsc(e.status) + '</span>' : ''}</td>
      </tr>`).join('')
    }</tbody></table>`;
  for (const tr of log.querySelectorAll('tr.usage-row')) {
    tr.addEventListener('click', () => usageOpenDetail(tr.dataset.id));
  }
}

function usageSparkline(samples) {
  if (!samples || !samples.length) return '<p class="muted">No activity recorded for this request.</p>';
  const w = 600;
  const h = 120;
  const max = Math.max(1, ...samples);
  const step = samples.length > 1 ? w / (samples.length - 1) : w;
  const pts = samples.map((v, i) => `${(i * step).toFixed(1)},${(h - (v / max) * (h - 8) - 4).toFixed(1)}`).join(' ');
  const area = `0,${h} ${pts} ${(samples.length - 1) * step},${h}`;
  return `<svg class="usage-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="tokens per second over time">
      <polygon points="${area}" class="usage-spark-area"></polygon>
      <polyline points="${pts}" class="usage-spark-line"></polyline>
    </svg>
    <div class="muted usage-spark-axis"><span>0 s</span><span>max ${usageFmt(max)} chunks/s</span><span>${samples.length} s</span></div>`;
}

async function usageOpenDetail(id) {
  let e;
  try {
    e = await usageJson(`/api/usage/events/${encodeURIComponent(id)}`);
  } catch {
    return;
  }
  const facts = [
    ['Machine', `${e.node}${e.client_ip ? ` (${e.client_ip})` : ''}`],
    ['Model', e.model || '—'],
    ['Endpoint', e.endpoint],
    ['Started', usageTime(e.started_at)],
    ['Duration', usageDuration((e.ended_at || e.started_at) - e.started_at)],
    ['Status', e.error ? `${e.status}: ${e.error}` : e.status],
    ['Mode', e.stream ? 'streamed' : 'single response'],
    ['Thinking', e.thinking ? 'yes' : 'no'],
    ['Streams on this engine', usageFmt(e.concurrency)],
    ['Prompt tokens', usageFmt(e.prompt_tokens)],
    ['Completion tokens', usageFmt(e.completion_tokens)],
    ['Cached prompt tokens', usageFmt(e.cached_tokens)],
    ['Cache hit', usagePct(e.cache_hit_pct)],
    ['Time to first token', e.ttft_ms == null ? '—' : `${usageFmt(e.ttft_ms)} ms`],
    ['Decode time', usageDuration(e.decode_secs)],
    ['Tok/s (exact tokens ÷ time from first token to end)', usageFmt(e.tok_s, 1)],
  ];
  $u('#usageModalTitle').textContent = `Request ${e.id}`;
  $u('#usageModalBody').innerHTML = `
    <dl class="usage-facts">${facts.map(([k, v]) => `<dt>${usageEsc(k)}</dt><dd>${usageEsc(v)}</dd>`).join('')}</dl>
    <h4>Stream activity</h4>
    ${usageSparkline(e.samples)}`;
  $u('#usageModal').hidden = false;
}
