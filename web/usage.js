// Spark Usage tab: what the engines are serving right now and what they have
// served. Polls the /api/usage/* endpoints only while the tab is open.

const $u = (sel, root = document) => root.querySelector(sel);

// ---------- i18n -----------------------------------------------------------
// usage.js is an ES module; i18n.js is a classic script loaded before it, so
// window.t is normally available. Fall back to the key itself if it did not
// load, so a missing i18n.js degrades to English-ish keys instead of throwing.
const t = (typeof window !== 'undefined' && typeof window.t === 'function')
  ? window.t
  : (key) => key;


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
  return `<div class="usage-actual-head"><h3><i class="fa-solid fa-microchip"></i> ${t('ss.usage.actual_compute')}</h3>
      <span class="muted">${t('ss.usage.actual_compute_hint')}</span></div>
    <div class="usage-actual-grid">
      ${card(t('ss.usage.actual_prefill'), `${usageFmt(uncached)} <span class="muted">${t('ss.usage.unit_tokens')}</span>`,
        t('ss.usage.actual_prefill_hint', { pct: usagePct(uncachedPct), total: usageFmt(prompt) }))}
      ${card(t('ss.usage.actual_decode'), `${usageFmt(completion)} <span class="muted">${t('ss.usage.unit_tokens')}</span>`, t('ss.usage.actual_decode_hint'))}
      ${card(t('ss.usage.actual_total'), `${usageFmt(uncached + completion)} <span class="muted">${t('ss.usage.unit_tokens')}</span>`,
        t('ss.usage.actual_total_hint'))}
      ${card(t('ss.usage.cache_savings'), `${usageFmt(cached)} <span class="muted">${t('ss.usage.unit_tokens')}</span>`,
        cachePct == null ? t('ss.usage.no_prompt_tokens_yet') : t('ss.usage.cache_savings_hint', { pct: usagePct(cachePct) }))}
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
      $u('#usageImportNote').textContent = t('ss.usage.import_done', {
        hermes: usageFmt(counts.hermes_sessions),
        engine: usageFmt(counts.engine_samples),
      });
      usageLoadAll(true);
    } catch {
      $u('#usageImportNote').textContent = t('ss.usage.import_failed');
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
    ? `<table class="usage-table"><thead><tr><th>${t('ss.usage.th_tool')}</th><th>${t('ss.usage.th_calls')}</th><th>${t('ss.usage.th_requests')}</th><th>${t('ss.usage.th_avg_args')}</th></tr></thead><tbody>${
        byName.map((r) => `<tr><td class="mono">${usageEsc(r.name)}</td><td>${usageFmt(r.calls)}</td><td>${usageFmt(r.requests)}</td><td>${t('ss.usage.unit_chars', { n: usageFmt(r.avg_args_chars) })}</td></tr>`).join('')
      }</tbody></table>`
    : `<p class="muted">${t('ss.usage.no_tool_calls')}</p>`;
  const recent = data.recent || [];
  $u('#usageToolsRecent').innerHTML = recent.length
    ? `<table class="usage-table"><thead><tr><th>${t('ss.usage.th_when')}</th><th>${t('ss.usage.th_machine')}</th><th>${t('ss.usage.th_tool')}</th></tr></thead><tbody>${
        recent.slice(0, 20).map((r) => `<tr><td>${usageTime(r.ts)}</td><td>${usageEsc(r.node)}</td><td class="mono">${usageEsc(r.name)}</td></tr>`).join('')
      }</tbody></table>`
    : `<p class="muted">${t('ss.usage.no_recent_tool_calls')}</p>`;
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
    ? `<span class="muted">${t('ss.usage.engine')}</span>
       <span class="mono">${usageFmt(eng.running)}</span> ${t('ss.usage.running')} ·
       <span class="mono">${usageFmt(eng.waiting)}</span> ${t('ss.usage.waiting')} ·
       ${t('ss.usage.cache')} <span class="mono">${usagePct(eng.cache_hit_pct)}</span>`
    : `<span class="muted">${t('ss.usage.no_engine_metrics')}</span>`;
  const list = $u('#usageLiveList');
  if (!live.streams.length) {
    list.innerHTML = `<p class="muted">${t('ss.usage.nothing_generating')}</p>`;
    return;
  }
  list.innerHTML = live.streams.map((s) => {
    const phase = s.phase === 'thinking'
      ? `<span class="usage-badge think"><i class="fa-solid fa-brain"></i> ${t('ss.usage.phase_thinking')}</span>`
      : s.phase === 'answering'
        ? `<span class="usage-badge answer"><i class="fa-solid fa-pen"></i> ${t('ss.usage.phase_answering')}</span>`
        : `<span class="usage-badge"><i class="fa-solid fa-hourglass-half"></i> ${t('ss.usage.phase_waiting')}</span>`;
    return `<div class="usage-live-row">
      <div class="usage-live-who"><strong>${usageEsc(s.node)}</strong> <span class="muted mono">${usageEsc(s.model || t('ss.usage.model'))}</span></div>
      ${phase}
      <div class="usage-live-rate">${s.tok_s == null ? `<span class="muted">${t('ss.usage.counting')}</span>` : `<span class="mono">${usageFmt(s.tok_s, 1)}</span> ${t('ss.usage.unit_tok_s', { n: '' }).trim()}`}</div>
      <div class="usage-live-tools">${(s.tools || []).map((n) => `<span class="usage-badge"><i class="fa-solid fa-wrench"></i> ${usageEsc(n)}</span>`).join('')}</div>
      <div class="usage-live-trend">${usageMiniSpark(s.recent, { w: 200, h: 34, label: t('ss.usage.trend_aria') })}</div>
      <div class="muted">${s.tokens_exact ? '' : '≈ '}${usageFmt(s.tokens_so_far)} ${t('ss.usage.unit_tokens')} · ${usageDuration(s.elapsed)}</div>
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
    [t('ss.usage.tile_requests'), usageFmt(t.requests), t('ss.usage.tile_failed', { n: usageFmt(t.failed_requests) })],
    [t('ss.usage.tile_prompt_tokens'), usageFmt(t.prompt_tokens), t('ss.usage.tile_in')],
    [t('ss.usage.tile_completion_tokens'), usageFmt(t.completion_tokens), t('ss.usage.tile_out')],
    [t('ss.usage.tile_cache_hit'), usagePct(t.cache_hit_pct), t('ss.usage.tile_cached_tokens', { n: usageFmt(t.cached_tokens) })],
    [t('ss.usage.tile_avg_tok_s'), usageFmt(t.avg_tok_s, 1), t('ss.usage.tile_avg_tok_s_hint')],
    [t('ss.usage.tile_peak_tok_s'), usageFmt(t.peak_tok_s, 1), t('ss.usage.tile_peak_tok_s_hint')],
    [t('ss.usage.tile_thinking'), t.requests ? usagePct((100 * t.thinking_requests) / t.requests) : '—', t('ss.usage.tile_requests_n', { n: usageFmt(t.thinking_requests) })],
    [t('ss.usage.tile_peak_streams'), usageFmt(t.peak_streams), t('ss.usage.tile_at_once')],
  ];
  $u('#usageTiles').innerHTML = tiles.map(([label, value, hint]) =>
    `<div class="usage-tile"><div class="usage-tile-label">${label}</div><div class="usage-tile-value">${value}</div><div class="muted usage-tile-hint">${usageEsc(hint)}</div></div>`
  ).join('');

  const table = (rows, key, head) => {
    if (!rows.length) return `<p class="muted">${t('ss.usage.no_requests_period')}</p>`;
    return `<table class="usage-table"><thead><tr><th>${head}</th><th>${t('ss.usage.th_requests')}</th><th>${t('ss.usage.th_completion')}</th><th>${t('ss.usage.th_tok_s')}</th><th>${t('ss.usage.tile_peak_tok_s')}</th><th>${t('ss.usage.th_cache')}</th></tr></thead><tbody>${
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
  $u('#usageByNode').innerHTML = table(data.by_node, 'node', t('ss.usage.by_machine'));
  $u('#usageByModel').innerHTML = table(data.by_model, 'model', t('ss.usage.by_model'));
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
    log.innerHTML = `<p class="muted">${t('ss.usage.no_requests_yet')}</p>`;
    return;
  }
  log.innerHTML = `<table class="usage-table usage-log"><thead><tr>
      <th>${t('ss.usage.th_when')}</th><th>${t('ss.usage.th_machine')}</th><th>${t('ss.usage.th_model')}</th><th>${t('ss.usage.th_endpoint')}</th><th>${t('ss.usage.th_prompt')}</th><th>${t('ss.usage.th_completion')}</th><th>${t('ss.usage.th_cache')}</th><th>${t('ss.usage.th_tok_s')}</th><th></th>
    </tr></thead><tbody>${
      usageEvents.map((e) => `<tr class="usage-row ${e.status !== 'ok' ? 'failed' : ''}" data-id="${usageEsc(e.id)}">
        <td>${usageTime(e.started_at)}</td>
        <td>${usageEsc(e.node)}</td>
        <td class="mono">${usageEsc(e.model || '—')}${String(e.engine_run_id || '').startsWith('backfill') ? ` <span class="usage-badge">${t('ss.usage.imported')}</span>` : ''}</td>
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
  if (!samples || !samples.length) return `<p class="muted">${t('ss.usage.no_activity')}</p>`;
  const w = 600;
  const h = 120;
  const max = Math.max(1, ...samples);
  const step = samples.length > 1 ? w / (samples.length - 1) : w;
  const pts = samples.map((v, i) => `${(i * step).toFixed(1)},${(h - (v / max) * (h - 8) - 4).toFixed(1)}`).join(' ');
  const area = `0,${h} ${pts} ${(samples.length - 1) * step},${h}`;
  return `<svg class="usage-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="${t('ss.usage.sparkline_aria')}">
      <polygon points="${area}" class="usage-spark-area"></polygon>
      <polyline points="${pts}" class="usage-spark-line"></polyline>
    </svg>
    <div class="muted usage-spark-axis"><span>${t('ss.usage.axis_zero')}</span><span>${t('ss.usage.axis_max', { n: usageFmt(max) })}</span><span>${t('ss.usage.axis_seconds', { n: samples.length })}</span></div>`;
}

async function usageOpenDetail(id) {
  let e;
  try {
    e = await usageJson(`/api/usage/events/${encodeURIComponent(id)}`);
  } catch {
    return;
  }
  const facts = [
    [t('ss.usage.fact_machine'), `${e.node}${e.client_ip ? ` (${e.client_ip})` : ''}`],
    [t('ss.usage.fact_model'), e.model || '—'],
    [t('ss.usage.fact_endpoint'), e.endpoint],
    [t('ss.usage.fact_started'), usageTime(e.started_at)],
    [t('ss.usage.fact_duration'), usageDuration((e.ended_at || e.started_at) - e.started_at)],
    [t('ss.usage.fact_status'), e.error ? `${e.status}: ${e.error}` : e.status],
    [t('ss.usage.fact_mode'), e.stream ? t('ss.usage.mode_streamed') : t('ss.usage.mode_single')],
    [t('ss.usage.fact_thinking'), e.thinking ? t('ss.usage.yes') : t('ss.usage.no')],
    [t('ss.usage.fact_streams'), usageFmt(e.concurrency)],
    [t('ss.usage.fact_prompt_tokens'), usageFmt(e.prompt_tokens)],
    [t('ss.usage.fact_completion_tokens'), usageFmt(e.completion_tokens)],
    [t('ss.usage.fact_cached_tokens'), usageFmt(e.cached_tokens)],
    [t('ss.usage.fact_cache_hit'), usagePct(e.cache_hit_pct)],
    [t('ss.usage.fact_ttft'), e.ttft_ms == null ? '—' : `${usageFmt(e.ttft_ms)} ms`],
    [t('ss.usage.fact_decode'), usageDuration(e.decode_secs)],
    [t('ss.usage.fact_tok_s'), usageFmt(e.tok_s, 1)],
  ];
  $u('#usageModalTitle').textContent = t('ss.usage.request_title', { id: e.id });
  $u('#usageModalBody').innerHTML = `
    <dl class="usage-facts">${facts.map(([k, v]) => `<dt>${usageEsc(k)}</dt><dd>${usageEsc(v)}</dd>`).join('')}</dl>
    <h4>${t('ss.usage.stream_activity')}</h4>
    ${usageSparkline(e.samples)}`;
  $u('#usageModal').hidden = false;
}
