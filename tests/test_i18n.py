"""i18n system tests.

Covers the runtime (web/i18n.js), the static markup (web/index.html), the
dynamic call sites (web/app.js), and the FastAPI asset serving (server.py):

* EN/FR key parity (both dictionaries are strict JSON, parsed with
  json.JSONDecoder.raw_decode so braces inside string values are safe).
* Placeholder parity: every {var} in an EN value appears in the FR value.
* Every key referenced from index.html (data-i18n*) and app.js (t('ss.*'))
  exists in the EN dictionary.
* No known translated literal remains hardcoded in app.js.
* The language selector and the i18n script are wired in index.html.
* server.py version-busts i18n.js alongside app.js / style.css.
* The runtime itself: node --check on i18n.js, and (when node is available)
  a smoke test of window.t / ssLang / fallback / plural via a stubbed DOM.
"""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
REPO = Path(__file__).resolve().parent.parent
I18N_JS = WEB / "i18n.js"
INDEX_HTML = WEB / "index.html"
APP_JS = WEB / "app.js"
SERVER_PY = REPO / "server.py"

KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def load_dict(name: str) -> dict:
    """Extract the `name: { ... }` dictionary block from i18n.js.

    Uses json.JSONDecoder.raw_decode from the first '{' after the label, so
    braces inside JSON string values can never confuse the boundaries.
    """
    text = I18N_JS.read_text(encoding="utf-8")
    m = re.search(rf"\b{name}:\s*\{{", text)
    assert m, f"dictionary block '{name}' not found in i18n.js"
    start = text.index("{", m.start())
    obj, _end = json.JSONDecoder().raw_decode(text[start:])
    return obj


class I18nStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.en = load_dict("en")
        cls.fr = load_dict("fr")

    def test_dictionaries_are_non_empty(self):
        self.assertGreater(len(self.en), 100, "EN dictionary suspiciously small")
        self.assertGreater(len(self.fr), 100, "FR dictionary suspiciously small")

    def test_en_fr_key_parity(self):
        self.assertEqual(set(self.en), set(self.fr),
                         "EN and FR dictionaries must contain exactly the same keys")

    def test_no_empty_values(self):
        for lang, d in (("en", self.en), ("fr", self.fr)):
            empty = [k for k, v in d.items() if not isinstance(v, str) or not v.strip()]
            self.assertEqual(empty, [], f"empty values in {lang}: {empty}")

    def test_all_keys_match_key_shape(self):
        for d in (self.en, self.fr):
            bad = [k for k in d if not KEY_RE.match(k)]
            self.assertEqual(bad, [], f"keys not matching ss.domain.name shape: {bad}")

    def test_french_is_actually_translated(self):
        """Sanity: the FR dict must differ from EN on a large majority of keys,
        so a merge that copies EN over FR cannot slip through."""
        identical = [k for k, v in self.fr.items() if v == self.en[k]]
        ratio = len(identical) / len(self.fr)
        self.assertLess(ratio, 0.15,
                        f"{len(identical)} FR values identical to EN ({ratio:.0%}) — "
                        f"French looks un-translated: {identical[:5]}")

    def test_french_critical_strings(self):
        """Spot-check strings the closed PR #2 established as correct French."""
        expected = {
            "ss.nav.overview": "Aperçu",
            "ss.app.title": "Spark Studio",
        }
        for key, want in expected.items():
            self.assertEqual(self.fr.get(key), want, f"{key} unexpected FR value")

    def test_placeholder_parity(self):
        for key in self.en:
            en_vars = set(PLACEHOLDER_RE.findall(self.en[key]))
            fr_vars = set(PLACEHOLDER_RE.findall(self.fr[key]))
            self.assertEqual(en_vars, fr_vars,
                             f"{key}: placeholders differ EN={sorted(en_vars)} FR={sorted(fr_vars)}")

    def test_app_title_key_exists(self):
        self.assertIn("ss.app.title", self.en,
                      "i18n.js sets document.title from this key")


class I18nReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.en = load_dict("en")
        cls.index_html = INDEX_HTML.read_text(encoding="utf-8")
        cls.app_js = APP_JS.read_text(encoding="utf-8")

    def test_html_data_i18n_keys_exist_in_en(self):
        used = set()
        for attr in ("data-i18n", "data-i18n-title", "data-i18n-placeholder",
                     "data-i18n-aria", "data-i18n-value", "data-i18n-html"):
            used.update(re.findall(rf'{attr}="(ss\.[a-z0-9_.]+)"', self.index_html))
        missing = sorted(k for k in used if k not in self.en)
        self.assertEqual(missing, [], f"data-i18n keys missing from EN dict: {missing}")
        self.assertGreater(len(used), 50, "expected a substantial number of tagged elements")

    def test_app_js_t_call_keys_exist_in_en(self):
        used = set(re.findall(r"t\('(ss\.[a-z0-9_.]+)'", self.app_js))
        missing = sorted(k for k in used if k not in self.en)
        self.assertEqual(missing, [], f"t() keys missing from EN dict: {missing}")
        self.assertGreaterEqual(len(used), 30, "expected 30+ dynamic t() call sites")

    def test_no_stale_translated_literals_in_app_js(self):
        """Literals that were moved to the dictionary must be gone from app.js."""
        gone = [
            "toast('Recipe saved')",
            "toast('Database reset')",
            "toast('Invalid JSON', 'danger')",
            "toast('Popup blocked', 'danger')",
            "confirm('Delete recipe?')",
            "document.title = 'Spark Studio';",
            "title=\"Last run succeeded\"",
            "title=\"Click for update options\"",
            "title=\"Click to expand live vitals\"",
            "toast('Building bug report…')",
        ]
        found = [lit for lit in gone if lit in self.app_js]
        self.assertEqual(found, [], f"stale literals still hardcoded: {found}")

    def test_recipe_history_uses_i18n(self):
        """The history UI added on main must use the selected language."""
        required = (
            "ss.t.recipe_history_hint", "ss.t.file_history_hint",
            "ss.history.button", "ss.history.title.recipe",
            "ss.history.title.bundled", "ss.t.restored_recipe",
            "ss.t.restored_file", "ss.history.restore", "ss.history.empty",
            "ss.t.recipe_check",
        )
        used = set(re.findall(r"t\('(ss\.[a-z0-9_.]+)'", self.app_js))
        self.assertEqual(sorted(set(required) - used), [])
        stale = (
            "showHistoryModal('Recipe history'", "showHistoryModal(`Bundled recipe history",
            "toast(`Restored ", "toast(`Recipe check: ", ">⟲ History</button>",
            "}, '⟲ Restore')", "'No snapshots yet — one is taken automatically",
        )
        self.assertEqual([s for s in stale if s in self.app_js], [])

    def test_bench_sweep_uses_i18n(self):
        """The bench concurrency sweep added on main must use i18n."""
        required = (
            "ss.bench.sweep_running",
            "ss.bench.sweep.streams",
            "ss.bench.sweep.aggregate",
            "ss.bench.sweep.per_stream",
            "ss.bench.sweep.tokens",
            "ss.bench.sweep.failed",
            "ss.bench.sweep.footer",
        )
        used = set(re.findall(r"t\('(ss\.[a-z0-9_.]+)'", self.app_js))
        self.assertEqual(sorted(set(required) - used), [])
        stale = (
            "Concurrency sweep running",
            "<th>Streams</th><th>Aggregate tok/s</th>",
            "Aggregate = all streams combined",
        )
        self.assertEqual([s for s in stale if s in self.app_js], [])

    def test_hermes_reconnect_uses_i18n(self):
        """Hermes Chat reconnect/retry messages added on main must use i18n."""
        required = (
            "ss.t.hermes_https_interrupted",
            "ss.t.hermes_reconnected",
            "ss.t.hermes_phase_dropped",
            "ss.t.hermes_phase_reconnecting",
            "ss.t.hermes_phase_starting",
            "ss.t.hermes_retrying",
            "ss.t.hermes_connection_lost",
            "ss.t.hermes_ws_fallback",
            "ss.t.hermes_terminal_continued",
            "ss.t.hermes_session_ended",
            "ss.t.hermes_https_retrying",
            "ss.t.telemetry_reconnecting",
            "ss.t.telemetry_reconnecting_hint",
        )
        used = set(re.findall(r"t\('(ss\.[a-z0-9_.]+)'", self.app_js))
        self.assertEqual(sorted(set(required) - used), [])
        stale = (
            "HTTPS terminal interrupted — reconnecting:",
            "setHermesTuiState('Hermes reconnected over HTTPS'",
            "opened ? 'connection dropped'",
            "Lost the connection to Spark Studio — press Start to reconnect",
            "WebSocket unavailable — switching to secure HTTPS transport",
            "Previous terminal ended — picked up your last conversation.",
            "Previous session ended — started a new Hermes.",
            "HTTPS terminal is still starting — retrying in",
            "Telemetry reconnecting · model stays online",
        )
        self.assertEqual([s for s in stale if s in self.app_js], [])

    def test_language_selector_present(self):
        self.assertIn('id="langSelect"', self.index_html,
                      "language <select> missing from the sidebar")
        self.assertIn("fr", self.index_html)
        self.assertIn("en", self.index_html)

    def test_i18n_script_loaded_before_app_js(self):
        i18n_at = self.index_html.find("i18n.js")
        app_at = self.index_html.find('type="module" src="/static/app.js"')
        self.assertNotEqual(i18n_at, -1, "i18n.js script tag missing")
        self.assertNotEqual(app_at, -1, "app.js module script missing")
        self.assertLess(i18n_at, app_at,
                        "i18n.js must load before the deferred app.js module")


class I18nServerTests(unittest.TestCase):
    def test_server_version_busts_i18n_js(self):
        src = SERVER_PY.read_text(encoding="utf-8")
        m = re.search(r'for asset in \(([^)]+)\)', src)
        self.assertIsNotNone(m, "asset version-busting loop not found in server.py")
        assets = re.findall(r'"([\w.]+)"', m.group(1))
        for name in ("app.js", "style.css", "i18n.js"):
            self.assertIn(name, assets, f"{name} not in cache-busted assets")


class I18nRuntimeTests(unittest.TestCase):
    """Exercise web/i18n.js itself with Node, if available."""

    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which("node")

    def test_node_syntax_check(self):
        if not self.node:
            self.skipTest("node not installed")
        proc = subprocess.run([self.node, "--check", str(I18N_JS)],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0,
                         f"node --check failed: {proc.stderr}")

    def test_runtime_smoke(self):
        if not self.node:
            self.skipTest("node not installed")
        # Stub DOM + load the runtime, then probe window.t / ssLang.
        # Language switching in the runtime is by design a page reload
        # (ssSetLang persists to localStorage and location.reload()); so the
        # "reload in French" step pre-seeds localStorage and re-evaluates the
        # script, exactly like a fresh page load would.
        probe = r"""
globalThis.document = {
  readyState: 'complete',
  documentElement: { lang: '' },
  querySelectorAll: () => [],
  getElementById: () => null,
  addEventListener: () => {},
  title: '',
};
globalThis.window = globalThis;
globalThis.localStorage = { _s: {}, getItem(k){ return this._s[k] ?? null; }, setItem(k, v){ this._s[k] = v; } };
globalThis.navigator = { language: 'en-US' };
const src = require('fs').readFileSync(process.argv[1], 'utf8');
eval(src); // fresh load, no stored language -> en
// EN by default.
console.log(JSON.stringify({
  lang: window.ssLang,
  overview: window.t('ss.nav.overview'),
  title: window.t('ss.app.title'),
  missing: window.t('zzz.missing'),
  vars: window.t('ss.t.started_run', { id: 'abc123' }),
}));
// Simulate the reload-after-switch: persist 'fr' then re-load the runtime.
localStorage.setItem('ss_lang', 'fr');
eval(src);
console.log(JSON.stringify({
  lang: window.ssLang,
  overview: window.t('ss.nav.overview'),
  plural2: window.t('ss.t.models_on_disk', { n: 3, gb: 12 }),
  plural1: window.t('ss.t.models_on_disk', { n: 1, gb: 8 }),
}));
"""
        proc = subprocess.run(
            [self.node, "-e", probe, str(I18N_JS)],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, f"runtime probe failed: {proc.stderr}")
        out = proc.stdout.strip().splitlines()
        en_probe = json.loads(out[0])
        fr_probe = json.loads(out[1])
        self.assertEqual(en_probe["lang"], "en")
        self.assertEqual(en_probe["overview"], "Overview")
        self.assertEqual(en_probe["title"], "Spark Studio")
        self.assertEqual(en_probe["missing"], "zzz.missing",
                         "unknown key must fall back to the key itself")
        self.assertIn("abc123", en_probe["vars"])
        self.assertEqual(fr_probe["overview"], "Aperçu")
        # {plural} derivation: fr 1 => "modèle", 3 => "modèles".
        self.assertIn("modèle ·", fr_probe["plural1"])
        self.assertIn("modèles ·", fr_probe["plural2"])


if __name__ == "__main__":
    unittest.main()
