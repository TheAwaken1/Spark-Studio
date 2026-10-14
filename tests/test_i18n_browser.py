"""Real-browser regression for the static i18n pass (EN and FR).

applyStaticI18n() rewrites the textContent of every [data-i18n] element.
Markup mistakes (data-i18n on an element that wraps controls, unclosed
<i> icons) only show up in a real DOM: the unit suite stays green while the
Engine Chat sliders vanish and sending fails with
"Cannot read properties of null (reading 'value')".

This test loads web/index.html + the real static assets in headless
Chromium, for each UI language, and checks that:

* every id present in the markup still exists after i18n ran;
* the Engine Chat sliders are still range inputs, with a translated label;
* no stray <i> icon was re-parented under <body>;
* sending a message in Engine Chat posts temperature/max_tokens taken from
  the sliders and renders the (stubbed) streamed reply, with no page error.

Only POST /api/chat is answered (by a local stub); every other network
request is aborted, so nothing leaves the machine.

Optional: skipped when Playwright or a Chromium binary is unavailable (the
GitHub CI image has neither). Point SPARK_STUDIO_TEST_CHROMIUM at a Chromium
executable to use a specific binary.
"""

import json
import mimetypes
import os
import re
import unittest
from pathlib import Path
from urllib.parse import urlsplit

WEB = Path(__file__).resolve().parent.parent / "web"
ORIGIN = "http://studio.test"
ID_RE = re.compile(r'\sid="([^"]+)"')

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - depends on the environment
    sync_playwright = None


def _fr_value(key: str) -> str:
    try:  # `discover -s tests` puts tests/ itself on sys.path
        from test_i18n import load_dict
    except ImportError:  # run as tests.test_i18n_browser from the repo root
        from tests.test_i18n import load_dict
    return load_dict("fr")[key]


@unittest.skipIf(sync_playwright is None, "playwright not installed")
class StaticI18nBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._pw = sync_playwright().start()
        kwargs = {"headless": True, "args": ["--no-sandbox"]}
        exe = os.environ.get("SPARK_STUDIO_TEST_CHROMIUM")
        if exe:
            kwargs["executable_path"] = exe
        try:
            cls.browser = cls._pw.chromium.launch(**kwargs)
        except Exception as exc:  # no browser downloaded / not runnable
            cls._pw.stop()
            raise unittest.SkipTest(f"chromium unavailable: {exc}")
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")
        cls.markup_ids = set(ID_RE.findall(cls.html))

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls._pw.stop()

    def _open(self, lang: str, with_app: bool = True):
        page = self.browser.new_page(locale="en-US")
        errors, chat_bodies = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        # Top frame only: sandboxed iframes (canvas preview) have no storage.
        page.add_init_script(
            "if (window === window.top) { try { localStorage.setItem('ss_lang', %s); }"
            " catch (e) {} }" % json.dumps(lang))

        def route(r):
            url = r.request.url
            path = urlsplit(url).path
            if url.rstrip("/") == ORIGIN:
                return r.fulfill(status=200, content_type="text/html", body=self.html)
            if path.startswith("/static/") and (with_app or path != "/static/app.js"):
                f = WEB / path[len("/static/"):]
                if f.is_file():
                    ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                    return r.fulfill(status=200, content_type=ctype, body=f.read_bytes())
            if path == "/api/chat" and r.request.method == "POST":
                chat_bodies.append(json.loads(r.request.post_data or "{}"))
                sse = ('data: {"choices":[{"delta":{"content":"pong"}}]}\n\n'
                       "data: [DONE]\n\n")
                return r.fulfill(status=200, content_type="text/event-stream", body=sse)
            return r.abort()

        page.route("**/*", route)
        page.goto(ORIGIN + "/", wait_until="load")
        page.wait_for_function("() => window.ssLang !== undefined")
        return page, errors, chat_bodies

    def _check_static_pass(self, lang: str):
        # i18n.js alone (app.js blocked): app.js legitimately re-renders some
        # containers (e.g. #registryStatus when its API call fails), so the
        # "no id lost" invariant is checked on the static i18n pass only.
        page, errors, _ = self._open(lang, with_app=False)
        try:
            self.assertEqual(page.evaluate("window.ssLang"), lang)
            self.assertEqual(page.evaluate("document.documentElement.lang"), lang)
            dom_ids = set(page.evaluate("[...document.querySelectorAll('[id]')].map(e => e.id)"))
            self.assertEqual(sorted(self.markup_ids - dom_ids), [],
                             f"[{lang}] ids deleted by the static i18n pass")
            self.assertEqual(page.evaluate("document.querySelectorAll('body > i').length"), 0,
                             f"[{lang}] an unclosed <i> icon leaked under <body>")
            self.assertEqual(errors, [], f"[{lang}] page errors")
        finally:
            page.close()

    def _check_engine_chat(self, lang: str):
        page, errors, chat_bodies = self._open(lang)
        try:
            self.assertEqual(page.evaluate("window.ssLang"), lang)

            sliders = page.evaluate("""() => ['wgTemp', 'wgMaxTokens'].map(id => {
                const el = document.getElementById(id);
                return el && { type: el.type, label: el.closest('label')
                    && el.closest('label').querySelector('[data-i18n]').textContent.trim() };
            })""")
            self.assertTrue(all(sliders), f"[{lang}] sliders missing: {sliders}")
            self.assertEqual([s["type"] for s in sliders], ["range", "range"])
            if lang == "fr":
                self.assertEqual(sliders[0]["label"], _fr_value("ss.webgpu.temperature"))
                self.assertEqual(sliders[1]["label"], _fr_value("ss.webgpu.max_tokens"))
            else:
                self.assertEqual([s["label"] for s in sliders], ["Temperature", "Max tokens"])

            slider_values = page.evaluate(
                "[Number(document.querySelector('#wgTemp').value),"
                " Number(document.querySelector('#wgMaxTokens').value)]")
            # DOM clicks (not pointer clicks): onboarding overlays may cover
            # the panel in a fresh profile, which is irrelevant here.
            page.evaluate("document.querySelector('[data-tab=webgpu]').click();"
                          "document.querySelector('#wgInput').value = 'ping';"
                          "document.querySelector('#wgSend').click();")
            page.wait_for_function(
                "() => /pong|Error/.test(document.querySelector('#wgMessages').innerText)",
                timeout=10_000)
            messages = page.inner_text("#wgMessages")
            self.assertNotIn("Error", messages, f"[{lang}] send failed: {messages!r}")
            self.assertIn("pong", messages)
            self.assertEqual(len(chat_bodies), 1)
            # The request carries the slider values (the range snaps the
            # 4096 markup default to its step grid, hence reading them back).
            self.assertEqual(slider_values[0], 0.7)
            self.assertEqual([chat_bodies[0]["temperature"], chat_bodies[0]["max_tokens"]],
                             slider_values)
            self.assertEqual(errors, [], f"[{lang}] page errors")
        finally:
            page.close()

    def test_english_static_pass(self):
        self._check_static_pass("en")

    def test_french_static_pass(self):
        self._check_static_pass("fr")

    def test_english_engine_chat_send(self):
        self._check_engine_chat("en")

    def test_french_engine_chat_send(self):
        self._check_engine_chat("fr")


if __name__ == "__main__":
    unittest.main()
