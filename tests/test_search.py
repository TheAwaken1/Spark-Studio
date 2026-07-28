import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import sparkstudio_cli
import sparkstudio_mcp
import studio_search


class SearchClientTests(unittest.TestCase):
    def test_search_calls_studio_pipeline_and_clamps_limit(self):
        payload = {
            "query": "DGX Spark",
            "url": "searxng (bundled)",
            "results": [{"title": "Result", "url": "https://example.com"}],
        }
        response = mock.Mock()
        response.json.return_value = payload
        client = mock.Mock()
        client.get.return_value = response
        context = mock.MagicMock()
        context.__enter__.return_value = client
        with mock.patch.object(studio_search.httpx, "Client", return_value=context):
            result = studio_search.search(
                "http://127.0.0.1:7860/",
                "  DGX Spark  ",
                limit=99,
                enrich=True,
            )

        self.assertEqual(result, payload)
        client.get.assert_called_once_with(
            "http://127.0.0.1:7860/api/search",
            params={"q": "DGX Spark", "limit": 10, "enrich": "true"},
        )
        response.raise_for_status.assert_called_once_with()

    def test_search_rejects_empty_or_oversized_queries(self):
        with self.assertRaises(ValueError):
            studio_search.search("http://127.0.0.1:7860", "  ")
        with self.assertRaises(ValueError):
            studio_search.search("http://127.0.0.1:7860", "x" * 501)

    def test_formatter_keeps_titles_urls_and_compact_content(self):
        text = studio_search.format_results({
            "query": "current docs",
            "url": "duckduckgo",
            "results": [{
                "title": "Official docs",
                "url": "https://example.com/docs",
                "content": "A  current\nsource.",
            }],
        })
        self.assertIn("Backend: duckduckgo", text)
        self.assertIn("1. Official docs", text)
        self.assertIn("https://example.com/docs", text)
        self.assertIn("A current source.", text)


class SearchMcpTests(unittest.TestCase):
    def test_mcp_tool_routes_only_through_studio_search(self):
        payload = {
            "query": "latest release",
            "url": "http://127.0.0.1:8888",
            "results": [{"title": "Release", "url": "https://example.com/release"}],
        }
        with mock.patch.object(studio_search, "search", return_value=payload) as search:
            result = sparkstudio_mcp.run_search_tool(
                "http://127.0.0.1:7860", "latest release", 4, True
            )

        search.assert_called_once_with(
            "http://127.0.0.1:7860",
            "latest release",
            limit=4,
            enrich=True,
            timeout=60,
        )
        self.assertEqual(result["backend"], "http://127.0.0.1:8888")
        self.assertEqual(result["results"], payload["results"])


class SearchCliTests(unittest.TestCase):
    def test_human_search_output_uses_shared_formatter(self):
        args = sparkstudio_cli.build_parser().parse_args(["search", "DGX Spark"])
        payload = {"query": "DGX Spark", "url": "duckduckgo", "results": []}
        with (
            mock.patch.object(studio_search, "search", return_value=payload),
            redirect_stdout(io.StringIO()) as output,
        ):
            code = args.func(args)

        self.assertEqual(code, 0)
        self.assertIn("Backend: duckduckgo", output.getvalue())
        self.assertIn("No results.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
