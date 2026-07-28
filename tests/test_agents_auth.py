import asyncio
import json
import subprocess
import unittest
from unittest import mock

import agents


class HuggingFaceStatusTests(unittest.TestCase):
    def test_missing_cli_is_reported(self):
        with mock.patch.object(agents, "_which", return_value=None):
            self.assertEqual(
                agents.huggingface_status(),
                {"installed": False, "logged_in": False, "username": None},
            )

    def test_whoami_returns_only_public_identity(self):
        output = json.dumps({
            "status": "Logged in",
            "user": "spark-user",
            "auth": {"accessToken": "must-not-leak"},
        })
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
        with (
            mock.patch.object(agents, "_which", return_value="/project/env/bin/hf"),
            mock.patch.object(agents._subprocess, "run", return_value=result) as run,
        ):
            status = agents.huggingface_status()

        self.assertEqual(
            status,
            {"installed": True, "logged_in": True, "username": "spark-user"},
        )
        self.assertNotIn("must-not-leak", repr(status))
        self.assertEqual(
            run.call_args.args[0],
            ["/project/env/bin/hf", "auth", "whoami", "--format", "json"],
        )

    def test_failed_whoami_is_logged_out(self):
        result = subprocess.CompletedProcess([], 1, stdout="", stderr="Not logged in")
        with (
            mock.patch.object(agents, "_which", return_value="/project/env/bin/hf"),
            mock.patch.object(agents._subprocess, "run", return_value=result),
        ):
            self.assertEqual(
                agents.huggingface_status(),
                {"installed": True, "logged_in": False, "username": None},
            )

    def test_non_identity_strings_are_never_returned(self):
        output = json.dumps({"auth": {"accessToken": "must-not-leak"}})
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
        with (
            mock.patch.object(agents, "_which", return_value="/project/env/bin/hf"),
            mock.patch.object(agents._subprocess, "run", return_value=result),
        ):
            status = agents.huggingface_status()

        self.assertTrue(status["logged_in"])
        self.assertIsNone(status["username"])
        self.assertNotIn("must-not-leak", repr(status))


class _AsyncLines:
    def __init__(self, lines):
        self._lines = lines

    def __aiter__(self):
        self._iterator = iter(self._lines)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _Process:
    def __init__(self):
        self.stdout = _AsyncLines([b"Open https://huggingface.co/device\n"])
        self.returncode = 0

    async def wait(self):
        return self.returncode


class HuggingFaceLoginTests(unittest.TestCase):
    def test_login_uses_official_browser_flow(self):
        async def collect():
            return [line async for line in agents.login_stream("huggingface")]

        spawn = mock.AsyncMock(return_value=_Process())
        with (
            mock.patch.object(agents, "_which", return_value="/project/env/bin/hf"),
            mock.patch.object(agents.asyncio, "create_subprocess_exec", spawn),
        ):
            lines = asyncio.run(collect())

        self.assertIn("Spark Studio never sees your token.", lines[1])
        self.assertIn("https://huggingface.co/device", lines[2])
        self.assertEqual(
            spawn.call_args.args[:4],
            ("/project/env/bin/hf", "auth", "login", "--force"),
        )


if __name__ == "__main__":
    unittest.main()
