"""OOM guard: keep the dashboard alive, make the model the OOM victim.

oomguard never kills anything itself — it only nudges oom_score_adj through
procfs. builtins.open is mocked so no real /proc file is read or written.
"""

import unittest
from unittest import mock

import oomguard


class FakeProc:
    """Minimal procfs stand-in: path -> contents, with optional per-mode errors."""

    def __init__(self, files=None, read_error=False, write_error=False):
        self.files = dict(files or {})
        self.read_error = read_error
        self.write_error = write_error
        self.writes = []

    def open(self, path, mode="r", *a, **k):
        if "w" in mode:
            if self.write_error:
                raise PermissionError(13, "Permission denied", path)
            proc = self

            class _W:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def write(self, data):
                    proc.writes.append((path, data))
                    proc.files[path] = data

            return _W()
        if self.read_error or path not in self.files:
            raise FileNotFoundError(2, "No such file", path)
        return mock.mock_open(read_data=self.files[path])()


def _patched(fake):
    return mock.patch("builtins.open", side_effect=fake.open)


class ConstantsTests(unittest.TestCase):
    def test_engine_is_pushed_above_dashboard_within_kernel_range(self):
        # Kernel accepts -1000..1000; engines must be well above the default 0.
        self.assertGreater(oomguard._ENGINE_OOM_SCORE_ADJ, 0)
        self.assertLessEqual(oomguard._ENGINE_OOM_SCORE_ADJ, 1000)
        self.assertLess(oomguard._SELF_OOM_SCORE_ADJ, 0)
        self.assertGreaterEqual(oomguard._SELF_OOM_SCORE_ADJ, -1000)


class ProtectSelfTests(unittest.TestCase):
    PATH = "/proc/self/oom_score_adj"

    def test_no_procfs_is_skipped(self):
        fake = FakeProc(read_error=True)
        with _patched(fake):
            msg = oomguard.protect_self()
        self.assertIn("procfs unavailable", msg)
        self.assertEqual(fake.writes, [])

    def test_already_at_or_below_target_is_left_alone(self):
        for cur in (str(oomguard._SELF_OOM_SCORE_ADJ), "-1000"):
            fake = FakeProc({self.PATH: cur + "\n"})
            with _patched(fake):
                msg = oomguard.protect_self()
            self.assertEqual(msg, f"OOM guard: already protected (oom_score_adj={cur})")
            self.assertEqual(fake.writes, [])

    def test_lowers_score_when_privileged(self):
        fake = FakeProc({self.PATH: "0\n"})
        with _patched(fake):
            msg = oomguard.protect_self()
        self.assertEqual(fake.writes, [(self.PATH, str(oomguard._SELF_OOM_SCORE_ADJ))])
        self.assertIn(f"lowered own oom_score_adj to {oomguard._SELF_OOM_SCORE_ADJ}", msg)

    def test_empty_procfs_value_is_treated_as_zero(self):
        fake = FakeProc({self.PATH: ""})
        with _patched(fake):
            oomguard.protect_self()
        self.assertEqual(fake.writes, [(self.PATH, str(oomguard._SELF_OOM_SCORE_ADJ))])

    def test_unprivileged_write_failure_explains_the_fallback(self):
        fake = FakeProc({self.PATH: "200"}, write_error=True)
        with _patched(fake):
            msg = oomguard.protect_self()
        self.assertIn("can't lower own OOM priority without privilege", msg)
        self.assertIn("README", msg)


class DeprioritizeTests(unittest.TestCase):
    def test_raises_engine_score(self):
        fake = FakeProc()
        with _patched(fake):
            self.assertTrue(oomguard.deprioritize(4242))
        self.assertEqual(fake.writes, [("/proc/4242/oom_score_adj",
                                        str(oomguard._ENGINE_OOM_SCORE_ADJ))])

    def test_failure_returns_false(self):
        fake = FakeProc(write_error=True)
        with _patched(fake):
            self.assertFalse(oomguard.deprioritize(4242))

    def test_vanished_process_returns_false(self):
        with mock.patch("builtins.open", side_effect=FileNotFoundError(2, "gone")):
            self.assertFalse(oomguard.deprioritize(999999))


if __name__ == "__main__":
    unittest.main()
