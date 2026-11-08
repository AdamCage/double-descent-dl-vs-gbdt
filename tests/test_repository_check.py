"""The offline check must not report success for an empty test suite."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.check_repository import main, run_unit_tests


class RepositoryDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "tests").mkdir()

    def test_empty_discovery_cannot_pass_repository_check(self) -> None:
        output = self.root / "checks.json"
        arguments = ["check_repository.py", "--root", str(self.root), "--output", str(output)]
        with patch("sys.argv", arguments), patch(
                "scripts.check_repository.structural_checks", return_value=[]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertNotEqual(caught.exception.code, 0)
        summary = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(summary["tests_run"], 0)
        self.assertFalse(summary["passed"])

    def test_real_test_is_counted(self) -> None:
        (self.root / "tests/test_sample.py").write_text(
            "import unittest\n"
            "class Sample(unittest.TestCase):\n"
            "    def test_pass(self):\n"
            "        self.assertEqual(2 + 2, 4)\n", encoding="utf-8")
        _, result, count = run_unit_tests(self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(count, 1)

    def test_failed_test_retains_failure_and_count(self) -> None:
        (self.root / "tests/test_sample.py").write_text(
            "import unittest\n"
            "class Sample(unittest.TestCase):\n"
            "    def test_fail(self):\n"
            "        self.fail('expected fixture failure')\n", encoding="utf-8")
        _, result, count = run_unit_tests(self.root)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
