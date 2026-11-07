"""The R6 audit must refuse a download and a fit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class R6AuditTest(unittest.TestCase):
    def test_audit_writes_not_run_without_opening_a_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/r6_audit.py")],
                cwd=directory, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((Path(directory) / "disposition.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["disposition"], "not_run")
        self.assertEqual(payload["download_bytes"], 0)
        self.assertEqual(payload["models_fit"], 0)
        self.assertEqual(payload["datasets_opened"], [])
        self.assertFalse(payload["candidate_considered"]["opened"])
        self.assertIn("R5-replication-not-one-mechanism", payload["claim_ids"])


if __name__ == "__main__":
    unittest.main()
