"""Charter accounting must count failures and must not invent missing durations."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.resource_ledger import LedgerError, append_entry, report


def charter() -> dict:
    return {
        "budgets": {
            "total_runner_wall_seconds": 100,
            "max_stage_wall_seconds": 40,
            "new_storage_bytes": 1000,
            "public_dataset_download_bytes": 10,
        },
        "r1_budgets": {
            "document_download_bytes": 100,
            "max_document_bytes": 50,
            "new_storage_bytes": 500,
            "dataset_download_bytes": 0,
            "max_http_requests": 48,
        },
    }


class LedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = self.root / "research/decisions"
        path.mkdir(parents=True)
        (path / "autonomous-research-authorization.json").write_text(
            json.dumps(charter()), encoding="utf-8")

    def test_failed_elapsed_counts_and_missing_elapsed_is_not_zero(self) -> None:
        failed = self.root / "artifacts/research/R0/failed"
        failed.mkdir(parents=True)
        (failed / "manifest.json").write_text(
            json.dumps({"elapsed_seconds": 3, "status": "failed"}), encoding="utf-8")
        missing = self.root / "artifacts/research/R1/interrupted"
        missing.mkdir(parents=True)
        (self.root / "result").mkdir()
        (self.root / "result/ignored.bin").write_bytes(b"x" * 5000)
        (self.root / "artifacts/research/counted.txt").write_bytes(b"y" * 25)
        document = report(self.root)
        charged = {item["attempt_id"]: item for item in document["attempts"]}
        self.assertEqual(charged["failed"]["charged_seconds"], 3)
        self.assertTrue(charged["failed"]["measured"])
        self.assertEqual(charged["interrupted"]["charged_seconds"], 40)
        self.assertFalse(charged["interrupted"]["charge_is_measurement"])
        self.assertEqual(document["charged_wall_seconds"], 43)
        self.assertFalse(document["computational_activation_ready"])
        self.assertGreaterEqual(document["storage"]["bytes"], 25)
        self.assertLess(document["storage"]["bytes"], 1000)
        self.assertIn("result", document["storage"]["excluded_roots"])

    def test_ledger_cannot_contradict_a_manifest(self) -> None:
        attempt = self.root / "artifacts/research/R0/done"
        attempt.mkdir(parents=True)
        (attempt / "manifest.json").write_text(
            json.dumps({"elapsed_seconds": 2, "status": "complete"}), encoding="utf-8")
        append_entry(self.root, {"kind": "attempt", "stage_id": "R0", "attempt_id": "done",
                                 "elapsed_seconds": 9})
        document = report(self.root)
        self.assertTrue(document["reconciliation_problems"])
        self.assertFalse(document["computational_activation_ready"])

    def test_boolean_elapsed_is_unmeasured(self) -> None:
        attempt = self.root / "artifacts/research/R0/bad"
        attempt.mkdir(parents=True)
        (attempt / "manifest.json").write_text(
            json.dumps({"elapsed_seconds": True, "status": "failed"}), encoding="utf-8")
        document = report(self.root)
        self.assertFalse(document["attempts"][0]["measured"])
        self.assertNotEqual(document["attempts"][0]["charged_seconds"], 1)

    def test_append_rejects_an_empty_entry(self) -> None:
        with self.assertRaises(LedgerError):
            append_entry(self.root, {})
