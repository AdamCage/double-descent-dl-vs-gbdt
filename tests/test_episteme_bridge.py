"""Question planning and provenance guards; fixtures are not research evidence."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from episteme.store import Store

from scripts import episteme_bridge as bridge
from scripts import episteme_study, export_episteme_bundle


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.root = self.base / bridge.STUDY
        self.program = self.base / "program.md"
        self.program.write_text("Frozen test question; no experiment.\n", encoding="utf-8")

    def test_status_is_read_only_and_does_not_initialize(self):
        self.assertEqual(bridge.inspect(self.root)["status"], "not_initialized")
        self.assertFalse(self.root.exists())

    def test_bootstrap_is_idempotent_question_only(self):
        first = bridge.bootstrap(self.root, self.program)
        second = bridge.bootstrap(self.root, self.program)
        self.assertEqual(first, second)
        self.assertEqual(first["events"], 1)
        self.assertEqual(first["receipts"], 1)
        self.assertEqual(first["registration_kind"], "research_question_only")
        self.assertEqual(first["scientific_validity"], "not_assessed")
        self.assertEqual(first["review"], "not_requested")
        self.assertFalse((self.root / "evaluation_pointer.json").exists())

    def test_changed_program_does_not_rewrite_registration(self):
        original = bridge.bootstrap(self.root, self.program)
        self.program.write_text("Changed after registration", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after"):
            bridge.bootstrap(self.root, self.program)
        self.assertEqual(original, bridge.inspect(self.root))

    def test_lost_pointer_recovers_receipt_without_duplicate_event(self):
        first = bridge.bootstrap(self.root, self.program)
        (self.root / bridge.POINTER).unlink()
        self.assertEqual(first, bridge.bootstrap(self.root, self.program))

    def test_crash_after_empty_store_creation_can_resume(self):
        with Store(self.root):
            pass
        self.assertEqual(bridge.inspect(self.root)["status"], "not_initialized")
        self.assertEqual(bridge.bootstrap(self.root, self.program)["events"], 1)

    def test_pointer_tamper_is_rejected(self):
        bridge.bootstrap(self.root, self.program)
        pointer = self.root / bridge.POINTER
        pointer.write_text(pointer.read_text(encoding="utf-8").replace(
            '"not_assessed"', '"approved"'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pointer mismatch"):
            bridge.inspect(self.root)

    def test_legacy_root_is_rejected_before_creation(self):
        historical = self.base / "double-descent"
        with self.assertRaisesRegex(ValueError, "only supports"):
            bridge.bootstrap(historical, self.program)
        self.assertFalse(historical.exists())

    def test_inspect_does_not_change_database_bytes(self):
        bridge.bootstrap(self.root, self.program)
        database = self.root / "state.sqlite3"
        before = hashlib.sha256(database.read_bytes()).hexdigest()
        bridge.inspect(self.root)
        self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())


class HistoricalEntryPointTest(unittest.TestCase):
    def test_obsolete_entrypoints_refuse_without_loading_artifacts(self):
        for function in (episteme_study.question, episteme_study.evaluate,
                         export_episteme_bundle.main):
            with self.subTest(function=function.__qualname__):
                with self.assertRaisesRegex(SystemExit, "Historical"):
                    function()


if __name__ == "__main__":
    unittest.main()
