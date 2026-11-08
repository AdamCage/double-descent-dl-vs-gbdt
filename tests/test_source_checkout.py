"""Source CI must expose absent evidence and reject corrupt source controls."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.check_source_checkout import REQUIRED, RELEASE_MANIFEST, SourceAudit, main
from src.research_claims import validate_claims


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SourceCheckoutTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for relative in REQUIRED:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        self.write("research/stages.json", {"schema_version": 1, "stages": [{
            "id": "R0", "kind": "engineering", "depends_on": [],
            "plan": "research/plans/R0.json", "execution_authorized": True}]})
        self.write("research/plans/R0.json", {"schema_version": 1, "stage_id": "R0", "tasks": [{
            "id": "fixture", "inputs": ["src/source.py", "artifacts/intake/data.parquet"],
            "outputs": ["out.json"]}]})
        path = self.root / "src/source.py"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"source bytes\n")
        self.evidence = [{"path": "artifacts/research/R0/attempt/metrics.json", "sha256": "a" * 64},
                         {"path": "src/source.py", "sha256": digest(path.read_bytes())}]
        self.claim = {"id": "fixture", "statement": "A bounded fixture statement.",
                      "scope": "Fixture", "limitations": ["Fixture only"],
                      "evidence": self.evidence}
        self.decision = {"claim_id": "fixture", "statement_sha256": digest(self.claim["statement"].encode()),
                         "verdict": "accepted", "harness": "codex", "reviewer": "fixture reviewer",
                         "evidence": self.evidence}
        self.save_decision()
        self.write("research/evidence/legacy-2026-11-07/inventory.json", {"files": []})

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def save_decision(self):
        path = self.write("research/decisions/fixture.json", self.decision)
        self.claim["scientific_review"] = {"path": "research/decisions/fixture.json",
                                           "sha256": digest(path.read_bytes())}
        self.write("research/claims.json", {"schema_version": 1, "claims": [self.claim]})

    def audit(self):
        return SourceAudit(self.root).run()

    def test_missing_evidence_is_reported_and_full_claim_gate_still_rejects(self):
        result = self.audit()
        self.assertEqual(result["source_problems"], [])
        self.assertEqual(result["evidence_validation"], "not_performed_source_checkout")
        self.assertEqual(result["scientific_claim_admission"], "not_performed")
        self.assertEqual(len(result["unverified_artifact_references"]), 3)
        doc = json.loads((self.root / "research/claims.json").read_text())
        self.assertTrue(validate_claims(self.root, doc))

    def test_existing_artifact_is_still_not_claimed_as_verified(self):
        self.write("artifacts/research/R0/attempt/metrics.json", {"arbitrary": "not the expected hash"})
        result = self.audit()
        self.assertEqual(result["source_problems"], [])
        self.assertTrue(any(x["path"].endswith("metrics.json") for x in result["unverified_artifact_references"]))

    def test_source_hash_corruption_is_a_failure(self):
        (self.root / "src/source.py").write_bytes(b"changed source\n")
        self.assertTrue(any("SHA-256 mismatch" in x for x in self.audit()["source_problems"]))

    def test_missing_source_cannot_be_classified_as_release_data(self):
        (self.root / "src/source.py").unlink()
        self.assertTrue(any("missing source/control" in x for x in self.audit()["source_problems"]))

    def test_artifact_traversal_is_rejected(self):
        path = self.root / "research/plans/R0.json"
        plan = json.loads(path.read_text())
        plan["tasks"][0]["inputs"] = ["artifacts/../src/source.py"]
        self.write("research/plans/R0.json", plan)
        self.assertTrue(any("escapes" in x for x in self.audit()["source_problems"]))

    def test_changed_decision_and_mismatched_evidence_are_rejected(self):
        self.decision["evidence"] = [self.evidence[0]]
        self.save_decision()
        self.assertTrue(any("exact evidence metadata mismatch" in x for x in self.audit()["source_problems"]))

    def test_claim_statement_binding_is_checked(self):
        self.claim["statement"] = "A changed claim."
        self.save_decision()
        self.assertTrue(any("statement/ID/verdict mismatch" in x for x in self.audit()["source_problems"]))

    def test_release_manifest_does_not_require_asset_bytes_but_rejects_invalid_hash(self):
        asset = {"path": "research/evidence/missing.zip", "bytes": 123, "sha256": "b" * 64}
        self.write(RELEASE_MANIFEST, {"schema_version": 1, "assets": [asset]})
        self.assertEqual(self.audit()["source_problems"], [])
        asset["sha256"] = "invalid"
        self.write(RELEASE_MANIFEST, {"schema_version": 1, "assets": [asset]})
        self.assertTrue(any("release manifest" in x for x in self.audit()["source_problems"]))

    def test_empty_unit_discovery_fails_even_when_source_audit_passes(self):
        (self.root / "tests").mkdir()
        output = self.root / "checks.json"
        with patch("sys.argv", ["check_source_checkout.py", "--root", str(self.root), "--output", str(output)]), \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertNotEqual(caught.exception.code, 0)
        summary = json.loads(output.read_text())
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["tests_run"], 0)


if __name__ == "__main__":
    unittest.main()
