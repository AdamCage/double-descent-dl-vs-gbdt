"""Integrity gates for future claims; all approvals below are test fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.research_claims import validate_claims


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ClaimGateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "metrics.json").write_bytes(b'{"loss":0.1}\n')
        self.evidence = [{"path": "metrics.json", "sha256": sha((self.root / "metrics.json").read_bytes())}]
        self.statement = "Fixture measured loss is 0.1."
        self.decision = {
            "claim_id": "fixture-claim",
            "statement_sha256": sha(self.statement.encode("utf-8")),
            "evidence": copy.deepcopy(self.evidence),
            "reviewer": "test-reviewer-label",
            "harness": "codex",
            "verdict": "accepted",
        }
        self.claim = {
            "id": "fixture-claim", "statement": self.statement,
            "scope": {"dataset": "unit test fixture only"},
            "limitations": ["Not real research evidence"],
            "evidence": copy.deepcopy(self.evidence),
        }
        self.doc = {"schema_version": 1, "claims": [self.claim]}
        self.save_decision()

    def save_decision(self, raw: bytes | None = None):
        target = self.root / "decision.json"
        target.write_bytes(raw if raw is not None else json.dumps(self.decision).encode("utf-8"))
        self.claim["scientific_review"] = {"path": "decision.json", "sha256": sha(target.read_bytes())}

    def authorize_cursor(self):
        authorization = {
            "schema_version": 1, "decision_id": "DD-AUTO-2026-11-07",
            "authority": "user", "status": "accepted", "cursor_claim_review": True,
            "require_distinct_review_subagent": True,
        }
        target = self.root / "authorization.json"
        target.write_text(json.dumps(authorization), encoding="utf-8")
        self.doc["autonomy_authorization"] = {
            "path": "authorization.json", "sha256": sha(target.read_bytes())}
        self.claim.update(assessment="evidence_supported", review_status="cursor_team_reviewed")
        self.decision.update(
            harness="cursor", review_mode="subagent_review", author_agent="cursor-author",
            reviewer_agent="cursor-reviewer", authorization_decision=copy.deepcopy(self.doc["autonomy_authorization"]))
        self.save_decision()
        return authorization

    def test_exact_bound_decision_is_accepted_mechanically(self):
        self.assertEqual(validate_claims(self.root, self.doc), [])
        self.decision["harness"] = "human"
        self.save_decision()
        self.assertEqual(validate_claims(self.root, self.doc), [])

    def test_empty_claim_registry_needs_no_synthetic_approval(self):
        self.assertEqual(validate_claims(self.root, {"schema_version": 1, "claims": []}), [])

    def test_changed_statement_cannot_reuse_review(self):
        self.claim["statement"] += " Broader conclusion."
        self.assertTrue(any("statement_sha256 mismatch" in issue for issue in validate_claims(self.root, self.doc)))

    def test_changed_evidence_bytes_are_rejected(self):
        (self.root / "metrics.json").write_bytes(b'{"loss":0.01}\n')
        self.assertTrue(any("SHA-256 mismatch" in issue for issue in validate_claims(self.root, self.doc)))

    def test_changed_decision_bytes_are_rejected(self):
        (self.root / "decision.json").write_bytes(b'{}')
        self.assertTrue(any("scientific_review: SHA-256 mismatch" in issue for issue in validate_claims(self.root, self.doc)))

    def test_decision_cannot_approve_different_or_partial_evidence(self):
        (self.root / "other.json").write_bytes(b"other evidence")
        self.decision["evidence"] = [{"path": "other.json", "sha256": sha(b"other evidence")}]
        self.save_decision()
        self.assertTrue(any("match exact" in issue for issue in validate_claims(self.root, self.doc)))

    def test_unauthorized_cursor_and_missing_reviewer_are_rejected(self):
        for field, value in (("harness", "cursor"), ("harness", ["codex"]),
                             ("reviewer", " "), ("verdict", "proposed"),
                             ("claim_id", "different-claim")):
            with self.subTest(field=field, value=value):
                saved = self.decision[field]
                self.decision[field] = value
                self.save_decision()
                self.assertTrue(validate_claims(self.root, self.doc))
                self.decision[field] = saved

    def test_authorized_cursor_subagent_review_is_admitted_as_local_team_review(self):
        self.authorize_cursor()
        self.assertEqual(validate_claims(self.root, self.doc), [])

    def test_cursor_requires_exact_authorization_binding(self):
        self.authorize_cursor()
        del self.doc["autonomy_authorization"]
        self.assertTrue(any("valid autonomy_authorization" in issue
                            for issue in validate_claims(self.root, self.doc)))
        self.authorize_cursor()
        self.decision["authorization_decision"]["sha256"] = "0" * 64
        self.save_decision()
        self.assertTrue(any("exact charter path/hash" in issue
                            for issue in validate_claims(self.root, self.doc)))

    def test_changed_charter_is_rejected_even_for_empty_registry(self):
        self.authorize_cursor()
        (self.root / "authorization.json").write_text("{}", encoding="utf-8")
        self.doc["claims"] = []
        self.assertTrue(any("autonomy_authorization: SHA-256 mismatch" in issue
                            for issue in validate_claims(self.root, self.doc)))

    def test_charter_must_explicitly_enable_cursor_and_subagent_review(self):
        authorization = self.authorize_cursor()
        for field, value in (("schema_version", True), ("decision_id", " "),
                             ("authority", "cursor"), ("status", "draft"),
                             ("cursor_claim_review", False), ("cursor_claim_review", 1),
                             ("require_distinct_review_subagent", False)):
            with self.subTest(field=field, value=value):
                modified = dict(authorization, **{field: value})
                target = self.root / "authorization.json"
                target.write_text(json.dumps(modified), encoding="utf-8")
                self.doc["autonomy_authorization"]["sha256"] = sha(target.read_bytes())
                self.decision["authorization_decision"] = copy.deepcopy(self.doc["autonomy_authorization"])
                self.save_decision()
                self.assertTrue(any("explicitly enable Cursor" in issue
                                    for issue in validate_claims(self.root, self.doc)))

    def test_cursor_sole_self_review_and_missing_agent_labels_are_rejected(self):
        self.authorize_cursor()
        for field, value in (("review_mode", "self_review"),
                             ("author_agent", " "), ("reviewer_agent", None),
                             ("reviewer_agent", "cursor-author"),
                             ("reviewer_agent", " Cursor-Author ")):
            with self.subTest(field=field, value=value):
                saved = self.decision[field]
                self.decision[field] = value
                self.save_decision()
                self.assertTrue(validate_claims(self.root, self.doc))
                self.decision[field] = saved

    def test_cursor_claims_require_honest_local_assessment_labels(self):
        self.authorize_cursor()
        for field, value in (("assessment", "confirmed_truth"),
                             ("review_status", "human_approved"),
                             ("scientific_validity", "approved")):
            with self.subTest(field=field, value=value):
                modified = copy.deepcopy(self.doc)
                modified["claims"][0][field] = value
                self.assertTrue(validate_claims(self.root, modified))
        self.decision["scientific_validity"] = "approved"
        self.save_decision()
        self.assertTrue(any("cannot certify" in issue for issue in validate_claims(self.root, self.doc)))

    def test_invalid_charter_json_and_unsafe_binding_are_rejected(self):
        self.authorize_cursor()
        for raw in (b"[]", b'{"cursor_claim_review":true,"cursor_claim_review":false}',
                    b'{"budget":NaN}', b"\xff"):
            with self.subTest(raw=raw):
                target = self.root / "authorization.json"
                target.write_bytes(raw)
                self.doc["autonomy_authorization"]["sha256"] = sha(raw)
                self.assertTrue(validate_claims(self.root, self.doc))
        self.doc["autonomy_authorization"] = {"path": "../outside.json", "sha256": "0" * 64}
        self.assertTrue(validate_claims(self.root, self.doc))

    def test_missing_statement_scope_or_limitations_is_rejected(self):
        for field in ("statement", "scope", "limitations", "evidence", "scientific_review"):
            with self.subTest(field=field):
                doc = copy.deepcopy(self.doc)
                del doc["claims"][0][field]
                self.assertTrue(validate_claims(self.root, doc))

    def test_malformed_user_json_returns_diagnostics(self):
        for value in (None, [], "text", {}, {"schema_version": True, "claims": []},
                      {"schema_version": 1, "claims": {}},
                      {"schema_version": 1, "claims": [None, [], {"id": []}]},
                      {"schema_version": 1, "claims": [{"evidence": [None, [], {"path": []}]}]}):
            with self.subTest(value=value):
                self.assertTrue(validate_claims(self.root, value))
        self.assertTrue(validate_claims(None, self.doc))

    def test_invalid_decision_json_is_rejected(self):
        for raw in (b"not json", b"[]", b'{"verdict":"accepted","verdict":"rejected"}',
                    b'{"metric":NaN}', b"\xff"):
            with self.subTest(raw=raw):
                self.save_decision(raw)
                self.assertTrue(validate_claims(self.root, self.doc))

    def test_lone_surrogates_return_diagnostics(self):
        self.claim["statement"] = "\ud800"
        self.assertTrue(validate_claims(self.root, self.doc))
        self.claim["statement"] = self.statement
        self.claim["evidence"][0]["path"] = "\ud800"
        self.assertTrue(validate_claims(self.root, self.doc))

    def test_duplicate_ids_and_duplicate_evidence_are_rejected(self):
        self.doc["claims"].append(copy.deepcopy(self.claim))
        self.assertTrue(any("duplicate claim id" in issue for issue in validate_claims(self.root, self.doc)))
        self.doc["claims"].pop()
        self.claim["evidence"].append(copy.deepcopy(self.evidence[0]))
        self.assertTrue(any("duplicate evidence" in issue for issue in validate_claims(self.root, self.doc)))

    def test_absolute_parent_and_drive_relative_paths_are_rejected(self):
        for path in ("../outside.json", str(self.root / "metrics.json"), "C:metrics.json",
                     "\\\\server\\share\\metrics.json", "/etc/passwd", "\x00"):
            with self.subTest(path=path):
                self.claim["evidence"] = [{"path": path, "sha256": "0" * 64}]
                self.assertTrue(validate_claims(self.root, self.doc))

    def test_symlink_outside_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "outside.json"
            target.write_bytes(b"outside")
            link = self.root / "linked.json"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            self.claim["evidence"] = [{"path": "linked.json", "sha256": sha(b"outside")}]
            self.assertTrue(any("escapes" in issue for issue in validate_claims(self.root, self.doc)))


if __name__ == "__main__":
    unittest.main()
