"""R1 structural guards: hashes, access level, coverage and distinct review."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.r1_audit import (AuditError, audit_card, audit_design_review, audit_legacy,
                          audit_literature, audit_preparation, audit_r2, build_intake_index,
                          check_observation)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LegacyAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root / "notes/log.txt"
        self.evidence.parent.mkdir(parents=True)
        self.evidence.write_bytes(b"width=512 val_loss=0.0038\n")
        self.inventory = {"files": [
            {"path": "result/big.npy", "bytes": 99, "sha256": None, "status": "local_only_not_hashed"},
        ]}
        (self.root / "result").mkdir()
        (self.root / "result/big.npy").write_bytes(b"x" * 99)

    def notes(self) -> dict:
        attested = {
            "observation_id": "LEG-LOG",
            "statement": "The preserved log records one rounded validation loss.",
            "evidence_path": "notes/log.txt",
            "sha256": digest("width=512 val_loss=0.0038\n"),
            "locator": "line 1",
            "locator_line": 1,
            "quote": "val_loss=0.0038",
            "disposition": "supported",
            "scope": "One logged validation point only.",
            "limitations": "Rounded text is not a full-precision residual.",
            "content_attestation": True,
        }
        unhashed = {
            "observation_id": "LEG-BIG",
            "statement": "The matrix is listed without a content hash and was not read.",
            "evidence_path": "result/big.npy",
            "sha256": None,
            "locator": "inventory entry result/big.npy",
            "disposition": "not_assessed",
            "scope": "Byte count from the inventory only.",
            "limitations": "Size does not identify the matrix values or their row order.",
            "content_attestation": False,
            "inventory_bytes": 99,
        }
        return {
            "author_agent": "author-1",
            "observations": [attested, unhashed],
            "prediction_assessments": [{
                "id": "P1", "author_outcome": "consistent", "observation_ids": ["LEG-LOG", "LEG-BIG"],
                "reason": "The small log does not establish double descent on the corpus.",
            }],
        }

    def test_quotes_and_unhashed_matrices_are_checked(self) -> None:
        audit = audit_legacy(self.root, self.notes(), self.inventory, required_ids={"LEG-LOG", "LEG-BIG"})
        self.assertEqual(audit["negative_and_not_assessed"], ["LEG-BIG"])
        self.assertTrue(audit["observations"][1]["live_bytes_match_inventory"])

    def test_hash_mismatch_and_abstract_promotion_fail(self) -> None:
        notes = self.notes()
        notes["observations"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(AuditError, "SHA-256"):
            audit_legacy(self.root, notes, self.inventory, required_ids={"LEG-LOG", "LEG-BIG"})
        notes = self.notes()
        notes["observations"][1]["disposition"] = "supported"
        with self.assertRaisesRegex(AuditError, "not_assessed"):
            check_observation(self.root, notes["observations"][1],
                              {row["path"]: row for row in self.inventory["files"]})

    def test_promoted_full_text_without_snapshot_fails(self) -> None:
        card = {
            "source_id": "L01", "title": "T", "authors": ["A"], "version": "v1",
            "registry_version": "v1", "retrieved_utc": "2026-11-07T00:00:00+00:00",
            "access_status": "full_text", "review_level": "full_text",
            "snapshot_path": "missing.pdf", "snapshot_sha256": "abc", "text_path": "missing.txt",
            "text_sha256": "abc", "unavailable_reason": None, "locators": [],
            "model_and_data": "not_located_in_extract", "loss": "not_located_in_extract",
            "varied_axis": "not_located_in_extract", "held_fixed_axes": "not_located_in_extract",
            "interpolation_definition": "not_located_in_extract", "assumptions": ["not located in the extract"],
            "result_type": "not_located_in_extract", "transfer_limits": "No full text was stored.",
        }
        with self.assertRaises(AuditError):
            audit_card(self.root, card, {"L01": {"id": "L01", "title": "T", "authors": ["A"], "version": "v1"}})

    def test_intake_attempt_cap_and_distinct_review(self) -> None:
        records = [
            {"source_id": "L01", "attempt": 1, "url": "https://example.test/a", "status": "http_error"},
            {"source_id": "L01", "attempt": 2, "url": "https://example.test/b", "status": "http_error"},
            {"source_id": "L01", "attempt": 3, "url": "https://example.test/c", "status": "http_error"},
        ]
        with self.assertRaisesRegex(AuditError, "two retrieval"):
            build_intake_index(self.root, records, "intake")
        with self.assertRaisesRegex(AuditError, "different agent"):
            audit_design_review({
                "author_agent": "same", "reviewer_agent": "same", "review_mode": "subagent_review",
                "harness": "cursor", "verdict": "accept", "blocking_issues": [],
                "notes": "This note is long enough to be a review and still shares one author.",
            })

    def test_r2_rejects_execution_and_preparation_must_be_sealed(self) -> None:
        design = {key: "A" * 50 for key in (
            "teacher_and_maximum_feature_space", "nested_feature_selection",
            "feature_and_ridge_objective_normalization", "separate_data_noise_feature_model_seeds",
            "planned_p_over_n_grid", "noise_snr_and_clean_evaluation", "discovery_confirmation_exposure",
            "svd_solver_rcond_rank_residual_norm_spectrum", "estimand_and_uncertainty",
            "independent_draws_common_evaluation_points", "timing_pilot_and_cumulative_budget",
            "readiness_and_open_issues")}
        design["interventions_predictions_falsifiers_inconclusive_rules"] = [{
            "intervention": "Increase label noise on a fixed design.",
            "prediction": "The clean-risk peak near interpolation increases.",
            "falsifier": "The peak does not increase while the norm stays flat.",
            "inconclusive": "The grid does not bracket a numerical interpolation point.",
        }] * 3
        design.update(models_executed=True, readiness="ready_for_autonomous_planning",
                      open_issues=["Timing is not yet measured."], author_agent="author-1",
                      discovery_data_reused_as_confirmation=False,
                      prediction_assessments=[{
                          "id": "P3", "author_outcome": "consistent", "observation_ids": ["R2"],
                          "reason": "The contrasts are specified without fitting a model.",
                      }])
        with self.assertRaisesRegex(AuditError, "no model"):
            audit_r2(design)
        with self.assertRaisesRegex(AuditError, "sealed"):
            audit_preparation([{"kind": "status", "ts": "2026-11-07T00:00:00+00:00"}])

    def test_literature_round_trip_rejects_index_tampering(self) -> None:
        pdf = self.root / "intake/snapshots/L01.pdf"
        text = self.root / "intake/text/L01.txt"
        pdf.parent.mkdir(parents=True)
        text.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF-1.4\n")
        text.write_bytes("The risk curve is not universal.\n".encode("utf-8"))
        pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
        text_hash = hashlib.sha256(text.read_bytes()).hexdigest()
        records = [{"source_id": "L01", "attempt": 1, "url": "https://example.test/l01.pdf",
                    "status": "retrieved", "bytes": pdf.stat().st_size, "sha256": pdf_hash,
                    "snapshot_path": "snapshots/L01.pdf"}]
        index = build_intake_index(self.root, records, "intake")
        card = {
            "source_id": "L01", "title": "Title", "authors": ["Author"], "version": "published",
            "registry_version": "published", "retrieved_utc": "2026-11-07T00:00:00+00:00",
            "retrieved_url": "https://example.test/l01.pdf",
            "access_status": "full_text", "review_level": "full_text",
            "snapshot_path": "intake/snapshots/L01.pdf", "snapshot_sha256": pdf_hash,
            "text_path": "intake/text/L01.txt", "text_sha256": text_hash,
            "unavailable_reason": None,
            "locators": [{"label": "opening", "quote": "The risk curve is not universal."}],
            "model_and_data": "A short descriptive model and data statement.",
            "loss": "Squared prediction risk in the paper.",
            "varied_axis": "A declared model-complexity axis.",
            "held_fixed_axes": "Other training choices stay fixed.",
            "interpolation_definition": "Interpolation is defined in the paper text.",
            "assumptions": ["The quoted setting is not our corpus."],
            "result_type": "Theoretical or empirical result as quoted.",
            "transfer_limits": "The result does not transfer without its assumptions.",
        }
        document = {
            "author_agent": "author-1", "intake_path": "intake", "cards": [card],
            "tree_axis_comparison": {
                "sources": ["L05", "L06", "L07"], "resolved": False,
                "statement": "The available extracts do not justify treating these axes as the same intervention.",
            },
            "prediction_assessments": [{
                "id": "P2", "author_outcome": "unresolved", "observation_ids": ["L01"],
                "reason": "One card cannot settle the tree-axis comparison.",
            }],
        }
        registry = {"sources": [
            {"id": source_id, "title": "Title", "authors": ["Author"], "version": "published"}
            for source_id in ("L01", "L02", "L03", "L04", "L05", "L06", "L07", "L08")
        ]}
        with self.assertRaisesRegex(AuditError, "Missing baseline"):
            audit_literature(self.root, document, registry, index, records)
        for source_id in ("L02", "L03", "L04", "L05", "L06", "L07", "L08"):
            missing = dict(card)
            missing.update(source_id=source_id, access_status="unavailable", review_level="unavailable",
                           snapshot_path=None, snapshot_sha256=None, text_path=None, text_sha256=None,
                           locators=[], unavailable_reason="No permitted PDF was stored for this test card.")
            document["cards"].append(missing)
        audit_literature(self.root, document, registry, index, records)
        index["download_bytes"] = 0
        with self.assertRaisesRegex(AuditError, "Intake index"):
            audit_literature(self.root, document, registry, index, records)
