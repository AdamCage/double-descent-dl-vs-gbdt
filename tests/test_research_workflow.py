"""Offline contract tests: corrupted evidence must never unlock another stage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.research_workflow import Workflow, WorkflowError, confined, load_json, sha256, write_json


class WorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "research/plans").mkdir(parents=True)
        (self.root / "input.txt").write_text("source input\n", encoding="utf-8")
        (self.root / "report.md").write_text("Engineering smoke report.\n", encoding="utf-8")
        self.registry = {"schema_version": 1, "stages": [
            {"id": "R0", "depends_on": [], "plan": "research/plans/R0.json", "kind": "engineering"},
            {"id": "R1", "depends_on": ["R0"], "plan": "research/plans/R1.json", "kind": "science"},
        ]}
        write_json(self.root / "research/stages.json", self.registry)
        self.plan = {
            "schema_version": 1, "stage_id": "R0", "question": "Does the ledger work?",
            "scope": "Offline engineering verification", "budget": {"max_wall_seconds": 20},
            "tasks": [{"id": "smoke", "argv": ["{python}", "-c", "from pathlib import Path; Path('answer.txt').write_text('42'); print('recorded')"],
                       "inputs": ["input.txt"], "outputs": ["answer.txt"], "timeout_seconds": 10}],
            "exit_criteria": [{"id": "C1", "description": "Output exists with intact evidence", "task_ids": ["smoke"]}],
            "predictions": [{"id": "P1", "statement": "Offline task succeeds"}],
        }
        self.save_plan()
        self.workflow = Workflow(self.root)

    def save_plan(self) -> None:
        write_json(self.root / "research/plans/R0.json", self.plan)

    def run_smoke(self) -> Path:
        result = self.workflow.run("R0")
        return self.workflow.attempt("R0", result["attempt_id"])

    def accepted_review(self, attempt: Path, stage: str = "R0") -> Path:
        review = self.workflow.review_template(stage, "report.md", attempt.name)
        review["reviewer"].update(name="test reviewer", harness="offline-test")
        review["criteria"][0].update(verdict="pass", evidence=["smoke/answer.txt"])
        review["predictions"][0].update(outcome="Observed", evidence=["smoke/answer.txt"])
        review.update(verdict="accept", limitations=["Engineering test only; no scientific finding"])
        path = self.root / "scored-review.json"
        write_json(path, review)
        return path

    def test_complete_run_review_close_unlocks_next_without_scientific_approval(self) -> None:
        self.assertEqual(self.workflow.next_stage()["next"]["id"], "R0")
        attempt = self.run_smoke()
        self.assertEqual((attempt / "inputs/input.txt").read_text(encoding="utf-8"), "source input\n")
        result = self.workflow.verify("R0")
        self.assertGreaterEqual(result["checked_files"], 3)
        self.workflow.review("R0", self.accepted_review(attempt), attempt.name)
        self.workflow.close("R0", attempt.name)
        self.assertEqual(self.workflow.verify_closure("R0")["scientific_validity"], "not_assessed")
        self.assertEqual(self.workflow.next_stage()["next"]["id"], "R1")
        with self.assertRaisesRegex(WorkflowError, "not authorized"):
            self.workflow.run("R1")

    def bind_authorization(self, stage: str = "R0") -> dict[str, str]:
        decision = self.root / "research/charters/autonomy.md"
        decision.parent.mkdir(parents=True, exist_ok=True)
        decision.write_text("Bounded autonomous preparation and execution; no scientific result accepted.\n", encoding="utf-8")
        binding = {"path": "research/charters/autonomy.md", "sha256": sha256(decision)}
        entry = next(item for item in self.registry["stages"] if item["id"] == stage)
        entry["authorization_decision"] = binding
        write_json(self.root / "research/stages.json", self.registry)
        self.workflow = Workflow(self.root)
        return binding

    def test_historical_nonaccepting_review_preserves_retry_history(self) -> None:
        attempt = self.run_smoke()
        path = self.accepted_review(attempt)
        review = load_json(path)
        review.update(verdict="revise")
        review["criteria"][0]["verdict"] = "partial"
        write_json(path, review)
        (self.root / "input.txt").write_text("fixed source for a new attempt\n", encoding="utf-8")
        with self.assertRaises(WorkflowError):
            self.workflow.review("R0", path, attempt.name)
        self.assertTrue(self.workflow.review("R0", path, attempt.name, historical=True)["review_recorded"])
        self.workflow._check_review("R0", attempt, load_json(attempt / "review.json"), live_sources=False)
        with self.assertRaises(WorkflowError):
            self.workflow.close("R0", attempt.name)

    def test_historical_review_cannot_admit_an_accepting_decision(self) -> None:
        attempt = self.run_smoke()
        with self.assertRaisesRegex(WorkflowError, "only permits revise or reject"):
            self.workflow.review("R0", self.accepted_review(attempt), attempt.name, historical=True)
        self.assertFalse((attempt / "review.json").exists())

    def test_historical_nonaccepting_review_still_rejects_corrupt_evidence(self) -> None:
        attempt = self.run_smoke()
        path = self.accepted_review(attempt)
        review = load_json(path)
        review["verdict"] = "reject"
        write_json(path, review)
        (attempt / "tasks/smoke/answer.txt").write_text("corrupted", encoding="utf-8")
        with self.assertRaises(WorkflowError):
            self.workflow.review("R0", path, attempt.name, historical=True)
        self.assertFalse((attempt / "review.json").exists())

    def test_preparation_action_is_distinct_from_execution_and_unmet_dependencies(self) -> None:
        self.registry["stages"][0].update(execution_authorized=False, preparation_authorized=False)
        self.registry["stages"][1].update(execution_authorized=False, preparation_authorized=True)
        write_json(self.root / "research/stages.json", self.registry)
        self.workflow = Workflow(self.root)
        self.assertEqual(self.workflow.next_stage()["recommended_action"], "await_authorization")
        self.registry["stages"][0]["preparation_authorized"] = True
        write_json(self.root / "research/stages.json", self.registry)
        self.workflow = Workflow(self.root)
        next_stage = self.workflow.next_stage()
        self.assertEqual(next_stage["recommended_action"], "prepare")
        self.assertTrue(next_stage["next"]["preparation_authorized"])
        self.assertFalse(next_stage["next"]["execution_authorized"])
        with self.assertRaisesRegex(WorkflowError, "not authorized"):
            self.workflow.run("R0")
        waiting = self.workflow.status()["stages"][1]
        self.assertEqual(waiting["state"], "waiting")
        self.assertEqual(waiting["recommended_action"], "wait_for_dependencies")
        self.registry["stages"][0]["execution_authorized"] = True
        write_json(self.root / "research/stages.json", self.registry)
        self.assertEqual(Workflow(self.root).next_stage()["recommended_action"], "execute")

    def test_optional_authority_flags_must_be_boolean(self) -> None:
        for flag in ("preparation_authorized", "execution_authorized"):
            for value in ("true", 1, None):
                with self.subTest(flag=flag, value=value):
                    entry = dict(self.registry["stages"][0], **{flag: value})
                    write_json(self.root / "research/stages.json", {"schema_version": 1, "stages": [entry]})
                    with self.assertRaisesRegex(WorkflowError, "must be boolean"):
                        Workflow(self.root)

    def test_invalid_authorization_binding_is_rejected_without_reading_it(self) -> None:
        for binding in ({"path": "../outside.md", "sha256": "0" * 64},
                        {"path": ".env", "sha256": "0" * 64},
                        {"path": "input.txt", "sha256": "invalid"},
                        {"path": "input.txt", "sha256": "0" * 64, "approved": True}):
            with self.subTest(binding=binding):
                self.registry["stages"][0]["authorization_decision"] = binding
                write_json(self.root / "research/stages.json", self.registry)
                with self.assertRaises(WorkflowError):
                    Workflow(self.root)

    def test_changed_authorization_blocks_validation_run_and_next(self) -> None:
        binding = self.bind_authorization()
        self.assertEqual(self.workflow.validate_authorization("R0"), binding)
        self.assertEqual(self.workflow.next_stage()["next"]["authorization_decision"], binding)
        (self.root / binding["path"]).write_text("Unrecorded wider authorization", encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "Authorization decision changed"):
            self.workflow.validate_plan("R0")
        with self.assertRaisesRegex(WorkflowError, "Authorization decision changed"):
            self.workflow.run("R0")
        row = self.workflow.status()["stages"][0]
        self.assertEqual((row["state"], row["recommended_action"]), ("invalid", "escalate"))
        self.assertIn("Authorization decision changed", row["authorization_error"])
        self.assertEqual(self.workflow.next_stage(), {
            "next": None, "recommended_action": "escalate", "blocking_stages": ["R0"]})
        self.assertFalse((self.root / "artifacts/research/R0").exists())

    def test_authorization_is_snapshotted_automatically_and_history_is_self_contained(self) -> None:
        binding = self.bind_authorization()
        attempt = self.run_smoke()
        manifest = load_json(attempt / "manifest.json")
        self.assertEqual(manifest["authorization_decision"], binding)
        self.assertNotIn(binding["path"], self.plan["tasks"][0]["inputs"])
        self.assertEqual(manifest["input_hashes"][binding["path"]], binding["sha256"])
        self.assertEqual(sha256(attempt / "inputs" / binding["path"]), binding["sha256"])
        self.workflow.review("R0", self.accepted_review(attempt))
        self.workflow.close("R0")
        (self.root / binding["path"]).unlink()
        self.assertTrue(self.workflow.verify("R0", live_sources=False)["verified"])
        self.assertEqual(self.workflow.verify_closure("R0")["status"], "operationally_closed")
        self.assertEqual(self.workflow.status()["stages"][0]["state"], "closed")
        with self.assertRaises(WorkflowError):
            self.workflow.verify("R0")
        (attempt / "inputs" / binding["path"]).write_text("Corrupt frozen authority", encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "Input snapshot changed"):
            self.workflow.verify("R0", live_sources=False)

    def test_authorization_cannot_be_removed_from_attempt_inventory(self) -> None:
        binding = self.bind_authorization()
        attempt = self.run_smoke()
        manifest = load_json(attempt / "manifest.json")
        manifest["input_hashes"].pop(binding["path"])
        write_json(attempt / "manifest.json", manifest)
        with self.assertRaisesRegex(WorkflowError, "does not bind the authorization"):
            self.workflow.verify("R0", live_sources=False)

    def test_preparation_and_delegated_activation_keep_science_freeze_gate(self) -> None:
        self.close_chain_stage("R0")
        self.registry["stages"][1].update(preparation_authorized=True, execution_authorized=False)
        self.bind_authorization("R1")
        plan = dict(self.plan, stage_id="R1", status="draft")
        write_json(self.root / "research/plans/R1.json", plan)
        self.assertEqual(self.workflow.next_stage()["recommended_action"], "prepare")
        with self.assertRaisesRegex(WorkflowError, "not authorized"):
            self.workflow.run("R1")
        self.registry["stages"][1]["execution_authorized"] = True
        write_json(self.root / "research/stages.json", self.registry)
        self.workflow = Workflow(self.root)
        with self.assertRaisesRegex(WorkflowError, "frozen plan"):
            self.workflow.run("R1")
        plan["status"] = "frozen"
        write_json(self.root / "research/plans/R1.json", plan)
        self.assertTrue(self.workflow.validate_plan("R1")["valid"])
        self.assertEqual(self.workflow.run("R1")["status"], "complete")

    def bind_subagent_review_policy(self) -> dict[str, str]:
        decision = self.root / "research/charters/autonomy.json"
        write_json(decision, {"require_distinct_stage_review_subagent": True})
        binding = {"path": "research/charters/autonomy.json", "sha256": sha256(decision)}
        self.registry["stages"][0]["authorization_decision"] = binding
        write_json(self.root / "research/stages.json", self.registry)
        self.workflow = Workflow(self.root)
        return binding

    def test_chartered_cursor_review_requires_a_distinct_reviewer_subagent(self) -> None:
        self.bind_subagent_review_policy()
        attempt = self.run_smoke()
        template = self.workflow.review_template("R0", "report.md", attempt.name)
        self.assertEqual((template["author_agent"], template["reviewer_agent"], template["review_mode"]),
                         ("", "", "subagent_review"))
        path = self.accepted_review(attempt)
        review = load_json(path)
        review["reviewer"]["harness"] = "cursor"
        for author, reviewer in (("", ""), ("worker-1", "worker-1"), ("worker-1", " WORKER-1 ")):
            review.update(author_agent=author, reviewer_agent=reviewer)
            write_json(path, review)
            with self.assertRaisesRegex(WorkflowError, "distinct author_agent"):
                self.workflow.review("R0", path)
        review.update(author_agent="worker-1", reviewer_agent="reviewer-2", review_mode="self_review")
        write_json(path, review)
        with self.assertRaisesRegex(WorkflowError, "review_mode=subagent_review"):
            self.workflow.review("R0", path)
        review["review_mode"] = "subagent_review"
        write_json(path, review)
        self.assertTrue(self.workflow.review("R0", path)["review_recorded"])
        self.assertEqual(self.workflow.close("R0")["status"], "operationally_closed")

    def test_codex_cannot_bypass_distinct_subagent_policy_or_historical_check(self) -> None:
        binding = self.bind_subagent_review_policy()
        attempt = self.run_smoke()
        path = self.accepted_review(attempt)
        review = load_json(path)
        review["reviewer"]["harness"] = "codex"
        for author, reviewer in (("", ""), ("worker", " WORKER ")):
            review.update(author_agent=author, reviewer_agent=reviewer)
            write_json(path, review)
            with self.assertRaisesRegex(WorkflowError, "distinct author_agent"):
                self.workflow.review("R0", path)
        review.update(author_agent="codex-author", reviewer_agent="codex-reviewer", review_mode="self_review")
        write_json(path, review)
        with self.assertRaisesRegex(WorkflowError, "review_mode=subagent_review"):
            self.workflow.review("R0", path)
        review["review_mode"] = "subagent_review"
        write_json(path, review)
        self.workflow.review("R0", path)
        self.workflow.close("R0")
        write_json(self.root / binding["path"], {"require_distinct_stage_review_subagent": False})
        self.assertEqual(self.workflow.verify_closure("R0")["status"], "operationally_closed")
        review["reviewer_agent"] = review["author_agent"]
        write_json(attempt / "review.json", review)
        closure = load_json(self.workflow.closure_path("R0"))
        closure["review_sha256"] = sha256(attempt / "review.json")
        write_json(self.workflow.closure_path("R0"), closure)
        with self.assertRaisesRegex(WorkflowError, "distinct author_agent"):
            self.workflow.verify_closure("R0")

    def test_unknown_harness_cannot_bypass_charter_but_actual_external_review_is_allowed(self) -> None:
        self.bind_subagent_review_policy()
        attempt = self.run_smoke()
        path = self.accepted_review(attempt)
        with self.assertRaisesRegex(WorkflowError, "Cursor subagent or an actual human/Codex"):
            self.workflow.review("R0", path)
        review = load_json(path)
        review["reviewer"]["harness"] = "human"
        review["review_mode"] = "external_review"
        write_json(path, review)
        self.assertTrue(self.workflow.review("R0", path)["review_recorded"])

    def test_historical_subagent_gate_uses_frozen_charter(self) -> None:
        binding = self.bind_subagent_review_policy()
        attempt = self.run_smoke()
        path = self.accepted_review(attempt)
        review = load_json(path)
        review["reviewer"]["harness"] = "cursor"
        review.update(author_agent="worker-1", reviewer_agent="reviewer-2")
        write_json(path, review)
        self.workflow.review("R0", path)
        self.workflow.close("R0")
        write_json(self.root / binding["path"], {"require_distinct_stage_review_subagent": False})
        self.assertEqual(self.workflow.verify_closure("R0")["status"], "operationally_closed")
        frozen_review = attempt / "review.json"
        review["reviewer_agent"] = review["author_agent"]
        write_json(frozen_review, review)
        closure = load_json(self.workflow.closure_path("R0"))
        closure["review_sha256"] = sha256(frozen_review)
        write_json(self.workflow.closure_path("R0"), closure)
        with self.assertRaisesRegex(WorkflowError, "distinct author_agent"):
            self.workflow.verify_closure("R0")

    def test_changed_output_cannot_verify_or_close(self) -> None:
        attempt = self.run_smoke()
        (attempt / "tasks/smoke/answer.txt").write_text("altered", encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "Output changed"):
            self.workflow.verify("R0")

    def test_missing_output_is_failed_attempt_with_logs_and_manifest(self) -> None:
        self.plan["tasks"][0]["outputs"] = ["missing.txt"]
        self.save_plan()
        with self.assertRaisesRegex(WorkflowError, "evidence retained"):
            self.workflow.run("R0")
        attempt = self.workflow.attempt("R0")
        manifest = load_json(attempt / "manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["tasks"][0]["exit_code"], 0)
        self.assertTrue((attempt / "tasks/smoke/stdout.log").is_file())
        self.assertFalse((self.root / ".research/workflow/active.lock").exists())

    def test_failed_exit_is_retained_and_not_reused(self) -> None:
        self.plan["tasks"][0]["argv"] = ["{python}", "-c", "raise SystemExit(7)"]
        self.save_plan()
        with self.assertRaises(WorkflowError):
            self.workflow.run("R0")
        first = self.workflow.attempt("R0")
        self.assertEqual(load_json(first / "manifest.json")["tasks"][0]["exit_code"], 7)
        with self.assertRaises(WorkflowError):
            self.workflow.run("R0")
        self.assertEqual(len(list(first.parent.glob("*/manifest.json"))), 2)

    def test_timeout_stops_task_and_records_failure(self) -> None:
        self.plan["tasks"][0].update(argv=["{python}", "-c", "import time; time.sleep(20)"], timeout_seconds=0.15)
        self.save_plan()
        with self.assertRaises(WorkflowError):
            self.workflow.run("R0")
        record = load_json(self.workflow.attempt("R0") / "manifest.json")["tasks"][0]
        self.assertTrue(record["timed_out"])

    def test_live_source_change_requires_rerun_but_history_still_verifies(self) -> None:
        attempt = self.run_smoke()
        (self.root / "input.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "Live input changed"):
            self.workflow.verify("R0")
        self.assertTrue(self.workflow.verify("R0", attempt.name, live_sources=False)["verified"])
        (attempt / "inputs/input.txt").write_text("corrupt snapshot", encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "snapshot changed"):
            self.workflow.verify("R0", attempt.name, live_sources=False)

    def test_review_cannot_approve_an_edited_report(self) -> None:
        attempt = self.run_smoke()
        review_path = self.accepted_review(attempt)
        (self.root / "report.md").write_text("New conclusion", encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "report changed"):
            self.workflow.review("R0", review_path)

    def test_review_must_score_registered_criteria_and_cite_outputs(self) -> None:
        attempt = self.run_smoke()
        path = self.accepted_review(attempt)
        review = load_json(path)
        review["criteria"][0]["evidence"] = ["unregistered/file.txt"]
        write_json(path, review)
        with self.assertRaisesRegex(WorkflowError, "registered output evidence"):
            self.workflow.review("R0", path)
        review["criteria"] = []
        write_json(path, review)
        with self.assertRaisesRegex(WorkflowError, "every registered criteria"):
            self.workflow.review("R0", path)

    def test_stage_dependencies_cannot_be_skipped(self) -> None:
        self.registry["stages"][1]["execution_authorized"] = True
        write_json(self.root / "research/stages.json", self.registry)
        workflow = Workflow(self.root)
        with self.assertRaisesRegex(WorkflowError, "Dependency is not closed"):
            workflow.run("R1")

    def test_duplicate_matrix_empty_outputs_and_path_escape_are_rejected(self) -> None:
        self.plan["tasks"].append(self.plan["tasks"][0].copy())
        self.save_plan()
        with self.assertRaisesRegex(WorkflowError, "Duplicate task"):
            self.workflow.validate_plan("R0")
        self.plan["tasks"].pop()
        self.plan["tasks"][0]["outputs"] = []
        self.save_plan()
        with self.assertRaisesRegex(WorkflowError, "nonempty outputs"):
            self.workflow.validate_plan("R0")
        with self.assertRaises(WorkflowError):
            confined(self.root, "../outside.txt")

    def test_lock_prevents_duplicate_execution(self) -> None:
        with self.workflow.lock("R0"):
            with self.assertRaisesRegex(WorkflowError, "lock exists"):
                self.workflow.run("R0")

    def test_registry_cycles_and_symlink_inputs_are_rejected(self) -> None:
        self.registry["stages"][0]["depends_on"] = ["R1"]
        write_json(self.root / "research/stages.json", self.registry)
        with self.assertRaisesRegex(WorkflowError, "cycle"):
            Workflow(self.root)
        link = self.root / "shortcut.txt"
        try:
            link.symlink_to(self.root / "input.txt")
        except OSError:
            return  # Windows without symlink privilege still tests lexical confinement above.
        with self.assertRaisesRegex(WorkflowError, "Symlinks"):
            confined(self.root, "shortcut.txt", must_exist=True)

    def test_closed_record_hash_tampering_is_detected(self) -> None:
        attempt = self.run_smoke()
        self.workflow.review("R0", self.accepted_review(attempt))
        self.workflow.close("R0")
        review_path = attempt / "review.json"
        review = load_json(review_path)
        review["limitations"].append("unrecorded edit")
        write_json(review_path, review)
        with self.assertRaisesRegex(WorkflowError, "Closed evidence changed"):
            self.workflow.verify_closure("R0")

    def test_retry_cap_counts_interrupted_attempt_directories(self) -> None:
        self.plan["budget"]["max_attempts"] = 1
        self.save_plan()
        interrupted = self.root / "artifacts/research/R0/interrupted"
        interrupted.mkdir(parents=True)
        with self.assertRaisesRegex(WorkflowError, "max_attempts exhausted"):
            self.workflow.run("R0")
        self.assertEqual(len(list(interrupted.parent.iterdir())), 1)

    def test_cumulative_wall_budget_counts_previous_failed_runs(self) -> None:
        self.plan["budget"]["total_wall_seconds"] = 2
        self.save_plan()
        write_json(self.root / "artifacts/research/R0/previous/manifest.json",
                   {"elapsed_seconds": 2.1, "status": "failed"})
        with self.assertRaisesRegex(WorkflowError, "total_wall_seconds exhausted"):
            self.workflow.run("R0")

    def bind_resource_policy(self, *, total_wall: float = 100) -> dict[str, str]:
        decision = self.root / "research/charters/resources.json"
        write_json(decision, {
            "budgets": {
                "total_runner_wall_seconds": total_wall, "max_stage_wall_seconds": 40,
                "new_storage_bytes": 1000000, "public_dataset_download_bytes": 10,
            },
            "r1_budgets": {
                "document_download_bytes": 100, "max_document_bytes": 50,
                "new_storage_bytes": 1000000, "dataset_download_bytes": 0,
                "max_http_requests": 48,
            },
        })
        binding = {"path": "research/charters/resources.json", "sha256": sha256(decision)}
        self.registry["stages"][0]["authorization_decision"] = binding
        write_json(self.root / "research/stages.json", self.registry)
        self.workflow = Workflow(self.root)
        return binding

    def test_remaining_global_wall_must_cover_the_declared_attempt(self) -> None:
        self.bind_resource_policy()
        previous = self.root / "artifacts/research/OTHER/failed/manifest.json"
        write_json(previous, {"elapsed_seconds": 90, "status": "failed"})
        with self.assertRaisesRegex(WorkflowError, "remaining wall budget"):
            self.workflow.run("R0")
        self.assertFalse((self.root / "artifacts/research/R0").exists())
        self.assertEqual(load_json(previous)["elapsed_seconds"], 90)

    def test_unmeasured_or_contradictory_global_attempt_blocks_execution(self) -> None:
        from src.resource_ledger import append_entry

        self.bind_resource_policy()
        previous = self.root / "artifacts/research/OTHER/interrupted"
        previous.mkdir(parents=True)
        with self.assertRaisesRegex(WorkflowError, "resource preflight is not ready"):
            self.workflow.run("R0")
        write_json(previous / "manifest.json", {"elapsed_seconds": 1, "status": "failed"})
        append_entry(self.root, {"kind": "attempt", "stage_id": "OTHER", "attempt_id": "interrupted",
                                 "elapsed_seconds": 2, "status": "failed"})
        with self.assertRaisesRegex(WorkflowError, "resource preflight is not ready"):
            self.workflow.run("R0")
        self.assertFalse((self.root / "artifacts/research/R0").exists())

    def test_global_storage_exhaustion_blocks_before_attempt_creation(self) -> None:
        self.bind_resource_policy()
        (self.root / "research/evidence").mkdir()
        (self.root / "research/evidence/full.bin").write_bytes(b"x" * 1000001)
        with self.assertRaisesRegex(WorkflowError, "resource preflight is not ready"):
            self.workflow.run("R0")
        self.assertFalse((self.root / "artifacts/research/R0").exists())

    def test_success_and_failure_are_automatically_charged_without_losing_history(self) -> None:
        from src.resource_ledger import ledger_entries, report

        binding = self.bind_resource_policy()
        first = self.run_smoke()
        self.plan["tasks"][0]["argv"] = ["{python}", "-c", "raise SystemExit(7)"]
        self.save_plan()
        with self.assertRaisesRegex(WorkflowError, "evidence retained"):
            self.workflow.run("R0")
        attempts = list(first.parent.glob("*/manifest.json"))
        entries = ledger_entries(self.root)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(entries), 2)
        for path in attempts:
            manifest = load_json(path)
            entry = next(item for item in entries if item["attempt_id"] == manifest["attempt_id"])
            self.assertEqual(entry["elapsed_seconds"], manifest["elapsed_seconds"])
            self.assertEqual(entry["status"], manifest["status"])
        self.assertEqual({item["status"] for item in entries}, {"complete", "failed"})
        resources = report(self.root, charter_path=binding["path"], expected_sha256=binding["sha256"])
        self.assertEqual(resources["reconciliation_problems"], [])
        self.assertAlmostEqual(resources["charged_wall_seconds"], sum(item["elapsed_seconds"] for item in entries))
        self.assertTrue(self.workflow.verify("R0", first.name, live_sources=False)["verified"])

    def test_declared_but_malformed_global_budget_is_not_a_legacy_exemption(self) -> None:
        binding = self.bind_resource_policy()
        decision = self.root / binding["path"]
        write_json(decision, {"budgets": {}})
        self.registry["stages"][0]["authorization_decision"]["sha256"] = sha256(decision)
        write_json(self.root / "research/stages.json", self.registry)
        self.workflow = Workflow(self.root)
        with self.assertRaisesRegex(WorkflowError, "Cannot validate cumulative resource budget"):
            self.workflow.run("R0")
        self.assertFalse((self.root / "artifacts/research/R0").exists())

    def test_finite_budgets_and_secret_input_guards(self) -> None:
        for value in (float("inf"), float("nan"), True, -1):
            self.plan["budget"]["max_wall_seconds"] = value
            # JSON intentionally permits these invalid external values for validation.
            (self.root / "research/plans/R0.json").write_text(json.dumps(self.plan), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "max_wall_seconds"):
                self.workflow.validate_plan("R0")
        self.plan["budget"]["max_wall_seconds"] = 20
        for value in (".env", ".env.local", ".git/config", ".venv/pyvenv.cfg"):
            target = self.root / value
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("test-only dummy content", encoding="utf-8")
            self.plan["tasks"][0]["inputs"] = [value]
            self.save_plan()
            with self.assertRaisesRegex(WorkflowError, "cannot be snapshotted"):
                self.workflow.validate_plan("R0")

    def test_historical_closure_checks_frozen_report_and_binds_review(self) -> None:
        attempt = self.run_smoke()
        self.workflow.review("R0", self.accepted_review(attempt))
        self.workflow.close("R0")
        (self.root / "report.md").unlink()
        self.assertEqual(self.workflow.verify_closure("R0")["status"], "operationally_closed")
        (attempt / "report.md").write_text("corrupt", encoding="utf-8")
        self.assertEqual(self.workflow.status()["stages"][0]["state"], "invalid")
        self.assertEqual(self.workflow.status()["stages"][1]["state"], "waiting")

    def test_keyboard_interrupt_terminates_owned_process_tree(self) -> None:
        child = MagicMock()
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        child.poll.return_value = None
        child.returncode = -9
        with patch("src.research_workflow.subprocess.Popen", return_value=child), \
             patch("src.research_workflow.subprocess.run") as runner, \
             patch("src.research_workflow.os.killpg", create=True) as killpg:
            with self.assertRaises(KeyboardInterrupt):
                self.workflow._execute(["unused"], self.root, 10)
            import os
            if os.name == "nt":
                self.assertEqual(runner.call_args.args[0][0], "taskkill")
            else:
                killpg.assert_called_once()
            self.assertEqual(child.wait.call_count, 2)

    def setup_three_stage_chain(self) -> None:
        self.registry["stages"][1]["execution_authorized"] = True
        self.registry["stages"].append({"id": "R2", "depends_on": ["R1"],
                                        "plan": "research/plans/R2.json", "kind": "science",
                                        "execution_authorized": True})
        write_json(self.root / "research/stages.json", self.registry)
        for stage in ("R1", "R2"):
            plan = dict(self.plan, stage_id=stage, status="frozen")
            write_json(self.root / f"research/plans/{stage}.json", plan)
        self.workflow = Workflow(self.root)

    def close_chain_stage(self, stage: str) -> Path:
        result = self.workflow.run(stage)
        attempt = self.workflow.attempt(stage, result["attempt_id"])
        self.workflow.review(stage, self.accepted_review(attempt, stage), attempt.name)
        self.workflow.close(stage, attempt.name)
        return attempt

    def test_corrupt_ancestor_invalidates_descendant_closure_and_blocks_third_stage(self) -> None:
        self.setup_three_stage_chain()
        first = self.close_chain_stage("R0")
        second = self.close_chain_stage("R1")
        dependency_hashes = load_json(second / "manifest.json")["dependency_closure_hashes"]
        self.assertEqual(set(dependency_hashes), {"R0"})
        self.assertEqual(load_json(self.workflow.closure_path("R1"))["dependency_closure_hashes"], dependency_hashes)
        (first / "tasks/smoke/answer.txt").write_text("corrupted ancestor", encoding="utf-8")
        states = {row["id"]: row["state"] for row in self.workflow.status()["stages"]}
        self.assertEqual(states, {"R0": "invalid", "R1": "invalid", "R2": "waiting"})
        with self.assertRaisesRegex(WorkflowError, "Output changed"):
            self.workflow.run("R2")

    def test_dependency_closure_swap_is_rejected_even_if_ancestor_still_valid(self) -> None:
        self.setup_three_stage_chain()
        self.close_chain_stage("R0")
        self.close_chain_stage("R1")
        path = self.workflow.closure_path("R0")
        closure = load_json(path)
        closure["closed_at"] = "changed after downstream acceptance"
        write_json(path, closure)
        self.assertEqual(self.workflow.verify_closure("R0")["status"], "operationally_closed")
        with self.assertRaisesRegex(WorkflowError, "Dependency closure changed"):
            self.workflow.verify_closure("R1")
        with self.assertRaisesRegex(WorkflowError, "Dependency closure changed"):
            self.workflow.run("R2")

    def test_dependency_closure_swap_between_execution_and_review_is_rejected(self) -> None:
        self.setup_three_stage_chain()
        self.close_chain_stage("R0")
        result = self.workflow.run("R1")
        attempt = self.workflow.attempt("R1", result["attempt_id"])
        review_path = self.accepted_review(attempt, "R1")
        path = self.workflow.closure_path("R0")
        closure = load_json(path)
        closure["closed_at"] = "swapped during downstream review"
        write_json(path, closure)
        with self.assertRaisesRegex(WorkflowError, "Dependency closure changed"):
            self.workflow.review("R1", review_path, attempt.name)


if __name__ == "__main__":
    unittest.main()
