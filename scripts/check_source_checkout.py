"""Check a source checkout without pretending that Release evidence was restored.

The full local gate remains scripts/check_repository.py. This command checks
source/control integrity and runs the same unit suite. Artifact input paths
and claim bindings are inventoried, never counted as verified evidence here.
It does not validate an executable stage or admit a scientific claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_repository import run_unit_tests
from src.research_claims import _cursor_authorization, _cursor_review
from src.research_workflow import Workflow, confined, identifier, load_json, sha256


REQUIRED = (
    "AGENTS.md", "README.md", "docs/automation.md", "docs/research-program.md",
    "docs/legacy-assessment.md", "docs/literature/sources.json", "research/STATE.md",
    "research/stages.json", "research/claims.json", "uv.lock",
    ".cursor/rules/00-research-contract.mdc",
)
RELEASE_MANIFEST = "docs/releases/research-cot-audit-2026-11-08-assets.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def external_artifact(path: str) -> bool:
    return path.startswith("artifacts/") or (
        path.startswith("research/evidence/") and path.endswith((".zip", ".tar.gz", ".tgz")))


class SourceAudit:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.problems: list[str] = []
        self.unverified: list[dict] = []
        self.checked: set[str] = set()

    def input(self, relative: str, label: str, expected: str | None = None) -> Path:
        path = confined(self.root, relative)
        Workflow._safe_input(relative)
        if external_artifact(relative):
            self.unverified.append({"path": relative, "sha256": expected,
                                    "reference": label,
                                    "reason": "Release/live artifact excluded from source-only verification"})
        elif not path.is_file():
            self.problems.append(f"{label}: missing source/control file: {relative}")
        elif expected is not None and sha256(path) != expected:
            self.problems.append(f"{label}: source/control SHA-256 mismatch: {relative}")
        else:
            self.checked.add(relative)
        return path

    def binding(self, value: dict, label: str, *, source_required: bool = False) -> Path:
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError(f"{label}: expected exactly path and sha256")
        if not isinstance(value["sha256"], str) or not SHA256.fullmatch(value["sha256"]):
            raise ValueError(f"{label}: invalid SHA-256")
        path = self.input(value["path"], label, value["sha256"])
        if source_required and external_artifact(value["path"]):
            raise ValueError(f"{label}: review/authorization must be a source control file")
        return path

    def evidence(self, value: list, label: str) -> dict[str, str]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{label}: nonempty evidence list required")
        result = {}
        for binding in value:
            self.binding(binding, label)
            if binding["path"] in result:
                raise ValueError(f"{label}: duplicate evidence path")
            result[binding["path"]] = binding["sha256"]
        return result

    def claims(self) -> None:
        doc = load_json(self.root / "research/claims.json")
        if type(doc.get("schema_version")) is not int or doc["schema_version"] != 1:
            raise ValueError("claims: invalid schema_version")
        claims = doc.get("claims")
        if not isinstance(claims, list):
            raise ValueError("claims: expected claims list")
        if "autonomy_authorization" in doc:
            self.binding(doc["autonomy_authorization"], "claim authorization", source_required=True)
        authority = _cursor_authorization(self.root, doc, self.problems)
        seen = set()
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError("claims: each claim must be an object")
            claim_id = claim.get("id")
            if not isinstance(claim_id, str) or not claim_id.strip() or claim_id in seen:
                raise ValueError("claims: missing/duplicate claim ID")
            seen.add(claim_id)
            label = f"claim {claim_id}"
            statement = claim.get("statement")
            if not isinstance(statement, str) or not statement.strip():
                raise ValueError(f"{label}: statement required")
            scope = claim.get("scope")
            if not (isinstance(scope, str) and scope.strip() or isinstance(scope, dict) and scope):
                raise ValueError(f"{label}: scope required")
            limits = claim.get("limitations")
            if not isinstance(limits, list) or not limits or not all(
                    isinstance(x, str) and x.strip() for x in limits):
                raise ValueError(f"{label}: nonempty limitations required")
            evidence = self.evidence(claim.get("evidence"), label)
            decision_path = self.binding(claim.get("scientific_review"), label, source_required=True)
            decision = load_json(decision_path)
            if (decision.get("claim_id") != claim_id
                    or decision.get("statement_sha256") != hashlib.sha256(statement.encode("utf-8")).hexdigest()
                    or decision.get("verdict") != "accepted"):
                raise ValueError(f"{label}: decision statement/ID/verdict mismatch")
            reviewer = decision.get("reviewer")
            if not isinstance(reviewer, str) or not reviewer.strip():
                raise ValueError(f"{label}: reviewer label required")
            if decision.get("harness") == "cursor":
                _cursor_review(claim, decision, authority, label, self.problems)
            elif decision.get("harness") not in ("codex", "human"):
                raise ValueError(f"{label}: invalid review harness")
            if self.evidence(decision.get("evidence"), label + " decision") != evidence:
                raise ValueError(f"{label}: exact evidence metadata mismatch")

    def release_manifest(self) -> None:
        path = self.root / RELEASE_MANIFEST
        if not path.is_file():
            return  # The checker also works before preparing a release.
        manifest = load_json(path)
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("assets"), list):
            raise ValueError("release manifest: schema_version=1 and assets list required")
        seen = set()
        for asset in manifest["assets"]:
            confined(self.root, asset["path"])
            if asset["path"] in seen:
                raise ValueError("release manifest: duplicate asset path")
            seen.add(asset["path"])
            if (type(asset.get("bytes")) is not int or asset["bytes"] <= 0
                    or not isinstance(asset.get("sha256"), str) or not SHA256.fullmatch(asset["sha256"])):
                raise ValueError("release manifest: positive byte count and SHA-256 required")
        self.checked.add(RELEASE_MANIFEST)

    def run(self) -> dict:
        try:
            for relative in REQUIRED:
                self.input(relative, "required control")
            if not self.problems:
                workflow = Workflow(self.root)
                seen = set()
                for stage_id, stage in workflow.stages.items():
                    if not set(stage["depends_on"]).issubset(seen):
                        raise ValueError(f"Non-topological registry: {stage_id}")
                    seen.add(stage_id)
                    workflow.validate_authorization(stage_id)
                    if stage.get("preparation_authorized") and not stage.get("authorization_decision"):
                        raise ValueError(f"Preparation lacks bound authorization: {stage_id}")
                    plan = load_json(confined(self.root, stage["plan"], must_exist=True))
                    self.checked.add(stage["plan"])
                    if plan.get("stage_id") != stage_id or plan.get("schema_version") != 1:
                        raise ValueError(f"Plan identity mismatch: {stage_id}")
                    tasks = plan.get("tasks")
                    if not isinstance(tasks, list) or not tasks:
                        raise ValueError(f"Plan lacks task matrix: {stage_id}")
                    task_ids = set()
                    for task in tasks:
                        task_id = identifier(task.get("id"))
                        if task_id in task_ids:
                            raise ValueError(f"Duplicate task ID: {stage_id}/{task_id}")
                        task_ids.add(task_id)
                        for key in ("inputs", "outputs"):
                            values = task.get(key)
                            if not isinstance(values, list) or not values or len(set(values)) != len(values):
                                raise ValueError(f"Missing/duplicate {key}: {stage_id}/{task_id}")
                            for relative in values:
                                if key == "inputs":
                                    self.input(relative, f"task {stage_id}/{task_id}")
                                else:
                                    confined(self.root / "artifacts/research" / stage_id / "validation", relative)
                self.claims()
                legacy = self.root / "research/evidence/legacy-2026-11-07"
                inventory = load_json(legacy / "inventory.json")
                for row in inventory["files"]:
                    if row["portable"]:
                        path = confined(legacy, row["snapshot"], must_exist=True)
                        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                            raise ValueError(f"Historical snapshot changed: {row['path']}")
                self.release_manifest()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.problems.append(f"Invalid source/control document: {exc}")
        return {"source_problems": self.problems,
                "checked_source_files": sorted(self.checked),
                "unverified_artifact_references": self.unverified,
                "evidence_validation": "not_performed_source_checkout",
                "stage_execution_preflight": "not_performed",
                "scientific_claim_admission": "not_performed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = SourceAudit(root).run()
    command, result, count = run_unit_tests(root)
    if count is None or count == 0:
        audit["source_problems"].append("Unit test discovery must execute at least one test")
    summary = dict(audit, schema_version=1,
                   checked_utc=datetime.now(timezone.utc).isoformat(),
                   test_command=command, tests_run=count, tests_returncode=result.returncode,
                   tests_stdout=result.stdout, tests_stderr=result.stderr,
                   passed=not audit["source_problems"] and result.returncode == 0,
                   scientific_validity="not_assessed",
                   scope="Source/control checks and offline unit tests only; Release evidence bytes, full plan preflight, scientific claims and scientific validity are not validated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    for problem in audit["source_problems"]:
        print(problem, file=sys.stderr)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
