"""Offline engineering verification. Does not train models or approve science."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research_claims import validate_claims
from src.research_workflow import Workflow


def structural_checks(root: Path) -> list[str]:
    problems = []
    required = ["AGENTS.md", "README.md", "docs/automation.md", "docs/research-program.md",
                "docs/legacy-assessment.md", "docs/literature/sources.json", "research/STATE.md",
                "research/stages.json", "research/claims.json", "uv.lock",
                ".cursor/rules/00-research-contract.mdc"]
    for name in required:
        if not (root / name).is_file():
            problems.append(f"Missing required control: {name}")
    if problems:
        return problems
    registry = json.loads((root / "research/stages.json").read_text(encoding="utf-8"))
    stages = registry["stages"]
    if len({s["id"] for s in stages}) != len(stages):
        problems.append("Duplicate stage IDs")
    seen = set()
    for stage in stages:
        if not set(stage["depends_on"]).issubset(seen):
            problems.append(f"Non-topological/missing dependency at {stage['id']}")
        seen.add(stage["id"])
        path = (root / stage["plan"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            problems.append(f"Invalid plan path: {stage['id']}")
            continue
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("stage_id") != stage["id"]:
            problems.append(f"Stage ID mismatch: {stage['id']}")
    claims = json.loads((root / "research/claims.json").read_text(encoding="utf-8"))
    problems.extend(validate_claims(root, claims))
    workflow = Workflow(root)
    for stage in stages:
        if stage.get("preparation_authorized") or stage.get("authorization_decision"):
            try:
                binding = workflow.validate_authorization(stage["id"])
                if stage.get("preparation_authorized") and binding is None:
                    problems.append(f"Preparation lacks bound authorization: {stage['id']}")
            except (ValueError, OSError, KeyError, TypeError) as exc:
                problems.append(f"Preparation authorization invalid: {stage['id']}: {exc}")
        if stage.get("execution_authorized"):
            try:
                workflow.validate_plan(stage["id"])
            except (ValueError, OSError, KeyError, TypeError) as exc:
                problems.append(f"Authorized plan invalid: {stage['id']}: {exc}")
    legacy = root / "research/evidence/legacy-2026-11-07"
    inventory = legacy / "inventory.json"
    if not inventory.is_file():
        problems.append("Missing historical inventory")
    else:
        for row in json.loads(inventory.read_text(encoding="utf-8"))["files"]:
            if not row["portable"]:
                continue
            path = (legacy / row["snapshot"]).resolve()
            if not path.is_relative_to(legacy) or not path.is_file():
                problems.append(f"Missing/unsafe historical snapshot: {row['path']}")
                continue
            if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                problems.append(f"Historical snapshot changed: {row['path']}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        problems = structural_checks(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems = [f"Invalid control document: {exc}"]
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, encoding="utf-8", errors="replace")
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    summary = {
        "schema_version": 1, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "structural_problems": problems, "test_command": command,
        "tests_returncode": result.returncode,
        "tests_stdout": result.stdout, "tests_stderr": result.stderr,
        "passed": not problems and result.returncode == 0,
        "scientific_validity": "not_assessed",
        "scope": "Offline infrastructure and protocol unit tests only; no new research experiment",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for problem in problems:
        print(problem, file=sys.stderr)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
