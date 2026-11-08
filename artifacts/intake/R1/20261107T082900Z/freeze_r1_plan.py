"""Add the real R1 input closure and mark the machine plan frozen."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[4]
plan_path = root / "research/plans/R1.json"
plan = json.loads(plan_path.read_text(encoding="utf-8"))
notes = json.loads((root / "research/stages/R1/inputs/legacy-notes.json").read_text(encoding="utf-8"))
cards = json.loads((root / "research/stages/R1/inputs/source-cards.json").read_text(encoding="utf-8"))
intake = cards["intake_path"]
attested = sorted({
    item["evidence_path"]
    for item in notes["observations"]
    if item.get("content_attestation", True) is not False
})
snapshots = []
for card in cards["cards"]:
    if card["access_status"] == "full_text":
        snapshots.extend((card["snapshot_path"], card["text_path"]))
code = [
    "scripts/r1_capture.py",
    "src/r1_audit.py",
    "src/resource_ledger.py",
    "src/research_workflow.py",
]
literature_inputs = [
    f"{intake}/acquisition.jsonl",
    f"{intake}/extract_text.py",
    f"{intake}/retrieve_sources.py",
    *snapshots,
]
validate_inputs = [
    "research/stages/R1/inputs/design-review.json",
    "research/stages/R1/inputs/legacy-notes.json",
    "research/stages/R1/inputs/source-cards.json",
    "research/stages/R1/inputs/intake-index.json",
    "research/stages/R1/inputs/R2-design.json",
    "research/stages/R1/inputs/dnn-directory-listing.json",
    "research/resources/ledger.jsonl",
    *attested,
    *literature_inputs,
]
check_inputs = [
    "scripts/r1_capture.py",
    "src/r1_audit.py",
    "src/resource_ledger.py",
    "tests/test_r1_audit.py",
    "tests/test_resource_ledger.py",
]
extra = {
    "legacy-evidence": code + attested,
    "literature": code + literature_inputs,
    "r2-design": code,
    "audit-validate": code + validate_inputs,
    "repository-check": check_inputs,
}
for task in plan["tasks"]:
    current = list(task["inputs"])
    for path in extra[task["id"]]:
        if path not in current:
            current.append(path)
        file_path = root / path
        if not file_path.is_file():
            raise SystemExit(f"missing input {path}")
    task["inputs"] = current
plan["status"] = "frozen"
plan["implementation_state"] = (
    "Inputs, capture scripts and the distinct-subagent design review exist. "
    "status=frozen records that closure. Execution is activated separately."
)
plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("frozen", {task["id"]: len(task["inputs"]) for task in plan["tasks"]})
