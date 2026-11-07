"""Preparation check for literature notes."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
from src.r1_audit import INPUTS, audit_literature, load_jsonl, load_object, repo_file

notes = load_object(repo_file(root, INPUTS["source_cards"]))
records = load_jsonl(repo_file(root, f"{notes['intake_path']}/acquisition.jsonl"))
audit = audit_literature(
    root, notes, load_object(repo_file(root, INPUTS["registry"])),
    load_object(repo_file(root, INPUTS["intake_index"])), records,
)
print("literature_ok", "unavailable", audit["unavailable"], "resolved", audit["tree_axis_comparison"]["resolved"])
