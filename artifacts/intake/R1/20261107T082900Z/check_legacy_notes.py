"""Preparation check for legacy notes. Not a stage task."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))

from src.r1_audit import INPUTS, audit_legacy, load_object, repo_file
audit = audit_legacy(
    root,
    load_object(repo_file(root, INPUTS["legacy_notes"])),
    load_object(repo_file(root, INPUTS["inventory"])),
)
print(f"observations={len(audit['observations'])} not_assessed={len(audit['negative_and_not_assessed'])}")
