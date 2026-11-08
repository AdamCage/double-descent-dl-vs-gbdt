"""Preparation check for the R2 proposal."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
from src.r1_audit import INPUTS, audit_r2, load_object, repo_file

audit = audit_r2(load_object(repo_file(root, INPUTS["r2_design"])))
print(audit["readiness"], "interventions", len(audit["interventions"]), "models", audit["models_executed"])
