"""Audit a CSV loss trajectory using a frozen calibration artifact.

Example: uv run python scripts/audit_loss.py --csv curves.csv
  --calibration calibration.json --output audit.json
CSV columns: iteration, train_loss; optionally eval_loss and train_errors.
Synthetic thresholds are descriptive on model trajectories, not p-values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.acceleration_audit import slope_scan


def audit(path: Path, calibration_path: Path) -> dict:
    raw = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    times = [float(r["iteration"]) for r in rows]
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Iteration must increase strictly")
    step = times[1] - times[0]
    if any(abs((b - a) - step) > 1e-9 * max(1, abs(step)) for a, b in zip(times, times[1:])):
        raise ValueError("The declared scan requires uniformly sampled iterations")
    calibration = json.loads(calibration_path.read_bytes())
    threshold = calibration["deployment_descriptive_threshold"]
    result = {"input_sha256": hashlib.sha256(raw).hexdigest(),
              "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
              "threshold_scope": "descriptive_only; synthetic-null exchangeability is not established",
              "double_descent_assessed": False, "mechanism_identified": False, "series": {}}
    for key in ("train_loss", "eval_loss"):
        if key not in rows[0]:
            continue
        score = slope_scan([float(r[key]) for r in rows], calibration["windows"])
        score["descriptive_candidate"] = score["score"] > threshold
        score["threshold"] = threshold
        score["iteration_at_candidate"] = times[score["t_star"]] if score["t_star"] is not None else None
        result["series"][key] = score
    if "train_loss" not in result["series"]:
        raise ValueError("CSV requires train_loss")
    if "train_errors" in rows[0]:
        errors = [int(r["train_errors"]) for r in rows]
        result["first_zero_train_errors_iteration"] = next((t for t, e in zip(times, errors) if e == 0), None)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit(args.csv, args.calibration), indent=2, allow_nan=False) + "\n", encoding="utf-8")
