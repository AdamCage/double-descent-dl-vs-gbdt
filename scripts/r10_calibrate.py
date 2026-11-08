"""Execute the frozen synthetic calibration/evaluation matrix once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from src.acceleration_audit import analytic_curve, noisy_curve, slope_scan, rank_threshold, binomial_interval
from src.detector_catalog import local_velocity_change
from src.dynamics import describe_change


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(protocol: dict, out: Path, protocol_sha256: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows, curves, thresholds = [], [], {}
    groups = protocol["noise_conditions"]
    windows = protocol["windows"]
    nulls, alternatives = protocol["null_families"], protocol["alternative_families"]
    alpha = protocol["alpha"]
    for phase, families, count in (("calibration", nulls, protocol["calibration_count"]),
                                    ("evaluation", nulls + alternatives, protocol["evaluation_count"])):
        for gi, noise in enumerate(groups):
            for fi, family in enumerate(families):
                base, transition = analytic_curve(family, protocol["length"])
                for rep in range(count):
                    entropy = [protocol["salt"], 1 if phase == "calibration" else 2, gi, fi, rep]
                    y = noisy_curve(base, noise["sigma"], noise["rho"], entropy)
                    scan = slope_scan(y, windows)
                    index = len(curves)
                    curves.append(y)
                    row = {"curve_index": index, "phase": phase, "family": family,
                           "noise_id": noise["id"], "replicate": rep, "seed_entropy": entropy,
                           "known_transition": transition, "scan": scan}
                    if phase == "evaluation":
                        row["global"] = describe_change(y)
                        row["local"] = local_velocity_change(y, window=protocol["legacy_local_window"],
                                                              rss_improvement_min=.05)
                        row["scale_invariance_max_error"] = max(abs(slope_scan(scale * y + 3, windows)["score"] - scan["score"])
                                                                 for scale in protocol["amplitude_scales"])
                    rows.append(row)
            if phase == "calibration":
                per_family = {family: rank_threshold([r["scan"]["score"] for r in rows
                    if r["phase"] == phase and r["noise_id"] == noise["id"] and r["family"] == family], alpha)
                    for family in nulls}
                thresholds[noise["id"]] = {"threshold": max(x["threshold"] for x in per_family.values()),
                                           "per_family": per_family}
    summaries = []
    for noise in groups:
        threshold = thresholds[noise["id"]]["threshold"]
        for family in nulls + alternatives:
            subset = [r for r in rows if r["phase"] == "evaluation" and r["noise_id"] == noise["id"] and r["family"] == family]
            for r in subset:
                r["scan"]["candidate"] = r["scan"]["score"] > threshold
                r["scan"]["threshold"] = threshold
            for detector in ("global", "local", "scan"):
                selected = [r for r in subset if r[detector]["candidate"]]
                errors = [abs(r[detector]["t_star"] - r["known_transition"]) for r in selected if r["known_transition"] is not None]
                summaries.append({"noise_id": noise["id"], "family": family, "detector": detector,
                    "role": "null" if family in nulls else "alternative", "n": len(subset),
                    "flagged": len(selected), "rate": len(selected) / len(subset),
                    "pointwise_95_cp": binomial_interval(len(selected), len(subset)),
                    "median_absolute_localization_error_given_flag": float(np.median(errors)) if errors else None,
                    "localized_within_max_window": sum(e <= max(windows) for e in errors) / len(subset) if errors else 0.0})
    write_json(out / "calibration.json", {"schema_version": 1, "protocol_sha256": protocol_sha256,
        "windows": windows, "alpha": alpha, "thresholds": thresholds,
        "deployment_descriptive_threshold": max(x["threshold"] for x in thresholds.values()),
        "guarantee_scope": "Marginal rank bound for each included whole-curve null law; not conditional on calibration. No arbitrary model-trajectory guarantee."})
    with (out / "curves.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    np.savez_compressed(out / "curves.npz", loss=np.asarray(curves))
    summary = {"schema_version": 1, "phase": "independent_evaluation", "n_curves": len(rows),
        "n_calibration": sum(r["phase"] == "calibration" for r in rows),
        "n_evaluation": sum(r["phase"] == "evaluation" for r in rows),
        "protocol_sha256": protocol_sha256, "rates": summaries,
        "max_scale_invariance_error": max(r.get("scale_invariance_max_error", 0) for r in rows),
        "independent_unit": "whole generated trajectory; paired detectors and scaled copies are not additional repetitions",
        "interval_scope": "Pointwise exact binomial intervals, conditional on the shared frozen calibration. Not simultaneous intervals; calibration uncertainty is not included.",
        "mechanism_claim": False, "double_descent_declared": False, "scientific_validity": "not_assessed",
        "download_bytes": 0, "paid_api_calls": 0, "gpu_jobs": 0}
    write_json(out / "analysis.json", summary)
    write_json(out / "environment.json", {"python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(), "torch_imported": "torch" in sys.modules,
        "thread_limits": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}})
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    args = p.parse_args()
    raw = args.protocol.read_bytes()
    run(json.loads(raw), Path.cwd(), hashlib.sha256(raw).hexdigest())
