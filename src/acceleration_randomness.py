"""Fixed continuous-fit CatBoost split-score-randomness intervention.

The random_strength parameter is varied once for each whole run. This cannot
isolate its time-varying internal implementation from the resulting changes in
trees and gradients. Synthetic scan calibration is descriptive in this study.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.acceleration_audit import slope_scan
from src.acceleration_boosting import (
    SPLITS, array_hash, boundary_contrast, file_hash, paired_summary,
    prepare_dataset, raw_trajectories, trajectory_metrics, write_json,
)
from src.detector_catalog import local_velocity_change
from src.dynamics import describe_change


def validate_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "protocol_id": "R12-split-score-randomness-2026-11-07",
        "datasets": ["synthetic", "breast_cancer"],
        "seeds": list(range(13001, 13013)),
        "random_strengths": [0.0, 1.0, 10.0],
        "iterations": 1024,
        "learning_rate": 0.05,
        "scan_windows": [8, 16, 32],
        "legacy_local_window": 5,
        "boundary": 512,
        "contrast_points": 64,
        "natural_gate_min_drop": 0.01,
        "natural_gate_positive_seed_count": 10,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Registered R12 field differs: {key}")
    if protocol.get("cpu_threads") != 4:
        raise ValueError("R12 requires four CPU threads")
    if protocol.get("max_storage_bytes") != 2 * 1024 ** 3:
        raise ValueError("R12 storage cap differs")


def fit_random_strength(data: dict[str, Any], random_strength: float, *, iterations: int,
                        learning_rate: float, threads: int):
    """One continuous fit, without an initial model or schedule change."""
    from catboost import CatBoostClassifier

    train = data["indices"]["train"]
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate,
        depth=4, l2_leaf_reg=3.0, bootstrap_type="No", random_strength=random_strength,
        loss_function="Logloss", thread_count=threads,
        allow_writing_files=False, use_best_model=False, boost_from_average=True,
        verbose=False, task_type="CPU", random_seed=data["seed_channels"]["model"],
        border_count=254, leaf_estimation_iterations=1, boosting_type="Plain",
    )
    model.fit(data["x"][train], data["y"][train])
    if model.tree_count_ != iterations:
        raise ValueError("Incomplete registered tree budget")
    return model


def describe_trajectory(loss: list[float], windows: list[int], *, local_window: int = 5,
                        threshold: float | None = None) -> dict[str, Any]:
    scan = slope_scan(loss, windows)
    scan["descriptive_candidate"] = None if threshold is None else scan["score"] > threshold
    scan["descriptive_threshold"] = threshold
    return {
        "global": describe_change(loss),
        "local": local_velocity_change(loss, window=local_window, rss_improvement_min=0.05),
        "scan": scan,
        "threshold_scope": "descriptive only; synthetic whole-curve nulls are not exchangeable with CatBoost trajectories",
    }


def first_zero_errors(errors: list[int]) -> int | None:
    if not errors or any(type(value) is not int or value < 0 for value in errors):
        raise ValueError("Exact nonnegative error counts required")
    return next((index for index, value in enumerate(errors) if value == 0), None)


def summarize_cells(cells: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    """Summarize predeclared paired scan contrasts and fixed-boundary gates."""
    comparisons = []
    fixed_boundary_contrasts = []
    gates = []
    for dataset in protocol["datasets"]:
        subset = [row for row in cells if row["dataset"] == dataset]
        lookup = {(row["seed"], row["random_strength"]): row for row in subset}
        if len(subset) != len(lookup) or len(lookup) != len(protocol["seeds"]) * len(protocol["random_strengths"]):
            raise ValueError("Incomplete or duplicate paired matrix")
        for strength in (1.0, 10.0):
            score_differences = [lookup[(seed, strength)]["diagnostics"]["train"]["scan"]["score"] - lookup[(seed, 0.0)]["diagnostics"]["train"]["scan"]["score"] for seed in protocol["seeds"]]
            summary = paired_summary(score_differences, comparisons=4)
            lo, hi = summary["ci_family_bonferroni"]
            outcome = "exploratory_no_sign_prediction" if strength == 1.0 else "supported" if lo > 0 else "refuted" if hi < 0 else "inconclusive"
            comparisons.append({"dataset": dataset, "random_strength_minus_zero": strength,
                                "predicted_sign": None if strength == 1.0 else "positive",
                                "outcome": outcome, **summary})
            gain_differences = [lookup[(seed, strength)]["boundary_contrast"]["train"]["velocity_gain"] - lookup[(seed, 0.0)]["boundary_contrast"]["train"]["velocity_gain"] for seed in protocol["seeds"]]
            fixed_boundary_contrasts.append({"dataset": dataset, "random_strength_minus_zero": strength,
                                             "interpretation": "Prespecified signed fixed-boundary diagnostic; no sign prediction.",
                                             **paired_summary(gain_differences, comparisons=4)})
        for strength in protocol["random_strengths"]:
            rows = [lookup[(seed, strength)] for seed in protocol["seeds"]]
            gains = [row["boundary_contrast"]["train"]["velocity_gain"] for row in rows]
            drops = [row["fixed_after_drop"]["train"] for row in rows]
            gain_summary = paired_summary(gains, comparisons=12)
            drop_summary = paired_summary(drops, comparisons=12)
            positive_count = sum(value > 0 for value in gains)
            gain_positive = gain_summary["ci_family_bonferroni"][0] > 0
            drop_practical = drop_summary["ci_family_bonferroni"][0] > protocol["natural_gate_min_drop"]
            reproduced_sign = positive_count >= protocol["natural_gate_positive_seed_count"]
            gates.append({"dataset": dataset, "random_strength": strength,
                          "fixed_boundary": protocol["boundary"], "contrast_points": protocol["contrast_points"],
                          "mean_gain": gain_summary, "mean_after_drop": drop_summary,
                          "positive_gain_seed_count": positive_count,
                          "practical_drop_threshold": protocol["natural_gate_min_drop"],
                          "positive_gain_ci": gain_positive, "practical_drop_ci": drop_practical,
                          "reproduced_gain_sign": reproduced_sign,
                          "outcome": "fixed_late_acceleration_established" if gain_positive and drop_practical and reproduced_sign else "not_established",
                          "interpretation": "Operational evidence for a prespecified sustained-window event only. No identification of split-noise annealing mechanism or universal natural acceleration."})
    return {"scan_score_contrasts": comparisons, "fixed_boundary_gain_contrasts": fixed_boundary_contrasts,
            "natural_acceleration_gates": gates}


def run_study(protocol: dict[str, Any], output: Path, protocol_path: Path,
              calibration_path: Path | None = None) -> dict[str, Any]:
    validate_protocol(protocol)
    import catboost
    import scipy
    import sklearn

    started = time.monotonic()
    threshold = None
    calibration_binding = None
    if calibration_path is not None:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("windows") != protocol["scan_windows"]:
            raise ValueError("Calibration windows differ from registered scan windows")
        threshold = float(calibration["deployment_descriptive_threshold"])
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid descriptive calibration threshold")
        calibration_binding = {"path": str(calibration_path), "sha256": file_hash(calibration_path),
                               "threshold": threshold, "scope": "descriptive only; model trajectories have different length and unknown null law"}
    output.mkdir(parents=True, exist_ok=True)
    predictions = {}
    data_arrays = {}
    cells = []
    model_records = []
    cell_path = output / "cells.jsonl"
    with cell_path.open("x", encoding="utf-8") as stream:
        for dataset in protocol["datasets"]:
            for seed in protocol["seeds"]:
                data = prepare_dataset(dataset, seed)
                key = f"{dataset}_{seed}"
                data_arrays[f"{key}_x"] = data["x"]
                data_arrays[f"{key}_y"] = data["y"]
                for split, indices in data["indices"].items():
                    data_arrays[f"{key}_{split}_indices"] = indices
                for strength in protocol["random_strengths"]:
                    cell_started = time.monotonic()
                    cell_id = f"{key}_rs{strength:g}"
                    print(f"dataset={dataset} seed={seed} random_strength={strength:g}", flush=True)
                    model = fit_random_strength(data, strength, iterations=protocol["iterations"], learning_rate=protocol["learning_rate"], threads=protocol["cpu_threads"])
                    raw, baseline = raw_trajectories(model, data, threads=protocol["cpu_threads"])
                    metrics = {}
                    for split, matrix in raw.items():
                        predictions[f"{cell_id}_{split}"] = matrix
                        metrics[split] = trajectory_metrics(matrix, data["y"][data["indices"][split]])
                    leaf_counts = np.asarray(model.get_tree_leaf_counts(), dtype=np.int64)
                    train_leaf_indices = np.asarray(model.calc_leaf_indexes(data["x"][data["indices"]["train"]], thread_count=protocol["cpu_threads"]), dtype=np.int64)
                    if leaf_counts.shape != (protocol["iterations"],) or train_leaf_indices.shape != (len(data["indices"]["train"]), protocol["iterations"]):
                        raise ValueError("Incomplete realized tree structure")
                    occupied_leaves = [int(len(np.unique(train_leaf_indices[:, tree]))) for tree in range(protocol["iterations"])]
                    predictions[f"{cell_id}_train_leaf_indices"] = train_leaf_indices
                    model_path = output / "models" / f"{cell_id}.cbm"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    model.save_model(str(model_path), format="cbm")
                    model_record = {"path": f"models/{cell_id}.cbm", "sha256": file_hash(model_path), "bytes": model_path.stat().st_size}
                    model_records.append(model_record)
                    row = {
                        "id": cell_id, "dataset": dataset, "seed": seed, "random_strength": strength,
                        "seed_channels": data["seed_channels"], "data_hashes": data["hashes"],
                        "split_sizes": {split: len(indices) for split, indices in data["indices"].items()},
                        "n_trees": int(model.tree_count_), "baseline": baseline, "model": model_record,
                        "resolved_model_parameters": model.get_all_params(),
                        "predictions_keys": {split: f"{cell_id}_{split}" for split in SPLITS},
                        "prediction_hashes": {split: array_hash(matrix) for split, matrix in raw.items()},
                        "train_leaf_indices_key": f"{cell_id}_train_leaf_indices",
                        "train_leaf_indices_sha256": array_hash(train_leaf_indices),
                        "realized_tree_leaf_counts": leaf_counts.tolist(),
                        "occupied_train_leaf_counts": occupied_leaves,
                        "first_zero_train_errors_iteration": first_zero_errors(metrics["train"]["errors"]),
                        "first_zero_errors_does_not_mean_zero_logloss": True,
                        "metrics": metrics,
                        "boundary_contrast": {split: boundary_contrast(metrics[split]["logloss"], protocol["boundary"], protocol["contrast_points"]) for split in SPLITS},
                        "fixed_after_drop": {split: float(metrics[split]["logloss"][protocol["boundary"]] - metrics[split]["logloss"][protocol["boundary"] + protocol["contrast_points"] - 1]) for split in SPLITS},
                        "diagnostics": {split: describe_trajectory(metrics[split]["logloss"], protocol["scan_windows"], local_window=protocol["legacy_local_window"], threshold=threshold) for split in ("train", "test")},
                        "elapsed_seconds": time.monotonic() - cell_started,
                    }
                    cells.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
    np.savez_compressed(output / "predictions.npz", **predictions)
    np.savez_compressed(output / "data.npz", **data_arrays)
    write_json(output / "models.json", {"models": model_records})
    analysis = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"], "protocol_sha256": file_hash(protocol_path),
        "n_cells": len(cells), "expected_cells": 72, "seeds": protocol["seeds"],
        "datasets": protocol["datasets"], "random_strengths": protocol["random_strengths"],
        "single_continuous_fit": True, "constant_learning_rate": protocol["learning_rate"],
        "calibration_binding": calibration_binding,
        **summarize_cells(cells, protocol),
        "used_test_to_choose_grid": False, "double_descent_declared": False,
        "noise_annealing_mechanism_identified": False, "synthetic_null_pvalues_transportable": False,
        "gpu_jobs": 0, "paid_api_spend": 0, "download_bytes": 0,
        "elapsed_seconds": time.monotonic() - started,
        "artifact_hashes": {name: file_hash(output / name) for name in ("cells.jsonl", "predictions.npz", "data.npz", "models.json")},
        "scientific_validity": "not_assessed",
        "limitations": ["Two fixed small environments and 12 paired replicates per environment.",
                        "Breast-cancer split/model repetitions are conditional on one fixed corpus and do not sample new corpora.",
                        "Whole-run random_strength interventions change split decisions and paths; they do not isolate time-varying split-score noise.",
                        "Student-t intervals use approximate inference for independent replicate differences.",
                        "A maximum scan is a descriptive scalar; synthetic-null p-values cannot be transported to these model trajectories.",
                        "Failure of a prespecified fixed-boundary gate is not evidence against all acceleration locations or configurations.",
                        "Exact zero classification errors are separate from zero LogLoss and do not by themselves establish double descent."],
    }
    write_json(output / "analysis.json", analysis)
    write_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "catboost": catboost.__version__, "cpu_threads": protocol["cpu_threads"], "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}, "gpu_jobs": 0, "torch_imported": "torch" in sys.modules, "download_bytes": 0})
    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if total_bytes > protocol["max_storage_bytes"]:
        raise RuntimeError(f"Storage budget exceeded: {total_bytes}")
    return analysis
