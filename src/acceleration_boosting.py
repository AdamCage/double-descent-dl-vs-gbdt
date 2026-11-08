"""Fixed CatBoost schedule intervention study, with raw trajectory evidence.

This module does not use a detector to choose data, settings, or outcomes.
Every schedule uses two identical continuation calls. A scheduled acceleration
is a positive control for interpretation, not a discovered natural mechanism.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "validation", "test")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    descriptor = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(descriptor + b"\0" + value.tobytes()).hexdigest()


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("protocol_id") != "R11-schedule-controls-2026-11-07":
        raise ValueError("Unsupported R11 protocol")
    seeds = protocol.get("seeds", [])
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Unique nonnegative integer seeds required")
    if protocol.get("datasets") != ["synthetic", "breast_cancer"]:
        raise ValueError("Dataset menu must be fixed")
    schedules = protocol.get("schedules", {})
    if set(schedules) != {"constant", "stepup", "stepdown"}:
        raise ValueError("Three registered schedules required")
    for rates in schedules.values():
        if len(rates) != 2 or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("Two positive learning rates per schedule required")
    totals = [sum(rates) for rates in schedules.values()]
    if not np.allclose(totals, totals[0], rtol=0, atol=1e-14):
        raise ValueError("Integrated learning rates differ")
    half = protocol.get("trees_per_segment")
    window = protocol.get("contrast_points")
    if type(half) is not int or type(window) is not int or not 3 <= window <= half:
        raise ValueError("Insufficient schedule or slope window length")
    if protocol.get("cpu_threads") not in (1, 2, 3, 4):
        raise ValueError("CPU thread cap exceeded")


def prepare_dataset(name: str, seed: int, *, synthetic_n: int = 768) -> dict[str, Any]:
    """Use one data/split/model replicate; breast cancer is one fixed corpus."""
    from sklearn.datasets import load_breast_cancer, make_classification
    from sklearn.model_selection import train_test_split

    data_seed, split_seed, model_seed = seed, seed + 100_000, seed + 200_000
    if name == "synthetic":
        x, y = make_classification(
            n_samples=synthetic_n, n_features=10, n_informative=5, n_redundant=0,
            n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
            flip_y=0.08, class_sep=0.8, hypercube=True, shift=0.0, scale=1.0,
            shuffle=True, random_state=data_seed,
        )
        heldout_fraction = 2 / 3
    elif name == "breast_cancer":
        loaded = load_breast_cancer()
        x, y = loaded.data, loaded.target
        heldout_fraction = 0.4
    else:
        raise ValueError(f"Unregistered dataset: {name}")
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64)
    indices = np.arange(len(y), dtype=np.int64)
    train, heldout = train_test_split(indices, test_size=heldout_fraction, stratify=y, random_state=split_seed)
    validation, test = train_test_split(heldout, test_size=0.5, stratify=y[heldout], random_state=split_seed + 1)
    return {
        "name": name, "seed": seed,
        "seed_channels": {"data": data_seed, "split": split_seed, "model": model_seed},
        "x": x, "y": y, "indices": {"train": train, "validation": validation, "test": test},
        "hashes": {"x": array_hash(x), "y": array_hash(y), **{f"{key}_indices": array_hash(value) for key, value in {"train": train, "validation": validation, "test": test}.items()}},
    }


def fit_schedule(data: dict[str, Any], rates: list[float], *, trees_per_segment: int, threads: int,
                 return_initial: bool = False):
    """All arms, including constant LR, use the same two-stage continuation."""
    from catboost import CatBoostClassifier

    train = data["indices"]["train"]
    common = {
        "iterations": trees_per_segment, "depth": 4, "l2_leaf_reg": 3.0,
        "bootstrap_type": "No", "random_strength": 1.0, "loss_function": "Logloss",
        "thread_count": threads, "allow_writing_files": False, "use_best_model": False,
        "verbose": False, "task_type": "CPU",
        "random_seed": data["seed_channels"]["model"], "border_count": 254,
        "leaf_estimation_iterations": 1, "boosting_type": "Plain",
    }
    first = CatBoostClassifier(**common, learning_rate=rates[0], boost_from_average=True)
    first.fit(data["x"][train], data["y"][train])
    # CatBoost forbids boost_from_average=True together with init_model.
    # The continuation retains the fitted bias from the initial model.
    second = CatBoostClassifier(**common, learning_rate=rates[1], boost_from_average=False)
    second.fit(data["x"][train], data["y"][train], init_model=first)
    if second.tree_count_ != 2 * trees_per_segment:
        raise ValueError("Continuation did not preserve the full registered tree budget")
    return (first, second) if return_initial else second


def raw_trajectories(model: Any, data: dict[str, Any], *, threads: int = 4) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("Inference threads must be explicitly bounded by four")
    train_y = data["y"][data["indices"]["train"]]
    prevalence = float(train_y.mean())
    if not 0 < prevalence < 1:
        raise ValueError("Both train classes required")
    intercept = math.log(prevalence / (1 - prevalence))
    scale, bias = model.get_scale_and_bias()
    if not math.isclose(float(scale), 1.0, abs_tol=1e-12) or not math.isclose(float(bias), intercept, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"Model base margin differs from prevalence logit: {bias}, {intercept}")
    # The fitted bias is retained exactly, after verifying its interpretation.
    intercept = float(bias)
    result = {}
    for split, indices in data["indices"].items():
        stages = [np.asarray(value, dtype=np.float64) for value in model.staged_predict(data["x"][indices], prediction_type="RawFormulaVal", eval_period=1, thread_count=threads)]
        matrix = np.vstack([np.full(len(indices), intercept), *stages])
        if matrix.shape != (model.tree_count_ + 1, len(indices)) or not np.isfinite(matrix).all():
            raise ValueError("Incomplete/nonfinite raw trajectory")
        final = np.asarray(model.predict(data["x"][indices], prediction_type="RawFormulaVal", thread_count=threads))
        if not np.allclose(final, matrix[-1], rtol=1e-12, atol=1e-12):
            raise ValueError("Final staged prediction mismatch")
        result[split] = matrix
    return result, {"train_prevalence": prevalence, "base_margin": intercept, "base_margin_source": "verified fitted model bias"}


def trajectory_metrics(raw: np.ndarray, labels: np.ndarray) -> dict[str, list[float] | list[int]]:
    """Stable per-iteration metrics on every object, including exact errors."""
    from scipy.special import expit

    labels = np.asarray(labels, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != len(labels) or not np.isfinite(raw).all():
        raise ValueError("Invalid raw prediction matrix")
    signed = raw * (2 * labels[None, :] - 1)
    loss = np.mean(np.logaddexp(0, -signed), axis=1)
    gradients = expit(-signed)
    count = max(1, int(math.ceil(0.1 * len(labels))))
    top = np.sort(gradients, axis=1)[:, -count:].sum(axis=1)
    total = gradients.sum(axis=1)
    concentration = np.divide(top, total, out=np.zeros_like(top), where=total > 0)
    return {
        "logloss": loss.tolist(),
        "errors": np.sum((raw > 0) != labels[None, :], axis=1).astype(int).tolist(),
        "min_margin": np.min(signed, axis=1).tolist(),
        "q10_margin": np.quantile(signed, 0.1, axis=1).tolist(),
        "gradient_rms": np.sqrt(np.mean(gradients * gradients, axis=1)).tolist(),
        "top10_gradient_mass": concentration.tolist(),
    }


def slope(values: np.ndarray, x: np.ndarray | None = None) -> float:
    coordinates = np.arange(len(values), dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    if coordinates.shape != values.shape or not np.isfinite(coordinates).all() or not np.all(np.diff(coordinates) > 0):
        raise ValueError("Slope coordinates must be finite and strictly increasing")
    centered = coordinates - coordinates.mean()
    return float(np.dot(centered, values) / np.dot(centered, centered))


def boundary_contrast(loss: list[float] | np.ndarray, boundary: int, points: int, x: np.ndarray | None = None) -> dict[str, float | list[int]]:
    values = np.asarray(loss, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or points < 3 or boundary < points - 1 or boundary + points > len(values):
        raise ValueError("Invalid boundary/window")
    before_range = [boundary - points + 1, boundary + 1]
    after_range = [boundary, boundary + points]
    coordinates = np.arange(len(values), dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    if coordinates.shape != values.shape or not np.isfinite(coordinates).all() or not np.all(np.diff(coordinates) > 0):
        raise ValueError("Contrast coordinates must be finite and strictly increasing")
    before = -slope(values[slice(*before_range)], coordinates[slice(*before_range)])
    after = -slope(values[slice(*after_range)], coordinates[slice(*after_range)])
    return {"velocity_before": before, "velocity_after": after, "velocity_gain": after - before, "before_slice": before_range, "after_slice": after_range}


def paired_summary(values: list[float], *, family_alpha: float = 0.05, comparisons: int = 4) -> dict[str, Any]:
    from scipy.stats import t

    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("At least two finite paired observations required")
    mean, se = float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array)))
    half95 = float(t.ppf(0.975, len(array) - 1)) * se
    half_adjusted = float(t.ppf(1 - family_alpha / (2 * comparisons), len(array) - 1)) * se
    return {"n": len(array), "values": array.tolist(), "mean": mean, "standard_error": se,
            "ci95_student_t": [mean - half95, mean + half95],
            "ci_family_bonferroni": [mean - half_adjusted, mean + half_adjusted],
            "family_alpha": family_alpha, "family_comparisons": comparisons,
            "assumptions": "Independent replicate differences and approximate t inference; fixed-corpus split repetitions do not sample new corpora."}


def run_study(protocol: dict[str, Any], output: Path, protocol_path: Path) -> dict[str, Any]:
    validate_protocol(protocol)
    import catboost
    import scipy
    import sklearn

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, np.ndarray] = {}
    data_arrays: dict[str, np.ndarray] = {}
    cells: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []
    cell_path = output / "cells.jsonl"
    if cell_path.exists():
        raise FileExistsError("Refuse to overwrite scientific output")
    with cell_path.open("x", encoding="utf-8") as stream:
        for dataset in protocol["datasets"]:
            for seed in protocol["seeds"]:
                data = prepare_dataset(dataset, seed)
                key = f"{dataset}_{seed}"
                data_arrays[f"{key}_x"] = data["x"]
                data_arrays[f"{key}_y"] = data["y"]
                for split, indices in data["indices"].items():
                    data_arrays[f"{key}_{split}_indices"] = indices
                for schedule, rates in protocol["schedules"].items():
                    cell_started = time.monotonic()
                    print(f"dataset={dataset} seed={seed} schedule={schedule}", flush=True)
                    model = fit_schedule(data, rates, trees_per_segment=protocol["trees_per_segment"], threads=protocol["cpu_threads"])
                    raw, baseline = raw_trajectories(model, data, threads=protocol["cpu_threads"])
                    cell_id = f"{key}_{schedule}"
                    learning_rate_clock = np.r_[0.0, np.cumsum(np.repeat(rates, protocol["trees_per_segment"]))]
                    metrics = {}
                    for split, matrix in raw.items():
                        predictions[f"{cell_id}_{split}"] = matrix
                        metrics[split] = trajectory_metrics(matrix, data["y"][data["indices"][split]])
                    model_path = output / "models" / f"{cell_id}.cbm"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    model.save_model(str(model_path), format="cbm")
                    model_record = {"path": f"models/{cell_id}.cbm", "sha256": file_hash(model_path), "bytes": model_path.stat().st_size}
                    model_manifest.append(model_record)
                    row = {
                        "id": cell_id, "dataset": dataset, "seed": seed, "schedule": schedule,
                        "rates": rates, "seed_channels": data["seed_channels"], "data_hashes": data["hashes"],
                        "split_sizes": {split: len(indices) for split, indices in data["indices"].items()},
                        "n_trees": int(model.tree_count_), "baseline": baseline, "model": model_record,
                        "predictions_keys": {split: f"{cell_id}_{split}" for split in SPLITS},
                        "prediction_hashes": {split: array_hash(matrix) for split, matrix in raw.items()},
                        "metrics": metrics,
                        "cumulative_learning_rate": learning_rate_clock.tolist(),
                        "boundary_contrast": {split: boundary_contrast(metrics[split]["logloss"], protocol["trees_per_segment"], protocol["contrast_points"]) for split in SPLITS},
                        "boundary_contrast_lr_clock": {split: boundary_contrast(metrics[split]["logloss"], protocol["trees_per_segment"], protocol["contrast_points"], learning_rate_clock) for split in SPLITS},
                        "elapsed_seconds": time.monotonic() - cell_started,
                    }
                    cells.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
    np.savez_compressed(output / "predictions.npz", **predictions)
    np.savez_compressed(output / "data.npz", **data_arrays)
    write_json(output / "models.json", {"models": model_manifest})
    contrasts = []
    normalized_contrasts = []
    for dataset in protocol["datasets"]:
        lookup = {(row["seed"], row["schedule"]): row for row in cells if row["dataset"] == dataset}
        for schedule in ("stepup", "stepdown"):
            values = [lookup[(seed, schedule)]["boundary_contrast"]["train"]["velocity_gain"] - lookup[(seed, "constant")]["boundary_contrast"]["train"]["velocity_gain"] for seed in protocol["seeds"]]
            summary = paired_summary(values)
            predicted_sign = 1 if schedule == "stepup" else -1
            lo, hi = summary["ci_family_bonferroni"]
            supported = lo > 0 if predicted_sign == 1 else hi < 0
            refuted = hi < 0 if predicted_sign == 1 else lo > 0
            contrasts.append({"dataset": dataset, "schedule_minus_constant": schedule, "outcome": "supported" if supported else "refuted" if refuted else "inconclusive", **summary})
            normalized = [lookup[(seed, schedule)]["boundary_contrast_lr_clock"]["train"]["velocity_gain"] - lookup[(seed, "constant")]["boundary_contrast_lr_clock"]["train"]["velocity_gain"] for seed in protocol["seeds"]]
            normalized_contrasts.append({"dataset": dataset, "schedule_minus_constant": schedule, "interpretation": "Prespecified diagnostic; cumulative prescribed learning rate changes clock but does not hold learned tree partitions or gradient path fixed.", **paired_summary(normalized)})
    analysis = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"], "protocol_sha256": file_hash(protocol_path),
        "n_cells": len(cells), "expected_cells": len(protocol["datasets"]) * len(protocol["seeds"]) * len(protocol["schedules"]),
        "seeds": protocol["seeds"], "datasets": protocol["datasets"], "schedules": protocol["schedules"],
        "primary_contrasts": contrasts,
        "lr_clock_diagnostic_contrasts": normalized_contrasts,
        "primary_boundary": protocol["trees_per_segment"], "contrast_points": protocol["contrast_points"],
        "same_continuation_semantics": True, "equal_integrated_learning_rate": True,
        "used_test_to_choose_grid": False, "double_descent_declared": False,
        "natural_acceleration_mechanism_identified": False, "synthetic_null_pvalues_transportable": False,
        "gpu_jobs": 0, "paid_api_spend": 0, "download_bytes": 0,
        "elapsed_seconds": time.monotonic() - started,
        "artifact_hashes": {name: file_hash(output / name) for name in ("cells.jsonl", "predictions.npz", "data.npz", "models.json")},
        "scientific_validity": "not_assessed",
        "limitations": ["Three predeclared schedules on two small environments only.", "Scheduled acceleration is a positive control, not a naturally discovered phenomenon.", "Breast-cancer repetitions are conditional on one fixed corpus.", "Student-t intervals assume independent replicate differences and approximate distributional adequacy.", "Train-loss acceleration alone cannot establish double descent or its mechanism."],
    }
    write_json(output / "analysis.json", analysis)
    write_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "catboost": catboost.__version__, "cpu_threads": protocol["cpu_threads"], "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}, "gpu_jobs": 0, "torch_imported": "torch" in sys.modules, "download_bytes": 0})
    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if total_bytes > protocol["max_storage_bytes"]:
        raise RuntimeError(f"Storage budget exceeded: {total_bytes}")
    return analysis
