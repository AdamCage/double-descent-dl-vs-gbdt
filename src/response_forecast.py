"""Fixed discovery runner for forecasting finite-label frozen-replay error.

Only ``run_study`` generates study inputs or fits CatBoost. The forecast,
weighting and decision helpers are pure numerical functions, usable in default
offline tests without fitting a research model.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

from src.boosting_response import (
    canonicalize_targets, decomposition, fit_control, frozen_operators,
    nested_binary_features, representation_mse_floor,
)


BASELINE_IDS = ("B0", "B1", "B2", "B3", "B4")
MODEL_IDS = (*BASELINE_IDS, "A")
B1_FEATURES = ("log10_epsilon", "log10_prefix", "teacher_interaction")
B2_FEATURES = (*B1_FEATURES, "log10_1p_train_mse", "collision_mse_floor",
               "trajectory_weighted_per_tree_efc_proxy", "log10_1p_frozen_eval_noise_gain",
               "log10_1p_frozen_train_noise_gain", "log10_1p_largest_train_singular_value")
CHEAP_FEATURE = "log10_1p_cheap_discrepancy"
MODEL_FEATURES = {"B0": (), "B1": B1_FEATURES, "B2": B2_FEATURES,
                  "B3": (*B1_FEATURES, CHEAP_FEATURE), "B4": (CHEAP_FEATURE,),
                  "A": (*B2_FEATURES, CHEAP_FEATURE)}


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    description = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True).encode()
    return hashlib.sha256(description + b"\0" + array.tobytes()).hexdigest()


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("protocol_id") != "R14-finite-label-response-forecast-v1":
        raise ValueError("Unsupported finite-label-response protocol")
    if protocol.get("mode") != "discovery" or protocol.get("teachers") != ["linear", "interaction"]:
        raise ValueError("Registered discovery teachers required")
    for field in ("n_train", "n_evaluation", "n_raw_features", "iterations", "pilot_iterations", "pilot_directions", "response_directions", "threads"):
        if type(protocol.get(field)) is not int or protocol[field] < 1:
            raise ValueError(f"Positive integer required: {field}")
    if protocol["n_raw_features"] != 4 or protocol["n_train"] < 2 or not 1 <= protocol["threads"] <= 4:
        raise ValueError("Four raw features, n_train>=2 and CPU threads<=4 required")
    if protocol["pilot_iterations"] >= protocol["iterations"]:
        raise ValueError("Pilot fit must be shorter than full fit")
    seeds = protocol.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or any(type(seed) is not int or seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Distinct nonnegative data seeds required")
    if protocol.get("threshold_levels") != [1, 3, 7, 15] or protocol.get("prefix_budgets") != [4, 12, 28, 60]:
        raise ValueError("Registered nested threshold levels and prefixes required")
    amplitudes = protocol.get("epsilon_values")
    if not isinstance(amplitudes, list) or not amplitudes or len(set(amplitudes)) != len(amplitudes) or any(type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0 for eps in amplitudes):
        raise ValueError("Distinct positive perturbation amplitudes required")
    for key in ("train", "evaluation", "label_noise", "model", "pilot", "response"):
        if type(protocol.get("seed_offsets", {}).get(key)) is not int or protocol["seed_offsets"][key] < 0:
            raise ValueError(f"Explicit seed channel required: {key}")
    if len(set(protocol["seed_offsets"].values())) != len(protocol["seed_offsets"]):
        raise ValueError("Seed channels must be distinct")
    if not isinstance(protocol.get("train_noise_sigma"), (float, int)) or not math.isfinite(protocol["train_noise_sigma"]) or protocol["train_noise_sigma"] < 0:
        raise ValueError("Finite nonnegative train noise required")
    if protocol.get("replay_atol") != 1e-10 or protocol.get("replay_rtol") != 0:
        raise ValueError("Fixed baseline replay tolerance is atol1e-10/rtol0")
    forecast = protocol.get("forecast", {})
    if forecast.get("ridge_alpha") != 1 or forecast.get("baseline_ids") != list(MODEL_IDS):
        raise ValueError("Registered ridge alpha and all B0..B4/A forecasters required")
    gates = protocol.get("decision_gates", {})
    for key in ("relative_mae_improvement", "min_target_range", "max_cheap_full_wall_ratio"):
        if not isinstance(gates.get(key), (int, float)) or not math.isfinite(gates[key]) or gates[key] < 0:
            raise ValueError(f"Finite nonnegative decision gate required: {key}")
    if type(gates.get("min_draws_not_worse")) is not int or not 1 <= gates["min_draws_not_worse"] <= len(seeds):
        raise ValueError("Valid draw-count gate required")
    budget = protocol.get("budget", {})
    for key in ("max_wall_seconds", "max_storage_bytes"):
        if not isinstance(budget.get(key), (int, float)) or not math.isfinite(budget[key]) or budget[key] <= 0:
            raise ValueError(f"Positive task budget required: {key}")
    contexts = len(seeds) * len(protocol["teachers"]) * len(protocol["prefix_budgets"])
    fits = contexts * (1 + len(amplitudes) * (protocol["pilot_directions"] + protocol["response_directions"]))
    for key, expected in (("expected_contexts", contexts), ("expected_fits", fits), ("expected_forecast_rows", contexts * len(amplitudes))):
        if protocol.get(key) != expected:
            raise ValueError(f"Declared matrix count differs: {key}")


def trajectory_efc_proxy(train_leaves: np.ndarray, train_predictions: np.ndarray,
                         actual_y: np.ndarray) -> dict[str, Any]:
    """Explicit loss-weighted per-new-tree adaptation of published Eq.(2)-(3)."""
    leaves = np.asarray(train_leaves)
    predictions = np.asarray(train_predictions, dtype=np.float64)
    targets = np.asarray(actual_y, dtype=np.float64)
    if leaves.ndim != 2 or not np.issubdtype(leaves.dtype, np.integer) or (leaves < 0).any():
        raise ValueError("Nonnegative integer per-tree leaf assignments required")
    n, trees = leaves.shape
    if n < 2 or targets.shape != (n,) or predictions.shape != (trees + 1, n) or not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Complete initial-and-per-tree training predictions required")
    if not np.array_equal(predictions[0], np.zeros(n)):
        raise ValueError("EFC control requires the actual zero initial predictor")
    per_tree = []
    for tree in range(trees):
        counts = np.bincount(leaves[:, tree])
        per_tree.append(1.0 - float(np.sum(counts * (counts - 1))) / (n * (n - 1)))
    sse = np.sum((predictions - targets[None, :]) ** 2, axis=1)
    weights = np.maximum(sse[:-1] - sse[1:], 0.0)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Undefined trajectory EFC proxy: zero positive SSE-improvement weight")
    value = float(np.dot(per_tree, weights) / total)
    return {"value": value, "per_tree": np.asarray(per_tree), "weights": weights, "sse": sse,
            "weight_sum": total, "scope": "Loss-weighted per-new-weak-tree EFC proxy on training inputs; explicit adaptation, not a verified replica of the authors' boosting implementation."}


def geometry_features(a_train: np.ndarray, a_evaluation: np.ndarray) -> dict[str, float]:
    train, evaluation = np.asarray(a_train, dtype=float), np.asarray(a_evaluation, dtype=float)
    if train.ndim != 2 or train.shape[0] != train.shape[1] or not len(train) or evaluation.ndim != 2 or evaluation.shape[1] != train.shape[1] or not len(evaluation) or not np.isfinite(train).all() or not np.isfinite(evaluation).all():
        raise ValueError("Finite train/evaluation operators with compatible dimensions required")
    return {"frozen_eval_noise_gain": float(np.sum(evaluation ** 2) / len(evaluation)),
            "frozen_train_noise_gain": float(np.sum(train ** 2) / len(train)),
            "largest_train_singular_value": float(np.linalg.svd(train, compute_uv=False)[0])}


def forecast_features(row: dict[str, Any]) -> dict[str, float]:
    if row["teacher"] not in ("linear", "interaction") or row["epsilon"] <= 0 or row["prefix"] <= 0:
        raise ValueError("Invalid registered forecast row")
    nonnegative = ("train_mse", "collision_mse_floor", "frozen_eval_noise_gain",
                   "frozen_train_noise_gain", "largest_train_singular_value", "cheap_discrepancy")
    if any(not math.isfinite(float(row[key])) or row[key] < 0 for key in nonnegative):
        raise ValueError("Finite nonnegative frozen/probe diagnostic features required")
    efc = float(row["trajectory_weighted_per_tree_efc_proxy"])
    if not math.isfinite(efc) or not 0 <= efc <= 1:
        raise ValueError("Finite per-tree EFC proxy in [0,1] required")
    return {"log10_epsilon": math.log10(row["epsilon"]), "log10_prefix": math.log10(row["prefix"]),
            "teacher_interaction": float(row["teacher"] == "interaction"),
            "log10_1p_train_mse": math.log10(1 + row["train_mse"]),
            "collision_mse_floor": float(row["collision_mse_floor"]),
            "trajectory_weighted_per_tree_efc_proxy": efc,
            "log10_1p_frozen_eval_noise_gain": math.log10(1 + row["frozen_eval_noise_gain"]),
            "log10_1p_frozen_train_noise_gain": math.log10(1 + row["frozen_train_noise_gain"]),
            "log10_1p_largest_train_singular_value": math.log10(1 + row["largest_train_singular_value"]),
            CHEAP_FEATURE: math.log10(1 + row["cheap_discrepancy"])}


def ridge_prediction(x_train: np.ndarray, y_train: np.ndarray, x_evaluation: np.ndarray,
                     alpha: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Train-fold standardization and fitted intercept, with no alpha selection."""
    x_train, x_evaluation, y_train = np.asarray(x_train, dtype=float), np.asarray(x_evaluation, dtype=float), np.asarray(y_train, dtype=float)
    if x_train.ndim != 2 or not len(x_train) or y_train.shape != (len(x_train),) or x_evaluation.ndim != 2 or x_train.shape[1] != x_evaluation.shape[1] or not np.isfinite(x_train).all() or not np.isfinite(x_evaluation).all() or not np.isfinite(y_train).all() or alpha <= 0:
        raise ValueError("Finite compatible ridge training/evaluation arrays required")
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    train = (x_train - center) / scale
    evaluation = (x_evaluation - center) / scale
    intercept = float(y_train.mean())
    coefficients = np.linalg.solve(train.T @ train + alpha * np.eye(train.shape[1]), train.T @ (y_train - intercept))
    predictions = intercept + evaluation @ coefficients
    return predictions, {"alpha": alpha, "feature_mean": center.tolist(), "feature_scale": scale.tolist(),
                         "coefficients_standardized": coefficients.tolist(), "intercept_centered_features": intercept,
                         "zero_variance_policy": "scale1", "standardization_scope": "training draw folds only"}


def oof_forecasts(rows: list[dict[str, Any]], seeds: list[int], *, alpha: float = 1.0) -> dict[str, Any]:
    """All contexts/amplitudes of a held-out draw stay outside every fit."""
    if not rows or len({row["id"] for row in rows}) != len(rows) or {row["seed"] for row in rows} != set(seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Unique complete forecast rows and declared draw folds required")
    if any(not row.get("valid_directions", False) for row in rows):
        return {"status": "inconclusive_invalid_direction", "rows": rows, "folds": [], "model_ids": list(MODEL_IDS),
                "reason": "At least one actual native perturbation was zero; complete matrix retained without replacing/dropping observations.", "go": False}
    features = [forecast_features(row) for row in rows]
    if any(not math.isfinite(row["full_discrepancy"]) or row["full_discrepancy"] < 0 for row in rows):
        raise ValueError("Finite nonnegative full-response endpoints required")
    targets = np.asarray([math.log10(1 + row["full_discrepancy"]) for row in rows])
    output = [{**row, "features": feature, "target_u": float(target), "models": {}} for row, feature, target in zip(rows, features, targets)]
    fold_records = []
    for seed in seeds:
        training = np.asarray([i for i, row in enumerate(rows) if row["seed"] != seed])
        heldout = np.asarray([i for i, row in enumerate(rows) if row["seed"] == seed])
        if not len(training) or not len(heldout):
            raise ValueError("Empty training or held-out draw fold")
        for model_id in MODEL_IDS:
            feature_names = MODEL_FEATURES[model_id]
            if model_id == "B0":
                mean = float(targets[training].mean())
                predictions = np.full(len(heldout), mean)
                fit = {"intercept": mean}
            elif model_id == "B4":
                cheap = np.asarray([feature[CHEAP_FEATURE] for feature in features])
                offset = float(np.mean(targets[training] - cheap[training]))
                predictions = cheap[heldout] + offset
                fit = {"slope": 1.0, "training_mean_target_minus_cheap": offset}
            else:
                matrix = np.asarray([[feature[name] for name in feature_names] for feature in features])
                predictions, fit = ridge_prediction(matrix[training], targets[training], matrix[heldout], alpha)
            fold_records.append({"held_out_seed": seed, "model": model_id,
                                 "training_seeds": [s for s in seeds if s != seed],
                                 "training_row_ids": [rows[i]["id"] for i in training],
                                 "held_out_row_ids": [rows[i]["id"] for i in heldout],
                                 "feature_names": list(feature_names), "fit": fit})
            for at, prediction in zip(heldout, predictions):
                residual = float(prediction - targets[at])
                output[at]["models"][model_id] = {"prediction_u": float(prediction), "residual_u": residual,
                                                   "absolute_error_u": abs(residual)}
    summaries = {}
    for model_id in MODEL_IDS:
        draw_mae = {str(seed): float(np.mean([row["models"][model_id]["absolute_error_u"] for row in output if row["seed"] == seed])) for seed in seeds}
        teacher_mae = {}
        for teacher in sorted({row["teacher"] for row in rows}):
            per_draw = [np.mean([row["models"][model_id]["absolute_error_u"] for row in output if row["seed"] == seed and row["teacher"] == teacher]) for seed in seeds]
            if not np.isfinite(per_draw).all():
                raise ValueError("Each teacher must be present in each held-out draw")
            teacher_mae[teacher] = float(np.mean(per_draw))
        summaries[model_id] = {"mae_u": float(np.mean(list(draw_mae.values()))),
                               "draw_mae_u": draw_mae, "teacher_mae_u": teacher_mae}
    q10, q90 = np.quantile(targets, [.1, .9], method="linear")
    return {"status": "complete_discovery_cross_validation", "rows": output, "folds": fold_records,
            "model_ids": list(MODEL_IDS), "summaries": summaries,
            "target_q10": float(q10), "target_q90": float(q90), "target_q90_minus_q10": float(q90 - q10),
            "independent_unit": "draw seed; teachers/prefixes/amplitudes/directions are correlated within draw",
            "metric_weighting": "equal context/epsilon weight within each draw, then equal draw weight",
            "inference": "discovery cross-validation, no iid64 significance or confidence intervals"}


def decision_gates(forecast: dict[str, Any], protocol: dict[str, Any], *, replay_passed: bool,
                    cheap_wall: float, full_wall: float) -> dict[str, Any]:
    gates = protocol["decision_gates"]
    if not all(math.isfinite(value) and value >= 0 for value in (cheap_wall, full_wall)):
        raise ValueError("Measured finite nonnegative response costs required")
    cost_ratio = cheap_wall / full_wall if full_wall > 0 else None
    cost_pass = cost_ratio is not None and cost_ratio <= gates["max_cheap_full_wall_ratio"]
    if forecast.get("status") != "complete_discovery_cross_validation":
        return {"P1_replay": replay_passed, "P2_forecast": "inconclusive_invalid_direction",
                "P3_cost": cost_pass, "cheap_full_wall_ratio": cost_ratio, "go": False}
    summaries = forecast["summaries"]
    a = summaries["A"]
    aggregate = {name: summaries[name]["mae_u"] > 0 and a["mae_u"] <= (1 - gates["relative_mae_improvement"]) * summaries[name]["mae_u"] for name in BASELINE_IDS}
    relative_improvement = {name: {"value": 1 - a["mae_u"] / summaries[name]["mae_u"] if summaries[name]["mae_u"] > 0 else None,
                                   "defined": summaries[name]["mae_u"] > 0,
                                   "reason": None if summaries[name]["mae_u"] > 0 else "Baseline MAE is zero; a relative improvement cannot be established."}
                            for name in BASELINE_IDS}
    comparisons = ("B2", "B3", "B4")
    per_draw = {str(seed): {name: a["draw_mae_u"][str(seed)] <= summaries[name]["draw_mae_u"][str(seed)] for name in comparisons} for seed in protocol["seeds"]}
    jointly_not_worse = sum(all(values.values()) for values in per_draw.values())
    not_worse_counts = {baseline: sum(values[baseline] for values in per_draw.values()) for baseline in comparisons}
    teacher = {name: {baseline: value <= summaries[baseline]["teacher_mae_u"][name] for baseline in comparisons}
               for name, value in a["teacher_mae_u"].items()}
    forecast_pass = all(aggregate.values()) and all(count >= gates["min_draws_not_worse"] for count in not_worse_counts.values()) and all(all(values.values()) for values in teacher.values())
    range_pass = forecast["target_q90_minus_q10"] >= gates["min_target_range"]
    return {"P1_replay": replay_passed, "P2_forecast": forecast_pass, "P3_cost": cost_pass,
            "aggregate_at_least_relative_improvement": aggregate,
            "relative_improvement_vs_baseline": relative_improvement,
            "draw_not_worse_comparisons": per_draw, "draws_jointly_not_worse": jointly_not_worse,
            "draws_not_worse_per_baseline": not_worse_counts,
            "teacher_not_worse_comparisons": teacher, "range_gate": range_pass,
            "cheap_full_wall_ratio": cost_ratio,
            "go": bool(replay_passed and forecast_pass and cost_pass and range_pass),
            "outcome": "proceed_to_separately_frozen_confirmation" if replay_passed and forecast_pass and cost_pass and range_pass else "inconclusive_lack_of_challenge" if not range_pass else "stop_this_forecast_candidate",
            "scope": "Decision on this prespecified discovery candidate, not a novelty or publication certificate."}


def teacher_values(name: str, x: np.ndarray) -> np.ndarray:
    if name == "linear":
        return np.sum(x, axis=1) / 2
    if name == "interaction":
        return (x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]) / math.sqrt(2)
    raise ValueError("Unregistered teacher")


def draw_inputs(seed: int, protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    """Study-only generation; pure default unit tests never invoke this function."""
    offsets = protocol["seed_offsets"]
    n, m, p = protocol["n_train"], protocol["n_evaluation"], protocol["n_raw_features"]
    return {
        "raw_train": np.random.default_rng(seed + offsets["train"]).normal(size=(n, p)),
        "raw_evaluation": np.random.default_rng(seed + offsets["evaluation"]).normal(size=(m, p)),
        "train_noise": np.random.default_rng(seed + offsets["label_noise"]).normal(scale=protocol["train_noise_sigma"], size=n),
        "pilot_directions": np.random.default_rng(seed + offsets["pilot"]).normal(size=(protocol["pilot_directions"], n)),
        "response_directions": np.random.default_rng(seed + offsets["response"]).normal(size=(protocol["response_directions"], n)),
    }


def _append(stream, row: dict[str, Any]) -> None:
    stream.write(json.dumps(json_value(row), allow_nan=False) + "\n")
    stream.flush()


class _RunState:
    def __init__(self, output: Path, protocol: dict[str, Any]):
        self.output, self.protocol = output, protocol
        self.started = time.perf_counter()
        self.context = None
        self.fit_id = None
        self.contexts_completed = 0
        self.fits_completed = 0
        self.model_records = []
        self.data_arrays: dict[str, np.ndarray] = {}
        self.prediction_arrays: dict[str, np.ndarray] = {}
        self.operator_arrays: dict[str, np.ndarray] = {}
        self.budget_checks = []
        self.update("initializing")

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def update(self, phase: str, **extra) -> None:
        atomic_json(self.output / "progress.json", {"phase": phase, "context_id": self.context,
                    "fit_id": self.fit_id, "contexts_completed": self.contexts_completed,
                    "fits_completed": self.fits_completed, "elapsed_seconds": self.elapsed(),
                    "pid": os.getpid(), "serial_model_jobs": 1, **extra})

    def guard(self, phase: str, *, reserve_bytes: int = 0) -> None:
        stored = sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file())
        buffered = sum(array.nbytes for mapping in (self.data_arrays, self.prediction_arrays, self.operator_arrays) for array in mapping.values())
        record = {"phase": phase, "context_id": self.context, "elapsed_seconds": self.elapsed(),
                  "stored_bytes": stored, "buffered_uncompressed_bytes": buffered,
                  "reserved_next_context_bytes": reserve_bytes,
                  "storage_accounting": "Conservative stored+buffered+reserve; buffered data may duplicate stored compressed evidence."}
        self.budget_checks.append(record)
        atomic_json(self.output / "budget_checks.json", {"checks": self.budget_checks})
        if record["elapsed_seconds"] > self.protocol["budget"]["max_wall_seconds"]:
            raise RuntimeError("Registered R14 task wall budget exceeded")
        if stored + buffered + reserve_bytes > self.protocol["budget"]["max_storage_bytes"]:
            raise RuntimeError("Registered R14 task storage budget exceeded")

    def checkpoint_arrays(self) -> float:
        start = time.perf_counter()
        atomic_npz(self.output / "data.npz", self.data_arrays)
        atomic_npz(self.output / "predictions.npz", self.prediction_arrays)
        atomic_npz(self.output / "operators.npz", self.operator_arrays)
        return time.perf_counter() - start


def _predict_native(model, x_train: np.ndarray, x_eval: np.ndarray, threads: int) -> dict[str, np.ndarray]:
    """Deployment predictions only; leaf-audit extraction is timed separately."""
    result = {}
    for split, x in (("train", x_train), ("evaluation", x_eval)):
        prediction = np.asarray(model.predict(x, prediction_type="RawFormulaVal", thread_count=threads), dtype=np.float64)
        if prediction.shape != (len(x),) or not np.isfinite(prediction).all():
            raise ValueError("Incomplete/nonfinite native prediction evidence")
        result[f"{split}__prediction"] = prediction
    return result


def _leaf_evidence(model, x_train: np.ndarray, x_eval: np.ndarray, threads: int) -> dict[str, np.ndarray]:
    """Preserve structural evidence without charging it to prediction cost P3."""
    result = {}
    for split, x in (("train", x_train), ("evaluation", x_eval)):
        leaves = np.asarray(model.calc_leaf_indexes(x, thread_count=threads))
        if leaves.shape != (len(x), model.tree_count_) or not np.issubdtype(leaves.dtype, np.integer) or (leaves < 0).any() or (leaves > 255).any():
            raise ValueError("Incomplete/noninteger native leaf evidence")
        result[f"{split}__leaf_indices"] = leaves.astype(np.uint8)
    result["tree_leaf_counts"] = np.asarray(model.get_tree_leaf_counts(), dtype=np.int64)
    result["leaf_weights"] = np.asarray(model.get_leaf_weights(), dtype=np.float64)
    return result


def _execute_fit(state: _RunState, fit_id: str, role: str, x_train: np.ndarray,
                 x_eval: np.ndarray, actual_y: np.ndarray, iterations: int,
                 model_seed: int, input_arrays: dict[str, np.ndarray]):
    """Persist inputs before fitting and the complete fit atomically afterwards."""
    state.fit_id = fit_id
    raw_path = state.output / "fit_evidence" / f"{fit_id}.npz"
    io_started = time.perf_counter()
    atomic_npz(raw_path, input_arrays)
    io_seconds = time.perf_counter() - io_started
    state.update("fitting", role=role, trees=iterations, fit_inputs=raw_path.relative_to(state.output).as_posix())
    if state.elapsed() > state.protocol["budget"]["max_wall_seconds"]:
        raise RuntimeError("Task wall budget exhausted before next fixed-matrix fit")
    started = time.perf_counter()
    model = fit_control(x_train, actual_y, iterations=iterations, seed=model_seed, threads=state.protocol["threads"])
    fit_seconds = time.perf_counter() - started
    io_started = time.perf_counter()
    model_path = state.output / "models" / f"{fit_id}.cbm"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(model_path.name + ".tmp")
    model.save_model(str(temporary), format="cbm")
    os.replace(temporary, model_path)
    model_digest = file_hash(model_path)
    io_seconds += time.perf_counter() - io_started
    state.update("predicting", role=role, model=model_path.relative_to(state.output).as_posix())
    started = time.perf_counter()
    arrays = _predict_native(model, x_train, x_eval, state.protocol["threads"])
    prediction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    arrays.update(_leaf_evidence(model, x_train, x_eval, state.protocol["threads"]))
    leaf_evidence_seconds = time.perf_counter() - started
    io_started = time.perf_counter()
    atomic_npz(raw_path, {**input_arrays, **arrays})
    record = {"id": fit_id, "context_id": state.context, "role": role, "trees": iterations,
              "seed": model_seed, "threads": state.protocol["threads"],
              "path": model_path.relative_to(state.output).as_posix(), "sha256": model_digest,
              "bytes": model_path.stat().st_size,
              "raw_evidence": raw_path.relative_to(state.output).as_posix(), "raw_evidence_sha256": file_hash(raw_path),
              "actual_targets_sha256": array_hash(actual_y),
              "resolved_parameters": model.get_all_params()}
    state.model_records.append(record)
    atomic_json(state.output / "models.json", {"models": state.model_records})
    io_seconds += time.perf_counter() - io_started
    for key, array in arrays.items():
        state.prediction_arrays[f"{fit_id}__{key}"] = array
    state.fits_completed += 1
    state.update("fit_complete", role=role, trees=iterations)
    return model, arrays, {"fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
                           "leaf_evidence_seconds": leaf_evidence_seconds,
                           "io_seconds": io_seconds}, record


def _scalar_decomposition(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "structural_delta"}


def _run_study(protocol: dict[str, Any], output: Path, protocol_path: Path) -> dict[str, Any]:
    import catboost
    from threadpoolctl import threadpool_info

    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("progress.json", "analysis.json", "contexts.jsonl", "directions.jsonl", "models.json")):
        raise FileExistsError("Refuse to resume/overwrite a scientific attempt; use a new registered attempt")
    state = _RunState(output, protocol)
    context_rows, direction_rows, forecast_rows = [], [], []
    all_baseline_checks = []
    total_times = defaultdict(float)
    contexts_stream = (output / "contexts.jsonl").open("x", encoding="utf-8")
    directions_stream = (output / "directions.jsonl").open("x", encoding="utf-8")
    features_stream = (output / "feature_snapshots.jsonl").open("x", encoding="utf-8")
    try:
        # A loose predeclared-size reserve, not an outcome-dependent reduction.
        fits_per_context = 1 + len(protocol["epsilon_values"]) * (protocol["pilot_directions"] + protocol["response_directions"])
        reserve = fits_per_context * (1024 ** 2 + (protocol["n_train"] + protocol["n_evaluation"]) * protocol["iterations"] * 8) + 4 * (protocol["n_train"] + protocol["n_evaluation"]) * protocol["n_train"] * 8
        for seed in protocol["seeds"]:
            raw = draw_inputs(seed, protocol)
            view = nested_binary_features(raw["raw_train"], raw["raw_evaluation"],
                                          protocol["n_raw_features"], protocol["threshold_levels"])
            for key, array in raw.items():
                state.data_arrays[f"draw{seed}__{key}"] = array
            for teacher in protocol["teachers"]:
                clean_train = teacher_values(teacher, raw["raw_train"])
                clean_evaluation = teacher_values(teacher, raw["raw_evaluation"])
                base_labels = canonicalize_targets(clean_train + raw["train_noise"])
                for prefix in protocol["prefix_budgets"]:
                    context_started = time.perf_counter()
                    context_id = f"{teacher}_s{seed}_p{prefix}"
                    state.context, state.fit_id = context_id, None
                    state.guard("before_context", reserve_bytes=reserve)
                    state.update("starting_context")
                    x_train, x_eval = view.prefix(prefix)
                    context_inputs = {"raw_train": raw["raw_train"], "raw_evaluation": raw["raw_evaluation"],
                                      "train_noise": raw["train_noise"], "binary_train": x_train, "binary_evaluation": x_eval,
                                      "clean_train": clean_train, "clean_evaluation": clean_evaluation,
                                      "base_nominal_targets": base_labels["nominal"], "base_native_targets": base_labels["native"],
                                      "base_actual_targets": base_labels["actual"], "base_rounding_delta": base_labels["rounding_delta"],
                                      "pilot_directions": raw["pilot_directions"], "response_directions": raw["response_directions"]}
                    io_started = time.perf_counter()
                    context_path = output / "context_evidence" / f"{context_id}.npz"
                    atomic_npz(context_path, context_inputs)
                    context_io = time.perf_counter() - io_started
                    for key, array in context_inputs.items():
                        state.data_arrays[f"{context_id}__{key}"] = array
                    model_seed = seed + protocol["seed_offsets"]["model"]
                    base_id = f"{context_id}__base"
                    base_model, base_arrays, base_time, base_record = _execute_fit(
                        state, base_id, "base", x_train, x_eval, base_labels["actual"],
                        protocol["iterations"], model_seed,
                        {"actual_targets": base_labels["actual"], "native_targets": base_labels["native"],
                         "nominal_targets": base_labels["nominal"], "initial_rounding_delta": base_labels["rounding_delta"]})
                    started = time.perf_counter()
                    replay = frozen_operators(base_model, x_train, x_eval,
                                              [protocol["pilot_iterations"], protocol["iterations"]],
                                              targets=base_labels["actual"], threads=protocol["threads"],
                                              atol=protocol["replay_atol"], rtol=protocol["replay_rtol"])
                    operator_seconds = time.perf_counter() - started
                    all_baseline_checks.extend({"context_id": context_id, **check} for check in replay["baseline_checks"])
                    for at, operators in replay["operators"].items():
                        for split, array in operators.items():
                            state.operator_arrays[f"{context_id}__t{at}__{split}"] = array
                    started = time.perf_counter()
                    train_stages = [np.asarray(prediction, dtype=float) for prediction in base_model.staged_predict(x_train, prediction_type="RawFormulaVal", thread_count=protocol["threads"], eval_period=1)]
                    train_trajectory = np.vstack([np.zeros(len(x_train)), *train_stages])
                    if train_trajectory.shape != (protocol["iterations"] + 1, protocol["n_train"]) or not np.isfinite(train_trajectory).all() or not np.allclose(train_trajectory[-1], base_arrays["train__prediction"], atol=protocol["replay_atol"], rtol=0):
                        raise ValueError("Incomplete or mismatched base training trajectory")
                    base_predictions = {}
                    for at in (protocol["pilot_iterations"], protocol["iterations"]):
                        base_predictions[at] = {"train": train_trajectory[at],
                                                "evaluation": np.asarray(base_model.predict(x_eval, prediction_type="RawFormulaVal", ntree_end=at, thread_count=protocol["threads"]), dtype=float)}
                    state.prediction_arrays[f"{base_id}__train__trajectory"] = train_trajectory
                    for at, split_predictions in base_predictions.items():
                        for split, prediction in split_predictions.items():
                            state.prediction_arrays[f"{base_id}__t{at}__{split}"] = prediction
                    base_trajectory_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    efc = trajectory_efc_proxy(replay["train_leaf_indices"], train_trajectory, base_labels["actual"])
                    floor = representation_mse_floor(x_train, base_labels["actual"])
                    geometry = geometry_features(replay["operators"][protocol["iterations"]]["train"],
                                                 replay["operators"][protocol["iterations"]]["evaluation"])
                    train_mse = float(np.mean((base_predictions[protocol["iterations"]]["train"] - base_labels["actual"]) ** 2))
                    base_features = {"train_mse": train_mse, "collision_mse_floor": floor["mse_floor"],
                                     "trajectory_weighted_per_tree_efc_proxy": efc["value"], **geometry}
                    base_feature_seconds = time.perf_counter() - started
                    state.prediction_arrays[f"{base_id}__train__sse"] = efc["sse"]
                    state.data_arrays[f"{context_id}__per_tree_efc_proxy"] = efc["per_tree"]
                    state.data_arrays[f"{context_id}__efc_weights"] = efc["weights"]
                    state.data_arrays[f"{context_id}__collision_group_ids"] = floor["group_ids"]
                    state.data_arrays[f"{context_id}__collision_group_counts"] = floor["group_counts"]
                    state.data_arrays[f"{context_id}__collision_group_means"] = floor["group_means"]
                    # Preserve common controls before any perturbed refit.
                    context_io += state.checkpoint_arrays()
                    context = {"id": context_id, "seed": seed, "teacher": teacher, "prefix": prefix,
                               "seed_channels": {key: seed + offset for key, offset in protocol["seed_offsets"].items()},
                               "data_hashes": {key: array_hash(array) for key, array in context_inputs.items()},
                               "context_evidence": {"path": context_path.relative_to(output).as_posix(), "sha256": file_hash(context_path)},
                               "threshold_manifest": view.manifest(),
                               "prefix_unique_thresholds": len({(threshold.raw_feature, threshold.value) for threshold in view.thresholds[:prefix]}),
                               "n_train_signatures": floor["n_groups"], "base_features": base_features,
                               "base_clean_train_mse": float(np.mean((base_predictions[protocol["iterations"]]["train"] - clean_train) ** 2)),
                               "base_clean_evaluation_mse": float(np.mean((base_predictions[protocol["iterations"]]["evaluation"] - clean_evaluation) ** 2)),
                               "base_relative_train_residual": float(np.linalg.norm(base_predictions[protocol["iterations"]]["train"] - base_labels["actual"]) / np.linalg.norm(base_labels["actual"])),
                               "actual_learning_rate": replay["actual_learning_rate"], "actual_l2_leaf_reg": replay["actual_l2_leaf_reg"],
                               "base_model_id": base_id, "base_model_sha256": base_record["sha256"],
                               "baseline_checks": replay["baseline_checks"], "amplitudes": [],
                               "timings": {"base_fit_seconds": base_time["fit_seconds"],
                                           "base_prediction_seconds": base_time["prediction_seconds"] + base_trajectory_seconds,
                                           "base_leaf_evidence_seconds": base_time["leaf_evidence_seconds"],
                                           "base_operator_seconds": operator_seconds, "base_feature_seconds": base_feature_seconds,
                                           "cheap_fit_seconds": 0.0, "cheap_prediction_seconds": 0.0, "cheap_decomposition_seconds": 0.0,
                                           "cheap_leaf_evidence_seconds": 0.0,
                                           "full_fit_seconds": 0.0, "full_prediction_seconds": 0.0, "full_decomposition_seconds": 0.0,
                                           "full_leaf_evidence_seconds": 0.0,
                                           "label_preparation_seconds": 0.0, "io_seconds": context_io + base_time["io_seconds"]}}
                    for amplitude_index, epsilon in enumerate(protocol["epsilon_values"]):
                        amplitude = {"epsilon": epsilon, "amplitude_index": amplitude_index,
                                     "pilot_direction_ids": [], "response_direction_ids": [],
                                     "cheap_discrepancy_values": [], "full_discrepancy_values": [],
                                     "cheap_train_discrepancy_values": [], "full_train_discrepancy_values": [],
                                     "valid_directions": True}
                        snapshot_row = None
                        for phase, directions, trees in (("pilot", raw["pilot_directions"], protocol["pilot_iterations"]),
                                                         ("response", raw["response_directions"], protocol["iterations"])):
                            if phase == "response":
                                cheap = float(np.mean(amplitude["cheap_discrepancy_values"]))
                                snapshot_row = {"id": f"{context_id}__e{amplitude_index}", "context_id": context_id,
                                                "seed": seed, "teacher": teacher, "prefix": prefix, "epsilon": epsilon,
                                                **base_features, "cheap_discrepancy": cheap,
                                                "cheap_train_discrepancy": float(np.mean(amplitude["cheap_train_discrepancy_values"]))}
                                _append(features_stream, {**snapshot_row, "features": forecast_features(snapshot_row),
                                                          "scope": "Frozen before the four independent full-response refits for this context/amplitude",
                                                          "elapsed_seconds": state.elapsed()})
                            for direction_index, z in enumerate(directions):
                                fit_id = f"{context_id}__e{amplitude_index}__{phase}{direction_index}"
                                started = time.perf_counter()
                                nominal_delta = epsilon * z
                                nominal_perturbed = base_labels["actual"] + nominal_delta
                                perturbed = canonicalize_targets(nominal_perturbed)
                                actual_delta = perturbed["actual"] - base_labels["actual"]
                                valid = bool(np.any(actual_delta != 0))
                                actual_rounding_delta = actual_delta - nominal_delta
                                label_preparation_seconds = time.perf_counter() - started
                                context["timings"]["label_preparation_seconds"] += label_preparation_seconds
                                input_arrays = {"nominal_targets": nominal_perturbed, "native_targets": perturbed["native"],
                                                "actual_targets": perturbed["actual"], "z": z,
                                                "nominal_delta": nominal_delta, "actual_delta": actual_delta,
                                                "actual_delta_minus_nominal_delta": actual_rounding_delta}
                                for key, array in input_arrays.items():
                                    state.data_arrays[f"{fit_id}__{key}"] = array
                                _, arrays, timing, record = _execute_fit(state, fit_id, phase, x_train, x_eval,
                                                                         perturbed["actual"], trees, model_seed, input_arrays)
                                started = time.perf_counter()
                                metrics, array_keys = {}, {}
                                for split in ("train", "evaluation"):
                                    full_delta = arrays[f"{split}__prediction"] - base_predictions[trees][split]
                                    frozen_delta = replay["operators"][trees][split] @ actual_delta
                                    parts = decomposition(full_delta, frozen_delta, epsilon)
                                    keys = {"full_delta": f"{fit_id}__{split}__full_delta",
                                            "frozen_delta": f"{fit_id}__{split}__frozen_delta",
                                            "structural_delta": f"{fit_id}__{split}__structural_delta",
                                            "prediction": f"{fit_id}__{split}__prediction"}
                                    state.prediction_arrays[keys["full_delta"]] = full_delta
                                    state.prediction_arrays[keys["frozen_delta"]] = frozen_delta
                                    state.prediction_arrays[keys["structural_delta"]] = parts["structural_delta"]
                                    metrics[split] = _scalar_decomposition(parts)
                                    array_keys[split] = keys
                                decomposition_seconds = time.perf_counter() - started
                                prefix_time = "cheap" if phase == "pilot" else "full"
                                for label in ("fit", "prediction", "leaf_evidence"):
                                    context["timings"][f"{prefix_time}_{label}_seconds"] += timing[f"{label}_seconds"]
                                context["timings"][f"{prefix_time}_decomposition_seconds"] += decomposition_seconds
                                context["timings"]["io_seconds"] += timing["io_seconds"]
                                direction = {"id": fit_id, "context_id": context_id, "seed": seed, "teacher": teacher,
                                             "prefix": prefix, "epsilon": epsilon, "amplitude_index": amplitude_index,
                                             "phase": phase, "direction": direction_index, "trees": trees, "valid": valid,
                                             "zero_actual_delta": not valid, "actual_delta_sha256": array_hash(actual_delta),
                                             "nominal_z_sha256": array_hash(z), "actual_delta_l2": float(np.linalg.norm(actual_delta)),
                                             "rounding_residual_l2": float(np.linalg.norm(actual_rounding_delta)),
                                             "metrics": metrics, "array_keys": array_keys,
                                             "clean_train_mse": float(np.mean((arrays["train__prediction"] - clean_train) ** 2)),
                                             "clean_evaluation_mse": float(np.mean((arrays["evaluation__prediction"] - clean_evaluation) ** 2)),
                                             "train_mse_to_actual_perturbed_targets": float(np.mean((arrays["train__prediction"] - perturbed["actual"]) ** 2)),
                                             "model_sha256": record["sha256"], "raw_evidence": record["raw_evidence"],
                                             "timings": {**timing, "decomposition_seconds": decomposition_seconds,
                                                         "label_preparation_seconds": label_preparation_seconds}}
                                direction_rows.append(direction)
                                _append(directions_stream, direction)
                                amplitude[f"{phase}_direction_ids"].append(fit_id)
                                key = "cheap" if phase == "pilot" else "full"
                                amplitude[f"{key}_discrepancy_values"].append(metrics["evaluation"]["normalized_structural_discrepancy"])
                                amplitude[f"{key}_train_discrepancy_values"].append(metrics["train"]["normalized_structural_discrepancy"])
                                amplitude["valid_directions"] = amplitude["valid_directions"] and valid
                        amplitude["cheap_discrepancy"] = float(np.mean(amplitude["cheap_discrepancy_values"]))
                        amplitude["full_discrepancy"] = float(np.mean(amplitude["full_discrepancy_values"]))
                        forecast_rows.append({**snapshot_row, "full_discrepancy": amplitude["full_discrepancy"],
                                              "full_train_discrepancy": float(np.mean(amplitude["full_train_discrepancy_values"])),
                                              "valid_directions": amplitude["valid_directions"]})
                        context["amplitudes"].append(amplitude)
                    context["timings"]["io_seconds"] += state.checkpoint_arrays()
                    context["timings"]["context_elapsed_seconds"] = time.perf_counter() - context_started
                    context_rows.append(context)
                    _append(contexts_stream, context)
                    for key, value in context["timings"].items():
                        total_times[key] += value
                    state.contexts_completed += 1
                    state.fit_id = None
                    state.guard("after_context")
                    state.update("context_complete")
        if len(context_rows) != protocol["expected_contexts"] or state.fits_completed != protocol["expected_fits"] or len(forecast_rows) != protocol["expected_forecast_rows"]:
            raise ValueError("Completed matrix differs from the frozen declared counts")
        started = time.perf_counter()
        forecast = oof_forecasts(forecast_rows, protocol["seeds"], alpha=protocol["forecast"]["ridge_alpha"])
        total_times["forecast_seconds"] = time.perf_counter() - started
        with (output / "forecast_rows.jsonl").open("x", encoding="utf-8") as stream:
            for row in forecast["rows"]:
                _append(stream, row)
        cheap_wall = sum(total_times[f"cheap_{name}_seconds"] for name in ("fit", "prediction", "decomposition"))
        full_wall = sum(total_times[f"full_{name}_seconds"] for name in ("fit", "prediction", "decomposition"))
        common_base_wall = sum(total_times[name] for name in ("base_fit_seconds", "base_prediction_seconds", "base_operator_seconds", "base_feature_seconds"))
        decisions = decision_gates(forecast, protocol, replay_passed=all(check["passed"] for check in all_baseline_checks), cheap_wall=cheap_wall, full_wall=full_wall)
        analysis = {"schema_version": 1, "protocol_id": protocol["protocol_id"], "protocol_sha256": file_hash(protocol_path),
                    "mode": "discovery", "n_contexts": len(context_rows), "n_forecast_rows": len(forecast_rows),
                    "n_directions": len(direction_rows), "n_fits": state.fits_completed,
                    "seeds": protocol["seeds"], "teachers": protocol["teachers"], "prefix_budgets": protocol["prefix_budgets"],
                    "epsilon_values": protocol["epsilon_values"], "baseline_checks": all_baseline_checks,
                    "max_baseline_absolute_error": max(check["max_absolute_error"] for check in all_baseline_checks),
                    "max_absolute_norm_closure_error": max(abs(row["metrics"][split]["closure_error"]) for row in direction_rows for split in ("train", "evaluation")),
                    "invalid_zero_delta_direction_count": sum(not row["valid"] for row in direction_rows),
                    "forecast": {key: value for key, value in forecast.items() if key != "rows"},
                    "decision": decisions, "timings": dict(total_times),
                    "costs": {"cheap_fit_prediction_decomposition_seconds": cheap_wall,
                              "full_fit_prediction_decomposition_seconds": full_wall,
                              "common_base_fit_predictions_operator_features_seconds": common_base_wall,
                              "extra_cheap_full_ratio": cheap_wall / full_wall if full_wall > 0 else None,
                              "secondary_end_to_end_ratio_common_base_in_both": (common_base_wall + cheap_wall + total_times["forecast_seconds"]) / (common_base_wall + full_wall) if common_base_wall + full_wall > 0 else None,
                              "secondary_end_to_end_forecast_cost_over_full_refits": (common_base_wall + cheap_wall + total_times["forecast_seconds"]) / full_wall if full_wall > 0 else None,
                              "io_seconds_separate": total_times["io_seconds"],
                              "leaf_evidence_seconds_separate": {role: total_times[f"{role}_leaf_evidence_seconds"] for role in ("base", "cheap", "full")},
                              "scope": "Primary P3 assumes the full basefit and frozen operator are already available; secondary ratios include their measured costs explicitly. Environment-specific timings, not universal speedup."},
                    "forecast_row_file": "forecast_rows.jsonl", "feature_snapshot_file": "feature_snapshots.jsonl",
                    "direction_streams_shared_across_teachers_prefixes_and_amplitudes": True,
                    "pilot_response_streams_disjoint": True,
                    "unlabeled_evaluation_covariates_used_for_forecast": True,
                    "clean_teacher_labels_used_for_forecast": False,
                    "test_adapted_grid": False, "double_descent_declared": False,
                    "independent_confirmation": False, "priority_established": False,
                    "gpu_jobs": 0, "paid_api_spend": 0, "download_bytes": 0,
                    "elapsed_seconds": state.elapsed(), "scientific_validity": "not_assessed"}
        atomic_json(output / "timings.json", {"totals": dict(total_times), "costs": analysis["costs"],
                                              "per_context": [{"context_id": row["id"], **row["timings"]} for row in context_rows]})
        atomic_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(),
                                                   "numpy": np.__version__, "catboost": catboost.__version__,
                                                   "threads": protocol["threads"], "threadpools": threadpool_info(),
                                                   "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                                                   "torch_imported": "torch" in sys.modules, "gpu_jobs": 0,
                                                   "download_bytes": 0, "paid_api_spend": 0})
        state.guard("before_completion")
        analysis["artifact_hashes"] = {name: file_hash(output / name) for name in ("contexts.jsonl", "directions.jsonl", "data.npz", "predictions.npz", "operators.npz", "models.json", "forecast_rows.jsonl", "feature_snapshots.jsonl", "timings.json", "environment.json")}
        atomic_json(output / "analysis.json", analysis)
        state.update("complete", decision=decisions)
        return analysis
    except BaseException as exc:
        failure = {"exception_type": type(exc).__name__, "message": str(exc),
                   "context_id": state.context, "fit_id": state.fit_id,
                   "fits_completed": state.fits_completed, "contexts_completed": state.contexts_completed,
                   "elapsed_seconds": state.elapsed(), "scientific_interpretation": "not_available; partial attempt retained"}
        atomic_json(output / "failure.json", failure)
        state.update("failed", failure=failure)
        raise
    finally:
        contexts_stream.close()
        directions_stream.close()
        features_stream.close()


def run_study(protocol: dict[str, Any], output: Path, protocol_path: Path) -> dict[str, Any]:
    """The sole study execution entry point; always dispatched after plan freeze."""
    validate_protocol(protocol)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=protocol["threads"]):
        return _run_study(protocol, output, protocol_path)
