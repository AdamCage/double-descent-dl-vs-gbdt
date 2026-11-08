"""R2 discovery and confirmation driver for linear ridge and frozen ReLU features.

The confirmatory menu is chosen from discovery wall-clock time only. Clean risk,
coefficient norms and confirmation seeds are not inputs to that choice.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from src.controlled_synthetic import (
    apply_latent_spectrum,
    column_std,
    latent_standardizer,
    prepare_draw,
    prepare_evaluation,
    sigma_from_snr,
    teacher_signal,
)
from src.ridge_svd import fit_ridge_path, interpolation_flag, predict

FORBIDDEN_EXPORT_KEYS = {
    "clean_mse",
    "noisy_mse",
    "coefficient_l2",
    "condition_number",
    "prediction_outcomes",
    "cells",
}
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def limit_threads(limit: int = 4) -> None:
    if limit > 4:
        raise RuntimeError("The charter allows at most 4 CPU threads")
    for key in THREAD_KEYS:
        os.environ[key] = str(limit)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(pythonize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def pythonize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): pythonize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [pythonize(item) for item in value]
    if isinstance(value, np.ndarray):
        return pythonize(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Refusing a non-finite JSON number")
        return float(format(value, ".17g"))
    return value


def _number_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    numbers = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{name} contains a non-finite entry")
        numbers.append(float(item))
    return numbers


def _int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    numbers = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{name} must contain nonnegative integers")
        numbers.append(item)
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{name} contains duplicates")
    return numbers


def validate_protocol(protocol: dict[str, Any]) -> None:
    n_train = protocol.get("n_train")
    if isinstance(n_train, bool) or not isinstance(n_train, int) or n_train < 2:
        raise ValueError("n_train must be an integer >= 2")
    columns = _int_list(protocol.get("columns"), "columns")
    if columns != sorted(columns):
        raise ValueError("columns must be increasing")
    if int(protocol.get("p_max")) != max(columns):
        raise ValueError("p_max must equal the largest column")
    if n_train * max(columns) > 2_000_000:
        raise ValueError("design exceeds the predeclared size cap")
    n_eval = protocol.get("n_eval")
    if isinstance(n_eval, bool) or not isinstance(n_eval, int) or not 2 <= n_eval <= 10000:
        raise ValueError("n_eval must be an integer from 2 to 10000")
    window = _int_list(protocol.get("window_columns"), "window_columns")
    if not set(window).issubset(columns):
        raise ValueError("window columns must be on the menu")
    fallback = _int_list(protocol["reduction_rule"]["fallback_columns"], "fallback_columns")
    if not set(fallback).issubset(columns):
        raise ValueError("fallback columns must be a subset of the menu")
    if not any(column < n_train for column in fallback) or not any(column > n_train for column in fallback):
        raise ValueError("fallback menu must place columns on both sides of n_train")
    if n_train not in fallback or not set(window).issubset(fallback):
        raise ValueError("fallback menu must keep n_train and the primary window")
    discovery = _int_list(protocol.get("discovery_data_seeds"), "discovery_data_seeds")
    confirmation = _int_list(protocol.get("confirmation_data_seeds"), "confirmation_data_seeds")
    if set(discovery) & set(confirmation):
        raise ValueError("discovery and confirmation seeds overlap")
    eval_seeds = [
        protocol.get("discovery_eval_seed"),
        protocol.get("discovery_eval_noise_seed"),
        protocol.get("confirmation_eval_seed"),
        protocol.get("confirmation_eval_noise_seed"),
    ]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in eval_seeds):
        raise ValueError("evaluation seeds must be integers")
    if set(eval_seeds) & (set(discovery) | set(confirmation)) or len(set(eval_seeds)) != 4:
        raise ValueError("evaluation seeds must be unique and disjoint from draw seeds")
    snr = _number_list(protocol.get("snr"), "snr")
    if snr != [1.0, 5.0]:
        raise ValueError("snr menu is fixed at 1 and 5 before any fit")
    spectra = protocol.get("spectra")
    if spectra != ["isotropic", "geometric"]:
        raise ValueError("linear spectra must be isotropic then geometric")
    ratio = protocol.get("geometric_ratio")
    beta_ratio = protocol.get("beta_geometric_ratio")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) < 1
           for value in (ratio, beta_ratio)):
        raise ValueError("geometric ratios must lie in (0, 1)")
    lambdas = _number_list(protocol.get("lambdas"), "lambdas")
    if lambdas[0] != 0.0 or any(value < 0 for value in lambdas):
        raise ValueError("lambdas must start with 0 and be nonnegative")
    tolerance = protocol.get("relative_train_residual_tolerance")
    rcond = protocol.get("rcond_relative")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
           for value in (tolerance, rcond)):
        raise ValueError("tolerance and rcond must be positive")
    if int(protocol.get("threads")) != 4 or int(protocol.get("gpu_jobs")) != 0:
        raise ValueError("protocol must keep 4 CPU threads and 0 GPU jobs")
    pilot = protocol.get("timing_pilot_cell")
    if not isinstance(pilot, dict):
        raise ValueError("timing pilot cell is missing")
    if pilot.get("data_seed") not in discovery or pilot.get("column") not in columns:
        raise ValueError("timing pilot cell is outside the discovery menu")
    if pilot.get("family") != "linear" or pilot.get("spectrum") != "isotropic":
        raise ValueError("timing pilot must be the isotropic linear cell")
    if float(pilot.get("snr")) not in snr or float(pilot.get("ridge_lambda")) != 0.0:
        raise ValueError("timing pilot snr or lambda is not on the menu")
    rule = protocol.get("reduction_rule")
    if rule.get("statistic") != "discovery_wall_seconds_per_draw":
        raise ValueError("reduction statistic must be discovery wall-clock per draw")
    if float(rule.get("safety_factor")) != 1.5:
        raise ValueError("reduction safety factor must stay 1.5")
    cap = rule.get("keep_full_menu_if_projected_confirmation_seconds_below")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap <= 0:
        raise ValueError("confirmation time cap must be positive")
    features = protocol.get("random_features")
    if features.get("spectra") != ["isotropic"] or features.get("lambdas") != [0.0]:
        raise ValueError("random features stay on isotropic inputs and lambda 0")
    if _number_list(features.get("snr"), "random_features.snr") != snr:
        raise ValueError("random-feature snr menu must match the linear snr menu")
    minimum = protocol.get("minimum_bracketed_draws")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_bracketed_draws must be a positive integer")
    margins = protocol.get("contrast_margins")
    for key in ("clean_risk_absolute", "coefficient_norm_relative", "condition_number_relative"):
        margin = margins.get(key)
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin <= 0:
            raise ValueError(f"contrast margin {key} must be positive")


def phase_seeds(protocol: dict[str, Any], phase: str) -> list[int]:
    if phase == "discovery":
        return list(protocol["discovery_data_seeds"])
    if phase == "confirmation":
        return list(protocol["confirmation_data_seeds"])
    raise ValueError(phase)


def reduce_menu(discovery_wall_seconds: float, n_discovery_draws: int,
                n_confirmation_draws: int, protocol: dict[str, Any]) -> dict[str, Any]:
    """Choose the confirmatory columns from elapsed time only."""
    if not math.isfinite(discovery_wall_seconds) or discovery_wall_seconds < 0:
        raise ValueError("discovery wall time must be finite and nonnegative")
    if n_discovery_draws < 1 or n_confirmation_draws < 1:
        raise ValueError("both phases need at least one draw")
    rule = protocol["reduction_rule"]
    per_draw = discovery_wall_seconds / n_discovery_draws
    projected = per_draw * n_confirmation_draws * float(rule["safety_factor"])
    keep_full = projected < float(rule["keep_full_menu_if_projected_confirmation_seconds_below"])
    kept = list(protocol["columns"] if keep_full else rule["fallback_columns"])
    return {
        "rule": "time_only",
        "statistic": rule["statistic"],
        "used_clean_risk": False,
        "discovery_wall_seconds": discovery_wall_seconds,
        "seconds_per_discovery_draw": per_draw,
        "projected_confirmation_seconds": projected,
        "kept_full_menu": keep_full,
        "kept_columns": kept,
    }


def freeze_confirmation_grid(export: dict[str, Any], protocol: dict[str, Any],
                             export_sha256: str) -> dict[str, Any]:
    if FORBIDDEN_EXPORT_KEYS & set(export):
        raise RuntimeError("Refusing to freeze a grid from an export that carries risk fields")
    reduction = export.get("reduction")
    if not isinstance(reduction, dict) or reduction.get("used_clean_risk") is not False or reduction.get("rule") != "time_only":
        raise RuntimeError("Discovery export has no time-only reduction")
    kept = reduction.get("kept_columns")
    full = list(protocol["columns"])
    fallback = list(protocol["reduction_rule"]["fallback_columns"])
    if kept not in (full, fallback):
        raise RuntimeError("Kept columns are not one of the two predeclared menus")
    if export.get("protocol_sha256") != sha256_bytes(canonical_protocol(protocol)):
        raise RuntimeError("Discovery export was built from a different protocol")
    audit = export.get("seed_audit")
    if not isinstance(audit, dict) or audit.get("confirmation_seeds_absent") is not True:
        raise RuntimeError("Discovery export does not attest that confirmation seeds were absent")
    if audit.get("data_seeds") != list(protocol["discovery_data_seeds"]):
        raise RuntimeError("Discovery export seeds differ from the protocol")
    return {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "kept_columns": kept,
        "used_clean_risk": False,
        "rule": "time_only",
        "discovery_export_sha256": export_sha256,
        "discovery_protocol_sha256": export["protocol_sha256"],
        "projected_confirmation_seconds": reduction["projected_confirmation_seconds"],
        "median_note": "The projection uses discovery wall-clock per draw, not a cell chosen by risk.",
    }


def canonical_protocol(protocol: dict[str, Any]) -> bytes:
    return (json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mean_se(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("contrast values must be finite and nonempty")
    mean = float(array.mean())
    se = None if array.size < 2 else float(array.std(ddof=1) / math.sqrt(array.size))
    return {"mean": mean, "se": se, "minimum": float(array.min()), "maximum": float(array.max()), "n": int(array.size)}


def _directional(values: list[float], margin: float) -> str:
    stats = _mean_se(values)
    mean = float(stats["mean"])
    se = stats["se"]
    if mean <= 0:
        return "nonpositive"
    if mean < margin or se is None or mean <= float(se):
        return "below"
    return "pass"


def _combine(parts: list[str]) -> str:
    if all(part == "pass" for part in parts):
        return "supported"
    if all(part == "nonpositive" for part in parts):
        return "refuted"
    if "pass" in parts and "nonpositive" in parts:
        return "not_uniform"
    return "inconclusive_within_noise"


def _rows(cells: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    selected = []
    for cell in cells:
        if all(cell.get(key) == value for key, value in filters.items()):
            selected.append(cell)
    return selected


def _bracketed(cells: list[dict[str, Any]], protocol: dict[str, Any], family: str,
               spectrum: str, snr: float) -> set[int]:
    n_train = int(protocol["n_train"])
    tolerance = float(protocol["relative_train_residual_tolerance"])
    seeds = {int(cell["data_seed"]) for cell in cells
             if cell["family"] == family and cell["spectrum"] == spectrum and cell["snr"] == snr
             and cell["ridge_lambda"] == 0.0}
    kept = set()
    for seed in seeds:
        rows = _rows(cells, family=family, spectrum=spectrum, snr=snr, ridge_lambda=0.0, data_seed=seed)
        under = [row for row in rows if int(row["column"]) < n_train and row["residual_relative"] > tolerance
                 and row["interpolating"] is False]
        over = [row for row in rows if int(row["column"]) > n_train and row["interpolating"] is True]
        if under and over:
            kept.add(seed)
    return kept


def _window_value(cells: list[dict[str, Any]], protocol: dict[str, Any], seed: int, field: str,
                  **filters: Any) -> float:
    rows = _rows(cells, data_seed=seed, **filters)
    by_column = {int(row["column"]): row for row in rows}
    values = []
    for column in protocol["window_columns"]:
        if column not in by_column:
            raise ValueError(f"Missing window column {column} for seed {seed}")
        values.append(float(by_column[column][field]))
    return float(np.mean(values))


def _curve(cells: list[dict[str, Any]], columns: list[int], field: str, **filters: Any) -> list[float]:
    means = []
    for column in columns:
        rows = _rows(cells, column=column, **filters)
        if not rows:
            raise ValueError(f"Missing curve column {column}")
        means.append(float(np.mean([float(row[field]) for row in rows])))
    return means


def _interior_peak(values: list[float]) -> bool:
    if len(values) < 3:
        return False
    for index in range(1, len(values) - 1):
        if values[index] > values[index - 1] and values[index] > values[index + 1]:
            return True
    return False


def evaluate_predictions(cells: list[dict[str, Any]], protocol: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_cells = [cell for cell in cells if cell["phase"] == phase]
    ordered_snr = sorted(float(value) for value in protocol["snr"])
    if ordered_snr != [1.0, 5.0]:
        raise ValueError("The contrast definitions are bound to snr 1 and snr 5")
    noisier, quieter = ordered_snr
    minimum = int(protocol["minimum_bracketed_draws"])
    margins = protocol["contrast_margins"]
    risk_margin = float(margins["clean_risk_absolute"])
    norm_margin = float(margins["coefficient_norm_relative"])
    condition_margin = float(margins["condition_number_relative"])

    def gate(seeds: set[int]) -> str | None:
        if not seeds:
            return "inconclusive_no_bracket"
        if len(seeds) < minimum:
            return "inconclusive_few_brackets"
        return None

    def noise_block(spectrum: str) -> dict[str, Any]:
        low = _bracketed(phase_cells, protocol, "linear", spectrum, noisier)
        high = _bracketed(phase_cells, protocol, "linear", spectrum, quieter)
        seeds = low & high
        blocked = gate(seeds)
        if blocked:
            return {"spectrum": spectrum, "outcome": blocked, "n_bracketed": len(seeds)}
        risk_deltas, norm_deltas = [], []
        for seed in sorted(seeds):
            risk_low = _window_value(phase_cells, protocol, seed, "clean_mse",
                                     family="linear", spectrum=spectrum, snr=noisier, ridge_lambda=0.0)
            risk_high = _window_value(phase_cells, protocol, seed, "clean_mse",
                                      family="linear", spectrum=spectrum, snr=quieter, ridge_lambda=0.0)
            norm_low = _window_value(phase_cells, protocol, seed, "coefficient_l2",
                                     family="linear", spectrum=spectrum, snr=noisier, ridge_lambda=0.0)
            norm_high = _window_value(phase_cells, protocol, seed, "coefficient_l2",
                                      family="linear", spectrum=spectrum, snr=quieter, ridge_lambda=0.0)
            risk_deltas.append(risk_low - risk_high)
            norm_deltas.append((norm_low - norm_high) / max(norm_high, 1e-12))
        parts = [_directional(risk_deltas, risk_margin), _directional(norm_deltas, norm_margin)]
        return {
            "spectrum": spectrum,
            "outcome": _combine(parts),
            "n_bracketed": len(seeds),
            "clean_risk_delta": _mean_se(risk_deltas),
            "coefficient_norm_relative_delta": _mean_se(norm_deltas),
            "components": parts,
        }

    def spectrum_block(snr: float) -> dict[str, Any]:
        geometric = _bracketed(phase_cells, protocol, "linear", "geometric", snr)
        isotropic = _bracketed(phase_cells, protocol, "linear", "isotropic", snr)
        seeds = geometric & isotropic
        blocked = gate(seeds)
        if blocked:
            return {"snr": snr, "outcome": blocked, "n_bracketed": len(seeds)}
        risk_deltas, condition_deltas = [], []
        for seed in sorted(seeds):
            risk_geometric = _window_value(phase_cells, protocol, seed, "clean_mse",
                                           family="linear", spectrum="geometric", snr=snr, ridge_lambda=0.0)
            risk_isotropic = _window_value(phase_cells, protocol, seed, "clean_mse",
                                           family="linear", spectrum="isotropic", snr=snr, ridge_lambda=0.0)
            condition_geometric = _rows(phase_cells, family="linear", spectrum="geometric", snr=snr,
                                        ridge_lambda=0.0, data_seed=seed, column=int(protocol["n_train"]))
            condition_isotropic = _rows(phase_cells, family="linear", spectrum="isotropic", snr=snr,
                                        ridge_lambda=0.0, data_seed=seed, column=int(protocol["n_train"]))
            if len(condition_geometric) != 1 or len(condition_isotropic) != 1:
                raise ValueError("Expected one square-design condition per draw")
            base = float(condition_isotropic[0]["condition_number"])
            gap = (float(condition_geometric[0]["condition_number"]) - base) / max(base, 1e-12)
            risk_deltas.append(risk_geometric - risk_isotropic)
            condition_deltas.append(gap)
        parts = [_directional(risk_deltas, risk_margin), _directional(condition_deltas, condition_margin)]
        return {
            "snr": snr,
            "outcome": _combine(parts),
            "n_bracketed": len(seeds),
            "clean_risk_delta": _mean_se(risk_deltas),
            "condition_relative_delta": _mean_se(condition_deltas),
            "components": parts,
        }

    def ridge_block(spectrum: str, snr: float) -> dict[str, Any]:
        seeds = _bracketed(phase_cells, protocol, "linear", spectrum, snr)
        blocked = gate(seeds)
        if blocked:
            return {"spectrum": spectrum, "snr": snr, "outcome": blocked, "n_bracketed": len(seeds)}
        parts = []
        by_lambda = []
        for ridge_lambda in protocol["lambdas"]:
            if ridge_lambda == 0.0:
                continue
            risk_deltas, norm_deltas = [], []
            for seed in sorted(seeds):
                base_risk = _window_value(phase_cells, protocol, seed, "clean_mse",
                                          family="linear", spectrum=spectrum, snr=snr, ridge_lambda=0.0)
                new_risk = _window_value(phase_cells, protocol, seed, "clean_mse",
                                         family="linear", spectrum=spectrum, snr=snr, ridge_lambda=ridge_lambda)
                base_norm = _window_value(phase_cells, protocol, seed, "coefficient_l2",
                                          family="linear", spectrum=spectrum, snr=snr, ridge_lambda=0.0)
                new_norm = _window_value(phase_cells, protocol, seed, "coefficient_l2",
                                         family="linear", spectrum=spectrum, snr=snr, ridge_lambda=ridge_lambda)
                risk_deltas.append(base_risk - new_risk)
                norm_deltas.append((base_norm - new_norm) / max(base_norm, 1e-12))
            component = [_directional(risk_deltas, risk_margin), _directional(norm_deltas, norm_margin)]
            parts.extend(component)
            by_lambda.append({
                "ridge_lambda": ridge_lambda,
                "components": component,
                "clean_risk_delta": _mean_se(risk_deltas),
                "coefficient_norm_relative_delta": _mean_se(norm_deltas),
            })
        return {
            "spectrum": spectrum,
            "snr": snr,
            "outcome": _combine(parts),
            "n_bracketed": len(seeds),
            "lambdas": by_lambda,
        }

    def random_block(snr: float) -> dict[str, Any]:
        seeds = _bracketed(phase_cells, protocol, "random_features", "isotropic", snr)
        columns = [int(cell["column"]) for cell in phase_cells if cell["family"] == "random_features"]
        menu = sorted(set(columns))
        blocked = gate(seeds)
        recorded = bool(_rows(phase_cells, family="random_features", spectrum="isotropic", snr=snr, ridge_lambda=0.0))
        if blocked:
            return {"snr": snr, "recorded": recorded, "coupling_outcome": blocked, "n_bracketed": len(seeds)}
        n_train = int(protocol["n_train"])
        under = max(column for column in menu if column < n_train)
        over = min(column for column in menu if column > n_train)
        selected = [cell for cell in phase_cells if cell["family"] == "random_features" and cell["snr"] == snr
                    and cell["data_seed"] in seeds and cell["ridge_lambda"] == 0.0]
        under_values, over_values = [], []
        for seed in sorted(seeds):
            under_row = _rows(selected, data_seed=seed, column=under)[0]
            over_row = _rows(selected, data_seed=seed, column=over)[0]
            under_values.append(float(under_row["gram_condition_number"]))
            over_values.append(float(over_row["gram_condition_number"]))
        relative = [(new - old) / max(old, 1e-12) for old, new in zip(under_values, over_values)]
        condition_status = _directional(relative, condition_margin)
        curve = _curve(selected, menu, "clean_mse", family="random_features", spectrum="isotropic",
                       snr=snr, ridge_lambda=0.0)
        # The curve helper filters selected further; pass the already selected rows.
        peak = _interior_peak(curve)
        if condition_status == "pass" and not peak:
            coupling = "coupling_refuted"
        elif condition_status == "pass" and peak:
            coupling = "coupling_consistent"
        elif condition_status == "nonpositive":
            coupling = "recorded_no_condition_increase"
        else:
            coupling = "inconclusive_within_noise"
        return {
            "snr": snr,
            "recorded": recorded,
            "coupling_outcome": coupling,
            "n_bracketed": len(seeds),
            "condition_relative_delta": _mean_se(relative),
            "interior_clean_risk_peak": peak,
            "clean_risk_curve": curve,
            "columns": menu,
        }

    linear_curve_cells = _rows(phase_cells, family="linear", spectrum="isotropic", snr=1.0, ridge_lambda=0.0)
    linear_columns = sorted({int(cell["column"]) for cell in linear_curve_cells})
    descriptive = {
        "family": "linear",
        "spectrum": "isotropic",
        "snr": 1.0,
        "ridge_lambda": 0.0,
        "interior_clean_risk_peak": _interior_peak(_curve(linear_curve_cells, linear_columns, "clean_mse",
                                                          family="linear", spectrum="isotropic",
                                                          snr=1.0, ridge_lambda=0.0)) if linear_columns else False,
        "note": "The interior-peak flag is descriptive. It is not a registered success condition and it does not change the menu.",
    }
    return {
        "phase": phase,
        "label": "exploratory_discovery" if phase == "discovery" else "confirmation",
        "noise": {"primary": noise_block("isotropic"), "secondary": noise_block("geometric")},
        "spectrum": {"primary": spectrum_block(noisier), "secondary": spectrum_block(quieter)},
        "ridge": {"primary": ridge_block("isotropic", noisier), "secondary": ridge_block("geometric", noisier)},
        "random_features": {"snr_1": random_block(noisier), "snr_5": random_block(quieter)},
        "descriptive_shape": descriptive,
        "scientific_validity": "not_assessed",
    }


def _cell_record(common: dict[str, Any], result: dict[str, Any], solution: dict[str, Any],
                 clean_mse: float, noisy_mse: float, tolerance: float, elapsed: float) -> dict[str, Any]:
    flag = interpolation_flag(solution, result, tolerance)
    return {
        **common,
        "ridge_lambda": float(solution["ridge_lambda"]),
        "rank": result["rank"],
        "rank_target": result["rank_target"],
        "residual_l2": solution["residual_l2"],
        "residual_relative": solution["residual_relative"],
        "coefficient_l2": solution["coefficient_l2"],
        "intercept": solution["intercept"],
        "train_mse": solution["train_mse"],
        "clean_mse": clean_mse,
        "noisy_mse": noisy_mse,
        "singular_value_max": result["singular_value_max"],
        "singular_value_min_kept": result["singular_value_min_kept"],
        "condition_number": result["condition_number"],
        "gram_condition_number": result["gram_condition_number"],
        "zero_variance_columns": result["fit"]["zero_variance_columns"],
        "interpolating": flag,
        "elapsed_seconds": elapsed,
        "rcond_relative": common["rcond_relative"],
        "residual_tolerance": tolerance,
    }


def _risks(prediction: np.ndarray, truth: np.ndarray, noisy: np.ndarray) -> tuple[float, float]:
    if not np.isfinite(prediction).all():
        raise RuntimeError("prediction is not finite")
    return float(np.mean((prediction - truth) ** 2)), float(np.mean((prediction - noisy) ** 2))


def run_phase(protocol: dict[str, Any], phase: str, columns: list[int], out_dir: Path) -> dict[str, Any]:
    validate_protocol(protocol)
    if any(column not in protocol["columns"] for column in columns):
        raise ValueError("requested columns are outside the protocol menu")
    if not columns:
        raise ValueError("column menu is empty")
    limit_threads(int(protocol["threads"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tolerance = float(protocol["relative_train_residual_tolerance"])
    rcond = float(protocol["rcond_relative"])
    seeds = phase_seeds(protocol, phase)
    other = set(phase_seeds(protocol, "confirmation" if phase == "discovery" else "discovery"))
    if set(seeds) & other:
        raise RuntimeError("phase seeds overlap")
    evaluation = prepare_evaluation(protocol, phase)
    cells: list[dict[str, Any]] = []
    pilot: dict[str, Any] | None = None
    with threadpool_limits(limits=int(protocol["threads"])):
        np.linalg.svd(np.eye(2, dtype=np.float64))
        runtime = _runtime_report(protocol)
        for data_seed in seeds:
            if data_seed in other:
                raise RuntimeError("A confirmation seed entered the other phase")
            draw = prepare_draw(protocol, phase, data_seed)
            captured = {
                int(column): float(np.sum(draw["beta"][:column] ** 2))
                for column in columns
            }
            latent = latent_standardizer(draw["covariates"])
            for spectrum in protocol["spectra"]:
                scales = column_std(protocol, spectrum, int(protocol["p_max"]))
                train = apply_latent_spectrum(draw["covariates"], latent, scales)
                evaluate = apply_latent_spectrum(evaluation["covariates"], latent, scales)
                truth = teacher_signal(evaluate, draw["beta"])
                population = float(np.sum((draw["beta"] ** 2) * (scales ** 2)))
                empirical = float(np.sum(teacher_signal(train, draw["beta"]) ** 2))
                for snr in protocol["snr"]:
                    sigma = sigma_from_snr(float(snr))
                    response = teacher_signal(train, draw["beta"]) + sigma * draw["label_noise"]
                    noisy_target = truth + sigma * evaluation["label_noise"]
                    for column in columns:
                        block_started = time.perf_counter()
                        fitted = fit_ridge_path(
                            train[:, :column], response, list(protocol["lambdas"]), rcond,
                            rescale_columns=False,
                        )
                        elapsed = time.perf_counter() - block_started
                        for solution in fitted["solutions"]:
                            predicted = predict(evaluate[:, :column], fitted, solution)
                            clean_mse, noisy_mse = _risks(predicted, truth, noisy_target)
                            record = _cell_record({
                                "phase": phase,
                                "family": "linear",
                                "data_seed": data_seed,
                                "spectrum": spectrum,
                                "snr": float(snr),
                                "sigma": sigma,
                                "column": int(column),
                                "p_over_n": float(column) / float(protocol["n_train"]),
                                "n_train": int(protocol["n_train"]),
                                "n_eval": int(protocol["n_eval"]),
                                "teacher_energy_captured": captured[int(column)],
                                "population_signal_energy": population,
                                "empirical_train_signal_energy": empirical,
                                "empirical_snr": empirical / (int(protocol["n_train"]) * sigma ** 2),
                                "rcond_relative": rcond,
                                "channels": draw["channels"],
                            }, fitted, solution, clean_mse, noisy_mse, tolerance, elapsed)
                            cells.append(record)
                            if _is_pilot(protocol, record):
                                pilot = {
                                    "family": "linear",
                                    "data_seed": data_seed,
                                    "column": int(column),
                                    "spectrum": spectrum,
                                    "snr": float(snr),
                                    "ridge_lambda": 0.0,
                                    "elapsed_seconds": elapsed,
                                    "residual_relative": solution["residual_relative"],
                                    "rank": fitted["rank"],
                                    "rank_target": fitted["rank_target"],
                                    "interpolating": record["interpolating"],
                                }
            feature_spectra = protocol["random_features"]["spectra"]
            for spectrum in feature_spectra:
                scales = column_std(protocol, spectrum, int(protocol["p_max"]))
                train = apply_latent_spectrum(draw["covariates"], latent, scales)
                evaluate = apply_latent_spectrum(evaluation["covariates"], latent, scales)
                truth = teacher_signal(evaluate, draw["beta"])
                hidden_train = np.maximum(train @ draw["feature_weights"], 0.0)
                hidden_eval = np.maximum(evaluate @ draw["feature_weights"], 0.0)
                for snr in protocol["random_features"]["snr"]:
                    sigma = sigma_from_snr(float(snr))
                    response = teacher_signal(train, draw["beta"]) + sigma * draw["label_noise"]
                    noisy_target = truth + sigma * evaluation["label_noise"]
                    for column in columns:
                        block_started = time.perf_counter()
                        fitted = fit_ridge_path(
                            hidden_train[:, :column], response, list(protocol["random_features"]["lambdas"]), rcond,
                        )
                        elapsed = time.perf_counter() - block_started
                        for solution in fitted["solutions"]:
                            predicted = predict(hidden_eval[:, :column], fitted, solution)
                            clean_mse, noisy_mse = _risks(predicted, truth, noisy_target)
                            cells.append(_cell_record({
                                "phase": phase,
                                "family": "random_features",
                                "data_seed": data_seed,
                                "spectrum": spectrum,
                                "snr": float(snr),
                                "sigma": sigma,
                                "column": int(column),
                                "p_over_n": float(column) / float(protocol["n_train"]),
                                "n_train": int(protocol["n_train"]),
                                "n_eval": int(protocol["n_eval"]),
                                "teacher_energy_captured": 1.0,
                                "population_signal_energy": float(np.sum((draw["beta"] ** 2) * (scales ** 2))),
                                "empirical_train_signal_energy": float(np.sum(teacher_signal(train, draw["beta"]) ** 2)),
                                "empirical_snr": float(np.sum(teacher_signal(train, draw["beta"]) ** 2))
                                / (int(protocol["n_train"]) * sigma ** 2),
                                "rcond_relative": rcond,
                                "channels": draw["channels"],
                            }, fitted, solution, clean_mse, noisy_mse, tolerance, elapsed))
        runtime = _runtime_report(protocol)
    expected = _expected_count(protocol, phase, columns)
    if len(cells) != expected:
        raise RuntimeError(f"Cell count {len(cells)} != {expected}")
    identities = [(cell["family"], cell["data_seed"], cell["spectrum"], cell["snr"], cell["ridge_lambda"], cell["column"])
                  for cell in cells]
    if len(set(identities)) != len(identities):
        raise RuntimeError("Duplicate cells")
    used = sorted({int(cell["data_seed"]) for cell in cells})
    if used != sorted(seeds):
        raise RuntimeError("Recorded seeds differ from the phase list")
    audit = {
        "phase": phase,
        "data_seeds": used,
        "confirmation_seeds_absent": phase == "discovery" and not (set(used) & set(protocol["confirmation_data_seeds"])),
        "discovery_seeds_absent": phase == "confirmation" and not (set(used) & set(protocol["discovery_data_seeds"])),
        "eval_seed": evaluation["eval_seed"],
        "eval_noise_seed": evaluation["eval_noise_seed"],
        "eval_channels": evaluation["channels"],
    }
    if phase == "discovery" and not audit["confirmation_seeds_absent"]:
        raise RuntimeError("Confirmation seeds appeared in discovery")
    if phase == "confirmation" and not audit["discovery_seeds_absent"]:
        raise RuntimeError("Discovery seeds appeared in confirmation")
    if phase == "discovery" and pilot is None:
        raise RuntimeError("Timing pilot cell was not executed")
    analysis = evaluate_predictions(cells, protocol, phase)
    wall = time.perf_counter() - started
    return {
        "cells": cells,
        "pilot": pilot,
        "seed_audit": audit,
        "analysis": analysis,
        "runtime": runtime,
        "wall_seconds": wall,
        "n_cells": len(cells),
    }


def _is_pilot(protocol: dict[str, Any], record: dict[str, Any]) -> bool:
    pilot = protocol["timing_pilot_cell"]
    return (
        record["family"] == pilot["family"]
        and record["data_seed"] == pilot["data_seed"]
        and record["column"] == pilot["column"]
        and record["spectrum"] == pilot["spectrum"]
        and record["snr"] == float(pilot["snr"])
        and record["ridge_lambda"] == float(pilot["ridge_lambda"])
    )


def _expected_count(protocol: dict[str, Any], phase: str, columns: list[int]) -> int:
    draws = len(phase_seeds(protocol, phase))
    linear = draws * len(protocol["spectra"]) * len(protocol["snr"]) * len(columns) * len(protocol["lambdas"])
    features = protocol["random_features"]
    random = draws * len(features["spectra"]) * len(features["snr"]) * len(columns) * len(features["lambdas"])
    return linear + random


def _runtime_report(protocol: dict[str, Any]) -> dict[str, Any]:
    pools = []
    for row in threadpool_info():
        pools.append({
            "internal_api": row.get("internal_api"),
            "num_threads": row.get("num_threads"),
            "prefix": row.get("prefix"),
        })
        threads = row.get("num_threads")
        if isinstance(threads, int) and threads > int(protocol["threads"]):
            raise RuntimeError(f"Thread pool exceeds the cap: {row}")
    return {
        "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_imported": "torch" in sys.modules,
        "pools": pools,
        "scientific_validity": "not_assessed",
    }


def _assert_runtime(runtime: dict[str, Any]) -> None:
    if runtime.get("torch_imported") or runtime.get("cuda_visible_devices") not in ("", None):
        raise RuntimeError("GPU/torch runtime is outside the R2 protocol")
    for key, value in runtime["thread_environment"].items():
        if value != "4":
            raise RuntimeError(f"{key} is {value}, expected 4")


def write_phase_outputs(protocol: dict[str, Any], phase: str, result: dict[str, Any], out_dir: Path,
                        export_name: str | None = None) -> None:
    _assert_runtime(result["runtime"])
    cells_path = out_dir / "cells.jsonl"
    with cells_path.open("w", encoding="utf-8", newline="\n") as handle:
        for cell in result["cells"]:
            handle.write(json.dumps(pythonize(cell), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    timing = {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "pilot": result["pilot"],
        "residual_tolerance": protocol["relative_train_residual_tolerance"],
        "rank_target_rule": protocol["rank_target_rule"],
        "wall_seconds": result["wall_seconds"],
    }
    analysis = dict(result["analysis"])
    analysis["n_cells"] = result["n_cells"]
    analysis["cells_sha256"] = sha256_file(cells_path)
    write_json(out_dir / "timing.json", timing)
    write_json(out_dir / "analysis.json", analysis)
    write_json(out_dir / "runtime.json", result["runtime"])
    write_json(out_dir / "seed-audit.json", result["seed_audit"])
    if phase == "discovery":
        reduction = reduce_menu(
            result["wall_seconds"],
            len(protocol["discovery_data_seeds"]),
            len(protocol["confirmation_data_seeds"]),
            protocol,
        )
        write_json(out_dir / "reduction.json", reduction)
        export = {
            "schema_version": 1,
            "phase": "discovery",
            "scientific_validity": "not_assessed",
            "protocol_sha256": sha256_bytes(canonical_protocol(protocol)),
            "timing": {
                "pilot_elapsed_seconds": None if result["pilot"] is None else result["pilot"]["elapsed_seconds"],
                "wall_seconds": result["wall_seconds"],
            },
            "reduction": reduction,
            "seed_audit": result["seed_audit"],
            "n_cells": result["n_cells"],
        }
        write_json(out_dir / (export_name or "discovery-export.json"), export)
    else:
        # Confirmation must not invent a new menu. The caller writes the bound grid echo.
        write_json(out_dir / "reduction.json", {
            "rule": "not_applicable",
            "used_clean_risk": False,
            "note": "Confirmation uses the frozen menu and does not reduce it.",
        })


def build_discovery_export(protocol_path: Path, out_dir: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    result = run_phase(protocol, "discovery", list(protocol["columns"]), out_dir)
    write_phase_outputs(protocol, "discovery", result, out_dir)
    return result


def build_confirmation(protocol_path: Path, grid_path: Path, export_path: Path, out_dir: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    grid = load_json(grid_path)
    export_bytes = export_path.read_bytes()
    export = json.loads(export_bytes.decode("utf-8"))
    expected = freeze_confirmation_grid(export, protocol, hashlib.sha256(export_bytes).hexdigest())
    if grid.get("kept_columns") != expected["kept_columns"] or grid.get("used_clean_risk") is not False:
        raise RuntimeError("Frozen grid does not match the time-only discovery rule")
    if grid.get("discovery_export_sha256") != expected["discovery_export_sha256"]:
        raise RuntimeError("Frozen grid is not bound to the discovery export bytes")
    result = run_phase(protocol, "confirmation", list(grid["kept_columns"]), out_dir)
    write_phase_outputs(protocol, "confirmation", result, out_dir)
    write_json(out_dir / "bound-discovery.json", {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "discovery_export_sha256": expected["discovery_export_sha256"],
        "kept_columns": expected["kept_columns"],
        "used_clean_risk": False,
        "rule": "time_only",
        "seed_audit": result["seed_audit"],
    })
    return result
