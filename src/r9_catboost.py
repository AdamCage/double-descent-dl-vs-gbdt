"""R9 discovery: one CatBoost iteration axis on a frozen synthetic teacher.

Confirmation seeds are reserved and are not fit. Clean risk is recorded and
does not change the iteration grid. Double descent is not declared.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.controlled_synthetic import (
    apply_latent_spectrum,
    column_std,
    latent_standardizer,
    prepare_draw,
    prepare_evaluation,
    sigma_from_snr,
    teacher_signal,
)


def pythonize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): pythonize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [pythonize(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite")
        return float(format(number, ".17g"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(pythonize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("phase") != "discovery":
        raise ValueError("R9 runs discovery only")
    if int(protocol.get("gpu_jobs")) != 0 or int(protocol.get("n_jobs")) != 1 or int(protocol.get("threads")) != 4:
        raise ValueError("R9 uses one CPU job and four threads")
    if float(protocol["relative_train_residual_tolerance"]) != 1e-8:
        raise ValueError("residual tolerance is fixed at 1e-8")
    if protocol.get("double_descent_declared") is not False:
        raise ValueError("R9 does not declare double descent")
    discovery = list(protocol["discovery_data_seeds"])
    confirmation = list(protocol["confirmation_data_seeds"])
    if len(discovery) < 1 or set(discovery) & set(confirmation):
        raise ValueError("discovery and confirmation seeds overlap or discovery is empty")
    iterations = list(protocol["iterations"])
    if iterations != sorted(set(iterations)) or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in iterations):
        raise ValueError("iterations must be strictly increasing positive integers")
    fixed = protocol["fixed"]
    if fixed != {
        "depth": 4,
        "learning_rate": 0.3,
        "l2_leaf_reg": 1.0,
        "loss_function": "RMSE",
        "bootstrap_type": "No",
        "thread_count": 4,
        "logging_level": "Silent",
    }:
        raise ValueError("fixed CatBoost controls changed")
    if int(protocol["input_columns"]) >= int(protocol["p_max"]):
        raise ValueError("the model must see a proper prefix of the teacher features")
    if protocol["spectrum"] != "isotropic" or float(protocol["snr"]) != 1.0:
        raise ValueError("R9 spectrum and SNR are fixed")


def summarize_seed(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Bracket rule fixed before any R9 fit. Ties in clean MSE use the smallest iteration."""
    if not cells:
        raise ValueError("empty seed")
    ordered = sorted(cells, key=lambda cell: int(cell["iterations"]))
    seed = int(ordered[0]["data_seed"])
    if any(int(cell["data_seed"]) != seed for cell in ordered):
        raise ValueError("summarize_seed received several draws")
    first_true = next((cell for cell in ordered if cell["interpolating"]), None)
    has_false_before = first_true is not None and any(
        (not cell["interpolating"]) and int(cell["iterations"]) < int(first_true["iterations"]) for cell in ordered
    )
    status = "bracketed" if has_false_before else "not_identifiable"
    peak = min(ordered, key=lambda cell: (-float(cell["clean_mse"]), int(cell["iterations"])))
    first_iteration = None if first_true is None else int(first_true["iterations"])
    flags = [bool(cell["interpolating"]) for cell in ordered]
    monotone = all(not (flags[index] and not flags[index + 1]) for index in range(len(flags) - 1))
    same = None if status != "bracketed" else first_iteration == int(peak["iterations"])
    return {
        "data_seed": seed,
        "bracket_status": status,
        "first_interpolating_iteration": first_iteration,
        "max_clean_mse_iteration": int(peak["iterations"]),
        "same_iteration": same,
        "interpolation_flag_nonmonotone": not monotone,
    }


def _fit(features: np.ndarray, response: np.ndarray, protocol: dict[str, Any], iterations: int, seed: int) -> tuple[Any, int]:
    from catboost import CatBoostRegressor

    fixed = protocol["fixed"]
    model = CatBoostRegressor(
        iterations=int(iterations),
        depth=int(fixed["depth"]),
        learning_rate=float(fixed["learning_rate"]),
        l2_leaf_reg=float(fixed["l2_leaf_reg"]),
        loss_function=str(fixed["loss_function"]),
        bootstrap_type=str(fixed["bootstrap_type"]),
        random_seed=int(seed),
        thread_count=int(fixed["thread_count"]),
        logging_level=str(fixed["logging_level"]),
        allow_writing_files=False,
    )
    model.fit(features, response)
    tree_count = int(model.tree_count_)
    return model, tree_count


def run_discovery(protocol: dict[str, Any], out_dir: Path, protocol_path: Path) -> dict[str, Any]:
    validate_protocol(protocol)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation = prepare_evaluation(protocol, "discovery")
    input_dim = int(protocol["input_columns"])
    tolerance = float(protocol["relative_train_residual_tolerance"])
    sigma = sigma_from_snr(float(protocol["snr"]))
    cells: list[dict[str, Any]] = []
    for data_seed in protocol["discovery_data_seeds"]:
        if data_seed in set(protocol["confirmation_data_seeds"]):
            raise RuntimeError("confirmation seed was selected")
        draw = prepare_draw(protocol, "discovery", int(data_seed))
        latent = latent_standardizer(draw["covariates"])
        scales = column_std(protocol, "isotropic", int(protocol["p_max"]))
        train = apply_latent_spectrum(draw["covariates"], latent, scales)[:, :input_dim]
        evaluate = apply_latent_spectrum(evaluation["covariates"], latent, scales)[:, :input_dim]
        full_train = apply_latent_spectrum(draw["covariates"], latent, scales)
        full_eval = apply_latent_spectrum(evaluation["covariates"], latent, scales)
        response = teacher_signal(full_train, draw["beta"]) + sigma * draw["label_noise"]
        truth = teacher_signal(full_eval, draw["beta"])
        noisy = truth + sigma * evaluation["label_noise"]
        response_norm = float(np.linalg.norm(response))
        for iterations in protocol["iterations"]:
            model, tree_count = _fit(train, response, protocol, int(iterations), int(data_seed))
            train_prediction = np.asarray(model.predict(train), dtype=np.float64)
            eval_prediction = np.asarray(model.predict(evaluate), dtype=np.float64)
            if not np.isfinite(train_prediction).all() or not np.isfinite(eval_prediction).all():
                raise RuntimeError("non-finite CatBoost prediction")
            residual = float(np.linalg.norm(train_prediction - response) / max(response_norm, 1e-30))
            cells.append({
                "phase": "discovery",
                "model": "catboost",
                "data_seed": int(data_seed),
                "model_seed": int(data_seed),
                "iterations": int(iterations),
                "depth": int(protocol["fixed"]["depth"]),
                "tree_count": tree_count,
                "residual_relative": residual,
                "interpolating": residual <= tolerance,
                "clean_mse": float(np.mean((eval_prediction - truth) ** 2)),
                "noisy_mse": float(np.mean((eval_prediction - noisy) ** 2)),
            })
    expected = len(protocol["discovery_data_seeds"]) * len(protocol["iterations"])
    if len(cells) != expected:
        raise RuntimeError("cell count does not match the frozen grid")
    used = sorted({int(cell["data_seed"]) for cell in cells})
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        by_seed.setdefault(int(cell["data_seed"]), []).append(cell)
    summaries = [summarize_seed(by_seed[seed]) for seed in used]
    analysis = {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "phase": "discovery",
        "iteration_grid": list(protocol["iterations"]),
        "used_clean_risk_to_choose_grid": False,
        "double_descent_declared": False,
        "models_fit": len(cells),
        "download_bytes": 0,
        "http_requests": 0,
        "n_cells": len(cells),
        "peak_vs_threshold": summaries,
        "tolerance": tolerance,
    }
    audit = {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "data_seeds": used,
        "confirmation_seeds": list(protocol["confirmation_data_seeds"]),
        "confirmation_seeds_absent": not (set(used) & set(protocol["confirmation_data_seeds"])),
        "models_fit": len(cells),
    }
    if not audit["confirmation_seeds_absent"]:
        raise RuntimeError("confirmation seed appeared in discovery cells")
    lines = [json.dumps(pythonize(cell), ensure_ascii=False, allow_nan=False) for cell in cells]
    (out_dir / "cells.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / "analysis.json", analysis)
    write_json(out_dir / "seed-audit.json", audit)
    return analysis
