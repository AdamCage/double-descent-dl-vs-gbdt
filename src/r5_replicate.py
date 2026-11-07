"""R5 replication of two closed contrasts on new seeds.

The MLP budget result and the boosting-depth curve are not re-fit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

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


def pythonize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): pythonize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [pythonize(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite")
        return float(format(number, ".17g"))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(pythonize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def canonical(protocol: dict[str, Any]) -> bytes:
    return (json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def validate_protocol(protocol: dict[str, Any]) -> None:
    if set(protocol["discovery_data_seeds"]) & set(protocol["confirmation_data_seeds"]):
        raise ValueError("R5 seeds overlap")
    if protocol["linear_columns"] != [150, 200, 250]:
        raise ValueError("linear replication window changed")
    if protocol["tree_leaf_values"] != [2, 256] or protocol["forest_sizes"] != [1, 50]:
        raise ValueError("tree replication endpoints changed")
    if int(protocol["gpu_jobs"]) != 0 or int(protocol["n_jobs"]) != 1:
        raise ValueError("R5 stays on one CPU job")
    if "R3-confirmation-budget" not in protocol["not_reestimated"][0] and "40 epochs" not in " ".join(protocol["not_reestimated"]):
        raise ValueError("the unreplicated MLP result must stay explicit")


def _mean_se(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    se = None if array.size < 2 else float(array.std(ddof=1) / math.sqrt(array.size))
    return {"mean": float(array.mean()), "se": se, "minimum": float(array.min()), "maximum": float(array.max()), "n": int(array.size)}


def _directional(values: list[float], margin: float, minimum: int) -> str:
    if len(values) < minimum:
        return "inconclusive_few_brackets" if values else "inconclusive_no_bracket"
    stats = _mean_se(values)
    mean, se = float(stats["mean"]), stats["se"]
    if mean <= 0:
        return "refuted"
    if mean < margin or se is None or mean <= float(se):
        return "inconclusive_within_noise"
    return "supported"


def evaluate(cells: list[dict[str, Any]], protocol: dict[str, Any], phase: str) -> dict[str, Any]:
    rows = [cell for cell in cells if cell["phase"] == phase]
    margin = float(protocol["contrast_margins"]["clean_risk_absolute"])
    minimum = int(protocol["minimum_bracketed_draws"])
    seeds = sorted({cell["data_seed"] for cell in rows})
    noise_deltas, leaf_residual, leaf_clean, forest_clean = [], [], [], []
    bracketed = []
    for seed in seeds:
        selected = [cell for cell in rows if cell["data_seed"] == seed and cell["family"] == "linear"]
        low = [cell for cell in selected if cell["column"] == 150 and cell["snr"] == 1.0]
        high = [cell for cell in selected if cell["column"] == 250 and cell["snr"] == 1.0]
        if low and high and (not low[0]["interpolating"]) and high[0]["interpolating"]:
            bracketed.append(seed)
            def window(snr: float) -> float:
                vals = [cell["clean_mse"] for cell in selected if cell["snr"] == snr]
                return float(np.mean(vals))
            noise_deltas.append(window(1.0) - window(5.0))
        leaves = {cell["value"]: cell for cell in rows if cell["data_seed"] == seed and cell["family"] == "tree_leaves"}
        if 2 in leaves and 256 in leaves:
            leaf_residual.append(leaves[2]["residual_relative"] - leaves[256]["residual_relative"])
            leaf_clean.append(leaves[256]["clean_mse"] - leaves[2]["clean_mse"])
        forest = {cell["value"]: cell for cell in rows if cell["data_seed"] == seed and cell["family"] == "forest_size_restricted"}
        if 1 in forest and 50 in forest:
            forest_clean.append(forest[1]["clean_mse"] - forest[50]["clean_mse"])
    return {
        "phase": phase,
        "label": "exploratory_discovery" if phase == "discovery" else "confirmation",
        "noise": {"outcome": _directional(noise_deltas, margin, minimum), "n_bracketed": len(bracketed),
                  "clean_risk_delta": _mean_se(noise_deltas) if noise_deltas else None},
        "leaf_residual": {"outcome": _directional(leaf_residual, margin, minimum),
                          "delta": _mean_se(leaf_residual) if leaf_residual else None,
                          "note": "Positive means the 2-leaf residual exceeds the 256-leaf residual."},
        "leaf_clean_risk": {"outcome": _directional(leaf_clean, margin, minimum),
                            "delta": _mean_se(leaf_clean) if leaf_clean else None,
                            "note": "Positive means clean risk is higher at 256 leaves than at 2."},
        "forest_clean_risk": {"outcome": _directional(forest_clean, margin, minimum),
                              "delta": _mean_se(forest_clean) if forest_clean else None,
                              "note": "Positive means one restricted tree has higher clean risk than 50 trees."},
        "nonidentifiability": protocol["nonidentifiability"],
        "not_reestimated": protocol["not_reestimated"],
        "scientific_validity": "not_assessed",
    }


def run_phase(protocol: dict[str, Any], phase: str) -> dict[str, Any]:
    validate_protocol(protocol)
    started = time.perf_counter()
    seeds = protocol["discovery_data_seeds"] if phase == "discovery" else protocol["confirmation_data_seeds"]
    other = set(protocol["confirmation_data_seeds"] if phase == "discovery" else protocol["discovery_data_seeds"])
    evaluation = prepare_evaluation(protocol, phase)
    cells = []
    for data_seed in seeds:
        if data_seed in other:
            raise RuntimeError("phase seeds overlap")
        draw = prepare_draw(protocol, phase, data_seed)
        latent = latent_standardizer(draw["covariates"])
        scales = column_std(protocol, "isotropic", int(protocol["p_max"]))
        train_full = apply_latent_spectrum(draw["covariates"], latent, scales)
        eval_full = apply_latent_spectrum(evaluation["covariates"], latent, scales)
        truth = teacher_signal(eval_full, draw["beta"])
        captured = float(np.sum(draw["beta"][: int(protocol["input_columns"])] ** 2))
        for snr in protocol["snr"]:
            sigma = sigma_from_snr(float(snr))
            response = teacher_signal(train_full, draw["beta"]) + sigma * draw["label_noise"]
            noisy = truth + sigma * evaluation["label_noise"]
            for column in protocol["linear_columns"]:
                fitted = fit_ridge_path(train_full[:, :column], response, [0.0], 1e-10, rescale_columns=False)
                solution = fitted["solutions"][0]
                prediction = predict(eval_full[:, :column], fitted, solution)
                cells.append({
                    "phase": phase, "family": "linear", "data_seed": int(data_seed), "snr": float(snr),
                    "column": int(column), "value": int(column),
                    "residual_relative": solution["residual_relative"],
                    "interpolating": interpolation_flag(solution, fitted, float(protocol["linear_tolerance"])),
                    "clean_mse": float(np.mean((prediction - truth) ** 2)),
                    "noisy_mse": float(np.mean((prediction - noisy) ** 2)),
                    "teacher_energy_captured": captured,
                })
        sigma = sigma_from_snr(1.0)
        tree_x = train_full[:, : int(protocol["input_columns"])]
        tree_eval = eval_full[:, : int(protocol["input_columns"])]
        tree_y = teacher_signal(train_full, draw["beta"]) + sigma * draw["label_noise"]
        tree_truth = truth
        tree_noisy = truth + sigma * evaluation["label_noise"]
        response_norm = float(np.linalg.norm(tree_y))
        for leaves in protocol["tree_leaf_values"]:
            model = DecisionTreeRegressor(max_leaf_nodes=int(leaves), random_state=int(data_seed))
            model.fit(tree_x, tree_y)
            train_hat = model.predict(tree_x)
            pred = model.predict(tree_eval)
            residual = float(np.linalg.norm(train_hat - tree_y) / max(response_norm, 1e-30))
            cells.append({
                "phase": phase, "family": "tree_leaves", "data_seed": int(data_seed), "snr": 1.0,
                "value": int(leaves), "n_leaves": int(model.get_n_leaves()),
                "residual_relative": residual, "interpolating": residual <= float(protocol["tree_tolerance"]),
                "clean_mse": float(np.mean((pred - tree_truth) ** 2)),
                "noisy_mse": float(np.mean((pred - tree_noisy) ** 2)),
            })
        for size in protocol["forest_sizes"]:
            model = RandomForestRegressor(n_estimators=int(size), max_leaf_nodes=int(protocol["forest_leaf_cap"]),
                                          n_jobs=1, random_state=int(data_seed))
            model.fit(tree_x, tree_y)
            train_hat = model.predict(tree_x)
            pred = model.predict(tree_eval)
            residual = float(np.linalg.norm(train_hat - tree_y) / max(response_norm, 1e-30))
            cells.append({
                "phase": phase, "family": "forest_size_restricted", "data_seed": int(data_seed), "snr": 1.0,
                "value": int(size), "residual_relative": residual,
                "interpolating": residual <= float(protocol["tree_tolerance"]),
                "clean_mse": float(np.mean((pred - tree_truth) ** 2)),
                "noisy_mse": float(np.mean((pred - tree_noisy) ** 2)),
            })
    used = sorted({cell["data_seed"] for cell in cells})
    audit = {
        "phase": phase, "data_seeds": used,
        "confirmation_seeds_absent": phase == "discovery" and not (set(used) & set(protocol["confirmation_data_seeds"])),
        "discovery_seeds_absent": phase == "confirmation" and not (set(used) & set(protocol["discovery_data_seeds"])),
    }
    if phase == "discovery" and not audit["confirmation_seeds_absent"]:
        raise RuntimeError("confirmation seeds appeared")
    if phase == "confirmation" and not audit["discovery_seeds_absent"]:
        raise RuntimeError("discovery seeds appeared")
    wall = time.perf_counter() - started
    return {"cells": cells, "analysis": evaluate(cells, protocol, phase), "seed_audit": audit, "wall_seconds": wall,
            "runtime": {"n_jobs": 1, "gpu_jobs": 0, "torch_imported": False, "scientific_validity": "not_assessed"}}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_outputs(protocol: dict[str, Any], phase: str, result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cells.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for cell in result["cells"]:
            handle.write(json.dumps(pythonize(cell), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    analysis = dict(result["analysis"])
    analysis["n_cells"] = len(result["cells"])
    analysis["cells_sha256"] = sha256_file(path)
    write_json(out_dir / "analysis.json", analysis)
    write_json(out_dir / "seed-audit.json", result["seed_audit"])
    write_json(out_dir / "runtime.json", result["runtime"])
    write_json(out_dir / "timing.json", {"schema_version": 1, "scientific_validity": "not_assessed", "wall_seconds": result["wall_seconds"]})
    projected = (result["wall_seconds"] / len(protocol["discovery_data_seeds"])) * len(protocol["confirmation_data_seeds"]) * 1.5
    reduction = {
        "rule": "time_only", "used_clean_risk": False, "kept_full_menu": projected < 1800,
        "projected_confirmation_seconds": projected, "discovery_wall_seconds": result["wall_seconds"],
        "blocked_if_over_cap": projected >= 1800,
    }
    if phase == "discovery":
        write_json(out_dir / "reduction.json", reduction)
        write_json(out_dir / "discovery-export.json", {
            "schema_version": 1, "scientific_validity": "not_assessed",
            "protocol_sha256": hashlib.sha256(canonical(protocol)).hexdigest(),
            "reduction": reduction, "seed_audit": result["seed_audit"], "n_cells": len(result["cells"]),
        })
    else:
        write_json(out_dir / "reduction.json", {"rule": "not_applicable", "used_clean_risk": False})


def build_discovery(protocol_path: Path, out_dir: Path) -> None:
    protocol = load_json(protocol_path)
    result = run_phase(protocol, "discovery")
    write_outputs(protocol, "discovery", result, out_dir)


def build_confirmation(protocol_path: Path, grid_path: Path, export_path: Path, out_dir: Path) -> None:
    protocol = load_json(protocol_path)
    grid = load_json(grid_path)
    export = load_json(export_path)
    if export["reduction"].get("kept_full_menu") is not True or grid.get("kept_full_menu") is not True:
        raise RuntimeError("replication grid was not kept in full")
    if grid.get("discovery_export_sha256") != sha256_file(export_path):
        raise RuntimeError("export hash mismatch")
    if hashlib.sha256(canonical(protocol)).hexdigest() != export.get("protocol_sha256"):
        raise RuntimeError("protocol hash mismatch")
    result = run_phase(protocol, "confirmation")
    write_outputs(protocol, "confirmation", result, out_dir)
    write_json(out_dir / "bound-discovery.json", {
        "schema_version": 1, "scientific_validity": "not_assessed",
        "discovery_export_sha256": grid["discovery_export_sha256"],
        "kept_full_menu": True, "used_clean_risk": False,
        "nonidentifiability": protocol["nonidentifiability"],
        "seed_audit": result["seed_audit"],
    })
