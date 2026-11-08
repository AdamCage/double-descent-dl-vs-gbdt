"""R4 separate tree, forest, and gradient-boosting axes.

Each grid changes one control. CatBoost is not fit. Training uses one job.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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


def canonical(protocol: dict[str, Any]) -> bytes:
    return (json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def validate_protocol(protocol: dict[str, Any]) -> None:
    if int(protocol.get("gpu_jobs")) != 0 or int(protocol.get("n_jobs")) != 1 or int(protocol.get("threads")) != 4:
        raise ValueError("R4 uses one CPU job and four threads")
    if float(protocol["relative_train_residual_tolerance"]) != 1e-8:
        raise ValueError("tree residual tolerance is fixed at 1e-8")
    if set(protocol["discovery_data_seeds"]) & set(protocol["confirmation_data_seeds"]):
        raise ValueError("R4 seeds overlap")
    names = set(protocol["grids"])
    if names != {"tree_leaves", "forest_size_restricted", "forest_size_large", "gbdt_steps", "gbdt_depth", "gbdt_split_fraction"}:
        raise ValueError("R4 grids changed")
    if protocol["grids"]["forest_size_restricted"]["max_leaf_nodes"] == protocol["grids"]["forest_size_large"]["max_leaf_nodes"]:
        raise ValueError("forest slices must keep different leaf caps")


def jobs_for(protocol: dict[str, Any], reduced: bool = False) -> list[dict[str, Any]]:
    jobs = []
    for name, spec in protocol["grids"].items():
        varied = next(key for key, value in spec.items() if isinstance(value, list))
        values = list(spec[varied])
        if reduced:
            middle = values[len(values) // 2]
            values = list(dict.fromkeys([values[0], middle, values[-1]]))
        for value in values:
            setting = {key: item for key, item in spec.items() if key != "model"}
            setting[varied] = value
            jobs.append({"grid": name, "model": spec["model"], "varied": varied, "value": value, "setting": setting})
    return jobs


def reduce_menu(wall_seconds: float, n_discovery: int, n_confirmation: int, protocol: dict[str, Any]) -> dict[str, Any]:
    rule = protocol["reduction_rule"]
    projected = (wall_seconds / n_discovery) * n_confirmation * float(rule["safety_factor"])
    keep_full = projected < float(rule["keep_full_menu_if_projected_confirmation_seconds_below"])
    return {
        "rule": "time_only",
        "used_clean_risk": False,
        "projected_confirmation_seconds": projected,
        "kept_full_menu": keep_full,
        "discovery_wall_seconds": wall_seconds,
    }


def _fit(job: dict[str, Any], features: np.ndarray, response: np.ndarray, seed: int):
    setting = job["setting"]
    if job["model"] == "decision_tree":
        model = DecisionTreeRegressor(max_leaf_nodes=int(setting["max_leaf_nodes"]), random_state=seed)
    elif job["model"] == "random_forest":
        model = RandomForestRegressor(
            n_estimators=int(setting["n_estimators"]),
            max_leaf_nodes=int(setting["max_leaf_nodes"]),
            n_jobs=1,
            random_state=seed,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=int(setting["n_estimators"]),
            max_depth=int(setting["max_depth"]),
            max_features=float(setting["max_features"]),
            learning_rate=0.1,
            random_state=seed,
        )
    model.fit(features, response)
    if job["model"] == "decision_tree":
        leaves = int(model.get_n_leaves())
    elif job["model"] == "random_forest":
        leaves = int(sum(tree.get_n_leaves() for tree in model.estimators_))
    else:
        leaves = int(sum(stage.tree_.n_leaves for stage in model.estimators_.ravel()))
    return model, leaves


def _predict(model, features: np.ndarray) -> np.ndarray:
    prediction = np.asarray(model.predict(features), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite tree prediction")
    return prediction


def run_phase(protocol: dict[str, Any], phase: str, keep_full: bool, out_dir: Path) -> dict[str, Any]:
    import time
    validate_protocol(protocol)
    started = time.perf_counter()
    seeds = protocol["discovery_data_seeds"] if phase == "discovery" else protocol["confirmation_data_seeds"]
    other = set(protocol["confirmation_data_seeds"] if phase == "discovery" else protocol["discovery_data_seeds"])
    evaluation = prepare_evaluation(protocol, phase)
    input_dim = int(protocol["input_columns"])
    tolerance = float(protocol["relative_train_residual_tolerance"])
    sigma = sigma_from_snr(float(protocol["snr"]))
    cells = []
    pilot = None
    jobs = jobs_for(protocol, reduced=not keep_full)
    for data_seed in seeds:
        if data_seed in other:
            raise RuntimeError("phase seeds overlap")
        draw = prepare_draw(protocol, phase, data_seed)
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
        for job in jobs:
            block = time.perf_counter()
            model, leaves = _fit(job, train, response, int(data_seed))
            train_prediction = _predict(model, train)
            eval_prediction = _predict(model, evaluate)
            residual = float(np.linalg.norm(train_prediction - response) / max(response_norm, 1e-30))
            record = {
                "phase": phase,
                "grid": job["grid"],
                "model": job["model"],
                "varied": job["varied"],
                "value": job["value"],
                "data_seed": int(data_seed),
                "n_leaves": leaves,
                "residual_relative": residual,
                "interpolating": residual <= tolerance,
                "clean_mse": float(np.mean((eval_prediction - truth) ** 2)),
                "noisy_mse": float(np.mean((eval_prediction - noisy) ** 2)),
                "elapsed_seconds": time.perf_counter() - block,
                "setting": job["setting"],
            }
            cells.append(record)
            pilot_spec = protocol["timing_pilot_cell"]
            if (data_seed == pilot_spec["data_seed"] and job["grid"] == pilot_spec["grid"]
                    and job["varied"] == "max_leaf_nodes" and int(job["value"]) == int(pilot_spec["max_leaf_nodes"])):
                pilot = {"elapsed_seconds": record["elapsed_seconds"], "grid": job["grid"], "value": job["value"],
                         "residual_relative": residual, "n_leaves": leaves}
    if phase == "discovery" and pilot is None:
        raise RuntimeError("timing pilot missing")
    used = sorted({cell["data_seed"] for cell in cells})
    audit = {
        "phase": phase,
        "data_seeds": used,
        "confirmation_seeds_absent": phase == "discovery" and not (set(used) & set(protocol["confirmation_data_seeds"])),
        "discovery_seeds_absent": phase == "confirmation" and not (set(used) & set(protocol["discovery_data_seeds"])),
    }
    if phase == "discovery" and not audit["confirmation_seeds_absent"]:
        raise RuntimeError("confirmation seeds appeared")
    if phase == "confirmation" and not audit["discovery_seeds_absent"]:
        raise RuntimeError("discovery seeds appeared")
    grids = sorted({cell["grid"] for cell in cells})
    return {
        "cells": cells,
        "pilot": pilot,
        "seed_audit": audit,
        "analysis": {
            "phase": phase,
            "label": "exploratory_discovery" if phase == "discovery" else "confirmation",
            "grids": grids,
            "n_cells": len(cells),
            "scientific_validity": "not_assessed",
            "note": "Axes stay in separate grids. No single capacity curve is formed.",
        },
        "runtime": {"n_jobs": 1, "gpu_jobs": 0, "torch_imported": "torch" in __import__("sys").modules, "scientific_validity": "not_assessed"},
        "wall_seconds": time.perf_counter() - started,
        "keep_full": keep_full,
    }


def write_outputs(protocol: dict[str, Any], phase: str, result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cells.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for cell in result["cells"]:
            handle.write(json.dumps(pythonize(cell), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    analysis = dict(result["analysis"])
    analysis["cells_sha256"] = sha256_file(path)
    write_json(out_dir / "analysis.json", analysis)
    write_json(out_dir / "runtime.json", result["runtime"])
    write_json(out_dir / "seed-audit.json", result["seed_audit"])
    write_json(out_dir / "timing.json", {
        "schema_version": 1, "scientific_validity": "not_assessed", "pilot": result["pilot"],
        "wall_seconds": result["wall_seconds"], "residual_tolerance": protocol["relative_train_residual_tolerance"],
    })
    if phase == "discovery":
        reduction = reduce_menu(result["wall_seconds"], len(protocol["discovery_data_seeds"]),
                                len(protocol["confirmation_data_seeds"]), protocol)
        write_json(out_dir / "reduction.json", reduction)
        write_json(out_dir / "discovery-export.json", {
            "schema_version": 1, "phase": "discovery", "scientific_validity": "not_assessed",
            "protocol_sha256": hashlib.sha256(canonical(protocol)).hexdigest(),
            "reduction": reduction, "seed_audit": result["seed_audit"], "n_cells": len(result["cells"]),
        })
    else:
        write_json(out_dir / "reduction.json", {"rule": "not_applicable", "used_clean_risk": False})


def build_discovery(protocol_path: Path, out_dir: Path) -> None:
    protocol = load_json(protocol_path)
    result = run_phase(protocol, "discovery", True, out_dir)
    write_outputs(protocol, "discovery", result, out_dir)


def freeze_grid(export: dict[str, Any], protocol: dict[str, Any], export_sha256: str) -> dict[str, Any]:
    if "clean_mse" in export:
        raise RuntimeError("refusing risk fields")
    reduction = export["reduction"]
    if reduction.get("used_clean_risk") is not False:
        raise RuntimeError("reduction is not time-only")
    if hashlib.sha256(canonical(protocol)).hexdigest() != export.get("protocol_sha256"):
        raise RuntimeError("protocol hash mismatch")
    return {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "kept_full_menu": bool(reduction["kept_full_menu"]),
        "used_clean_risk": False,
        "rule": "time_only",
        "discovery_export_sha256": export_sha256,
        "projected_confirmation_seconds": reduction["projected_confirmation_seconds"],
    }


def build_confirmation(protocol_path: Path, grid_path: Path, export_path: Path, out_dir: Path) -> None:
    protocol = load_json(protocol_path)
    grid = load_json(grid_path)
    export_bytes = export_path.read_bytes()
    expected = freeze_grid(json.loads(export_bytes.decode("utf-8")), protocol, hashlib.sha256(export_bytes).hexdigest())
    if grid.get("kept_full_menu") != expected["kept_full_menu"] or grid.get("discovery_export_sha256") != expected["discovery_export_sha256"]:
        raise RuntimeError("frozen grid mismatch")
    result = run_phase(protocol, "confirmation", bool(grid["kept_full_menu"]), out_dir)
    write_outputs(protocol, "confirmation", result, out_dir)
    write_json(out_dir / "bound-discovery.json", expected | {"seed_audit": result["seed_audit"]})
