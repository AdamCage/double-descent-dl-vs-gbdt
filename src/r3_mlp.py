"""R3 MLP width, budget, and noise contrasts on the isotropic latent design.

The network is trained on CPU. A GPU being visible is recorded and not used.
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

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

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

THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_threads() -> None:
    import torch

    for key in THREAD_KEYS:
        os.environ[key] = "4"
    torch.set_num_threads(4)


def pythonize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): pythonize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [pythonize(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return float(format(value, ".17g"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
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


def canonical_protocol(protocol: dict[str, Any]) -> bytes:
    return (json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("device") != "cpu" or int(protocol.get("gpu_jobs")) != 0 or int(protocol.get("threads")) != 4:
        raise ValueError("R3 stays on 4 CPU threads and zero GPU jobs")
    if protocol.get("spectrum") != "isotropic":
        raise ValueError("R3 fixes the isotropic spectrum")
    widths = protocol.get("widths")
    fallback = protocol["reduction_rule"]["fallback_widths"]
    if not isinstance(widths, list) or widths != sorted(set(widths)):
        raise ValueError("widths must be a strictly increasing list")
    if not set(fallback).issubset(widths) or len(fallback) < 2:
        raise ValueError("fallback widths must be a subset with both sides available")
    discovery = protocol["discovery_data_seeds"]
    confirmation = protocol["confirmation_data_seeds"]
    if set(discovery) & set(confirmation):
        raise ValueError("R3 seeds overlap")
    if protocol["timing_pilot_cell"]["data_seed"] not in discovery:
        raise ValueError("timing pilot seed is not a discovery seed")
    if float(protocol["relative_train_residual_tolerance"]) != 0.001:
        raise ValueError("MLP residual tolerance is fixed at 0.001")
    if int(protocol["fixed_epochs"]) != 40 or int(protocol["max_epochs"]) != 500:
        raise ValueError("epoch budgets are fixed")
    if protocol["procedures"] != ["epochs_40", "until_residual"]:
        raise ValueError("procedures changed")


def phase_seeds(protocol: dict[str, Any], phase: str) -> list[int]:
    key = "discovery_data_seeds" if phase == "discovery" else "confirmation_data_seeds"
    if phase not in ("discovery", "confirmation"):
        raise ValueError(phase)
    return list(protocol[key])


def reduce_menu(discovery_wall_seconds: float, n_discovery: int, n_confirmation: int,
                protocol: dict[str, Any]) -> dict[str, Any]:
    rule = protocol["reduction_rule"]
    per_draw = discovery_wall_seconds / n_discovery
    projected = per_draw * n_confirmation * float(rule["safety_factor"])
    keep_full = projected < float(rule["keep_full_menu_if_projected_confirmation_seconds_below"])
    return {
        "rule": "time_only",
        "used_clean_risk": False,
        "discovery_wall_seconds": discovery_wall_seconds,
        "projected_confirmation_seconds": projected,
        "kept_full_menu": keep_full,
        "kept_widths": list(protocol["widths"] if keep_full else rule["fallback_widths"]),
    }


def freeze_confirmation_grid(export: dict[str, Any], protocol: dict[str, Any], export_sha256: str) -> dict[str, Any]:
    if "clean_mse" in export or "cells" in export:
        raise RuntimeError("refusing a grid frozen from risk fields")
    reduction = export["reduction"]
    if reduction.get("used_clean_risk") is not False or reduction.get("rule") != "time_only":
        raise RuntimeError("missing time-only reduction")
    kept = reduction["kept_widths"]
    if kept not in (list(protocol["widths"]), list(protocol["reduction_rule"]["fallback_widths"])):
        raise RuntimeError("kept widths are not predeclared")
    digest = hashlib.sha256(canonical_protocol(protocol)).hexdigest()
    if export.get("protocol_sha256") != digest:
        raise RuntimeError("protocol hash mismatch")
    if export["seed_audit"].get("confirmation_seeds_absent") is not True:
        raise RuntimeError("discovery export used confirmation seeds")
    return {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "kept_widths": kept,
        "used_clean_risk": False,
        "rule": "time_only",
        "discovery_export_sha256": export_sha256,
        "projected_confirmation_seconds": reduction["projected_confirmation_seconds"],
    }


def _init_network(input_dim: int, width: int, generator: torch.Generator) -> torch.nn.Module:
    import torch

    hidden = torch.nn.Linear(input_dim, width, dtype=torch.float64)
    output = torch.nn.Linear(width, 1, dtype=torch.float64)
    with torch.no_grad():
        hidden.weight.copy_(torch.randn(hidden.weight.shape, generator=generator, dtype=torch.float64) * math.sqrt(2.0 / input_dim))
        output.weight.copy_(torch.randn(output.weight.shape, generator=generator, dtype=torch.float64) * math.sqrt(1.0 / width))
        hidden.bias.zero_()
        output.bias.zero_()
    network = torch.nn.Sequential(hidden, torch.nn.ReLU(), output)
    if any(parameter.device.type != "cpu" for parameter in network.parameters()):
        raise RuntimeError("MLP parameters left the CPU")
    return network


def _parameter_l2(network: torch.nn.Module) -> float:
    import torch

    total = 0.0
    for parameter in network.parameters():
        total += float(torch.sum(parameter.detach() ** 2))
    return math.sqrt(total)


def _grad_norm(network: torch.nn.Module) -> float:
    import torch

    total = 0.0
    for parameter in network.parameters():
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2))
    return math.sqrt(total)


def train_mlp(features: np.ndarray, response: np.ndarray, width: int, procedure: str,
              weight_decay: float, protocol: dict[str, Any], model_seed: int) -> dict[str, Any]:
    import torch
    limit_threads()
    tolerance = float(protocol["relative_train_residual_tolerance"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(model_seed) % (2**63 - 1))
    network = _init_network(features.shape[1], width, generator)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01, weight_decay=float(weight_decay))
    inputs = torch.as_tensor(features, dtype=torch.float64)
    target = torch.as_tensor(response, dtype=torch.float64).reshape(-1, 1)
    target_norm = float(torch.linalg.vector_norm(target))
    max_epochs = int(protocol["fixed_epochs"] if procedure == "epochs_40" else protocol["max_epochs"])
    stop_reason = "budget"
    residual_relative = None
    train_mse = None
    grad_norm = None
    epoch_ran = 0
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = network(inputs)
        residual = prediction - target
        loss = torch.mean(residual ** 2)
        loss.backward()
        grad_norm = _grad_norm(network)
        optimizer.step()
        epoch_ran = epoch
        with torch.no_grad():
            fitted = network(inputs)
            residual_norm = float(torch.linalg.vector_norm(fitted - target))
        residual_relative = residual_norm / max(target_norm, 1e-30)
        train_mse = float(loss.detach())
        if procedure == "until_residual" and residual_relative <= tolerance:
            stop_reason = "residual_tolerance"
            break
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    return {
        "network": network,
        "epochs_ran": epoch_ran,
        "stop_reason": stop_reason,
        "train_mse": train_mse,
        "residual_relative": residual_relative,
        "grad_norm": grad_norm,
        "parameter_l2": _parameter_l2(network),
        "parameter_count": int(parameter_count),
        "interpolating": bool(residual_relative is not None and residual_relative <= tolerance),
    }


def predict_mlp(network: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        prediction = network(torch.as_tensor(features, dtype=torch.float64))
    array = prediction.reshape(-1).cpu().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError("non-finite MLP prediction")
    return array


def _mean_se(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    se = None if array.size < 2 else float(array.std(ddof=1) / math.sqrt(array.size))
    return {"mean": float(array.mean()), "se": se, "minimum": float(array.min()), "maximum": float(array.max()), "n": int(array.size)}


def _directional(values: list[float], margin: float) -> str:
    stats = _mean_se(values)
    mean = float(stats["mean"])
    se = stats["se"]
    if mean <= 0:
        return "nonpositive"
    if mean < margin or se is None or mean <= float(se):
        return "below"
    return "pass"


def evaluate_predictions(cells: list[dict[str, Any]], protocol: dict[str, Any], phase: str) -> dict[str, Any]:
    rows = [cell for cell in cells if cell["phase"] == phase]
    margin = float(protocol["contrast_margins"]["clean_risk_absolute"])
    minimum = int(protocol["minimum_bracketed_draws"])
    widths = sorted({int(cell["width"]) for cell in rows if cell["weight_decay"] == 0.0 and cell["procedure"] == "until_residual"})

    def bracketed(snr: float) -> set[int]:
        seeds = {int(cell["data_seed"]) for cell in rows if cell["procedure"] == "until_residual" and cell["snr"] == snr and cell["weight_decay"] == 0.0}
        kept = set()
        for seed in seeds:
            selected = [cell for cell in rows if cell["data_seed"] == seed and cell["procedure"] == "until_residual"
                        and cell["snr"] == snr and cell["weight_decay"] == 0.0]
            if any(not cell["interpolating"] and cell["width"] == min(widths) for cell in selected) and any(
                    cell["interpolating"] and cell["width"] == max(widths) for cell in selected):
                kept.add(seed)
        return kept

    def blocked(seeds: set[int]) -> str | None:
        if not seeds:
            return "inconclusive_no_bracket"
        if len(seeds) < minimum:
            return "inconclusive_few_brackets"
        return None

    def noise() -> dict[str, Any]:
        seeds = bracketed(1.0) & bracketed(5.0)
        reason = blocked(seeds)
        if reason:
            return {"outcome": reason, "n_bracketed": len(seeds)}
        deltas = []
        for seed in sorted(seeds):
            def average(snr: float) -> float:
                selected = [cell["clean_mse"] for cell in rows if cell["data_seed"] == seed and cell["snr"] == snr
                            and cell["procedure"] == "until_residual" and cell["weight_decay"] == 0.0]
                return float(np.mean(selected))
            deltas.append(average(1.0) - average(5.0))
        return {"outcome": {"pass": "supported", "nonpositive": "refuted"}.get(_directional(deltas, margin), "inconclusive_within_noise"),
                "n_bracketed": len(seeds), "clean_risk_delta": _mean_se(deltas)}

    def budget() -> dict[str, Any]:
        seeds = bracketed(1.0)
        reason = blocked(seeds)
        if reason:
            return {"outcome": reason, "n_bracketed": len(seeds)}
        deltas = []
        for seed in sorted(seeds):
            short = [cell["clean_mse"] for cell in rows if cell["data_seed"] == seed and cell["snr"] == 1.0
                     and cell["procedure"] == "epochs_40" and cell["weight_decay"] == 0.0]
            long = [cell["clean_mse"] for cell in rows if cell["data_seed"] == seed and cell["snr"] == 1.0
                    and cell["procedure"] == "until_residual" and cell["weight_decay"] == 0.0]
            deltas.append(float(np.mean(short) - np.mean(long)))
        status = _directional(deltas, margin)
        outcome = {"pass": "supported", "nonpositive": "refuted"}.get(status, "inconclusive_within_noise")
        return {"outcome": outcome, "n_bracketed": len(seeds), "clean_risk_delta": _mean_se(deltas),
                "note": "Positive means the 40-epoch budget has higher clean risk than until_residual."}

    def decay() -> dict[str, Any]:
        seeds = bracketed(1.0)
        reason = blocked(seeds)
        if reason:
            return {"outcome": reason, "n_bracketed": len(seeds)}
        deltas = []
        for seed in sorted(seeds):
            base = [cell for cell in rows if cell["data_seed"] == seed and cell["width"] == 64 and cell["snr"] == 1.0
                    and cell["procedure"] == "until_residual" and cell["weight_decay"] == 0.0]
            penalized = [cell for cell in rows if cell["data_seed"] == seed and cell["width"] == 64 and cell["snr"] == 1.0
                         and cell["procedure"] == "until_residual" and cell["weight_decay"] == 0.001]
            if len(base) != 1 or len(penalized) != 1:
                raise RuntimeError("weight-decay slice is incomplete")
            deltas.append(float(base[0]["clean_mse"] - penalized[0]["clean_mse"]))
        status = _directional(deltas, margin)
        return {"outcome": {"pass": "supported", "nonpositive": "refuted"}.get(status, "inconclusive_within_noise"),
                "n_bracketed": len(seeds), "clean_risk_delta": _mean_se(deltas)}

    return {
        "phase": phase,
        "label": "exploratory_discovery" if phase == "discovery" else "confirmation",
        "noise": noise(),
        "budget": budget(),
        "weight_decay": decay(),
        "scientific_validity": "not_assessed",
    }


def run_phase(protocol: dict[str, Any], phase: str, widths: list[int], out_dir: Path) -> dict[str, Any]:
    import torch

    validate_protocol(protocol)
    limit_threads()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized")
    started = time.perf_counter()
    seeds = phase_seeds(protocol, phase)
    other = set(phase_seeds(protocol, "confirmation" if phase == "discovery" else "discovery"))
    evaluation = prepare_evaluation(protocol, phase)
    cells: list[dict[str, Any]] = []
    pilot = None
    input_dim = int(protocol["input_columns"])
    for data_seed in seeds:
        if data_seed in other:
            raise RuntimeError("phase seeds overlap")
        draw = prepare_draw(protocol, phase, data_seed)
        latent = latent_standardizer(draw["covariates"])
        scales = column_std(protocol, "isotropic", int(protocol["p_max"]))
        train = apply_latent_spectrum(draw["covariates"], latent, scales)
        evaluate = apply_latent_spectrum(evaluation["covariates"], latent, scales)
        truth = teacher_signal(evaluate, draw["beta"])
        captured = float(np.sum(draw["beta"][:input_dim] ** 2))
        model_seed = int(np.random.SeedSequence([int(protocol["protocol_salt"]), data_seed, 4]).generate_state(1)[0])
        for snr in protocol["snr"]:
            sigma = sigma_from_snr(float(snr))
            response = teacher_signal(train, draw["beta"]) + sigma * draw["label_noise"]
            noisy = truth + sigma * evaluation["label_noise"]
            jobs = [(width, procedure, float(protocol["main_weight_decay"]))
                    for width in widths for procedure in protocol["procedures"]]
            slice_spec = protocol["weight_decay_slice"]
            for width in slice_spec["widths"]:
                if width in widths and float(snr) in slice_spec["snr"]:
                    jobs.append((width, slice_spec["procedure"], float(slice_spec["weight_decay"])))
            for width, procedure, decay in jobs:
                block = time.perf_counter()
                trained = train_mlp(train[:, :input_dim], response, int(width), procedure, decay, protocol, model_seed)
                prediction = predict_mlp(trained["network"], evaluate[:, :input_dim])
                record = {
                    "phase": phase,
                    "data_seed": int(data_seed),
                    "model_seed": model_seed,
                    "width": int(width),
                    "snr": float(snr),
                    "sigma": sigma,
                    "procedure": procedure,
                    "weight_decay": decay,
                    "input_columns": input_dim,
                    "parameter_count": trained["parameter_count"],
                    "epochs_ran": trained["epochs_ran"],
                    "stop_reason": trained["stop_reason"],
                    "train_mse": trained["train_mse"],
                    "residual_relative": trained["residual_relative"],
                    "grad_norm": trained["grad_norm"],
                    "parameter_l2": trained["parameter_l2"],
                    "clean_mse": float(np.mean((prediction - truth) ** 2)),
                    "noisy_mse": float(np.mean((prediction - noisy) ** 2)),
                    "teacher_energy_captured": captured,
                    "interpolating": trained["interpolating"],
                    "elapsed_seconds": time.perf_counter() - block,
                    "device": "cpu",
                }
                cells.append(record)
                pilot_spec = protocol["timing_pilot_cell"]
                if (data_seed == pilot_spec["data_seed"] and width == pilot_spec["width"] and float(snr) == float(pilot_spec["snr"])
                        and procedure == pilot_spec["procedure"] and decay == float(pilot_spec["weight_decay"])):
                    pilot = {"elapsed_seconds": record["elapsed_seconds"], "width": width, "data_seed": data_seed,
                             "residual_relative": record["residual_relative"], "epochs_ran": record["epochs_ran"],
                             "stop_reason": record["stop_reason"]}
    if phase == "discovery" and pilot is None:
        raise RuntimeError("timing pilot cell missing")
    used = sorted({cell["data_seed"] for cell in cells})
    audit = {
        "phase": phase,
        "data_seeds": used,
        "confirmation_seeds_absent": phase == "discovery" and not (set(used) & set(protocol["confirmation_data_seeds"])),
        "discovery_seeds_absent": phase == "confirmation" and not (set(used) & set(protocol["discovery_data_seeds"])),
        "eval_seed": evaluation["eval_seed"],
    }
    if phase == "discovery" and not audit["confirmation_seeds_absent"]:
        raise RuntimeError("confirmation seeds appeared")
    if phase == "confirmation" and not audit["discovery_seeds_absent"]:
        raise RuntimeError("discovery seeds appeared")
    analysis = evaluate_predictions(cells, protocol, phase)
    runtime = {
        "device": "cpu",
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "torch_threads": int(torch.get_num_threads()),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
        "scientific_validity": "not_assessed",
    }
    if runtime["cuda_initialized"] or runtime["torch_threads"] > 4:
        raise RuntimeError("CPU thread or CUDA guard failed")
    return {"cells": cells, "pilot": pilot, "seed_audit": audit, "analysis": analysis,
            "runtime": runtime, "wall_seconds": time.perf_counter() - started}


def write_phase_outputs(protocol: dict[str, Any], phase: str, result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "cells.jsonl"
    with cells_path.open("w", encoding="utf-8", newline="\n") as handle:
        for cell in result["cells"]:
            handle.write(json.dumps(pythonize(cell), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    analysis = dict(result["analysis"])
    analysis["n_cells"] = len(result["cells"])
    analysis["cells_sha256"] = sha256_file(cells_path)
    write_json(out_dir / "analysis.json", analysis)
    write_json(out_dir / "runtime.json", result["runtime"])
    write_json(out_dir / "seed-audit.json", result["seed_audit"])
    write_json(out_dir / "timing.json", {"schema_version": 1, "scientific_validity": "not_assessed",
                                         "pilot": result["pilot"], "wall_seconds": result["wall_seconds"],
                                         "residual_tolerance": protocol["relative_train_residual_tolerance"]})
    if phase == "discovery":
        reduction = reduce_menu(result["wall_seconds"], len(protocol["discovery_data_seeds"]),
                                len(protocol["confirmation_data_seeds"]), protocol)
        write_json(out_dir / "reduction.json", reduction)
        write_json(out_dir / "discovery-export.json", {
            "schema_version": 1,
            "phase": "discovery",
            "scientific_validity": "not_assessed",
            "protocol_sha256": hashlib.sha256(canonical_protocol(protocol)).hexdigest(),
            "timing": {"pilot_elapsed_seconds": None if result["pilot"] is None else result["pilot"]["elapsed_seconds"],
                       "wall_seconds": result["wall_seconds"]},
            "reduction": reduction,
            "seed_audit": result["seed_audit"],
            "n_cells": len(result["cells"]),
        })
    else:
        write_json(out_dir / "reduction.json", {"rule": "not_applicable", "used_clean_risk": False})


def build_discovery(protocol_path: Path, out_dir: Path) -> None:
    protocol = load_json(protocol_path)
    result = run_phase(protocol, "discovery", list(protocol["widths"]), out_dir)
    write_phase_outputs(protocol, "discovery", result, out_dir)


def build_confirmation(protocol_path: Path, grid_path: Path, export_path: Path, out_dir: Path) -> None:
    protocol = load_json(protocol_path)
    grid = load_json(grid_path)
    export_bytes = export_path.read_bytes()
    export = json.loads(export_bytes.decode("utf-8"))
    expected = freeze_confirmation_grid(export, protocol, hashlib.sha256(export_bytes).hexdigest())
    if grid.get("kept_widths") != expected["kept_widths"] or grid.get("discovery_export_sha256") != expected["discovery_export_sha256"]:
        raise RuntimeError("frozen grid does not match the time-only export")
    result = run_phase(protocol, "confirmation", list(grid["kept_widths"]), out_dir)
    write_phase_outputs(protocol, "confirmation", result, out_dir)
    write_json(out_dir / "bound-discovery.json", {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "discovery_export_sha256": expected["discovery_export_sha256"],
        "kept_widths": expected["kept_widths"],
        "used_clean_risk": False,
        "rule": "time_only",
        "seed_audit": result["seed_audit"],
    })
