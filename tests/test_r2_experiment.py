"""Solver, seed and menu-rule tests for the R2 protocol. No confirmatory fit."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.controlled_synthetic import (
    apply_latent_spectrum,
    column_std,
    generator,
    latent_standardizer,
    prepare_draw,
    teacher_beta,
)
from src.r2_experiment import (
    FORBIDDEN_EXPORT_KEYS,
    evaluate_predictions,
    freeze_confirmation_grid,
    load_json,
    reduce_menu,
    run_phase,
    sha256_bytes,
    canonical_protocol,
    validate_protocol,
    write_phase_outputs,
)
from src.ridge_svd import fit_ridge_path, interpolation_flag, predict, standardize_train


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research/stages/R2/inputs/protocol.json"


def repo_protocol() -> dict:
    return load_json(PROTOCOL_PATH)


def tiny_protocol() -> dict:
    protocol = repo_protocol()
    protocol.update({
        "n_train": 40,
        "n_eval": 25,
        "p_max": 80,
        "columns": [10, 40, 80],
        "window_columns": [40],
        "discovery_data_seeds": [7],
        "confirmation_data_seeds": [8],
        "minimum_bracketed_draws": 1,
        "timing_pilot_cell": {
            "family": "linear",
            "data_seed": 7,
            "column": 40,
            "spectrum": "isotropic",
            "snr": 1.0,
            "ridge_lambda": 0.0,
        },
    })
    protocol["reduction_rule"] = dict(protocol["reduction_rule"])
    protocol["reduction_rule"]["fallback_columns"] = [10, 40, 80]
    validate_protocol(protocol)
    return protocol


class ProtocolTest(unittest.TestCase):
    def test_repo_protocol_locks_the_numeric_rules(self) -> None:
        protocol = repo_protocol()
        validate_protocol(protocol)
        self.assertEqual(protocol["relative_train_residual_tolerance"], 1e-8)
        self.assertEqual(protocol["rcond_relative"], 1e-10)
        self.assertEqual(protocol["n_train"], 200)
        self.assertEqual(protocol["columns"], [50, 100, 150, 200, 250, 300, 400, 800])
        self.assertEqual(protocol["window_columns"], [150, 200, 250])
        self.assertEqual(protocol["discovery_data_seeds"], [1101, 1102, 1103, 1104, 1105])
        self.assertEqual(protocol["confirmation_data_seeds"], list(range(2101, 2111)))
        self.assertEqual(protocol["contrast_margins"]["clean_risk_absolute"], 0.01)
        self.assertFalse(set(protocol["discovery_data_seeds"]) & set(protocol["confirmation_data_seeds"]))
        beta = teacher_beta(protocol, generator(protocol, "discovery", "data", 1101), 800)
        self.assertAlmostEqual(float(np.linalg.norm(beta)), 1.0, places=12)
        self.assertGreater(float(np.sum(beta[:50] ** 2)), 0.999)

    def test_draws_repeat_and_keep_channels_separate(self) -> None:
        protocol = repo_protocol()
        first = prepare_draw(protocol, "discovery", 1101)
        second = prepare_draw(protocol, "discovery", 1101)
        other_phase = prepare_draw(protocol, "confirmation", 1101)
        self.assertTrue(np.array_equal(first["beta"], second["beta"]))
        self.assertTrue(np.array_equal(first["covariates"], second["covariates"]))
        self.assertFalse(np.array_equal(first["covariates"], other_phase["covariates"]))
        self.assertNotEqual(first["channels"]["data"], first["channels"]["noise"])


class SolverTest(unittest.TestCase):
    def test_ridge_matches_the_centered_normal_equation(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.standard_normal((20, 6))
        response = features @ rng.standard_normal(6) + rng.standard_normal(20)
        fitted = fit_ridge_path(features, response, [0.1], 1e-10)
        solution = fitted["solutions"][0]
        standardized = fitted["fit"]["standardized"]
        centered = response - response.mean()
        gram = standardized.T @ standardized + 20 * 0.1 * np.eye(6)
        reference = np.linalg.solve(gram, standardized.T @ centered)
        self.assertLess(float(np.linalg.norm(solution["slope"] - reference)), 1e-8)

    def test_minimum_norm_interpolates_only_once_the_row_space_is_full(self) -> None:
        rng = np.random.default_rng(1)
        response_noise = rng.standard_normal(30)
        wide = rng.standard_normal((30, 45))
        square = rng.standard_normal((30, 30))
        narrow = rng.standard_normal((30, 8))
        signal = rng.standard_normal(8)
        response = narrow @ signal + response_noise
        for features, expect in ((wide, True), (square, True), (narrow, False)):
            fitted = fit_ridge_path(features, response, [0.0], 1e-10)
            flag = interpolation_flag(fitted["solutions"][0], fitted, 1e-8)
            self.assertIs(flag, expect)
            self.assertEqual(fitted["rank_target"], min(29, features.shape[1]))
        under = fit_ridge_path(narrow, response, [0.0], 1e-10)
        self.assertGreater(under["solutions"][0]["residual_relative"], 1e-3)

    def test_evaluation_reuses_training_location_and_scale(self) -> None:
        rng = np.random.default_rng(2)
        train = rng.standard_normal((15, 4)) + np.array([5.0, -2.0, 0.0, 3.0])
        fitted = fit_ridge_path(train, rng.standard_normal(15), [0.0], 1e-10)
        shifted = train + 10.0
        held = predict(shifted, fitted, fitted["solutions"][0])
        original = predict(train, fitted, fitted["solutions"][0])
        self.assertFalse(np.allclose(held, original))
        self.assertTrue(np.allclose(fitted["fit"]["mean"], train.mean(axis=0)))

    def test_geometric_spectrum_reaches_the_decomposed_matrix(self) -> None:
        protocol = repo_protocol()
        draw = prepare_draw(protocol, "discovery", 3)
        self.assertNotIn(3, protocol["discovery_data_seeds"] + protocol["confirmation_data_seeds"])
        latent = latent_standardizer(draw["covariates"])
        isotropic = apply_latent_spectrum(
            draw["covariates"], latent, column_std(protocol, "isotropic", protocol["p_max"]))
        geometric = apply_latent_spectrum(
            draw["covariates"], latent, column_std(protocol, "geometric", protocol["p_max"]))
        width = 200
        self.assertFalse(np.allclose(isotropic[:, :width], geometric[:, :width]))
        cancelled_isotropic = standardize_train(isotropic[:, :width])["standardized"]
        cancelled_geometric = standardize_train(geometric[:, :width])["standardized"]
        self.assertTrue(np.allclose(cancelled_isotropic, cancelled_geometric, atol=1e-6, rtol=1e-6))
        response = draw["label_noise"]
        isotropic_fit = fit_ridge_path(isotropic[:, :width], response, [0.0], 1e-10, rescale_columns=False)
        geometric_fit = fit_ridge_path(geometric[:, :width], response, [0.0], 1e-10, rescale_columns=False)
        self.assertFalse(isotropic_fit["fit"]["rescaled_columns"])
        self.assertGreater(geometric_fit["condition_number"], isotropic_fit["condition_number"])


class MenuRuleTest(unittest.TestCase):
    def test_time_keeps_or_falls_back_without_reading_risk(self) -> None:
        protocol = repo_protocol()
        full = reduce_menu(1.0, 5, 10, protocol)
        self.assertEqual(full["kept_columns"], protocol["columns"])
        self.assertFalse(full["used_clean_risk"])
        fallback = reduce_menu(10000.0, 5, 10, protocol)
        self.assertEqual(fallback["kept_columns"], protocol["reduction_rule"]["fallback_columns"])
        self.assertLess(fallback["projected_confirmation_seconds"], 1e9)

    def test_freeze_rejects_risk_fields_and_ignores_extra_diagnostics(self) -> None:
        protocol = repo_protocol()
        reduction = reduce_menu(1.0, 5, 10, protocol)
        export = {
            "protocol_sha256": sha256_bytes(canonical_protocol(protocol)),
            "reduction": reduction,
            "seed_audit": {
                "confirmation_seeds_absent": True,
                "data_seeds": protocol["discovery_data_seeds"],
            },
            "bracket_counts": {"should_not_change_the_menu": 0},
        }
        grid = freeze_confirmation_grid(export, protocol, "abc")
        self.assertEqual(grid["kept_columns"], protocol["columns"])
        self.assertFalse(grid["used_clean_risk"])
        contaminated = dict(export)
        contaminated["clean_mse"] = [1.0]
        self.assertIn("clean_mse", FORBIDDEN_EXPORT_KEYS)
        with self.assertRaises(RuntimeError):
            freeze_confirmation_grid(contaminated, protocol, "abc")


class ContrastRuleTest(unittest.TestCase):
    def test_predeclared_directions_are_scored_from_cells(self) -> None:
        protocol = repo_protocol()
        protocol = copy.deepcopy(protocol)
        protocol["minimum_bracketed_draws"] = 5
        cells = []
        for seed in protocol["discovery_data_seeds"]:
            for spectrum, risk, norm, condition in (
                ("isotropic", 1.0, 2.0, 10.0),
                ("geometric", 1.4, 2.0, 40.0),
            ):
                for column, interpolating, residual in ((50, False, 0.4), (150, False, 0.2),
                                                        (200, True, 1e-12), (250, True, 1e-12),
                                                        (800, True, 1e-12)):
                    for snr, risk_scale, norm_scale in ((1.0, 1.0, 1.0), (5.0, 0.5, 0.5)):
                        cells.append({
                            "phase": "discovery",
                            "family": "linear",
                            "data_seed": seed,
                            "spectrum": spectrum,
                            "snr": snr,
                            "ridge_lambda": 0.0,
                            "column": column,
                            "clean_mse": risk * risk_scale,
                            "coefficient_l2": norm * norm_scale,
                            "condition_number": condition,
                            "gram_condition_number": condition ** 2,
                            "residual_relative": residual,
                            "interpolating": interpolating,
                        })
                        for ridge_lambda, factor in ((0.01, 0.5), (0.1, 0.4), (1.0, 0.3)):
                            cells.append({
                                "phase": "discovery",
                                "family": "linear",
                                "data_seed": seed,
                                "spectrum": spectrum,
                                "snr": snr,
                                "ridge_lambda": ridge_lambda,
                                "column": column,
                                "clean_mse": risk * risk_scale * factor,
                                "coefficient_l2": norm * norm_scale * factor,
                                "condition_number": condition,
                                "gram_condition_number": condition ** 2,
                                "residual_relative": 0.2,
                                "interpolating": None,
                            })
            for column, interpolating, residual, clean, gram in (
                (50, False, 0.3, 0.5, 4.0),
                (150, False, 0.2, 0.7, 8.0),
                (200, True, 1e-12, 0.4, 20.0),
                (250, True, 1e-12, 0.9, 30.0),
                (800, True, 1e-12, 0.3, 18.0),
            ):
                for snr in (1.0, 5.0):
                    cells.append({
                        "phase": "discovery",
                        "family": "random_features",
                        "data_seed": seed,
                        "spectrum": "isotropic",
                        "snr": snr,
                        "ridge_lambda": 0.0,
                        "column": column,
                        "clean_mse": clean,
                        "coefficient_l2": 1.0,
                        "condition_number": math.sqrt(gram),
                        "gram_condition_number": gram,
                        "residual_relative": residual,
                        "interpolating": interpolating,
                    })
        scored = evaluate_predictions(cells, protocol, "discovery")
        self.assertEqual(scored["noise"]["primary"]["outcome"], "supported")
        self.assertEqual(scored["spectrum"]["primary"]["outcome"], "supported")
        self.assertEqual(scored["ridge"]["primary"]["outcome"], "supported")
        self.assertEqual(scored["random_features"]["snr_1"]["coupling_outcome"], "coupling_consistent")
        self.assertEqual(scored["label"], "exploratory_discovery")


class TinyPhaseTest(unittest.TestCase):
    def test_discovery_phase_brackets_and_leaves_confirmation_seeds_unused(self) -> None:
        protocol = tiny_protocol()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            result = run_phase(protocol, "discovery", list(protocol["columns"]), out)
            write_phase_outputs(protocol, "discovery", result, out)
            seeds = {cell["data_seed"] for cell in result["cells"]}
            self.assertEqual(seeds, {7})
            self.assertTrue(result["seed_audit"]["confirmation_seeds_absent"])
            self.assertFalse(json.loads((out / "reduction.json").read_text(encoding="utf-8"))["used_clean_risk"])
            square = [cell for cell in result["cells"]
                      if cell["family"] == "linear" and cell["spectrum"] == "isotropic"
                      and cell["column"] == 80 and cell["ridge_lambda"] == 0.0 and cell["snr"] == 1.0]
            self.assertEqual(len(square), 1)
            self.assertTrue(square[0]["interpolating"])
            self.assertLess(square[0]["residual_relative"], 1e-8)
            self.assertIsNotNone(result["pilot"])


if __name__ == "__main__":
    unittest.main()
