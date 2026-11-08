"""Pure numerical and mocked failure checks; no study generation/model fitting."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.response_forecast import (
    BASELINE_IDS, MODEL_IDS, _RunState, _execute_fit,
    decision_gates, forecast_features, geometry_features, oof_forecasts,
    ridge_prediction, run_study, trajectory_efc_proxy, validate_protocol,
)
from src.boosting_response import operators_from_partitions


def protocol_fixture():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "research/stages/R14/inputs/protocol.json").read_text(encoding="utf-8"))
    # All mocked tests use non-study seed labels; no random generator is called.
    protocol["seeds"] = [5, 6, 7, 8]
    return protocol


def forecast_fixture():
    rows = []
    for index, seed in enumerate([5, 6, 7, 8]):
        for teacher in ("linear", "interaction"):
            for amplitude, epsilon in enumerate((.01, .1)):
                cheap = .05 + .02 * index + .01 * amplitude + .03 * (teacher == "interaction")
                rows.append({"id": f"toy_{seed}_{teacher}_{amplitude}", "context_id": f"toy_{seed}_{teacher}",
                             "seed": seed, "teacher": teacher, "prefix": 4, "epsilon": epsilon,
                             "train_mse": .1 + index * .01, "collision_mse_floor": .02,
                             "trajectory_weighted_per_tree_efc_proxy": .6,
                             "frozen_eval_noise_gain": .5, "frozen_train_noise_gain": .7,
                             "largest_train_singular_value": .9, "cheap_discrepancy": cheap,
                             "full_discrepancy": 2 * cheap + .1, "valid_directions": True})
    return rows


def decision_fixture():
    summaries = {model: {"mae_u": 1.0, "draw_mae_u": {str(seed): 1.0 for seed in [5, 6, 7, 8]},
                         "teacher_mae_u": {"linear": 1.0, "interaction": 1.0}} for model in MODEL_IDS}
    summaries["A"] = {"mae_u": .1, "draw_mae_u": {str(seed): .1 for seed in [5, 6, 7, 8]},
                       "teacher_mae_u": {"linear": .1, "interaction": .1}}
    for baseline, bad_seed in (("B2", 5), ("B3", 6), ("B4", 7)):
        summaries[baseline] = {"mae_u": .7625,
                               "draw_mae_u": {str(seed): .05 if seed == bad_seed else 1.0 for seed in [5, 6, 7, 8]},
                               "teacher_mae_u": {"linear": .7625, "interaction": .7625}}
    return {"status": "complete_discovery_cross_validation", "summaries": summaries, "target_q90_minus_q10": .2}


class OneLeafMockModel:
    """Deterministic arithmetic fixture, never a fitted CatBoost model."""
    def __init__(self, actual_y, iterations, clock=None):
        self.actual_y = np.asarray(actual_y, dtype=float)
        self.tree_count_ = iterations
        self.clock = clock

    def predict(self, features, *, ntree_end=0, **kwargs):
        if self.clock is not None:
            self.clock[0] += 4.0
        trees = ntree_end or self.tree_count_
        gain = 1 - (1 - float(np.float32(.1)) * len(self.actual_y) / (len(self.actual_y) + 1)) ** trees
        return np.full(len(features), gain * np.mean(self.actual_y))

    def calc_leaf_indexes(self, features, **kwargs):
        if self.clock is not None:
            self.clock[0] += 50.0
        return np.zeros((len(features), self.tree_count_), dtype=np.int64)

    def get_tree_leaf_counts(self):
        return np.ones(self.tree_count_, dtype=np.int64)

    def get_leaf_weights(self):
        return np.full(self.tree_count_, len(self.actual_y), dtype=float)

    def get_all_params(self):
        return {"fixture": "pure-one-leaf-mock", "iterations": self.tree_count_}

    def save_model(self, path, **kwargs):
        Path(path).write_bytes(b"pure-mock-not-a-CatBoost-model")

    def staged_predict(self, features, **kwargs):
        for at in range(1, self.tree_count_ + 1):
            yield self.predict(features, ntree_end=at)


def mocked_replay(model, x_train, x_evaluation, checkpoints, *, targets, **kwargs):
    train_leaves = np.zeros((len(x_train), model.tree_count_), dtype=np.int64)
    evaluation_leaves = np.zeros((len(x_evaluation), model.tree_count_), dtype=np.int64)
    operators = operators_from_partitions(train_leaves, evaluation_leaves,
                                         eta=float(np.float32(.1)), l2=1,
                                         checkpoints=checkpoints, leaf_counts=[1] * model.tree_count_)
    checks = []
    for at, split_operators in operators.items():
        for split, features in (("train", x_train), ("evaluation", x_evaluation)):
            error = float(np.max(np.abs(split_operators[split] @ targets - model.predict(features, ntree_end=at))))
            if error > 1e-10:
                raise AssertionError("Arithmetic mock lost its exact conditional replay")
            checks.append({"trees": at, "split": split, "n": len(features), "passed": True,
                           "atol": 1e-10, "rtol": 0, "max_absolute_error": error})
    return {"operators": operators, "train_leaf_indices": train_leaves,
            "evaluation_leaf_indices": evaluation_leaves,
            "actual_learning_rate": float(np.float32(.1)), "actual_l2_leaf_reg": 1,
            "baseline_checks": checks}


class ResponseForecastTests(unittest.TestCase):
    def test_protocol_matrix_counts_and_replay_tolerance_are_checked(self):
        protocol = protocol_fixture()
        validate_protocol(protocol)
        for key, changed in (("expected_fits", protocol["expected_fits"] - 1), ("replay_atol", 1e-8)):
            bad = copy.deepcopy(protocol)
            bad[key] = changed
            with self.assertRaises(ValueError):
                validate_protocol(bad)

    def test_efc_excludes_diagonal_and_uses_positive_sse_weights(self):
        y = np.asarray([1, 1, -1, -1], dtype=float)
        leaves = np.asarray([[0, 0], [0, 1], [1, 2], [1, 3]])
        predictions = np.asarray([np.zeros(4), .5 * y, .75 * y])
        result = trajectory_efc_proxy(leaves, predictions, y)
        np.testing.assert_allclose(result["per_tree"], [2 / 3, 1])
        np.testing.assert_allclose(result["weights"], [3, .75])
        self.assertAlmostEqual(result["value"], 11 / 15)
        with self.assertRaisesRegex(ValueError, "Undefined"):
            trajectory_efc_proxy(leaves, np.zeros((3, 4)), np.zeros(4))

    def test_frozen_geometry_normalization(self):
        result = geometry_features(np.diag([1.0, .5]), np.asarray([[1.0, 0]]))
        self.assertEqual(result["frozen_eval_noise_gain"], 1)
        self.assertEqual(result["frozen_train_noise_gain"], .625)
        self.assertEqual(result["largest_train_singular_value"], 1)

    def test_ridge_fits_intercept_and_train_only_scaling(self):
        predictions, fit = ridge_prediction(np.asarray([[0, 1], [2, 1]]), np.asarray([0, 2]), np.asarray([[100, 1]]))
        self.assertEqual(fit["feature_mean"], [1, 1])
        self.assertEqual(fit["feature_scale"], [1, 1])
        self.assertEqual(fit["intercept_centered_features"], 1)
        self.assertAlmostEqual(predictions[0], 67)

    def test_oof_holds_out_all_amplitudes_and_teachers_of_one_draw(self):
        rows = forecast_fixture()
        original = oof_forecasts(rows, [5, 6, 7, 8])
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["seed"] == 5:
                row["full_discrepancy"] = 1e8
        perturbed_targets = oof_forecasts(changed, [5, 6, 7, 8])
        original_heldout = [row for row in original["rows"] if row["seed"] == 5]
        changed_heldout = [row for row in perturbed_targets["rows"] if row["seed"] == 5]
        for before, after in zip(original_heldout, changed_heldout):
            for model in MODEL_IDS:
                self.assertEqual(before["models"][model]["prediction_u"], after["models"][model]["prediction_u"])
        for fold in original["folds"]:
            self.assertNotIn(fold["held_out_seed"], fold["training_seeds"])
            self.assertTrue(all(not identifier.startswith(f"toy_{fold['held_out_seed']}_") for identifier in fold["training_row_ids"]))
        self.assertEqual(len(original["rows"]), 16)
        self.assertEqual(set(original["summaries"]), set(MODEL_IDS))

    def test_identity_probe_baseline_uses_only_training_offset(self):
        rows = forecast_fixture()
        forecast = oof_forecasts(rows, [5, 6, 7, 8])
        heldout = next(row for row in forecast["rows"] if row["seed"] == 5)
        training = [row for row in rows if row["seed"] != 5]
        offset = np.mean([np.log10(1 + row["full_discrepancy"]) - np.log10(1 + row["cheap_discrepancy"]) for row in training])
        self.assertAlmostEqual(heldout["models"]["B4"]["prediction_u"], np.log10(1 + heldout["cheap_discrepancy"]) + offset)
        fit = next(fold for fold in forecast["folds"] if fold["model"] == "B4" and fold["held_out_seed"] == 5)
        self.assertEqual(fit["fit"]["slope"], 1)

    def test_invalid_direction_keeps_rows_without_fabricated_oof_values(self):
        rows = forecast_fixture()
        rows[0]["valid_directions"] = False
        forecast = oof_forecasts(rows, [5, 6, 7, 8])
        self.assertEqual(forecast["status"], "inconclusive_invalid_direction")
        self.assertEqual(len(forecast["rows"]), 16)
        self.assertFalse(forecast["go"])
        self.assertTrue(all("models" not in row for row in forecast["rows"]))

    def test_feature_transform_does_not_read_full_response_endpoint(self):
        row = forecast_fixture()[0]
        original = forecast_features(row)
        row["full_discrepancy"] = float("nan")
        self.assertEqual(original, forecast_features(row))

    def test_draw_gate_is_per_baseline_not_silently_joint(self):
        decision = decision_gates(decision_fixture(), protocol_fixture(), replay_passed=True, cheap_wall=.5, full_wall=2)
        self.assertTrue(decision["P2_forecast"])
        self.assertEqual(decision["draws_jointly_not_worse"], 1)
        self.assertEqual(decision["draws_not_worse_per_baseline"], {"B2": 3, "B3": 3, "B4": 3})
        self.assertTrue(decision["go"])

    def test_zero_baseline_mae_has_no_relative_improvement(self):
        forecast = decision_fixture()
        forecast["summaries"]["A"]["mae_u"] = 0
        forecast["summaries"]["B4"]["mae_u"] = 0
        decision = decision_gates(forecast, protocol_fixture(), replay_passed=True, cheap_wall=.5, full_wall=2)
        self.assertFalse(decision["aggregate_at_least_relative_improvement"]["B4"])
        self.assertFalse(decision["relative_improvement_vs_baseline"]["B4"]["defined"])
        self.assertIsNone(decision["relative_improvement_vs_baseline"]["B4"]["value"])
        self.assertIn("zero", decision["relative_improvement_vs_baseline"]["B4"]["reason"])
        self.assertFalse(decision["go"])

    def test_task_failure_is_saved_without_generating_or_fitting(self):
        protocol = protocol_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            output = root / "task"
            with patch("src.response_forecast.draw_inputs", side_effect=RuntimeError("technical test interruption")), patch("src.response_forecast.fit_control") as fitting:
                with self.assertRaisesRegex(RuntimeError, "technical test interruption"):
                    run_study(protocol, output, protocol_path)
                fitting.assert_not_called()
            progress = json.loads((output / "progress.json").read_text())
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(progress["phase"], "failed")
            self.assertEqual(failure["fits_completed"], 0)
            self.assertTrue((output / "contexts.jsonl").exists())

    def test_fit_inputs_exist_before_a_mocked_fit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = _RunState(root, protocol_fixture())
            state.context = "pure_toy"
            y = np.asarray([1.0, -1.0])
            with patch("src.response_forecast.fit_control", side_effect=RuntimeError("mocked failure")):
                with self.assertRaisesRegex(RuntimeError, "mocked failure"):
                    _execute_fit(state, "pure_toy_base", "base", np.asarray([[0], [1]]), np.asarray([[0]]), y, 1, 3, {"actual_targets": y})
            with np.load(root / "fit_evidence/pure_toy_base.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["actual_targets"], y)
            self.assertEqual(state.fits_completed, 0)

    def test_leaf_audit_time_is_excluded_from_prediction_time(self):
        clock = [0.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            y = np.asarray([1.0, -1.0])

            def mock_fit(*args, **kwargs):
                clock[0] += 3.0
                return OneLeafMockModel(y, 2, clock=clock)

            with patch("src.response_forecast.time.perf_counter", side_effect=lambda: clock[0]), patch("src.response_forecast.fit_control", side_effect=mock_fit):
                state = _RunState(root, protocol_fixture())
                state.context = "pure_timing_toy"
                _, arrays, timing, _ = _execute_fit(state, "toy", "pilot", np.asarray([[0], [1]]), np.asarray([[0]]), y, 2, 3, {"actual_targets": y})
            self.assertEqual(timing["fit_seconds"], 3)
            self.assertEqual(timing["prediction_seconds"], 8)
            self.assertEqual(timing["leaf_evidence_seconds"], 100)
            self.assertIn("train__leaf_indices", arrays)
            self.assertIn("evaluation__leaf_indices", arrays)

    def test_mocked_pipeline_preserves_full_output_contract_without_generation_or_fit(self):
        protocol = protocol_fixture()
        protocol.update(seeds=[5, 6], n_train=12, n_evaluation=3,
                        iterations=5, pilot_iterations=2, epsilon_values=[.1],
                        pilot_directions=1, response_directions=1,
                        expected_contexts=16, expected_fits=48, expected_forecast_rows=16)
        protocol["decision_gates"]["min_draws_not_worse"] = 2
        raw_train = np.asarray([[i / 4, i % 3, (i * 3) % 5, (i * 5) % 7] for i in range(12)], dtype=float)
        raw = {"raw_train": raw_train,
               "raw_evaluation": np.asarray([[.5, 1, 2, 3], [1.5, 0, 4, 2], [2.5, 2, 1, 1]], dtype=float),
               "train_noise": np.ones(12) * .1,
               "pilot_directions": np.asarray([[1, -1] * 6], dtype=float),
               "response_directions": np.asarray([[1, 1, -1] * 4], dtype=float)}

        def mock_fit(x_train, actual_y, *, iterations, **kwargs):
            return OneLeafMockModel(actual_y, iterations)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            output = root / "task"
            with patch("src.response_forecast.draw_inputs", side_effect=lambda *args: {key: value.copy() for key, value in raw.items()}) as generated, patch("src.response_forecast.fit_control", side_effect=mock_fit) as fitted, patch("src.response_forecast.frozen_operators", side_effect=mocked_replay):
                analysis = run_study(protocol, output, protocol_path)
            self.assertEqual(generated.call_count, 2)  # Patched literal fixture, no RNG.
            self.assertEqual(fitted.call_count, 48)  # Pure mock; no CatBoost fit.
            self.assertEqual(analysis["n_contexts"], 16)
            self.assertEqual(analysis["n_forecast_rows"], 16)
            self.assertEqual(analysis["n_directions"], 32)
            self.assertEqual(analysis["n_fits"], 48)
            for filename in ("analysis.json", "contexts.jsonl", "directions.jsonl", "forecast_rows.jsonl",
                             "feature_snapshots.jsonl", "data.npz", "predictions.npz", "operators.npz",
                             "models.json", "timings.json", "environment.json", "budget_checks.json", "progress.json"):
                self.assertTrue((output / filename).is_file(), filename)
            self.assertEqual(len(list((output / "models").glob("*.cbm"))), 48)
            self.assertEqual(len(list((output / "fit_evidence").glob("*.npz"))), 48)
            self.assertEqual(len(list((output / "context_evidence").glob("*.npz"))), 16)
            self.assertEqual(json.loads((output / "progress.json").read_text())["phase"], "complete")
            self.assertFalse(analysis["decision"]["go"])
            self.assertFalse(analysis["decision"]["range_gate"])
            self.assertEqual(analysis["forecast"]["status"], "complete_discovery_cross_validation")
            self.assertEqual(set(analysis["costs"]["leaf_evidence_seconds_separate"]), {"base", "cheap", "full"})
            for name in ("data.npz", "predictions.npz", "operators.npz"):
                with np.load(output / name, allow_pickle=False) as archive:
                    self.assertTrue(archive.files)


if __name__ == "__main__":
    unittest.main()
