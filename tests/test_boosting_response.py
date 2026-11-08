"""Pure algebra by default; CatBoost precision smoke requires explicit opt-in."""

from __future__ import annotations

import os
import json
from importlib.metadata import version
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.boosting_response import (
    canonicalize_targets, decomposition, fit_control, frozen_operators,
    nested_binary_features, operators_from_partitions, representation_mse_floor,
)


class BoostingResponsePureTests(unittest.TestCase):
    def test_train_only_dyadic_prefixes_and_all_raw_features(self):
        train = np.arange(32, dtype=float).reshape(8, 4)
        evaluation = np.asarray([[1000, -1000, 1000, -1000]], dtype=float)
        view = nested_binary_features(train, evaluation)
        other = nested_binary_features(train, -evaluation)
        self.assertEqual(view.prefix_budgets, (4, 12, 28, 60))
        self.assertEqual(view.train.shape, (8, 60))
        self.assertEqual(view.evaluation.shape, (1, 60))
        self.assertEqual(view.thresholds, other.thresholds)
        self.assertEqual([t.raw_feature for t in view.thresholds[:4]], [0, 1, 2, 3])
        self.assertEqual([view.thresholds[i].probability for i in (0, 4, 8, 12)], [.5, .25, .75, .125])
        for budget in view.prefix_budgets:
            x_train, x_eval = view.prefix(budget)
            np.testing.assert_array_equal(x_train, view.train[:, :budget])
            np.testing.assert_array_equal(x_eval, view.evaluation[:, :budget])
        with self.assertRaises(ValueError):
            view.train[0, 0] = 1
        self.assertEqual(view.manifest()["fit_on"], "training_covariates_only")

    def test_tied_quantiles_retained_and_empty_evaluation_supported(self):
        view = nested_binary_features(np.ones((4, 4)), np.empty((0, 4)))
        self.assertEqual(view.train.shape, (4, 60))
        self.assertEqual(view.evaluation.shape, (0, 60))
        self.assertEqual(int(view.train.sum()), 0)
        self.assertEqual(len(view.thresholds), 60)

    def test_invalid_quantile_design_rejected(self):
        with self.assertRaises(ValueError):
            nested_binary_features(np.ones((4, 4)), np.ones((2, 3)))
        with self.assertRaises(ValueError):
            nested_binary_features(np.ones((4, 4)), np.ones((2, 4)), threshold_levels=(1, 2))

    def test_one_tree_operator_has_regularized_leaf_means_and_empty_leaf(self):
        train_leaves = np.asarray([[0], [0], [1]])
        eval_leaves = np.asarray([[1], [2]])
        out = operators_from_partitions(train_leaves, eval_leaves, eta=.5, l2=1, checkpoints=[0, 1], leaf_counts=[3])
        expected_train = np.asarray([[1 / 6, 1 / 6, 0], [1 / 6, 1 / 6, 0], [0, 0, .25]])
        np.testing.assert_allclose(out[1]["train"], expected_train, atol=1e-15)
        np.testing.assert_allclose(out[1]["evaluation"], [[0, 0, .25], [0, 0, 0]], atol=1e-15)
        np.testing.assert_array_equal(out[0]["train"], np.zeros((3, 3)))
        self.assertFalse(out[1]["train"].flags.writeable)

    def test_two_tree_residual_update_uses_preupdate_train_operator(self):
        train_leaves = np.asarray([[0, 0], [0, 1]])
        eval_leaves = np.asarray([[0, 1]])
        out = operators_from_partitions(train_leaves, eval_leaves, eta=.5, l2=0, checkpoints=[1, 2])
        np.testing.assert_allclose(out[1]["train"], [[.25, .25], [.25, .25]])
        np.testing.assert_allclose(out[2]["train"], [[.625, .125], [.125, .625]])
        np.testing.assert_allclose(out[2]["evaluation"], [[.125, .625]])

    def test_decomposition_signed_negative_cross_is_retained(self):
        result = decomposition([1, 0], [2, 0], epsilon=.5)
        self.assertEqual(result["normalized_full"], 2)
        self.assertEqual(result["normalized_frozen"], 8)
        self.assertEqual(result["normalized_structural_discrepancy"], 2)
        self.assertEqual(result["normalized_twice_cross"], -8)
        self.assertEqual(result["closure_error"], 0)
        np.testing.assert_array_equal(result["structural_delta"], [-1, 0])

    def test_collision_floor_is_groupwise_empirical_control(self):
        signatures = [[0, 1], [0, 1], [1, 0], [1, 1]]
        result = representation_mse_floor(signatures, [1, 3, 2, 8])
        self.assertEqual(result["n_groups"], 3)
        self.assertEqual(result["mse_floor"], .5)
        np.testing.assert_array_equal(result["fitted"], [2, 2, 2, 8])
        self.assertTrue(result["known_control"])

    def test_native_rounding_delta_is_explicit(self):
        result = canonicalize_targets([.1, .5])
        self.assertEqual(result["native"].dtype, np.float32)
        self.assertNotEqual(result["rounding_delta"][0], 0)
        self.assertEqual(result["rounding_delta"][1], 0)
        actual_delta = canonicalize_targets([.2, .75])["actual"] - result["actual"]
        nominal_delta = np.asarray([.1, .25])
        self.assertNotEqual(actual_delta[0], nominal_delta[0])


@unittest.skipUnless(os.environ.get("RESEARCH_MODEL_SMOKE") == "1", "Explicit tiny CatBoost precision smoke only")
class BoostingResponseModelSmoke(unittest.TestCase):
    def test_fresh_short_fit_matches_full_prefix_and_long_replay(self):
        # The cheap scientific probe is a fresh short fit. This non-study toy
        # checks that the iteration budget itself does not change its baseline.
        raw_train = np.asarray([[i, i % 3, (i * 3) % 5, (i * 5) % 7] for i in range(12)], dtype=float)
        raw_eval = np.asarray([[.5, 1, 2, 3], [5.5, 0, 4, 2], [9.5, 2, 1, 1]], dtype=float)
        x_train, x_eval = nested_binary_features(raw_train, raw_eval).prefix(12)
        actual_y = np.asarray([-.5, .25, 1, -.25, .5, -1, .75, -.75, 1.25, -.5, .25, 0], dtype=np.float32)
        full = fit_control(x_train, actual_y, iterations=512, seed=71, threads=1)
        short = fit_control(x_train, actual_y, iterations=32, seed=71, threads=1)
        full_counts = np.asarray(full.get_tree_leaf_counts())[:32]
        np.testing.assert_array_equal(short.get_tree_leaf_counts(), full_counts)
        leaf_value_count = int(np.sum(full_counts))
        np.testing.assert_array_equal(short.get_leaf_values(), np.asarray(full.get_leaf_values())[:leaf_value_count])
        differences = {}
        for split, features in (("train", x_train), ("evaluation", x_eval)):
            short_prediction = np.asarray(short.predict(features, prediction_type="RawFormulaVal", thread_count=1))
            full_prefix = np.asarray(full.predict(features, prediction_type="RawFormulaVal", ntree_end=32, thread_count=1))
            np.testing.assert_allclose(short_prediction, full_prefix, atol=1e-10, rtol=0)
            np.testing.assert_array_equal(short.calc_leaf_indexes(features, thread_count=1),
                                          np.asarray(full.calc_leaf_indexes(features, thread_count=1))[:, :32])
            differences[split] = float(np.max(np.abs(short_prediction - full_prefix)))
        replay = frozen_operators(full, x_train, x_eval, checkpoints=[32, 512], targets=actual_y,
                                  threads=1, atol=1e-10, rtol=0)
        resolved_keys = ("iterations", "depth", "learning_rate", "l2_leaf_reg", "loss_function",
                         "boosting_type", "random_strength", "bootstrap_type", "leaf_estimation_method",
                         "leaf_estimation_iterations", "leaf_estimation_backtracking", "score_function",
                         "boost_from_average", "border_count", "grow_policy", "task_type", "rsm", "model_shrink_rate")
        resolved_full = {key: full.get_all_params()[key] for key in resolved_keys}
        resolved_short = {key: short.get_all_params()[key] for key in resolved_keys}
        self.assertEqual({key: value for key, value in resolved_full.items() if key != "iterations"},
                         {key: value for key, value in resolved_short.items() if key != "iterations"})
        print(json.dumps({"technical_smoke": "fresh-short32-vs-full512-prefix32",
                          "catboost": version("catboost"), "seed": 71, "threads": 1,
                          "n_train": 12, "n_evaluation": 3,
                          "resolved_full_control": resolved_full, "resolved_short_control": resolved_short,
                          "actual_scale_and_bias": full.get_scale_and_bias(),
                          "short_full_prefix_max_absolute_error": differences,
                          "leaf_indices_and_values_exactly_equal": True,
                          "baseline_checks": replay["baseline_checks"]}, allow_nan=False))

    def test_prefix_operator_precision_and_changed_label_frozen_replay(self):
        # Deterministic toy covariates and dyadic native labels, not study seeds.
        raw_train = np.asarray([[i, i % 3, (i * 3) % 5, (i * 5) % 7] for i in range(12)], dtype=float)
        raw_eval = np.asarray([[.5, 1, 2, 3], [5.5, 0, 4, 2], [9.5, 2, 1, 1]], dtype=float)
        view = nested_binary_features(raw_train, raw_eval)
        x_train, x_eval = view.prefix(12)
        actual_y = np.asarray([-.5, .25, 1, -.25, .5, -1, .75, -.75, 1.25, -.5, .25, 0], dtype=np.float32)
        model = fit_control(x_train, actual_y, iterations=5, seed=71, threads=1)
        replay = frozen_operators(model, x_train, x_eval, checkpoints=[0, 1, 3, 5], targets=actual_y, threads=1, atol=1e-10, rtol=0)
        self.assertTrue(all(row["passed"] for row in replay["baseline_checks"]))
        self.assertEqual(replay["actual_learning_rate"], float(np.float32(.1)))
        # Refit values while retaining the actual saved tree partitions.
        changed_y = actual_y.astype(float) + np.asarray([1, -1] * 6) * .125
        fitted = np.zeros(len(actual_y))
        per_tree_values = []
        for tree, count in enumerate(replay["leaf_counts"]):
            ids = replay["train_leaf_indices"][:, tree]
            sums = np.bincount(ids, weights=changed_y - fitted, minlength=int(count))
            occupancy = np.bincount(ids, minlength=int(count))
            values = replay["actual_learning_rate"] * sums / (occupancy + replay["actual_l2_leaf_reg"])
            fitted += values[ids]
            per_tree_values.append(values)
        # CatBoost copies use a non-solid serialized representation in this
        # version. Mutate only this disposable original toy model after the
        # baseline checks; no scientific model is edited.
        changed_values = np.concatenate(per_tree_values)
        model.set_leaf_values(changed_values)
        np.testing.assert_array_equal(model.get_leaf_values(), changed_values)
        # Reload to obtain a fresh evaluation calcer after the leaf mutation.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed-toy.cbm"
            model.save_model(str(path))
            loaded = type(model)()
            loaded.load_model(str(path))
            expected = loaded.predict(x_eval, prediction_type="RawFormulaVal", thread_count=1)
        np.testing.assert_allclose(replay["operators"][5]["evaluation"] @ changed_y, expected, atol=1e-10, rtol=0)
        print(json.dumps({"technical_smoke": "fixed-partition-native-label-control",
                          "catboost": version("catboost"), "seed": 71, "threads": 1,
                          "n_train": 12, "n_evaluation": 3, "trees": 5,
                          "actual_learning_rate": replay["actual_learning_rate"],
                          "actual_l2_leaf_reg": replay["actual_l2_leaf_reg"],
                          "baseline_checks": replay["baseline_checks"],
                          "changed_label_max_absolute_error": float(np.max(np.abs(replay["operators"][5]["evaluation"] @ changed_y - expected))),
                          "float32_point1_rounding_delta": float(canonicalize_targets([.1])["rounding_delta"][0])}, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
