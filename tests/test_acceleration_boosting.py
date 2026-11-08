"""Small numerical and continuation checks; never run the scientific grid."""

from __future__ import annotations

import copy
import json
import os
import unittest
from pathlib import Path

import numpy as np

from src.acceleration_boosting import (
    array_hash, boundary_contrast, fit_schedule, paired_summary, prepare_dataset,
    raw_trajectories, trajectory_metrics, validate_protocol,
)


class AccelerationBoostingTests(unittest.TestCase):
    def test_stable_logloss_errors_and_gradients(self):
        raw = np.asarray([[0.0, 0.0], [-1000.0, 1000.0], [1000.0, -1000.0]])
        result = trajectory_metrics(raw, np.asarray([0, 1]))
        np.testing.assert_allclose(result["logloss"], [np.log(2), 0.0, 1000.0])
        self.assertEqual(result["errors"], [1, 0, 2])
        self.assertEqual(result["min_margin"], [0.0, 1000.0, -1000.0])
        self.assertTrue(np.isfinite(result["gradient_rms"]).all())

    def test_known_gain_and_clock_normalization(self):
        # A change of rate alone: linear loss in the cumulative-LR clock.
        clock = np.r_[0.0, np.cumsum([0.1] * 8 + [0.4] * 8)]
        loss = 2.0 - 0.2 * clock
        tree = boundary_contrast(loss, 8, 5)
        normalized = boundary_contrast(loss, 8, 5, clock)
        self.assertAlmostEqual(tree["velocity_gain"], 0.06)
        self.assertAlmostEqual(normalized["velocity_gain"], 0.0)
        self.assertEqual(tree["before_slice"], [4, 9])
        self.assertEqual(tree["after_slice"], [8, 13])

    def test_invalid_clock_and_short_window_rejected(self):
        with self.assertRaises(ValueError):
            boundary_contrast([1, 2, 3, 4], 2, 3)
        with self.assertRaises(ValueError):
            boundary_contrast(np.arange(10), 5, 3, np.ones(10))

    def test_inference_explicitly_forwards_thread_cap(self):
        calls = []

        class Model:
            tree_count_ = 1

            def get_scale_and_bias(self):
                return 1., 0.

            def staged_predict(self, rows, **kwargs):
                calls.append(("staged", kwargs["thread_count"]))
                yield np.zeros(len(rows))

            def predict(self, rows, **kwargs):
                calls.append(("predict", kwargs["thread_count"]))
                return np.zeros(len(rows))

        data = {"x": np.zeros((4, 2)), "y": np.array([0, 1, 0, 1]),
                "indices": {key: np.arange(4) for key in ("train", "validation", "test")}}
        raw_trajectories(Model(), data, threads=2)
        self.assertEqual(calls, [("staged", 2), ("predict", 2)] * 3)
        with self.assertRaises(ValueError):
            raw_trajectories(Model(), data, threads=5)

    def test_pair_summary_keeps_observations_and_widens_for_family(self):
        values = [0.5, 1.0, 1.5, 2.0]
        result = paired_summary(values)
        self.assertEqual(result["values"], values)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["mean"], 1.25)
        self.assertLess(result["ci_family_bonferroni"][0], result["ci95_student_t"][0])
        self.assertGreater(result["ci_family_bonferroni"][1], result["ci95_student_t"][1])

    def test_dataset_rows_and_seeds_are_reproducible(self):
        data = prepare_dataset("synthetic", 23, synthetic_n=48)
        repeated = prepare_dataset("synthetic", 23, synthetic_n=48)
        self.assertEqual(data["hashes"], repeated["hashes"])
        self.assertEqual(data["seed_channels"], {"data": 23, "split": 100023, "model": 200023})
        parts = list(data["indices"].values())
        self.assertEqual([len(part) for part in parts], [16, 16, 16])
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(48))
        self.assertNotEqual(array_hash(data["x"]), array_hash(data["x"][:, ::-1]))

    def test_protocol_rejects_unequal_integrated_rates(self):
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root / "research/stages/R11/inputs/protocol.json").read_text(encoding="utf-8"))
        validate_protocol(protocol)
        changed = copy.deepcopy(protocol)
        changed["schedules"]["constant"] = [0.12, 0.12]
        with self.assertRaisesRegex(ValueError, "Integrated"):
            validate_protocol(changed)

    @unittest.skipUnless(os.environ.get("RESEARCH_MODEL_SMOKE") == "1", "explicit research preflight only")
    def test_tiny_two_stage_model_preserves_bias_and_stages(self):
        data = prepare_dataset("synthetic", 29, synthetic_n=48)
        model = fit_schedule(data, [0.03, 0.1], trees_per_segment=3, threads=1)
        raw, baseline = raw_trajectories(model, data, threads=1)
        self.assertEqual(model.tree_count_, 6)
        self.assertAlmostEqual(baseline["train_prevalence"], data["y"][data["indices"]["train"]].mean())
        for split in ("train", "validation", "test"):
            self.assertEqual(raw[split].shape, (7, 16))
            np.testing.assert_array_equal(raw[split][0], np.full(16, baseline["base_margin"]))

    @unittest.skipUnless(os.environ.get("RESEARCH_MODEL_SMOKE") == "1", "explicit research preflight only")
    def test_unbalanced_continuation_retains_bias_and_prediction_prefix(self):
        data = prepare_dataset("synthetic", 31, synthetic_n=48)
        train = data["indices"]["train"]
        data["y"][train] = np.r_[np.ones(5, dtype=int), np.zeros(11, dtype=int)]
        initial, final = fit_schedule(data, [.03, .1], trees_per_segment=3,
                                      threads=1, return_initial=True)
        expected_bias = np.log(5 / 11)
        self.assertAlmostEqual(initial.get_scale_and_bias()[1], expected_bias, places=5)
        self.assertAlmostEqual(final.get_scale_and_bias()[1], expected_bias, places=5)
        for indices in data["indices"].values():
            initial_raw = initial.predict(data["x"][indices], prediction_type="RawFormulaVal", thread_count=1)
            prefix_raw = final.predict(data["x"][indices], prediction_type="RawFormulaVal", ntree_end=3, thread_count=1)
            np.testing.assert_allclose(initial_raw, prefix_raw, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
