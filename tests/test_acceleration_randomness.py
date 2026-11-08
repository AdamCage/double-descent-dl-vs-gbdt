"""Pure R12 diagnostic checks; fitting is an explicit opt-in tiny smoke only."""

from __future__ import annotations

import copy
import json
import os
import unittest
from pathlib import Path

import numpy as np

from src.acceleration_boosting import prepare_dataset, raw_trajectories
from src.acceleration_randomness import (
    describe_trajectory, first_zero_errors, fit_random_strength,
    summarize_cells, validate_protocol,
)


def protocol_fixture():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "research/stages/R12/inputs/protocol.json").read_text(encoding="utf-8"))


def cells_fixture(protocol, *, gain=0.002, drop=0.02):
    cells = []
    for dataset in protocol["datasets"]:
        for seed in protocol["seeds"]:
            for strength in protocol["random_strengths"]:
                cells.append({"dataset": dataset, "seed": seed, "random_strength": strength,
                              "diagnostics": {"train": {"scan": {"score": 0.01 + 0.01 * strength}}},
                              "boundary_contrast": {"train": {"velocity_gain": gain}},
                              "fixed_after_drop": {"train": drop}})
    return cells


class AccelerationRandomnessTests(unittest.TestCase):
    def test_registered_grid_cannot_be_adapted(self):
        protocol = protocol_fixture()
        validate_protocol(protocol)
        for key, value in (("iterations", 2048), ("random_strengths", [0, 1, 20]),
                           ("boundary", 600), ("natural_gate_min_drop", 0.005)):
            altered = copy.deepcopy(protocol)
            altered[key] = value
            with self.assertRaisesRegex(ValueError, key):
                validate_protocol(altered)

    def test_exact_zero_errors_are_distinct_from_loss(self):
        self.assertEqual(first_zero_errors([4, 2, 1, 0, 0]), 3)
        self.assertIsNone(first_zero_errors([4, 3, 1]))
        with self.assertRaises(ValueError):
            first_zero_errors([2, True])

    def test_scan_threshold_is_descriptive_only(self):
        t = np.arange(80)
        loss = 1 - 0.001 * t - 0.008 * np.maximum(t - 55, 0)
        diagnostic = describe_trajectory(loss.tolist(), [4, 8], threshold=0.001)
        self.assertTrue(diagnostic["scan"]["descriptive_candidate"])
        self.assertIn("descriptive only", diagnostic["threshold_scope"])
        uncalibrated = describe_trajectory(loss.tolist(), [4, 8])
        self.assertIsNone(uncalibrated["scan"]["descriptive_candidate"])
        self.assertGreater(diagnostic["scan"]["score"], 0)

    def test_full_signed_contrasts_and_gate_are_preserved(self):
        protocol = protocol_fixture()
        result = summarize_cells(cells_fixture(protocol), protocol)
        self.assertEqual(len(result["scan_score_contrasts"]), 4)
        self.assertEqual(len(result["fixed_boundary_gain_contrasts"]), 4)
        self.assertEqual(len(result["natural_acceleration_gates"]), 6)
        rs10 = [row for row in result["scan_score_contrasts"] if row["random_strength_minus_zero"] == 10]
        self.assertTrue(all(row["outcome"] == "supported" for row in rs10))
        self.assertTrue(all(len(row["values"]) == 12 for row in rs10))
        for row in result["natural_acceleration_gates"]:
            self.assertEqual(row["outcome"], "fixed_late_acceleration_established")
            self.assertEqual(row["positive_gain_seed_count"], 12)
            self.assertEqual(row["mean_gain"]["family_comparisons"], 12)
            self.assertEqual(row["mean_after_drop"]["family_comparisons"], 12)

    def test_nonpractical_drop_and_negative_gain_do_not_establish_event(self):
        protocol = protocol_fixture()
        for gain, drop in ((-0.002, 0.02), (0.002, 0.005)):
            result = summarize_cells(cells_fixture(protocol, gain=gain, drop=drop), protocol)
            self.assertTrue(all(row["outcome"] == "not_established" for row in result["natural_acceleration_gates"]))

    def test_repeated_or_missing_cells_rejected(self):
        protocol = protocol_fixture()
        cells = cells_fixture(protocol)
        with self.assertRaisesRegex(ValueError, "Incomplete or duplicate"):
            summarize_cells(cells[:-1], protocol)
        with self.assertRaisesRegex(ValueError, "Incomplete or duplicate"):
            summarize_cells(cells + [cells[0]], protocol)

    @unittest.skipUnless(os.environ.get("RESEARCH_MODEL_SMOKE") == "1", "Explicit tiny model-fit smoke only")
    def test_optin_tiny_continuous_fit(self):
        data = prepare_dataset("synthetic", 37, synthetic_n=48)
        model = fit_random_strength(data, 0.0, iterations=4, learning_rate=0.05, threads=1)
        raw, baseline = raw_trajectories(model, data, threads=1)
        self.assertEqual(model.tree_count_, 4)
        self.assertEqual(len(model.get_tree_leaf_counts()), 4)
        self.assertEqual(raw["train"].shape, (5, 16))
        self.assertTrue(np.isfinite(raw["test"]).all())
        self.assertIn("base_margin", baseline)


if __name__ == "__main__":
    unittest.main()
