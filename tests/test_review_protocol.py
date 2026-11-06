"""Протокол ревью 2026-11-04. Фикстуры не являются результатом по корпусу."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np

from src.cache import compute_hash
from src.dynamics import describe_change, replicate_change_points, segment_rss
from src.protocol import (
    choose_pca_dimensions,
    comparison_status,
    complexity_axis_grid,
    h1_verdict,
    h4_interior_peak,
    h4_verdict,
    held_out_threshold_report,
    normalized_time_closer,
    optimization_settled,
    select_comparison_index,
    select_operating_points,
    summarize_replicates,
)


def _legacy_hash(**kwargs: object) -> str:
    digest = hashlib.sha256()
    for item in sorted(kwargs.items()):
        digest.update(str(item).encode())
    return digest.hexdigest()[:16]


class CacheRecipeTest(unittest.TestCase):
    def test_compute_hash_matches_the_recipe_s03_already_stored(self) -> None:
        kwargs = dict(
            backend="routerai",
            model="qwen/qwen3-embedding-8b",
            batch_size=32,
            n_texts=10,
            first_text_hash="abcd1234",
            last_text_hash="ef567890",
        )
        self.assertEqual(compute_hash(**kwargs), _legacy_hash(**kwargs))
        self.assertEqual(len(compute_hash(**kwargs)), 16)


class PcaRuleTest(unittest.TestCase):
    def test_does_not_collapse_to_128_when_only_256_meets_the_target(self) -> None:
        cumulative = np.concatenate([
            np.linspace(0.01, 0.80, 128),
            np.linspace(0.81, 0.99, 128),
        ])
        decision = choose_pca_dimensions(cumulative, original_dim=4096, target=0.95)
        expected = int(np.searchsorted(cumulative, 0.95, side="left")) + 1
        self.assertGreater(expected, 128)
        self.assertEqual(decision["final_dims"], expected)
        self.assertNotEqual(decision["final_dims"], 128)
        self.assertEqual(decision["rule"], "smallest_k_meeting_variance_target")

    def test_keeps_original_dim_when_the_cap_misses_the_target(self) -> None:
        cumulative = np.linspace(0.20, 0.90, 256)
        decision = choose_pca_dimensions(cumulative, original_dim=4096, target=0.95)
        self.assertEqual(decision["final_dims"], 4096)
        self.assertEqual(decision["rule"], "variance_target_not_met_within_cap")

    def test_uses_the_earlier_crossing_instead_of_a_fixed_128(self) -> None:
        cumulative = np.concatenate([
            np.linspace(0.90, 0.96, 40),
            np.linspace(0.961, 0.999, 216),
        ])
        decision = choose_pca_dimensions(cumulative, original_dim=4096, target=0.95)
        expected = int(np.searchsorted(cumulative, 0.95, side="left")) + 1
        self.assertLess(expected, 128)
        self.assertEqual(decision["final_dims"], expected)


class ThresholdProtocolTest(unittest.TestCase):
    def test_thresholds_come_from_the_selection_split(self) -> None:
        y_selection = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        proba_selection = np.array([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95])
        y_evaluation = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        proba_evaluation = np.array([0.40, 0.42, 0.55, 0.58, 0.46, 0.60, 0.63, 0.90])
        selected = select_operating_points(y_selection, proba_selection)
        biased = select_operating_points(y_evaluation, proba_evaluation)
        report = held_out_threshold_report(
            y_selection, proba_selection, y_evaluation, proba_evaluation,
        )
        self.assertNotEqual(selected["f1_best"], biased["f1_best"])
        self.assertEqual(report["f1_best"]["threshold"], selected["f1_best"])
        self.assertEqual(report["threshold_source"], "validation")
        self.assertEqual(report["evaluation_split"], "test")
        self.assertEqual(report["selection_metrics"]["f1_best"]["threshold"], selected["f1_best"])

    def test_same_array_is_rejected(self) -> None:
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.7])
        with self.assertRaises(ValueError):
            held_out_threshold_report(labels, scores, labels, np.array([0.1, 0.2, 0.8, 0.9]))


class DynamicsTest(unittest.TestCase):
    def test_segment_rss_matches_a_straight_line(self) -> None:
        values = [0.2 * index + 1.0 for index in range(12)]
        self.assertAlmostEqual(segment_rss(values, 0, len(values)), 0.0, places=6)

    def test_late_drop_is_a_candidate_and_smooth_decay_is_not(self) -> None:
        late = [1.0] * 40 + list(np.linspace(1.0, 0.2, 40))
        found = describe_change(late)
        self.assertTrue(found["candidate"])
        self.assertGreaterEqual(found["t_star"], 30)
        self.assertLessEqual(found["t_star"], 50)

        smooth = list(np.exp(-np.linspace(0.0, 2.0, 80)))
        self.assertFalse(describe_change(smooth)["candidate"])
        self.assertFalse(describe_change([0.4] * 40)["candidate"])

    def test_one_run_is_not_a_replication(self) -> None:
        one = describe_change([1.0] * 40 + list(np.linspace(1.0, 0.2, 40)))
        summary = replicate_change_points([one])
        self.assertEqual(summary["inference"], "not_available")
        self.assertFalse(summary["interval_available"])

    def test_replication_requires_agreement(self) -> None:
        agreed = [
            {"candidate": True, "t_star": 40, "n": 80},
            {"candidate": True, "t_star": 42, "n": 80},
            {"candidate": True, "t_star": 38, "n": 80},
        ]
        disagreed = [
            {"candidate": True, "t_star": 16, "n": 80},
            {"candidate": True, "t_star": 40, "n": 80},
            {"candidate": True, "t_star": 64, "n": 80},
        ]
        self.assertEqual(replicate_change_points(agreed)["inference"], "reproduced")
        self.assertEqual(replicate_change_points(disagreed)["inference"], "not_reproduced")


class ComparisonProtocolTest(unittest.TestCase):
    def test_noise_regimes_must_match(self) -> None:
        self.assertEqual(comparison_status(0.0, 0.0), "same_noise_regime")
        self.assertEqual(comparison_status(0.1, 0.0), "blocked_noise_regime_mismatch")
        self.assertEqual(comparison_status(None, 0.0), "noise_regime_not_recorded")

    def test_selection_prefers_a_settled_width_over_the_widest(self) -> None:
        rows = [
            {"width": 8, "val_loss": 0.2, "optimization_settled": False},
            {"width": 32, "val_loss": 0.5, "optimization_settled": True},
            {"width": 128, "val_loss": 0.4, "optimization_settled": True},
        ]
        index, rule = select_comparison_index(rows)
        self.assertEqual(rows[index]["width"], 128)
        self.assertEqual(rule, "settled_validation_logloss")
        unsettled = [
            {"width": 8, "val_loss": 0.2, "optimization_settled": False},
            {"width": 128, "val_loss": 0.4, "optimization_settled": False},
        ]
        index, rule = select_comparison_index(unsettled)
        self.assertEqual(unsettled[index]["width"], 8)
        self.assertEqual(rule, "no_width_settled_best_validation_logloss")

    def test_convergence_flag_uses_the_last_window(self) -> None:
        self.assertTrue(optimization_settled([1.0, 0.9, 0.8, 0.7, 0.69]))
        self.assertFalse(optimization_settled([1.0, 0.8, 0.6, 0.4, 0.2]))

    def test_single_run_has_no_interval(self) -> None:
        summary = summarize_replicates([0.2])
        self.assertFalse(summary["interval_available"])
        repeated = summarize_replicates([0.2, 0.3, 0.25])
        self.assertEqual(repeated["interval_method"], "observed_min_max")
        self.assertNotIn("95", repeated["interval_method"])

    def test_axes_are_not_a_cartesian_product(self) -> None:
        reference = {
            "depth": 6,
            "border_count": 32,
            "iterations": 200,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
        }
        grid = complexity_axis_grid(
            reference, depths=[4, 6, 8], border_counts=[8, 32, 128], ensemble_sizes=[50, 200, 400],
        )
        for spec in grid:
            changed = []
            if spec["depth"] != 6:
                changed.append("depth")
            if spec["border_count"] != 32:
                changed.append("border")
            if spec["iterations"] != 200:
                changed.append("iterations")
            self.assertLessEqual(len(changed), 1)
        self.assertTrue(any(spec["axis"] == "depth" and spec["depth"] == 4 for spec in grid))
        self.assertFalse(any(spec["depth"] == 4 and spec["border_count"] == 8 for spec in grid))

    def test_hypothesis_labels_stay_closed(self) -> None:
        self.assertEqual(
            h1_verdict({"inference": "reproduced"}, {"inference": "not_reproduced"}),
            "supports",
        )
        self.assertEqual(
            h1_verdict({"inference": "not_reproduced"}, {"inference": "not_reproduced"}),
            "inconclusive",
        )
        self.assertEqual(
            h1_verdict({"inference": "reproduced"}, {"inference": "reproduced"}),
            "refutes",
        )
        self.assertFalse(h4_interior_peak([0.40, 0.20, 0.10]))
        self.assertTrue(h4_interior_peak([0.20, 0.40, 0.15]))
        self.assertEqual(h4_verdict([True, True, True]), "coarse_peak_not_double_descent")
        self.assertEqual(h4_verdict([False, False, False]), "not_observed")
        self.assertNotEqual(h4_verdict([True, True, True]), "supports")
        self.assertTrue(normalized_time_closer([100, 400], [0.2, 0.05]))
        self.assertIsNone(normalized_time_closer([100], [0.2]))


if __name__ == "__main__":
    unittest.main()
