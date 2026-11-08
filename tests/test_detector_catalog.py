"""Engineering checks for the R8 formulas and the local-window rule.

These tests do not score research/stages/R8/inputs/catalog.json with either
detector. The stage task is the first registered score of that catalog.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.detector_catalog import exponential_decay, local_velocity_change, materialize, piecewise_linear, score_catalog


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "research" / "stages" / "R8" / "inputs" / "catalog.json"


class FormulaTest(unittest.TestCase):
    def test_piecewise_hits_the_knots(self) -> None:
        values = piecewise_linear(4, [[0, 1.0], [2, 1.0], [3, 0.0]])
        self.assertEqual(values, [1.0, 1.0, 1.0, 0.0])

    def test_catalog_file_matches_its_formulas(self) -> None:
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        self.assertEqual(catalog["length"], 100)
        self.assertEqual(catalog["local_window"], 5)
        self.assertEqual(catalog["global_min_fraction"], 0.2)
        ids = [item["id"] for item in catalog["series"]]
        self.assertEqual(ids, [
            "fast_plateau_moderate", "smooth_exponential", "late_tail",
            "single_late_drop", "two_drops",
        ])
        roles = {item["id"]: item["claim_role"] for item in catalog["series"]}
        self.assertEqual(roles["two_drops"], "excluded_multiple")
        fast = materialize(catalog["series"][0], 100)
        self.assertAlmostEqual(fast[0], 1.0)
        self.assertAlmostEqual(fast[29], 0.42)
        self.assertAlmostEqual(fast[70], 0.42)
        self.assertAlmostEqual(fast[50], 0.42)
        self.assertAlmostEqual(fast[99], 0.275)
        smooth = materialize(catalog["series"][1], 100)
        self.assertEqual(smooth, exponential_decay(100, 0.04))

    def test_local_window_finds_a_toy_drop(self) -> None:
        values = piecewise_linear(20, [[0, 1.0], [12, 1.0], [19, 0.0]])
        found = local_velocity_change(values, window=3, rss_improvement_min=0.05)
        self.assertTrue(found["candidate"])
        self.assertEqual(found["t_star"], 12)
        self.assertEqual(found["window"], 3)

    def test_toy_catalog_records_no_fit(self) -> None:
        catalog = {
            "schema_version": 1,
            "catalog_id": "toy",
            "length": 20,
            "local_window": 3,
            "global_min_fraction": 0.2,
            "rss_improvement_min": 0.05,
            "exclusions": ["toy"],
            "series": [{
                "id": "toy_drop",
                "formula": "piecewise_linear",
                "claim_role": "toy",
                "knots": [[0, 1.0], [12, 1.0], [19, 0.0]],
            }],
        }
        scored = score_catalog(catalog)
        self.assertEqual(scored["models_fit"], 0)
        self.assertFalse(scored["mechanism_claim"])
        self.assertEqual(scored["series"][0]["id"], "toy_drop")
        self.assertIn("candidate", scored["series"][0]["global"])
        self.assertIn("candidate", scored["series"][0]["local"])


if __name__ == "__main__":
    unittest.main()
