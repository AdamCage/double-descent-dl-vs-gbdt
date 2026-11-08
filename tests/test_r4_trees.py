"""Axis-separation checks for R4. Research seeds are not used."""

from __future__ import annotations

import unittest
import os
from pathlib import Path

import numpy as np

from src.r4_trees import jobs_for, load_json, reduce_menu, validate_protocol
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]


class R4AxisTest(unittest.TestCase):
    def test_grids_do_not_share_one_axis(self) -> None:
        protocol = load_json(ROOT / "research/stages/R4/inputs/protocol.json")
        validate_protocol(protocol)
        jobs = jobs_for(protocol, reduced=False)
        grids = {job["grid"] for job in jobs}
        self.assertEqual(len(grids), 6)
        varied = {job["grid"]: job["varied"] for job in jobs}
        self.assertEqual(varied["tree_leaves"], "max_leaf_nodes")
        self.assertEqual(varied["forest_size_restricted"], "n_estimators")
        self.assertEqual(varied["gbdt_steps"], "n_estimators")
        self.assertEqual(varied["gbdt_depth"], "max_depth")
        self.assertEqual(varied["gbdt_split_fraction"], "max_features")
        self.assertNotEqual(
            protocol["grids"]["forest_size_restricted"]["max_leaf_nodes"],
            protocol["grids"]["forest_size_large"]["max_leaf_nodes"],
        )
        self.assertFalse(set(protocol["discovery_data_seeds"]) & set(protocol["confirmation_data_seeds"]))
        self.assertFalse(reduce_menu(1.0, 2, 5, protocol)["used_clean_risk"])

    @unittest.skipUnless(os.environ.get("RESEARCH_MODEL_SMOKE") == "1", "explicit research preflight only")
    def test_a_large_leaf_cap_can_interpolate_outside_research_seeds(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.standard_normal((40, 4))
        response = features @ np.array([1.0, -0.5, 0.25, 0.1]) + rng.normal(scale=0.1, size=40)
        small = DecisionTreeRegressor(max_leaf_nodes=2, random_state=0).fit(features, response)
        large = DecisionTreeRegressor(max_leaf_nodes=40, random_state=0).fit(features, response)
        small_residual = np.linalg.norm(small.predict(features) - response) / np.linalg.norm(response)
        large_residual = np.linalg.norm(large.predict(features) - response) / np.linalg.norm(response)
        self.assertGreater(small_residual, 1e-8)
        self.assertLessEqual(large_residual, 1e-8)


if __name__ == "__main__":
    unittest.main()
