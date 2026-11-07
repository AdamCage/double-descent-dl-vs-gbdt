"""CPU checks for the R3 MLP protocol. Research seeds are not fitted."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.r3_mlp import load_json, reduce_menu, train_mlp, validate_protocol


ROOT = Path(__file__).resolve().parents[1]


class R3ProtocolTest(unittest.TestCase):
    def test_protocol_locks_budgets_and_seeds(self) -> None:
        protocol = load_json(ROOT / "research/stages/R3/inputs/protocol.json")
        validate_protocol(protocol)
        self.assertEqual(protocol["relative_train_residual_tolerance"], 0.001)
        self.assertEqual(protocol["discovery_data_seeds"], [3101, 3102])
        self.assertEqual(protocol["confirmation_data_seeds"], [4101, 4102, 4103, 4104, 4105])
        self.assertFalse(set(protocol["discovery_data_seeds"]) & set(protocol["confirmation_data_seeds"]))
        self.assertEqual(protocol["device"], "cpu")
        kept = reduce_menu(2.0, 2, 5, protocol)
        self.assertFalse(kept["used_clean_risk"])
        self.assertEqual(kept["kept_widths"], protocol["widths"])
        fallback = reduce_menu(10000.0, 2, 5, protocol)
        self.assertEqual(fallback["kept_widths"], protocol["reduction_rule"]["fallback_widths"])

    def test_short_budget_trains_on_cpu_and_can_interpolate(self) -> None:
        protocol = load_json(ROOT / "research/stages/R3/inputs/protocol.json")
        rng = np.random.default_rng(0)
        features = rng.standard_normal((200, 20))
        beta = rng.standard_normal(20)
        beta /= np.linalg.norm(beta)
        response = features @ beta + rng.standard_normal(200)
        short = train_mlp(features, response, 4, "epochs_40", 0.0, protocol, 11)
        self.assertEqual(short["epochs_ran"], 40)
        self.assertEqual(short["stop_reason"], "budget")
        self.assertFalse(short["interpolating"])
        long = train_mlp(features, response, 256, "until_residual", 0.0, protocol, 11)
        self.assertTrue(long["interpolating"])
        self.assertLessEqual(long["residual_relative"], 0.001)
        self.assertLess(long["epochs_ran"], 500)
        self.assertEqual(long["stop_reason"], "residual_tolerance")


if __name__ == "__main__":
    unittest.main()
