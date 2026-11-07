"""Scoring checks for the R5 replication. Research seeds are not fit."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.r5_replicate import evaluate, load_json, validate_protocol


ROOT = Path(__file__).resolve().parents[1]


class R5ScoreTest(unittest.TestCase):
    def test_protocol_keeps_endpoints_and_unreplicated_alternatives(self) -> None:
        protocol = load_json(ROOT / "research/stages/R5/inputs/protocol.json")
        validate_protocol(protocol)
        self.assertFalse(set(protocol["discovery_data_seeds"]) & set(protocol["confirmation_data_seeds"]))
        text = " ".join(protocol["not_reestimated"])
        self.assertIn("40 epochs", text)
        self.assertIn("CatBoost", text)

    def test_directional_rule_on_a_fixture(self) -> None:
        protocol = load_json(ROOT / "research/stages/R5/inputs/protocol.json")
        cells = []
        for seed in protocol["confirmation_data_seeds"]:
            for column, interpolating, residual in ((150, False, 0.4), (200, True, 1e-12), (250, True, 1e-12)):
                for snr, risk in ((1.0, 2.0), (5.0, 0.5)):
                    cells.append({"phase": "confirmation", "family": "linear", "data_seed": seed,
                                  "snr": snr, "column": column, "clean_mse": risk, "interpolating": interpolating,
                                  "residual_relative": residual})
            cells.append({"phase": "confirmation", "family": "tree_leaves", "data_seed": seed, "value": 2,
                          "residual_relative": 0.8, "clean_mse": 0.5})
            cells.append({"phase": "confirmation", "family": "tree_leaves", "data_seed": seed, "value": 256,
                          "residual_relative": 0.0, "clean_mse": 1.5})
            cells.append({"phase": "confirmation", "family": "forest_size_restricted", "data_seed": seed,
                          "value": 1, "clean_mse": 1.2})
            cells.append({"phase": "confirmation", "family": "forest_size_restricted", "data_seed": seed,
                          "value": 50, "clean_mse": 0.4})
        scored = evaluate(cells, protocol, "confirmation")
        self.assertEqual(scored["noise"]["outcome"], "supported")
        self.assertEqual(scored["leaf_residual"]["outcome"], "supported")
        self.assertEqual(scored["leaf_clean_risk"]["outcome"], "supported")
        self.assertEqual(scored["forest_clean_risk"]["outcome"], "supported")
        self.assertEqual(scored["label"], "confirmation")


if __name__ == "__main__":
    unittest.main()
