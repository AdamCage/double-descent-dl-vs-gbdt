"""Engineering checks for the R9 bracket rule and a tiny CatBoost fit.

The tiny fit does not use seeds 7101, 7102, or 9101-9105.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.r9_catboost import load_json, run_discovery, summarize_seed, validate_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "stages" / "R9" / "inputs" / "protocol.json"


class BracketRuleTest(unittest.TestCase):
    def test_false_then_true_is_bracketed(self) -> None:
        cells = [
            {"iterations": 1, "interpolating": False, "clean_mse": 0.2, "data_seed": 1},
            {"iterations": 4, "interpolating": True, "clean_mse": 0.5, "data_seed": 1},
        ]
        summary = summarize_seed(cells)
        self.assertEqual(summary["bracket_status"], "bracketed")
        self.assertEqual(summary["first_interpolating_iteration"], 4)
        self.assertEqual(summary["max_clean_mse_iteration"], 4)
        self.assertIs(summary["same_iteration"], True)

    def test_no_interpolation_has_null_comparison(self) -> None:
        cells = [
            {"iterations": 1, "interpolating": False, "clean_mse": 0.4, "data_seed": 3},
            {"iterations": 2, "interpolating": False, "clean_mse": 0.4, "data_seed": 3},
        ]
        summary = summarize_seed(cells)
        self.assertEqual(summary["bracket_status"], "not_identifiable")
        self.assertIsNone(summary["same_iteration"])
        self.assertEqual(summary["max_clean_mse_iteration"], 1)

    def test_frozen_protocol_file(self) -> None:
        protocol = load_json(PROTOCOL)
        validate_protocol(protocol)
        self.assertEqual(protocol["iterations"], [1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(protocol["discovery_data_seeds"], [7101, 7102])
        self.assertNotIn(7101, protocol["confirmation_data_seeds"])


class TinyFitTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("RESEARCH_MODEL_SMOKE") == "1", "explicit research preflight only")
    def test_two_cells_do_not_declare_double_descent(self) -> None:
        protocol = json.loads(json.dumps(load_json(PROTOCOL)))
        protocol["protocol_id"] = "R9-tiny-engineering"
        protocol["n_train"] = 24
        protocol["n_eval"] = 40
        protocol["p_max"] = 12
        protocol["input_columns"] = 4
        protocol["discovery_data_seeds"] = [3]
        protocol["confirmation_data_seeds"] = [4]
        protocol["iterations"] = [1, 2]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            analysis = run_discovery(protocol, out, PROTOCOL)
            self.assertEqual(analysis["models_fit"], 2)
            self.assertFalse(analysis["double_descent_declared"])
            self.assertFalse(analysis["used_clean_risk_to_choose_grid"])
            self.assertEqual(analysis["iteration_grid"], [1, 2])
            audit = json.loads((out / "seed-audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["data_seeds"], [3])
            self.assertTrue(audit["confirmation_seeds_absent"])
            names = sorted(path.name for path in out.iterdir())
            self.assertEqual(names, ["analysis.json", "cells.jsonl", "seed-audit.json"])
            lines = (out / "cells.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            residual = json.loads(lines[0])["residual_relative"]
            self.assertGreater(residual, 0.0)


if __name__ == "__main__":
    unittest.main()
