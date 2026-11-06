"""Rules for double_descent_text_v1. The fixture is not the corpus study."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pack_fixtures
from episteme.domains.packs.double_descent_text_v1.model import (
    boosting_pattern, modelwise_pattern)


class DoubleDescentRuleTests(unittest.TestCase):
    def test_planted_modelwise_peak_is_support_and_a_monotone_curve_is_not(self):
        bundle = json.loads(pack_fixtures.double_descent_fixture_bytes())
        found = modelwise_pattern(bundle["modelwise"])
        self.assertEqual(found["verdict"], "supports")
        self.assertEqual(found["interpolation_index"], 1)
        self.assertAlmostEqual(found["peak_excess"], 0.5)
        monotone = [
            {"width": 8, "num_params": 20, "train_loss": 0.2, "val_loss": 0.9,
             "train_acc": 0.99, "val_acc": 0.4},
            {"width": 16, "num_params": 80, "train_loss": 0.1, "val_loss": 0.5,
             "train_acc": 1.0, "val_acc": 0.6},
            {"width": 32, "num_params": 200, "train_loss": 0.05, "val_loss": 0.4,
             "train_acc": 1.0, "val_acc": 0.7},
        ]
        self.assertEqual(modelwise_pattern(monotone)["verdict"], "refutes")
        edge = monotone[:2]
        self.assertEqual(modelwise_pattern(edge)["verdict"], "inconclusive")

    def test_planted_boosting_curve_supports_and_a_short_curve_refutes(self):
        bundle = json.loads(pack_fixtures.double_descent_fixture_bytes())
        found = boosting_pattern(bundle["catboost_main"]["train_logloss"],
                                 bundle["catboost_main"]["val_logloss"])
        self.assertEqual(found["verdict"], "supports")
        self.assertEqual(found["plateau_start_iteration"], 100)
        short = bundle["sweeps"]["learning_rate"][0]
        self.assertEqual(boosting_pattern(short["train_logloss"], short["val_logloss"])["verdict"],
                         "refutes")

    def test_runner_metric_matches_the_pack_rule(self):
        from episteme.domains import registry
        loaded = registry.load_pack("double_descent_text_v1")
        with TemporaryDirectory() as work:
            request = pack_fixtures._double_descent_request(Path(work) / "inputs")
            draft = loaded.hook("compile_protocol")(request)
            plan = loaded.hook("compile_execution")(request)
            slots, blobs = pack_fixtures.run_programs(plan, draft.roster, Path(work) / "run")
        primary = next(row for row in slots if row["mode"] == "primary")
        metrics = json.loads(blobs[primary["outputs"]["metrics"]])
        self.assertAlmostEqual(metrics["modelwise_peak_excess"], 0.5)
        raw = json.loads(blobs[primary["outputs"]["raw_data"]])
        self.assertEqual(raw["origin"], "synthetic_fixture")
        self.assertEqual(raw["catboost"]["verdict"], "supports")
        self.assertNotEqual(raw["origin"], "full_pipeline")
