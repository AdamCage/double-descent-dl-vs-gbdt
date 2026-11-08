"""Hand-built response evidence only: no data generator, model fit or study seed."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.plot_response_forecast import (
    FEATURES, MODELS, ResponseGallery, Study, delivery_spec, load_study,
    plt, render, validate_delivery,
)


def hand_evidence(*, seeds=(71, 72), prefixes=(4,), n_train=3, n_evaluation=4, invalid=False):
    """Fabricate prediction deltas directly; never call any model/data-generation API."""
    contexts, rows, directions, arrays = [], [], [], {}
    eps_values = (.01, .1)
    for seed_index, seed in enumerate(seeds):
        for teacher in ("linear", "interaction"):
            for prefix in prefixes:
                context_id = f"toy_{teacher}_s{seed}_p{prefix}"
                feature = {key: .1 + i*.01 for i, (key, _) in enumerate(FEATURES)}
                contexts.append({"id": context_id, "seed": seed, "teacher": teacher, "prefix": prefix,
                                 "base_features": feature, "base_clean_train_mse": .1,
                                 "base_clean_evaluation_mse": .2,
                                 "prefix_unique_thresholds": prefix, "n_train_signatures": n_train,
                                 "timings": {"base_fit_seconds": .05, "cheap_fit_seconds": .04, "full_fit_seconds": .3}})
                for amplitude, eps in enumerate(eps_values):
                    grouped = {"pilot": [], "response": []}
                    for phase, count in (("pilot", 2), ("response", 4)):
                        for direction in range(count):
                            direction_id = f"{context_id}_e{amplitude}_{phase}_d{direction}"
                            valid = not (invalid and not directions)
                            keys, metrics = {}, {}
                            for split, n in (("train", n_train), ("evaluation", n_evaluation)):
                                frozen = eps * np.arange(1, n+1) * (1 + direction*.1)
                                discrepancy = eps * np.resize([1., -.5], n) * (.2+seed_index*.02+direction*.01) * (1 if phase == "pilot" else 2)
                                if not valid:
                                    frozen, discrepancy = np.zeros(n), np.zeros(n)
                                full = frozen + discrepancy
                                keys[split] = {key: f"{direction_id}_{split}_{key}" for key in ("full_delta", "frozen_delta", "structural_delta")}
                                for key, value in zip(("full_delta", "frozen_delta", "structural_delta"), (full, frozen, discrepancy)):
                                    arrays[keys[split][key]] = value
                                denominator = eps**2 * n
                                metrics[split] = {"n_evaluation": n,
                                                  "normalized_full": float(full@full/denominator),
                                                  "normalized_frozen": float(frozen@frozen/denominator),
                                                  "normalized_structural_discrepancy": float(discrepancy@discrepancy/denominator),
                                                  "normalized_twice_cross": float(2*frozen@discrepancy/denominator)}
                            record = {"id": direction_id, "context_id": context_id, "seed": seed, "teacher": teacher,
                                      "prefix": prefix, "epsilon": eps, "phase": phase, "direction": direction,
                                      "valid": valid, "metrics": metrics, "array_keys": keys,
                                      "clean_train_mse": .12, "clean_evaluation_mse": .22,
                                      "timings": {"fit_seconds": .01, "prediction_seconds": .001, "io_seconds": .002,
                                                  "decomposition_seconds": .0005, "label_preparation_seconds": .0002}}
                            directions.append(record)
                            grouped[phase].append(record)
                    row = {"id": f"{context_id}_e{amplitude}", "context_id": context_id, "seed": seed,
                           "teacher": teacher, "prefix": prefix, "epsilon": eps, **feature,
                           "cheap_discrepancy": float(np.mean([r["metrics"]["evaluation"]["normalized_structural_discrepancy"] for r in grouped["pilot"]])),
                           "full_discrepancy": float(np.mean([r["metrics"]["evaluation"]["normalized_structural_discrepancy"] for r in grouped["response"]])),
                           "valid_directions": all(r["valid"] for group in grouped.values() for r in group)}
                    rows.append(row)
    forecast = {"status": "inconclusive_invalid_direction", "folds": []}
    if not invalid:
        for row in rows:
            row["target_u"] = float(np.log10(1+row["full_discrepancy"]))
            row["models"] = {}
            for i, model in enumerate(MODELS):
                prediction = row["target_u"] + (i-2)*.03
                residual = prediction - row["target_u"]
                row["models"][model] = {"prediction_u": prediction, "residual_u": residual,
                                         "absolute_error_u": abs(residual)}
        summaries = {}
        for model in MODELS:
            draws = {str(seed): float(np.mean([r["models"][model]["absolute_error_u"] for r in rows if r["seed"] == seed])) for seed in seeds}
            teachers = {teacher: float(np.mean([np.mean([r["models"][model]["absolute_error_u"] for r in rows if r["seed"] == seed and r["teacher"] == teacher]) for seed in seeds])) for teacher in ("linear", "interaction")}
            summaries[model] = {"mae_u": float(np.mean(list(draws.values()))), "draw_mae_u": draws, "teacher_mae_u": teachers}
        forecast = {"status": "complete_discovery_cross_validation", "summaries": summaries,
                    "folds": [{"held_out_seed": seed, "model": model,
                               "training_seeds": [s for s in seeds if s != seed]} for seed in seeds for model in MODELS]}
    analysis = {"n_contexts": len(contexts), "n_forecast_rows": len(rows), "n_directions": len(directions),
                "n_fits": len(contexts)+len(directions), "seeds": list(seeds),
                "teachers": ["linear", "interaction"], "prefix_budgets": list(prefixes),
                "epsilon_values": list(eps_values), "forecast": forecast,
                "invalid_zero_delta_direction_count": sum(not r["valid"] for r in directions),
                "timings": {"forecast_seconds": .002},
                "costs": {"cheap_fit_prediction_decomposition_seconds": 1.,
                          "full_fit_prediction_decomposition_seconds": 4.,
                          "common_base_fit_predictions_operator_features_seconds": 2.,
                          "extra_cheap_full_ratio": .25,
                          "secondary_end_to_end_ratio_common_base_in_both": 3.002/6,
                          "secondary_end_to_end_forecast_cost_over_full_refits": 3.002/4,
                          "leaf_evidence_seconds_separate": {"base": .001, "cheap": .002, "full": .004},
                          "io_seconds_separate": .2, "scope": "Hand-built test fixture, no measured scientific result"},
                "decision": {"P1_replay": True, "P2_forecast": "inconclusive_invalid_direction" if invalid else False,
                             "P3_cost": True, "range_gate": False, "cheap_full_wall_ratio": .25,
                             "go": False, "outcome": "inconclusive_lack_of_challenge"}}
    return Study(analysis, rows, contexts, directions, arrays)


def write_hand_evidence(directory: Path, study: Study):
    directory.mkdir()
    (directory / "analysis.json").write_text(json.dumps(study.analysis, allow_nan=False), encoding="utf-8")
    for name, rows in (("forecast_rows", study.rows), ("contexts", study.contexts), ("directions", study.directions)):
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows), encoding="utf-8")
    np.savez(directory / "predictions.npz", **study.arrays)


class RecordingGallery(ResponseGallery):
    """Exercise every panel and CSV path without writing dozens of raster files."""
    def save(self, fig, name, title, caption, group, selection="All declared observations"):
        self.figures.append({"id": name, "paths": {ext: f"figures/{name}.{ext}" for ext in ("png", "svg")}})
        plt.close(fig)


class ResponseForecastPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_registered_full_matrix_counts_both_outcomes(self):
        for invalid, expected_pairs, expected_tables in ((False, 113, 7), (True, 89, 6)):
            study = hand_evidence(seeds=(71, 72, 73, 74), prefixes=(4, 12, 28, 60), n_train=128, n_evaluation=256, invalid=invalid)
            spec = delivery_spec(study)
            self.assertEqual(spec["static_png_svg_pairs"], expected_pairs)
            self.assertEqual(spec["source_tables"], expected_tables)
            self.assertEqual(spec["table_rows"]["tables/r14_all_raw_responses.csv"], 147456)
            self.assertEqual(spec["table_rows"]["tables/r14_all_forecast_rows.csv"], 64)
            self.assertEqual(spec["table_rows"]["tables/r14_cost_scopes.csv"], 11)
            if not invalid:
                self.assertEqual(spec["table_rows"]["tables/r14_oof_fold_fits.csv"], 24)

    def test_complete_loader_and_raw_norm_validation(self):
        study = hand_evidence()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "study"; write_hand_evidence(source, study)
            loaded = load_study(ResponseGallery(root / "gallery"), source)
            self.assertEqual(len(loaded.directions), 48)
            changed = copy.deepcopy(study)
            changed.directions[0]["metrics"]["evaluation"]["normalized_twice_cross"] += .5
            other = root / "corrupt"; write_hand_evidence(other, changed)
            with self.assertRaisesRegex(ValueError, "metric differs"):
                load_study(ResponseGallery(root / "other-gallery"), other)

    def test_missing_direction_and_wrong_context_identity_fail(self):
        for mutation in ("missing", "identity"):
            study = hand_evidence()
            if mutation == "missing":
                study.directions.pop()
                study.analysis["n_directions"] -= 1
                study.analysis["n_fits"] -= 1
            else:
                study.rows[0]["teacher"] = "interaction"
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source = root / "study"; write_hand_evidence(source, study)
                with self.assertRaises(ValueError):
                    load_study(ResponseGallery(root / "gallery"), source)

    def test_incomplete_oof_cannot_be_disguised_as_inconclusive(self):
        study = hand_evidence(); study.rows[0]["models"].pop("B3")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "study"; write_hand_evidence(source, study)
            with self.assertRaisesRegex(ValueError, "all declared model"):
                load_study(ResponseGallery(root / "gallery"), source)

    def test_every_render_branch_matches_declared_delivery_and_keeps_sources(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source = root / "study"
                write_hand_evidence(source, hand_evidence(invalid=invalid))
                before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
                gallery = RecordingGallery(root / "gallery"); study = load_study(gallery, source)
                render(gallery, study)
                validate_delivery(gallery.figures, gallery.tables, gallery.expected_delivery)
                self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
                ids = {r["id"] for r in gallery.figures}
                self.assertEqual("r14_oof_unavailable" in ids, invalid)
                self.assertEqual("r14_oof_prediction_A" in ids, not invalid)
                if invalid:
                    self.assertNotIn("tables/r14_oof_fold_fits.csv", {r["path"] for r in gallery.tables})
                    raw_text = (gallery.output / "tables/r14_all_forecast_rows.csv").read_text()
                    self.assertNotIn("prediction_u", raw_text)

    def test_finish_relabels_r14_and_refuses_reusing_nonempty_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gallery"; gallery = ResponseGallery(path)
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            ax.plot([1, 2], [-1e-5, 2e-5]); ax.set_xlabel("Hand-built x"); ax.set_ylabel("Signed toy response")
            gallery.save(fig, "r14_fixture", "R14 layout fixture", "Hand-built fixture only, no research evidence.", "Fixture")
            gallery.csv("fixture", [{"a": 1}], "Fixture")
            gallery.expected_delivery = {"static_png_svg_pairs": 1, "interactive_html": 0, "figure_ids": ["r14_fixture"],
                                         "source_tables": 1, "table_rows": {"tables/fixture.csv": 1}, "oof_available": False}
            gallery.finish()
            page = (path / "index.html").read_text(encoding="utf-8")
            manifest = json.loads((path / "figure_manifest.json").read_text())
            self.assertIn("R14: finite-label response forecast discovery", page)
            self.assertNotIn("R10–R12 evidence gallery", page)
            self.assertEqual(manifest["study_id"], "R14")
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                ResponseGallery(path)


if __name__ == "__main__":
    unittest.main()
