"""Pure handmade summaries; no released transcript or study outcomes are opened."""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

import matplotlib.pyplot as plt
import numpy as np

from src.cot_figures import (AuditFigures, COUNTS, DELTAS, FigureError, Gallery, GAPS, NOTE, PAPER_IDS,
                             INTAKE_COLUMNS, METRICS, OMITTED, QUALITY, SPLITS,
                             _matrix, delivery_spec, load_sources, paper_example_cells, render,
                             render_cells, render_paper_summaries, render_pairs, source_tables)


def fixtures(empty=False):
    models = [f"handmade/model-{i}" for i in range(8)]
    datasets = ["bbh", "gpqa", "mmlu"]
    settings = {"models": models, "datasets": datasets, "split_seed": "handmade-only",
                "bootstrap_replicates": 5000, "bootstrap_seed": {"discovery": 16016, "confirmation": 17017},
                "min_clusters": {"discovery": 50, "confirmation": 100},
                "split_modulus": 20, "discovery_buckets": list(range(6)), "confirmation_buckets": list(range(6, 13)),
                "reserve_buckets": list(range(13, 20)), "min_joint_finite_bootstrap_fraction": .99,
                "metadata_conflict_policy": "quarantine_entire_baseline_union_cluster_before_outcomes",
                "metadata_audit_sha256": "d" * 64,
                "context_definition": "scope_only",
                "population": "Handmade-only common-context question population with pre-outcome metadata quarantine",
                "min_overlap_coverage": .25, "practical_effect": .05, "tie_band": .02}
    analyses, quality = {}, {}
    for split in SPLITS:
        cells = []
        for dataset in datasets:
            for ai, bi in combinations(range(len(models)), 2):
                gap = (ai - bi) * .01
                estimates = dict(zip(METRICS, (gap, gap + .02, gap + .08, .08, .02, .06)))
                intervals = {m: {"low": estimates[m] - .04, "high": estimates[m] + .04,
                                  "se": .01, "critical": 4., "simultaneous_family": m,
                                  "status": "simultaneous_95_max_t", "joint_finite_fraction": 1.,
                                  "replicates_used": 5000, "family_size": 84} for m in DELTAS}
                counts = {k: 500 for k in COUNTS}
                counts.update(u0_questions=1000, u1_questions=500, u0_clusters=1000, u1_clusters=500,
                              a_observed_versions=1200, b_observed_versions=1000, common_exact_versions=800,
                              overlap_question_coverage=.5)
                if empty:
                    counts = {k: 0 for k in counts}
                    estimates = {k: None for k in METRICS}
                    for interval in intervals.values():
                        interval.update(low=None, high=None, se=None, critical=None, status="insufficient_support")
                cells.append({"cell_id": f"{dataset}|{models[ai]}|{models[bi]}", "dataset": dataset,
                              "model_a": models[ai], "model_b": models[bi], "counts": counts,
                              "estimates": estimates, "intervals": intervals, "eligible": not empty,
                              "support_eligible": not empty, "material_effect": not empty,
                              "ineligible_reasons": ["insufficient_overlap_clusters"] if empty else [],
                              "decomposition_closure_error": None if empty else 0.})
        analyses[split] = {"schema_version": 1, "split": split,
                           "context_definition": "scope_only",
                           "estimand": "question-weighted released acknowledgment proxy", "models": models,
                           "datasets": datasets, "settings": deepcopy(settings), "cells": cells,
                           "input_binding": {"slim_sha256": "a" * 64, "spec_sha256": "b" * 64}, "global": {}}
        recalc(analyses[split])
        quality[split] = {"split": split, "selected_questions": 0 if empty else 1000,
                          "context_definition": "scope_only",
                          "selected_clusters": 0 if empty else 1000, "bootstrap_clusters": 0 if empty else 2000,
                          "unselected_outcomes_used": False, "parquet_io_may_decode_unselected_batch_rows": True,
                          "metadata_integrity": {"conflict_ledger": {"policy": settings["metadata_conflict_policy"],
                              "excluded_cluster_keys": [], "excluded_cluster_count": 0,
                              "acknowledgment_or_generated_answer_values_used": False,
                              "original_cluster_keys_and_bucket_assignments_preserved": True}},
                          "ack_false_may_include_source_parse_failure": True,
                          "models": {f"{d}|{m}": {k: 0 if empty else 100 for k in QUALITY} for d in datasets for m in models}}
    return analyses, quality


def recalc(analysis):
    cells = analysis["cells"]
    support = [c for c in cells if c["support_eligible"]]
    effects = [c for c in cells if c["material_effect"]]
    sensitivity = len(effects) >= 2 and len({c["dataset"] for c in effects}) >= 2
    analysis["global"] = {"eligible_cells": sum(c["eligible"] for c in cells),
                          "support_eligible_cells": len(support), "material_effect_cells": len(effects),
                          "hypothesis": "not_evaluated_on_discovery" if analysis["split"] == "discovery" else "supported" if sensitivity else "not_supported",
                          "sensitivity_rule_pass": sensitivity, "robustness": "inconclusive",
                          "discovery_go": len(support) >= 4 and len({c["dataset"] for c in support}) >= 2}


def intake_fixture():
    return {"rows": 12, "source_files": 2, "source_profiles": [{"source_file": "data/toy-a.parquet", "rows": 5},
            {"source_file": "data/toy-b.parquet", "rows": 7}], "retained_columns": [k for k in INTAKE_COLUMNS if k not in OMITTED],
            "hashed_text_columns": list(OMITTED), "nulls": {k: 0 for k in INTAKE_COLUMNS},
            "slim": {"bytes": 1234, "sha256": "a" * 64}}


class RecordingGallery:
    """Exercise every renderer without writing hundreds of fixture images."""
    def __init__(self):
        self.figures, self.tables, self.rows, self.notes, self.expected = [], [], {}, [], None

    def save(self, fig, identifier, title, caption, group):
        self.figures.append({"id": identifier, "title": title, "caption": caption, "group": group,
                             "annotations": [text.get_text() for ax in fig.axes for text in ax.texts]})
        if identifier in PAPER_IDS:
            self.figures[-1].update(axis_lines=[[list(line.get_xdata()) for line in ax.lines] for ax in fig.axes],
                scatter_points=sum(len(collection.get_offsets()) for ax in fig.axes for collection in ax.collections),
                containers=[[type(container).__name__ for container in ax.containers] for ax in fig.axes],
                axis_labels=[(ax.get_xlabel(), ax.get_ylabel()) for ax in fig.axes],
                ytick_labels=[[tick.get_text() for tick in ax.get_yticklabels()] for ax in fig.axes])
        plt.close(fig)

    def csv(self, identifier, rows, description):
        self.tables.append({"id": identifier, "rows": len(rows)}); self.rows[identifier] = rows

    def finish(self):
        if [r["id"] for r in self.figures] != self.expected["figure_ids"]:
            raise AssertionError("Incomplete renderer matrix")
        if {r["id"]: r["rows"] for r in self.tables} != self.expected["tables"]:
            raise AssertionError("Incomplete source table matrix")


class GeometryGallery(RecordingGallery):
    """Measure actual Agg text/axes geometry for specified handmade figures."""
    def __init__(self, watched):
        super().__init__(); self.watched = set(watched); self.geometry = {}

    def save(self, fig, identifier, title, caption, group):
        if identifier in self.watched:
            # Include the production title/footer before constrained layout runs.
            fig.suptitle(title, fontsize=14, fontweight="bold")
            fig.text(.01, -.025, textwrap.fill(caption + " " + NOTE, 145), fontsize=8, ha="left", va="top")
            fig.canvas.draw(); renderer = fig.canvas.get_renderer()
            self.geometry[identifier] = [{
                "axes": ax.get_window_extent(renderer).frozen(),
                "annotations": [{"text": text.get_text(), "position": text.get_position(),
                                  "bbox": text.get_window_extent(renderer).frozen()} for text in ax.texts],
                "x_ticklabels": [text.get_window_extent(renderer).frozen() for text in ax.get_xticklabels() if text.get_visible()],
                "marks": [artist.get_window_extent(renderer).frozen() for artist in list(ax.lines) + list(ax.patches)
                          # Empty Line2D objects are legend handles, with no drawn data marks.
                          if (not hasattr(artist, "get_xdata") or len(artist.get_xdata()))
                          and np.isfinite(artist.get_window_extent(renderer).extents).all()],
            } for ax in fig.axes]
        super().save(fig, identifier, title, caption, group)


class CotFiguresTests(unittest.TestCase):
    def assert_annotation_geometry(self, identifier, panels):
        for panel_index, panel in enumerate(panels):
            axes = panel["axes"]
            for label in panel["annotations"]:
                box = label["bbox"]; message = f"{identifier} panel {panel_index}: {label['text']} at {label['position']}"
                self.assertGreaterEqual(box.x0, axes.x0 - .5, message)
                self.assertLessEqual(box.x1, axes.x1 + .5, message)
                self.assertGreaterEqual(box.y0, axes.y0 - .5, message)
                self.assertLessEqual(box.y1, axes.y1 + .5, message)
                self.assertFalse(any(box.overlaps(tick) for tick in panel["x_ticklabels"]), message + " collides with an x tick label")
            for box in panel["marks"]:
                self.assertGreaterEqual(box.y0, axes.y0 - .5, f"{identifier} panel {panel_index}: marker/bar below axes")
                self.assertLessEqual(box.y1, axes.y1 + .5, f"{identifier} panel {panel_index}: marker/bar above axes")

    def test_complete_matrix_has_111_figures_and_eight_source_tables(self):
        a, q = fixtures(); study = AuditFigures(a, q, intake_fixture()); gallery = RecordingGallery()
        render(gallery, study)
        self.assertEqual(len(gallery.figures), 111)
        self.assertEqual(len(set(r["id"] for r in gallery.figures)), 111)
        self.assertEqual(gallery.expected["tables"], {"all_cells": 168, "all_estimates": 1008,
            "all_intervals": 504, "model_legend": 8, "dataquality": 48, "split_summary": 2,
            "cross_split": 84, "intake_columns": 24})
        self.assertEqual(len(gallery.tables), 8)
        self.assertTrue(all("Every model pair" == r["group"] for r in gallery.figures[36:92]))
        self.assertEqual([row["id"] for row in gallery.figures[-4:]], list(PAPER_IDS))

    def test_all_empty_groups_keep_full_gallery_and_nulls(self):
        a, q = fixtures(empty=True); study = AuditFigures(a, q); gallery = RecordingGallery()
        render(gallery, study)
        self.assertEqual(len(gallery.figures), 111)
        self.assertTrue(all(r["value"] is None for r in gallery.rows["all_estimates"]))
        self.assertTrue(all(r["low"] is None for r in gallery.rows["all_intervals"]))
        self.assertEqual(len(gallery.rows["cross_split"]), 84)
        self.assertTrue(all(r["null_count"] is None for r in gallery.rows["intake_columns"]))
        self.assertEqual(gallery.figures[-2]["selected_cell_ids"], [])

    def test_paper_summaries_keep_every_interval_and_support_cell(self):
        a, q = fixtures(); study = AuditFigures(a, q); gallery = RecordingGallery()
        render_paper_summaries(gallery, study)
        forest, support, examples, design = gallery.figures
        self.assertEqual(len(forest["ytick_labels"]), 2)
        self.assertTrue(all(len(labels) == 28 for labels in forest["ytick_labels"]))
        # The first recorded interval is converted from probability units to pp.
        self.assertTrue(np.allclose(forest["axis_lines"][0][0], [4., 12.], atol=1e-12, rtol=0))
        self.assertEqual(support["scatter_points"], 84)
        self.assertIn("100 U1 clusters", support["caption"])
        self.assertTrue(all(types == ["BarContainer"] for types in examples["containers"][:2]))
        self.assertTrue(all("ErrorbarContainer" not in types for types in examples["containers"]))
        self.assertIn("Hypothetical binary question", design["caption"])
        self.assertIn("Source-only", design["caption"])

    def test_paper_examples_selection_is_absolute_eligible_and_tie_stable(self):
        a, q = fixtures(); initial = AuditFigures(a, q)
        self.assertEqual(paper_example_cells(initial)["bbh"]["cell_id"], initial.group("confirmation", "bbh")[0]["cell_id"])
        def changed(cell, u0, u1, exact, eligible=True):
            cell["estimates"] = dict(zip(METRICS, (u0, u1, exact, exact - u0, u1 - u0, exact - u1)))
            for metric in DELTAS:
                value = cell["estimates"][metric]
                cell["intervals"][metric].update(low=value - .03, high=value + .03)
            cell.update(eligible=eligible, material_effect=eligible)
        selected_bbh = a["confirmation"]["cells"][5]
        selected_mmlu = a["confirmation"]["cells"][56 + 3]
        changed(selected_bbh, -.1, 0., .25)
        changed(selected_mmlu, .3, .25, -.1)
        changed(a["confirmation"]["cells"][0], -.4, 0., .4, eligible=False)
        recalc(a["confirmation"]); study = AuditFigures(a, q)
        selected = paper_example_cells(study)
        self.assertEqual(selected["bbh"]["cell_id"], selected_bbh["cell_id"])
        self.assertEqual(selected["mmlu"]["cell_id"], selected_mmlu["cell_id"])
        self.assertLess(selected["mmlu"]["estimates"]["delta_total"], 0)
        gallery = RecordingGallery(); render_paper_summaries(gallery, study)
        self.assertEqual(gallery.figures[2]["selected_cell_ids"], [selected_bbh["cell_id"], selected_mmlu["cell_id"]])

    def test_paper_support_rejects_inconsistent_fixed_gates(self):
        a, q = fixtures()
        for split in SPLITS: a[split]["settings"]["min_clusters"]["confirmation"] = 200
        study = AuditFigures(a, q)
        with self.assertRaises(FigureError): render_paper_summaries(RecordingGallery(), study)

    def test_sparse_missing_and_observed_zero_remain_distinct(self):
        a, q = fixtures(empty=True)
        cell = a["discovery"]["cells"][0]
        cell["estimates"].update({m: 0. for m in METRICS})
        cell["support_eligible"] = True
        cell["ineligible_reasons"] = ["degenerate_or_nonfinite_bootstrap"]
        for interval in cell["intervals"].values(): interval["status"] = "degenerate_or_nonfinite_bootstrap"
        recalc(a["discovery"])
        study = AuditFigures(a, q); gallery = RecordingGallery(); source_tables(gallery, study)
        estimates = gallery.rows["all_estimates"]
        self.assertEqual([r["value"] for r in estimates[:6]], [0.] * 6)
        self.assertTrue(all(r["value"] is None for r in estimates[6:]))
        matrix = _matrix(study, study.group("discovery", "bbh"), lambda c: c["estimates"]["native_u0_gap"], antisymmetric=True)
        self.assertEqual(matrix[0, 1], 0.)
        self.assertTrue(np.isnan(matrix[0, 2]))
        self.assertFalse(gallery.rows["all_cells"][0]["eligible"])

    def test_secondary_unavailable_interval_is_explicit_in_pair_plot(self):
        a, q = fixtures()
        interval = a["discovery"]["cells"][0]["intervals"]["delta_version"]
        interval.update(low=None, high=None, se=None, critical=None, status="degenerate_or_nonfinite_bootstrap")
        study = AuditFigures(a, q); gallery = RecordingGallery(); render_pairs(gallery, study)
        self.assertIn("Version shift: N/A", gallery.figures[0]["annotations"])

    def test_pair_extreme_category_annotations_and_marks_have_real_bbox_clearance(self):
        a, q = fixtures()
        for split in SPLITS:
            for cell in a[split]["cells"]:
                cell["intervals"]["delta_version"].update(low=None, high=None, se=None, critical=None,
                    status="degenerate_or_nonfinite_bootstrap")
        study = AuditFigures(a, q)
        wanted = {f"{split}_pair_m01_m02" for split in SPLITS}
        gallery = GeometryGallery(wanted); render_pairs(gallery, study)
        self.assertEqual(set(gallery.geometry), wanted)
        for identifier, panels in gallery.geometry.items():
            version = [label for label in panels[1]["annotations"] if label["text"] == "Version shift: N/A"]
            self.assertEqual([label["position"][1] for label in version], [.2, 1.2, 2.2])
            self.assert_annotation_geometry(identifier, panels)

    def test_all_unavailable_forests_keep_first_and_last_label_bboxes_inside(self):
        a, q = fixtures(empty=True); study = AuditFigures(a, q)
        wanted = {"discovery_bbh_intervals", "confirmation_mmlu_intervals"}
        gallery = GeometryGallery(wanted); render_cells(gallery, study)
        self.assertEqual(set(gallery.geometry), wanted)
        for identifier, panels in gallery.geometry.items():
            self.assertEqual(len(panels), 3)
            for panel in panels:
                self.assertEqual(len(panel["annotations"]), 28)
                self.assertEqual(panel["annotations"][0]["position"][1], 0)
                self.assertEqual(panel["annotations"][-1]["position"][1], 27)
            self.assert_annotation_geometry(identifier, panels)

    def test_rejects_corrupt_scientific_contract(self):
        corruptions = [
            lambda a, q: a["discovery"]["cells"][0]["estimates"].pop("delta_version"),
            lambda a, q: a["discovery"]["cells"].pop(),
            lambda a, q: a["discovery"]["cells"][0]["counts"].pop("u1_clusters"),
            lambda a, q: a["discovery"]["cells"][0]["estimates"].update(delta_total=float("nan")),
            lambda a, q: a["discovery"]["cells"][0]["estimates"].update(delta_total=.2),
            lambda a, q: a["confirmation"]["settings"].update(practical_effect=.2),
            lambda a, q: a["confirmation"]["input_binding"].update(slim_sha256="c" * 64),
            lambda a, q: a["confirmation"]["input_binding"].update(spec_sha256="c" * 64),
            lambda a, q: a["confirmation"]["input_binding"].update(spec_sha256=None),
            lambda a, q: a["confirmation"]["settings"].update(confirmation_buckets=list(range(7, 14))),
            lambda a, q: a["confirmation"]["settings"].update(min_joint_finite_bootstrap_fraction=.95),
            lambda a, q: a["confirmation"]["settings"].update(metadata_audit_sha256="e" * 64),
            lambda a, q: a["confirmation"]["settings"].update(metadata_conflict_policy="ignore"),
            lambda a, q: a["confirmation"]["settings"].update(population="Different population"),
            lambda a, q: a["confirmation"]["settings"].update(context_definition="scope_and_literal_cue"),
            lambda a, q: a["confirmation"].update(context_definition="scope_and_literal_cue"),
            lambda a, q: q["confirmation"].update(context_definition="scope_and_literal_cue"),
            lambda a, q: q["confirmation"]["metadata_integrity"]["conflict_ledger"].update(excluded_cluster_keys=["different"], excluded_cluster_count=1),
            lambda a, q: a["discovery"]["cells"][0]["intervals"]["delta_total"].update(low=None),
            lambda a, q: a["discovery"]["cells"][0]["intervals"]["delta_total"].update(simultaneous_family="pooled"),
            lambda a, q: a["discovery"]["cells"][0].update(material_effect=False),
            lambda a, q: q["confirmation"]["models"].pop("bbh|handmade/model-0"),
            lambda a, q: q["discovery"].update(unselected_outcomes_used=True),
        ]
        for mutation in corruptions:
            with self.subTest(mutation=mutation):
                a, q = fixtures(); mutation(a, q)
                with self.assertRaises(FigureError): AuditFigures(a, q)

    def test_even_identical_overlapping_split_allocations_are_rejected(self):
        a, q = fixtures()
        for split in SPLITS: a[split]["settings"]["reserve_buckets"][0] = 0
        with self.assertRaises(FigureError): AuditFigures(a, q)

    def test_even_consistently_labeled_h1_inputs_are_not_h2_gallery_inputs(self):
        a, q = fixtures()
        for split in SPLITS:
            a[split]["settings"]["context_definition"] = "scope_and_literal_cue"
            a[split]["context_definition"] = q[split]["context_definition"] = "scope_and_literal_cue"
        with self.assertRaises(FigureError): AuditFigures(a, q)

    def test_profile_must_match_frozen_slim_and_all_rows(self):
        a, q = fixtures(); profile = intake_fixture()
        profile["slim"]["sha256"] = "c" * 64
        with self.assertRaises(FigureError): AuditFigures(a, q, profile)
        profile = intake_fixture(); profile["nulls"].pop("input")
        with self.assertRaises(FigureError): AuditFigures(a, q, profile)

    def test_loads_only_bound_summary_contract(self):
        a, q = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in SPLITS:
                (root / split).mkdir()
                for name, record in (("analysis", a[split]), ("dataquality", q[split]), ("protocol", a[split]["settings"])):
                    (root / split / (name + ".json")).write_text(json.dumps(record), encoding="utf-8")
            gallery = Gallery(root / "gallery")
            study = load_sources(gallery, root / "discovery", root / "confirmation")
            self.assertEqual(len(gallery.sources), 6)
            self.assertEqual(len(study.cells["confirmation"]), 84)
            (root / "confirmation" / "protocol.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FigureError): load_sources(gallery, root / "discovery", root / "confirmation")

    def test_one_handmade_actual_layout_png_svg_and_offline_index(self):
        with tempfile.TemporaryDirectory() as directory:
            gallery = Gallery(Path(directory) / "gallery")
            gallery.expected = {"figure_ids": ["handmade_layout"], "tables": {"toy_source": 2}}
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            ax.plot([0, 1], [0, .08], "o-"); ax.set_xlabel("Handmade question coverage"); ax.set_ylabel("Handmade proxy shift")
            gallery.save(fig, "handmade_layout", "Handmade fixture: complete layout check", "Two manually specified points; no research outcome data.", "Fixture")
            gallery.csv("toy_source", [{"value": 0.}, {"value": None}], "Zero and unavailable remain distinct.")
            gallery.finish()
            manifest = json.loads((gallery.output / "figure_manifest.json").read_text())
            self.assertEqual(manifest["figure_count"], 1)
            for extension in ("png", "svg"):
                path = gallery.output / "figures" / ("handmade_layout." + extension)
                self.assertGreater(path.stat().st_size, 1000)
            from PIL import Image
            with Image.open(gallery.output / "figures" / "handmade_layout.png") as image:
                self.assertGreater(image.width, 1000); self.assertGreater(image.height, 650)
            page = (gallery.output / "index.html").read_text(encoding="utf-8")
            self.assertIn("Released CoT acknowledgment audit", page)
            self.assertNotIn("R13", page)
            self.assertNotIn("https://", page)
            self.assertEqual((gallery.output / "tables" / "toy_source.csv").read_text().splitlines(), ["value", "0.0", '""'])

    def test_source_change_is_caught_before_finishing_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "handmade.json"; source.write_text("{}")
            gallery = Gallery(root / "gallery"); gallery.source(source)
            gallery.expected = {"figure_ids": [], "tables": {}}
            source.write_text('{"changed": true}')
            with self.assertRaises(FigureError): gallery.finish()


if __name__ == "__main__":
    unittest.main()
