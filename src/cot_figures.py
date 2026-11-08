"""Complete, source-bound figures for the released acknowledgment-proxy audit.

No scientific estimator, model, network request or corpus parser is run here.
Only separately completed discovery/confirmation summaries are rendered.
"""
from __future__ import annotations

import csv
import hashlib
import html
from itertools import combinations
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "cot-ack-audit-v1"})


SPLITS = ("discovery", "confirmation")
GAPS = ("native_u0_gap", "native_u1_gap", "exact_u1_gap")
DELTAS = ("delta_total", "delta_composition", "delta_version")
METRICS = GAPS + DELTAS
LABELS = {"native_u0_gap": "Native / U0", "native_u1_gap": "Native / U1 (all versions)",
          "exact_u1_gap": "Exact-version / U1", "delta_total": "Total shift",
          "delta_composition": "Composition shift", "delta_version": "Version shift"}
COLORS = {GAPS[0]: "#718096", GAPS[1]: "#276f9b", GAPS[2]: "#138777",
          DELTAS[0]: "#138777", DELTAS[1]: "#ae6b21", DELTAS[2]: "#8262a0"}
COUNTS = ("u0_questions", "u1_questions", "u0_clusters", "u1_clusters", "u0_contexts", "u1_contexts",
          "a_observed_versions", "b_observed_versions", "common_exact_versions")
QUALITY = ("selected_rows", "baseline_rows", "cued_scope_rows", "has_cue_not_true", "ack_missing_or_nonbool",
           "invalid_context", "valid_ack_rows", "missing_exact_version_rows")
INTAKE_COLUMNS = ("core_id", "model", "id", "scope", "input", "output", "reasoning", "model_answer",
                  "is_correct", "cue_acknowledgment", "state_transition", "switched_to_cue", "factor_utilization_rate",
                  "factor_utilization_used", "factor_utilization_total", "correct_answer", "correct_answer_index",
                  "correct_text", "choices", "causal_factors", "baseline_model_answer", "cue", "cue_target_answer", "has_cue")
OMITTED = ("input", "output", "reasoning")
GLOBAL_IDS = ("model_legend", "global_gates", "eligibility_overview", "transfer_delta_total",
              "transfer_delta_composition", "transfer_delta_version", "transfer_native_gaps",
              "interval_width_vs_coverage", "version_overlap", "quality_model_coverage",
              "support_eligibility_region", "interval_width_distributions", "intake_file_rows", "intake_nulls", "intake_provenance")
PAPER_IDS = ("paper_total_shift", "paper_support", "paper_examples", "paper_design_support")
PAPER_DATASETS = ("bbh", "mmlu")
PAPER_EXAMPLE_RULE = ("Descriptive selection after confirmation: largest absolute eligible total shift within each fixed BBH/MMLU dataset; "
                      "ties use the frozen model-pair order; no new test or claim is based on selection")
PAPER_NOTE = ("Historical released judge-ACK proxy; pre-outcome metadata quarantine; H2 cue-kind contexts. "
              "Confirmation uses held-out questions within these benchmarks. Recorded conditional approximate simultaneous intervals; no causal interpretation or overall ranking.")
DESIGN_NOTE_PATH = "docs/research-reset/cot-design-support-note-2026-11-08.md"
CELL_VIEWS = ("coverage", "gaps", "decomposition", "intervals", "sampling_units", "support_precision")
NOTE = ("Released historical-model acknowledgment proxy; not latent faithfulness or a new frontier-model evaluation. "
        "Population excludes entire baseline-input clusters with conflicting released reference metadata; exclusions are fixed before ACK analysis. "
        "Context is cue kind (scope_only); exact-version matching is unchanged. Exploratory discovery reuses inspected questions; "
        "confirmation uses new questions within the same benchmarks, not an out-of-domain evaluation. "
        "Gray N/A denotes unavailable or inferentially unsupported comparisons; every source estimate remains in CSV. "
        "Approximate cluster-bootstrap intervals are conditional and simultaneous separately for each delta endpoint within a split, "
        "not jointly across endpoints/splits. No causal cue-version attribution or corrected overall ranking.")


class FigureError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def number(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise FigureError("Recorded numeric values must be finite or explicit null")
    return float(value)


def delivery_spec(models: list[str], datasets: list[str]) -> dict:
    pairs = list(combinations(range(len(models)), 2))
    figures = [f"{split}_{dataset}_{view}" for split in SPLITS for dataset in datasets for view in CELL_VIEWS]
    figures += [f"{split}_pair_m{a + 1:02d}_m{b + 1:02d}" for split in SPLITS for a, b in pairs]
    figures += list(GLOBAL_IDS)
    figures += list(PAPER_IDS)
    cells = len(SPLITS) * len(datasets) * len(pairs)
    return {"schema_version": 1, "static_png_svg_pairs": len(figures), "interactive_html": 0,
            "figure_ids": figures, "tables": {"all_cells": cells, "all_estimates": cells * len(METRICS),
                "all_intervals": cells * len(DELTAS), "model_legend": len(models),
                "dataquality": len(SPLITS) * len(datasets) * len(models), "split_summary": len(SPLITS),
                "cross_split": len(datasets) * len(pairs), "intake_columns": len(INTAKE_COLUMNS)},
            "coverage_policy": "Every declared split, dataset and model pair; no data-dependent figure selection",
            "renderer_config": {"backend": "Agg", "png_dpi": 155, "formats": ["png", "svg"], "font": "DejaVu Sans",
                                "scientific_power_limits": [-3, 4], "layout": "constrained_layout plus tight bounding box",
                                "missing_values": "gray N/A, never zero imputation", "point_gap_axis": [-1, 1],
                                "svg_fonttype": "none", "svg_hashsalt": "cot-ack-audit-v1"},
            "paper_figures": {"ids": list(PAPER_IDS), "size_inches": [12, 6], "main_font_points": 12,
                              "footer_font_points": 10, "gaps_and_shifts": "percentage points",
                              "support_gates": {"u1_clusters": 100, "overlap_question_coverage": .25},
                              "example_rule": PAPER_EXAMPLE_RULE, "design_note": DESIGN_NOTE_PATH},
            "uncertainty": NOTE,
            "heatmap_color_policy": "Native gaps +/-1; delta color saturation +/-max(.1,4*practical_effect), printed values unclipped",
            "source_implementation_authorship": {"renderer": "/root/cot_intake", "scientific_analysis": "/root/alignment_brainstorm"}}


class AuditFigures:
    def __init__(self, analyses: dict[str, dict], qualities: dict[str, dict], intake: dict | None = None):
        self.analyses, self.qualities, self.intake = analyses, qualities, intake
        first = analyses["discovery"]
        self.models, self.datasets = first["models"], first["datasets"]
        self.pairs = list(combinations(self.models, 2))
        self.model_ids = {model: f"M{i + 1}" for i, model in enumerate(self.models)}
        self.settings = first["settings"]
        self.effects = float(self.settings["practical_effect"])
        self.tie = float(self.settings["tie_band"])
        self.cells = {split: {c["cell_id"]: c for c in analyses[split]["cells"]} for split in SPLITS}
        self._validate()

    def _validate(self) -> None:
        if len(set(self.models)) != len(self.models) or len(self.models) < 2 or not self.datasets or len(set(self.datasets)) != len(self.datasets):
            raise FigureError("Unique model and dataset metadata required")
        if any(not isinstance(v, str) or not v or "|" in v for v in self.models + self.datasets):
            raise FigureError("Invalid frozen model/dataset identifier")
        if any(not d.replace("_", "").isalnum() for d in self.datasets):
            raise FigureError("Dataset identifiers must be safe filenames")
        if not math.isfinite(self.effects) or not math.isfinite(self.tie) or not 0 < self.effects <= 1 or not 0 <= self.tie < self.effects:
            raise FigureError("Invalid frozen effect/tie thresholds")
        required_settings = ("models", "datasets", "split_seed", "bootstrap_replicates", "bootstrap_seed", "min_clusters",
                             "min_overlap_coverage", "practical_effect", "tie_band", "split_modulus", "discovery_buckets",
                             "confirmation_buckets", "reserve_buckets", "min_joint_finite_bootstrap_fraction",
                             "metadata_conflict_policy", "metadata_audit_sha256", "population", "context_definition")
        for split in SPLITS:
            a = self.analyses[split]
            if a.get("schema_version") != 1 or a.get("split") != split or a.get("models") != self.models or a.get("datasets") != self.datasets:
                raise FigureError("Analysis metadata or split mismatch")
            for name in required_settings:
                if name not in a["settings"] or a["settings"][name] != self.settings[name]:
                    raise FigureError(f"Frozen scientific settings differ: {name}")
            allocation = [a["settings"][name] for name in ("discovery_buckets", "confirmation_buckets", "reserve_buckets")]
            modulus = a["settings"]["split_modulus"]
            if type(modulus) is not int or modulus < 3 or any(not isinstance(v, list) or not v for v in allocation):
                raise FigureError("Explicit nonempty three-way split allocation required")
            buckets = [bucket for group in allocation for bucket in group]
            if any(type(bucket) is not int for bucket in buckets) or len(set(buckets)) != len(buckets) or set(buckets) != set(range(modulus)):
                raise FigureError("Split allocation must be disjoint and cover the frozen modulus")
            finite_fraction = number(a["settings"]["min_joint_finite_bootstrap_fraction"])
            if finite_fraction is None or not 0 < finite_fraction <= 1:
                raise FigureError("Invalid finite-bootstrap support requirement")
            if a["settings"]["metadata_conflict_policy"] != "quarantine_entire_baseline_union_cluster_before_outcomes":
                raise FigureError("Unexpected metadata-conflict policy")
            if a["settings"]["context_definition"] != "scope_only" or a.get("context_definition", "scope_only") != "scope_only":
                raise FigureError("R18 requires the same H2 cue-kind scope_only context definition")
            if not isinstance(a["settings"]["population"], str) or not a["settings"]["population"].strip():
                raise FigureError("Explicit frozen analysis population required")
            metadata_hash = a["settings"]["metadata_audit_sha256"]
            if not isinstance(metadata_hash, str) or len(metadata_hash) != 64 or any(c not in "0123456789abcdef" for c in metadata_hash):
                raise FigureError("Missing valid frozen metadata-audit hash")
            if a.get("estimand") != "question-weighted released acknowledgment proxy":
                raise FigureError("Unexpected estimand; do not relabel an unknown result")
            expected = [f"{d}|{ma}|{mb}" for d in self.datasets for ma, mb in self.pairs]
            if [c["cell_id"] for c in a["cells"]] != expected or len(self.cells[split]) != len(expected):
                raise FigureError("Incomplete, duplicate or reordered cell matrix")
            for cell in a["cells"]:
                if cell["cell_id"] != f"{cell['dataset']}|{cell['model_a']}|{cell['model_b']}":
                    raise FigureError("Cell identity mismatch")
                for field in ("eligible", "support_eligible", "material_effect"):
                    if type(cell.get(field)) is not bool:
                        raise FigureError(f"Explicit eligibility flag required: {field}")
                for key in COUNTS:
                    if type(cell["counts"].get(key)) is not int or cell["counts"][key] < 0:
                        raise FigureError("Every support count must be explicit and nonnegative")
                coverage = number(cell["counts"].get("overlap_question_coverage"))
                if coverage is None or not 0 <= coverage <= 1:
                    raise FigureError("Explicit overlap coverage required")
                if set(cell["estimates"]) != set(METRICS) or set(cell["intervals"]) != set(DELTAS):
                    raise FigureError("Incomplete metric/interval matrix")
                e = {metric: number(value) for metric, value in cell["estimates"].items()}
                for metric in GAPS:
                    if e[metric] is not None and abs(e[metric]) > 1 + 1e-10:
                        raise FigureError("Acknowledgment gap outside its range")
                for metric, left, right in ((DELTAS[0], GAPS[2], GAPS[0]), (DELTAS[1], GAPS[1], GAPS[0]), (DELTAS[2], GAPS[2], GAPS[1])):
                    if e[left] is None or e[right] is None:
                        if e[metric] is not None:
                            raise FigureError("A difference with an unavailable constituent cannot be recorded")
                    elif e[metric] is None or not math.isclose(e[metric], e[left] - e[right], abs_tol=1e-10, rel_tol=1e-10):
                        raise FigureError("Stored decomposition does not close")
                for metric in DELTAS:
                    interval = cell["intervals"][metric]
                    if interval.get("simultaneous_family") != metric or not isinstance(interval.get("status"), str) or interval["status"] == "pending":
                        raise FigureError("Unassessed or mislabelled interval family")
                    lo, hi = number(interval.get("low")), number(interval.get("high"))
                    if (lo is None) != (hi is None) or (lo is not None and lo > hi):
                        raise FigureError("Invalid interval bounds")
                    if lo is not None and (e[metric] is None or interval["status"] != "simultaneous_95_max_t"):
                        raise FigureError("Unexpected source interval semantics")
                    for name in ("se", "critical"):
                        value = number(interval.get(name))
                        if value is not None and value < 0:
                            raise FigureError("Invalid source interval scale")
                if cell["eligible"] and (not cell["support_eligible"] or any(v is None for v in e.values()) or cell["intervals"][DELTAS[0]]["low"] is None):
                    raise FigureError("Eligible cell lacks primary evidence")
                expected_material = bool(cell["eligible"] and abs(e[DELTAS[0]]) >= self.effects and
                    (cell["intervals"][DELTAS[0]]["low"] > self.tie or cell["intervals"][DELTAS[0]]["high"] < -self.tie))
                if cell["material_effect"] != expected_material:
                    raise FigureError("Source material-effect flag disagrees with its frozen rule")
            for name in ("eligible_cells", "support_eligible_cells", "material_effect_cells"):
                key = {"eligible_cells": "eligible", "support_eligible_cells": "support_eligible", "material_effect_cells": "material_effect"}[name]
                if a["global"].get(name) != sum(c[key] for c in a["cells"]):
                    raise FigureError("Global counts disagree with complete cells")
            quality = self.qualities[split]
            if quality.get("split") != split or quality.get("unselected_outcomes_used") is not False:
                raise FigureError("Missing split-isolation quality record")
            if quality.get("context_definition", "scope_only") != a["settings"]["context_definition"]:
                raise FigureError("Data-quality context definition differs from the H2 analysis protocol")
            ledger = quality.get("metadata_integrity", {}).get("conflict_ledger")
            if not isinstance(ledger, dict) or ledger.get("policy") != a["settings"]["metadata_conflict_policy"]:
                raise FigureError("Missing frozen metadata-conflict ledger")
            excluded = ledger.get("excluded_cluster_keys")
            if not isinstance(excluded, list) or any(not isinstance(k, str) or not k for k in excluded) or len(set(excluded)) != len(excluded):
                raise FigureError("Explicit unique metadata-exclusion keys required")
            if ledger.get("excluded_cluster_count") != len(excluded) or ledger.get("acknowledgment_or_generated_answer_values_used") is not False or ledger.get("original_cluster_keys_and_bucket_assignments_preserved") is not True:
                raise FigureError("Metadata exclusions must precede outcomes and preserve cluster assignment")
            for d in self.datasets:
                for m in self.models:
                    record = quality.get("models", {}).get(f"{d}|{m}")
                    if not isinstance(record, dict) or any(type(record.get(k)) is not int or record[k] < 0 for k in QUALITY):
                        raise FigureError("Quality counters must be explicit; missing is not zero")
        if self.qualities["discovery"]["metadata_integrity"]["conflict_ledger"] != self.qualities["confirmation"]["metadata_integrity"]["conflict_ledger"]:
            raise FigureError("Discovery and confirmation have different metadata-excluded populations")
        for hash_key in ("slim_sha256", "spec_sha256"):
            bound_hashes = [self.analyses[split]["input_binding"].get(hash_key) for split in SPLITS]
            if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in bound_hashes):
                raise FigureError(f"Missing valid frozen source hash: {hash_key}")
            if bound_hashes[0] != bound_hashes[1]:
                raise FigureError(f"Discovery/confirmation frozen source bindings differ: {hash_key}")
        source_hashes = [self.analyses[split]["input_binding"]["slim_sha256"] for split in SPLITS]
        if self.intake is not None:
            if type(self.intake.get("rows")) is not int or self.intake["rows"] <= 0:
                raise FigureError("Invalid optional intake profile")
            if set(self.intake["retained_columns"] + self.intake["hashed_text_columns"]) != set(INTAKE_COLUMNS):
                raise FigureError("Unexpected intake column policy")
            for key in INTAKE_COLUMNS:
                if type(self.intake["nulls"].get(key)) is not int or not 0 <= self.intake["nulls"][key] <= self.intake["rows"]:
                    raise FigureError("Missing intake null count cannot become zero")
            profiles = self.intake.get("source_profiles")
            if not isinstance(profiles, list) or len(profiles) != self.intake.get("source_files") or not profiles or any(type(p.get("rows")) is not int or p["rows"] < 0 or not isinstance(p.get("source_file"), str) for p in profiles):
                raise FigureError("Incomplete intake file-row profile")
            if sum(p["rows"] for p in profiles) != self.intake["rows"] or self.intake.get("slim", {}).get("sha256") != source_hashes[0]:
                raise FigureError("Intake profile is not bound to the same complete slim source")

    def group(self, split: str, dataset: str) -> list[dict]:
        return [self.cells[split][f"{dataset}|{a}|{b}"] for a, b in self.pairs]

    def pair_label(self, cell: dict) -> str:
        return f"{self.model_ids[cell['model_a']]} − {self.model_ids[cell['model_b']]}"


class Gallery:
    def __init__(self, output: Path):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "figures").mkdir()
        (self.output / "tables").mkdir()
        self.sources, self.figures, self.tables = {}, [], []
        self.notes = [NOTE]
        self.expected = None

    def source(self, path: Path) -> Path:
        if path.is_symlink():
            raise FigureError("Source must not be a symbolic link")
        path = path.resolve()
        if path.is_symlink() or not path.is_file():
            raise FigureError("Source must be a regular immutable file")
        self.sources[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        return path

    def load(self, path: Path) -> dict:
        return json.loads(self.source(path).read_text(encoding="utf-8"))

    def save(self, fig, identifier: str, title: str, caption: str, group: str) -> None:
        if any(r["id"] == identifier for r in self.figures):
            raise FigureError("Duplicate figure ID")
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                formatter = axis.get_major_formatter()
                if isinstance(formatter, ScalarFormatter):
                    formatter.set_scientific(True); formatter.set_powerlimits((-3, 4)); formatter.set_useMathText(True)
        paper = identifier in PAPER_IDS
        fig.suptitle(title, fontsize=16 if paper else 14, fontweight="bold")
        note = ("Source-only hypothetical illustration; no observed-model or effect inference, and no new-theorem claim."
                if identifier == "paper_design_support" else PAPER_NOTE) if paper else NOTE
        fig.text(.01, -.025, textwrap.fill(caption + " " + note, 145), fontsize=10 if paper else 8, ha="left", va="top")
        paths, hashes = {}, {}
        for extension in ("png", "svg"):
            path = self.output / "figures" / f"{identifier}.{extension}"
            if path.exists():
                raise FigureError("Figure output already exists")
            fig.savefig(path, dpi=155, bbox_inches="tight", facecolor="white", metadata={"Date": None} if extension == "svg" else None)
            paths[extension], hashes[extension] = path.relative_to(self.output).as_posix(), sha256(path)
        plt.close(fig)
        self.figures.append({"id": identifier, "title": title, "caption": caption, "group": group,
                             "selection": "Complete prespecified matrix", "paths": paths, "sha256": hashes})

    def csv(self, identifier: str, rows: list[dict], description: str) -> None:
        path = self.output / "tables" / f"{identifier}.csv"
        names = list(dict.fromkeys(k for row in rows for k in row))
        if not rows or not names:
            raise FigureError("Fixed source tables must not disappear when results are unsupported")
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        self.tables.append({"path": path.relative_to(self.output).as_posix(), "id": identifier,
                            "rows": len(rows), "description": description, "sha256": sha256(path)})

    def finish(self) -> None:
        if self.expected is None or [r["id"] for r in self.figures] != self.expected["figure_ids"]:
            raise FigureError("Rendered figure matrix differs from the frozen delivery specification")
        if {r["id"]: r["rows"] for r in self.tables} != self.expected["tables"]:
            raise FigureError("Source table matrix differs from the frozen delivery specification")
        for path, record in self.sources.items():
            if sha256(Path(path)) != record["sha256"]:
                raise FigureError("Source changed during rendering")
        manifest = {"schema_version": 1, "study_id": "cot-released-acknowledgment-audit", "renderer": str(Path(__file__).resolve()),
                    "renderer_sha256": sha256(Path(__file__)), "sources": self.sources, "figures": self.figures,
                    "tables": self.tables, "notes": self.notes, "figure_count": len(self.figures), "expected_delivery": self.expected,
                    "environment": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "backend": matplotlib.get_backend()},
                    "interpretation": NOTE, "scientific_validity": "not_assessed"}
        (self.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
        sections = []
        for group in dict.fromkeys(r["group"] for r in self.figures):
            cards = []
            for row in (r for r in self.figures if r["group"] == group):
                links = " · ".join(f'<a href="{html.escape(p)}">{ext.upper()}</a>' for ext, p in row["paths"].items())
                cards.append(f'<article><h3>{html.escape(row["title"])}</h3><img loading="lazy" src="{html.escape(row["paths"]["png"])}" alt="{html.escape(row["title"])}"><p>{html.escape(row["caption"])}</p><p>{links}</p></article>')
            sections.append(f'<section><h2>{html.escape(group)}</h2><div class="grid">{"".join(cards)}</div></section>')
        tables = "".join(f'<li><a href="{html.escape(r["path"])}">{html.escape(r["id"])}</a> ({r["rows"]} rows): {html.escape(r["description"])}</li>' for r in self.tables)
        page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoT acknowledgment audit: complete evidence gallery</title><style>body{font:16px system-ui;background:#f4f7f8;color:#18323c;max-width:1550px;margin:auto;padding:24px}h1,h2{color:#164755}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:18px}article{background:white;padding:16px;border:1px solid #dce4e8;border-radius:8px}img{width:100%;height:auto}p{line-height:1.45}a{color:#08758b}@media(max-width:600px){.grid{grid-template-columns:1fr}body{padding:12px}}</style>'
        page += '<h1>Released CoT acknowledgment audit</h1><p>' + html.escape(NOTE) + '</p><p>All declared comparisons, including unsupported and null results. Model identifiers follow the frozen protocol; no performance ordering.</p>'
        page += "".join(sections) + '<h2>Complete source tables</h2><ul>' + tables + '</ul><p><a href="figure_manifest.json">Input/output hash manifest</a></p></html>'
        (self.output / "index.html").write_text(page, encoding="utf-8")


def load_sources(gallery: Gallery, discovery: Path, confirmation: Path, intake_profile: Path | None = None) -> AuditFigures:
    analyses, qualities = {}, {}
    for split, directory in zip(SPLITS, (discovery, confirmation)):
        analyses[split] = gallery.load(directory / "analysis.json")
        qualities[split] = gallery.load(directory / "dataquality.json")
        protocol = gallery.load(directory / "protocol.json")
        if analyses[split].get("settings") != protocol:
            raise FigureError("Copied protocol semantics differ from analysis settings")
    return AuditFigures(analyses, qualities, gallery.load(intake_profile) if intake_profile else None)


def source_tables(g: Gallery, study: AuditFigures) -> None:
    cells, estimates, intervals, quality_rows, summaries, cross = [], [], [], [], [], []
    for split in SPLITS:
        for cell in study.analyses[split]["cells"]:
            cells.append({"split": split, **cell})
            for metric in METRICS:
                estimates.append({"split": split, "cell_id": cell["cell_id"], "dataset": cell["dataset"],
                                  "model_a": cell["model_a"], "model_b": cell["model_b"], "metric": metric,
                                  "value": cell["estimates"][metric], "primary_eligible": cell["eligible"],
                                  "support_eligible": cell["support_eligible"]})
            for metric in DELTAS:
                intervals.append({"split": split, "cell_id": cell["cell_id"], "metric": metric, **cell["intervals"][metric]})
        for dataset in study.datasets:
            for model in study.models:
                quality_rows.append({"split": split, "dataset": dataset, "model": model,
                                     **study.qualities[split]["models"][f"{dataset}|{model}"]})
        summaries.append({"split": split, **study.analyses[split]["global"],
                          "practical_effect": study.effects, "tie_band": study.tie,
                          "min_clusters": study.settings["min_clusters"][split],
                          "min_overlap_coverage": study.settings["min_overlap_coverage"]})
    for identifier in study.cells["discovery"]:
        a, b = (study.cells[s][identifier] for s in SPLITS)
        row = {"cell_id": identifier, "dataset": a["dataset"], "model_a": a["model_a"], "model_b": a["model_b"]}
        for split, cell in zip(SPLITS, (a, b)):
            row.update({f"{split}_{metric}": cell["estimates"][metric] for metric in METRICS})
            row.update({f"{split}_{key}": cell[key] for key in ("eligible", "support_eligible", "material_effect")})
        cross.append(row)
    intake_rows = [{"column": name, "policy": "exact UTF8 hash and lengths" if name in OMITTED else "retained unchanged",
                    "null_count": study.intake["nulls"][name] if study.intake else None,
                    "source_rows": study.intake["rows"] if study.intake else None} for name in INTAKE_COLUMNS]
    for name, rows, description in (
        ("all_cells", cells, "All source cell objects, including unsupported estimates and reasons; nested objects remain JSON."),
        ("all_estimates", estimates, "Every source point estimate; null remains blank, never imputed zero."),
        ("all_intervals", intervals, "All recorded delta intervals and endpoint-specific statuses; native gap CIs were not supplied."),
        ("model_legend", [{"index": study.model_ids[m], "model": m, "order": i} for i, m in enumerate(study.models)], "Frozen model order, independent of observed performance."),
        ("dataquality", quality_rows, "Explicit source counters for all split/dataset/model records."),
        ("split_summary", summaries, "Source global gates and fixed thresholds; no new success rule."),
        ("cross_split", cross, "Every same-cell discovery/confirmation comparison; these are different held-out question groups."),
        ("intake_columns", intake_rows, "Full raw-column retention policy and optional recorded null counts.")):
        g.csv(name, rows, description)


def _axes(rows=1, cols=1, width=12, height=5):
    return plt.subplots(rows, cols, figsize=(width, height), constrained_layout=True, squeeze=False)


def _na(ax, message="N/A — no eligible comparison") -> None:
    ax.set_facecolor("#eef1f2")
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", color="#697982", fontsize=11, wrap=True)


def _matrix(study: AuditFigures, cells: list[dict], getter, *, antisymmetric=False) -> np.ndarray:
    matrix = np.full((len(study.models), len(study.models)), np.nan)
    for cell in cells:
        value = getter(cell)
        if value is None:
            continue
        a, b = study.models.index(cell["model_a"]), study.models.index(cell["model_b"])
        matrix[a, b], matrix[b, a] = value, -value if antisymmetric else value
    return matrix


def _heatmap(fig, ax, matrix, labels, title, *, diverging=False, vmax=None, fraction=False):
    cmap = plt.get_cmap("RdBu_r" if diverging else "Blues").copy(); cmap.set_bad("#e5e9eb")
    maximum = vmax if vmax is not None else max(1., float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-maximum if diverging else 0, vmax=maximum)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title, fontsize=11)
    for (i, j), value in np.ndenumerate(matrix):
        text = "—" if i == j else "N/A" if not np.isfinite(value) else f"{value:.2f}" if fraction or diverging else f"{value:g}"
        ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if np.isfinite(value) and abs(value) > .7 * maximum else "#263a44")
    fig.colorbar(image, ax=ax, fraction=.045, pad=.03, extend="both" if diverging else "neither")


def _interval(ax, y, cell, metric, offset=0., color=None) -> bool:
    interval, value = cell["intervals"][metric], cell["estimates"][metric]
    if not cell["eligible"] or value is None or interval["low"] is None:
        return False
    color = color or COLORS[metric]
    ax.plot([interval["low"], interval["high"]], [y + offset] * 2, color=color, lw=1.3)
    ax.plot(value, y + offset, "o", color=color, ms=4)
    return True


def render_cells(g: Gallery, s: AuditFigures) -> None:
    labels = [s.model_ids[m] for m in s.models]
    for split in SPLITS:
        for dataset in s.datasets:
            cells, prefix = s.group(split, dataset), f"{split}_{dataset}"
            fig, axes = _axes(cols=3, width=15, height=5)
            for ax, field, title in zip(axes.flat, ("u0_questions", "u1_questions", "overlap_question_coverage"), ("U0 questions", "U1 exact-overlap questions", "Overlap / U0 question coverage")):
                _heatmap(fig, ax, _matrix(s, cells, lambda c, f=field: c["counts"][f]), labels, title, fraction="coverage" in field, vmax=1 if "coverage" in field else None)
            g.save(fig, prefix + "_coverage", f"{split.title()} · {dataset.upper()}: complete pair coverage", "All support counts, including observed zeros. Diagonal comparisons are undefined; off-diagonal values are mirrored for display only.", "Support by dataset")
            fig, axes = _axes(cols=3, width=15, height=5)
            for ax, metric in zip(axes.flat, GAPS):
                _heatmap(fig, ax, _matrix(s, cells, lambda c, m=metric: c["estimates"][m] if c["eligible"] else None, antisymmetric=True), labels, LABELS[metric], diverging=True, vmax=1)
            g.save(fig, prefix + "_gaps", f"{split.title()} · {dataset.upper()}: three support-dependent gaps", "Points displayed as row-model minus column-model acknowledgment. Inferentially eligible cells only; all other source estimates remain in CSV. No native-gap CI was supplied.", "Support-dependent contrasts")
            fig, axes = _axes(cols=3, width=15, height=5)
            limit = max(.1, 4 * s.effects)
            for ax, metric in zip(axes.flat, DELTAS):
                _heatmap(fig, ax, _matrix(s, cells, lambda c, m=metric: c["estimates"][m] if c["eligible"] else None, antisymmetric=True), labels, LABELS[metric], diverging=True, vmax=limit)
            g.save(fig, prefix + "_decomposition", f"{split.title()} · {dataset.upper()}: total = composition + version", f"Color saturates at ±{limit:g}; printed estimates are unclipped. These are descriptive support/version differences, not causal effects.", "Support-dependent contrasts")
            fig, axes = _axes(cols=3, width=16, height=max(6, .29 * len(cells) + 2))
            for ax, metric in zip(axes.flat, DELTAS):
                any_shown = False
                for i, cell in enumerate(cells):
                    shown = _interval(ax, i, cell, metric); any_shown |= shown
                    if not shown:
                        ax.text(.98, i, "N/A", transform=ax.get_yaxis_transform(), ha="right", color="#7a858b", fontsize=7)
                ax.axvline(0, color="#52636b", lw=.8); ax.axvspan(-s.tie, s.tie, color="#cbd5dc", alpha=.25)
                ax.set_yticks(range(len(cells)), [s.pair_label(c) for c in cells], fontsize=8); ax.set_ylim(len(cells) - .5, -.5)
                ax.set_title(LABELS[metric]); ax.set_xlabel("Acknowledgment-gap shift")
                ax.grid(axis="x", alpha=.2)
                if not any_shown:
                    ax.set_xlim(-.1, .1)
            g.save(fig, prefix + "_intervals", f"{split.title()} · {dataset.upper()}: all pair interval families", f"Recorded simultaneous intervals are shown only for primary-eligible cells. Gray band is ±{s.tie:g}, the frozen tie band; point practical effect threshold is {s.effects:g}.", "Endpoint-specific uncertainty")
            fig, axes = _axes(cols=3, width=16, height=max(6, .29 * len(cells) + 2))
            for ax, fields, title in zip(axes.flat, (("u0_questions", "u1_questions"), ("u0_clusters", "u1_clusters"), ("u0_contexts", "u1_contexts")), ("Questions", "Deduplicated baseline clusters", "Question × cue-kind contexts")):
                for offset, field, color in ((-.13, fields[0], "#7b8c97"), (.13, fields[1], "#138777")):
                    ax.plot([c["counts"][field] for c in cells], np.arange(len(cells)) + offset, "o", color=color, ms=4, label=field)
                ax.set_yticks(range(len(cells)), [s.pair_label(c) for c in cells], fontsize=8); ax.invert_yaxis(); ax.set_title(title)
                ax.legend(fontsize=8); ax.set_xlabel("Observed source count"); ax.grid(axis="x", alpha=.2)
            g.save(fig, prefix + "_sampling_units", f"{split.title()} · {dataset.upper()}: observations and analysis units", "Every pair retained. Repeated generations, contexts, questions and deduplicated clusters have different weights and are not interchangeable sample sizes.", "Support by dataset")
            fig, axes = _axes(cols=2, width=12, height=5)
            shown = 0
            for cell in cells:
                interval = cell["intervals"][DELTAS[0]]
                if not cell["eligible"] or interval["low"] is None:
                    continue
                width = interval["high"] - interval["low"]
                axes[0, 0].scatter(cell["counts"]["u1_clusters"], width, color="#276f9b", s=22)
                axes[0, 1].scatter(cell["counts"]["overlap_question_coverage"], cell["estimates"][DELTAS[0]], color="#138777", s=22)
                shown += 1
            axes[0, 0].set(xlabel="U1 clusters", ylabel="Total-shift CI width")
            axes[0, 1].set(xlabel="U1/U0 question coverage", ylabel="Total gap shift"); axes[0, 1].axhline(0, color="gray", lw=.8)
            for ax in axes.flat:
                ax.grid(alpha=.2)
                if not shown:
                    _na(ax)
            g.save(fig, prefix + "_support_precision", f"{split.title()} · {dataset.upper()}: support and precision", f"All {shown} primary-eligible comparisons shown; {len(cells) - shown} unsupported comparisons are explicitly unavailable, not zero-valued points.", "Endpoint-specific uncertainty")


def render_pairs(g: Gallery, s: AuditFigures) -> None:
    for split in SPLITS:
        for a, b in combinations(range(len(s.models)), 2):
            cells = [s.cells[split][f"{d}|{s.models[a]}|{s.models[b]}"] for d in s.datasets]
            fig, axes = _axes(cols=3, width=15, height=max(5, .7 * len(s.datasets) + 2))
            for metric, offset in zip(GAPS, (-.2, 0, .2)):
                seen = False
                for i, cell in enumerate(cells):
                    value = cell["estimates"][metric]
                    if cell["eligible"] and value is not None:
                        axes[0, 0].plot(value, i + offset, "o", color=COLORS[metric], ms=5, label=LABELS[metric] if not seen else None); seen = True
                if not seen:
                    axes[0, 0].plot([], [], "o", color=COLORS[metric], label=LABELS[metric] + " (N/A)")
            for metric, offset in zip(DELTAS, (-.2, 0, .2)):
                for i, cell in enumerate(cells):
                    if not _interval(axes[0, 1], i, cell, metric, offset):
                        axes[0, 1].text(.98, i + offset, LABELS[metric] + ": N/A", transform=axes[0, 1].get_yaxis_transform(), ha="right", color="#7a858b", fontsize=6)
                axes[0, 1].plot([], [], "o", color=COLORS[metric], label=LABELS[metric])
            for ax in axes[0, :2]:
                ax.axvline(0, color="gray", lw=.8); ax.set_yticks(range(len(cells)), [d.upper() for d in s.datasets]); ax.set_ylim(len(cells) - .5, -.5); ax.grid(axis="x", alpha=.2)
                ax.legend(fontsize=8); ax.set_xlabel("Acknowledgment gap" if ax is axes[0, 0] else "Gap shift with recorded CI")
                for i, cell in enumerate(cells):
                    if not cell["eligible"] and ax is axes[0, 0]:
                        ax.text(.98, i, "N/A", transform=ax.get_yaxis_transform(), ha="right", color="#7a858b", fontsize=8)
            ax = axes[0, 2]
            for i, cell in enumerate(cells):
                ax.barh(i - .16, cell["counts"]["u0_questions"], height=.28, color="#80919b", label="U0 questions" if not i else None)
                ax.barh(i + .16, cell["counts"]["u1_questions"], height=.28, color="#138777", label="U1 questions" if not i else None)
            ax.set_yticks(range(len(cells)), [d.upper() for d in s.datasets]); ax.set_ylim(len(cells) - .5, -.5); ax.set_xlabel("Observed support count"); ax.legend(fontsize=8)
            identifier = f"{split}_pair_m{a + 1:02d}_m{b + 1:02d}"
            g.save(fig, identifier, f"{split.title()}: M{a + 1} − M{b + 1}, all datasets", "Left: source point gaps without invented native-gap CIs. Middle: three endpoint-specific shift families. Right: actual support counts. No dataset or pair selected for its observed effect.", "Every model pair")


def _width(cell: dict, metric: str = "delta_total") -> float | None:
    interval = cell["intervals"][metric]
    return None if not cell["eligible"] or interval["low"] is None else interval["high"] - interval["low"]


def _legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)


def render_globals(g: Gallery, s: AuditFigures) -> None:
    # The order below is part of the frozen delivery contract, not a result-driven selection.
    fig, axes = _axes(width=13, height=max(5, .55 * len(s.models) + 2))
    ax = axes[0, 0]; ax.axis("off")
    table = ax.table(cellText=[[s.model_ids[m], m] for m in s.models], colLabels=["ID", "Source model identifier"],
                     colWidths=[.12, .88], cellLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
    g.save(fig, "model_legend", "Frozen source model identifiers", "Identifiers follow the protocol order. This table is not an absolute model ranking.", "Global overview and validation")

    fig, axes = _axes(width=14, height=5); ax = axes[0, 0]; ax.axis("off")
    fields = ("hypothesis", "sensitivity_rule_pass", "robustness", "eligible_cells", "support_eligible_cells", "material_effect_cells", "discovery_go")
    values = [[field] + [str(s.analyses[split]["global"].get(field, "N/A")) for split in SPLITS] for field in fields]
    table = ax.table(cellText=values, colLabels=["Recorded gate", "Discovery", "Confirmation"], cellLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.8)
    g.save(fig, "global_gates", "Recorded stage gates and scientific scope", "Source gate values are reported verbatim. Discovery-go is an operational discovery gate, not independent confirmation. A null or inconclusive result is retained.", "Global overview and validation")

    fig, axes = _axes(cols=2, width=14, height=5)
    for ax, split in zip(axes.flat, SPLITS):
        positions = np.arange(len(s.datasets))
        for offset, field, color in ((-.25, "support_eligible", "#718096"), (0, "eligible", "#276f9b"), (.25, "material_effect", "#138777")):
            counts = [sum(c[field] for c in s.group(split, d)) for d in s.datasets]
            ax.bar(positions + offset, counts, width=.24, color=color, label=field.replace("_", " "))
        ax.set_xticks(positions, [d.upper() for d in s.datasets]); ax.set_ylim(0, len(s.pairs) + 1)
        ax.set_ylabel("Number of declared model pairs"); ax.set_title(split.title()); ax.legend(fontsize=8)
    g.save(fig, "eligibility_overview", "Eligibility and material-effect counts, every dataset", "Observed counts of source flags, including zeros. Support eligibility and valid-primary-interval eligibility are distinct; effect count is not a model ranking.", "Global overview and validation")

    for metric in DELTAS:
        fig, axes = _axes(width=7, height=6); ax = axes[0, 0]; seen = 0
        for dataset in s.datasets:
            for a in s.group("discovery", dataset):
                b = s.cells["confirmation"][a["cell_id"]]
                if not (a["eligible"] and b["eligible"]) or a["estimates"][metric] is None or b["estimates"][metric] is None:
                    continue
                label = dataset.upper() if not any(line.get_label() == dataset.upper() for line in ax.lines) else None
                ax.plot(a["estimates"][metric], b["estimates"][metric], "o", alpha=.65, ms=5, label=label); seen += 1
        bound = max(.1, 4 * s.effects, *(abs(c["estimates"][metric]) for split in SPLITS for c in s.analyses[split]["cells"] if c["eligible"] and c["estimates"][metric] is not None))
        ax.plot([-bound, bound], [-bound, bound], "--", color="gray", lw=1)
        ax.axhline(0, color="gray", lw=.6); ax.axvline(0, color="gray", lw=.6)
        ax.set_xlabel("Discovery point estimate"); ax.set_ylabel("Confirmation point estimate"); _legend(ax)
        if not seen: _na(ax)
        g.save(fig, "transfer_" + metric, f"Discovery versus confirmation: {LABELS[metric].lower()}", f"Every same-cell pair eligible in both splits is shown ({seen} of {len(s.pairs) * len(s.datasets)}). These are distinct held-out question groups; the diagonal is a visual reference, not a new test.", "Discovery and confirmation")

    fig, axes = _axes(cols=3, width=15, height=5)
    for ax, metric in zip(axes.flat, GAPS):
        seen = 0
        for dataset in s.datasets:
            pairs = [(c, s.cells["confirmation"][c["cell_id"]]) for c in s.group("discovery", dataset)]
            points = [(a["estimates"][metric], b["estimates"][metric]) for a, b in pairs if a["eligible"] and b["eligible"]]
            if points:
                ax.scatter(*zip(*points), s=20, alpha=.6, label=dataset.upper()); seen += len(points)
        ax.plot([-1, 1], [-1, 1], "--", color="gray", lw=.8); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_title(LABELS[metric]); ax.set_xlabel("Discovery point gap"); ax.set_ylabel("Confirmation point gap"); _legend(ax)
        if not seen: _na(ax)
    g.save(fig, "transfer_native_gaps", "All three source gap estimands across held-out splits", "Each point is a declared pair in a declared dataset eligible in both splits. Native-gap uncertainty intervals were not supplied, so none are invented.", "Discovery and confirmation")

    fig, axes = _axes(cols=2, width=14, height=5)
    for ax, split in zip(axes.flat, SPLITS):
        seen = 0
        for dataset in s.datasets:
            cells = [c for c in s.group(split, dataset) if _width(c) is not None]
            if cells:
                ax.scatter([c["counts"]["overlap_question_coverage"] for c in cells], [_width(c) for c in cells], label=dataset.upper(), alpha=.65, s=22); seen += len(cells)
        ax.set_xlabel("U1 / U0 question coverage"); ax.set_ylabel("Total-shift simultaneous CI width"); ax.set_title(split.title()); ax.set_xlim(0, 1); _legend(ax)
        if not seen: _na(ax)
    g.save(fig, "interval_width_vs_coverage", "Precision and exact-overlap question coverage", "All eligible total-shift intervals are plotted, without a fitted trend or causal interpretation of support.", "Support and annotation quality")

    fig, axes = _axes(rows=2, cols=len(s.datasets), width=max(9, 5 * len(s.datasets)), height=9)
    for row, split in enumerate(SPLITS):
        for col, dataset in enumerate(s.datasets):
            def overlap(c):
                denominator = min(c["counts"]["a_observed_versions"], c["counts"]["b_observed_versions"])
                return c["counts"]["common_exact_versions"] / denominator if denominator else None
            _heatmap(fig, axes[row, col], _matrix(s, s.group(split, dataset), overlap), [s.model_ids[m] for m in s.models], f"{split.title()} / {dataset.upper()}", fraction=True, vmax=1)
    g.save(fig, "version_overlap", "Exact common-version count relative to the smaller version count", "Ratio = common exact U1 versions / min(model A observed U0 versions, model B observed U0 versions). Zero denominator is N/A. Counts are included in all_cells.csv; this is not a probability of causal equivalence.", "Support and annotation quality")

    fig, axes = _axes(rows=2, cols=len(s.datasets), width=max(9, 5 * len(s.datasets)), height=9)
    qfields = ("has_cue_not_true", "ack_missing_or_nonbool", "invalid_context", "missing_exact_version_rows")
    for row, split in enumerate(SPLITS):
        for col, dataset in enumerate(s.datasets):
            ax = axes[row, col]; matrix = np.full((len(s.models), len(qfields)), np.nan)
            for i, model in enumerate(s.models):
                q = s.qualities[split]["models"][f"{dataset}|{model}"]
                if q["cued_scope_rows"]:
                    matrix[i] = [q[k] / q["cued_scope_rows"] for k in qfields]
            cmap = plt.get_cmap("Oranges").copy(); cmap.set_bad("#e5e9eb")
            im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_yticks(range(len(s.models)), [s.model_ids[m] for m in s.models])
            ax.set_xticks(range(4), ["cue flag", "ACK type", "context", "version hash"], rotation=35, ha="right")
            ax.set_title(f"{split.title()} / {dataset.upper()}")
            for (i, j), v in np.ndenumerate(matrix): ax.text(j, i, "N/A" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
    g.save(fig, "quality_model_coverage", "Recorded annotation and context exclusions by model", "Each counter is divided by all selected cued-scope rows; categories may overlap. No cued rows gives N/A. False ACK annotations may include source parse failures: this chart cannot resolve annotation accuracy.", "Support and annotation quality")

    fig, axes = _axes(cols=2, width=14, height=5)
    for ax, split in zip(axes.flat, SPLITS):
        for flag, marker, color in ((False, "x", "#8c989e"), (True, "o", "#138777")):
            cells = [c for c in s.analyses[split]["cells"] if c["support_eligible"] == flag]
            ax.scatter([c["counts"]["u1_clusters"] for c in cells], [c["counts"]["overlap_question_coverage"] for c in cells], marker=marker, color=color, s=25, alpha=.7, label="support eligible" if flag else "support ineligible")
        ax.axvline(s.settings["min_clusters"][split], color="#276f9b", ls="--", lw=1)
        ax.axhline(s.settings["min_overlap_coverage"], color="#276f9b", ls="--", lw=1)
        ax.set_title(split.title()); ax.set_xlabel("U1 shared question clusters"); ax.set_ylabel("Overlap question coverage"); ax.set_ylim(-.03, 1.03); ax.legend(fontsize=8)
    g.save(fig, "support_eligibility_region", "Frozen support thresholds and every declared cell", "All cells are shown, including observed zero support. Dashed thresholds are fixed by each split's protocol; support eligibility can also require finite estimands, so crossing these lines alone does not establish eligibility.", "Support and annotation quality")

    fig, axes = _axes(cols=3, width=15, height=5)
    for ax, metric in zip(axes.flat, DELTAS):
        values = [[_width(c, metric) for c in s.analyses[split]["cells"] if _width(c, metric) is not None] for split in SPLITS]
        for i, (split, widths) in enumerate(zip(SPLITS, values)):
            if widths:
                # Every finite width appears; no bins or outlier suppression.
                offsets = np.linspace(-.12, .12, len(widths))
                ax.scatter(i + offsets, widths, s=15, alpha=.6, label=split.title())
            else: ax.text(i, .5, "N/A", transform=ax.get_xaxis_transform(), ha="center", color="#7a858b")
        ax.set_xticks(range(2), [v.title() for v in SPLITS]); ax.set_xlim(-.4, 1.4)
        ax.set_title(LABELS[metric]); ax.set_ylabel("Simultaneous CI width")
    g.save(fig, "interval_width_distributions", "All available endpoint-specific interval widths", "Every finite source width is retained as an individual point, with deterministic display offsets. Each endpoint has its own simultaneous family; intervals are not pooled into a new inferential claim.", "Support and annotation quality")

    fig, axes = _axes(width=14, height=6); ax = axes[0, 0]
    if s.intake:
        profiles = s.intake["source_profiles"]
        ax.barh(range(len(profiles)), [p["rows"] for p in profiles], color="#276f9b")
        ax.set_yticks(range(len(profiles)), [Path(p["source_file"]).name for p in profiles], fontsize=6); ax.invert_yaxis(); ax.set_xlabel("Recorded source rows")
    else: _na(ax, "N/A — optional intake profile was not supplied")
    g.save(fig, "intake_file_rows", "Source-file row counts from the frozen intake profile", "All input source-file profiles are shown in their recorded order, without opening transcript Parquet payloads or using outcomes.", "Intake integrity and provenance")

    fig, axes = _axes(width=13, height=7); ax = axes[0, 0]
    if s.intake:
        ax.barh(range(len(INTAKE_COLUMNS)), [s.intake["nulls"][k] / s.intake["rows"] for k in INTAKE_COLUMNS], color=["#ae6b21" if k in OMITTED else "#276f9b" for k in INTAKE_COLUMNS])
        ax.set_yticks(range(len(INTAKE_COLUMNS)), INTAKE_COLUMNS, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Null fraction among every retained source row")
    else: _na(ax, "N/A — optional intake profile was not supplied")
    g.save(fig, "intake_nulls", "Nulls across all raw transcript-schema columns", "Every raw column is listed. Orange marks text replaced in slim storage by exact UTF8 hashes and lengths; raw payloads are retained separately. Empty strings are not nulls.", "Intake integrity and provenance")

    fig, axes = _axes(width=13, height=6); ax = axes[0, 0]; ax.axis("off")
    if s.intake:
        lines = [f"All source rows retained: {s.intake['rows']:,}", f"Source files: {s.intake['source_files']}",
                 f"Slim bytes: {s.intake['slim']['bytes']:,}", "Slim SHA256: " + s.intake["slim"]["sha256"],
                 "Full transcript strings replaced by exact hashes and lengths: input, output, reasoning",
                 "Each row retains source file, row index, payload hash and content hash.",
                 "Ordered choices hash is preserved for exact-version joins.",
                 "Null differs from an empty UTF8 string; no value normalization."]
        ax.text(.02, .95, "\n\n".join(lines), transform=ax.transAxes, va="top", fontsize=10)
    else: _na(ax, "N/A — optional intake profile was not supplied")
    g.save(fig, "intake_provenance", "Storage and row-provenance contract", "This is a descriptive source-identity check, not a new outcome analysis. File hashes for every rendered input and artifact are in figure_manifest.json.", "Intake integrity and provenance")


def _paper_style(axes) -> None:
    for ax in np.asarray(axes).flat:
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(12); ax.yaxis.label.set_size(12)
        ax.title.set_fontsize(13)


def paper_example_cells(s: AuditFigures) -> dict[str, dict | None]:
    """A declared descriptive display rule, never used for estimation or testing."""
    result = {}
    for dataset in PAPER_DATASETS:
        eligible = [c for c in s.group("confirmation", dataset) if c["eligible"]]
        result[dataset] = max(eligible, key=lambda c: abs(c["estimates"]["delta_total"])) if eligible else None
    return result


def render_paper_summaries(g: Gallery, s: AuditFigures) -> None:
    if s.datasets != ["bbh", "gpqa", "mmlu"]:
        raise FigureError("Paper summaries require the fixed BBH/GPQA/MMLU dataset matrix")
    if s.settings["min_clusters"]["confirmation"] != 100 or s.settings["min_overlap_coverage"] != .25:
        raise FigureError("Paper support gates must match the frozen 100-cluster / .25 protocol")
    group = "Paper summary figures"
    fig, axes = _axes(cols=2, width=12, height=6)
    shown_cells = [c for d in PAPER_DATASETS for c in s.group("confirmation", d) if c["eligible"]]
    limits = [abs(c["intervals"]["delta_total"][bound] * 100) for c in shown_cells for bound in ("low", "high")]
    bound = max(6., 1.08 * max(limits, default=0.))
    for ax, dataset in zip(axes.flat, PAPER_DATASETS):
        cells = s.group("confirmation", dataset)
        for i, cell in enumerate(cells):
            if cell["eligible"]:
                interval = cell["intervals"]["delta_total"]
                color = "#138777" if cell["material_effect"] else "#526d7a"
                ax.plot([interval["low"] * 100, interval["high"] * 100], [i, i], color=color, lw=1.4)
                ax.plot(cell["estimates"]["delta_total"] * 100, i, "o", color=color, ms=4)
            else:
                ax.text(.98, i, "N/A", transform=ax.get_yaxis_transform(), ha="right", va="center", color="#7a858b", fontsize=10)
        ax.axvspan(-100 * s.tie, 100 * s.tie, color="#cbd5dc", alpha=.25)
        ax.axvline(0, color="#607782", lw=.8)
        ax.set_xlim(-bound, bound)
        ax.set_yticks(range(len(cells)), [s.pair_label(c) for c in cells]); ax.invert_yaxis()
        ax.set_xlabel("Exact minus native gap (percentage points)")
        ax.set_title(f"{dataset.upper()}: {sum(c['eligible'] for c in cells)}/{len(cells)} with total-shift CI")
        ax.grid(axis="x", alpha=.2)
    _paper_style(axes)
    gpqa = s.group("confirmation", "gpqa"); gpqa_n = sum(c["eligible"] for c in gpqa)
    gpqa_note = (f"GPQA: all {len(gpqa)} pairs unsupported (no eligible total-shift CI)." if not gpqa_n else
                 f"GPQA: {gpqa_n}/{len(gpqa)} pairs eligible; fixed paper forest panels show BBH and MMLU.")
    g.save(fig, "paper_total_shift", "Held-out confirmation: every BBH and MMLU model-pair shift",
           "All 28 pairs per panel, in frozen order; green marks the source material-effect rule. Gray band: frozen tie region. " + gpqa_note +
           " Existing source simultaneous total-shift family retained; intervals were not recalculated for these panels.", group)
    g.figures[-1]["selection"] = "Every pair in the two fixed paper datasets, with GPQA eligibility explicitly reported"

    fig, axes = _axes(cols=3, width=12, height=6)
    for ax, dataset in zip(axes.flat, s.datasets):
        cells = s.group("confirmation", dataset)
        for flag, marker, color in ((False, "x", "#8c989e"), (True, "o", "#138777")):
            selected = [c for c in cells if c["support_eligible"] == flag]
            ax.scatter([c["counts"]["u1_clusters"] for c in selected], [c["counts"]["overlap_question_coverage"] for c in selected],
                       marker=marker, color=color, s=44, alpha=.8, label="Support eligible" if flag else "Support ineligible")
        ax.axvline(100, color="#276f9b", ls="--", lw=1.2); ax.axhline(.25, color="#276f9b", ls="--", lw=1.2)
        ax.set_xlim(-max(5., max((c["counts"]["u1_clusters"] for c in cells), default=0) * .03),
                    max(125., max((c["counts"]["u1_clusters"] for c in cells), default=0) * 1.08))
        ax.set_ylim(-.04, 1.08); ax.set_xlabel("U1 question clusters"); ax.set_ylabel("U1 / U0 question coverage")
        supported = sum(c["support_eligible"] for c in cells); primary = sum(c["eligible"] for c in cells)
        ax.set_title(f"{dataset.upper()}\n{len(cells)} pairs | support: {supported} | CI: {primary}")
        ax.text(.98, .27, "25% gate", transform=ax.get_yaxis_transform(), ha="right", fontsize=10, color="#276f9b")
        ax.grid(alpha=.15)
    _paper_style(axes); axes[0, 0].legend(loc="lower right", fontsize=10)
    g.save(fig, "paper_support", "Held-out confirmation support: all 84 declared comparisons",
           "Every pair is plotted, including zero support; coincident points can overlap. Gates are fixed at 100 U1 clusters and 25% question coverage. Panel counts retain every comparison. Support eligibility and availability of a valid total-shift CI are distinct; no trend is fitted.", group)

    choices = paper_example_cells(s)
    fig, axes = _axes(rows=2, cols=2, width=12, height=6)
    selected_cells = [cell for cell in choices.values() if cell is not None]
    gap_values = [100 * c["estimates"][m] for c in selected_cells for m in ("native_u0_gap", "exact_u1_gap")]
    gap_min, gap_max = min([0.] + gap_values), max([0.] + gap_values)
    gap_padding = max(4., (gap_max - gap_min) * .3)
    for col, dataset in enumerate(PAPER_DATASETS):
        top, bottom = axes[:, col]; cell = choices[dataset]
        if cell is None:
            _na(top, "N/A - no eligible comparison"); _na(bottom, "N/A - no total-shift interval")
            top.set_title(dataset.upper()); continue
        values = [100 * cell["estimates"][metric] for metric in ("native_u0_gap", "exact_u1_gap")]
        top.bar([0, 1], values, color=["#718096", "#138777"], width=.55)
        for x, value in enumerate(values):
            top.annotate(f"{value:+.1f} pp", (x, value), xytext=(0, 5 if value >= 0 else -6), textcoords="offset points",
                         ha="center", va="bottom" if value >= 0 else "top", fontsize=12, fontweight="bold")
        top.axhline(0, color="#607782", lw=.8); top.set_ylim(gap_min - gap_padding, gap_max + gap_padding)
        top.set_xticks([0, 1], ["Native / U0\npoint gap", "Exact / U1\npoint gap"])
        top.set_ylabel("ACK gap (percentage points)"); top.set_title(f"{dataset.upper()}: {s.pair_label(cell)}")
        interval = cell["intervals"]["delta_total"]; estimate = 100 * cell["estimates"]["delta_total"]
        lo, hi = 100 * interval["low"], 100 * interval["high"]
        bottom.plot([lo, hi], [0, 0], color="#138777", lw=2); bottom.plot(estimate, 0, "o", color="#138777", ms=7)
        bottom.axvline(0, color="#607782", lw=.8); bottom.set_yticks([]); bottom.set_ylim(-.7, .7)
        padding = max(2., (hi - lo) * .18); bottom.set_xlim(min(0., lo) - padding, max(0., hi) + padding)
        bottom.set_xlabel("Total shift with source simultaneous CI (pp)")
        bottom.text(.02, .95, f"{estimate:+.2f} pp  [{lo:+.2f}, {hi:+.2f}]", transform=bottom.transAxes, va="top", fontsize=11)
        counts = cell["counts"]
        bottom.set_title(f"U0/U1 questions: {counts['u0_questions']:,}/{counts['u1_questions']:,}; U1 clusters: {counts['u1_clusters']:,}", fontsize=10)
    _paper_style(axes)
    g.save(fig, "paper_examples", "Two descriptive examples: point gaps and total-shift intervals",
           "Within each fixed BBH/MMLU dataset, select the largest absolute eligible total shift; ties follow frozen pair order. Selection is descriptive after confirmation. Bars are point gaps without invented gap CIs. Only the separately labeled total shift receives its existing simultaneous CI; all pairs remain in the complete forest.", group)
    g.figures[-1].update(selection=PAPER_EXAMPLE_RULE, selected_cell_ids=[cell["cell_id"] for cell in selected_cells])

    fig, axes = _axes(cols=2, width=12, height=6)
    left, right = axes[0]
    for ax in (left, right): ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    left.set_title("Contrastive targets can have empty shared support", fontsize=13, pad=16)
    for y, name, baseline, target in ((.73, "Hypothetical model X", "A", "B"), (.44, "Hypothetical model Y", "B", "A")):
        left.add_patch(Rectangle((.02, y - .105), .42, .21, facecolor="#eef3f5", edgecolor="#8ba1ac"))
        left.text(.23, y, name + "\nBaseline " + baseline, ha="center", va="center", fontsize=12)
        left.add_patch(FancyArrowPatch((.46, y), (.61, y), arrowstyle="-|>", mutation_scale=16, color="#276f9b", lw=1.5))
        left.add_patch(Rectangle((.62, y - .105), .35, .21, facecolor="#e7f4ef", edgecolor="#138777"))
        left.text(.795, y, "Allowed target\n{" + target + "}", ha="center", va="center", fontsize=12)
    left.text(.5, .20, "Shared contrastive target:\n{B} intersection {A} = empty", ha="center", va="center", fontsize=14, fontweight="bold", color="#9b4f2e")
    left.text(.5, .04, "Reweighting cannot create an unavailable target.", ha="center", va="center", fontsize=11)
    right.set_title("A fully crossed alternative includes own-baseline cues", fontsize=13, pad=16)
    right.text(.58, .88, "Cued target A          Cued target B", ha="center", va="center", fontsize=12)
    for row, (y, label) in enumerate(((.68, "X: baseline A"), (.40, "Y: baseline B"))):
        right.text(.01, y, label, ha="left", va="center", fontsize=11)
        for col, x in enumerate((.46, .80)):
            own = row == col
            right.add_patch(Rectangle((x - .145, y - .115), .29, .23, facecolor="#e9eef7" if own else "#e7f4ef", edgecolor="#718096"))
            right.text(x, y, "Cued target =\nown baseline" if own else "Contrastive\ncue", ha="center", va="center", fontsize=11)
    right.text(.5, .17, "Observe both targets for both models;\nreport common-target and contrastive policies separately.", ha="center", va="center", fontsize=11)
    right.text(.5, .04, "An uncued baseline does not replace a cued own-baseline cell.", ha="center", va="center", fontsize=10)
    g.save(fig, "paper_design_support", "An elementary support constraint and an alternative evaluation design",
           "Hypothetical binary question; aligned answer identities and valid frozen baseline references. Contrastive means different from each model's recorded baseline, not necessarily objectively wrong. The fully crossed design includes cued own-baseline conditions. Source-only assumptions/proof: " + DESIGN_NOTE_PATH + ".", group)
    g.figures[-1]["selection"] = "Hypothetical source-only binary illustration; no observed-model, data or effect selection"


def render(gallery: Gallery, study: AuditFigures) -> None:
    gallery.expected = delivery_spec(study.models, study.datasets)
    gallery.notes.append("Model-pair signs use the frozen model order: model A minus model B; symmetric heatmap halves are display mirrors.")
    source_tables(gallery, study)
    render_cells(gallery, study)
    render_pairs(gallery, study)
    render_globals(gallery, study)
    render_paper_summaries(gallery, study)
    gallery.finish()
