"""Render complete R14 response-forecast evidence without fitting or network use.

The independent unit is the data draw. All uncertainty shown here is an
observed range, never a confidence interval; OOF predictions are discovery.
The shared Gallery implementation is imported without modifying it.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import hashlib
import os
from pathlib import Path
import re
import sys
from typing import Any

# Set before importing the shared Matplotlib/NumPy renderer.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_v2_research import Gallery, plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = ("B0", "B1", "B2", "B3", "B4", "A")
COLORS = {"B0": "#84939e", "B1": "#8768a9", "B2": "#c08e37", "B3": "#4573a0", "B4": "#b86667", "A": "#15887b"}
FEATURES = (("train_mse", "Baseline train MSE"), ("collision_mse_floor", "Empirical train representation MSE floor"),
            ("trajectory_weighted_per_tree_efc_proxy", "Loss-weighted per-new-tree EFC proxy"),
            ("frozen_eval_noise_gain", "Frozen evaluation operator Frobenius gain"),
            ("frozen_train_noise_gain", "Frozen train operator Frobenius gain"),
            ("largest_train_singular_value", "Largest train operator singular value"))
UNIT_NOTE = "Independent unit: data draw; contexts, amplitudes, directions and evaluation rows are correlated within draw. Descriptive discovery, no confidence intervals or double-descent claim."


class ResponseGallery(Gallery):
    """Relabel the derived gallery locally; shared R13 source stays untouched."""
    def __init__(self, output: Path):
        if output.exists() and any(output.iterdir()):
            raise ValueError("R14 gallery output must be new or empty; existing evidence is never overwritten")
        super().__init__(output)
        self.expected_delivery = None

    def finish(self) -> None:
        if self.expected_delivery is None:
            raise ValueError("A complete R14 delivery matrix must be declared before finish")
        validate_delivery(self.figures, self.tables, self.expected_delivery)
        super().finish()
        index = self.output / "index.html"
        page = index.read_text(encoding="utf-8")
        page = page.replace("<title>R10–R12 evidence gallery</title>", "<title>R14 response forecast evidence</title>")
        page = page.replace("Acceleration audit: complete evidence gallery", "R14: finite-label response forecast discovery")
        page = page.replace("Static PNG/SVG, offline interactive companions and source tables.", "Complete static PNG/SVG pairs and machine-readable source tables for R14.")
        index.write_text(page, encoding="utf-8")
        manifest_path = self.output / "figure_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["shared_gallery_renderer"] = {"path": manifest["renderer"], "sha256": manifest["renderer_sha256"]}
        manifest["renderer"] = str(Path(__file__).resolve())
        manifest["renderer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["study_id"] = "R14"
        manifest["independent_unit"] = "data draw; discovery only"
        manifest["interpretation"] = "Fixed R14 source evidence only; observed ranges are not confidence intervals. OOF is discovery, not independent confirmation. No model fitting, source mutation, novelty certificate or double-descent claim."
        manifest["expected_delivery"] = self.expected_delivery
        manifest["source_implementation_authorship"] = ["/root/literature", "/root/r14_figures"]
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def number(value: Any) -> float:
    return float(value) if isinstance(value, (float, int, np.floating, np.integer)) and np.isfinite(value) else float("nan")


def average(values) -> float:
    values = np.asarray([number(v) for v in values])
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")


def model_record(row, model):
    return row.get("models", {}).get(model, {})


@dataclass
class Study:
    analysis: dict
    rows: list[dict]
    contexts: list[dict]
    directions: list[dict]
    arrays: dict[str, np.ndarray]

    @property
    def seeds(self): return sorted({row["seed"] for row in self.rows})
    @property
    def teachers(self): return sorted({row["teacher"] for row in self.rows})
    @property
    def amplitudes(self): return sorted({row["epsilon"] for row in self.rows})
    @property
    def prefixes(self): return sorted({row["prefix"] for row in self.rows})

    @property
    def oof_available(self):
        return self.analysis["forecast"]["status"] == "complete_discovery_cross_validation"


def finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)) or not np.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"Finite {'nonnegative ' if nonnegative else ''}source value required: {label}")
    return float(value)


def delivery_spec(s: Study) -> dict:
    """Declare every figure and table before rendering, including the invalid branch."""
    ids = [f"r14_{phase}_{context['id']}" for context in s.contexts for phase in ("response", "pilot")]
    ids += ["r14_cheap_vs_full", "r14_early_late_gap"]
    ids += [f"r14_prefix_discrepancy_{teacher}_e{eps:g}" for teacher in s.teachers for eps in s.amplitudes]
    ids += [f"r14_decomposition_{teacher}" for teacher in s.teachers]
    ids += [f"r14_feature_{key}" for key, _ in FEATURES]
    ids += ["r14_train_vs_evaluation_discrepancy", "r14_geometry_by_prefix", "r14_representation_realization", "r14_baseline_clean_quality", "r14_perturbed_clean_quality"]
    if s.oof_available:
        ids += [f"r14_oof_{kind}_{model}" for model in MODELS for kind in ("prediction", "residual")]
        ids += ["r14_oof_absolute_error_matrix", "r14_mae_aggregate", "r14_mae_by_draw", "r14_mae_by_teacher", "r14_A_error_advantage"]
        ids += [f"r14_oof_context_{teacher}_s{seed}" for teacher in s.teachers for seed in s.seeds]
    else:
        ids += ["r14_oof_unavailable"]
    ids += ["r14_runtime_components", "r14_runtime_direction_distributions", "r14_runtime_by_context", "r14_declared_decision", "r14_cost_scopes"]
    raw_count = sum(len(s.arrays[r["array_keys"][split]["full_delta"]]) for r in s.directions for split in ("train", "evaluation"))
    tables = {"r14_all_forecast_rows": len(s.rows), "r14_all_contexts": len(s.contexts),
              "r14_all_direction_metrics": len(s.directions), "r14_all_raw_responses": raw_count,
              "r14_all_timing_components": len(timing_rows(s)), "r14_cost_scopes": len(cost_rows(s))}
    if s.oof_available:
        tables["r14_oof_fold_fits"] = len(s.analysis["forecast"]["folds"])
    return {"static_png_svg_pairs": len(ids), "interactive_html": 0, "figure_ids": ids,
            "source_tables": len(tables), "table_rows": {f"tables/{key}.csv": value for key, value in tables.items()},
            "oof_available": s.oof_available}


def validate_delivery(figures: list[dict], tables: list[dict], spec: dict) -> None:
    ids = [r["id"] for r in figures]
    if len(ids) != len(set(ids)) or set(ids) != set(spec["figure_ids"]) or len(ids) != spec["static_png_svg_pairs"]:
        raise ValueError("R14 complete declared figure matrix differs")
    paths = [path for row in figures for path in row["paths"].values()]
    if len(paths) != len(set(paths)) or any(set(row["paths"]) != {"png", "svg"} for row in figures):
        raise ValueError("Every R14 figure must have one distinct PNG/SVG pair")
    actual = {r["path"]: r["rows"] for r in tables}
    if len(actual) != len(tables) or actual != spec["table_rows"] or len(tables) != spec["source_tables"]:
        raise ValueError("R14 complete declared source-table matrix differs")


def load_study(g: Gallery, directory: Path) -> Study:
    analysis = g.load(directory / "analysis.json")
    rows = g.jsonl(directory / "forecast_rows.jsonl")
    contexts = g.jsonl(directory / "contexts.jsonl")
    directions = g.jsonl(directory / "directions.jsonl")
    if not rows or not contexts or not directions or len({r["id"] for r in rows}) != len(rows) or len({r["id"] for r in contexts}) != len(contexts) or len({r["id"] for r in directions}) != len(directions):
        raise ValueError("Complete nonempty unique R14 evidence is required")
    if any(not isinstance(r["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", r["id"]) for r in contexts):
        raise ValueError("Context IDs must be safe single filename components")
    for key, count in (("n_contexts", len(contexts)), ("n_forecast_rows", len(rows)), ("n_directions", len(directions)), ("n_fits", len(contexts)+len(directions))):
        if analysis.get(key) != count: raise ValueError(f"R14 source count differs: {key}")
    expected = {(r["context_id"], r["epsilon"]) for r in rows}
    if len(expected) != len(rows) or {r["context_id"] for r in rows} != {r["id"] for r in contexts}:
        raise ValueError("R14 context/amplitude matrix is incomplete or duplicated")
    seeds = {r["seed"] for r in rows}; teachers = {r["teacher"] for r in rows}
    prefixes = {r["prefix"] for r in rows}; amplitudes = {r["epsilon"] for r in rows}
    if any(set(analysis.get(key, [])) != values for key, values in (("seeds", seeds), ("teachers", teachers), ("prefix_budgets", prefixes), ("epsilon_values", amplitudes))):
        raise ValueError("R14 evidence axes differ from source analysis")
    context_by_id = {r["id"]: r for r in contexts}
    actual_contexts = {(r["seed"], r["teacher"], r["prefix"]) for r in contexts}
    actual_rows = {(r["seed"], r["teacher"], r["prefix"], r["epsilon"]) for r in rows}
    if actual_contexts != {(seed, teacher, prefix) for seed in seeds for teacher in teachers for prefix in prefixes} or actual_rows != {(seed, teacher, prefix, eps) for seed in seeds for teacher in teachers for prefix in prefixes for eps in amplitudes}:
        raise ValueError("R14 Cartesian forecast matrix is incomplete")
    for r in rows:
        context = context_by_id[r["context_id"]]
        if any(r[key] != context[key] for key in ("seed", "teacher", "prefix")) or type(r.get("valid_directions")) is not bool:
            raise ValueError("Forecast row identity/validity differs from its context")
        if finite(r["epsilon"], "epsilon", nonnegative=True) == 0:
            raise ValueError("Positive perturbation amplitude required")
        for key in ("cheap_discrepancy", "full_discrepancy", *[key for key, _ in FEATURES]):
            finite(r[key], key, nonnegative=True)
        if not 0 <= r["trajectory_weighted_per_tree_efc_proxy"] <= 1:
            raise ValueError("Per-tree EFC adaptation must be in [0,1]")
    for context in contexts:
        for key in ("base_clean_train_mse", "base_clean_evaluation_mse"):
            finite(context[key], key, nonnegative=True)
        if type(context.get("prefix_unique_thresholds")) is not int or not 0 <= context["prefix_unique_thresholds"] <= context["prefix"] or type(context.get("n_train_signatures")) is not int or context["n_train_signatures"] < 1:
            raise ValueError("Actual unique thresholds and train signatures must be recorded")
        for key, value in context["timings"].items():
            finite(value, f"context timing {key}", nonnegative=True)
    groups = defaultdict(list)
    for r in directions:
        if (r["context_id"], r["epsilon"]) not in expected or r["phase"] not in ("pilot", "response"):
            raise ValueError("Direction is outside the registered forecast matrix")
        context = context_by_id[r["context_id"]]
        if any(r[key] != context[key] for key in ("seed", "teacher", "prefix")) or type(r.get("valid")) is not bool:
            raise ValueError("Direction identity/validity differs from its context")
        for key in ("clean_train_mse", "clean_evaluation_mse"):
            finite(r[key], key, nonnegative=True)
        for key, value in r["timings"].items():
            finite(value, f"direction timing {key}", nonnegative=True)
        groups[(r["context_id"], r["epsilon"], r["phase"])].append(r)
    for context, eps in expected:
        for phase, count in (("pilot", 2), ("response", 4)):
            group = groups[(context, eps, phase)]
            if len(group) != count or {r["direction"] for r in group} != set(range(count)):
                raise ValueError("R14 requires every declared two-pilot/four-response direction")
    arrays = {}
    split_sizes = {}
    with np.load(g.source(directory / "predictions.npz"), allow_pickle=False) as archive:
        for r in directions:
            for split in ("train", "evaluation"):
                for kind in ("full_delta", "frozen_delta", "structural_delta"):
                    key = r["array_keys"][split][kind]
                    values = np.asarray(archive[key], dtype=float)
                    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                        raise ValueError("Finite complete response arrays required")
                    arrays[key] = values
                keys = r["array_keys"][split]
                full, frozen, structural = (arrays[keys[k]] for k in ("full_delta", "frozen_delta", "structural_delta"))
                if full.shape != frozen.shape or full.shape != structural.shape or not np.allclose(full - frozen, structural, atol=1e-10, rtol=1e-10):
                    raise ValueError("Raw full/frozen/structural arrays do not close")
                if split in split_sizes and len(full) != split_sizes[split]:
                    raise ValueError("Response arrays have inconsistent split sizes")
                split_sizes[split] = len(full)
                metrics = r["metrics"][split]
                if metrics["n_evaluation"] != len(full):
                    raise ValueError("Response array size differs from its metric denominator")
                denominator = len(full) * r["epsilon"] ** 2
                expected_metrics = {"normalized_full": full @ full / denominator,
                                    "normalized_frozen": frozen @ frozen / denominator,
                                    "normalized_structural_discrepancy": structural @ structural / denominator,
                                    "normalized_twice_cross": 2 * frozen @ structural / denominator}
                for metric, value in expected_metrics.items():
                    source_value = finite(metrics[metric], metric)
                    if not np.isclose(source_value, value, atol=1e-10, rtol=1e-10):
                        raise ValueError(f"Response metric differs from all raw values: {metric}")
    for r in rows:
        for phase, key in (("pilot", "cheap_discrepancy"), ("response", "full_discrepancy")):
            value = np.mean([d["metrics"]["evaluation"]["normalized_structural_discrepancy"] for d in groups[(r["context_id"], r["epsilon"], phase)]])
            if not np.isclose(r[key], value, atol=1e-10, rtol=1e-10):
                raise ValueError("Context/amplitude endpoint differs from its complete direction group")
        group = [d for phase in ("pilot", "response") for d in groups[(r["context_id"], r["epsilon"], phase)]]
        if r["valid_directions"] != all(d["valid"] for d in group):
            raise ValueError("Context/amplitude validity differs from its directions")
    if any(context["n_train_signatures"] > split_sizes["train"] for context in contexts):
        raise ValueError("Unique training signatures cannot exceed training rows")
    invalid = sum(not r["valid"] for r in directions)
    if analysis.get("invalid_zero_delta_direction_count") != invalid:
        raise ValueError("Invalid-direction count differs from complete raw matrix")
    status = analysis.get("forecast", {}).get("status")
    if status == "complete_discovery_cross_validation":
        if invalid:
            raise ValueError("Complete OOF analysis cannot contain an invalid direction")
        for r in rows:
            target = finite(r["target_u"], "target_u")
            if set(r.get("models", {})) != set(MODELS) or not np.isclose(target, np.log10(1+r["full_discrepancy"]), atol=1e-12, rtol=0):
                raise ValueError("Every complete forecast row needs all declared model outputs and target")
            for model in MODELS:
                prediction, residual, error = (finite(r["models"][model][key], key) for key in ("prediction_u", "residual_u", "absolute_error_u"))
                if not np.isclose(residual, prediction-target, atol=1e-12, rtol=0) or not np.isclose(error, abs(residual), atol=1e-12, rtol=0):
                    raise ValueError("OOF residual/absolute error differs from recorded prediction")
        folds = analysis["forecast"]["folds"]
        if len(folds) != len(seeds)*len(MODELS) or {(r["held_out_seed"], r["model"]) for r in folds} != {(seed, model) for seed in seeds for model in MODELS}:
            raise ValueError("Complete OOF fold/model matrix required")
        summaries = analysis["forecast"]["summaries"]
        for model in MODELS:
            draw_maes = {str(seed): float(np.mean([r["models"][model]["absolute_error_u"] for r in rows if r["seed"] == seed])) for seed in seeds}
            teacher_maes = {teacher: float(np.mean([np.mean([r["models"][model]["absolute_error_u"] for r in rows if r["seed"] == seed and r["teacher"] == teacher]) for seed in seeds])) for teacher in teachers}
            source = summaries[model]
            if not np.isclose(finite(source["mae_u"], "aggregate MAE", nonnegative=True), np.mean(list(draw_maes.values())), atol=1e-12, rtol=0) or any(not np.isclose(source["draw_mae_u"][key], value, atol=1e-12, rtol=0) for key, value in draw_maes.items()) or any(not np.isclose(source["teacher_mae_u"][key], value, atol=1e-12, rtol=0) for key, value in teacher_maes.items()):
                raise ValueError("Source-computed aggregate MAE differs from all OOF rows")
    elif status == "inconclusive_invalid_direction":
        if not invalid or analysis.get("decision", {}).get("go") is not False or any(r.get("models") for r in rows):
            raise ValueError("Invalid-direction OOF must stay explicitly inconclusive with no invented forecasts")
    else:
        raise ValueError("R14 OOF is either complete discovery or explicit invalid-direction inconclusive")
    if (directory / "environment.json").exists(): g.load(directory / "environment.json")
    return Study(analysis, rows, contexts, directions, arrays)


def panels(study, *, size=(13, 8)):
    fig, axes = plt.subplots(len(study.teachers), len(study.amplitudes), figsize=size, squeeze=False, constrained_layout=True)
    for i, teacher in enumerate(study.teachers):
        for j, eps in enumerate(study.amplitudes):
            axes[i, j].set_title(f"{teacher}; ε={eps:g}")
    return fig, axes


def select(rows, teacher, eps): return [r for r in rows if r["teacher"] == teacher and r["epsilon"] == eps]


def by_prefix(ax, rows, key, label, color, *, transform=False):
    xs = sorted({r["prefix"] for r in rows})
    arrays = [np.asarray([number(r.get(key)) for r in rows if r["prefix"] == x]) for x in xs]
    if transform: arrays = [np.log10(1 + a) for a in arrays]
    ax.plot(xs, [average(a) for a in arrays], "o-", label=label, color=color, markersize=4)
    lo = [np.min(a[np.isfinite(a)]) if np.isfinite(a).any() else np.nan for a in arrays]
    hi = [np.max(a[np.isfinite(a)]) if np.isfinite(a).any() else np.nan for a in arrays]
    ax.fill_between(xs, lo, hi, color=color, alpha=.12)
    ax.set_xscale("log", base=2); ax.set_xticks(xs, [str(x) for x in xs]); ax.grid(alpha=.18)


def source_tables(g, s):
    g.csv("r14_all_forecast_rows", s.rows, "Every context/amplitude and OOF prediction; nested fields preserved as JSON. Data draw is the independent cluster.")
    g.csv("r14_all_contexts", s.contexts, "Every context, including baseline quality, fixed geometry, costs and amplitude metadata.")
    g.csv("r14_all_direction_metrics", s.directions, "All pilot/full directions, validity flags, normalized norms, signed cross terms, keys and measured costs.")
    raw = []
    for r in s.directions:
        for split in ("train", "evaluation"):
            keys = r["array_keys"][split]
            values = [s.arrays[keys[k]] for k in ("full_delta", "frozen_delta", "structural_delta")]
            for point, (full, frozen, structural) in enumerate(zip(*values)):
                raw.append({"direction_id": r["id"], "context_id": r["context_id"], "seed": r["seed"], "teacher": r["teacher"], "prefix": r["prefix"],
                            "epsilon": r["epsilon"], "phase": r["phase"], "direction": r["direction"], "valid": r["valid"], "split": split,
                            "row_in_split": point, "full_delta": full, "frozen_delta": frozen, "structural_delta": structural})
    g.csv("r14_all_raw_responses", raw, "Every row of every train/evaluation response array; full minus frozen equals structural. No selected examples.")
    g.csv("r14_oof_fold_fits", s.analysis.get("forecast", {}).get("folds", []), "All fitted discovery fold models; standardization is within training draws only.")


def render_direction_panels(g, s):
    grouped = defaultdict(list)
    for r in s.directions: grouped[(r["context_id"], r["epsilon"], r["phase"])].append(r)
    for context in s.contexts:
        for phase, count in (("response", 4), ("pilot", 2)):
            fig, axes = plt.subplots(count, len(s.amplitudes), figsize=(12, 2.8 * count), squeeze=False, constrained_layout=True)
            for j, eps in enumerate(s.amplitudes):
                group = sorted(grouped[(context["id"], eps, phase)], key=lambda r: r["direction"])
                for i, r in enumerate(group):
                    ax = axes[i, j]; keys = r["array_keys"]["evaluation"]
                    full, frozen = s.arrays[keys["full_delta"]], s.arrays[keys["frozen_delta"]]
                    ax.scatter(frozen, full, s=5, alpha=.45, linewidths=0, color="#167d9b" if phase == "response" else "#c56739")
                    lo, hi = min(full.min(), frozen.min()), max(full.max(), frozen.max())
                    if lo == hi: lo, hi = lo-.01, hi+.01
                    ax.plot([lo, hi], [lo, hi], "k--", linewidth=.8)
                    d = number(r.get("metrics", {}).get("evaluation", {}).get("normalized_structural_discrepancy"))
                    ax.set_title(f"ε={eps:g}; direction {r['direction']}; {'D' if phase=='response' else 'C'}={d:.3g}" + ("; INVALID" if not r["valid"] else ""), fontsize=9)
                    ax.set_xlabel("Conditional frozen response Δprediction"); ax.set_ylabel("Full-refit response Δprediction"); ax.grid(alpha=.12)
            g.save(fig, f"r14_{phase}_{context['id']}", f"R14 — {context['teacher']}, draw {context['seed']}, prefix {context['prefix']}: all {phase} directions",
                   "Every evaluation row is plotted. Dashed diagonal means exact frozen agreement; these rows/directions are not independent data draws. Invalid directions remain explicit. " + UNIT_NOTE,
                   "R14 every full direction" if phase == "response" else "R14 every cheap direction", f"All declared {phase} directions and both amplitudes for context {context['id']}")


def render_measurements(g, s):
    cmap = plt.get_cmap("viridis"); pc = {p: cmap(i / max(1, len(s.prefixes)-1)) for i, p in enumerate(s.prefixes)}
    fig, axes = panels(s)
    for i, teacher in enumerate(s.teachers):
        for j, eps in enumerate(s.amplitudes):
            ax = axes[i, j]; rows = select(s.rows, teacher, eps)
            for p in s.prefixes:
                r = [r for r in rows if r["prefix"] == p]
                ax.scatter([np.log10(1+number(x.get("cheap_discrepancy"))) for x in r], [np.log10(1+number(x.get("full_discrepancy"))) for x in r], color=pc[p], label=f"prefix {p}")
            ax.set_xlabel("v = log10(1 + cheap discrepancy C32)"); ax.set_ylabel("u = log10(1 + full discrepancy D512)"); ax.grid(alpha=.15); ax.legend(fontsize=7)
    g.save(fig, "r14_cheap_vs_full", "R14 — early versus late conditional-response error", "All context/amplitude endpoints. Each dot is one context, clustered into data draws; no regression is fitted by this renderer. " + UNIT_NOTE, "R14 measurements")
    fig, axes = panels(s)
    for i, teacher in enumerate(s.teachers):
        for j, eps in enumerate(s.amplitudes):
            rows = select(s.rows, teacher, eps); ax = axes[i, j]
            for seed in s.seeds:
                r = sorted([r for r in rows if r["seed"] == seed], key=lambda x: x["prefix"])
                ax.plot([x["prefix"] for x in r], [np.log10(1+number(x.get("full_discrepancy")))-np.log10(1+number(x.get("cheap_discrepancy"))) for x in r], "o-", label=f"draw {seed}")
            ax.axhline(0, color="black", linestyle="--", linewidth=.8); ax.set_xscale("log", base=2); ax.set_xlabel("Nested candidate prefix"); ax.set_ylabel("u − v (late minus early transformed error)"); ax.legend(fontsize=7); ax.grid(alpha=.15)
    g.save(fig, "r14_early_late_gap", "R14 — every draw's early/late calibration gap", "Same fixed amplitudes and prefixes, no selected contexts. This is descriptive and is not the fitted B4 offset. " + UNIT_NOTE, "R14 measurements")
    for teacher in s.teachers:
        for eps in s.amplitudes:
            fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True); rows = select(s.rows, teacher, eps)
            by_prefix(ax, rows, "cheap_discrepancy", "C32: two cheap directions", "#c56739", transform=True)
            by_prefix(ax, rows, "full_discrepancy", "D512: four new full directions", "#167d9b", transform=True)
            ax.set_xlabel("Nested candidate prefix (a declared intervention)"); ax.set_ylabel("log10(1 + normalized squared discrepancy)"); ax.legend(fontsize=8)
            g.save(fig, f"r14_prefix_discrepancy_{teacher}_e{eps:g}", f"R14 — {teacher}, ε={eps:g}: error by prefix", "Lines: context means across draws. Shading: observed draw min–max, NOT a confidence interval. This is not a test-risk/double-descent curve. " + UNIT_NOTE, "R14 measurements")
    norms = (("normalized_full", "Full response norm"), ("normalized_frozen", "Frozen response norm"), ("normalized_structural_discrepancy", "Approximation discrepancy norm"), ("normalized_twice_cross", "Signed twice-cross term"))
    for teacher in s.teachers:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        row_ids = sorted({r["context_id"] for r in s.rows if r["teacher"] == teacher})
        for ax, (metric, title) in zip(axes.flat, norms):
            values = np.asarray([[average(r.get("metrics", {}).get("evaluation", {}).get(metric) for r in s.directions if r["context_id"] == context and r["epsilon"] == eps and r["phase"] == "response") for eps in s.amplitudes] for context in row_ids])
            transformed = np.sign(values) * np.log10(1 + np.abs(values))
            limit = np.max(np.abs(transformed[np.isfinite(transformed)])) if np.isfinite(transformed).any() else 1
            image = ax.imshow(transformed, aspect="auto", cmap="RdBu_r", vmin=-max(limit, 1e-12), vmax=max(limit, 1e-12))
            ax.set_xticks(range(len(s.amplitudes)), [f"ε={e:g}" for e in s.amplitudes]); ax.set_yticks(range(len(row_ids)), row_ids, fontsize=7); ax.set_title(title); fig.colorbar(image, ax=ax, label="sign(x) log10(1+|x|)", shrink=.7)
        g.save(fig, f"r14_decomposition_{teacher}", f"R14 — {teacher}: complete norm decomposition", "Means over four full directions. The signed cross term can be negative; discrepancy is not an additive extra-variance estimate. Colors explicitly transform magnitudes; missing values remain missing. " + UNIT_NOTE, "R14 decomposition")
    fig, axes = panels(s)
    for i, teacher in enumerate(s.teachers):
        for j, eps in enumerate(s.amplitudes):
            ax = axes[i, j]
            for phase, color in (("pilot", "#c56739"), ("response", "#167d9b")):
                group = [r for r in s.rows if r["teacher"] == teacher and r["epsilon"] == eps]
                train, evaluation = [], []
                for row in group:
                    directions = [d for d in s.directions if d["context_id"] == row["context_id"] and d["epsilon"] == eps and d["phase"] == phase]
                    train.append(np.log10(1+average(d["metrics"]["train"]["normalized_structural_discrepancy"] for d in directions)))
                    evaluation.append(np.log10(1+average(d["metrics"]["evaluation"]["normalized_structural_discrepancy"] for d in directions)))
                ax.scatter(train, evaluation, color=color, label=f"{phase}: {'32' if phase == 'pilot' else '512'} trees")
            lim = max(ax.get_xlim()[1], ax.get_ylim()[1], .01)
            ax.plot([0, lim], [0, lim], "k--", linewidth=.8)
            ax.set_xlabel("log10(1 + mean training response discrepancy)")
            ax.set_ylabel("log10(1 + mean evaluation response discrepancy)")
            ax.legend(fontsize=7); ax.grid(alpha=.15)
    g.save(fig, "r14_train_vs_evaluation_discrepancy", "R14 — training versus out-of-sample response discrepancy",
           "Every context appears for both phases and amplitudes. Separate native split denominators are retained; this is not teacher prediction risk. Direction averages are descriptive, not independent repeats. " + UNIT_NOTE, "R14 decomposition")
    for key, label in FEATURES:
        fig, axes = panels(s)
        for i, teacher in enumerate(s.teachers):
            for j, eps in enumerate(s.amplitudes):
                ax=axes[i,j]
                for seed in s.seeds:
                    rows=select([r for r in s.rows if r["seed"]==seed],teacher,eps)
                    ax.scatter([number(r.get(key)) for r in rows],[np.log10(1+number(r.get("full_discrepancy"))) for r in rows],label=f"draw {seed}")
                ax.set_xlabel(label);ax.set_ylabel("u = log10(1+D512)");ax.grid(alpha=.15);ax.legend(fontsize=7)
        g.save(fig,f"r14_feature_{key}",f"R14 — diagnostic feature: {label}","All points are displayed. This plot does not select a feature, fit a correlation or establish mechanism. EFC is an explicit per-new-tree adaptation. "+UNIT_NOTE,"R14 geometry and quality")
    fig, axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,(key,label) in zip(axes.flat,FEATURES):
        for teacher,color in zip(s.teachers,("#167d9b","#c56739")):
            # Geometry is context-level; select the first amplitude by declared index to avoid counting it twice.
            rows=select(s.rows,teacher,s.amplitudes[0]);by_prefix(ax,rows,key,teacher,color)
        ax.set_title(label,fontsize=9);ax.set_xlabel("Nested candidate prefix");ax.legend(fontsize=7)
    g.save(fig,"r14_geometry_by_prefix","R14 — all fixed geometry and train-quality controls","Geometry is counted once per context, not once per amplitude. Means and observed min–max across draws, not confidence intervals. "+UNIT_NOTE,"R14 geometry and quality")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, key, label in zip(axes, ("prefix_unique_thresholds", "n_train_signatures"), ("Realized unique feature/threshold pairs", "Realized distinct training signatures")):
        for teacher, marker in zip(s.teachers, ("o", "x")):
            for seed in s.seeds:
                group = sorted([r for r in s.contexts if r["teacher"] == teacher and r["seed"] == seed], key=lambda r: r["prefix"])
                ax.plot([r["prefix"] for r in group], [r[key] for r in group], marker=marker, linewidth=.7, markersize=4, alpha=.7, label=f"{teacher}, draw {seed}")
        ax.set_xlabel("Nominal nested candidate prefix"); ax.set_ylabel(label)
        ax.set_xscale("log", base=2); ax.set_xticks(s.prefixes, [str(p) for p in s.prefixes]); ax.grid(alpha=.15)
    axes[0].legend(fontsize=7, ncol=2)
    g.save(fig, "r14_representation_realization", "R14 — nominal prefixes and actual representation counts",
           "All contexts are retained, including coincident counts for paired teachers. Tied thresholds are not replaced. Signature count and a zero collision floor do not by themselves prove finite depth-limited boosting interpolation. " + UNIT_NOTE, "R14 geometry and quality")
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,key,label in zip(axes,("base_clean_train_mse","base_clean_evaluation_mse"),("Clean teacher MSE on train inputs","Clean teacher MSE on evaluation inputs")):
        for teacher,color in zip(s.teachers,("#167d9b","#c56739")):
            rows=[r for r in s.contexts if r["teacher"]==teacher];by_prefix(ax,rows,key,teacher,color)
        ax.set_xlabel("Nested candidate prefix");ax.set_ylabel(label);ax.legend(fontsize=8)
    g.save(fig,"r14_baseline_clean_quality","R14 — secondary baseline prediction quality","Means and observed draw min–max, not confidence intervals. Synthetic teacher quality is a separate descriptive endpoint; the forecast target is approximation discrepancy. "+UNIT_NOTE,"R14 geometry and quality")
    fig,axes=panels(s)
    for i,teacher in enumerate(s.teachers):
        for j,eps in enumerate(s.amplitudes):
            ax=axes[i,j]
            for phase,color in (("pilot","#c56739"),("response","#167d9b")):
                group=[r for r in s.directions if r["teacher"]==teacher and r["epsilon"]==eps and r["phase"]==phase]
                by_prefix(ax,group,"clean_evaluation_mse",f"{phase}: {'32' if phase=='pilot' else '512'} trees",color)
            ax.set_xlabel("Nested candidate prefix");ax.set_ylabel("Clean evaluation teacher MSE");ax.legend(fontsize=7)
    g.save(fig,"r14_perturbed_clean_quality","R14 — secondary quality of every perturbed fit","Observed ranges include all directions and draws, not confidence intervals. Pilot/full curves differ in duration and perturbation directions; their difference does not identify a causal regularization effect. "+UNIT_NOTE,"R14 geometry and quality")


def render_forecasts(g,s):
    complete=all(all(key in model_record(r,m) for key in ("prediction_u","residual_u","absolute_error_u")) for r in s.rows for m in MODELS)
    if not complete:
        g.notes.append("OOF outputs are incomplete/inconclusive; no missing prediction was converted to zero or silently dropped. Raw complete matrices remain in CSV and direction panels.")
        fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True);ax.axis("off");ax.text(.03,.8,"OOF unavailable: source evidence is incomplete or invalid-direction inconclusive.\nNo forecast success is inferred.",transform=ax.transAxes,fontsize=13)
        g.save(fig,"r14_oof_unavailable","R14 — explicit unavailable/inconclusive OOF result",UNIT_NOTE,"R14 OOF discovery")
        return
    for model in MODELS:
        for kind in ("prediction","residual"):
            fig,axes=panels(s)
            for i,teacher in enumerate(s.teachers):
                for j,eps in enumerate(s.amplitudes):
                    ax=axes[i,j];rows=select(s.rows,teacher,eps)
                    for seed in s.seeds:
                        group=[r for r in rows if r["seed"]==seed];x=[r["target_u"] for r in group]
                        y=[model_record(r,model)["prediction_u" if kind=="prediction" else "residual_u"] for r in group]
                        ax.scatter(x,y,label=f"draw {seed}")
                    if kind=="prediction":
                        lo=min(min(r["target_u"],model_record(r,model)["prediction_u"]) for r in rows);hi=max(max(r["target_u"],model_record(r,model)["prediction_u"]) for r in rows);ax.plot([lo,hi],[lo,hi],"k--",linewidth=.8)
                    else:ax.axhline(0,color="black",linestyle="--",linewidth=.8)
                    ax.set_xlabel("Observed u = log10(1+D512)");ax.set_ylabel("Held-out prediction u" if kind=="prediction" else "Prediction − observed u");ax.legend(fontsize=7);ax.grid(alpha=.15)
            g.save(fig,f"r14_oof_{kind}_{model}",f"R14 — {model}: all held-out {kind}s","All teachers/prefixes/amplitudes of each held-out draw were excluded from its fit. No IID-row confidence interval is implied. "+UNIT_NOTE,"R14 OOF discovery")
    fig,axes=panels(s,size=(15,10))
    for i,teacher in enumerate(s.teachers):
        for j,eps in enumerate(s.amplitudes):
            rows=sorted(select(s.rows,teacher,eps),key=lambda r:(r["seed"],r["prefix"]));ax=axes[i,j];v=np.asarray([[model_record(r,m)["absolute_error_u"] for m in MODELS] for r in rows])
            im=ax.imshow(v,aspect="auto",cmap="magma",vmin=0);ax.set_xticks(range(6),MODELS);ax.set_yticks(range(len(rows)),[f"{r['seed']} / p{r['prefix']}" for r in rows],fontsize=7);fig.colorbar(im,ax=ax,label="Absolute OOF error in u",shrink=.7)
    g.save(fig,"r14_oof_absolute_error_matrix","R14 — every model's OOF error, every row",UNIT_NOTE,"R14 OOF discovery")
    summaries=s.analysis.get("forecast",{}).get("summaries",{})
    if all(m in summaries for m in MODELS):
        fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);ax.bar(MODELS,[summaries[m]["mae_u"] for m in MODELS],color=[COLORS[m] for m in MODELS]);ax.set_ylabel("MAE(u): equal weight within draw, then across draws");ax.grid(axis="y",alpha=.2)
        g.save(fig,"r14_mae_aggregate","R14 — declared aggregate discovery MAE","Bars are source-computed point estimates, not confidence intervals. "+UNIT_NOTE,"R14 forecast comparison")
        for grouping,values,field in (("draw",s.seeds,"draw_mae_u"),("teacher",s.teachers,"teacher_mae_u")):
            fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True);x=np.arange(len(values))
            for k,m in enumerate(MODELS):ax.bar(x+(k-2.5)*.13,[summaries[m][field][str(v)] for v in values],width=.12,color=COLORS[m],label=m)
            ax.set_xticks(x,[str(v) for v in values]);ax.set_xlabel(grouping);ax.set_ylabel("Source-computed MAE(u)");ax.legend(ncol=6);ax.grid(axis="y",alpha=.2)
            g.save(fig,f"r14_mae_by_{grouping}",f"R14 — discovery MAE by {grouping}","Both teachers occur in training folds; these plots are not unseen-teacher transfer tests. "+UNIT_NOTE,"R14 forecast comparison")
    fig,axes=panels(s,size=(15,10))
    for i,teacher in enumerate(s.teachers):
        for j,eps in enumerate(s.amplitudes):
            rows=sorted(select(s.rows,teacher,eps),key=lambda r:(r["seed"],r["prefix"]));ax=axes[i,j]
            v=np.asarray([[model_record(r,m)["absolute_error_u"]-model_record(r,"A")["absolute_error_u"] for m in ("B2","B3","B4")] for r in rows]);lim=max(float(np.max(np.abs(v))),1e-12)
            im=ax.imshow(v,aspect="auto",cmap="RdBu",vmin=-lim,vmax=lim);ax.set_xticks(range(3),("B2 − A","B3 − A","B4 − A"));ax.set_yticks(range(len(rows)),[f"{r['seed']} / p{r['prefix']}" for r in rows],fontsize=7);fig.colorbar(im,ax=ax,label="Positive = smaller absolute error for A",shrink=.7)
    g.save(fig,"r14_A_error_advantage","R14 — signed error comparison against every strong comparator","No comparator was selected after seeing endpoint. Cellwise differences are descriptive; source stop/go gates govern interpretation. "+UNIT_NOTE,"R14 forecast comparison")
    for teacher in s.teachers:
        for seed in s.seeds:
            fig,axes=plt.subplots(1,len(s.amplitudes),figsize=(13,4),squeeze=False,constrained_layout=True)
            for ax,eps in zip(axes.flat,s.amplitudes):
                rows=sorted(select([r for r in s.rows if r["seed"]==seed],teacher,eps),key=lambda r:r["prefix"])
                for m in MODELS:ax.plot([r["prefix"] for r in rows],[model_record(r,m)["residual_u"] for r in rows],"o-",label=m,color=COLORS[m],linewidth=1,markersize=3)
                ax.axhline(0,color="black",linestyle="--",linewidth=.8);ax.set_title(f"ε={eps:g}");ax.set_xscale("log",base=2);ax.set_xlabel("Nested candidate prefix");ax.set_ylabel("Held-out prediction − observed u");ax.grid(alpha=.15);ax.legend(ncol=3,fontsize=7)
            g.save(fig,f"r14_oof_context_{teacher}_s{seed}",f"R14 — all model residuals: {teacher}, draw {seed}",UNIT_NOTE,"R14 every OOF draw",f"Every amplitude/prefix/model of held-out draw {seed}, teacher {teacher}")


def timing_rows(s):
    records=[]
    for scope,items in (("context",s.contexts),("direction",s.directions)):
        for row in items:
            for key,value in row.get("timings",{}).items():
                if isinstance(value,(int,float)):
                    records.append({"scope":scope,"id":row["id"],"seed":row["seed"],"teacher":row["teacher"],"prefix":row["prefix"],"phase":row.get("phase","base/shared"),"component":key,"seconds":value})
    return records


def cost_rows(s):
    records = []
    for key, value in s.analysis["costs"].items():
        if key == "scope":
            continue
        if isinstance(value, dict):
            for role, seconds in value.items():
                records.append({"component": f"{key}/{role}", "value": finite(seconds, f"cost {key}/{role}", nonnegative=True),
                                "units": "seconds", "status": "recorded", "scope": "Separate evidence extraction cost, excluded from primary P3"})
            continue
        if value is not None:
            finite(value, f"cost {key}", nonnegative=True)
        records.append({"component": key, "value": value,
                        "units": "ratio" if "ratio" in key or key.endswith("over_full_refits") else "seconds",
                        "status": "undefined" if value is None else "recorded",
                        "scope": s.analysis["costs"]["scope"]})
    records.append({"component": "forecast_seconds", "value": finite(s.analysis["timings"]["forecast_seconds"], "forecast_seconds", nonnegative=True),
                    "units": "seconds", "status": "recorded", "scope": "Measured discovery forecast computation; no model refit"})
    return records


def render_cost_scopes(g, s):
    records = cost_rows(s)
    g.csv("r14_cost_scopes", records, "All source-computed deployment cost scopes and measured forecast time. Undefined ratios remain blank/explicitly undefined, never zero.")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    costs = s.analysis["costs"]
    labels = ["Extra cheap refits\nfit + predict + decomposition", "Full-response refits\nfit + predict + decomposition",
              "Shared full baseline\nfit/predict/operator/features", "Forecast calculation"]
    values = [costs["cheap_fit_prediction_decomposition_seconds"], costs["full_fit_prediction_decomposition_seconds"],
              costs["common_base_fit_predictions_operator_features_seconds"], s.analysis["timings"]["forecast_seconds"]]
    axes[0].bar(range(len(labels)), values, color=["#c56739", "#167d9b", "#84939e", "#15887b"])
    axes[0].set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Measured seconds; shared scope shown separately")
    for i, value in enumerate(values):
        axes[0].annotate(f"{value:.3g}", (i, value), ha="center", xytext=(0, 4), textcoords="offset points", fontsize=8)
    axes[0].margins(y=.15); axes[0].grid(axis="y", alpha=.15)
    ratio_keys = [("extra_cheap_full_ratio", "Extra cheap / full refits\n(primary conditional P3)"),
                  ("secondary_end_to_end_ratio_common_base_in_both", "With shared base in both\n(secondary end-to-end)"),
                  ("secondary_end_to_end_forecast_cost_over_full_refits", "Total forecast cost / full refits\n(secondary end-to-end)")]
    for i, (key, _) in enumerate(ratio_keys):
        value = costs[key]
        if value is None:
            axes[1].text(i, .04, "undefined", transform=axes[1].get_xaxis_transform(), ha="center", rotation=90, fontsize=8)
        else:
            axes[1].bar(i, value, color="#c56739" if i == 0 else "#84939e")
            axes[1].annotate(f"{value:.3g}", (i, value), ha="center", xytext=(0, 4), textcoords="offset points", fontsize=8)
    axes[1].axhline(.5, color="black", linestyle="--", linewidth=.8)
    axes[1].set_xticks(range(len(ratio_keys)), [label for _, label in ratio_keys], rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Source-computed cost ratio"); axes[1].margins(y=.15); axes[1].grid(axis="y", alpha=.15)
    g.save(fig, "r14_cost_scopes", "R14 — conditional and end-to-end computational scopes",
           "The .5 threshold applies only to the first ratio (primary P3). Other ratios are secondary, not alternative success gates. All measured evidence-extraction/I/O costs remain separate in source tables; timings depend on this environment. " + UNIT_NOTE,
           "R14 computational cost")


def render_costs(g,s):
    records=timing_rows(s);g.csv("r14_all_timing_components",records,"Every scalar source timing component, with context/direction identity; no tree-count-to-speed conversion.")
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for ax,scope in zip(axes,("context","direction")):
        group=[r for r in records if r["scope"]==scope];components=sorted({r["component"] for r in group})
        if components:
            ax.barh(range(len(components)),[sum(r["seconds"] for r in group if r["component"]==c) for c in components],color="#167d9b");ax.set_yticks(range(len(components)),components,fontsize=8)
        else:ax.text(.1,.5,"Timing components unavailable in source",transform=ax.transAxes)
        ax.set_title(f"{scope.capitalize()} component totals");ax.set_xlabel("Measured seconds; components may be nested, do not add bars");ax.grid(axis="x",alpha=.15)
    g.save(fig,"r14_runtime_components","R14 — measured cost components","Source component totals are displayed separately; nested timers must not be double counted. Shared base/operator and extra probes have different deployment scopes. "+UNIT_NOTE,"R14 computational cost")
    direction_components=sorted({r["component"] for r in records if r["scope"]=="direction"})
    fig,ax=plt.subplots(figsize=(12,max(5,len(direction_components)*.9)),constrained_layout=True)
    components=sorted({r["component"] for r in records if r["scope"]=="direction"});position=0;labels=[]
    for component in components:
        for phase,color in (("pilot","#c56739"),("response","#167d9b")):
            values=[r["seconds"] for r in records if r["scope"]=="direction" and r["phase"]==phase and r["component"]==component]
            if values:ax.scatter(values,position+np.linspace(-.2,.2,len(values)),color=color,s=9,alpha=.5);labels.append(f"{component}\n{phase}");position+=1
    ax.set_yticks(range(len(labels)),labels,fontsize=8);ax.set_xlabel("Measured seconds per direction");ax.grid(axis="x",alpha=.2)
    g.save(fig,"r14_runtime_direction_distributions","R14 — every direction's measured costs","Points are individual recorded costs, not independent performance trials or confidence intervals. File I/O and analysis costs remain separately recorded.","R14 computational cost")
    fig,ax=plt.subplots(figsize=(14,6),constrained_layout=True);contexts=sorted(s.contexts,key=lambda r:r["id"]);components=sorted({r["component"] for r in records if r["scope"]=="context"})
    for component in components:
        values=[number(r.get("timings",{}).get(component)) for r in contexts];ax.plot(range(len(contexts)),values,"o-",label=component,linewidth=.8,markersize=3)
    ax.set_xticks(range(len(contexts)),[r["id"] for r in contexts],rotation=90,fontsize=7);ax.set_ylabel("Measured seconds; timers shown separately");ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15)
    g.save(fig,"r14_runtime_by_context","R14 — costs for every context","All 32 contexts are retained. This is a conditional audit of an available basefit/operator; no general standalone-service speedup is implied.","R14 computational cost")
    decision=s.analysis.get("decision",{});fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True);ax.axis("off")
    text="Source decision (renderer does not recompute gates):\n\n"+"\n".join(f"{key}: {decision.get(key,'not recorded')}" for key in ("P1_replay","P2_forecast","P3_cost","range_gate","cheap_full_wall_ratio","go","outcome"))
    ax.text(.03,.95,text,va="top",transform=ax.transAxes,fontsize=12)
    g.save(fig,"r14_declared_decision","R14 — declared scientific stop/go outcome","A failed or inconclusive gate is retained. Cost feasibility does not establish forecast accuracy, novelty, publication readiness or double descent. "+UNIT_NOTE,"R14 computational cost")
    render_cost_scopes(g, s)


def render(g: Gallery, s: Study) -> None:
    g.expected_delivery = delivery_spec(s)
    g.notes.extend((UNIT_NOTE,f"Complete source matrix: {len(s.contexts)} contexts; {len(s.rows)} context/amplitude rows; {len(s.directions)} directions; {len(s.seeds)} independent draw clusters.",
                    "Unlabeled evaluation covariates are available to the conditional audit. Approximation discrepancy is not teacher prediction error or an additive extra-variance term.",
                    "Static graphs replot fixed evidence only; this renderer fits no models, changes no forecasts and accesses no network."))
    source_tables(g,s);render_direction_panels(g,s);render_measurements(g,s);render_forecasts(g,s);render_costs(g,s)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--study",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    source,output=args.study.resolve(),args.output.resolve()
    if output==source or source in output.parents or output in source.parents:parser.error("Derived plots must be outside scientific task directories and use a separate output directory")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    g=ResponseGallery(output);g.source(Path(__file__));g.source(ROOT/"scripts"/"plot_v2_research.py");s=load_study(g,source);render(g,s);g.finish();return 0


if __name__=="__main__":raise SystemExit(main())
