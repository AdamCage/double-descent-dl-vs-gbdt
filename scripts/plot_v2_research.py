"""Render audited R10/R11/R12 evidence; never fit or alter scientific inputs.

Every static figure is written as PNG and SVG.  An offline HTML gallery,
interactive overview plots, CSV source tables, and hashes accompany the figures.
Trajectory envelopes are observed seed ranges, never confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


DETECTORS = ("global", "local", "scan")
FAMILIES = ("linear", "exponential", "early_plateau", "late_hinge", "fast_plateau_tail", "last_tenth")
NOISE = ("small_iid", "small_ar", "large_iid", "large_ar")
SCHEDULES = ("constant", "stepup", "stepdown")
SPLITS = ("train", "validation", "test")
COLORS = {"global": "#64809a", "local": "#d18637", "scan": "#15887b",
          "constant": "#64748b", "stepup": "#167d9b", "stepdown": "#c56739"}
LABELS = {"global": "Global rule", "local": "Local window 5", "scan": "OLS scan (calibrated only in R10)",
          "constant": "Constant .11 → .11", "stepup": "Step-up .02 → .20", "stepdown": "Step-down .20 → .02",
          "small_iid": "σ=.002, ρ=0", "small_ar": "σ=.002, ρ=.8",
          "large_iid": "σ=.01, ρ=0", "large_ar": "σ=.01, ρ=.8",
          "breast_cancer": "Breast Cancer (fixed corpus)", "synthetic": "Synthetic (new data per seed)"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pretty(value: str) -> str:
    return LABELS.get(value, value.replace("_", " "))


def format_numeric_ticks(fig) -> None:
    """Keep small numeric tick labels readable on every saved static panel.

    Categorical and logarithmic formatters retain their existing behavior.
    The shared exponent records the original units without long decimals.
    """
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()
            if isinstance(formatter, ScalarFormatter):
                formatter.set_scientific(True)
                formatter.set_powerlimits((-3, 4))
                formatter.set_useMathText(True)


class Gallery:
    def __init__(self, output: Path):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "figures").mkdir(exist_ok=True)
        (self.output / "tables").mkdir(exist_ok=True)
        self.sources: dict[str, dict[str, Any]] = {}
        self.figures: list[dict[str, Any]] = []
        self.tables: list[dict[str, Any]] = []
        self.notes: list[str] = []

    def source(self, path: Path) -> Path:
        resolved = path.resolve()
        self.sources[str(resolved)] = {"sha256": digest(resolved), "bytes": resolved.stat().st_size}
        return resolved

    def load(self, path: Path) -> Any:
        return json.loads(self.source(path).read_text(encoding="utf-8"))

    def jsonl(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.source(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    def csv(self, name: str, rows: list[dict[str, Any]], description: str) -> None:
        if not rows:
            return
        path = self.output / "tables" / f"{name}.csv"
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
        self.tables.append({"path": path.relative_to(self.output).as_posix(), "rows": len(rows),
                            "description": description, "sha256": digest(path)})

    def save(self, fig, name: str, title: str, caption: str, group: str, selection: str = "All declared observations") -> None:
        format_numeric_ticks(fig)
        # Let constrained_layout reserve space above subplot headers.
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.text(.01, -.018, caption, fontsize=8, ha="left", va="top", wrap=True)
        paths = {}
        for extension in ("png", "svg"):
            path = self.output / "figures" / f"{name}.{extension}"
            fig.savefig(path, dpi=155, bbox_inches="tight", facecolor="white")
            paths[extension] = path.relative_to(self.output).as_posix()
        plt.close(fig)
        self.figures.append({"id": name, "title": title, "caption": caption, "group": group,
                             "selection": selection, "paths": paths,
                             "sha256": {ext: digest(self.output / path) for ext, path in paths.items()}})

    def interactive(self, figure, name: str, title: str) -> None:
        path = self.output / "figures" / f"{name}.html"
        figure.write_html(str(path), include_plotlyjs=True, full_html=True)
        self.figures.append({"id": name, "title": title, "caption": "Interactive offline companion; same source estimates as static figures.",
                             "group": "Interactive", "selection": "All declared summary cells", "paths": {"html": path.relative_to(self.output).as_posix()},
                             "sha256": {"html": digest(path)}})

    def finish(self) -> None:
        # Rehash source files after rendering so concurrent source changes fail.
        for source, record in self.sources.items():
            if digest(Path(source)) != record["sha256"]:
                raise RuntimeError(f"Scientific source changed during rendering: {source}")
        manifest = {"schema_version": 1, "renderer": str(Path(__file__).resolve()),
                    "renderer_sha256": digest(Path(__file__)), "sources": self.sources,
                    "figures": self.figures, "tables": self.tables, "notes": self.notes,
                    "interpretation": "Derived plots only. CI, observed ranges, and conditional localization are labeled separately. No model fitting or scientific source mutation.",
                    "figure_count": len(self.figures)}
        (self.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        groups = list(dict.fromkeys(row["group"] for row in self.figures))
        sections = []
        for group in groups:
            cards = []
            for row in self.figures:
                if row["group"] != group:
                    continue
                links = " · ".join(f'<a href="{html.escape(path)}">{extension.upper()}</a>' for extension, path in row["paths"].items())
                preview = f'<img loading="lazy" src="{html.escape(row["paths"]["png"])}" alt="{html.escape(row["title"])}">' if "png" in row["paths"] else ""
                cards.append(f'<article><h3>{html.escape(row["title"])}</h3>{preview}<p>{html.escape(row["caption"])}</p><small>{html.escape(row["selection"])}</small><p>{links}</p></article>')
            sections.append(f'<section id="{html.escape(group)}"><h2>{html.escape(group)}</h2><div class="grid">{"".join(cards)}</div></section>')
        table_links = "".join(f'<li><a href="{html.escape(row["path"])}">{html.escape(row["path"])}</a> ({row["rows"]} rows): {html.escape(row["description"])}</li>' for row in self.tables)
        nav = " · ".join(f'<a href="#{html.escape(group)}">{html.escape(group)}</a>' for group in groups)
        notes = "".join(f'<li>{html.escape(note)}</li>' for note in self.notes)
        page = f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>R10–R12 evidence gallery</title><style>body{{font:16px system-ui,sans-serif;color:#162b36;background:#f5f7f8;max-width:1600px;margin:auto;padding:28px}}h1,h2{{color:#143d49}}nav{{position:sticky;top:0;background:#fff;padding:12px;z-index:1}}a{{color:#086d87}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:20px}}article{{background:white;border:1px solid #dce4e8;padding:16px;border-radius:8px}}img{{width:100%;height:auto}}small{{color:#536773}}p{{line-height:1.45}}@media(max-width:600px){{.grid{{grid-template-columns:1fr}}body{{padding:12px}}}}</style><h1>Acceleration audit: complete evidence gallery</h1><p>Static PNG/SVG, offline interactive companions and source tables. Figures summarize declared data only; no training occurs in this renderer.</p><nav>{nav}</nav><ul>{notes}</ul>{"".join(sections)}<h2>Source tables</h2><ul>{table_links}</ul><p><a href="figure_manifest.json">Figure manifest with source and output SHA-256 hashes</a></p></html>'
        (self.output / "index.html").write_text(page, encoding="utf-8")
        print(json.dumps({"output": str(self.output), "figures": len(self.figures), "tables": len(self.tables)}, indent=2))


def render_r10(g: Gallery, directory: Path) -> None:
    analysis = g.load(directory / "analysis.json")
    calibration = g.load(directory / "calibration.json")
    curves = g.jsonl(directory / "curves.jsonl")
    with np.load(g.source(directory / "curves.npz"), allow_pickle=False) as z:
        losses = z["loss"]
    if len(curves) != losses.shape[0] or len(curves) != analysis["n_curves"]:
        raise ValueError("R10 raw curve count mismatch")
    rates = analysis["rates"]
    lookup = {(row["detector"], row["family"], row["noise_id"]): row for row in rates}
    g.csv("r10_rates", rates, "All declared rates; exact pointwise CP intervals condition on shared calibration, not simultaneous intervals.")
    g.csv("r10_scan_scores", [{key: row[key] for key in ("curve_index", "phase", "family", "noise_id", "replicate", "known_transition")} | row["scan"] for row in curves], "Every curve's whole-trajectory scan score, threshold and selected boundary.")
    g.notes.append("R10 intervals are exact pointwise 95% Clopper–Pearson intervals conditional on the shared calibration. They are not simultaneous and do not cover calibration uncertainty. The rank bound is marginal over calibration for included null laws only.")
    for role, families in (("null", FAMILIES[:3]), ("alternative", FAMILIES[3:])):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, detector in zip(axes, DETECTORS):
            values = np.array([[lookup[(detector, family, noise)]["rate"] for noise in NOISE] for family in families])
            image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(4), [pretty(noise) for noise in NOISE], rotation=30, ha="right")
            ax.set_yticks(range(3), [pretty(family) for family in families])
            ax.set_title(pretty(detector))
            for (i, j), value in np.ndenumerate(values):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < .45 else "black", fontsize=9)
        fig.colorbar(image, ax=axes, label="Observed flag proportion", shrink=.8)
        g.save(fig, f"r10_{role}_rate_heatmap", f"R10 — {'false flags on null curves' if role == 'null' else 'power on declared alternatives'}", "n=200 independent evaluation curves per cell. Heatmap shows point estimates; exact pointwise intervals appear in companion figures and CSV.", "R10 overview")
        for noise in NOISE:
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            positions = np.arange(len(families))
            for i, detector in enumerate(DETECTORS):
                rows = [lookup[(detector, family, noise)] for family in families]
                means = np.array([row["rate"] for row in rows])
                intervals = np.array([row["pointwise_95_cp"] for row in rows])
                ax.errorbar(means, positions + (i - 1) * .2, xerr=np.array([means - intervals[:, 0], intervals[:, 1] - means]), fmt="o", capsize=3, color=COLORS[detector], label=pretty(detector))
            ax.set_yticks(positions, [pretty(family) for family in families])
            ax.set_xlim(-.03, 1.03)
            ax.set_xlabel("Observed flag proportion with pointwise exact 95% interval")
            if role == "null":
                ax.axvline(.05, color="#a24e4e", linestyle="--", linewidth=1, label=".05 marginal calibration target")
                ax.axvline(.10, color="#a24e4e", linestyle=":", linewidth=1, label=".10 empirical prediction boundary")
            ax.grid(axis="x", alpha=.2)
            ax.legend(fontsize=8)
            g.save(fig, f"r10_{role}_intervals_{noise}", f"R10 — {role} flags, {pretty(noise)}", "Pointwise Clopper–Pearson intervals conditional on calibration; each detector sees the same 200 curves. No simultaneous coverage claim.", "R10 uncertainty")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, noise in zip(axes.flat, NOISE):
        for family in FAMILIES[:3]:
            scores = [row["scan"]["score"] for row in curves if row["phase"] == "calibration" and row["family"] == family and row["noise_id"] == noise]
            ax.hist(scores, bins=24, histtype="step", linewidth=1.5, label=pretty(family))
        threshold = calibration["thresholds"][noise]["threshold"]
        ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold {threshold:.4g}")
        ax.set_title(pretty(noise))
        ax.set_xlabel("Whole-trajectory scan score")
        ax.set_ylabel("Calibration curves (count)")
        ax.legend(fontsize=7)
    g.save(fig, "r10_calibration_distributions", "R10 — all null calibration scores", "199 curves per null/noise cell. Threshold is the maximum of the three within-family 190th order statistics. Histograms are observed distributions, not uncertainty intervals.", "R10 calibration")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, noise in zip(axes.flat, NOISE):
        values = [[row["scan"]["score"] for row in curves if row["phase"] == "evaluation" and row["noise_id"] == noise and row["family"] == family] for family in FAMILIES]
        ax.boxplot(values, tick_labels=[pretty(f) for f in FAMILIES], showfliers=True)
        ax.axhline(calibration["thresholds"][noise]["threshold"], color="#a24e4e", linestyle="--", label="Declared threshold")
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_title(pretty(noise))
        ax.set_ylabel("Scan score")
        ax.legend(fontsize=8)
    g.save(fig, "r10_evaluation_score_distributions", "R10 — score distributions on every evaluation family", "Box: observed Q1–Q3; line: median; whiskers: 1.5 IQR; points: observations beyond whiskers. These are distributions, not confidence intervals.", "R10 calibration")
    localization_rows = []
    for family in FAMILIES[3:]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, noise in zip(axes.flat, NOISE):
            for detector in DETECTORS:
                selected = [row for row in curves if row["phase"] == "evaluation" and row["noise_id"] == noise and row["family"] == family and row[detector]["candidate"]]
                errors = [row[detector]["t_star"] - row["known_transition"] for row in selected]
                for row, error in zip(selected, errors):
                    localization_rows.append({"curve_index": row["curve_index"], "family": family, "noise_id": noise, "detector": detector, "signed_error": error})
                if errors:
                    ax.hist(errors, bins=np.arange(-240, 241, 8), histtype="step", linewidth=1.5, color=COLORS[detector], label=f"{pretty(detector)}; flagged {len(errors)}/200")
                else:
                    ax.plot([], [], color=COLORS[detector], label=f"{pretty(detector)}; flagged 0/200")
            ax.axvline(0, color="black", linewidth=.8)
            ax.axvspan(-32, 32, color="#14887b", alpha=.08)
            ax.set_xlim(-240, 240)
            ax.set_title(pretty(noise))
            ax.set_xlabel("Detected boundary − true boundary (points)")
            ax.set_ylabel("Flagged curves (count)")
            ax.legend(fontsize=7)
        g.save(fig, f"r10_localization_{family}", f"R10 — boundary error: {pretty(family)}", "Conditional on flagging; legends show the denominator retained. Shaded ±32 points is the declared localization tolerance. Histograms are observed counts, not confidence intervals.", "R10 localization")
    g.csv("r10_localization_flagged", localization_rows, "Signed boundary errors among flagged alternatives only; missed curves remain counted in the rate table.")
    for family in FAMILIES[:3]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, noise in zip(axes.flat, NOISE):
            for detector in DETECTORS:
                boundaries = [row[detector]["t_star"] for row in curves if row["phase"] == "evaluation" and row["noise_id"] == noise and row["family"] == family and row[detector]["candidate"]]
                if boundaries:
                    ax.hist(boundaries, bins=np.arange(0, 257, 8), histtype="step", linewidth=1.5, color=COLORS[detector], label=f"{pretty(detector)}; {len(boundaries)}/200")
                else:
                    ax.plot([], [], color=COLORS[detector], label=f"{pretty(detector)}; 0/200")
            ax.set_title(pretty(noise))
            ax.set_xlabel("Reported boundary index on a null curve")
            ax.set_ylabel("Falsely flagged curves (count)")
            ax.legend(fontsize=7)
        g.save(fig, f"r10_null_flag_locations_{family}", f"R10 — where false flags occur: {pretty(family)}", "These null curves have no true acceleration boundary. Histograms show the observed locations of false flags only; legend denominators retain the full 200 evaluation curves.", "R10 localization")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, detector in zip(axes, DETECTORS):
        values = np.array([[lookup[(detector, family, noise)]["localized_within_max_window"] for noise in NOISE] for family in FAMILIES[3:]])
        image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(4), [pretty(n) for n in NOISE], rotation=30, ha="right")
        ax.set_yticks(range(3), [pretty(f) for f in FAMILIES[3:]])
        ax.set_title(pretty(detector))
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < .45 else "black")
    fig.colorbar(image, ax=axes, label="Fraction of all 200 curves", shrink=.8)
    g.save(fig, "r10_localized_detection", "R10 — detection within 32 points, counting misses", "Denominator is every alternative curve, not only flags. Observed proportions; this plot does not display confidence intervals.", "R10 localization")
    sample_rows = []
    for family in FAMILIES:
        for noise in NOISE:
            samples = sorted([row for row in curves if row["phase"] == "evaluation" and row["family"] == family and row["noise_id"] == noise], key=lambda row: row["replicate"])[:5]
            fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
            for ax, row in zip(axes, samples):
                y = losses[row["curve_index"]]
                ax.plot(np.arange(len(y)), y, color="#344f60", linewidth=.9)
                if row["known_transition"] is not None:
                    ax.axvline(row["known_transition"], color="black", linestyle="--", label="True transition")
                for detector in DETECTORS:
                    if row[detector]["candidate"]:
                        ax.axvline(row[detector]["t_star"], color=COLORS[detector], alpha=.8, linestyle=":", label=pretty(detector))
                ax.set_ylabel(f"rep {row['replicate']}\nloss")
                ax.grid(alpha=.15)
                ax.legend(fontsize=6, loc="upper right", ncol=4)
                sample_rows.extend({"family": family, "noise_id": noise, "replicate": row["replicate"], "curve_index": row["curve_index"], "time": t, "loss": value} for t, value in enumerate(y))
            axes[-1].set_xlabel("Observation index")
            g.save(fig, f"r10_examples_{family}_{noise}", f"R10 — first five: {pretty(family)}, {pretty(noise)}", "First five evaluation replicates in index order in every declared cell; identical selection policy for all families and noise laws. Lines mark detected boundaries only when flagged.", "R10 raw trajectory examples", "Evaluation replicates 0–4, selected by index without outcome filtering")
            if family in FAMILIES[3:]:
                fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
                for ax, row in zip(axes, samples):
                    y = losses[row["curve_index"]]
                    start = max(0, row["known_transition"] - 50)
                    ax.plot(np.arange(start, len(y)), y[start:], color="#344f60", linewidth=1)
                    ax.axvline(row["known_transition"], color="black", linestyle="--", label="True transition")
                    for detector in DETECTORS:
                        if row[detector]["candidate"] and row[detector]["t_star"] >= start:
                            ax.axvline(row[detector]["t_star"], color=COLORS[detector], alpha=.8, linestyle=":", label=pretty(detector))
                    ax.set_ylabel(f"rep {row['replicate']}\nloss")
                    ax.grid(alpha=.15)
                    ax.legend(fontsize=6, loc="upper right", ncol=4)
                axes[-1].set_xlabel("Observation index; zoom begins 50 points before declared transition")
                g.save(fig, f"r10_tail_zoom_{family}_{noise}", f"R10 — fixed tail zoom: {pretty(family)}, {pretty(noise)}", "Same first-five examples as full-trajectory panels. View limits depend only on the known simulated boundary, not on detector outputs; out-of-view flags remain visible in full panels.", "R10 tail resolution", "Evaluation replicates 0–4; window [true boundary − 50, end]")
    g.csv("r10_example_trajectories", sample_rows, "All plotted first-five evaluation curves, including the selection index and full time series.")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        plot = make_subplots(rows=1, cols=3, subplot_titles=[pretty(d) for d in DETECTORS])
        for i, detector in enumerate(DETECTORS, 1):
            values = [[lookup[(detector, family, noise)]["rate"] for noise in NOISE] for family in FAMILIES]
            plot.add_trace(go.Heatmap(z=values, x=[pretty(n) for n in NOISE], y=[pretty(f) for f in FAMILIES], zmin=0, zmax=1, coloraxis="coloraxis", hovertemplate="%{y}<br>%{x}<br>Observed proportion: %{z:.3f}<extra></extra>"), row=1, col=i)
        plot.update_layout(title="R10 — all 72 observed rates; see static figures/CSV for exact pointwise intervals", height=600, coloraxis={"colorscale": "Viridis", "cmin": 0, "cmax": 1})
        g.interactive(plot, "r10_interactive_rates", "R10 — interactive complete rate table")
    except ImportError:
        g.notes.append("Plotly unavailable; all static PNG/SVG figures are still provided.")


def render_r11(g: Gallery, directory: Path) -> None:
    analysis = g.load(directory / "analysis.json")
    cells = g.jsonl(directory / "cells.jsonl")
    if len(cells) != analysis["expected_cells"]:
        raise ValueError("R11 matrix incomplete; renderer does not summarize partial studies")
    g.csv("r11_primary_contrasts", analysis["primary_contrasts"], "Four fixed-boundary primary paired contrasts; ordinary and familywise Bonferroni Student-t intervals.")
    g.csv("r11_clock_diagnostic_contrasts", analysis["lr_clock_diagnostic_contrasts"], "Separate diagnostic family under cumulative prescribed LR; no predicted sign or mechanism inference.")
    g.notes.append("R11 mean-trajectory shading is the observed minimum–maximum across 12 seeds, not a confidence interval. Breast Cancer repeats are conditional on one fixed corpus. The intervention is scheduled, not a discovered natural acceleration.")
    metrics = ("logloss", "error_rate", "min_margin", "q10_margin", "gradient_rms", "top10_gradient_mass")
    metric_labels = {"logloss": "LogLoss", "error_rate": "Classification error fraction", "min_margin": "Minimum signed margin", "q10_margin": "10th-percentile signed margin", "gradient_rms": "Gradient RMS", "top10_gradient_mass": "Top-decile share of |gradient|"}
    long_rows = []
    for row in cells:
        for split in SPLITS:
            for t in range(row["n_trees"] + 1):
                long_rows.append({"dataset": row["dataset"], "seed": row["seed"], "schedule": row["schedule"], "split": split, "tree": t,
                                  "cumulative_learning_rate": row["cumulative_learning_rate"][t], "split_size": row["split_sizes"][split],
                                  **{metric: values[t] for metric, values in row["metrics"][split].items()}})
    g.csv("r11_all_trajectory_metrics", long_rows, "Every iteration of every split/model/seed; exact errors are counts, envelopes in plots are observed ranges.")
    datasets = analysis["datasets"]
    boundary = analysis["primary_boundary"]
    for dataset in datasets:
        selected = [row for row in cells if row["dataset"] == dataset]
        for metric in metrics:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
            for ax, split in zip(axes, SPLITS):
                for schedule in SCHEDULES:
                    group = [row for row in selected if row["schedule"] == schedule]
                    values = np.array([np.array(row["metrics"][split]["errors"]) / row["split_sizes"][split] if metric == "error_rate" else row["metrics"][split][metric] for row in group])
                    x = np.arange(values.shape[1])
                    ax.fill_between(x, values.min(axis=0), values.max(axis=0), alpha=.10, color=COLORS[schedule])
                    ax.plot(x, values.mean(axis=0), color=COLORS[schedule], label=pretty(schedule), linewidth=1.5)
                ax.axvline(boundary, color="black", linestyle="--", linewidth=.8)
                ax.set_title(split.capitalize())
                ax.set_xlabel("Trees")
                ax.set_ylabel(metric_labels[metric])
                ax.grid(alpha=.15)
                ax.legend(fontsize=7)
            g.save(fig, f"r11_{dataset}_{metric}", f"R11 — {pretty(dataset)}: {metric_labels[metric]}", "Solid lines: seed means. Shading: observed min–max across seeds, not a confidence interval. Dashed line: fixed continuation/intervention boundary. All schedules use continuation.", "R11 complete trajectories")
        fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
        for i, split in enumerate(SPLITS):
            for j, clock in enumerate(("tree", "lr")):
                ax = axes[i, j]
                for schedule in SCHEDULES:
                    group = [row for row in selected if row["schedule"] == schedule]
                    values = np.array([row["metrics"][split]["logloss"] for row in group])
                    x = np.arange(values.shape[1]) if clock == "tree" else np.asarray(group[0]["cumulative_learning_rate"])
                    ax.fill_between(x, values.min(axis=0), values.max(axis=0), alpha=.1, color=COLORS[schedule])
                    ax.plot(x, values.mean(axis=0), color=COLORS[schedule], label=pretty(schedule))
                ax.set_title(f"{split.capitalize()} — {'tree clock' if clock == 'tree' else 'prescribed LR clock'}")
                ax.set_xlabel("Trees" if clock == "tree" else "Cumulative prescribed learning rate")
                ax.set_ylabel("LogLoss")
                ax.grid(alpha=.15)
                ax.legend(fontsize=7)
        g.save(fig, f"r11_{dataset}_clock_comparison", f"R11 — {pretty(dataset)}: two views of the same trajectories", "Means and observed min–max across seeds. LR-clock rescaling does not hold learned partitions or optimization paths fixed; it is not an exact time-equivalence theorem.", "R11 clock controls")
        for seed in analysis["seeds"]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            for ax, split in zip(axes, SPLITS):
                for schedule in SCHEDULES:
                    row = next(row for row in selected if row["seed"] == seed and row["schedule"] == schedule)
                    ax.plot(row["metrics"][split]["logloss"], color=COLORS[schedule], label=pretty(schedule))
                ax.axvline(boundary, color="black", linestyle="--", linewidth=.8)
                ax.set_title(split.capitalize())
                ax.set_xlabel("Trees")
                ax.set_ylabel("LogLoss")
                ax.grid(alpha=.15)
                ax.legend(fontsize=7)
            g.save(fig, f"r11_seed_{dataset}_{seed}", f"R11 — {pretty(dataset)}, seed {seed}", "Every declared seed is shown, with all three schedules and splits; no outcome-based example selection. No uncertainty band is inferred from one seed.", "R11 every seed", f"Declared seed {seed}; all paired schedules")
        for split in SPLITS:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            for j, metric in enumerate(("logloss", "error_rate")):
                ax = axes[j]
                for seed in analysis["seeds"]:
                    endpoints = []
                    for schedule in SCHEDULES:
                        row = next(row for row in selected if row["seed"] == seed and row["schedule"] == schedule)
                        value = row["metrics"][split]["logloss"][-1] if metric == "logloss" else row["metrics"][split]["errors"][-1] / row["split_sizes"][split]
                        endpoints.append(value)
                    ax.plot(range(3), endpoints, color="#9aa8b2", alpha=.5, linewidth=.7)
                    for x, schedule, value in zip(range(3), SCHEDULES, endpoints):
                        ax.scatter(x, value, s=25, color=COLORS[schedule], alpha=.8)
                ax.set_xticks(range(3), [pretty(s) for s in SCHEDULES], rotation=15, ha="right")
                ax.set_ylabel(metric_labels[metric])
                ax.set_title(f"Final {metric_labels[metric]}")
                ax.grid(axis="y", alpha=.2)
            g.save(fig, f"r11_endpoints_{dataset}_{split}", f"R11 — paired final quality: {pretty(dataset)}, {split}", "Every seed is connected across schedules. Points are observed endpoints at 256 trees, not estimated confidence intervals. These are descriptive outcomes and do not select primary claims.", "R11 final quality")
    for key, clock in (("primary_contrasts", "tree"), ("lr_clock_diagnostic_contrasts", "lr")):
        contrasts = analysis[key]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        labels = [f"{pretty(row['dataset'])}\n{pretty(row['schedule_minus_constant'])}" for row in contrasts]
        for i, row in enumerate(contrasts):
            mean = row["mean"]
            lo, hi = row["ci_family_bonferroni"]
            l95, h95 = row["ci95_student_t"]
            color = COLORS[row["schedule_minus_constant"]]
            axes[0].errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], color=color, fmt="o", capsize=5, linewidth=1, zorder=2)
            axes[0].plot([l95, h95], [i, i], color=color, linewidth=5, solid_capstyle="butt", zorder=1)
            values = np.asarray(row["values"])
            axes[1].scatter(values, i + np.linspace(-.13, .13, len(values)), s=24, color=color, alpha=.8)
            axes[1].plot(mean, i, "D", color="black", markersize=6)
        for ax in axes:
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_yticks(range(len(contrasts)), labels)
            ax.set_xlabel("Paired descent-speed gain difference: schedule − constant")
            ax.grid(axis="x", alpha=.2)
        axes[0].set_title("Means with declared uncertainty intervals")
        axes[1].set_title("All 12 seed differences; diamond = mean")
        g.save(fig, f"r11_{clock}_paired_contrasts", f"R11 — {'primary tree-clock' if clock == 'tree' else 'diagnostic LR-clock'} contrasts", "Thick segment: ordinary 95% t interval. Thin capped segment: Bonferroni familywise 95% interval over four contrasts. Seed points are observed values, not confidence intervals. LR diagnostics have no predicted sign.", "R11 paired inference")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        plot = make_subplots(rows=2, cols=3, subplot_titles=[f"{pretty(d)} — {s}" for d in datasets for s in SPLITS])
        for i, dataset in enumerate(datasets, 1):
            for j, split in enumerate(SPLITS, 1):
                for schedule in SCHEDULES:
                    group = [row for row in cells if row["dataset"] == dataset and row["schedule"] == schedule]
                    values = np.asarray([row["metrics"][split]["logloss"] for row in group])
                    plot.add_trace(go.Scatter(x=np.arange(values.shape[1]), y=values.mean(axis=0), name=pretty(schedule), legendgroup=schedule, showlegend=i==1 and j==1, line={"color": COLORS[schedule]}), row=i, col=j)
        plot.update_layout(title="R11 — all schedule mean LogLoss trajectories (hover/zoom; observed ranges in static plots)", height=850)
        g.interactive(plot, "r11_interactive_logloss", "R11 — interactive loss trajectories")
    except ImportError:
        pass


def render_r12(g: Gallery, directory: Path) -> None:
    """Continuous-fit randomness intervention; all scans remain descriptive."""
    analysis = g.load(directory / "analysis.json")
    cells = g.jsonl(directory / "cells.jsonl")
    if len(cells) != analysis["expected_cells"]:
        raise ValueError("R12 matrix incomplete; renderer does not summarize partial studies")
    strengths = tuple(analysis["random_strengths"])
    colors = {strength: color for strength, color in zip(strengths, ("#64748b", "#167d9b", "#c56739"))}
    label = lambda strength: f"random_strength={strength:g}"
    boundary, points = 512, 64
    gates = analysis["natural_acceleration_gates"]
    if any(row["fixed_boundary"] != boundary or row["contrast_points"] != points for row in gates):
        raise ValueError("R12 fixed late window differs from declared 512/64")
    if any(row["n_trees"] != 1024 for row in cells):
        raise ValueError("R12 requires complete 1024-tree trajectories")
    g.csv("r12_scan_score_contrasts", analysis["scan_score_contrasts"], "Declared paired maximum-scan differences; four-contrast Bonferroni interval family. Scan scores are descriptive, not transported p-values.")
    g.csv("r12_fixed_boundary_gain_contrasts", analysis["fixed_boundary_gain_contrasts"], "Paired signed gain differences at boundary 512 with 64 points per OLS segment; four-contrast diagnostic interval family.")
    g.csv("r12_fixed_late_acceleration_gates", gates, "Six gates and 12 gain/drop intervals; a failed fixed-window gate does not rule out every event location or configuration.")
    g.notes.append("R12 uses one continuous 1024-tree fit and constant learning rate .05. Whole-run random_strength interventions do not isolate internal noise annealing. Maximum scans are descriptive: exchangeability with synthetic R10 nulls is unestablished, including different trajectory lengths.")
    g.notes.append("R12 fixed late-window gates use boundary 512 and 64 OLS points on each side. Gain/drop intervals form a 12-interval Bonferroni family. Observed trajectory ranges are not confidence intervals; no natural annealing mechanism is identified.")
    metrics = ("logloss", "error_rate", "min_margin", "q10_margin", "gradient_rms", "top10_gradient_mass")
    metric_labels = {"logloss": "LogLoss", "error_rate": "Classification error fraction", "min_margin": "Minimum signed margin", "q10_margin": "10th-percentile signed margin", "gradient_rms": "Gradient RMS", "top10_gradient_mass": "Top-decile share of |gradient|"}
    table = []
    interpolation_table = []
    for row in cells:
        zero = row["first_zero_train_errors_iteration"]
        interpolation_table.append({"dataset": row["dataset"], "seed": row["seed"], "random_strength": row["random_strength"],
                                    "first_zero_train_errors_iteration": zero,
                                    "train_logloss_at_first_zero": None if zero is None else row["metrics"]["train"]["logloss"][zero],
                                    "train_logloss_final": row["metrics"]["train"]["logloss"][-1]})
        for split in SPLITS:
            for t in range(row["n_trees"] + 1):
                table.append({"dataset": row["dataset"], "seed": row["seed"], "random_strength": row["random_strength"],
                              "split": split, "tree": t, "split_size": row["split_sizes"][split],
                              **{metric: values[t] for metric, values in row["metrics"][split].items()}})
    g.csv("r12_all_trajectory_metrics", table, "All iterations of all R12 split/model/seed trajectories; errors are exact counts. No training is performed by the renderer.")
    del table
    g.csv("r12_first_zero_errors", interpolation_table, "Exact first zero train-error time, including missing events; train LogLoss at that time is a separate quantity.")
    diagnostics_table = []
    for dataset in analysis["datasets"]:
        selected = [row for row in cells if row["dataset"] == dataset]
        for metric in metrics:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
            for ax, split in zip(axes, SPLITS):
                for strength in strengths:
                    group = [row for row in selected if row["random_strength"] == strength]
                    values = np.asarray([np.asarray(row["metrics"][split]["errors"]) / row["split_sizes"][split] if metric == "error_rate" else row["metrics"][split][metric] for row in group])
                    x = np.arange(values.shape[1])
                    ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=colors[strength], alpha=.1)
                    ax.plot(x, values.mean(axis=0), color=colors[strength], label=label(strength))
                ax.axvline(boundary, color="black", linestyle="--", linewidth=.8)
                ax.set_title(split.capitalize())
                ax.set_xlabel("Trees (constant LR .05)")
                ax.set_ylabel(metric_labels[metric])
                ax.grid(alpha=.15)
                ax.legend(fontsize=7)
            g.save(fig, f"r12_{dataset}_{metric}", f"R12 — {pretty(dataset)}: {metric_labels[metric]}", "Mean across all seeds with observed min–max shading, not a confidence interval. The dashed line marks a predeclared assessment window, not an intervention or fitted change point. One continuous fit per model.", "R12 complete trajectories")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for ax, split in zip(axes, SPLITS):
            for strength in strengths:
                group = [row for row in selected if row["random_strength"] == strength]
                values = np.asarray([row["metrics"][split]["logloss"] for row in group])
                x = np.arange(values.shape[1])
                ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=colors[strength], alpha=.1)
                ax.plot(x, values.mean(axis=0), color=colors[strength], label=label(strength))
            ax.axvspan(boundary - points + 1, boundary, color="#64809a", alpha=.07)
            ax.axvspan(boundary, boundary + points - 1, color="#15887b", alpha=.07)
            ax.axvline(boundary, color="black", linestyle="--", linewidth=.8)
            ax.set_xlim(boundary - points, boundary + points)
            ax.set_title(split.capitalize())
            ax.set_xlabel("Trees: fixed 512 boundary, 64 points per OLS segment")
            ax.set_ylabel("LogLoss")
            ax.grid(alpha=.15)
            ax.legend(fontsize=7)
        g.save(fig, f"r12_{dataset}_fixed_late_zoom", f"R12 — {pretty(dataset)}: predeclared late-window view", "Identical fixed view for every arm; no outcome-based zoom selection. Shading around means is observed seed min–max, not CI. Colored backgrounds mark the before/after windows.", "R12 fixed late window")
        for seed in analysis["seeds"]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            for ax, split in zip(axes, SPLITS):
                for strength in strengths:
                    row = next(row for row in selected if row["seed"] == seed and row["random_strength"] == strength)
                    ax.plot(row["metrics"][split]["logloss"], color=colors[strength], label=label(strength))
                ax.axvline(boundary, color="black", linestyle="--", linewidth=.8)
                ax.set_title(split.capitalize())
                ax.set_xlabel("Trees")
                ax.set_ylabel("LogLoss")
                ax.grid(alpha=.15)
                ax.legend(fontsize=7)
            g.save(fig, f"r12_seed_{dataset}_{seed}", f"R12 — {pretty(dataset)}, seed {seed}", "Every declared seed and all three whole-run randomness settings are retained. No scheduled LR change or fit continuation is used. No uncertainty interval is inferred from a single seed.", "R12 every seed", f"Declared seed {seed}; all paired randomness settings")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for strength in strengths:
            group = [row for row in selected if row["random_strength"] == strength]
            realized = np.asarray([row["realized_tree_leaf_counts"] for row in group])
            occupied = np.asarray([row["occupied_train_leaf_counts"] for row in group])
            if np.any(realized <= 0):
                raise ValueError("Realized tree leaf counts must be positive")
            for ax, values, ylabel in zip(axes, (realized, occupied, occupied / realized), ("Realized leaves per tree", "Occupied train leaves per tree", "Occupied / realized leaf fraction")):
                x = np.arange(1, values.shape[1] + 1)
                ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=colors[strength], alpha=.1)
                ax.plot(x, values.mean(axis=0), color=colors[strength], label=label(strength), linewidth=.9)
                ax.set_xlabel("Tree index")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.15)
                ax.legend(fontsize=7)
        g.save(fig, f"r12_{dataset}_leaf_occupancy", f"R12 — {pretty(dataset)}: realized and occupied leaves", "Per-tree means and observed seed min–max, not confidence intervals. Occupancy changes are descriptive and do not isolate split-noise annealing or identify an effective kernel.", "R12 structure and interpolation")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for i, strength in enumerate(strengths):
            group = [row for row in selected if row["random_strength"] == strength]
            for offset, row in zip(np.linspace(-.15, .15, len(group)), group):
                zero = row["first_zero_train_errors_iteration"]
                axes[0].scatter(i + offset, 1090 if zero is None else zero, color=colors[strength], marker="x" if zero is None else "o", s=26)
                if zero is not None:
                    axes[1].scatter(i + offset, row["metrics"]["train"]["logloss"][zero], color=colors[strength], s=26)
            absent = sum(row["first_zero_train_errors_iteration"] is None for row in group)
            axes[0].text(i, 1160, f"not reached: {absent}/{len(group)}", ha="center", fontsize=8)
        for ax in axes:
            ax.set_xticks(range(len(strengths)), [label(s) for s in strengths], rotation=10)
            ax.grid(axis="y", alpha=.2)
        axes[0].axhline(1024, color="black", linestyle=":", linewidth=.8)
        axes[0].set_ylim(-30, 1220)
        axes[0].set_yticks([0, 256, 512, 768, 1024, 1090], ["0", "256", "512", "768", "1024", "Not reached"])
        axes[0].set_ylabel("First exact zero train-error time; missing events separate")
        axes[1].set_ylabel("Train LogLoss at the first exact zero-error time")
        g.save(fig, f"r12_{dataset}_zero_error_timing", f"R12 — {pretty(dataset)}: zero errors versus residual loss", "All seeds are plotted; crosses above the run horizon mean zero errors were not reached, not an event at that time. Right panel includes reached events only. Exact zero errors do not imply zero LogLoss.", "R12 structure and interpolation")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        for ax, split in zip(axes, ("train", "test")):
            for strength in strengths:
                group = [row for row in selected if row["random_strength"] == strength]
                positions = [row["diagnostics"][split]["scan"]["t_star"] for row in group if row["diagnostics"][split]["scan"]["t_star"] is not None]
                ax.hist(positions, bins=np.arange(0, 1025, 32), histtype="step", color=colors[strength], linewidth=1.5, label=f"{label(strength)}; maximum present {len(positions)}/{len(group)}")
            ax.axvline(boundary, color="black", linestyle="--", linewidth=.8)
            ax.set_title(split.capitalize())
            ax.set_xlabel("Boundary maximizing the scan; not a confirmed event")
            ax.set_ylabel("Seeds (count)")
            ax.legend(fontsize=7)
        g.save(fig, f"r12_{dataset}_scan_boundary_locations", f"R12 — {pretty(dataset)}: maximizing scan boundaries", "Locations are descriptive maximizers, not identified change points or natural mechanisms. R10 synthetic-null error rates do not transfer to these 1025-point model trajectories.", "R12 descriptive scans")
        for split in ("train", "test"):
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
            for i, strength in enumerate(strengths):
                group = [row for row in selected if row["random_strength"] == strength]
                scores = [row["diagnostics"][split]["scan"]["score"] for row in group]
                axes[0].scatter(i + np.linspace(-.15, .15, len(group)), scores, color=colors[strength], s=26)
                means = np.mean(scores)
                axes[0].plot([i-.2, i+.2], [means, means], color="black", linewidth=2)
                for detector_index, detector in enumerate(DETECTORS):
                    states = [row["diagnostics"][split][detector].get("descriptive_candidate" if detector == "scan" else "candidate") for row in group]
                    available = [state for state in states if state is not None]
                    proportion = sum(bool(state) for state in available) / len(available) if available else np.nan
                    xpos = i + (detector_index - 1) * .23
                    axes[1].bar(xpos, proportion, width=.21, color=COLORS[detector], label=pretty(detector) if i == 0 else None)
                    if not available:
                        axes[1].text(xpos, .03, "n/a", rotation=90, ha="center", fontsize=7)
                for row in group:
                    diagnostics_table.append({"dataset": dataset, "seed": row["seed"], "random_strength": strength, "split": split, "scan": row["diagnostics"][split]["scan"], "global": row["diagnostics"][split]["global"], "local": row["diagnostics"][split]["local"]})
            for ax in axes:
                ax.set_xticks(range(len(strengths)), [label(s) for s in strengths], rotation=10)
                ax.grid(axis="y", alpha=.2)
            axes[0].set_ylabel("Maximum scan score; black segment = observed mean")
            axes[1].set_ylabel("Descriptive flag proportion; no calibrated error rate")
            axes[1].set_ylim(0, 1.05)
            axes[1].legend(fontsize=8)
            g.save(fig, f"r12_{dataset}_{split}_descriptive_scan_scores", f"R12 — {pretty(dataset)}, {split}: scan scores and flags", "Points retain every seed. Bars are observed descriptive flag proportions, not confidence intervals; synthetic thresholds have unknown validity for CatBoost and a different trajectory length. No event is established by these flags alone.", "R12 descriptive scans")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for strength in strengths:
            group = [row for row in selected if row["random_strength"] == strength]
            before = [row["boundary_contrast"]["train"]["velocity_before"] for row in group]
            after = [row["boundary_contrast"]["train"]["velocity_after"] for row in group]
            axes[0].scatter(before, after, color=colors[strength], label=label(strength), s=30)
            gains = [row["boundary_contrast"]["train"]["velocity_gain"] for row in group]
            drops = [row["fixed_after_drop"]["train"] for row in group]
            axes[1].scatter(gains, drops, color=colors[strength], label=label(strength), s=30)
        lower = min(axes[0].get_xlim()[0], axes[0].get_ylim()[0])
        upper = max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
        axes[0].plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=.8)
        axes[0].set_xlabel("Before-window train descent speed")
        axes[0].set_ylabel("After-window train descent speed")
        axes[1].axvline(0, color="black", linestyle="--", linewidth=.8)
        axes[1].axhline(.01, color="black", linestyle=":", linewidth=.8)
        axes[1].set_xlabel("After − before train descent speed")
        axes[1].set_ylabel("Train LogLoss drop over fixed after-window")
        for ax in axes:
            ax.legend(fontsize=7)
            ax.grid(alpha=.15)
        g.save(fig, f"r12_{dataset}_fixed_window_seed_values", f"R12 — {pretty(dataset)}: every seed's fixed-window measurements", "All paired seeds at boundary 512 and 64-point OLS windows. Points are observations, not uncertainty intervals. Dashed diagonal: equal before/after speed; dotted drop line: declared .01 practical threshold.", "R12 fixed late window")
        for split in SPLITS:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            for ax, metric in zip(axes, ("logloss", "error_rate")):
                for seed in analysis["seeds"]:
                    endpoints = []
                    for strength in strengths:
                        row = next(row for row in selected if row["seed"] == seed and row["random_strength"] == strength)
                        endpoints.append(row["metrics"][split]["logloss"][-1] if metric == "logloss" else row["metrics"][split]["errors"][-1] / row["split_sizes"][split])
                    ax.plot(range(len(strengths)), endpoints, color="#9aa8b2", alpha=.5, linewidth=.7)
                    for i, strength, value in zip(range(len(strengths)), strengths, endpoints):
                        ax.scatter(i, value, color=colors[strength], s=26)
                ax.set_xticks(range(len(strengths)), [label(s) for s in strengths], rotation=10)
                ax.set_ylabel(f"Final {metric_labels[metric]}")
                ax.grid(axis="y", alpha=.2)
            g.save(fig, f"r12_endpoints_{dataset}_{split}", f"R12 — paired final quality: {pretty(dataset)}, {split}", "Every seed is connected across whole-run randomness settings at 1024 trees. These observed endpoints are descriptive, not confidence intervals or a double-descent capacity sweep.", "R12 final quality")
    g.csv("r12_descriptive_diagnostics", diagnostics_table, "Every train/test scan and legacy detector diagnostic. Synthetic-threshold flags are descriptive only.")
    for key, identifier, title in (("scan_score_contrasts", "scan", "Maximum-scan score"), ("fixed_boundary_gain_contrasts", "fixed_gain", "Fixed-boundary descent gain")):
        contrasts = analysis[key]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        labels = [f"{pretty(row['dataset'])}\nRS {row['random_strength_minus_zero']:g} − RS 0" for row in contrasts]
        for i, row in enumerate(contrasts):
            mean = row["mean"]
            lo, hi = row["ci_family_bonferroni"]
            l95, h95 = row["ci95_student_t"]
            color = colors[row["random_strength_minus_zero"]]
            axes[0].errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt="o", color=color, capsize=5, linewidth=1)
            axes[0].plot([l95, h95], [i, i], color=color, linewidth=5)
            values = row["values"]
            axes[1].scatter(values, i + np.linspace(-.13, .13, len(values)), color=color, s=25)
            axes[1].plot(mean, i, "D", color="black", markersize=6)
        for ax in axes:
            ax.set_yticks(range(len(contrasts)), labels)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel(f"Paired difference in {title.lower()}")
            ax.grid(axis="x", alpha=.2)
        axes[0].set_title("Means with declared uncertainty intervals")
        axes[1].set_title("All 12 seed differences; diamond = mean")
        g.save(fig, f"r12_{identifier}_paired_contrasts", f"R12 — {title.lower()} contrasts", "Thick segment: ordinary 95% Student-t interval. Thin capped segment: Bonferroni familywise 95% across four contrasts. Maximum scans remain descriptive; a whole-run random_strength effect does not isolate time-varying noise annealing.", "R12 paired inference")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    labels = [f"{pretty(row['dataset'])}\n{label(row['random_strength'])}; positive gains {row['positive_gain_seed_count']}/12" for row in gates]
    for ax, key, reference, xlabel in ((axes[0], "mean_gain", 0, "Fixed-window train descent-speed gain"), (axes[1], "mean_after_drop", .01, "Train LogLoss drop over fixed after-window")):
        for i, row in enumerate(gates):
            summary = row[key]
            mean = summary["mean"]
            lo, hi = summary["ci_family_bonferroni"]
            color = colors[row["random_strength"]]
            ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], color=color, fmt="o", capsize=4)
        ax.set_yticks(range(len(gates)), labels)
        ax.axvline(reference, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.2)
    g.save(fig, "r12_fixed_late_acceleration_gates", "R12 — prespecified late-event gates, with full interval family", "Intervals: Bonferroni familywise 95% over 12 gain/drop intervals. Gate also requires positive gains in at least 10/12 seeds. Boundary 512; 64 points per OLS window. Failure is specific to this fixed event definition; no annealing mechanism is identified.", "R12 paired inference")


def draw_control_curve(ax, rows: list[dict[str, Any]], x_key: str, metric: str, label: str, color: str, *, logarithmic: bool = False) -> None:
    """Descriptive means and observed seed ranges on exactly one declared axis."""
    xs = sorted({row[x_key] for row in rows})
    arrays = [np.asarray([float(row[metric]) for row in rows if row[x_key] == x and row[metric] is not None]) for x in xs]
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError(f"Nonfinite recorded control metric: {metric}")
    missing = sum(row[metric] is None for row in rows)
    available = [a for a in arrays if len(a)]
    if not available:
        # None means unassessed/unavailable in the original protocol, never zero.
        if not any(len(line.get_xdata()) for line in ax.lines):
            ax.text(.5, .5, "Not assessed / recorded in source", ha="center", va="center", transform=ax.transAxes)
        ax.plot([], [], color=color, label=label + " (not available)")
        ax.grid(alpha=.18)
        return
    means = [a.mean() if len(a) else np.nan for a in arrays]
    source_label = label if not missing else f"{label} ({missing}/{len(rows)} unavailable)"
    ax.plot(xs, means, "o-", markersize=4, linewidth=1.2, color=color, label=source_label)
    if metric != "interpolating":
        ax.fill_between(xs, [a.min() if len(a) else np.nan for a in arrays],
                        [a.max() if len(a) else np.nan for a in arrays], color=color, alpha=.12)
    if logarithmic and all(a.min() > 0 for a in available):
        ax.set_yscale("log")
    if metric == "interpolating":
        ax.set_ylim(-.03, 1.03)
    ax.grid(alpha=.18)


def render_controls(g: Gallery, stage: str, directory: Path) -> None:
    rows = g.jsonl(directory / "cells.jsonl")
    if not rows or any(row.get("phase") != "confirmation" for row in rows):
        raise ValueError(f"{stage} controls require explicit confirmation evidence")
    g.csv(f"{stage.lower()}_existing_confirmation_cells", rows, "Existing secondary model-family controls; descriptive replot, no retraining or new scientific inference.")
    palette = ("#167d9b", "#c56739", "#15887b", "#8768a9")
    if stage == "R2":
        groups = sorted({(row["family"], row["spectrum"]) for row in rows})
        metrics = (("clean_mse", "Clean evaluation MSE", True), ("residual_relative", "Relative train residual", True),
                   ("coefficient_l2", "Coefficient L2 norm", True), ("condition_number", "Design condition number", True),
                   ("interpolating", "Fraction of draws interpolating", False))
        for family, spectrum in groups:
            subset = [row for row in rows if row["family"] == family and row["spectrum"] == spectrum]
            snrs = sorted({row["snr"] for row in subset})
            for metric, ylabel, logarithmic in metrics:
                fig, axes = plt.subplots(1, len(snrs), figsize=(6 * len(snrs), 4.5), squeeze=False, constrained_layout=True)
                for ax, snr in zip(axes.flat, snrs):
                    condition = [row for row in subset if row["snr"] == snr]
                    for i, ridge in enumerate(sorted({row["ridge_lambda"] for row in condition})):
                        draw_control_curve(ax, [row for row in condition if row["ridge_lambda"] == ridge], "column", metric, f"Ridge λ={ridge:g}", palette[i % len(palette)], logarithmic=logarithmic)
                    ax.set_xscale("log", base=2)
                    ax.set_xlabel("Number of available features" if family == "linear" else "Number of frozen random features")
                    ax.set_ylabel(ylabel)
                    ax.set_title(f"SNR {snr:g}")
                    ax.legend(fontsize=8)
                g.save(fig, f"controls_r2_{family}_{spectrum}_{metric}", f"Existing R2 control — {pretty(family)}, {spectrum}: {ylabel}", "Each line fixes family, spectrum, SNR and ridge. Mean across the original confirmation draws; shading is observed min–max, not CI. Interpolation curves are observed proportions without uncertainty bands. Risk/norm/residual/condition use log y when positive.", "Existing model-family controls")
    elif stage == "R3":
        subset = [row for row in rows if row["weight_decay"] == 0]
        metrics = (("clean_mse", "Clean evaluation MSE", True), ("residual_relative", "Relative train residual", True),
                   ("epochs_ran", "Epochs executed", False), ("parameter_l2", "Parameter L2 norm", True),
                   ("interpolating", "Fraction of draws interpolating", False))
        snrs = sorted({row["snr"] for row in subset})
        for metric, ylabel, logarithmic in metrics:
            fig, axes = plt.subplots(1, len(snrs), figsize=(6 * len(snrs), 4.5), squeeze=False, constrained_layout=True)
            for ax, snr in zip(axes.flat, snrs):
                condition = [row for row in subset if row["snr"] == snr]
                for i, procedure in enumerate(sorted({row["procedure"] for row in condition})):
                    draw_control_curve(ax, [row for row in condition if row["procedure"] == procedure], "width", metric, pretty(procedure), palette[i], logarithmic=logarithmic)
                ax.set_xscale("log", base=2)
                ax.set_xlabel("MLP hidden-layer width; weight decay fixed at 0")
                ax.set_ylabel(ylabel)
                ax.set_title(f"SNR {snr:g}")
                ax.legend(fontsize=8)
            g.save(fig, f"controls_r3_{metric}", f"Existing R3 control — MLP: {ylabel}", "Each line fixes SNR, procedure and weight decay. Mean and observed min–max across original confirmation draws, not confidence intervals. Interpolation proportions have no uncertainty band. The 40-epoch prediction was refuted in the original report; curves do not create a new claim.", "Existing model-family controls")
        # The only nonzero-decay slice is held at its declared width/SNR/procedure.
        positive = [row for row in rows if row["weight_decay"] > 0]
        contrasts = sorted({(row["width"], row["snr"], row["procedure"]) for row in positive})
        if contrasts:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            for ax, metric in zip(axes, ("clean_mse", "residual_relative")):
                for i, (width, snr, procedure) in enumerate(contrasts):
                    fixed = [row for row in rows if row["width"] == width and row["snr"] == snr and row["procedure"] == procedure]
                    draw_control_curve(ax, fixed, "weight_decay", metric, f"width {width}, SNR {snr:g}, {procedure}", palette[i % len(palette)])
                ax.set_xlabel("Weight decay; all other declared settings fixed")
                ax.set_ylabel(pretty(metric))
                ax.legend(fontsize=7)
            g.save(fig, "controls_r3_weight_decay_slice", "Existing R3 control — declared weight-decay slice", "The weight-decay axis is separate from width and training duration. Lines show means; shading is the observed draw range, not a confidence interval.", "Existing model-family controls")
    elif stage == "R4":
        for grid in sorted({row["grid"] for row in rows}):
            subset = [row for row in rows if row["grid"] == grid]
            varied = {row["varied"] for row in subset}
            if len(varied) != 1:
                raise ValueError(f"Mixed intervention axes in {grid}")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            for ax, metric in zip(axes.flat, ("clean_mse", "residual_relative", "n_leaves", "interpolating")):
                draw_control_curve(ax, subset, "value", metric, pretty(grid), palette[0])
                ax.set_xlabel(pretty(next(iter(varied))))
                ax.set_ylabel("Realized total leaves" if metric == "n_leaves" else "Fraction of draws interpolating" if metric == "interpolating" else pretty(metric))
                ax.set_title(pretty(metric))
            g.save(fig, f"controls_r4_{grid}", f"Existing R4 control — {pretty(grid)}", "Only this declared grid is connected. Other tree/ensemble/boosting axes remain separate. Mean and observed min–max across original confirmation draws, not confidence intervals; interpolation is an observed proportion.", "Existing model-family controls")
    g.notes.append(f"{stage} panels replot explicit existing confirmation evidence as secondary controls. Distinct capacity, ensembling, regularization and optimization axes are not merged.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r10", type=Path)
    parser.add_argument("--r11", type=Path)
    parser.add_argument("--r12", type=Path, help="Complete continuous-fit randomness-intervention task directory")
    parser.add_argument("--r2", type=Path, help="Explicit existing R2 confirmation task directory (secondary controls)")
    parser.add_argument("--r3", type=Path, help="Explicit existing R3 confirmation task directory (secondary controls)")
    parser.add_argument("--r4", type=Path, help="Explicit existing R4 confirmation task directory (secondary controls)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not any((args.r10, args.r11, args.r12, args.r2, args.r3, args.r4)):
        parser.error("At least one completed task directory is required")
    output = args.output.resolve()
    for source in (args.r10, args.r11, args.r12, args.r2, args.r3, args.r4):
        if source is not None and (output == source.resolve() or source.resolve() in output.parents):
            parser.error("Derived gallery must be outside scientific task directories")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10, "svg.fonttype": "none"})
    gallery = Gallery(output)
    if args.r10:
        render_r10(gallery, args.r10.resolve())
    if args.r11:
        render_r11(gallery, args.r11.resolve())
    else:
        gallery.notes.append("R11 plots are pending a complete verified study; absence is not an observed negative result.")
    if args.r12:
        render_r12(gallery, args.r12.resolve())
    for stage, source in (("R2", args.r2), ("R3", args.r3), ("R4", args.r4)):
        if source:
            render_controls(gallery, stage, source.resolve())
    gallery.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
