"""Stage 09 — CatBoost vs DNN comparison."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.cache import compute_hash, is_cached, write_hash
from src.config import Settings
from src.plotting import save_fig


def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
    out_dir = cfg.result_path / "09_comparison"

    h = compute_hash(mode=cfg.MODE, seed=cfg.SEED, stage="s09")
    if is_cached(out_dir, h):
        logger.info("Stage 09 cached — skipping")
        return

    # Load metrics
    dnn_metrics = (
        ctx.get("dnn_metrics")
        if ctx
        else json.loads((cfg.result_path / "06_dnn_metrics" / "threshold_analysis.json").read_text(encoding="utf-8"))
    )
    cb_metrics = (
        ctx.get("cb_metrics")
        if ctx
        else json.loads((cfg.result_path / "08_catboost_metrics" / "threshold_analysis.json").read_text(encoding="utf-8"))
    )

    test_df = ctx["test_df"] if ctx and "test_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "test.parquet")
    y_test = test_df["generated"].values.astype(np.float32)

    # Load probabilities
    if ctx and "dnn_proba_test" in ctx:
        dnn_proba = ctx["dnn_proba_test"]
    else:
        from src.stages.s05_dnn_train import MLP
        from src.stages.s06_dnn_metrics import predict_proba_dnn
        import torch

        X_test = np.load(cfg.result_path / "04_embedding_analysis" / "test_reduced.npy")
        widths = cfg.dnn_widths_list
        width = widths[-1] if widths else 1024
        model = MLP(X_test.shape[1], width)
        model.load_state_dict(torch.load(cfg.result_path / "05_dnn" / "best_model.pt", weights_only=True))
        dnn_proba = predict_proba_dnn(model, X_test)

    if ctx and "cb_proba_test" in ctx:
        cb_proba = ctx["cb_proba_test"]
    else:
        from catboost import CatBoostClassifier

        X_test = np.load(cfg.result_path / "04_embedding_analysis" / "test_reduced.npy")
        cb_model = CatBoostClassifier()
        cb_model.load_model(str(cfg.result_path / "07_catboost" / "best_model.cbm"))
        cb_proba = cb_model.predict_proba(X_test)[:, 1]

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Side-by-side metrics table ────────────────────────────────────────────
    rows = []
    for thr_name in ["f1_best", "roc_youden", "prevalence", "0.5"]:
        d = dnn_metrics.get(thr_name, {})
        c = cb_metrics.get(thr_name, {})
        rows.append({
            "threshold_type": thr_name,
            "dnn_threshold": d.get("threshold"),
            "dnn_accuracy": d.get("accuracy"),
            "dnn_f1": d.get("f1"),
            "cb_threshold": c.get("threshold"),
            "cb_accuracy": c.get("accuracy"),
            "cb_f1": c.get("f1"),
        })
    rows.append({
        "threshold_type": "overall",
        "dnn_threshold": None,
        "dnn_accuracy": None,
        "dnn_f1": None,
        "cb_threshold": None,
        "cb_accuracy": None,
        "cb_f1": None,
    })
    summary = {
        "dnn_auc": dnn_metrics.get("auc"),
        "dnn_logloss": dnn_metrics.get("logloss"),
        "cb_auc": cb_metrics.get("auc"),
        "cb_logloss": cb_metrics.get("logloss"),
    }

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(out_dir / "metrics_comparison.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plotly table
    fig_table = go.Figure(data=[go.Table(
        header=dict(values=list(comparison_df.columns)),
        cells=dict(values=[comparison_df[c].tolist() for c in comparison_df.columns]),
    )])
    fig_table.update_layout(title="Metrics comparison: DNN vs CatBoost")
    save_fig(fig_table, out_dir / "metrics_table")

    # ── Scatter: DNN proba vs CatBoost proba ──────────────────────────────────
    scatter_df = pd.DataFrame({
        "DNN proba": dnn_proba,
        "CatBoost proba": cb_proba,
        "y_true": np.where(y_test == 1, "AI", "Human"),
    })
    fig = px.scatter(
        scatter_df, x="DNN proba", y="CatBoost proba", color="y_true",
        title="DNN vs CatBoost predicted probabilities", opacity=0.4,
    )
    fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
    save_fig(fig, out_dir / "scatter_proba")

    # ── Probability distribution overlay ──────────────────────────────────────
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=["DNN", "CatBoost"])
    for i, (proba, name) in enumerate([(dnn_proba, "DNN"), (cb_proba, "CatBoost")], 1):
        for cls_val, cls_name, color in [(0, "Human", "#636EFA"), (1, "AI", "#EF553B")]:
            mask = y_test == cls_val
            fig2.add_trace(
                go.Histogram(x=proba[mask], name=f"{cls_name} ({name})", opacity=0.6,
                             marker_color=color, nbinsx=40, showlegend=(i == 1)),
                row=1, col=i,
            )
    fig2.update_layout(title="Probability distributions: DNN vs CatBoost", barmode="overlay")
    save_fig(fig2, out_dir / "prob_distributions_overlay")

    # ── Disagreement analysis ─────────────────────────────────────────────────
    dnn_pred = (dnn_proba >= 0.5).astype(int)
    cb_pred = (cb_proba >= 0.5).astype(int)
    agree = (dnn_pred == cb_pred)
    disagree_mask = ~agree

    disagree_info = {
        "total_test": int(len(y_test)),
        "agree": int(agree.sum()),
        "disagree": int(disagree_mask.sum()),
        "disagree_pct": float(disagree_mask.mean() * 100),
    }
    if disagree_mask.any():
        disagree_info["dnn_correct_when_disagree"] = int(
            (dnn_pred[disagree_mask] == y_test[disagree_mask].astype(int)).sum()
        )
        disagree_info["cb_correct_when_disagree"] = int(
            (cb_pred[disagree_mask] == y_test[disagree_mask].astype(int)).sum()
        )

    (out_dir / "disagreement.json").write_text(json.dumps(disagree_info, indent=2), encoding="utf-8")
    logger.info("Disagreement: %d / %d (%.1f%%)", disagree_info["disagree"], disagree_info["total_test"], disagree_info["disagree_pct"])

    write_hash(out_dir, h)
    logger.info("Stage 09 artifacts saved to %s", out_dir)
