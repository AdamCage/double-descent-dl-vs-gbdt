"""Stage 06 — DNN metrics: thresholds, calibration, probability distributions, UMAP."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.cache import compute_hash, is_cached, write_hash
from src.config import Settings
from src.plotting import save_fig
from src.stages.s05_dnn_train import MLP


# ── Shared metrics helpers (reused in s08) ────────────────────────────────────

def predict_proba_dnn(model: MLP, X: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def threshold_analysis(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    """Compute metrics at several threshold strategies."""
    results: dict[str, Any] = {}

    # F1-optimal
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, proba)
    f1_arr = 2 * prec_arr * rec_arr / np.maximum(prec_arr + rec_arr, 1e-8)
    best_idx = np.argmax(f1_arr)
    best_thr_f1 = float(thr_arr[min(best_idx, len(thr_arr) - 1)])

    # ROC-optimal (Youden's J)
    fpr, tpr, roc_thr = roc_curve(y_true, proba)
    j = tpr - fpr
    best_thr_roc = float(roc_thr[np.argmax(j)])

    # Prevalence
    prevalence_thr = float(y_true.mean())

    thresholds = {
        "f1_best": best_thr_f1,
        "roc_youden": best_thr_roc,
        "prevalence": prevalence_thr,
        "0.5": 0.5,
    }

    for name, thr in thresholds.items():
        preds = (proba >= thr).astype(int)
        results[name] = {
            "threshold": thr,
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
        }

    results["auc"] = float(roc_auc_score(y_true, proba))
    results["logloss"] = float(log_loss(y_true, proba))
    return results


def plot_probability_distributions(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    out: Path,
    title_prefix: str = "",
) -> None:
    """Histogram of predicted probabilities split by y_true, for each data split."""
    for split_name, (y, proba) in splits.items():
        df = pd.DataFrame({"proba": proba, "class": np.where(y == 1, "AI (1)", "Human (0)")})
        fig = px.histogram(
            df, x="proba", color="class", barmode="overlay", nbins=60, opacity=0.6,
            title=f"{title_prefix}Probability distribution — {split_name}",
            labels={"proba": "Predicted probability"},
        )
        save_fig(fig, out / f"prob_dist_{split_name}")


def plot_calibration(y_true: np.ndarray, proba: np.ndarray, out: Path, name: str = "") -> None:
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=15, strategy="uniform")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")))
    fig.update_layout(
        title=f"Calibration curve{' — ' + name if name else ''}",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
    )
    save_fig(fig, out / "calibration")


def plot_umap_proba(
    X: np.ndarray, y: np.ndarray, proba: np.ndarray,
    out: Path, title_prefix: str = "",
) -> None:
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X2d = reducer.fit_transform(X)

    df = pd.DataFrame({
        "UMAP-1": X2d[:, 0], "UMAP-2": X2d[:, 1],
        "proba": proba, "pred": np.where(proba >= 0.5, "AI", "Human"),
        "y_true": np.where(y == 1, "AI", "Human"),
    })

    for color_col, title_suffix in [
        ("proba", "by predicted probability"),
        ("pred", "by predicted class"),
        ("y_true", "by true class"),
    ]:
        if color_col == "proba":
            fig = px.scatter(
                df, x="UMAP-1", y="UMAP-2", color="proba",
                title=f"{title_prefix}UMAP — {title_suffix}",
                color_continuous_scale="RdYlGn_r", opacity=0.5,
            )
        else:
            fig = px.scatter(
                df, x="UMAP-1", y="UMAP-2", color=color_col,
                title=f"{title_prefix}UMAP — {title_suffix}", opacity=0.5,
            )
        fig.update_traces(marker_size=3)
        save_fig(fig, out / f"umap_{color_col}")


# ── Public entry point ────────────────────────────────────────────────────────

def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
    out_dir = cfg.result_path / "06_dnn_metrics"

    X_train = ctx["X_train"] if ctx and "X_train" in ctx else np.load(cfg.result_path / "04_embedding_analysis" / "train_reduced.npy")
    X_val = ctx["X_val"] if ctx and "X_val" in ctx else np.load(cfg.result_path / "04_embedding_analysis" / "val_reduced.npy")
    X_test = ctx["X_test"] if ctx and "X_test" in ctx else np.load(cfg.result_path / "04_embedding_analysis" / "test_reduced.npy")

    train_df = ctx["train_df"] if ctx and "train_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "train.parquet")
    val_df = ctx["val_df"] if ctx and "val_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "val.parquet")
    test_df = ctx["test_df"] if ctx and "test_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "test.parquet")
    y_train = train_df["generated"].values.astype(np.float32)
    y_val = val_df["generated"].values.astype(np.float32)
    y_test = test_df["generated"].values.astype(np.float32)

    h = compute_hash(mode=cfg.MODE, seed=cfg.SEED, n_test=len(y_test), stage="s06")
    if is_cached(out_dir, h):
        logger.info("Stage 06 cached — skipping")
        return

    # Load best DNN model
    if ctx and "dnn_model" in ctx:
        model = ctx["dnn_model"]
    else:
        widths = cfg.dnn_widths_list
        width = widths[-1] if widths else 1024
        model = MLP(X_train.shape[1], width)
        model.load_state_dict(torch.load(cfg.result_path / "05_dnn" / "best_model.pt", weights_only=True))

    out_dir.mkdir(parents=True, exist_ok=True)

    proba_train = predict_proba_dnn(model, X_train)
    proba_val = predict_proba_dnn(model, X_val)
    proba_test = predict_proba_dnn(model, X_test)

    # Threshold analysis on test
    logger.info("Threshold analysis (test set)...")
    thr_results = threshold_analysis(y_test, proba_test)
    (out_dir / "threshold_analysis.json").write_text(
        json.dumps(thr_results, indent=2), encoding="utf-8"
    )
    for name, m in thr_results.items():
        if isinstance(m, dict):
            logger.info("  %s: thr=%.3f  acc=%.3f  f1=%.3f", name, m["threshold"], m["accuracy"], m["f1"])
    logger.info("  AUC=%.4f  LogLoss=%.4f", thr_results["auc"], thr_results["logloss"])

    # Probability distributions
    plot_probability_distributions(
        {"train": (y_train, proba_train), "val": (y_val, proba_val), "test": (y_test, proba_test)},
        out_dir, title_prefix="DNN — ",
    )

    # Calibration
    plot_calibration(y_test, proba_test, out_dir, name="DNN (test)")

    # UMAP scatter (test set)
    logger.info("UMAP scatter on test embeddings...")
    plot_umap_proba(X_test, y_test, proba_test, out_dir, title_prefix="DNN — ")

    write_hash(out_dir, h)
    logger.info("Stage 06 artifacts saved to %s", out_dir)

    if ctx is not None:
        ctx["dnn_proba_test"] = proba_test
        ctx["dnn_metrics"] = thr_results
