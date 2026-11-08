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

from src.cache import code_digest, compute_hash, is_cached, write_hash
from src.config import Settings
from src.plotting import save_fig
from src.protocol import PROTOCOL_REVISION, held_out_threshold_report
from src.stages.s05_dnn_train import MLP
from src.viz import PLOT_ROWS, stratified_indices


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


def threshold_analysis(
    y_selection: np.ndarray,
    proba_selection: np.ndarray,
    y_evaluation: np.ndarray,
    proba_evaluation: np.ndarray,
) -> dict[str, Any]:
    """Пороги замораживаются на validation и оцениваются на test.

    Прежняя версия подбирала F1 и Youden на том же test, где считала метрики.
    """
    return held_out_threshold_report(
        y_selection, proba_selection, y_evaluation, proba_evaluation,
    )


def plot_probability_distributions(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    out: Path,
    title_prefix: str = "",
) -> None:
    """Histogram of predicted probabilities split by y_true, for each data split."""
    for split_name, (y, proba) in splits.items():
        fig = go.Figure()
        for class_name, class_value, color in (("Human (0)", 0, "#636EFA"), ("AI (1)", 1, "#EF553B")):
            mask = np.asarray(y) == class_value
            counts, edges = np.histogram(np.asarray(proba)[mask], bins=60, range=(0.0, 1.0))
            centers = (edges[:-1] + edges[1:]) / 2
            fig.add_trace(go.Bar(x=centers, y=counts, name=class_name, marker_color=color, opacity=0.6))
        fig.update_layout(barmode="overlay", title=f"{title_prefix}Probability distribution — {split_name}",
                          xaxis_title="Predicted probability", yaxis_title="Count")
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

    index = stratified_indices(y, PLOT_ROWS, 42)
    X, y, proba = X[index], y[index], proba[index]
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

    h = compute_hash(
        mode=cfg.MODE, seed=cfg.SEED, n_test=len(y_test), stage="s06",
        protocol=PROTOCOL_REVISION, threshold_rule="validation_frozen",
        code=code_digest(Path(__file__)),
    )
    if is_cached(out_dir, h):
        logger.info("Stage 06 cached — skipping")
        return

    if ctx and "dnn_model" in ctx:
        model = ctx["dnn_model"]
    else:
        selection_path = cfg.result_path / "05_dnn" / "model_selection.json"
        if selection_path.exists():
            width = int(json.loads(selection_path.read_text(encoding="utf-8"))["width"])
        else:
            widths = cfg.dnn_widths_list
            width = widths[-1] if widths else 1024
            logger.warning("model_selection.json is absent; loading the widest width")
        model = MLP(X_train.shape[1], width)
        model.load_state_dict(torch.load(cfg.result_path / "05_dnn" / "best_model.pt", weights_only=True))

    out_dir.mkdir(parents=True, exist_ok=True)

    proba_train = predict_proba_dnn(model, X_train)
    proba_val = predict_proba_dnn(model, X_val)
    proba_test = predict_proba_dnn(model, X_test)
    np.save(out_dir / "test_proba.npy", proba_test)
    np.save(out_dir / "y_test.npy", y_test)

    logger.info("Thresholds frozen on validation, metrics on test")
    thr_results = threshold_analysis(y_val, proba_val, y_test, proba_test)
    (out_dir / "threshold_analysis.json").write_text(
        json.dumps(thr_results, indent=2), encoding="utf-8"
    )
    for name, metrics in thr_results.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            logger.info(
                "  %s: thr=%.3f  acc=%.3f  f1=%.3f",
                name, metrics["threshold"], metrics["accuracy"], metrics["f1"],
            )
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
