"""Stage 08 — CatBoost metrics: identical analysis to S06 but for CatBoost."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.cache import compute_hash, is_cached, write_hash
from src.config import Settings
from src.stages.s06_dnn_metrics import (
    plot_calibration,
    plot_probability_distributions,
    plot_umap_proba,
    threshold_analysis,
)


def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
    out_dir = cfg.result_path / "08_catboost_metrics"

    X_train = ctx["X_train"] if ctx and "X_train" in ctx else np.load(cfg.result_path / "04_embedding_analysis" / "train_reduced.npy")
    X_val = ctx["X_val"] if ctx and "X_val" in ctx else np.load(cfg.result_path / "04_embedding_analysis" / "val_reduced.npy")
    X_test = ctx["X_test"] if ctx and "X_test" in ctx else np.load(cfg.result_path / "04_embedding_analysis" / "test_reduced.npy")

    train_df = ctx["train_df"] if ctx and "train_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "train.parquet")
    val_df = ctx["val_df"] if ctx and "val_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "val.parquet")
    test_df = ctx["test_df"] if ctx and "test_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "test.parquet")
    y_train = train_df["generated"].values.astype(np.float32)
    y_val = val_df["generated"].values.astype(np.float32)
    y_test = test_df["generated"].values.astype(np.float32)

    h = compute_hash(mode=cfg.MODE, seed=cfg.SEED, n_test=len(y_test), stage="s08")
    if is_cached(out_dir, h):
        logger.info("Stage 08 cached — skipping")
        return

    # Load model
    if ctx and "cb_model" in ctx:
        model = ctx["cb_model"]
    else:
        model = CatBoostClassifier()
        model.load_model(str(cfg.result_path / "07_catboost" / "best_model.cbm"))

    out_dir.mkdir(parents=True, exist_ok=True)

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    # Threshold analysis
    logger.info("CatBoost threshold analysis (test set)...")
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
        out_dir, title_prefix="CatBoost — ",
    )

    # Calibration
    plot_calibration(y_test, proba_test, out_dir, name="CatBoost (test)")

    # UMAP scatter
    logger.info("UMAP scatter on test embeddings (CatBoost predictions)...")
    plot_umap_proba(X_test, y_test, proba_test, out_dir, title_prefix="CatBoost — ")

    write_hash(out_dir, h)
    logger.info("Stage 08 artifacts saved to %s", out_dir)

    if ctx is not None:
        ctx["cb_proba_test"] = proba_test
        ctx["cb_metrics"] = thr_results
