"""Stage 07 — CatBoost training: double descent log-loss + ablation sweeps."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool

from src.cache import compute_hash, is_cached, write_hash
from src.config import Settings
from src.plotting import save_fig


# ── Margin statistics ─────────────────────────────────────────────────────────

def margin_stats(model: CatBoostClassifier, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    raw = model.predict(X, prediction_type="RawFormulaVal").flatten()
    y_signed = 2 * y - 1  # 0->-1, 1->+1
    y_margin = y_signed * raw
    return {
        "mean_raw": float(np.mean(raw)),
        "std_raw": float(np.std(raw)),
        "mean_y_margin": float(np.mean(y_margin)),
        "q10_y_margin": float(np.quantile(y_margin, 0.10)),
        "q50_y_margin": float(np.quantile(y_margin, 0.50)),
        "q90_y_margin": float(np.quantile(y_margin, 0.90)),
        "frac_y_margin_gt0": float(np.mean(y_margin > 0)),
        "frac_y_margin_gt2": float(np.mean(y_margin > 2)),
    }


# ── Train one CatBoost model ─────────────────────────────────────────────────

def _train_cb(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    params: dict[str, Any],
    logger: logging.Logger,
    label: str = "",
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    tr_pool = Pool(X_tr, y_tr)
    va_pool = Pool(X_va, y_va)

    model = CatBoostClassifier(**params)
    model.fit(
        tr_pool,
        eval_set=va_pool,
        verbose=max(1, params.get("iterations", 1000) // 20),
    )

    evals = model.get_evals_result()
    train_ll = evals["learn"]["Logloss"]
    val_ll = evals["validation"]["Logloss"]

    logger.info(
        "  [%s] final train_ll=%.4f  val_ll=%.4f  iters=%d",
        label, train_ll[-1], val_ll[-1], len(train_ll),
    )

    return model, {
        "label": label,
        "params": {k: v for k, v in params.items() if k != "verbose"},
        "train_logloss": [float(v) for v in train_ll],
        "val_logloss": [float(v) for v in val_ll],
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_logloss_curves(
    runs: list[dict[str, Any]], out: Path, title: str,
) -> None:
    fig = go.Figure()
    for r in runs:
        iters = list(range(1, len(r["train_logloss"]) + 1))
        fig.add_trace(go.Scatter(x=iters, y=r["train_logloss"], mode="lines", name=f'{r["label"]} train', opacity=0.7))
        fig.add_trace(go.Scatter(x=iters, y=r["val_logloss"], mode="lines", name=f'{r["label"]} val', line=dict(dash="dash"), opacity=0.9))
    fig.update_layout(title=title, xaxis_title="Iteration", yaxis_title="LogLoss")
    save_fig(fig, out)


def _plot_margin_evolution(
    margin_snapshots: list[dict[str, Any]], out: Path, title: str,
) -> None:
    iters = [s["iteration"] for s in margin_snapshots]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=[s["mean_y_margin"] for s in margin_snapshots], mode="lines+markers", name="mean y*margin"))
    fig.add_trace(go.Scatter(x=iters, y=[s["q10_y_margin"] for s in margin_snapshots], mode="lines", name="q10", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=iters, y=[s["q50_y_margin"] for s in margin_snapshots], mode="lines", name="q50"))
    fig.add_trace(go.Scatter(x=iters, y=[s["q90_y_margin"] for s in margin_snapshots], mode="lines", name="q90", line=dict(dash="dot")))
    fig.update_layout(title=title, xaxis_title="Iteration", yaxis_title="y * margin")
    save_fig(fig, out)


# ── Main experiment ───────────────────────────────────────────────────────────

def _main_experiment(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    cfg: Settings, logger: logging.Logger, out_dir: Path,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    params = dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=cfg.CB_ITERATIONS,
        learning_rate=cfg.CB_LR,
        depth=cfg.CB_DEPTH,
        l2_leaf_reg=cfg.CB_L2_LEAF_REG,
        random_strength=cfg.CB_RANDOM_STRENGTH,
        bootstrap_type="Bayesian",
        use_best_model=False,
        random_seed=cfg.SEED,
        task_type="CPU",
        verbose=0,
    )

    logger.info("Main CatBoost experiment: %s", {k: v for k, v in params.items() if k != "verbose"})
    model, run_info = _train_cb(X_tr, y_tr, X_va, y_va, params, logger, label="main")

    # Margin dynamics at checkpoints
    n_iters = len(run_info["train_logloss"])
    checkpoint_iters = sorted(set(
        [1] + list(range(0, n_iters, max(1, n_iters // 20))) + [n_iters - 1]
    ))
    margin_snapshots: list[dict[str, Any]] = []
    for it in checkpoint_iters:
        truncated = model.copy()
        truncated.shrink(ntree_start=0, ntree_end=it + 1)
        stats = margin_stats(truncated, X_tr, y_tr)
        stats["iteration"] = it + 1
        margin_snapshots.append(stats)

    _plot_logloss_curves([run_info], out_dir / "main_logloss", "CatBoost — Main experiment LogLoss")
    _plot_margin_evolution(margin_snapshots, out_dir / "main_margin_evolution", "CatBoost — Margin dynamics (train)")

    # Feature importance
    fi = model.get_feature_importance()
    fig = go.Figure(go.Bar(x=list(range(len(fi))), y=fi))
    fig.update_layout(title="CatBoost — Feature importance", xaxis_title="Feature index", yaxis_title="Importance")
    save_fig(fig, out_dir / "feature_importance")

    return model, {
        "main": run_info,
        "margin_snapshots": margin_snapshots,
    }


# ── Ablation sweeps ───────────────────────────────────────────────────────────

def _ablation_sweeps(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    cfg: Settings, logger: logging.Logger, out_dir: Path,
) -> dict[str, Any]:
    base_params = dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=cfg.CB_ITERATIONS,
        learning_rate=cfg.CB_LR,
        depth=cfg.CB_DEPTH,
        l2_leaf_reg=cfg.CB_L2_LEAF_REG,
        random_strength=cfg.CB_RANDOM_STRENGTH,
        bootstrap_type="Bayesian",
        use_best_model=False,
        random_seed=cfg.SEED,
        task_type="CPU",
        verbose=0,
    )
    all_sweep_results: dict[str, Any] = {}

    # Sweep random_strength
    logger.info("Ablation: random_strength sweep")
    rs_runs: list[dict[str, Any]] = []
    for rs in [0, 1, 5, 10, 20]:
        p = {**base_params, "random_strength": rs}
        _, info = _train_cb(X_tr, y_tr, X_va, y_va, p, logger, label=f"rs={rs}")
        rs_runs.append(info)
    _plot_logloss_curves(rs_runs, out_dir / "sweep_random_strength", "CatBoost — Sweep random_strength")
    all_sweep_results["random_strength"] = rs_runs

    # Sweep l2_leaf_reg
    logger.info("Ablation: l2_leaf_reg sweep")
    l2_runs: list[dict[str, Any]] = []
    for l2 in [1, 3, 10, 30, 100]:
        p = {**base_params, "l2_leaf_reg": l2}
        _, info = _train_cb(X_tr, y_tr, X_va, y_va, p, logger, label=f"l2={l2}")
        l2_runs.append(info)
    _plot_logloss_curves(l2_runs, out_dir / "sweep_l2_leaf_reg", "CatBoost — Sweep l2_leaf_reg")
    all_sweep_results["l2_leaf_reg"] = l2_runs

    # Sweep learning_rate at constant lr*iterations
    logger.info("Ablation: learning_rate sweep (constant lr*iters)")
    lr_runs: list[dict[str, Any]] = []
    budget = cfg.CB_LR * cfg.CB_ITERATIONS
    for lr in [0.1, 0.05, 0.01, 0.005]:
        iters = max(100, int(budget / lr))
        p = {**base_params, "learning_rate": lr, "iterations": iters}
        _, info = _train_cb(X_tr, y_tr, X_va, y_va, p, logger, label=f"lr={lr},it={iters}")
        lr_runs.append(info)
    _plot_logloss_curves(lr_runs, out_dir / "sweep_learning_rate", "CatBoost — Sweep learning_rate (constant budget)")
    all_sweep_results["learning_rate"] = lr_runs

    return all_sweep_results


# ── Public entry point ────────────────────────────────────────────────────────

def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
    out_dir = cfg.result_path / "07_catboost"

    X_train = ctx["X_train"] if ctx and "X_train" in ctx else np.load(cfg.result_path / "04_embedding_analysis" / "train_reduced.npy")
    X_val = ctx["X_val"] if ctx and "X_val" in ctx else np.load(cfg.result_path / "04_embedding_analysis" / "val_reduced.npy")

    train_df = ctx["train_df"] if ctx and "train_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "train.parquet")
    val_df = ctx["val_df"] if ctx and "val_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "val.parquet")
    y_train = train_df["generated"].values.astype(np.float32)
    y_val = val_df["generated"].values.astype(np.float32)

    h = compute_hash(
        mode=cfg.MODE, seed=cfg.SEED,
        n_train=len(X_train), input_dim=X_train.shape[1],
        cb_iters=cfg.CB_ITERATIONS, cb_lr=cfg.CB_LR,
        cb_depth=cfg.CB_DEPTH, cb_l2=cfg.CB_L2_LEAF_REG,
        cb_rs=cfg.CB_RANDOM_STRENGTH,
    )
    if is_cached(out_dir, h):
        logger.info("Stage 07 cached — skipping CatBoost training")
        model_path = out_dir / "best_model.cbm"
        if model_path.exists() and ctx is not None:
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            ctx["cb_model"] = model
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Main experiment
    best_model, main_results = _main_experiment(X_train, y_train, X_val, y_val, cfg, logger, out_dir)

    # Ablation sweeps
    sweep_results = _ablation_sweeps(X_train, y_train, X_val, y_val, cfg, logger, out_dir)

    # Save
    all_results = {**main_results, "sweeps": sweep_results}
    (out_dir / "catboost_results.json").write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )
    best_model.save_model(str(out_dir / "best_model.cbm"))

    write_hash(out_dir, h)
    logger.info("Stage 07 artifacts saved to %s", out_dir)

    if ctx is not None:
        ctx["cb_model"] = best_model
        ctx["cb_results"] = all_results
