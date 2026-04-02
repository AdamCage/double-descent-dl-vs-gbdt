"""Stage 04 — Embedding analysis: UMAP 2D, PCA dimensionality reduction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from src.cache import compute_hash, is_cached, write_hash
from src.config import Settings
from src.plotting import save_fig


def _umap_scatter(X: np.ndarray, y: np.ndarray, title: str, out: Path) -> None:
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X2d = reducer.fit_transform(X)
    df = pd.DataFrame({"UMAP-1": X2d[:, 0], "UMAP-2": X2d[:, 1], "class": y.astype(str)})
    fig = px.scatter(
        df, x="UMAP-1", y="UMAP-2", color="class",
        title=title, opacity=0.5,
        color_discrete_map={"0": "#636EFA", "1": "#EF553B"},
    )
    fig.update_traces(marker_size=3)
    save_fig(fig, out)


def _pca_analysis(
    X: np.ndarray, target_dims: int, logger: logging.Logger
) -> tuple[np.ndarray, dict[str, Any], PCA]:
    n_components = min(target_dims, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    info = {
        "n_components_fitted": int(n_components),
        "explained_variance_total": float(cum_var[-1]),
        "explained_variance_at_128": float(cum_var[min(127, len(cum_var) - 1)]),
        "explained_variance_at_256": float(cum_var[min(255, len(cum_var) - 1)]),
    }
    logger.info(
        "PCA %d dims: total var=%.4f  @128=%.4f  @256=%.4f",
        n_components, info["explained_variance_total"],
        info["explained_variance_at_128"], info["explained_variance_at_256"],
    )
    return X_pca, info, pca


def _text_features(df: pd.DataFrame) -> np.ndarray:
    """Extract informative text-level features: char length, word count, avg word length."""
    char_len = df["text"].str.len().values.astype(np.float32)
    word_count = df["text"].str.split().str.len().values.astype(np.float32)
    avg_word_len = np.where(word_count > 0, char_len / word_count, 0.0).astype(np.float32)
    return np.column_stack([char_len, word_count, avg_word_len])


def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
    out_dir = cfg.result_path / "04_embedding_analysis"

    emb_train = ctx["emb_train"] if ctx and "emb_train" in ctx else np.load(cfg.result_path / "03_embeddings" / "train.npy")
    emb_val = ctx["emb_val"] if ctx and "emb_val" in ctx else np.load(cfg.result_path / "03_embeddings" / "val.npy")
    emb_test = ctx["emb_test"] if ctx and "emb_test" in ctx else np.load(cfg.result_path / "03_embeddings" / "test.npy")

    train_df = ctx["train_df"] if ctx and "train_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "train.parquet")
    val_df = ctx["val_df"] if ctx and "val_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "val.parquet")
    test_df = ctx["test_df"] if ctx and "test_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "test.parquet")
    y_train = train_df["generated"].values

    h = compute_hash(shape=emb_train.shape, mode=cfg.MODE, seed=cfg.SEED)
    if is_cached(out_dir, h):
        logger.info("Stage 04 cached — loading reduced embeddings")
        X_train = np.load(out_dir / "train_reduced.npy")
        X_val = np.load(out_dir / "val_reduced.npy")
        X_test = np.load(out_dir / "test_reduced.npy")
        if ctx is not None:
            ctx["X_train"], ctx["X_val"], ctx["X_test"] = X_train, X_val, X_test
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # UMAP on original embeddings
    logger.info("Computing UMAP on original embeddings (%d dims)...", emb_train.shape[1])
    _umap_scatter(emb_train, y_train, "UMAP — original embeddings (train, by y_true)", out_dir / "umap_original")

    # PCA to 256
    logger.info("Running PCA (target 256 dims)...")
    X_pca_train, pca_info, pca_model = _pca_analysis(emb_train, 256, logger)

    # Explained variance plot
    cum_var = np.cumsum(pca_model.explained_variance_ratio_)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cum_var, mode="lines", name="Cumulative"))
    fig.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="95%")
    fig.update_layout(
        title="PCA — Cumulative explained variance",
        xaxis_title="Number of components",
        yaxis_title="Cumulative explained variance",
    )
    save_fig(fig, out_dir / "pca_explained_variance")

    # Decide final dimensionality
    if pca_info["explained_variance_at_256"] >= 0.95:
        final_dims = 128
        logger.info("95%% variance covered at 256 dims -> reducing to 128")
    else:
        final_dims = emb_train.shape[1]
        logger.info("95%% variance NOT covered at 256 dims -> keeping original %d dims", final_dims)

    if final_dims < emb_train.shape[1]:
        pca_final = PCA(n_components=final_dims, random_state=42)
        X_train = pca_final.fit_transform(emb_train).astype(np.float32)
        X_val = pca_final.transform(emb_val).astype(np.float32)
        X_test = pca_final.transform(emb_test).astype(np.float32)
    else:
        X_train, X_val, X_test = emb_train, emb_val, emb_test

    logger.info("Embedding dims after PCA: %d", X_train.shape[1])

    # Append text-level features (char_len, word_count, avg_word_len)
    logger.info("Extracting text-level features (char_len, word_count, avg_word_len)...")
    tf_train = _text_features(train_df)
    tf_val = _text_features(val_df)
    tf_test = _text_features(test_df)

    scaler = StandardScaler()
    tf_train = scaler.fit_transform(tf_train).astype(np.float32)
    tf_val = scaler.transform(tf_val).astype(np.float32)
    tf_test = scaler.transform(tf_test).astype(np.float32)

    X_train = np.hstack([X_train, tf_train])
    X_val = np.hstack([X_val, tf_val])
    X_test = np.hstack([X_test, tf_test])
    logger.info("Final feature dimensionality: %d (embeddings %d + text features %d)",
                X_train.shape[1], X_train.shape[1] - tf_train.shape[1], tf_train.shape[1])

    # UMAP on final features
    logger.info("Computing UMAP on final features (%d dims)...", X_train.shape[1])
    _umap_scatter(X_train, y_train, f"UMAP — final features ({X_train.shape[1]}d, train, by y_true)", out_dir / "umap_reduced")

    # Save
    np.save(out_dir / "train_reduced.npy", X_train)
    np.save(out_dir / "val_reduced.npy", X_val)
    np.save(out_dir / "test_reduced.npy", X_test)
    (out_dir / "pca_info.json").write_text(json.dumps(pca_info, indent=2), encoding="utf-8")
    write_hash(out_dir, h)
    logger.info("Stage 04 artifacts saved to %s", out_dir)

    if ctx is not None:
        ctx["X_train"], ctx["X_val"], ctx["X_test"] = X_train, X_val, X_test
