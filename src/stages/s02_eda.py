"""Stage 02 — Exploratory Data Analysis with Plotly charts."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.cache import compute_hash, is_cached, write_hash
from src.config import Settings
from src.plotting import save_fig
from src.viz import stratified_indices


def _class_balance(df: pd.DataFrame, out: Path) -> None:
    counts = df["generated"].value_counts().sort_index()
    labels = ["Human (0)", "AI (1)"]
    fig = go.Figure(
        go.Bar(x=labels, y=counts.values, text=counts.values, textposition="auto")
    )
    fig.update_layout(title="Class balance", xaxis_title="Class", yaxis_title="Count")
    save_fig(fig, out / "class_balance")


def _binned_overlay(df: pd.DataFrame, column: str, title: str, out: Path, bins: int = 80) -> None:
    fig = go.Figure()
    for label_name, color in (("Human", "#636EFA"), ("AI", "#EF553B")):
        values = df.loc[df["label"] == label_name, column].to_numpy()
        counts, edges = np.histogram(values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Bar(x=centers, y=counts, name=label_name, marker_color=color, opacity=0.6))
    fig.update_layout(barmode="overlay", title=f"{title} distribution by class",
                      xaxis_title=title, yaxis_title="Count")
    save_fig(fig, out)


def _text_length_distribution(df: pd.DataFrame, out: Path) -> None:
    df = df.copy()
    df["char_len"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["label"] = df["generated"].map({0: "Human", 1: "AI"})

    _binned_overlay(df, "char_len", "Character length", out / "dist_char_len")
    _binned_overlay(df, "word_count", "Word count", out / "dist_word_count")

    stats = df.groupby("label")[["char_len", "word_count"]].describe()
    stats.to_csv(out / "text_stats.csv")


def _top_ngrams(df: pd.DataFrame, out: Path, n: int = 2, top_k: int = 20) -> None:
    indices = stratified_indices(df["generated"].to_numpy())
    sampled = df.iloc[indices]
    for label_int, label_name in [(0, "Human"), (1, "AI")]:
        texts = sampled.loc[sampled["generated"] == label_int, "text"]
        counter: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            words = text.lower().split()
            for i in range(len(words) - n + 1):
                counter[tuple(words[i : i + n])] += 1
        most = counter.most_common(top_k)
        ngrams = [" ".join(ng) for ng, _ in most]
        freqs = [c for _, c in most]
        fig = go.Figure(go.Bar(x=freqs, y=ngrams, orientation="h"))
        fig.update_layout(
            title=f"Top-{top_k} {n}-grams — {label_name}",
            yaxis=dict(autorange="reversed"),
            xaxis_title="Frequency",
        )
        save_fig(fig, out / f"top_{n}grams_{label_name.lower()}")


def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
    out_dir = cfg.result_path / "02_eda"

    if ctx and "train_df" in ctx:
        train_df = ctx["train_df"]
    else:
        train_df = pd.read_parquet(cfg.result_path / "01_data" / "train.parquet")

    h = compute_hash(mode=cfg.MODE, seed=cfg.SEED, n_rows=len(train_df))
    if is_cached(out_dir, h):
        logger.info("Stage 02 cached — skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating EDA charts ...")

    _class_balance(train_df, out_dir)
    _text_length_distribution(train_df, out_dir)
    _top_ngrams(train_df, out_dir, n=2)
    _top_ngrams(train_df, out_dir, n=3)

    write_hash(out_dir, h)
    logger.info("Stage 02 artifacts saved to %s", out_dir)
