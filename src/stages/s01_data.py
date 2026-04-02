"""Stage 01 — Data loading, preprocessing, stratified train/val/test split."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cache import compute_hash, is_cached, write_hash
from src.config import Settings


def _load_csv(cfg: Settings, logger: logging.Logger) -> pd.DataFrame:
    csv_path = Path(cfg.DATA_PATH)
    if csv_path.exists():
        logger.info("Reading local CSV: %s", csv_path)
        return pd.read_csv(csv_path)

    logger.info("CSV not found locally — downloading from Kaggle: %s", cfg.KAGGLE_DATASET)
    import kagglehub

    downloaded = kagglehub.dataset_download(cfg.KAGGLE_DATASET)
    downloaded_path = Path(downloaded)
    candidates = list(downloaded_path.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {downloaded_path}")
    src = candidates[0]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(src, csv_path)
    logger.info("Saved dataset to %s", csv_path)
    return pd.read_csv(csv_path)


def _preprocess(df: pd.DataFrame, cfg: Settings, logger: logging.Logger) -> pd.DataFrame:
    n_before = len(df)
    df = df.dropna(subset=["text", "generated"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info("Preprocessing: %d -> %d rows", n_before, len(df))

    if cfg.is_fast:
        parts = []
        for cls in df["generated"].unique():
            sub = df[df["generated"] == cls]
            parts.append(sub.sample(n=min(1024, len(sub)), random_state=cfg.SEED))
        df = pd.concat(parts, ignore_index=True)
        logger.info("FAST mode: sampled to %d rows (balanced)", len(df))
    return df


def _split(
    df: pd.DataFrame, cfg: Settings, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["generated"]
    train_df, temp_df = train_test_split(
        df, test_size=1 - cfg.TRAIN_RATIO, stratify=y, random_state=cfg.SEED
    )
    relative_val = cfg.VAL_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - relative_val, stratify=temp_df["generated"], random_state=cfg.SEED
    )
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info("  %s: %d rows  (pos=%.1f%%)", name, len(part), part["generated"].mean() * 100)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
    out_dir = cfg.result_path / "01_data"
    h = compute_hash(
        mode=cfg.MODE,
        seed=cfg.SEED,
        data_path=cfg.DATA_PATH,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
    )

    if is_cached(out_dir, h):
        logger.info("Stage 01 cached — loading splits from parquet")
        train_df = pd.read_parquet(out_dir / "train.parquet")
        val_df = pd.read_parquet(out_dir / "val.parquet")
        test_df = pd.read_parquet(out_dir / "test.parquet")
    else:
        df = _load_csv(cfg, logger)
        df = _preprocess(df, cfg, logger)
        train_df, val_df, test_df = _split(df, cfg, logger)

        out_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(out_dir / "train.parquet", index=False)
        val_df.to_parquet(out_dir / "val.parquet", index=False)
        test_df.to_parquet(out_dir / "test.parquet", index=False)
        write_hash(out_dir, h)
        logger.info("Stage 01 artifacts saved to %s", out_dir)

    if ctx is not None:
        ctx["train_df"] = train_df
        ctx["val_df"] = val_df
        ctx["test_df"] = test_df
