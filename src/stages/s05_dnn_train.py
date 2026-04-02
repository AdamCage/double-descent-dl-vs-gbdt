"""Stage 05 — DNN training: model-wise and epoch-wise double descent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.cache import compute_hash, is_cached, write_hash
from src.config import Settings
from src.plotting import save_fig


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_loaders(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    batch_size: int = 512,
) -> tuple[DataLoader, DataLoader]:
    ds_tr = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).float(),
    )
    ds_va = TensorDataset(
        torch.from_numpy(X_va).float(),
        torch.from_numpy(y_va).float(),
    )
    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(ds_va, batch_size=batch_size, shuffle=False),
    )


@torch.no_grad()
def _evaluate(model: MLP, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    all_logits, all_y = [], []
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        n += len(yb)
        all_logits.append(logits.cpu())
        all_y.append(yb.cpu())
    avg_loss = total_loss / max(n, 1)
    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    y_np = torch.cat(all_y).numpy()
    acc = accuracy_score(y_np.round(), (probs > 0.5).astype(float))
    return avg_loss, acc


def _train_model(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    log_every: int = 1,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if epoch % log_every == 0 or epoch == epochs:
            tr_loss, tr_acc = _evaluate(model, train_loader, criterion, device)
            va_loss, va_acc = _evaluate(model, val_loader, criterion, device)
            history.append({
                "epoch": epoch,
                "train_loss": tr_loss, "val_loss": va_loss,
                "train_acc": tr_acc, "val_acc": va_acc,
            })
    return history


# ── Model-wise double descent ────────────────────────────────────────────────

def _model_wise(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    cfg: Settings, logger: logging.Logger, out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    widths = cfg.dnn_widths_list
    epochs = cfg.DNN_MODELWISE_EPOCHS
    lr = cfg.DNN_LR
    results: list[dict[str, Any]] = []

    for w in widths:
        model = MLP(X_tr.shape[1], w)
        logger.info("  Model-wise: width=%d  params=%d  epochs=%d", w, model.num_params, epochs)
        tr_loader, va_loader = _make_loaders(X_tr, y_tr, X_va, y_va)
        history = _train_model(model, tr_loader, va_loader, epochs, lr, device, log_every=max(1, epochs // 20))
        final = history[-1]
        results.append({
            "width": w,
            "num_params": model.num_params,
            **final,
        })
        logger.info(
            "    -> train_loss=%.4f  val_loss=%.4f  train_acc=%.3f  val_acc=%.3f",
            final["train_loss"], final["val_loss"], final["train_acc"], final["val_acc"],
        )

    # Plot model-wise curves
    params = [r["num_params"] for r in results]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=params, y=[r["train_loss"] for r in results], mode="lines+markers", name="Train loss"))
    fig.add_trace(go.Scatter(x=params, y=[r["val_loss"] for r in results], mode="lines+markers", name="Val loss"))
    fig.update_layout(
        title="Model-wise double descent: LogLoss vs #params",
        xaxis_title="Number of parameters", yaxis_title="LogLoss",
        xaxis_type="log",
    )
    save_fig(fig, out_dir / "model_wise_logloss")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=params, y=[r["train_acc"] for r in results], mode="lines+markers", name="Train acc"))
    fig2.add_trace(go.Scatter(x=params, y=[r["val_acc"] for r in results], mode="lines+markers", name="Val acc"))
    fig2.update_layout(
        title="Model-wise: Accuracy vs #params",
        xaxis_title="Number of parameters", yaxis_title="Accuracy",
        xaxis_type="log",
    )
    save_fig(fig2, out_dir / "model_wise_accuracy")

    return {"model_wise": results}


# ── Epoch-wise double descent ────────────────────────────────────────────────

def _epoch_wise(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    cfg: Settings, logger: logging.Logger, out_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], MLP]:
    widths = cfg.dnn_widths_list
    width = widths[-1] if widths else 1024
    epochs = cfg.DNN_EPOCHWISE_EPOCHS
    lr = cfg.DNN_LR

    model = MLP(X_tr.shape[1], width)
    logger.info("Epoch-wise: width=%d  params=%d  epochs=%d", width, model.num_params, epochs)
    tr_loader, va_loader = _make_loaders(X_tr, y_tr, X_va, y_va)
    history = _train_model(model, tr_loader, va_loader, epochs, lr, device, log_every=max(1, epochs // 200))

    ep = [h["epoch"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=[h["train_loss"] for h in history], mode="lines", name="Train loss"))
    fig.add_trace(go.Scatter(x=ep, y=[h["val_loss"] for h in history], mode="lines", name="Val loss"))
    fig.update_layout(
        title=f"Epoch-wise double descent (width={width}, params={model.num_params})",
        xaxis_title="Epoch", yaxis_title="LogLoss",
    )
    save_fig(fig, out_dir / "epoch_wise_logloss")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ep, y=[h["train_acc"] for h in history], mode="lines", name="Train acc"))
    fig2.add_trace(go.Scatter(x=ep, y=[h["val_acc"] for h in history], mode="lines", name="Val acc"))
    fig2.update_layout(
        title=f"Epoch-wise: Accuracy (width={width})",
        xaxis_title="Epoch", yaxis_title="Accuracy",
    )
    save_fig(fig2, out_dir / "epoch_wise_accuracy")

    return {"epoch_wise": history, "epoch_wise_width": width, "epoch_wise_params": model.num_params}, model


# ── Label noise injection ─────────────────────────────────────────────────────

def _inject_label_noise(y: np.ndarray, noise_rate: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    y_noisy = y.copy()
    n_flip = int(len(y) * noise_rate)
    idx = rng.choice(len(y), size=n_flip, replace=False)
    y_noisy[idx] = 1 - y_noisy[idx]
    return y_noisy


# ── Public entry point ────────────────────────────────────────────────────────

def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
    out_dir = cfg.result_path / "05_dnn"

    X_train = ctx["X_train"] if ctx and "X_train" in ctx else np.load(cfg.result_path / "04_embedding_analysis" / "train_reduced.npy")
    X_val = ctx["X_val"] if ctx and "X_val" in ctx else np.load(cfg.result_path / "04_embedding_analysis" / "val_reduced.npy")

    train_df = ctx["train_df"] if ctx and "train_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "train.parquet")
    val_df = ctx["val_df"] if ctx and "val_df" in ctx else pd.read_parquet(cfg.result_path / "01_data" / "val.parquet")
    y_train = train_df["generated"].values.astype(np.float32)
    y_val = val_df["generated"].values.astype(np.float32)

    h = compute_hash(
        mode=cfg.MODE, seed=cfg.SEED,
        input_dim=X_train.shape[1], n_train=len(X_train),
        widths=cfg.DNN_WIDTHS, mw_epochs=cfg.DNN_MODELWISE_EPOCHS,
        ew_epochs=cfg.DNN_EPOCHWISE_EPOCHS, lr=cfg.DNN_LR,
        noise=cfg.DNN_LABEL_NOISE,
    )
    if is_cached(out_dir, h):
        logger.info("Stage 05 cached — skipping DNN training")
        if ctx is not None:
            model_path = out_dir / "best_model.pt"
            if model_path.exists():
                widths = cfg.dnn_widths_list
                width = widths[-1] if widths else 1024
                best_model = MLP(X_train.shape[1], width)
                best_model.load_state_dict(torch.load(model_path, weights_only=True))
                ctx["dnn_model"] = best_model
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    device = _get_device()
    logger.info("Device: %s", device)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # Optionally inject label noise to amplify interpolation peak
    if cfg.DNN_LABEL_NOISE > 0:
        y_train_noisy = _inject_label_noise(y_train, cfg.DNN_LABEL_NOISE, cfg.SEED)
        logger.info("Injected %.0f%% label noise for model-wise experiment", cfg.DNN_LABEL_NOISE * 100)
    else:
        y_train_noisy = y_train

    # Model-wise experiment
    mw_results = _model_wise(X_train, y_train_noisy, X_val, y_val, cfg, logger, out_dir, device)

    # Epoch-wise experiment (clean labels)
    ew_results, best_model = _epoch_wise(X_train, y_train, X_val, y_val, cfg, logger, out_dir, device)

    # Save all results
    all_results = {**mw_results, **ew_results}
    (out_dir / "dnn_results.json").write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )
    torch.save(best_model.state_dict(), out_dir / "best_model.pt")

    write_hash(out_dir, h)
    logger.info("Stage 05 artifacts saved to %s", out_dir)

    if ctx is not None:
        ctx["dnn_model"] = best_model
        ctx["dnn_results"] = all_results
