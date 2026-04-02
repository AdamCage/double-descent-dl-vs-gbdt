"""Stage 10 — Metadata: hardware, versions, seed, timing."""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import Settings


def _hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "python_version": sys.version,
    }

    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
    except ImportError:
        pass

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch_version"] = "N/A"

    try:
        import catboost
        info["catboost_version"] = catboost.__version__
    except ImportError:
        info["catboost_version"] = "N/A"

    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas
        info["pandas_version"] = pandas.__version__
    except ImportError:
        pass

    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass

    try:
        import plotly
        info["plotly_version"] = plotly.__version__
    except ImportError:
        pass

    return info


def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
    out_dir = cfg.result_path
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": cfg.MODE,
        "seed": cfg.SEED,
        "hardware": _hardware_info(),
        "config": {
            "data_path": cfg.DATA_PATH,
            "embedding_backend": cfg.EMBEDDING_BACKEND,
            "embedding_model": cfg.EMBEDDING_MODEL,
            "dnn_widths": cfg.DNN_WIDTHS,
            "dnn_modelwise_epochs": cfg.DNN_MODELWISE_EPOCHS,
            "dnn_epochwise_epochs": cfg.DNN_EPOCHWISE_EPOCHS,
            "dnn_lr": cfg.DNN_LR,
            "dnn_label_noise": cfg.DNN_LABEL_NOISE,
            "cb_iterations": cfg.CB_ITERATIONS,
            "cb_lr": cfg.CB_LR,
            "cb_depth": cfg.CB_DEPTH,
            "cb_l2_leaf_reg": cfg.CB_L2_LEAF_REG,
            "cb_random_strength": cfg.CB_RANDOM_STRENGTH,
        },
    }

    if ctx and "timings" in ctx:
        meta["timings_sec"] = {k: round(v, 2) for k, v in ctx["timings"].items()}

    out_file = out_dir / "meta.json"
    out_file.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Metadata saved to %s", out_file)
