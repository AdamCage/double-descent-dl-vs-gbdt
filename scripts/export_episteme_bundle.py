"""Write the EpistemeOS capture directory from pipeline artifacts.

Prints only byte counts and a sha256. It does not print curve values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
OUT = ROOT / "episteme_capture"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(values):
    return [float(value) for value in values]


def main() -> None:
    raise SystemExit(
        "Historical exporter is disabled: it hardcodes embeddinggemma, "
        "10% label noise and seed 42 and cannot attest the clean Qwen/CPU run. "
        "No artifacts were read or written. Use the evidence workflow in "
        "docs/automation.md and scripts/episteme_bridge.py status."
    )
    # Historical recipe retained for audit, never used as current provenance.
    dnn = _load(RESULT / "05_dnn" / "dnn_results.json")
    catboost = _load(RESULT / "07_catboost" / "catboost_results.json")
    pca = _load(RESULT / "04_embedding_analysis" / "pca_info.json")
    run_config = _load(RESULT / "07_catboost" / "run_config.json")
    manifest = _load(ROOT / "data" / "source_manifest.json")
    y_test = np.load(RESULT / "06_dnn_metrics" / "y_test.npy")
    dnn_proba = np.load(RESULT / "06_dnn_metrics" / "test_proba.npy")
    cb_proba = np.load(RESULT / "08_catboost_metrics" / "test_proba.npy")
    if not (len(y_test) == len(dnn_proba) == len(cb_proba)):
        raise SystemExit("test prediction lengths differ")
    sweeps = {}
    for name, runs in catboost["sweeps"].items():
        sweeps[name] = [{
            "label": run["label"],
            "train_logloss": _finite(run["train_logloss"]),
            "val_logloss": _finite(run["val_logloss"]),
        } for run in runs]
    main_run = catboost["main"]
    bundle = {
        "schema_version": 1,
        "origin": "full_pipeline",
        "mode": "full",
        "seed": 42,
        "label_noise": 0.1,
        "modelwise": [{
            "width": int(row["width"]),
            "num_params": int(row["num_params"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "train_acc": float(row["train_acc"]),
            "val_acc": float(row["val_acc"]),
        } for row in dnn["model_wise"]],
        "epochwise": [{
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "train_acc": float(row["train_acc"]),
            "val_acc": float(row["val_acc"]),
        } for row in dnn["epoch_wise"]],
        "catboost_main": {
            "train_logloss": _finite(main_run["train_logloss"]),
            "val_logloss": _finite(main_run["val_logloss"]),
        },
        "sweeps": sweeps,
        "y_test": [int(value) for value in y_test.tolist()],
        "dnn_proba": _finite(dnn_proba.tolist()),
        "cb_proba": _finite(cb_proba.tolist()),
        "meta": {
            "n_train": None,
            "n_val": None,
            "n_test": int(len(y_test)),
            "feature_dim": None,
            "pca_explained_variance_at_256": float(pca["explained_variance_at_256"]),
            "umap_plot_rows": int(pca.get("umap_plot_rows", 0)),
            "catboost_task_type": str(run_config["task_type"]),
            "embedding_model": "embeddinggemma:latest",
            "source_csv_sha256": manifest["csv_sha256"],
            "source_note": manifest["note"],
            "epoch_wise_width": int(dnn["epoch_wise_width"]),
            "epoch_wise_params": int(dnn["epoch_wise_params"]),
        },
    }
    counts = json.loads((RESULT / "01_data" / "cache_hash.json").read_text(encoding="utf-8"))
    del counts
    split_counts = {}
    for name in ("train", "val", "test"):
        import pyarrow.parquet as pq
        split_counts[name] = pq.ParquetFile(RESULT / "01_data" / f"{name}.parquet").metadata.num_rows
    reduced = np.load(RESULT / "04_embedding_analysis" / "train_reduced.npy", mmap_mode="r")
    bundle["meta"]["n_train"] = int(split_counts["train"])
    bundle["meta"]["n_val"] = int(split_counts["val"])
    bundle["meta"]["feature_dim"] = int(reduced.shape[1])
    if bundle["meta"]["n_test"] != int(split_counts["test"]):
        raise SystemExit("y_test length differs from the test parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "bundle.json"
    payload = json.dumps(bundle, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    target.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    print(f"bytes={len(payload)} sha256={digest} n_test={bundle['meta']['n_test']} feature_dim={bundle['meta']['feature_dim']}")


if __name__ == "__main__":
    main()
