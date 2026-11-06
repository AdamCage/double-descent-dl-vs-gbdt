"""Малая проверка позднего ускорения CatBoost.

Это не измерение по корпусу эссе и не чтение эмбеддингов. CatBoost на CPU,
два потока, чтобы не отбирать ядра у идущего этапа s03. Правила вердиктов
зафиксированы в src.protocol и здесь не подстраиваются под полученную кривую.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import Settings
from src.dynamics import describe_change, replicate_change_points
from src.protocol import (
    h1_verdict,
    h4_interior_peak,
    h4_verdict,
    held_out_threshold_report,
    normalized_time_closer,
    select_operating_points,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result" / "pilot_late_acceleration"
SEEDS = [int(part) for part in Settings.model_fields["REPLICATE_SEEDS"].default.split(",")]
REFERENCE = {
    "iterations": 200,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 20.0,
    "border_count": 32,
}
H2_GRID = (
    {"learning_rate": 0.05, "l2_leaf_reg": 1.0, "iterations": 200},
    {"learning_rate": 0.05, "l2_leaf_reg": 10.0, "iterations": 200},
    {"learning_rate": 0.2, "l2_leaf_reg": 1.0, "iterations": 50},
    {"learning_rate": 0.2, "l2_leaf_reg": 10.0, "iterations": 50},
)
ENSEMBLE_CHECKPOINTS = (40, 100, 200)


def _split(features: np.ndarray, labels: np.ndarray, seed: int):
    x_train, x_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.5, stratify=labels, random_state=seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def _fit(x_train, y_train, x_val, y_val, seed: int, **overrides) -> CatBoostClassifier:
    params = {
        **REFERENCE,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "custom_metric": ["Accuracy", "AUC"],
        "bootstrap_type": "Bayesian",
        "use_best_model": False,
        "task_type": "CPU",
        "thread_count": 2,
        "allow_writing_files": False,
        "random_seed": seed,
        "verbose": False,
    }
    params.update(overrides)
    model = CatBoostClassifier(**params)
    model.fit(Pool(x_train, y_train), eval_set=Pool(x_val, y_val))
    return model


def _val_logloss(model: CatBoostClassifier) -> list[float]:
    return [float(value) for value in model.get_evals_result()["validation"]["Logloss"]]


def _zero_one(model: CatBoostClassifier, features: np.ndarray, labels: np.ndarray) -> float:
    predicted = np.asarray(model.predict(features)).reshape(-1).astype(int)
    return float(np.mean(predicted != np.asarray(labels).astype(int)))


def _checkpoint_error(model: CatBoostClassifier, features: np.ndarray, labels: np.ndarray, trees: int) -> float:
    clipped = model.copy()
    clipped.shrink(ntree_start=0, ntree_end=trees)
    return _zero_one(clipped, features, labels)


def _hard_margin(model: CatBoostClassifier, features: np.ndarray, labels: np.ndarray) -> float:
    raw = np.asarray(model.predict(features, prediction_type="RawFormulaVal")).reshape(-1)
    signed = (2 * np.asarray(labels).astype(float) - 1.0) * raw
    order = np.argsort(signed)
    count = max(1, int(round(0.2 * len(signed))))
    return float(np.mean(signed[order[:count]]))


def _dataset(name: str, seed: int):
    if name == "synthetic":
        features, labels = make_classification(
            n_samples=600,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            flip_y=0.05,
            class_sep=1.0,
            random_state=seed,
        )
        return features.astype(np.float32), labels.astype(int)
    features, labels = load_breast_cancer(return_X_y=True)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=int)


def _h2_record(curves: list[dict]) -> dict:
    comparisons = []
    seeds = sorted({row["seed"] for row in curves})
    for seed in seeds:
        for l2 in (1.0, 10.0):
            pair = [
                row for row in curves
                if row["seed"] == seed and row["l2_leaf_reg"] == l2
            ]
            pair = sorted(pair, key=lambda row: row["learning_rate"])
            if len(pair) != 2:
                continue
            closer = normalized_time_closer(
                [row["argmin_iteration"] for row in pair],
                [row["learning_rate"] for row in pair],
            )
            comparisons.append({
                "seed": seed,
                "l2_leaf_reg": l2,
                "learning_rates": [row["learning_rate"] for row in pair],
                "argmin_iteration": [row["argmin_iteration"] for row in pair],
                "normalized_time_closer": closer,
            })
    at_cap = sum(1 for row in curves if row["argmin_iteration"] == row["iterations"])
    return {
        "status": "descriptive",
        "comparisons": comparisons,
        "n_curves": len(curves),
        "argmin_at_iteration_cap": at_cap,
        "note": (
            "Меньший разброс по оси iteration×learning_rate не доказывает механизм регуляризации. "
            "Если argmin стоит на последней итерации, совпадение iteration×lr отражает общий бюджет, а не сближение динамики."
        ),
    }


def _run_dataset(name: str) -> dict:
    reference_changes = []
    zero_changes = []
    border_peaks = []
    depth_peaks = []
    ensemble_peaks = []
    h2_curves = []
    h3 = []
    threshold_checks = []
    for seed in SEEDS:
        features, labels = _dataset(name, seed)
        x_train, x_val, x_test, y_train, y_val, y_test = _split(features, labels, seed)
        reference = _fit(x_train, y_train, x_val, y_val, seed)
        zero = _fit(x_train, y_train, x_val, y_val, seed, random_strength=0.0)
        reference_loss = _val_logloss(reference)
        zero_loss = _val_logloss(zero)
        reference_changes.append(describe_change(reference_loss))
        zero_changes.append(describe_change(zero_loss))

        border_errors = []
        for border in (8, 32, 128):
            model = reference if border == 32 else _fit(
                x_train, y_train, x_val, y_val, seed, border_count=border,
            )
            border_errors.append(_zero_one(model, x_test, y_test))
        border_peaks.append(h4_interior_peak(border_errors))

        depth_errors = []
        for depth in (4, 6, 8):
            model = reference if depth == 6 else _fit(
                x_train, y_train, x_val, y_val, seed, depth=depth,
            )
            depth_errors.append(_zero_one(model, x_test, y_test))
        depth_peaks.append(h4_interior_peak(depth_errors))

        ensemble_errors = [
            _checkpoint_error(reference, x_test, y_test, trees)
            for trees in ENSEMBLE_CHECKPOINTS
        ]
        ensemble_peaks.append(h4_interior_peak(ensemble_errors))

        for spec in H2_GRID:
            model = _fit(x_train, y_train, x_val, y_val, seed, **spec)
            loss = _val_logloss(model)
            h2_curves.append({
                "seed": seed,
                "learning_rate": spec["learning_rate"],
                "l2_leaf_reg": spec["l2_leaf_reg"],
                "iterations": spec["iterations"],
                "argmin_iteration": int(np.argmin(loss)) + 1,
                "final_val_logloss": loss[-1],
            })

        midpoint = reference.copy()
        midpoint.shrink(ntree_start=0, ntree_end=len(reference_loss) // 2)
        h3.append({
            "seed": seed,
            "hard20_margin_mid": _hard_margin(midpoint, x_val, y_val),
            "hard20_margin_end": _hard_margin(reference, x_val, y_val),
            "auc_end": float(roc_auc_score(y_val, reference.predict_proba(x_val)[:, 1])),
        })
        if seed == SEEDS[0]:
            report = held_out_threshold_report(
                y_val, reference.predict_proba(x_val)[:, 1],
                y_test, reference.predict_proba(x_test)[:, 1],
            )
            biased = select_operating_points(y_test, reference.predict_proba(x_test)[:, 1])
            threshold_checks.append({
                "frozen_f1_threshold": report["f1_best"]["threshold"],
                "test_selected_f1_threshold": biased["f1_best"],
                "held_out_f1": report["f1_best"]["f1"],
                "held_out_logloss": report["logloss"],
                "thresholds_differ": report["f1_best"]["threshold"] != biased["f1_best"],
            })

    reference_replication = replicate_change_points(reference_changes)
    zero_replication = replicate_change_points(zero_changes)
    return {
        "origin": "small_data_pilot",
        "corpus_essay_status": "not_assessed",
        "n_independent_runs": len(SEEDS),
        "seeds": SEEDS,
        "change_point_reference": reference_replication,
        "change_point_random_strength_0": zero_replication,
        "H1": {
            "verdict": h1_verdict(reference_replication, zero_replication),
            "prediction": "При random_strength=0 воспроизведённое позднее ускорение пропадает.",
        },
        "H2": _h2_record(h2_curves),
        "H3": {
            "status": "descriptive",
            "runs": h3,
            "note": "Сдвиг margin трудных объектов не доказывает отдельную фазу и не заменяет AUC.",
        },
        "H4": {
            "border_count": h4_verdict(border_peaks),
            "depth": h4_verdict(depth_peaks),
            "ensemble_size": h4_verdict(ensemble_peaks),
            "note": "Три точки сетки не устанавливают интерполяционный порог. coarse_peak не означает double descent.",
        },
        "threshold_check": threshold_checks,
        "reference_change_points": reference_changes,
        "zero_strength_change_points": zero_changes,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "review-2026-11-04",
        "task_type": "CPU",
        "thread_count": 2,
        "embedding_corpus": "not_read",
        "scientific_validity": "not_assessed",
        "datasets": {
            "synthetic": _run_dataset("synthetic"),
            "breast_cancer": _run_dataset("breast_cancer"),
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "wrote": str(OUT / "summary.json"),
        "synthetic_H1": summary["datasets"]["synthetic"]["H1"]["verdict"],
        "breast_cancer_H1": summary["datasets"]["breast_cancer"]["H1"]["verdict"],
        "synthetic_change": summary["datasets"]["synthetic"]["change_point_reference"]["inference"],
        "breast_cancer_change": summary["datasets"]["breast_cancer"]["change_point_reference"]["inference"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
