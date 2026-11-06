"""Правила оценки из ревью 2026-11-04.

Функции ниже фиксируют протокол до просмотра кривых корпуса.
Они не объявляют научный успех и не заменяют независимые повторы.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROTOCOL_REVISION = "review-2026-11-04"
VARIANCE_TARGET = 0.95
CONVERGENCE_RELATIVE_TOLERANCE = 0.05
PEAK_MARGIN = 0.01


def sha256_array(array: np.ndarray) -> str:
    """Полный хеш массива. Нужен ключу кэша, а не усечённой выборке строк."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    view = memoryview(contiguous).cast("B")
    step = 1 << 20
    for offset in range(0, view.nbytes, step):
        digest.update(view[offset : offset + step])
    return digest.hexdigest()


def choose_pca_dimensions(
    cumulative_variance: Sequence[float] | np.ndarray,
    original_dim: int,
    target: float = VARIANCE_TARGET,
) -> dict[str, Any]:
    """Оставить наименьшее k с накопленной долей >= target.

    Прежняя ошибка: при 95% на 256 компонентах размерность сбрасывалась
    до 128, даже если 128 объясняли существенно меньше target.
    Если target не достигнут внутри посчитанных компонент, остаётся исходная
    размерность эмбеддинга.
    """
    cumulative = np.asarray(cumulative_variance, dtype=float)
    if cumulative.ndim != 1 or len(cumulative) == 0:
        raise ValueError("cumulative variance must be a nonempty 1d series")
    if original_dim < 1:
        raise ValueError("original_dim must be positive")
    capped = cumulative[: min(len(cumulative), original_dim)]

    def _at(k: int) -> float:
        return float(capped[min(k, len(capped)) - 1])

    diagnostics = {
        "explained_variance_at_128": _at(128),
        "explained_variance_at_256": _at(256),
        "variance_target": float(target),
        "search_components": int(len(capped)),
    }
    if float(capped[-1]) < target:
        return {
            **diagnostics,
            "final_dims": int(original_dim),
            "variance_at_choice": float(capped[-1]),
            "rule": "variance_target_not_met_within_cap",
        }
    index = int(np.searchsorted(capped, target, side="left"))
    k = index + 1
    return {
        **diagnostics,
        "final_dims": int(k),
        "variance_at_choice": float(capped[k - 1]),
        "rule": "smallest_k_meeting_variance_target",
    }


def select_operating_points(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Пороги только по selection-split. Evaluation-split сюда не передаётся."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        f1_threshold = 0.5
    else:
        f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        f1_for_thresholds = f1_scores[:-1]
        if len(f1_for_thresholds) != len(thresholds):
            f1_for_thresholds = f1_scores[: len(thresholds)]
        f1_threshold = float(thresholds[int(np.argmax(f1_for_thresholds))])

    _fpr, tpr, roc_thresholds = roc_curve(y_true, proba)
    finite = np.isfinite(roc_thresholds)
    if not np.any(finite):
        roc_threshold = 0.5
    else:
        youden = np.where(finite, tpr - _fpr, -np.inf)
        roc_threshold = float(roc_thresholds[int(np.argmax(youden))])

    return {
        "f1_best": f1_threshold,
        "roc_youden": roc_threshold,
        "prevalence": float(np.mean(y_true)),
        "0.5": 0.5,
    }


def _metric_row(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "zero_one_error": float(np.mean(predictions != np.asarray(y_true).astype(int))),
    }


def evaluate_operating_points(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0 - 1e-15)
    results: dict[str, Any] = {
        name: _metric_row(y_true, proba, threshold)
        for name, threshold in thresholds.items()
    }
    results["auc"] = float(roc_auc_score(y_true, proba))
    results["logloss"] = float(log_loss(y_true, proba))
    results["brier"] = float(brier_score_loss(y_true, proba))
    results["zero_one_error_at_0.5"] = results["0.5"]["zero_one_error"]
    return results


def held_out_threshold_report(
    y_selection: np.ndarray,
    proba_selection: np.ndarray,
    y_evaluation: np.ndarray,
    proba_evaluation: np.ndarray,
    *,
    selection_name: str = "validation",
    evaluation_name: str = "test",
) -> dict[str, Any]:
    """Заморозить пороги на selection-split и посчитать метрики на evaluation-split."""
    if y_selection is y_evaluation or proba_selection is proba_evaluation:
        raise ValueError("threshold selection and evaluation received the same array")
    thresholds = select_operating_points(y_selection, proba_selection)
    evaluation = evaluate_operating_points(y_evaluation, proba_evaluation, thresholds)
    selection = evaluate_operating_points(y_selection, proba_selection, thresholds)
    evaluation["thresholds"] = thresholds
    evaluation["threshold_source"] = selection_name
    evaluation["evaluation_split"] = evaluation_name
    evaluation["selection_metrics"] = selection
    evaluation["protocol"] = "thresholds_frozen_on_selection_split"
    evaluation["protocol_revision"] = PROTOCOL_REVISION
    return evaluation


def optimization_settled(
    train_losses: Sequence[float],
    relative_tolerance: float = CONVERGENCE_RELATIVE_TOLERANCE,
) -> bool:
    """True, если train loss на последней пятой записанных точек почти не падает.

    Одинаковый бюджет эпох сам по себе не означает сопоставимую сходимость.
    Проверка смотрит на записанную сетку, а не на каждую эпоху.
    """
    losses = [float(value) for value in train_losses]
    if len(losses) < 2:
        return False
    window = max(2, len(losses) // 5)
    start = losses[-window]
    end = losses[-1]
    scale = max(abs(start), 1e-8)
    return (start - end) / scale <= relative_tolerance


def select_comparison_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    loss_key: str = "val_loss",
    settled_key: str = "optimization_settled",
    width_key: str = "width",
) -> tuple[int, str]:
    """Индекс модели для сравнения: минимум val loss среди сошедшихся ширин.

    Если ни одна ширина не сошлась, берётся минимум val loss, а правило
    прямо говорит, что оптимум не подтверждён.
    """
    if not rows:
        raise ValueError("no rows to select from")
    settled = [index for index, row in enumerate(rows) if row[settled_key]]
    pool = settled if settled else list(range(len(rows)))
    chosen = min(pool, key=lambda index: (float(rows[index][loss_key]), int(rows[index][width_key])))
    rule = "settled_validation_logloss" if settled else "no_width_settled_best_validation_logloss"
    return chosen, rule


def noise_regime_name(rate: float) -> str:
    if float(rate) == 0.0:
        return "clean"
    return f"label_noise_{float(rate):.3f}"


def comparison_status(dnn_noise: float | None, catboost_noise: float | None) -> str:
    """Сравнение допустимо только при записанном и одинаковом шуме меток."""
    if dnn_noise is None or catboost_noise is None:
        return "noise_regime_not_recorded"
    if float(dnn_noise) != float(catboost_noise):
        return "blocked_noise_regime_mismatch"
    return "same_noise_regime"


def summarize_replicates(values: Sequence[float]) -> dict[str, Any]:
    """Диапазон независимых запусков. Это не популяционный доверительный интервал."""
    series = [float(value) for value in values]
    count = len(series)
    if count == 0:
        return {
            "n_independent_runs": 0,
            "mean": None,
            "interval_available": False,
            "interval_low": None,
            "interval_high": None,
            "interval_method": None,
        }
    mean = float(sum(series) / count)
    if count < 3:
        return {
            "n_independent_runs": count,
            "mean": mean,
            "interval_available": False,
            "interval_low": None,
            "interval_high": None,
            "interval_method": None,
        }
    ordered = sorted(series)
    return {
        "n_independent_runs": count,
        "mean": mean,
        "interval_available": True,
        "interval_low": ordered[0],
        "interval_high": ordered[-1],
        "interval_method": "observed_min_max",
        "interval_note": "Наблюдаемый минимум и максимум независимых запусков, не доверительный интервал популяции.",
    }


def complexity_axis_grid(
    reference: Mapping[str, Any],
    depths: Sequence[int],
    border_counts: Sequence[int],
    ensemble_sizes: Sequence[int],
) -> list[dict[str, Any]]:
    """Одна ось за раз. Декартово произведение depth × border × iterations не строится."""
    required = ("depth", "border_count", "iterations", "learning_rate", "l2_leaf_reg", "random_strength")
    missing = [key for key in required if key not in reference]
    if missing:
        raise ValueError(f"reference is missing {missing}")
    base = {key: reference[key] for key in required}
    specs = [{**base, "axis": "reference"}]
    for depth in depths:
        if int(depth) == int(base["depth"]):
            continue
        specs.append({**base, "axis": "depth", "depth": int(depth)})
    for border_count in border_counts:
        if int(border_count) == int(base["border_count"]):
            continue
        specs.append({**base, "axis": "border_count", "border_count": int(border_count)})
    for iterations in ensemble_sizes:
        if int(iterations) == int(base["iterations"]):
            continue
        specs.append({**base, "axis": "ensemble_size", "iterations": int(iterations)})
    return specs


def h1_verdict(reference_replication: Mapping[str, Any], zero_replication: Mapping[str, Any]) -> str:
    """H1: при random_strength=0 позднее ускорение пропадает.

    supports — только если на опорном random_strength ускорение воспроизвелось,
    а при нуле нет. Если опорного ускорения нет, гипотеза не проверена.
    """
    reference = reference_replication.get("inference")
    zero = zero_replication.get("inference")
    if reference == "reproduced" and zero == "not_reproduced":
        return "supports"
    if reference == "reproduced" and zero == "reproduced":
        return "refutes"
    return "inconclusive"


def h4_interior_peak(errors: Sequence[float], margin: float = PEAK_MARGIN) -> bool:
    """Грубый пик на трёх точках. Это не доказательство double descent."""
    if len(errors) != 3:
        raise ValueError("the pilot peak check expects exactly three grid points")
    left, middle, right = (float(value) for value in errors)
    return middle > left + margin and middle > right + margin


def h4_verdict(peaks: Sequence[bool]) -> str:
    if len(peaks) < 3:
        return "not_available"
    if all(peaks):
        return "coarse_peak_not_double_descent"
    if not any(peaks):
        return "not_observed"
    return "unstable"


def normalized_time_closer(iterations: Sequence[float], learning_rates: Sequence[float]) -> bool | None:
    """Описательное сравнение разброса argmin: итерации против итерации × learning rate."""
    if len(iterations) != len(learning_rates) or len(iterations) < 2:
        return None
    raw = [float(value) for value in iterations]
    scaled = [iteration * float(rate) for iteration, rate in zip(raw, learning_rates)]
    return (max(scaled) - min(scaled)) < (max(raw) - min(raw))
