"""Minimum-norm and ridge fits through one truncated SVD.

The objective on centered responses is

    (1/n) * ||X_standardized @ w - y_centered||^2 + ridge_lambda * ||w||^2.

Lambda 0 is the same path with the penalty removed. Column centering makes
rank n unattainable, so the interpolation rank target is min(n - 1, p).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def rank_target(n_train: int, n_features: int) -> int:
    if n_train < 2 or n_features < 1:
        raise ValueError("rank target needs at least two rows and one column")
    return min(n_train - 1, n_features)


def _as_2d(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("features must be a matrix with at least two rows and one column")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain a non-finite value")
    return matrix


def standardize_train(features: np.ndarray) -> dict[str, Any]:
    matrix = _as_2d(features)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    dead = scale <= 0
    safe = np.where(dead, 1.0, scale)
    standardized = (matrix - mean) / safe
    standardized[:, dead] = 0.0
    return {
        "mean": mean,
        "scale": scale,
        "dead": dead,
        "standardized": standardized,
        "zero_variance_columns": int(dead.sum()),
    }


def apply_standardizer(features: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != fitted["mean"].shape[0]:
        raise ValueError("evaluation features do not match the training width")
    if not np.isfinite(matrix).all():
        raise ValueError("evaluation features contain a non-finite value")
    safe = np.where(fitted["dead"], 1.0, fitted["scale"])
    standardized = (matrix - fitted["mean"]) / safe
    standardized[:, fitted["dead"]] = 0.0
    return standardized


def fit_ridge_path(
    features: np.ndarray,
    response: np.ndarray,
    lambdas: list[float],
    rcond_relative: float,
    *,
    rescale_columns: bool = True,
) -> dict[str, Any]:
    """Fit every lambda from one SVD.

    ``rescale_columns=False`` keeps a spectrum that was already applied.
    Dividing those columns by their empirical standard deviation would cancel it.
    """
    if not math.isfinite(rcond_relative) or rcond_relative <= 0:
        raise ValueError("rcond_relative must be positive")
    response = np.asarray(response, dtype=np.float64)
    matrix = _as_2d(features)
    if rescale_columns:
        fitted = standardize_train(matrix)
    else:
        mean = matrix.mean(axis=0)
        dead = np.all(matrix == 0.0, axis=0)
        standardized = matrix - mean
        standardized[:, dead] = 0.0
        fitted = {
            "mean": mean,
            "scale": np.ones(matrix.shape[1], dtype=np.float64),
            "dead": dead,
            "standardized": standardized,
            "zero_variance_columns": int(dead.sum()),
        }
    fitted["rescaled_columns"] = bool(rescale_columns)
    standardized = fitted["standardized"]
    n_train, n_features = standardized.shape
    if response.shape != (n_train,) or not np.isfinite(response).all():
        raise ValueError("response must be a finite vector with one entry per row")
    if not lambdas or any((not math.isfinite(value)) or value < 0 for value in lambdas):
        raise ValueError("lambdas must be nonnegative")
    intercept = float(response.mean())
    centered = response - intercept
    left, singular, right = np.linalg.svd(standardized, full_matrices=False)
    largest = float(singular[0]) if singular.size else 0.0
    cutoff = float(rcond_relative) * largest
    retained = singular > cutoff
    rank = int(retained.sum())
    projected = left.T @ centered
    target = rank_target(n_train, n_features)
    kept = singular[retained]
    if kept.size:
        condition = float(kept[0] / kept[-1])
        gram_condition = float(condition ** 2)
        smallest_kept = float(kept[-1])
    else:
        condition = None
        gram_condition = None
        smallest_kept = None
    solutions = []
    for ridge_lambda in lambdas:
        filter_ = np.zeros_like(singular)
        denominator = singular[retained] ** 2 + n_train * float(ridge_lambda)
        filter_[retained] = singular[retained] / denominator
        slope = right.T @ (filter_ * projected)
        training_prediction = standardized @ slope + intercept
        residual = training_prediction - response
        residual_l2 = float(np.linalg.norm(residual))
        response_l2 = float(np.linalg.norm(response))
        relative = residual_l2 / max(response_l2, 1e-30)
        solutions.append({
            "ridge_lambda": float(ridge_lambda),
            "slope": slope,
            "intercept": intercept,
            "residual_l2": residual_l2,
            "residual_relative": relative,
            "coefficient_l2": float(np.linalg.norm(slope)),
            "train_mse": float(np.mean(residual ** 2)),
            "training_prediction": training_prediction,
        })
    return {
        "fit": fitted,
        "n_train": n_train,
        "n_features": n_features,
        "rank": rank,
        "rank_target": target,
        "cutoff": cutoff,
        "singular_value_max": largest,
        "singular_value_min_kept": smallest_kept,
        "condition_number": condition,
        "gram_condition_number": gram_condition,
        "solutions": solutions,
    }


def predict(features: np.ndarray, result: dict[str, Any], solution: dict[str, Any]) -> np.ndarray:
    standardized = apply_standardizer(features, result["fit"])
    return standardized @ solution["slope"] + solution["intercept"]


def interpolation_flag(solution: dict[str, Any], result: dict[str, Any], tolerance: float) -> bool | None:
    if solution["ridge_lambda"] != 0.0:
        return None
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("residual tolerance must be positive")
    return bool(
        result["rank"] == result["rank_target"]
        and solution["residual_relative"] <= tolerance
    )
