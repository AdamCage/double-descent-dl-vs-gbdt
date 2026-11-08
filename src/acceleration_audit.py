"""Whole-trajectory calibration of a multiscale slope-increase scan.

This is a diagnostic adaptation of established change-point ideas, not a new
general change-point method. Synthetic-null calibration never supplies a
valid p-value for an arbitrary training trajectory.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import beta


def slope_scan(values: Sequence[float], windows: Sequence[int]) -> dict:
    """Maximize local descent-speed gain over the declared search in one curve.

Both OLS segments include the boundary. Gain times window is divided by the
curve's range, giving invariance to positive affine changes of loss units.
"""
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 5 or not np.isfinite(y).all():
        raise ValueError("A finite one-dimensional trajectory is required")
    if not windows or any(isinstance(w, bool) or not isinstance(w, int) or w < 2
                          or 2 * w >= len(y) for w in windows):
        raise ValueError("Each window must be integer >=2 and fit both sides")
    span = float(np.ptp(y))
    best = {"score": 0.0, "t_star": None, "window": None,
            "velocity_before": 0.0, "velocity_after": 0.0}
    if span == 0:
        return best
    for w in sorted(set(windows)):
        x = np.arange(w + 1, dtype=float) - w / 2
        slopes = np.correlate(y, x, mode="valid") / np.dot(x, x)
        before, after = slopes[:-w], slopes[w:]
        scores = (before - after) * w / span
        # An increase in loss followed by a smaller increase is not descent.
        scores = np.where(after < 0, scores, -np.inf)
        at = int(np.argmax(scores))
        score = float(scores[at])
        if score > best["score"]:
            best = {"score": score, "t_star": at + w, "window": w,
                    "velocity_before": float(-before[at]),
                    "velocity_after": float(-after[at])}
    return best


def rank_threshold(scores: Sequence[float], alpha: float) -> dict:
    """Finite calibration order statistic; reject strictly above threshold.

For m exchangeable calibration null scores and an independent test score,
P(test > score_(ceil((m+1)(1-alpha)))) <= alpha, marginal over calibration.
The guarantee is conditional on the specified null law, not on a fitted
threshold, and allows ties. The max over several such thresholds is conservative
for each included null law.
"""
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 0 < alpha < 1:
        raise ValueError("Invalid calibration scores or alpha")
    k = math.ceil((len(x) + 1) * (1 - alpha))
    if k > len(x):
        raise ValueError("Insufficient calibration count for finite threshold")
    return {"threshold": float(np.sort(x)[k - 1]), "order": k,
            "n": len(x), "alpha": alpha, "rejection": "strictly_greater"}


def binomial_interval(successes: int, count: int, alpha: float = .05) -> list[float]:
    """Exact pointwise Clopper-Pearson interval across independent curves."""
    if not 0 <= successes <= count or count < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid binomial count")
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, count - successes + 1))
    hi = 1.0 if successes == count else float(beta.ppf(1 - alpha / 2, successes + 1, count - successes))
    return [lo, hi]


def analytic_curve(family: str, length: int) -> tuple[np.ndarray, int | None]:
    u = np.linspace(0, 1, length)
    if family == "linear":
        return 1 - .5 * u, None
    if family == "exponential":
        return .2 + .8 * np.exp(-5 * u), None
    if family == "early_plateau":
        return 1 - .6 * np.minimum(u / .2, 1) - .02 * u, None
    if family == "late_hinge":
        at = int(.7 * (length - 1))
        return 1 - .05 * u - .3 * np.maximum(u - u[at], 0), at
    if family == "fast_plateau_tail":
        at = int(.7 * (length - 1))
        return 1 - .6 * np.minimum(u / .2, 1) - .02 * u - .6 * np.maximum(u - u[at], 0), at
    if family == "last_tenth":
        at = int(.9 * (length - 1))
        return 1 - .05 * u - 1.2 * np.maximum(u - u[at], 0), at
    raise ValueError(f"Unknown analytic family: {family}")


def noisy_curve(base: np.ndarray, sigma: float, rho: float, entropy: list[int]) -> np.ndarray:
    """Stationary Gaussian AR(1) level noise; timepoints are dependent."""
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    z = rng.normal(size=len(base))
    eps = np.empty(len(base))
    eps[0] = z[0]
    for i in range(1, len(base)):
        eps[i] = rho * eps[i - 1] + math.sqrt(1 - rho * rho) * z[i]
    return base + sigma * eps
