"""Описательная точка изменения скорости validation LogLoss.

Соседние итерации бустинга не считаются независимыми наблюдениями.
Вывод о воспроизводимости делается только по независимым запускам.
"""

from __future__ import annotations

from typing import Any, Sequence


MIN_SEGMENT_FRACTION = 0.2
RSS_IMPROVEMENT_MIN = 0.05
REPLICATE_TOLERANCE = 0.15
MIN_REPLICATES = 3


def _sum_squares_until(k: int) -> float:
    """sum_{i=0}^{k-1} i^2."""
    if k <= 1:
        return 0.0
    last = k - 1
    return last * k * (2 * k - 1) / 6.0


def segment_rss(values: Sequence[float], start: int, end: int) -> float:
    """RSS линейной регрессии values[start:end] на индекс точки."""
    if end - start < 2:
        return 0.0
    count = end - start
    sum_y = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    for index in range(start, end):
        value = float(values[index])
        sum_y += value
        sum_y2 += value * value
        sum_xy += index * value
    sum_x = (start + end - 1) * count / 2.0
    sum_x2 = _sum_squares_until(end) - _sum_squares_until(start)
    denominator = count * sum_x2 - sum_x * sum_x
    if denominator <= 0.0:
        mean = sum_y / count
        return max(0.0, sum_y2 - 2.0 * mean * sum_y + count * mean * mean)
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    rss = (
        count * intercept * intercept
        + slope * slope * sum_x2
        + sum_y2
        + 2.0 * intercept * slope * sum_x
        - 2.0 * intercept * sum_y
        - 2.0 * slope * sum_xy
    )
    return max(0.0, float(rss))


def describe_change(
    loss: Sequence[float],
    *,
    min_fraction: float = MIN_SEGMENT_FRACTION,
    rss_improvement_min: float = RSS_IMPROVEMENT_MIN,
) -> dict[str, Any]:
    """Ищет t*, после которого средняя скорость снижения L(t) устойчиво выше.

    Скорость v(t) = -dL/dt оценивается разностью соседних точек одной кривой.
    Кандидат требует и прироста скорости, и лучшего кусочно-линейного RSS.
    Один кандидат не является проверкой значимости.
    """
    series = [float(value) for value in loss]
    count = len(series)
    empty = {
        "candidate": False,
        "t_star": None,
        "n": count,
        "fraction": None,
        "velocity_before": None,
        "velocity_after": None,
        "velocity_gain": None,
        "rss_improvement": None,
        "reason": "series_too_short",
    }
    if count < 10:
        return empty
    min_segment = max(5, int(count * min_fraction))
    velocity = [-(series[index + 1] - series[index]) for index in range(count - 1)]
    prefix = [0.0]
    for value in velocity:
        prefix.append(prefix[-1] + value)

    best_t: int | None = None
    best_gain: float | None = None
    best_before: float | None = None
    best_after: float | None = None
    for boundary in range(min_segment, count - min_segment):
        before_count = boundary
        after_count = (count - 1) - boundary
        if before_count < min_segment or after_count < min_segment:
            continue
        before = prefix[boundary] / before_count
        after = (prefix[count - 1] - prefix[boundary]) / after_count
        gain = after - before
        if best_gain is None or gain > best_gain:
            best_t = boundary
            best_gain = gain
            best_before = before
            best_after = after

    if best_t is None or best_gain is None:
        empty["reason"] = "no_eligible_boundary"
        return empty

    single = segment_rss(series, 0, count)
    piecewise = segment_rss(series, 0, best_t) + segment_rss(series, best_t, count)
    improvement = 0.0 if single <= 1e-15 else (single - piecewise) / single
    candidate = best_gain > 0.0 and improvement >= rss_improvement_min
    if candidate:
        reason = "velocity_increases_and_piecewise_fit_improves"
    elif best_gain <= 0.0:
        reason = "velocity_did_not_increase"
    else:
        reason = "piecewise_fit_not_better"
    return {
        "candidate": candidate,
        "t_star": int(best_t),
        "n": count,
        "fraction": best_t / count,
        "velocity_before": best_before,
        "velocity_after": best_after,
        "velocity_gain": best_gain,
        "rss_improvement": improvement,
        "reason": reason,
    }


def replicate_change_points(
    runs: Sequence[dict[str, Any]],
    *,
    tolerance: float = REPLICATE_TOLERANCE,
    min_replicates: int = MIN_REPLICATES,
) -> dict[str, Any]:
    """Согласие t*/T между независимыми запусками. Итерации внутри кривой не являются n."""
    count = len(runs)
    base = {
        "n_independent_runs": count,
        "interval_available": False,
        "inference": "not_available",
        "median_fraction": None,
        "fractions": [],
    }
    if count < min_replicates:
        base["reason"] = "fewer than 3 independent runs"
        return base
    fractions = []
    for run in runs:
        if not run.get("candidate") or run.get("t_star") is None or not run.get("n"):
            base["inference"] = "not_reproduced"
            base["reason"] = "at least one run has no candidate"
            base["fractions"] = [
                None if not item.get("candidate") or not item.get("n") else item["t_star"] / item["n"]
                for item in runs
            ]
            return base
        fractions.append(float(run["t_star"]) / float(run["n"]))
    ordered = sorted(fractions)
    median = ordered[count // 2]
    spread = max(abs(value - median) for value in fractions)
    reproduced = spread <= tolerance
    return {
        "n_independent_runs": count,
        "interval_available": False,
        "inference": "reproduced" if reproduced else "not_reproduced",
        "median_fraction": median,
        "fractions": fractions,
        "max_abs_deviation": spread,
        "tolerance": tolerance,
        "reason": "fractions agree" if reproduced else "fractions disagree",
    }
