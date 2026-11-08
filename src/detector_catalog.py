"""Score a frozen catalog of synthetic loss trajectories.

The global score is the existing ``describe_change`` rule. The local score
compares mean velocities on a predeclared window and does not use the global
minimum-segment fraction. Neither score trains a model.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from src.dynamics import describe_change, segment_rss


def piecewise_linear(length: int, knots: Sequence[Sequence[float]]) -> list[float]:
    """Interpolate y at integer indices. Knots cover 0 and length-1."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise ValueError("length must be an integer >= 2")
    parsed: list[tuple[int, float]] = []
    for knot in knots:
        if len(knot) != 2:
            raise ValueError("each knot is an index and a value")
        index = knot[0]
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("knot index must be an integer")
        value = float(knot[1])
        if not math.isfinite(value):
            raise ValueError("knot value must be finite")
        parsed.append((index, value))
    if len(parsed) < 2 or parsed[0][0] != 0 or parsed[-1][0] != length - 1:
        raise ValueError("knots must start at 0 and end at length-1")
    for (left, _), (right, _) in zip(parsed, parsed[1:]):
        if right <= left:
            raise ValueError("knot indices must strictly increase")
    values = [0.0] * length
    for (left, left_value), (right, right_value) in zip(parsed, parsed[1:]):
        span = right - left
        for step in range(span + 1):
            values[left + step] = left_value + (right_value - left_value) * (step / span)
    return values


def exponential_decay(length: int, rate: float) -> list[float]:
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise ValueError("length must be an integer >= 2")
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive")
    return [math.exp(-rate * index) for index in range(length)]


def materialize(spec: dict[str, Any], length: int) -> list[float]:
    formula = spec.get("formula")
    if formula == "piecewise_linear":
        return piecewise_linear(length, spec["knots"])
    if formula == "exponential_decay":
        return exponential_decay(length, float(spec["rate"]))
    raise ValueError(f"Unknown formula: {formula}")


def local_velocity_change(
    loss: Sequence[float],
    *,
    window: int,
    rss_improvement_min: float,
) -> dict[str, Any]:
    """Pick the boundary that most increases mean velocity on a fixed window.

    ``candidate`` is true only when that local gain is positive. The piecewise
    RSS improvement is recorded and does not decide ``candidate``.
    ``rss_gated_candidate`` applies the same RSS floor as the global detector.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if not math.isfinite(rss_improvement_min) or rss_improvement_min < 0:
        raise ValueError("rss_improvement_min must be nonnegative")
    series = [float(value) for value in loss]
    count = len(series)
    empty: dict[str, Any] = {
        "candidate": False,
        "rss_gated_candidate": False,
        "t_star": None,
        "n": count,
        "window": window,
        "fraction": None,
        "velocity_before": None,
        "velocity_after": None,
        "velocity_gain": None,
        "rss_improvement": None,
        "reason": "series_too_short",
    }
    if count < 2 * window + 2:
        return empty
    velocity = [-(series[index + 1] - series[index]) for index in range(count - 1)]
    best_t: int | None = None
    best_gain: float | None = None
    best_before: float | None = None
    best_after: float | None = None
    last = count - 1 - window
    for boundary in range(window, last + 1):
        before = sum(velocity[boundary - window:boundary]) / window
        after = sum(velocity[boundary:boundary + window]) / window
        gain = after - before
        if best_gain is None or gain > best_gain:
            best_t = boundary
            best_gain = gain
            best_before = before
            best_after = after
    if best_t is None or best_gain is None or best_before is None or best_after is None:
        empty["reason"] = "no_eligible_boundary"
        return empty
    single = segment_rss(series, 0, count)
    piecewise = segment_rss(series, 0, best_t) + segment_rss(series, best_t, count)
    improvement = 0.0 if single <= 1e-15 else (single - piecewise) / single
    candidate = best_gain > 0.0
    return {
        "candidate": candidate,
        "rss_gated_candidate": candidate and improvement >= rss_improvement_min,
        "t_star": int(best_t),
        "n": count,
        "window": window,
        "fraction": best_t / count,
        "velocity_before": best_before,
        "velocity_after": best_after,
        "velocity_gain": best_gain,
        "rss_improvement": improvement,
        "reason": "local_velocity_increases" if candidate else "local_velocity_did_not_increase",
    }


def score_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Score every declared series once. This does not interpret a mechanism."""
    if catalog.get("schema_version") != 1:
        raise ValueError("catalog schema_version")
    length = catalog.get("length")
    window = catalog.get("local_window")
    min_fraction = catalog.get("global_min_fraction")
    rss_floor = catalog.get("rss_improvement_min")
    if isinstance(length, bool) or not isinstance(length, int) or length < 10:
        raise ValueError("catalog length")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("catalog window")
    if not isinstance(min_fraction, (int, float)) or isinstance(min_fraction, bool):
        raise ValueError("catalog min fraction")
    if not isinstance(rss_floor, (int, float)) or isinstance(rss_floor, bool):
        raise ValueError("catalog rss floor")
    series = catalog.get("series")
    if not isinstance(series, list) or not series:
        raise ValueError("catalog series")
    seen: set[str] = set()
    rows = []
    for spec in series:
        if not isinstance(spec, dict):
            raise ValueError("series entry")
        series_id = spec.get("id")
        role = spec.get("claim_role")
        if not isinstance(series_id, str) or not series_id or series_id in seen:
            raise ValueError("duplicate or empty series id")
        if not isinstance(role, str) or not role:
            raise ValueError("claim_role")
        seen.add(series_id)
        values = materialize(spec, length)
        rows.append({
            "id": series_id,
            "claim_role": role,
            "known_transition": spec.get("known_transition"),
            "n": len(values),
            "values": values,
            "global": describe_change(values, min_fraction=float(min_fraction), rss_improvement_min=float(rss_floor)),
            "local": local_velocity_change(values, window=window, rss_improvement_min=float(rss_floor)),
        })
    exclusions = catalog.get("exclusions")
    if not isinstance(exclusions, list) or not exclusions or any(not isinstance(item, str) or not item for item in exclusions):
        raise ValueError("exclusions")
    return {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "catalog_id": catalog.get("catalog_id"),
        "n_series": len(rows),
        "models_fit": 0,
        "download_bytes": 0,
        "http_requests": 0,
        "mechanism_claim": False,
        "local_window": window,
        "global_min_fraction": min_fraction,
        "rss_improvement_min": rss_floor,
        "exclusions": list(exclusions),
        "series": rows,
    }
