"""Robust aggregates for literal skeleton-ring crossing profiles."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from ._literal_crossings import LiteralCrossingRingProfile


@dataclass(frozen=True)
class LiteralCrossingZoneMetrics:
    """Primary and diagnostic summaries for one radial zone.

    Angular values remain in radians at the SDK boundary. Rates use radians
    per pixel and the rate gradient uses radians per pixel squared.
    """

    sustained_peak: float
    net_rotation: float
    rotation_rate: float
    consistency: float
    raw_peak: float
    percentile_90: float
    percentile_95: float
    median_magnitude: float
    absolute_area: float
    total_variation: float
    rate_gradient: float
    ring_support: float
    run_span_support: float
    median_resultant: float


def _missing_metrics(
    *,
    ring_support: float = math.nan,
    run_span_support: float = math.nan,
    median_resultant: float = math.nan,
) -> LiteralCrossingZoneMetrics:
    """Return an all-missing metric record with optional support values."""
    return LiteralCrossingZoneMetrics(
        sustained_peak=math.nan,
        net_rotation=math.nan,
        rotation_rate=math.nan,
        consistency=math.nan,
        raw_peak=math.nan,
        percentile_90=math.nan,
        percentile_95=math.nan,
        median_magnitude=math.nan,
        absolute_area=math.nan,
        total_variation=math.nan,
        rate_gradient=math.nan,
        ring_support=ring_support,
        run_span_support=run_span_support,
        median_resultant=median_resultant,
    )


def _validate_aggregate_settings(
    peak_window_rings: int,
    minimum_run_rings: int,
) -> None:
    """Validate public aggregate settings."""
    if (
        isinstance(peak_window_rings, (bool, np.bool_))
        or not isinstance(peak_window_rings, (int, np.integer))
        or peak_window_rings < 3
        or peak_window_rings % 2 == 0
    ):
        raise ValueError("peak_window_rings must be an odd integer >= 3")
    if (
        isinstance(minimum_run_rings, (bool, np.bool_))
        or not isinstance(minimum_run_rings, (int, np.integer))
        or minimum_run_rings < 3
    ):
        raise ValueError("minimum_run_rings must be an integer >= 3")


def _run_indices(
    profile: LiteralCrossingRingProfile,
    selected: NDArray[np.bool_],
) -> list[NDArray[np.int64]]:
    """Return selected indices grouped by the profile's contiguous run ID."""
    runs: list[NDArray[np.int64]] = []
    for run_id in np.unique(profile.run_id[selected & profile.supported]):
        if run_id < 0:
            continue
        indices = np.flatnonzero(selected & (profile.run_id == run_id))
        if indices.size:
            runs.append(indices)
    return runs


def _median_pairwise_slope(radii: np.ndarray, values: np.ndarray) -> float:
    """Return the median slope across all ordered radius pairs."""
    left, right = np.triu_indices(values.size, k=1)
    distances = radii[right] - radii[left]
    slopes = (values[right] - values[left]) / distances
    return float(np.median(slopes)) if slopes.size else math.nan


def _absolute_kendall_tau_b(values: np.ndarray) -> float:
    """Return absolute Kendall tau-b against strictly increasing radius."""
    concordant = 0
    discordant = 0
    ties = 0
    for left in range(values.size - 1):
        differences = values[left + 1 :] - values[left]
        concordant += int(np.count_nonzero(differences > 0.0))
        discordant += int(np.count_nonzero(differences < 0.0))
        ties += int(np.count_nonzero(differences == 0.0))
    comparable = concordant + discordant
    denominator = math.sqrt((comparable + ties) * comparable)
    if denominator == 0.0:
        return 0.0
    return abs((concordant - discordant) / denominator)


def _sustained_peak(
    values: np.ndarray,
    runs: list[NDArray[np.int64]],
    window: int,
) -> float:
    """Return the greatest within-run rolling median magnitude."""
    peaks: list[float] = []
    for indices in runs:
        if indices.size < window:
            continue
        run_values = np.abs(values[indices])
        peaks.extend(
            float(np.median(run_values[start : start + window]))
            for start in range(indices.size - window + 1)
        )
    return max(peaks) if peaks else math.nan


def _span_normalized_absolute_area(
    radii: np.ndarray,
    values: np.ndarray,
    runs: list[NDArray[np.int64]],
) -> float:
    """Return trapezoidal absolute area divided by supported radial span."""
    area = 0.0
    span = 0.0
    for indices in runs:
        if indices.size < 2:
            continue
        run_radii = radii[indices]
        run_values = np.abs(values[indices])
        widths = np.diff(run_radii)
        area += float(
            np.sum(0.5 * (run_values[:-1] + run_values[1:]) * widths)
        )
        span += float(np.sum(widths))
    return area / span if span > 0.0 else math.nan


def _total_variation(
    values: np.ndarray,
    runs: list[NDArray[np.int64]],
) -> float:
    """Return summed absolute adjacent change without bridging run gaps."""
    changes = [
        np.abs(np.diff(values[indices]))
        for indices in runs
        if indices.size >= 2
    ]
    if not changes:
        return math.nan
    return float(sum(float(np.sum(change)) for change in changes))


def aggregate_literal_crossing_zone(
    profile: LiteralCrossingRingProfile,
    lower_radius: float,
    upper_radius: float,
    *,
    peak_window_rings: int = 3,
    minimum_run_rings: int = 6,
) -> LiteralCrossingZoneMetrics:
    """Aggregate one literal crossing profile over a radial zone.

    Args:
        profile: Equal-crossing cumulative orientation profile in radians.
        lower_radius: Inclusive radial bound in pixels.
        upper_radius: Exclusive radial bound in pixels.
        peak_window_rings: Odd rolling-median window for sustained peak.
        minimum_run_rings: Minimum dominant-run length for net rotation,
            robust rate, and trend consistency.

    Returns:
        Primary and diagnostic zone summaries. Angular values remain in
        radians. Support values and consistency are dimensionless. Invalid or
        collapsed radial bounds return an all-missing record.

    Raises:
        ValueError: If aggregate settings are invalid.
    """
    _validate_aggregate_settings(peak_window_rings, minimum_run_rings)
    if (
        not np.isfinite(lower_radius)
        or not np.isfinite(upper_radius)
        or upper_radius <= lower_radius
    ):
        return _missing_metrics()

    selected = (profile.radii >= lower_radius) & (profile.radii < upper_radius)
    candidate_count = int(np.count_nonzero(selected))
    if candidate_count == 0:
        return _missing_metrics(ring_support=0.0, run_span_support=0.0)

    supported = selected & profile.supported
    support_count = int(np.count_nonzero(supported))
    ring_support = support_count / candidate_count
    eligible_resultants = profile.resultant[
        selected & np.isfinite(profile.resultant)
    ]
    median_resultant = (
        float(np.median(eligible_resultants))
        if eligible_resultants.size
        else math.nan
    )
    if support_count == 0:
        return _missing_metrics(
            ring_support=ring_support,
            run_span_support=0.0,
            median_resultant=median_resultant,
        )

    values = profile.contiguous_change
    supported_values = values[supported]
    runs = _run_indices(profile, selected)
    sustained_peak = _sustained_peak(values, runs, peak_window_rings)
    raw_peak = float(np.max(np.abs(supported_values)))
    percentile_90 = float(np.percentile(np.abs(supported_values), 90.0))
    percentile_95 = float(np.percentile(np.abs(supported_values), 95.0))
    median_magnitude = float(np.median(np.abs(supported_values)))
    absolute_area = _span_normalized_absolute_area(profile.radii, values, runs)
    total_variation = _total_variation(values, runs)

    dominant = max(
        runs,
        key=lambda indices: (
            float(profile.radii[indices[-1]] - profile.radii[indices[0]]),
            int(indices.size),
            -float(profile.radii[indices[0]]),
        ),
    )
    dominant_span = float(
        profile.radii[dominant[-1]] - profile.radii[dominant[0]]
    )
    candidate_radii = profile.radii[selected]
    candidate_span = float(candidate_radii[-1] - candidate_radii[0])
    run_span_support = (
        dominant_span / candidate_span if candidate_span > 0.0 else 0.0
    )

    net_rotation = math.nan
    rotation_rate = math.nan
    consistency = math.nan
    rate_gradient = math.nan
    if dominant.size >= minimum_run_rings:
        run_radii = profile.radii[dominant]
        run_values = values[dominant]
        endpoint_width = max(2, math.ceil(0.2 * dominant.size))
        net_rotation = float(
            np.median(run_values[-endpoint_width:])
            - np.median(run_values[:endpoint_width])
        )
        rotation_rate = _median_pairwise_slope(run_radii, run_values)
        consistency = _absolute_kendall_tau_b(run_values)
        if dominant.size >= 8:
            midpoint = dominant.size // 2
            inner_radii = run_radii[:midpoint]
            outer_radii = run_radii[midpoint:]
            if inner_radii.size >= 4 and outer_radii.size >= 4:
                inner_rate = _median_pairwise_slope(
                    inner_radii, run_values[:midpoint]
                )
                outer_rate = _median_pairwise_slope(
                    outer_radii, run_values[midpoint:]
                )
                radial_separation = float(
                    np.median(outer_radii) - np.median(inner_radii)
                )
                if radial_separation > 0.0:
                    rate_gradient = (
                        outer_rate - inner_rate
                    ) / radial_separation

    return LiteralCrossingZoneMetrics(
        sustained_peak=sustained_peak,
        net_rotation=net_rotation,
        rotation_rate=rotation_rate,
        consistency=consistency,
        raw_peak=raw_peak,
        percentile_90=percentile_90,
        percentile_95=percentile_95,
        median_magnitude=median_magnitude,
        absolute_area=absolute_area,
        total_variation=total_variation,
        rate_gradient=rate_gradient,
        ring_support=ring_support,
        run_span_support=run_span_support,
        median_resultant=median_resultant,
    )
