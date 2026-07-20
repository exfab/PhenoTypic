"""Validate robust aggregates proposed for literal ring-crossing profiles.

This script intentionally does not import ``phenotypic``. It re-derives the
load-bearing numerical claims in the accompanying design specification using
only NumPy.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def split_finite_runs(
    radii: FloatArray, values: FloatArray
) -> list[tuple[FloatArray, FloatArray]]:
    """Split a sampled profile into contiguous finite runs."""
    finite = np.isfinite(radii) & np.isfinite(values)
    runs: list[tuple[FloatArray, FloatArray]] = []
    start: int | None = None
    for index, accepted in enumerate(finite):
        if accepted and start is None:
            start = index
        if start is not None and (not accepted or index == len(finite) - 1):
            stop = index if not accepted else index + 1
            runs.append((radii[start:stop], values[start:stop]))
            start = None
    return runs


def select_dominant_run(
    radii: FloatArray, values: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Select the run with greatest span, count, then earliest start."""
    runs = split_finite_runs(radii, values)
    if not runs:
        return np.array([], dtype=float), np.array([], dtype=float)
    return max(
        runs,
        key=lambda run: (
            float(run[0][-1] - run[0][0]),
            len(run[0]),
            -float(run[0][0]),
        ),
    )


def sustained_peak(values: FloatArray, window: int = 3) -> float:
    """Return the maximum rolling median absolute rotation."""
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer of at least 3")
    if len(values) < window or not np.all(np.isfinite(values)):
        return math.nan
    medians = [
        float(np.median(np.abs(values[index : index + window])))
        for index in range(len(values) - window + 1)
    ]
    return max(medians)


def sustained_peak_across_runs(
    radii: FloatArray, values: FloatArray, window: int = 3
) -> float:
    """Return the greatest sustained peak without bridging run breaks."""
    peaks = [
        sustained_peak(run_values, window)
        for _, run_values in split_finite_runs(radii, values)
        if len(run_values) >= window
    ]
    return max(peaks) if peaks else math.nan


def robust_net(values: FloatArray) -> float:
    """Return outer endpoint median minus inner endpoint median."""
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return math.nan
    width = max(2, math.ceil(0.2 * len(values)))
    return float(np.median(values[-width:]) - np.median(values[:width]))


def median_pairwise_slope(radii: FloatArray, values: FloatArray) -> float:
    """Return the median slope across all ordered radius pairs."""
    if len(radii) < 2 or len(radii) != len(values):
        return math.nan
    slopes: list[float] = []
    for left in range(len(radii) - 1):
        for right in range(left + 1, len(radii)):
            distance = float(radii[right] - radii[left])
            if distance > 0:
                slopes.append(float((values[right] - values[left]) / distance))
    return float(np.median(slopes)) if slopes else math.nan


def absolute_kendall_consistency(
    radii: FloatArray, values: FloatArray
) -> float:
    """Return absolute Kendall tau-b with cumulative-value tie correction."""
    concordant = 0
    discordant = 0
    value_ties = 0
    for left in range(len(radii) - 1):
        for right in range(left + 1, len(radii)):
            value_delta = float(values[right] - values[left])
            radius_delta = float(radii[right] - radii[left])
            if radius_delta <= 0.0:
                raise ValueError("radii must be strictly increasing")
            if value_delta == 0.0:
                value_ties += 1
            elif value_delta * radius_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    denominator = math.sqrt((comparable + value_ties) * comparable)
    if denominator == 0.0:
        return 0.0
    return abs((concordant - discordant) / denominator)


def validate_linear_profiles() -> None:
    """Check rate, net, sign, scale, and consistency invariants."""
    radii = np.arange(8.0, 88.0, 8.0)
    values = 0.75 * radii - 11.0
    assert np.isclose(median_pairwise_slope(radii, values), 0.75)
    assert np.isclose(absolute_kendall_consistency(radii, values), 1.0)
    assert robust_net(values) > 0

    negative = -values
    assert np.isclose(median_pairwise_slope(radii, negative), -0.75)
    assert np.isclose(absolute_kendall_consistency(radii, negative), 1.0)
    assert np.isclose(robust_net(negative), -robust_net(values))
    assert np.isclose(sustained_peak(negative), sustained_peak(values))

    translated = values + 1234.5
    assert np.isclose(robust_net(translated), robust_net(values))
    assert np.isclose(median_pairwise_slope(radii, translated), 0.75)
    assert np.isclose(absolute_kendall_consistency(radii, translated), 1.0)

    assert np.isclose(median_pairwise_slope(2.0 * radii, values), 0.375)


def validate_outlier_resistance() -> None:
    """Check that one isolated spike cannot set the sustained peak."""
    baseline = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0])
    spiked = baseline.copy()
    spiked[3] = 500.0
    assert np.max(np.abs(spiked)) == 500.0
    assert sustained_peak(spiked) == sustained_peak(baseline)
    assert np.isclose(
        median_pairwise_slope(np.arange(7.0), spiked),
        median_pairwise_slope(np.arange(7.0), baseline),
    )

    twelve_rings = np.arange(12.0)
    twelve_rings[-2:] = [80.0, 100.0]
    p95 = float(np.percentile(twelve_rings, 95))
    assert 80.0 < p95 < 100.0
    assert np.isclose(p95, 89.0)


def validate_gap_and_run_rules() -> None:
    """Check that aggregation cannot bridge unsupported radial samples."""
    radii = np.arange(0.0, 80.0, 8.0)
    values = np.array(
        [0.0, 4.0, 8.0, np.nan, 0.0, 20.0, 40.0, 60.0, 80.0, np.nan]
    )
    runs = split_finite_runs(radii, values)
    assert [len(run[0]) for run in runs] == [3, 5]
    selected_radii, selected_values = select_dominant_run(radii, values)
    assert np.array_equal(selected_radii, radii[4:9])
    assert np.array_equal(selected_values, values[4:9])
    assert sustained_peak(selected_values) == 60.0

    localized_values = np.array(
        [0.0, 100.0, 101.0, np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, np.nan]
    )
    assert sustained_peak_across_runs(radii, localized_values) == 100.0
    _, dominant_values = select_dominant_run(radii, localized_values)
    assert sustained_peak(dominant_values) == 3.0

    tied_radii = np.array([0.0, 8.0, np.nan, 32.0, 40.0])
    tied_values = np.array([0.0, 1.0, np.nan, 0.0, 1.0])
    selected_radii, _ = select_dominant_run(tied_radii, tied_values)
    assert selected_radii[0] == 0.0


def validate_support_edge_cases() -> None:
    """Check explicit missing and tied-profile behavior."""
    assert math.isnan(sustained_peak(np.array([1.0, 2.0])))
    assert math.isnan(median_pairwise_slope(np.array([1.0]), np.array([2.0])))
    radii = np.arange(5.0)
    values = np.full(5, 22.0)
    assert absolute_kendall_consistency(radii, values) == 0.0
    assert median_pairwise_slope(radii, values) == 0.0


def validate_aggregate_metric_claims() -> None:
    """Run every independent numerical validation."""
    validate_linear_profiles()
    validate_outlier_resistance()
    validate_gap_and_run_rules()
    validate_support_edge_cases()
    print("PASS: orientation-zone aggregate metric claims validated")


if __name__ == "__main__":
    validate_aggregate_metric_claims()
