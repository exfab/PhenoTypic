"""Validate the numeric invariants of Method B radial change points.

This script intentionally does not import ``phenotypic``. It independently
re-derives the load-bearing Method B claims in the accompanying design using
only NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ChangePoints:
    """Two selected ring indexes and their summed within-segment error."""

    first: int
    second: int
    cost: float


def bridge_short_gaps(support: BoolArray, maximum_gap: int) -> BoolArray:
    """Bridge only bounded unsupported runs no longer than ``maximum_gap``.

    Args:
        support: One-dimensional Boolean support profile.
        maximum_gap: Largest bounded False run to bridge.

    Returns:
        A copied Boolean profile with eligible interior gaps filled.
    """
    bridged = np.asarray(support, dtype=bool).copy()
    if maximum_gap <= 0:
        return bridged
    padded = np.concatenate(([True], bridged, [True]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == -1)
    stops = np.flatnonzero(changes == 1)
    for start, stop in zip(starts, stops, strict=True):
        if start > 0 and stop < bridged.size and stop - start <= maximum_gap:
            bridged[start:stop] = True
    return bridged


def robust_standardize(matrix: FloatArray) -> FloatArray:
    """Median-impute and robust-standardize each continuous feature.

    Args:
        matrix: Two-dimensional rings-by-features matrix.

    Returns:
        Standardized copy with no non-finite values.
    """
    standardized = np.asarray(matrix, dtype=float).copy()
    for column in range(standardized.shape[1]):
        values = standardized[:, column]
        finite = np.isfinite(values)
        fill = float(np.median(values[finite])) if finite.any() else 0.0
        values[~finite] = fill
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = max(1.4826 * mad, float(np.std(values)), np.finfo(float).eps)
        standardized[:, column] = (values - median) / scale
    return standardized


def prefix_statistics(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return zero-prefixed sums and sums of squares."""
    zeros = np.zeros((1, matrix.shape[1]), dtype=float)
    return (
        np.vstack((zeros, np.cumsum(matrix, axis=0))),
        np.vstack((zeros, np.cumsum(np.square(matrix), axis=0))),
    )


def segment_sse(
    prefix: FloatArray,
    prefix_squared: FloatArray,
    start: int,
    stop: int,
) -> float:
    """Return multivariate SSE around the segment feature means."""
    count = stop - start
    if count <= 0:
        return np.inf
    total = prefix[stop] - prefix[start]
    total_squared = prefix_squared[stop] - prefix_squared[start]
    return float(np.sum(total_squared - np.square(total) / count))


def exact_two_change_points(
    continuous_features: FloatArray,
    support: BoolArray,
    *,
    minimum_segment: int,
    support_weight: float,
    outer_support_margin: float,
) -> ChangePoints | None:
    """Run the exact evaluated Method B two-change-point search.

    Args:
        continuous_features: Rings-by-continuous-features matrix.
        support: Boolean support indicator for every ring.
        minimum_segment: Minimum number of rings in each of three segments.
        support_weight: Multiplier for the appended support feature.
        outer_support_margin: Required outward support-fraction increase at the
            first boundary.

    Returns:
        Best change points, or ``None`` when no first boundary passes the
        support constraint.
    """
    normalized = robust_standardize(continuous_features)
    matrix = np.column_stack((normalized, support.astype(float) * support_weight))
    prefix, prefix_squared = prefix_statistics(matrix)
    n_rings = matrix.shape[0]
    best: tuple[float, int, int] | None = None
    for first in range(
        minimum_segment,
        n_rings - 2 * minimum_segment + 1,
    ):
        support_gain = float(support[first:].mean()) - float(
            support[:first].mean()
        )
        if support_gain < outer_support_margin:
            continue
        for second in range(
            first + minimum_segment,
            n_rings - minimum_segment + 1,
        ):
            cost = (
                segment_sse(prefix, prefix_squared, 0, first)
                + segment_sse(prefix, prefix_squared, first, second)
                + segment_sse(prefix, prefix_squared, second, n_rings)
            )
            candidate = (cost, first, second)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    return ChangePoints(first=best[1], second=best[2], cost=best[0])


def ring_boundary_radius(ring_centers: FloatArray, index: int, width: float) -> float:
    """Map a selected ring index to that ring's inner-edge radius."""
    return float(ring_centers[index] - width / 2.0)


def validate_known_three_segment_signal() -> None:
    """Recover two exact breaks in a deterministic multivariate profile."""
    first = np.zeros((4, 3), dtype=float)
    middle = np.full((4, 3), (2.0, -1.0, 0.5), dtype=float)
    outer = np.full((4, 3), (5.0, 3.0, -2.0), dtype=float)
    features = np.vstack((first, middle, outer))
    support = np.array(
        [False, False, False, False, True, True, True, True, True, True, True, True]
    )
    result = exact_two_change_points(
        features,
        support,
        minimum_segment=4,
        support_weight=4.0,
        outer_support_margin=0.0,
    )
    assert result is not None
    assert (result.first, result.second) == (4, 8)
    assert np.isclose(result.cost, 0.0, atol=1e-12)

    width = 8.0
    ring_centers = width / 2.0 + np.arange(12) * width
    assert ring_boundary_radius(ring_centers, result.first, width) == 32.0
    assert ring_boundary_radius(ring_centers, result.second, width) == 64.0


def validate_deterministic_ties() -> None:
    """Confirm lexicographic ties choose the earliest feasible boundaries."""
    features = np.zeros((9, 2), dtype=float)
    support = np.zeros(9, dtype=bool)
    result = exact_two_change_points(
        features,
        support,
        minimum_segment=2,
        support_weight=0.5,
        outer_support_margin=0.0,
    )
    assert result is not None
    assert (result.first, result.second) == (2, 4)
    assert result.cost == 0.0


def validate_support_constraint() -> None:
    """Reject every split when outward support decreases below the margin."""
    features = np.arange(18.0).reshape(9, 2)
    support = np.array([True, True, True, True, False, False, False, False, False])
    result = exact_two_change_points(
        features,
        support,
        minimum_segment=2,
        support_weight=4.0,
        outer_support_margin=0.01,
    )
    assert result is None


def validate_gap_bridging() -> None:
    """Bridge short interior gaps but never leading or trailing gaps."""
    profile = np.array(
        [False, True, True, False, True, True, False, False], dtype=bool
    )
    expected = np.array(
        [False, True, True, True, True, True, False, False], dtype=bool
    )
    assert np.array_equal(bridge_short_gaps(profile, 1), expected)
    assert np.array_equal(bridge_short_gaps(profile, 0), profile)


def validate_imputation_and_scaling() -> None:
    """Check finite output, median imputation, and offset/scale invariance."""
    features = np.array(
        [
            [1.0, np.nan, np.nan],
            [2.0, 10.0, np.nan],
            [3.0, 20.0, np.nan],
            [4.0, 30.0, np.nan],
        ]
    )
    standardized = robust_standardize(features)
    assert np.isfinite(standardized).all()
    assert np.all(standardized[:, 2] == 0.0)
    transformed = features.copy()
    transformed[:, 0] = transformed[:, 0] * 7.0 + 13.0
    assert np.allclose(
        standardized[:, 0],
        robust_standardize(transformed)[:, 0],
    )


def validate_method_b_change_point_claims() -> None:
    """Run every independent Method B numerical validation."""
    validate_known_three_segment_signal()
    validate_deterministic_ties()
    validate_support_constraint()
    validate_gap_bridging()
    validate_imputation_and_scaling()
    print("PASS: Method B change-point claims validated")


if __name__ == "__main__":
    validate_method_b_change_point_claims()
