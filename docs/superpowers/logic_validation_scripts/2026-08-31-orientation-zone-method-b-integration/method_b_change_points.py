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


@dataclass(frozen=True)
class OneChangePoint:
    """One selected ring index and its summed within-segment error."""

    boundary: int
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


def full_object_scale_limits(
    signal: FloatArray,
    target_mask: BoolArray,
) -> tuple[float, float]:
    """Return P2/P98 from finite target-object pixels at full extent."""
    source = np.asarray(signal, dtype=float)
    mask = np.asarray(target_mask, dtype=bool)
    if source.shape != mask.shape:
        raise ValueError("signal and target_mask must have matching shapes")
    population = source[mask & np.isfinite(source)]
    if population.size == 0:
        return 0.0, 0.0
    low, high = np.percentile(population, [2.0, 98.0], method="linear")
    return float(low), float(high)


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


def candidate_resolved_segments_supported(
    support: BoolArray,
    first: int,
    second: int,
) -> bool:
    """Return whether fixed middle and outer segments both contain support."""
    profile = np.asarray(support, dtype=bool)
    return bool(profile[first:second].any() and profile[second:].any())


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
    matrix = np.column_stack(
        (normalized, support.astype(float) * support_weight)
    )
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
            if not candidate_resolved_segments_supported(
                support, first, second
            ):
                continue
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


def ring_boundary_radius(
    ring_centers: FloatArray, index: int, width: float
) -> float:
    """Map a selected ring index to that ring's inner-edge radius."""
    return float(ring_centers[index] - width / 2.0)


def selected_outer_radius(distances: FloatArray, percentile: float) -> float:
    """Return the configured selected-mask radial percentile.

    The explicit 100th-percentile branch is part of the public contract: the
    default must be the exact furthest finite mask distance rather than a
    rounded ring boundary.

    Args:
        distances: Finite radial distances for selected-mask pixels.
        percentile: Requested percentile in ``(0, 100]``.

    Returns:
        Selected outer radius in pixels.
    """
    values = np.asarray(distances, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("distances must be a non-empty finite 1-D array")
    if isinstance(percentile, (bool, np.bool_)):
        raise ValueError("percentile must not be Boolean")
    if not np.isfinite(percentile) or not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be finite and in (0, 100]")
    if percentile == 100.0:
        return float(np.max(values))
    return float(np.percentile(values, percentile, method="linear"))


def center_origin_ring_centers(
    outer_radius: float, width: float
) -> FloatArray:
    """Return the complete center-origin ring grid through an outer radius."""
    if not np.isfinite(outer_radius) or outer_radius <= 0.0:
        raise ValueError("outer_radius must be finite and > 0")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and > 0")
    count = max(1, int(np.ceil(outer_radius / width)))
    return (np.arange(count, dtype=float) + 0.5) * width


def best_one_change(
    values: FloatArray,
    support: BoolArray,
    minimum_segment: int,
) -> OneChangePoint | None:
    """Return the earliest supported minimum-SSE one-change boundary."""
    signal = np.asarray(values, dtype=float).reshape(-1, 1)
    support_array = np.asarray(support, dtype=bool)
    if support_array.shape != (signal.shape[0],):
        raise ValueError("support must have one value per ring")
    if signal.shape[0] < 2 * minimum_segment:
        return None
    normalized = robust_standardize(signal)
    prefix, prefix_squared = prefix_statistics(normalized)
    candidates = []
    for boundary in range(
        minimum_segment,
        signal.shape[0] - minimum_segment + 1,
    ):
        if not support_array[boundary:].any():
            continue
        cost = segment_sse(prefix, prefix_squared, 0, boundary) + segment_sse(
            prefix,
            prefix_squared,
            boundary,
            signal.shape[0],
        )
        candidates.append((cost, boundary))
    if not candidates:
        return None
    cost, boundary = min(candidates)
    return OneChangePoint(boundary=boundary, cost=cost)


def validate_known_three_segment_signal() -> None:
    """Recover two exact breaks in a deterministic multivariate profile."""
    first = np.zeros((4, 3), dtype=float)
    middle = np.full((4, 3), (2.0, -1.0, 0.5), dtype=float)
    outer = np.full((4, 3), (5.0, 3.0, -2.0), dtype=float)
    features = np.vstack((first, middle, outer))
    support = np.array(
        [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
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
    support = np.ones(9, dtype=bool)
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
    support = np.array(
        [True, True, True, True, False, False, False, False, False]
    )
    result = exact_two_change_points(
        features,
        support,
        minimum_segment=2,
        support_weight=4.0,
        outer_support_margin=0.01,
    )
    assert result is None

    all_unresolved = exact_two_change_points(
        np.zeros((9, 2), dtype=float),
        np.zeros(9, dtype=bool),
        minimum_segment=2,
        support_weight=0.5,
        outer_support_margin=0.0,
    )
    assert all_unresolved is None

    outer_only = np.array(
        [False, False, False, False, False, False, True, True, True]
    )
    assert not candidate_resolved_segments_supported(outer_only, 2, 6)
    middle_only = np.array(
        [False, False, True, True, True, True, False, False, False]
    )
    assert not candidate_resolved_segments_supported(middle_only, 2, 6)
    both = np.array(
        [False, False, True, False, False, False, True, False, False]
    )
    assert candidate_resolved_segments_supported(both, 2, 6)


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

    signal = np.array(
        [
            [0.0, 10.0, 20.0, 30.0, 9999.0],
            [40.0, 50.0, np.nan, 60.0, -9999.0],
        ]
    )
    target_mask = np.array(
        [
            [True, True, True, True, False],
            [True, True, True, True, False],
        ]
    )
    limits = full_object_scale_limits(signal, target_mask)
    assert np.allclose(
        limits, np.percentile(np.arange(0.0, 70.0, 10.0), [2, 98])
    )
    p95_window = target_mask.copy()
    p95_window[1, 3] = False
    assert full_object_scale_limits(signal, target_mask) == limits
    assert full_object_scale_limits(signal, p95_window) != limits


def validate_prefix_sse_against_direct_computation() -> None:
    """Match prefix-sum SSE to direct centering for every valid segment."""
    generator = np.random.default_rng(20260901)
    matrix = generator.normal(size=(13, 5)) + np.arange(5, dtype=float)
    prefix, prefix_squared = prefix_statistics(matrix)
    for start in range(matrix.shape[0]):
        for stop in range(start + 1, matrix.shape[0] + 1):
            segment = matrix[start:stop]
            direct = float(np.square(segment - segment.mean(axis=0)).sum())
            assert np.isclose(
                segment_sse(prefix, prefix_squared, start, stop),
                direct,
                rtol=1e-12,
                atol=1e-12,
            )


def selected_mask_selector(
    distances: FloatArray,
    outer_radius: float,
) -> BoolArray:
    """Select distances through an inclusive floating-point outer radius."""
    values = np.asarray(distances, dtype=float)
    return values < np.nextafter(float(outer_radius), np.inf)


def clipped_annulus_selector(
    distances: FloatArray,
    ring_center: float,
    ring_width: float,
    outer_radius: float,
) -> BoolArray:
    """Select one annulus clipped through its inclusive outer radius."""
    values = np.asarray(distances, dtype=float)
    annulus = np.abs(values - ring_center) <= ring_width / 2.0
    return annulus & selected_mask_selector(values, outer_radius)


def validate_outer_percentile_and_ring_grid() -> None:
    """Check the full-extent default and non-rounded percentile behavior."""
    distances = np.arange(1.0, 101.0)
    assert selected_outer_radius(distances, 100.0) == 100.0
    assert np.isclose(selected_outer_radius(distances, 95.0), 95.05)
    for invalid in (True, np.bool_(False)):
        try:
            selected_outer_radius(distances, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Boolean percentile was accepted")

    full_selector = selected_mask_selector(
        distances,
        selected_outer_radius(distances, 100.0),
    )
    assert full_selector.all()
    assert full_selector[-1]
    percentile_selector = selected_mask_selector(
        distances,
        selected_outer_radius(distances, 95.0),
    )
    assert percentile_selector.sum() == 95
    assert np.isclose(percentile_selector.mean(), 0.95)

    centers = center_origin_ring_centers(95.05, 8.0)
    assert centers.size == 12
    assert centers[0] == 4.0
    assert centers[-1] == 92.0
    assert centers[-1] + 4.0 > 95.05

    terminal_inner = centers[-1] - 4.0
    assert terminal_inner < 95.05 < centers[-1] + 4.0
    ring_distances = np.array(
        [terminal_inner, 95.05, np.nextafter(95.05, np.inf)]
    )
    terminal_selector = clipped_annulus_selector(
        ring_distances,
        centers[-1],
        8.0,
        95.05,
    )
    assert np.array_equal(terminal_selector, [True, True, False])


def validate_collapsed_one_change_fallback() -> None:
    """Check the evaluated one-change fallback and insufficient-length case."""
    unresolved_evidence = np.array([1.0, 1.0, 0.0, 0.0])
    support = np.array([False, False, True, True])
    result = best_one_change(unresolved_evidence, support, minimum_segment=2)
    assert result is not None
    assert result.boundary == 2
    assert np.isclose(result.cost, 0.0, atol=1e-12)
    assert (
        best_one_change(
            unresolved_evidence,
            np.zeros(4, dtype=bool),
            minimum_segment=2,
        )
        is None
    )
    assert (
        best_one_change(
            unresolved_evidence[:3],
            support[:3],
            minimum_segment=2,
        )
        is None
    )


def validate_method_b_change_point_claims() -> None:
    """Run every independent Method B numerical validation."""
    validate_known_three_segment_signal()
    validate_deterministic_ties()
    validate_support_constraint()
    validate_gap_bridging()
    validate_imputation_and_scaling()
    validate_prefix_sse_against_direct_computation()
    validate_outer_percentile_and_ring_grid()
    validate_collapsed_one_change_fallback()
    print("PASS: Method B change-point claims validated")


if __name__ == "__main__":
    validate_method_b_change_point_claims()
