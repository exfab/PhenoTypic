"""Independently verify radial-relative orientation metric invariants.

This script deliberately does not import ``phenotypic``. It re-derives the
load-bearing geometry using only NumPy so implementation and tests can be
checked against an independent calculation.
"""

from __future__ import annotations

import numpy as np


def axial_wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap an axial angle to ``[-pi/2, pi/2)``."""
    return 0.5 * np.arctan2(np.sin(2.0 * angle), np.cos(2.0 * angle))


def signed_outward_turning(
    fiber_axis: np.ndarray,
    centre: tuple[float, float],
) -> np.ndarray:
    """Return the pi-safe signed radial derivative of axial tilt."""
    rows, cols = np.indices(fiber_axis.shape, dtype=float)
    delta_row = rows - centre[0]
    delta_col = cols - centre[1]
    distance = np.hypot(delta_row, delta_col)
    polar = np.arctan2(delta_row, delta_col)
    tilt = axial_wrap(fiber_axis - polar)
    cosine = np.cos(2.0 * tilt)
    sine = np.sin(2.0 * tilt)
    cosine_y, cosine_x = np.gradient(cosine)
    sine_y, sine_x = np.gradient(sine)
    radial_x = np.divide(
        delta_col,
        distance,
        out=np.zeros_like(distance),
        where=distance > 0,
    )
    radial_y = np.divide(
        delta_row,
        distance,
        out=np.zeros_like(distance),
        where=distance > 0,
    )
    cosine_r = cosine_x * radial_x + cosine_y * radial_y
    sine_r = sine_x * radial_x + sine_y * radial_y
    return 0.5 * (cosine * sine_r - sine * cosine_r)


def radial_metrics(
    fiber_axis: np.ndarray,
    selector: np.ndarray,
    centre: tuple[float, float],
    coherence: np.ndarray | None = None,
    n_sectors: int = 36,
) -> tuple[float, float, float]:
    """Calculate equal-sector absolute tilt and outward turning."""
    rows, cols = np.indices(fiber_axis.shape, dtype=float)
    delta_row = rows - centre[0]
    delta_col = cols - centre[1]
    distance = np.hypot(delta_row, delta_col)
    polar = np.arctan2(delta_row, delta_col)
    tilt = axial_wrap(fiber_axis - polar)
    cosine = np.cos(2.0 * tilt)
    sine = np.sin(2.0 * tilt)
    cosine_y, cosine_x = np.gradient(cosine)
    sine_y, sine_x = np.gradient(sine)
    radial_x = np.divide(
        delta_col,
        distance,
        out=np.zeros_like(distance),
        where=distance > 0,
    )
    radial_y = np.divide(
        delta_row,
        distance,
        out=np.zeros_like(distance),
        where=distance > 0,
    )
    turning = 0.5 * np.hypot(
        cosine_x * radial_x + cosine_y * radial_y,
        sine_x * radial_x + sine_y * radial_y,
    )
    if coherence is None:
        coherence = np.ones_like(fiber_axis)
    valid = selector & (distance > 0) & (coherence >= 0.15)
    sector_ids = (
        np.mod(polar[valid], 2.0 * np.pi) / (2.0 * np.pi) * n_sectors
    ).astype(int)
    sector_tilt = []
    sector_turning = []
    for sector in np.unique(sector_ids):
        chosen = sector_ids == sector
        if int(chosen.sum()) < 3:
            continue
        weights = coherence[valid][chosen]
        sector_tilt.append(
            float(
                np.sum(weights * np.abs(tilt[valid][chosen])) / weights.sum()
            )
        )
        sector_turning.append(
            float(np.sum(weights * turning[valid][chosen]) / weights.sum())
        )
    if not sector_tilt:
        return float("nan"), float("nan"), 0.0
    return (
        float(np.mean(sector_tilt)),
        float(np.mean(sector_turning)),
        len(sector_tilt) / n_sectors,
    )


def long_range_ring_rotation(
    signed_tilt: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
    distance: np.ndarray,
    polar: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    ring_width: float,
    radial_lag: float,
    n_sectors: int = 36,
) -> tuple[float, float, float]:
    """Independently derive equal-sector fixed-lag annular rotation."""
    n_rings = int(np.floor((outer_radius - inner_radius) / ring_width + 1e-9))
    centres = inner_radius + (np.arange(n_rings) + 0.5) * ring_width
    means = np.full((n_rings, n_sectors), np.nan)
    for ring_index, centre_radius in enumerate(centres):
        ring = (
            selector
            & (distance >= centre_radius - 0.5 * ring_width)
            & (distance < centre_radius + 0.5 * ring_width)
            & (coherence >= 0.15)
        )
        sector_id = np.minimum(
            (
                np.mod(polar[ring], 2.0 * np.pi) / (2.0 * np.pi) * n_sectors
            ).astype(int),
            n_sectors - 1,
        )
        for sector in np.unique(sector_id):
            chosen = sector_id == sector
            if int(chosen.sum()) < 3:
                continue
            weights = coherence[ring][chosen]
            angles = signed_tilt[ring][chosen]
            cosine = float(np.sum(weights * np.cos(2.0 * angles)))
            sine = float(np.sum(weights * np.sin(2.0 * angles)))
            resultant = np.hypot(cosine, sine) / float(weights.sum())
            if resultant < 0.15:
                continue
            means[ring_index, sector] = 0.5 * np.arctan2(sine, cosine)

    ring_step = int(round(radial_lag / ring_width))
    rotations = []
    for inner_index in range(max(0, n_rings - ring_step)):
        outer_index = inner_index + ring_step
        delta = axial_wrap(means[outer_index] - means[inner_index])
        rotations.append(delta)
    if not rotations:
        return float("nan"), float("nan"), 0.0
    rotation = np.vstack(rotations)
    finite = np.isfinite(rotation)
    support = float(finite.sum()) / float(rotation.size)
    if not finite.any():
        return float("nan"), float("nan"), support
    values = rotation[finite]
    return float(np.mean(np.abs(values))), float(np.mean(values)), support


def cumulative_axial_rotation(sector_tilt: np.ndarray) -> np.ndarray:
    """Independently unwrap adjacent axial changes within each sector."""
    cumulative = np.full_like(sector_tilt, np.nan, dtype=float)
    for sector in range(sector_tilt.shape[1]):
        supported = np.flatnonzero(np.isfinite(sector_tilt[:, sector]))
        if supported.size == 0:
            continue
        start = int(supported[0])
        cumulative[start, sector] = 0.0
        for ring in range(start + 1, sector_tilt.shape[0]):
            if not (
                np.isfinite(sector_tilt[ring - 1, sector])
                and np.isfinite(sector_tilt[ring, sector])
            ):
                break
            step = axial_wrap(
                sector_tilt[ring, sector] - sector_tilt[ring - 1, sector]
            )
            if np.isclose(abs(step), np.pi / 2.0, atol=1e-9, rtol=0.0):
                break
            cumulative[ring, sector] = cumulative[ring - 1, sector] + step
    return cumulative


def verify_radial_relative_metric_invariants() -> None:
    """Assert axis, branch-count, and outward-bend invariants."""
    size = 161
    centre = ((size - 1) / 2.0, (size - 1) / 2.0)
    rows, cols = np.indices((size, size), dtype=float)
    delta_row = rows - centre[0]
    delta_col = cols - centre[1]
    distance = np.hypot(delta_row, delta_col)
    polar = np.arctan2(delta_row, delta_col)
    annulus = (distance >= 15.0) & (distance < 70.0)
    tolerance = np.deg2rad(4.0)
    horizontal = annulus & (np.abs(axial_wrap(polar)) < tolerance)
    vertical = annulus & (np.abs(axial_wrap(polar - np.pi / 2.0)) < tolerance)

    straight_results = [
        radial_metrics(polar, selector, centre)
        for selector in (horizontal, vertical, annulus)
    ]
    for tilt, turning, support in straight_results:
        assert np.isclose(tilt, 0.0, atol=1e-12), (tilt, turning)
        assert np.isclose(turning, 0.0, atol=1e-12), (tilt, turning)
        assert support > 0.0

    oblique = polar + np.deg2rad(20.0)
    sparse_oblique = radial_metrics(oblique, horizontal, centre)
    dense_oblique = radial_metrics(oblique, annulus, centre)
    assert np.isclose(sparse_oblique[0], np.deg2rad(20.0), atol=1e-12)
    assert np.isclose(dense_oblique[0], np.deg2rad(20.0), atol=1e-12)
    assert np.isclose(sparse_oblique[1], 0.0, atol=1e-12)
    assert np.isclose(dense_oblique[1], 0.0, atol=1e-12)
    assert sparse_oblique[2] < dense_oblique[2]

    expected_rate = 0.004
    bent = polar + expected_rate * distance
    _, measured_rate, _ = radial_metrics(bent, annulus, centre)
    assert np.isclose(measured_rate, expected_rate, rtol=0.03), measured_rate

    for direction in (-1.0, 1.0):
        directional_bend = polar + direction * expected_rate * distance
        signed_rate = signed_outward_turning(directional_bend, centre)
        measured_signed_rate = float(np.mean(signed_rate[annulus]))
        assert np.isclose(
            measured_signed_rate,
            direction * expected_rate,
            rtol=0.03,
        ), measured_signed_rate

    # Crossing the axial +/-90-degree seam must not reverse or spike the sign.
    seam_crossing_bend = (
        polar + np.pi / 2.0 - 0.04 + expected_rate * (distance - 15.0)
    )
    seam_rate = signed_outward_turning(seam_crossing_bend, centre)
    assert np.isclose(
        float(np.mean(seam_rate[annulus])),
        expected_rate,
        rtol=0.03,
    )

    confidence = np.ones_like(polar)
    low_confidence_sector = annulus & (polar >= 0.0) & (polar < np.pi / 6.0)
    conflicting = polar.copy()
    conflicting[low_confidence_sector] += np.pi / 2.0
    confidence[low_confidence_sector] = 0.01
    filtered_tilt, _, filtered_support = radial_metrics(
        conflicting,
        annulus,
        centre,
        confidence,
    )
    assert np.isclose(filtered_tilt, 0.0, atol=1e-12), filtered_tilt
    assert 0.0 < filtered_support < 1.0

    # A broad turn that is only 0.002 rad/px locally accumulates to 0.032 rad
    # across a 16-pixel radial lag. Duplicating same-angle evidence across all
    # sectors leaves the long-range point estimate unchanged; support changes.
    long_rate = 0.002
    long_tilt = axial_wrap(long_rate * distance)
    long_annulus = (distance >= 16.0) & (distance < 72.0)
    sparse_arcs = long_annulus & (
        (np.mod(polar, 2.0 * np.pi) < np.deg2rad(28.0))
        | (np.mod(polar, 2.0 * np.pi) > np.deg2rad(332.0))
    )
    long_dense = long_range_ring_rotation(
        long_tilt,
        np.ones_like(distance),
        long_annulus,
        distance,
        polar,
        16.0,
        72.0,
        8.0,
        16.0,
    )
    long_sparse = long_range_ring_rotation(
        long_tilt,
        np.ones_like(distance),
        sparse_arcs,
        distance,
        polar,
        16.0,
        72.0,
        8.0,
        16.0,
    )
    expected_long_rotation = long_rate * 16.0
    assert np.isclose(long_dense[0], expected_long_rotation, rtol=0.04)
    assert np.isclose(long_dense[1], expected_long_rotation, rtol=0.04)
    assert np.isclose(long_sparse[0], long_dense[0], rtol=0.04)
    assert np.isclose(long_sparse[1], long_dense[1], rtol=0.04)
    assert long_sparse[2] < long_dense[2]

    # Four locally resolvable +30-degree steps cross the axial seam and
    # accumulate beyond the principal +/-90-degree representation. A missing
    # intermediate ring terminates, rather than bridges, the evidence chain.
    sector_tilt = np.deg2rad(
        np.array(
            [
                [80.0, np.nan, 10.0],
                [-70.0, 20.0, 40.0],
                [-40.0, 50.0, np.nan],
                [-10.0, 80.0, 70.0],
                [20.0, -70.0, -80.0],
            ]
        )
    )
    cumulative = np.degrees(cumulative_axial_rotation(sector_tilt))
    assert np.allclose(cumulative[:, 0], [0.0, 30.0, 60.0, 90.0, 120.0])
    assert np.isnan(cumulative[0, 1])
    assert np.allclose(cumulative[1:, 1], [0.0, 30.0, 60.0, 90.0])
    assert np.allclose(cumulative[:2, 2], [0.0, 30.0])
    assert np.isnan(cumulative[2:, 2]).all()

    positive_orthogonal = cumulative_axial_rotation(
        np.deg2rad(np.array([[0.0], [90.0], [80.0]]))
    )
    negative_orthogonal = cumulative_axial_rotation(
        np.deg2rad(np.array([[0.0], [-90.0], [-80.0]]))
    )
    assert positive_orthogonal[0, 0] == 0.0
    assert negative_orthogonal[0, 0] == 0.0
    assert np.isnan(positive_orthogonal[1:, 0]).all()
    assert np.isnan(negative_orthogonal[1:, 0]).all()


if __name__ == "__main__":
    verify_radial_relative_metric_invariants()
