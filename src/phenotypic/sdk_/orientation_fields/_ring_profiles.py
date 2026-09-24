"""Radial and ring-sector transforms of an axial orientation field.

Each function works on an already-computed orientation field (for example the
output of :func:`orientation_field`) and a radial geometry around one colony
centre. None of them detects structure or reads an ``Image``. They were moved
here from ``measure/_measure_orientation_zones.py`` without changing their
arithmetic; ``MeasureOrientationZones`` is their only production consumer
today.
"""

from __future__ import annotations

import numpy as np

from ._axial import _EPS, axial_change, is_orthogonal_change
from ._constants import (
    FIBER_AXIS_OFFSET,
    MIN_AXIAL_RESULTANT,
    MIN_PIXELS_PER_SECTOR,
    N_SECTORS,
    RELIABLE_PIXEL_COHERENCE,
)


def signed_radial_relative_field(
    phi: np.ndarray,
    centre: tuple[float, float],
    dist_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute signed radial-relative tilt and outward turning.

    This is the directional diagnostic underlying the outward-turning map.
    The signed derivative is recovered from the doubled-angle field, avoiding
    false sign changes at the axial ``-pi/2 <-> pi/2`` seam. In image
    coordinates, positive values mean the fiber tilt becomes more clockwise
    relative to its outward radial spoke; negative values mean increasingly
    counterclockwise tilt.

    Args:
        phi: Gradient-normal orientation field in radians.
        centre: Inoculum centre as ``(row, column)`` in the tile frame.
        dist_map: Distance from ``centre`` for every pixel, with ``phi.shape``.

    Returns:
        Tuple ``(signed_tilt, signed_outward_turning, outward_turning,
        polar_angle)``. The first value is in ``[-pi/2, pi/2]`` radians. The
        next two are signed and magnitude radial derivatives in radians per
        pixel. ``polar_angle`` is in ``[-pi, pi]``.
    """
    rows, cols = np.indices(phi.shape, dtype=np.float64)
    delta_row = rows - float(centre[0])
    delta_col = cols - float(centre[1])
    polar_angle = np.arctan2(delta_row, delta_col)
    fiber_axis = phi + FIBER_AXIS_OFFSET
    relative = fiber_axis - polar_angle
    signed_tilt = 0.5 * np.arctan2(
        np.sin(2.0 * relative),
        np.cos(2.0 * relative),
    )

    # Differentiate the continuous doubled-angle representation so the
    # -pi/2 <-> pi/2 axial seam cannot create a false turn.
    cosine = np.cos(2.0 * signed_tilt)
    sine = np.sin(2.0 * signed_tilt)
    cosine_y, cosine_x = np.gradient(cosine)
    sine_y, sine_x = np.gradient(sine)
    radial_x = np.divide(
        delta_col,
        dist_map,
        out=np.zeros_like(delta_col),
        where=dist_map > _EPS,
    )
    radial_y = np.divide(
        delta_row,
        dist_map,
        out=np.zeros_like(delta_row),
        where=dist_map > _EPS,
    )
    cosine_r = cosine_x * radial_x + cosine_y * radial_y
    sine_r = sine_x * radial_x + sine_y * radial_y
    signed_outward_turning = 0.5 * (cosine * sine_r - sine * cosine_r)
    outward_turning = 0.5 * np.hypot(cosine_r, sine_r)
    return (
        signed_tilt,
        signed_outward_turning,
        outward_turning,
        polar_angle,
    )


def axial_sector_means(
    signed_tilt: np.ndarray,
    polar_angle: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
    n_angular_bins: int,
    eps: float = _EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reliable coherence-weighted axial means by polar sector.

    Args:
        signed_tilt: Signed radial-relative axial angle in radians.
        polar_angle: Per-pixel polar position in radians.
        coherence: Structure-tensor coherence in ``[0, 1]``.
        selector: Boolean pixels contributing to this radial band or zone.
        n_angular_bins: Number of fixed polar sectors.
        eps: Numerical floor for coherence sums.

    Returns:
        ``(mean_tilt, resultant)`` arrays of length ``n_angular_bins``. Cells
        are ``NaN`` unless they contain at least three reliable pixels and
        their doubled-angle resultant is at least 0.15.

    Raises:
        ValueError: If array shapes differ or ``n_angular_bins < 1``.
    """
    arrays = (polar_angle, coherence, selector)
    if any(array.shape != signed_tilt.shape for array in arrays):
        raise ValueError("sector-orientation arrays must share one shape")
    if n_angular_bins < 1:
        raise ValueError("n_angular_bins must be >= 1")

    means = np.full(n_angular_bins, np.nan, dtype=np.float64)
    resultants = np.full(n_angular_bins, np.nan, dtype=np.float64)
    valid = (
        selector
        & np.isfinite(signed_tilt)
        & np.isfinite(polar_angle)
        & np.isfinite(coherence)
        & (coherence >= RELIABLE_PIXEL_COHERENCE)
    )
    if not valid.any():
        return means, resultants

    angle01 = np.mod(polar_angle[valid], 2.0 * np.pi) / (2.0 * np.pi)
    sector_ids = np.minimum(
        (angle01 * n_angular_bins).astype(np.int64),
        n_angular_bins - 1,
    )
    tilts = signed_tilt[valid]
    weights = coherence[valid]
    for sector in np.unique(sector_ids):
        chosen = sector_ids == sector
        if int(chosen.sum()) < MIN_PIXELS_PER_SECTOR:
            continue
        sector_weights = weights[chosen]
        weight_sum = float(sector_weights.sum())
        if weight_sum <= eps:
            continue
        mean_cosine = float(
            np.sum(sector_weights * np.cos(2.0 * tilts[chosen])) / weight_sum
        )
        mean_sine = float(
            np.sum(sector_weights * np.sin(2.0 * tilts[chosen])) / weight_sum
        )
        resultant = float(np.hypot(mean_cosine, mean_sine))
        if resultant < MIN_AXIAL_RESULTANT:
            continue
        means[int(sector)] = 0.5 * np.arctan2(mean_sine, mean_cosine)
        resultants[int(sector)] = resultant
    return means, resultants


def radial_ring_orientation_profile(
    signed_tilt: np.ndarray,
    polar_angle: np.ndarray,
    coherence: np.ndarray,
    dist_map: np.ndarray,
    structure_selector: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    ring_width: float,
    n_angular_bins: int = N_SECTORS,
    *,
    include_outer: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate a sectorized Sholl-style orientation profile.

    Complete, equal-width annular bands are placed from ``inner_radius``
    outward. A mathematical one-pixel circle is deliberately avoided because
    rasterized circumferences provide unstable and often sparse support. Each
    ring-sector cell is summarized by :func:`axial_sector_means`.

    Args:
        signed_tilt: Signed radial-relative axial angle in radians.
        polar_angle: Per-pixel polar position in radians.
        coherence: Structure-tensor coherence in ``[0, 1]``.
        dist_map: Distance from the inoculum centre in pixels.
        structure_selector: Detected-structure mask. The inoculum exclusion is
            controlled by ``inner_radius``.
        inner_radius: Inner edge of the first ring in pixels.
        outer_radius: Exclusive outer profile radius in pixels.
        ring_width: Radial averaging width in pixels.
        n_angular_bins: Number of fixed polar sectors.
        include_outer: Include pixels exactly on ``outer_radius`` when the
            final complete ring ends on that global boundary.

    Returns:
        ``(ring_centres, sector_tilt, sector_resultant)``. The two matrices have
        shape ``(n_rings, n_angular_bins)`` and contain ``NaN`` for unsupported
        cells. Angles remain in radians.

    Raises:
        ValueError: If shapes differ, radii are invalid, or ``ring_width <= 0``.
    """
    arrays = (polar_angle, coherence, dist_map, structure_selector)
    if any(array.shape != signed_tilt.shape for array in arrays):
        raise ValueError("radial-ring arrays must share one shape")
    if ring_width <= 0:
        raise ValueError("ring_width must be > 0")
    if n_angular_bins < 1:
        raise ValueError("n_angular_bins must be >= 1")
    if (
        not np.isfinite(inner_radius)
        or not np.isfinite(outer_radius)
        or outer_radius <= inner_radius
    ):
        empty = np.empty(0, dtype=np.float64)
        empty_cells = np.empty((0, n_angular_bins), dtype=np.float64)
        return empty, empty_cells, empty_cells.copy()

    n_rings = int(np.floor((outer_radius - inner_radius) / ring_width + _EPS))
    if n_rings < 1:
        empty = np.empty(0, dtype=np.float64)
        empty_cells = np.empty((0, n_angular_bins), dtype=np.float64)
        return empty, empty_cells, empty_cells.copy()
    starts = inner_radius + np.arange(n_rings, dtype=np.float64) * ring_width
    ring_centres = starts + 0.5 * ring_width
    sector_tilt = np.full((n_rings, n_angular_bins), np.nan, dtype=np.float64)
    sector_resultant = np.full_like(sector_tilt, np.nan)
    for ring_index, start in enumerate(starts):
        ring_upper = start + ring_width
        if (
            include_outer
            and ring_index == n_rings - 1
            and np.isclose(ring_upper, outer_radius, rtol=0.0, atol=_EPS)
        ):
            ring_upper = np.nextafter(outer_radius, np.inf)
        ring_selector = (
            structure_selector & (dist_map >= start) & (dist_map < ring_upper)
        )
        means, resultants = axial_sector_means(
            signed_tilt,
            polar_angle,
            coherence,
            ring_selector,
            n_angular_bins,
        )
        sector_tilt[ring_index] = means
        sector_resultant[ring_index] = resultants
    return ring_centres, sector_tilt, sector_resultant


def cumulative_ring_rotation_profile(sector_tilt: np.ndarray) -> np.ndarray:
    """Accumulate seam-safe axial rotation from the innermost ring.

    Each angular sector is unwrapped independently by summing the signed axial
    difference between adjacent rings. This differs from summing absolute
    orientation angles, which is not geometrically meaningful. Each sector
    starts at its first reliable ring outside the inoculum and requires support
    in every subsequent ring; cumulative values after a support gap remain
    ``NaN`` rather than silently bridging missing evidence.

    Args:
        sector_tilt: Signed axial ring means in radians with shape
            ``(n_rings, n_sectors)``.

    Returns:
        Cumulative signed rotation in radians with the same shape. The first
        supported cell in each sector is zero. Values may exceed the axial principal
        range ``[-pi / 2, pi / 2]`` because adjacent changes are unwrapped and
        summed. The signed unwrapping assumes the true change between adjacent
        rings is less than 90 degrees; an exactly 90-degree axial change has no
        identifiable turning direction.

    Raises:
        ValueError: If ``sector_tilt`` is not two-dimensional.
    """
    sector_tilt = np.asarray(sector_tilt, dtype=np.float64)
    if sector_tilt.ndim != 2:
        raise ValueError("sector_tilt must be a two-dimensional array")
    cumulative = np.full_like(sector_tilt, np.nan)
    if sector_tilt.shape[0] == 0:
        return cumulative

    for sector_index in range(sector_tilt.shape[1]):
        supported_rings = np.flatnonzero(
            np.isfinite(sector_tilt[:, sector_index])
        )
        if supported_rings.size == 0:
            continue
        start = int(supported_rings[0])
        cumulative[start, sector_index] = 0.0
        for ring_index in range(start + 1, sector_tilt.shape[0]):
            if not (
                np.isfinite(sector_tilt[ring_index - 1, sector_index])
                and np.isfinite(sector_tilt[ring_index, sector_index])
            ):
                break
            adjacent_change = float(
                axial_change(
                    sector_tilt[ring_index, sector_index],
                    sector_tilt[ring_index - 1, sector_index],
                )
            )
            if is_orthogonal_change(adjacent_change):
                break
            cumulative[ring_index, sector_index] = (
                cumulative[ring_index - 1, sector_index] + adjacent_change
            )
    return cumulative


def radial_ring_sector_field(
    ring_sector_values: np.ndarray,
    polar_angle: np.ndarray,
    dist_map: np.ndarray,
    structure_selector: np.ndarray,
    inner_radius: float,
    ring_width: float,
) -> np.ndarray:
    """Paint ring-sector summaries back onto their contributing pixels.

    Args:
        ring_sector_values: Values shaped ``(n_rings, n_sectors)``.
        polar_angle: Per-pixel polar position in radians.
        dist_map: Distance from the inoculum centre in pixels.
        structure_selector: Pixels eligible for the ring calculation.
        inner_radius: Inner edge of the first ring in pixels.
        ring_width: Width of each radial ring in pixels.

    Returns:
        A floating-point field matching ``dist_map.shape``. The inoculum,
        unsupported ring sectors, and pixels outside ``structure_selector`` are
        ``NaN``.

    Raises:
        ValueError: If array shapes differ, values are not two-dimensional, or
            the radial parameters are invalid.
    """
    values = np.asarray(ring_sector_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("ring_sector_values must be a two-dimensional array")
    if (
        polar_angle.shape != dist_map.shape
        or structure_selector.shape != dist_map.shape
    ):
        raise ValueError("radial-ring field arrays must share one shape")
    if not np.isfinite(inner_radius):
        raise ValueError("inner_radius must be finite")
    if not np.isfinite(ring_width) or ring_width <= 0:
        raise ValueError("ring_width must be finite and > 0")

    field = np.full(dist_map.shape, np.nan, dtype=np.float64)
    n_rings, n_sectors = values.shape
    if n_rings == 0 or n_sectors == 0:
        return field
    eligible = (
        structure_selector
        & np.isfinite(dist_map)
        & np.isfinite(polar_angle)
        & (dist_map >= inner_radius)
    )
    rows, cols = np.nonzero(eligible)
    if rows.size == 0:
        return field
    ring_ids = np.floor(
        (dist_map[rows, cols] - inner_radius) / ring_width
    ).astype(np.int64)
    angle01 = np.mod(polar_angle[rows, cols], 2.0 * np.pi) / (2.0 * np.pi)
    sector_ids = np.minimum(
        (angle01 * n_sectors).astype(np.int64),
        n_sectors - 1,
    )
    within_profile = (ring_ids >= 0) & (ring_ids < n_rings)
    rows = rows[within_profile]
    cols = cols[within_profile]
    sampled = values[
        ring_ids[within_profile],
        sector_ids[within_profile],
    ]
    finite = np.isfinite(sampled)
    field[rows[finite], cols[finite]] = sampled[finite]
    return field


def long_range_ring_rotation_profile(
    ring_centres: np.ndarray,
    sector_tilt: np.ndarray,
    radial_lag: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare matching ring sectors across a fixed radial lag.

    Args:
        ring_centres: Increasing ring-centre radii in pixels.
        sector_tilt: Signed axial means with shape ``(n_rings, n_sectors)``.
        radial_lag: Requested centre-to-centre comparison distance in pixels.

    Returns:
        ``(pair_midpoints, signed_rotation)``. ``signed_rotation`` has one row
        per ring pair and preserves unsupported cells as ``NaN``. Only pairs
        whose centre separation matches the requested lag to floating-point
        precision are retained.

    Raises:
        ValueError: If shapes are inconsistent or ``radial_lag <= 0``.
    """
    ring_centres = np.asarray(ring_centres, dtype=np.float64)
    sector_tilt = np.asarray(sector_tilt, dtype=np.float64)
    if sector_tilt.ndim != 2 or sector_tilt.shape[0] != ring_centres.size:
        raise ValueError("sector_tilt rows must match ring_centres")
    if radial_lag <= 0:
        raise ValueError("radial_lag must be > 0")
    if ring_centres.size < 2:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, sector_tilt.shape[1]), dtype=np.float64),
        )

    midpoint_rows: list[float] = []
    rotation_rows: list[np.ndarray] = []
    tolerance = max(_EPS, radial_lag * 1e-9)
    for inner_index, inner_radius in enumerate(ring_centres[:-1]):
        target = inner_radius + radial_lag
        outer_index = int(np.searchsorted(ring_centres, target, side="left"))
        if outer_index >= ring_centres.size:
            continue
        if abs(float(ring_centres[outer_index]) - target) > tolerance:
            continue
        delta = axial_change(
            sector_tilt[outer_index], sector_tilt[inner_index]
        )
        midpoint_rows.append(0.5 * (inner_radius + ring_centres[outer_index]))
        rotation_rows.append(delta)
    if not rotation_rows:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, sector_tilt.shape[1]), dtype=np.float64),
        )
    return np.asarray(midpoint_rows), np.vstack(rotation_rows)
