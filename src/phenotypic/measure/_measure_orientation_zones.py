"""Measure absolute and radial-relative hyphal orientation by radial zone."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr, field_validator, model_validator

# Control/FigureProvider/figure are re-exported from phenotypic.abc_ (this is
# exactly what _measure_symmetric_zones.py imports).
from phenotypic.abc_ import Control, FigureProvider, MeasureFeatures, figure
from phenotypic.schema import (
    OBJECT,
    ORIENTATION_ZONE_DIAGNOSTIC,
    ORIENTATION_ZONE_PRIMARY,
)
from phenotypic.sdk_.orientation_fields import (
    aggregate_literal_crossing_zone,
    literal_crossing_ring_profile,
    literal_skeleton_ring_crossings,
)
from phenotypic.util._matched_ring_rotation import (
    matched_ring_cumulative_rotation_profile,
    matched_tracks_to_ring_sector_values,
)
from phenotypic.util._nematic_bend import fiber_bend_field
from phenotypic.util._orientation_field import orientation_field
from phenotypic.measure._zone_segmentation import (
    ZoneSegmentation,
    ZoneSegmentationParams,
    compute_zone_segmentation,
    distance_from_point,
    expand_slice_around_center,
)

_VARIANTS = ("Radial", "Mask")
_ZONES = ("Overall", "Dense", "Sparse")
_EPS = 1e-9
_RADIAL_RELATIVE_N_SECTORS = 36
_RADIAL_RELATIVE_MIN_COHERENCE = 0.15
_RADIAL_RELATIVE_MIN_PIXELS_PER_SECTOR = 3
_RADIAL_RING_MIN_RESULTANT = 0.15
_LITERAL_CROSSING_HALF_WIDTH = 1.5
_LITERAL_CROSSING_MIN_POINTS = 3
_PRIMARY_OUTWARD_METRICS = (
    "OutwardRotationSustainedPeak",
    "OutwardRotationNet",
    "OutwardRotationRate",
    "OutwardRotationConsistency",
)
_DIAGNOSTIC_OUTWARD_METRICS = (
    "OutwardRotationRawPeak",
    "OutwardRotationP90",
    "OutwardRotationP95",
    "OutwardRotationMedianMagnitude",
    "OutwardRotationAbsoluteArea",
    "OutwardRotationTotalVariation",
    "OutwardRotationRateGradient",
    "OutwardRotationRingSupport",
    "OutwardRotationRunSpanSupport",
    "OutwardRotationMedianResultant",
)

# Okabe-Ito navy for figure text (matches MeasureSymmetricZones; family comes
# from the phenotypic plotly template applied by @figure).
_OI_NAVY = "#003660"
_OI_ORANGE = "#E69F00"
_OI_SKY = "#56B4E9"
_OI_GREEN = "#009E73"
_OI_VERMILION = "#D55E00"

# ``orientation_field`` returns the dominant image-gradient normal. Fibers run
# perpendicular to that axis. The scalar metrics are invariant to this global
# quarter-turn, but the diagnostic line field is not.
_FIBER_AXIS_OFFSET = np.pi / 2.0

# Three opacity levels make local structure-tensor confidence visible without
# creating one plotly trace per line segment. Blocks below the low cutoff are
# omitted because their orientation is not reliably defined.
_QUIVER_COHERENCE_BINS = (
    ("Low C", 0.15, 0.40, 0.30),
    ("Medium C", 0.40, 0.70, 0.55),
    ("High C", 0.70, np.inf, 0.90),
)

# Base-layer selector for inspect(); mirrors MeasureSymmetricZones.BASE_LAYER
# but defaults to "detect_mat" (the tensor/segmentation source for this op).
BASE_LAYER = Control(
    label="Base layer",
    kind="select",
    default="detect_mat",
    options=("rgb", "gray", "detect_mat"),
    help="Image array rendered behind the orientation-field overlay.",
)

_BEND_SCALE_PRESETS = {
    "fine": (2.0, 4.0, 8.0),
    "balanced": (4.0, 8.0, 16.0),
    "broad": (8.0, 16.0, 32.0),
}

# 72-vertex unit circle for zone-ring polygons (replicated locally from the
# nested helper in _measure_symmetric_zones._add_overlay_traces).
_N_CIRCLE_PTS = 72
_CIRCLE_THETA = np.linspace(0.0, 2.0 * np.pi, _N_CIRCLE_PTS, endpoint=True)


def _circle_xy(
    cx: float, cy: float, r: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) for a ``_N_CIRCLE_PTS``-vertex circle of radius ``r``.

    Args:
        cx: Circle centre x (column) in plate coordinates.
        cy: Circle centre y (row) in plate coordinates.
        r: Circle radius in pixels.

    Returns:
        Tuple ``(xs, ys)`` of closed-polygon vertex coordinates.
    """
    return cx + r * np.cos(_CIRCLE_THETA), cy + r * np.sin(_CIRCLE_THETA)


def zone_selector(dist_map, r_lo, r_hi, obj_mask, variant):
    """Boolean selector for a radial zone on a tile; ``Mask`` also ∩ obj_mask.

    Args:
        dist_map: Per-pixel distance-from-centre map (tile shape).
        r_lo: Inner radius (inclusive) of the zone in pixels.
        r_hi: Outer radius (exclusive) of the zone in pixels.
        obj_mask: Boolean object mask (tile shape) used by the ``Mask`` variant.
        variant: ``"Radial"`` (all tile pixels in the ring) or ``"Mask"``
            (the ring intersected with ``obj_mask``).

    Returns:
        Boolean array (tile shape). All-False when the radius range is invalid
        (non-finite or ``r_hi <= r_lo``).
    """
    if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo:
        return np.zeros(dist_map.shape, dtype=bool)
    radial = (dist_map >= r_lo) & (dist_map < r_hi)
    if variant == "Mask":
        return radial & obj_mask
    return radial


def aggregate_orientation(phi, coherence, grad_phi, selector, eps=_EPS):
    """Coherence-weighted (R, turning, mean-coherence) over a selector.

    Args:
        phi: Orientation field in radians (tile shape).
        coherence: Structure-tensor coherence in [0, 1] (tile shape).
        grad_phi: Orientation-gradient magnitude in rad/px (tile shape).
        selector: Boolean pixel selector (tile shape).
        eps: Numerical floor for the summed-coherence denominator.

    Returns:
        ``(R, turning, mean_coherence)`` scalars. Returns ``(nan, nan, nan)``
        when the selector is empty or ``sum(coherence) ~ 0``.
    """
    if not selector.any():
        return (np.nan, np.nan, np.nan)
    C = coherence[selector]
    sumC = float(C.sum())
    if sumC < eps:
        return (np.nan, np.nan, np.nan)
    c2 = np.cos(2.0 * phi[selector])
    s2 = np.sin(2.0 * phi[selector])
    Rx = float((C * c2).sum()) / sumC
    Ry = float((C * s2).sum()) / sumC
    R = float(np.hypot(Rx, Ry))
    turning = float((C * grad_phi[selector]).sum()) / sumC
    return (R, turning, float(C.mean()))


def radial_relative_field(
    phi: np.ndarray,
    centre: tuple[float, float],
    dist_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local axial tilt from radial and its outward derivative.

    The structure tensor returns the gradient-normal axis ``phi``. Fibers are
    perpendicular to that axis, so their local axis is ``phi + pi/2`` modulo
    ``pi``. At each pixel, the fiber axis is compared with the outward polar
    spoke from ``centre`` using a doubled-angle representation. The derivative
    is then projected onto that same outward spoke.

    Args:
        phi: Gradient-normal orientation field in radians.
        centre: Inoculum centre as ``(row, column)`` in the tile frame.
        dist_map: Distance from ``centre`` for every pixel, with ``phi.shape``.

    Returns:
        Tuple ``(absolute_tilt, outward_turning, polar_angle)``. Absolute tilt
        is in ``[0, pi/2]`` radians. Outward turning is the magnitude of the
        radial derivative of the signed axial tilt, in radians per pixel.
        Polar angle is in ``[-pi, pi]``. The centre pixel has zero outward
        direction and is excluded by :func:`aggregate_radial_relative`.
    """
    signed_tilt, signed_outward_turning, outward_turning, polar_angle = (
        signed_radial_relative_field(phi, centre, dist_map)
    )
    return np.abs(signed_tilt), outward_turning, polar_angle


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
    fiber_axis = phi + _FIBER_AXIS_OFFSET
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


def aggregate_radial_relative(
    absolute_tilt: np.ndarray,
    outward_turning: np.ndarray,
    polar_angle: np.ndarray,
    coherence: np.ndarray,
    dist_map: np.ndarray,
    selector: np.ndarray,
    n_angular_bins: int,
    eps: float = _EPS,
) -> tuple[float, float, float]:
    """Aggregate radial-relative metrics with equal occupied-sector weight.

    Pixels are coherence-weighted within fixed polar sectors. Occupied sectors
    are then averaged equally, preventing highly occupied sectors from
    dominating the phenotype. Within a fixed set of reliable sectors,
    multiplying evidence without changing the tilt distributions leaves the
    point estimate unchanged. A support-threshold crossing can add a newly
    reliable sector and therefore change the estimate. Mixed orientations within
    one sector remain pixel-weighted because this method deliberately does not
    identify individual branches.

    Args:
        absolute_tilt: Absolute axial fiber-to-radial difference in radians.
        outward_turning: Radial derivative magnitude in radians per pixel.
        polar_angle: Per-pixel polar position in radians.
        coherence: Structure-tensor coherence in ``[0, 1]``.
        dist_map: Per-pixel distance from the inoculum centre.
        selector: Boolean detected-structure selector for one radial zone.
        n_angular_bins: Number of equal polar sectors around the colony.
        eps: Numerical floor for sector coherence sums.

    Returns:
        ``(mean_absolute_tilt, mean_outward_turning, sector_support)``.
        ``sector_support`` is the fraction of all fixed angular sectors meeting
        the coherence and pixel-support thresholds. The two phenotype values
        are ``NaN`` when no sector is reliable; support is then zero.

    Raises:
        ValueError: If ``n_angular_bins`` is less than one or array shapes differ.
    """
    arrays = (
        absolute_tilt,
        outward_turning,
        polar_angle,
        coherence,
        dist_map,
        selector,
    )
    if any(array.shape != absolute_tilt.shape for array in arrays[1:]):
        raise ValueError(
            "radial-relative arrays and selector must share one shape"
        )
    if n_angular_bins < 1:
        raise ValueError("n_angular_bins must be >= 1")

    valid = (
        selector
        & (dist_map > eps)
        & np.isfinite(absolute_tilt)
        & np.isfinite(outward_turning)
        & np.isfinite(coherence)
        & (coherence >= _RADIAL_RELATIVE_MIN_COHERENCE)
    )
    if not valid.any():
        return (np.nan, np.nan, 0.0)

    angle01 = np.mod(polar_angle[valid], 2.0 * np.pi) / (2.0 * np.pi)
    sector_ids = np.minimum(
        (angle01 * n_angular_bins).astype(np.int64),
        n_angular_bins - 1,
    )
    weights = coherence[valid]
    tilts = absolute_tilt[valid]
    turns = outward_turning[valid]
    sector_tilts: list[float] = []
    sector_turns: list[float] = []
    for sector in np.unique(sector_ids):
        chosen = sector_ids == sector
        if int(chosen.sum()) < _RADIAL_RELATIVE_MIN_PIXELS_PER_SECTOR:
            continue
        sector_weights = weights[chosen]
        weight_sum = float(sector_weights.sum())
        if weight_sum <= eps:
            continue
        sector_tilts.append(
            float(np.sum(sector_weights * tilts[chosen]) / weight_sum)
        )
        sector_turns.append(
            float(np.sum(sector_weights * turns[chosen]) / weight_sum)
        )
    if not sector_tilts:
        return (np.nan, np.nan, 0.0)
    sector_support = len(sector_tilts) / float(n_angular_bins)
    return (
        float(np.mean(sector_tilts)),
        float(np.mean(sector_turns)),
        sector_support,
    )


def _axial_sector_means(
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
        & (coherence >= _RADIAL_RELATIVE_MIN_COHERENCE)
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
        if int(chosen.sum()) < _RADIAL_RELATIVE_MIN_PIXELS_PER_SECTOR:
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
        if resultant < _RADIAL_RING_MIN_RESULTANT:
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
    n_angular_bins: int = _RADIAL_RELATIVE_N_SECTORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate a sectorized Sholl-style orientation profile.

    Complete, equal-width annular bands are placed from ``inner_radius``
    outward. A mathematical one-pixel circle is deliberately avoided because
    rasterized circumferences provide unstable and often sparse support. Each
    ring-sector cell is summarized by :func:`_axial_sector_means`.

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
        ring_selector = (
            structure_selector
            & (dist_map >= start)
            & (dist_map < start + ring_width)
        )
        means, resultants = _axial_sector_means(
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
            adjacent_change = 0.5 * np.arctan2(
                np.sin(
                    2.0
                    * (
                        sector_tilt[ring_index, sector_index]
                        - sector_tilt[ring_index - 1, sector_index]
                    )
                ),
                np.cos(
                    2.0
                    * (
                        sector_tilt[ring_index, sector_index]
                        - sector_tilt[ring_index - 1, sector_index]
                    )
                ),
            )
            if np.isclose(
                abs(adjacent_change),
                np.pi / 2.0,
                atol=_EPS,
                rtol=0.0,
            ):
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
        inner = sector_tilt[inner_index]
        outer = sector_tilt[outer_index]
        delta = 0.5 * np.arctan2(
            np.sin(2.0 * (outer - inner)),
            np.cos(2.0 * (outer - inner)),
        )
        midpoint_rows.append(0.5 * (inner_radius + ring_centres[outer_index]))
        rotation_rows.append(delta)
    if not rotation_rows:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, sector_tilt.shape[1]), dtype=np.float64),
        )
    return np.asarray(midpoint_rows), np.vstack(rotation_rows)


def aggregate_long_range_rotation(
    pair_midpoints: np.ndarray,
    signed_rotation: np.ndarray,
    lower_radius: float,
    upper_radius: float,
) -> tuple[float, float, float]:
    """Aggregate fixed-lag ring rotations whose midpoint lies in one zone.

    Args:
        pair_midpoints: Radial midpoint of each ring pair in pixels.
        signed_rotation: Seam-safe axial changes in radians, shaped
            ``(n_pairs, n_sectors)``.
        lower_radius: Inclusive lower midpoint radius for the zone.
        upper_radius: Exclusive upper midpoint radius for the zone.

    Returns:
        ``(mean_absolute_rotation, mean_signed_rotation, paired_support)``.
        Rotation cells receive equal weight. Support is the fraction of all
        selected pair-sector cells that are reliable. A valid zone with no
        reliable cells returns ``(NaN, NaN, 0)``.

    Raises:
        ValueError: If the pair arrays are inconsistent.
    """
    pair_midpoints = np.asarray(pair_midpoints, dtype=np.float64)
    signed_rotation = np.asarray(signed_rotation, dtype=np.float64)
    if (
        signed_rotation.ndim != 2
        or signed_rotation.shape[0] != pair_midpoints.size
    ):
        raise ValueError("signed_rotation rows must match pair_midpoints")
    if (
        not np.isfinite(lower_radius)
        or not np.isfinite(upper_radius)
        or upper_radius <= lower_radius
    ):
        return (np.nan, np.nan, np.nan)
    selected_rows = (pair_midpoints >= lower_radius) & (
        pair_midpoints < upper_radius
    )
    if not selected_rows.any():
        return (np.nan, np.nan, 0.0)
    chosen = signed_rotation[selected_rows]
    finite = np.isfinite(chosen)
    support = float(finite.sum()) / float(chosen.size)
    if not finite.any():
        return (np.nan, np.nan, support)
    values = chosen[finite]
    return (float(np.mean(np.abs(values))), float(np.mean(values)), support)


def aggregate_paired_zone_rotation(
    inner_sector_tilt: np.ndarray,
    outer_sector_tilt: np.ndarray,
) -> tuple[float, float, float]:
    """Compare matching sector means between two broad radial zones.

    Args:
        inner_sector_tilt: Signed axial means for the inner zone in radians.
        outer_sector_tilt: Signed axial means for the outer zone in radians.

    Returns:
        ``(mean_absolute_rotation, mean_signed_rotation, paired_support)``.
        Support is the fraction of fixed sectors reliable in both zones.

    Raises:
        ValueError: If the arrays differ in shape or are not one-dimensional.
    """
    inner = np.asarray(inner_sector_tilt, dtype=np.float64)
    outer = np.asarray(outer_sector_tilt, dtype=np.float64)
    if inner.ndim != 1 or outer.shape != inner.shape:
        raise ValueError("paired zone-sector arrays must share one 1-D shape")
    valid = np.isfinite(inner) & np.isfinite(outer)
    support = float(valid.sum()) / float(inner.size) if inner.size else 0.0
    if not valid.any():
        return (np.nan, np.nan, support)
    delta = 0.5 * np.arctan2(
        np.sin(2.0 * (outer[valid] - inner[valid])),
        np.cos(2.0 * (outer[valid] - inner[valid])),
    )
    return (float(np.mean(np.abs(delta))), float(np.mean(delta)), support)


def _downsample_quiver(phi, coherence, block):
    """Block-mean the doubled-angle field → (rows, cols, phi_block, coh_block).

    Circular-averages cos2φ/sin2φ (coherence-weighted) and means coherence over
    block×block cells. Returns block-centre coords in the TILE frame plus per-block
    orientation and coherence — a few KB, the only array kept in the lean cache.

    Args:
        phi: Orientation field in radians (tile shape).
        coherence: Structure-tensor coherence in [0, 1] (tile shape).
        block: Block edge length in pixels.

    Returns:
        Tuple ``(rows, cols, phi_block, coh_block)`` of ``(nr, nc)`` arrays:
        block-centre row/col in tile coordinates, per-block orientation (NaN
        where the block coherence is ~0), and per-block mean coherence.
    """
    h, w = phi.shape
    block = max(1, int(block))
    nr, nc = max(h // block, 1), max(w // block, 1)
    rows = np.empty((nr, nc))
    cols = np.empty((nr, nc))
    pb = np.empty((nr, nc))
    cb = np.empty((nr, nc))
    c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
    for i in range(nr):
        for j in range(nc):
            rsl, csl = (
                slice(i * block, (i + 1) * block),
                slice(j * block, (j + 1) * block),
            )
            cc = coherence[rsl, csl]
            rows[i, j], cols[i, j] = (
                i * block + block / 2,
                j * block + block / 2,
            )
            cb[i, j] = float(cc.mean())
            wsum = float(cc.sum())
            pb[i, j] = (
                0.5
                * np.arctan2(
                    (cc * s2[rsl, csl]).sum(), (cc * c2[rsl, csl]).sum()
                )
                if wsum > 1e-12
                else np.nan
            )
    return rows, cols, pb, cb


def _resultant_direction(phi, coherence, selector):
    """Coherence-weighted mean orientation over a selector (for the inspect glyph).

    Args:
        phi: Orientation field in radians (tile shape).
        coherence: Structure-tensor coherence in [0, 1] (tile shape).
        selector: Boolean pixel selector (tile shape).

    Returns:
        Mean orientation in radians, or NaN when the selector is empty or the
        summed coherence is ~0.
    """
    if not selector.any():
        return np.nan
    C = coherence[selector]
    if float(C.sum()) < _EPS:
        return np.nan
    return 0.5 * np.arctan2(
        float((C * np.sin(2.0 * phi[selector])).sum()),
        float((C * np.cos(2.0 * phi[selector])).sum()),
    )


class MeasureOrientationZones(MeasureFeatures, FigureProvider):
    """Measure absolute and radial-relative hyphal orientation by growth zone.

    Computes the structure-tensor orientation field over a mask-free tile (grid
    section when the image is a GridImage, else an expanded crop) and aggregates
    coherence-weighted metrics over radially-defined zones bounded by the
    symmetric radius. Absolute concentration, turning, and coherence retain both
    ``Radial`` and raw ``Mask`` variants. Radial-relative tilt and outward
    turning use detected structure and equal-weight reliable angular sectors.
    Their point estimates are count-scale-invariant while reliable-sector
    membership is unchanged; a separate support diagnostic exposes threshold
    crossings. Longer-range rotation summarizes signed radial-relative tilt in
    fixed-width Sholl-style annular bands, compares matching sectors across a
    configurable radial lag, and separately reports the broad Dense-to-Sparse
    change. Emits the
    :class:`~phenotypic.schema.ORIENTATION_ZONE_PRIMARY` columns. Set
    ``include_diagnostics=True`` to also emit the validation, comparator, and
    legacy :class:`~phenotypic.schema.ORIENTATION_ZONE_DIAGNOSTIC` columns.

    Args:
        intensity_source: Image array for the structure tensor and zone
            segmentation (``"detect_mat"`` default, ``"gray"`` alternative).
        sigma_d: Gaussian-derivative (gradient) scale in pixels, ~ hypha width.
        sigma_i: Structure-tensor integration scale in pixels.
        radial_ring_width: Width in pixels of each Sholl-style annular band
            used for longer-range radial rotation.
        long_range_lag: Centre-to-centre radial comparison distance in pixels.
            Must be an integer multiple of ``radial_ring_width``.
        outward_peak_window_rings: Odd number of consecutive literal-crossing
            rings used by the sustained-peak rolling median.
        outward_min_run_rings: Minimum contiguous supported-ring count for the
            robust net, rate, and consistency metrics.
        include_diagnostics: Emit validation comparators, quality support, raw
            peak, rate-gradient, and legacy orientation-zone columns. Defaults
            to ``False`` so only primary outward-rotation metrics are returned.
        quiver_block: inspect() quiver downsample block size in pixels.
        n_annuli: Number of equal-area annuli in the shared zone segmentation.
        pelt_penalty: PELT penalty controlling core-changepoint sensitivity.
        symmetry_threshold: Minimum angular coverage for symmetric growth.
        n_angular_bins: Number of angular bins for the coverage diagnostic.
        smoothing_window: Moving-average window (annuli) for the coverage test.
        method: Inoculum-centre estimator (``"distance"`` or ``"intensity"``).
        extent_margin: Fractional expansion of the analysis tile past the mask.
        min_samples_per_ring: Minimum pixel count per ring before interpolation.
        tau_core: Colony-ness threshold for the core/dense boundary.
        tau_dense: Colony-ness threshold for the dense/sparse boundary.
        tau_sparse: Colony-ness threshold for the sparse/outside boundary.

    Examples:
        >>> from phenotypic.data import load_synth_filamentous_plate
        >>> from phenotypic.measure import MeasureOrientationZones
        >>> image = load_synth_filamentous_plate()
        >>> df = MeasureOrientationZones().measure(image)
        >>> 'OrientZones_OutwardRotationRate-Mask-Overall' in df.columns
        True
    """

    _measurement_infoclass: ClassVar[type] = ORIENTATION_ZONE_PRIMARY
    _measurement_infoclasses: ClassVar[list[type]] = [
        ORIENTATION_ZONE_DIAGNOSTIC
    ]

    intensity_source: Literal["gray", "detect_mat"] = "detect_mat"
    sigma_d: float = 1.5
    sigma_i: float = 4.0
    radial_ring_width: float = 8.0
    long_range_lag: float = 16.0
    outward_peak_window_rings: int = 3
    outward_min_run_rings: int = 6
    include_diagnostics: bool = False
    quiver_block: int = 12
    # --- zone passthrough (defaults identical to MeasureSymmetricZones) ---
    n_annuli: int = 100
    pelt_penalty: float = 5.0
    symmetry_threshold: float = 4 / 6
    n_angular_bins: int = 6
    smoothing_window: int = 3
    method: Literal["distance", "intensity"] = "distance"
    extent_margin: float = 0.05
    min_samples_per_ring: int = 5
    tau_core: float = 0.9
    tau_dense: float = 0.5
    tau_sparse: float = 0.1
    # Per-object figure intermediates, populated by _operate. PrivateAttr keeps
    # it out of model_dump()/JSON (mirrors MeasureSymmetricZones' cache pattern).
    _cache: dict = PrivateAttr(default_factory=dict)
    _cache_image: "object | None" = PrivateAttr(default=None)
    _cache_signature: str | None = PrivateAttr(default=None)

    @field_validator("sigma_d", "sigma_i")
    @classmethod
    def _positive_sigma(cls, v):
        if v <= 0:
            raise ValueError("sigma_d and sigma_i must be > 0")
        return v

    @field_validator("radial_ring_width", "long_range_lag")
    @classmethod
    def _positive_radial_scale(cls, value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                "radial_ring_width and long_range_lag must be finite and > 0"
            )
        return value

    @field_validator("outward_peak_window_rings", mode="before")
    @classmethod
    def _valid_outward_peak_window(cls, value):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 3
            or value % 2 == 0
        ):
            raise ValueError(
                "outward_peak_window_rings must be an odd integer >= 3"
            )
        return int(value)

    @field_validator("outward_min_run_rings", mode="before")
    @classmethod
    def _valid_outward_minimum_run(cls, value):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 3
        ):
            raise ValueError("outward_min_run_rings must be an integer >= 3")
        return int(value)

    @model_validator(mode="after")
    def _validate_long_range_scales(self):
        ratio = self.long_range_lag / self.radial_ring_width
        if ratio < 1.0 or not np.isclose(
            ratio,
            round(ratio),
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(
                "long_range_lag must be an integer multiple of "
                "radial_ring_width"
            )
        return self

    def _zone_params(self) -> ZoneSegmentationParams:
        return ZoneSegmentationParams(
            n_annuli=self.n_annuli,
            pelt_penalty=self.pelt_penalty,
            symmetry_threshold=self.symmetry_threshold,
            n_angular_bins=self.n_angular_bins,
            smoothing_window=self.smoothing_window,
            method=self.method,
            extent_margin=self.extent_margin,
            min_samples_per_ring=self.min_samples_per_ring,
            tau_core=self.tau_core,
            tau_dense=self.tau_dense,
            tau_sparse=self.tau_sparse,
            intensity_source=self.intensity_source,
        )

    def _resolve_tile(self, image, seg: ZoneSegmentation, prop, label2section):
        """Return (tile_intensity, obj_mask_tile, centre_rc) for one object.

        Preferred: the object's **grid section** via ``image.grid[idx]`` — an
        object-aware cropped Image (only this object's label survives; the crop
        preserves the complete object, so it is a superset of the object's
        pixels). Verified API: ``image.grid[section_idx]`` returns a cropped
        ``Image``; the crop origin is recovered by the public exact identity
        ``origin = prop.centroid(full) - regionprops(section)[label].centroid``.
        Falls back to the mask-free expanded crop when the image is not a
        GridImage, the section lookup fails, or the section does not cover the
        r_max disk around the centre (crowded/overgrown plate).
        """
        from skimage.measure import regionprops

        min_row, min_col, max_row, max_col = prop.bbox
        object_radius_bound = max(
            np.hypot(
                row - seg.centroid_global[0], col - seg.centroid_global[1]
            )
            for row, col in (
                (min_row, min_col),
                (min_row, max_col),
                (max_row, min_col),
                (max_row, max_col),
            )
        )
        r_max = max(
            max(seg.sparse_end_radius, seg.symmetric_radius)
            * (1 + self.extent_margin),
            object_radius_bound + self.radial_ring_width,
        )
        if hasattr(image, "grid") and seg.label in label2section:
            try:
                section = image.grid[label2section[seg.label]]
                sec_props = {
                    p.label: p for p in regionprops(section.objmap[:])
                }
                sp = sec_props.get(seg.label)
                if sp is not None:
                    origin = (
                        prop.centroid[0] - sp.centroid[0],
                        prop.centroid[1] - sp.centroid[1],
                    )
                    centre = (
                        seg.centroid_global[0] - origin[0],
                        seg.centroid_global[1] - origin[1],
                    )
                    H, W = section.objmap[:].shape[:2]
                    if (
                        centre[0] - r_max >= 0
                        and centre[0] + r_max <= H
                        and centre[1] - r_max >= 0
                        and centre[1] + r_max <= W
                    ):
                        tile = np.asarray(
                            getattr(section, self.intensity_source)[:],
                            dtype=np.float64,
                        )
                        return tile, (section.objmap[:] == seg.label), centre
            except (KeyError, IndexError, ValueError, AttributeError):
                pass
        # Fallback: expanded crop on the full plate (non-grid / clipped section).
        hw = image.gray[:].shape[:2]  # 2-tuple; image.shape is (H,W,3) for RGB
        sl = expand_slice_around_center(seg.centroid_global, r_max, hw)
        tile = np.asarray(
            getattr(image, self.intensity_source)[sl], dtype=np.float64
        )
        obj_mask = image.objmap[:][sl] == seg.label
        centre = (
            seg.centroid_global[0] - sl[0].start,
            seg.centroid_global[1] - sl[1].start,
        )
        return tile, obj_mask, centre

    def _zone_bounds(self, seg: ZoneSegmentation):
        return {
            "Overall": (0.0, seg.symmetric_radius),
            "Dense": (seg.core_end_radius, seg.dense_end_radius),
            "Sparse": (seg.dense_end_radius, seg.sparse_end_radius),
        }

    def _prep(self, image):
        """Regionprops + label→grid-section map, computed ONCE per image.

        grid.info() is slow on filamentous plates, so never call it per object.
        intensity_image is required so compute_zone_segmentation can read
        prop.centroid_weighted when method="intensity" (else AttributeError).
        """
        from skimage.measure import regionprops
        from phenotypic.schema import GRID

        props = regionprops(
            image.objmap[:],
            intensity_image=image.gray[:].astype(np.float64, copy=False),
        )
        label2section = {}
        if hasattr(image, "grid"):
            info = image.grid.info()
            lab, rmi = str(OBJECT.LABEL), str(GRID.ROW_MAJOR_IDX)
            label2section = dict(
                zip(info[lab].astype(int), info[rmi].astype(int))
            )
        return props, label2section

    def _iter_object_fields(self, image, props, label2section):
        """Yield (prop, seg, obj_mask, phi, coh, grad, dist_map, centre) per object.

        SINGLE source of truth for the heavy orientation compute — reused by
        _operate() (which keeps only compact summaries) and by dashboard()'s
        coherence panel (which recomputes on demand). The full-resolution arrays
        yielded here are consumed and discarded by each caller; nothing full-res
        is retained on the instance. Tiny objects (area<10) are skipped.
        """
        for prop in props:
            if prop.area < 10:
                continue
            seg = compute_zone_segmentation(
                image, prop, params=self._zone_params()
            )
            tile, obj_mask, centre = self._resolve_tile(
                image, seg, prop, label2section
            )
            phi, coh, grad = orientation_field(
                tile, self.sigma_d, self.sigma_i
            )
            dist_map = distance_from_point(tile.shape, centre)
            yield prop, seg, obj_mask, phi, coh, grad, dist_map, centre

    def _operate(self, image) -> pd.DataFrame:  # type: ignore[override]
        props, label2section = self._prep(image)
        headers = ORIENTATION_ZONE_PRIMARY.get_headers()
        if self.include_diagnostics:
            headers = [
                *headers,
                *ORIENTATION_ZONE_DIAGNOSTIC.get_headers(),
            ]
        # pre-seed every object's row with NaN so skipped/failed objects still appear
        base: dict[int, dict] = {}
        for prop in props:
            r: dict = {OBJECT.LABEL: prop.label}
            r.update({h: np.nan for h in headers})
            base[prop.label] = r
        self._cache.clear()  # compact per-object figure records only
        self._cache_image = (
            image  # single reference (not a copy) for no-arg figures
        )
        for (
            prop,
            seg,
            obj_mask,
            phi,
            coh,
            grad,
            dist_map,
            centre,
        ) in self._iter_object_fields(image, props, label2section):
            per_zone, radial_relative, long_range, ring_profile = (
                self._fill_metrics(
                    base[prop.label],
                    seg,
                    obj_mask,
                    phi,
                    coh,
                    grad,
                    dist_map,
                    centre,
                )
            )
            # LEAN CACHE: store compact summaries only — NO full-res tile/phi/coh/
            # grad/dist_map and NO seg dataclass. Bounds memory to O(objects*blocks).
            self._cache[prop.label] = {
                "centroid_global": tuple(seg.centroid_global),
                "centre": centre,
                "radii": {
                    "core": seg.core_radius,
                    "symmetric": seg.symmetric_radius,
                    "core_end": seg.core_end_radius,
                    "dense_end": seg.dense_end_radius,
                    "sparse_end": seg.sparse_end_radius,
                },
                "zones_computed": seg.zones_computed,
                "quiver": _downsample_quiver(
                    phi, coh, self.quiver_block
                ),  # block-res
                "per_zone": per_zone,
                "radial_relative": radial_relative,
                "long_range": long_range,
                "ring_profile": ring_profile,
            }
        self._cache_signature = self.model_dump_json()
        return pd.DataFrame(
            [base[p.label] for p in props], columns=[OBJECT.LABEL, *headers]
        )

    def _write_diagnostic(self, row: dict, header: str, value: float) -> None:
        """Write one opt-in diagnostic value."""
        if self.include_diagnostics:
            row[header] = value

    def _fill_literal_crossing_metrics(
        self,
        row: dict,
        seg: ZoneSegmentation,
        obj_mask: np.ndarray,
        phi: np.ndarray,
        coherence: np.ndarray,
        dist_map: np.ndarray,
        centre: tuple[float, float],
    ) -> None:
        """Write full-length literal-crossing primary and diagnostic metrics.

        The complete detected object is skeletonized before the inoculum
        selector is applied. Ring centers cover the complete detected radial
        extent and are never trimmed to the symmetric radius.
        """

        def _write_missing() -> None:
            for zone in _ZONES:
                for metric in _PRIMARY_OUTWARD_METRICS:
                    row[f"OrientZones_{metric}-Mask-{zone}"] = np.nan
                for metric in _DIAGNOSTIC_OUTWARD_METRICS:
                    self._write_diagnostic(
                        row,
                        f"OrientZones_{metric}-Mask-{zone}",
                        np.nan,
                    )

        if not seg.zones_computed:
            _write_missing()
            return
        inner_radius = float(seg.core_end_radius)
        outside_core = (
            obj_mask & np.isfinite(dist_map) & (dist_map >= inner_radius)
        )
        if not outside_core.any():
            _write_missing()
            return

        object_extent_radius = float(np.max(dist_map[outside_core]))
        radial_span = np.nextafter(object_extent_radius, np.inf) - inner_radius
        n_rings = max(
            1,
            int(np.ceil(radial_span / self.radial_ring_width)),
        )
        outer_radius = inner_radius + n_rings * self.radial_ring_width
        radii = (
            inner_radius
            + (np.arange(n_rings, dtype=np.float64) + 0.5)
            * self.radial_ring_width
        )
        selector = zone_selector(
            dist_map,
            inner_radius,
            outer_radius,
            obj_mask,
            "Mask",
        )
        transform = literal_skeleton_ring_crossings(
            obj_mask,
            phi + _FIBER_AXIS_OFFSET,
            coherence,
            dist_map,
            centre,
            radii,
            selector=selector,
            minimum_coherence=_RADIAL_RELATIVE_MIN_COHERENCE,
            crossing_half_width=_LITERAL_CROSSING_HALF_WIDTH,
            minimum_crossing_resultant=_RADIAL_RING_MIN_RESULTANT,
        )
        profile = literal_crossing_ring_profile(
            transform,
            minimum_points=_LITERAL_CROSSING_MIN_POINTS,
            minimum_resultant=_RADIAL_RING_MIN_RESULTANT,
        )
        bounds = {
            "Overall": (inner_radius, outer_radius),
            "Dense": (
                inner_radius,
                min(float(seg.dense_end_radius), outer_radius),
            ),
            "Sparse": (
                max(float(seg.dense_end_radius), inner_radius),
                outer_radius,
            ),
        }
        for zone, (lower, upper) in bounds.items():
            metrics = aggregate_literal_crossing_zone(
                profile,
                lower,
                upper,
                peak_window_rings=self.outward_peak_window_rings,
                minimum_run_rings=self.outward_min_run_rings,
            )
            primary_values = {
                "OutwardRotationSustainedPeak": float(
                    np.degrees(metrics.sustained_peak)
                ),
                "OutwardRotationNet": float(np.degrees(metrics.net_rotation)),
                "OutwardRotationRate": float(
                    np.degrees(metrics.rotation_rate)
                ),
                "OutwardRotationConsistency": metrics.consistency,
            }
            for metric, value in primary_values.items():
                row[f"OrientZones_{metric}-Mask-{zone}"] = value

            diagnostic_values = {
                "OutwardRotationRawPeak": float(np.degrees(metrics.raw_peak)),
                "OutwardRotationP90": float(np.degrees(metrics.percentile_90)),
                "OutwardRotationP95": float(np.degrees(metrics.percentile_95)),
                "OutwardRotationMedianMagnitude": float(
                    np.degrees(metrics.median_magnitude)
                ),
                "OutwardRotationAbsoluteArea": float(
                    np.degrees(metrics.absolute_area)
                ),
                "OutwardRotationTotalVariation": float(
                    np.degrees(metrics.total_variation)
                ),
                "OutwardRotationRateGradient": float(
                    np.degrees(metrics.rate_gradient)
                ),
                "OutwardRotationRingSupport": metrics.ring_support,
                "OutwardRotationRunSpanSupport": metrics.run_span_support,
                "OutwardRotationMedianResultant": metrics.median_resultant,
            }
            for metric, value in diagnostic_values.items():
                self._write_diagnostic(
                    row,
                    f"OrientZones_{metric}-Mask-{zone}",
                    value,
                )

    def _fill_metrics(
        self,
        row,
        seg,
        obj_mask,
        phi,
        coh,
        grad,
        dist_map,
        centre,
    ):
        """Write public degree-based zone columns for one object.

        The structure-tensor and axial calculations remain in radians. Angular
        values are converted only at this output/cache boundary so exported
        measurements and diagnostic figures use degrees consistently.
        """
        per_zone = {}
        radial_relative = {}
        self._fill_literal_crossing_metrics(
            row,
            seg,
            obj_mask,
            phi,
            coh,
            dist_map,
            centre,
        )
        signed_tilt, _signed_turning, outward_turning, polar_angle = (
            signed_radial_relative_field(phi, centre, dist_map)
        )
        absolute_tilt = np.abs(signed_tilt)
        for zone, (r_lo, r_hi) in self._zone_bounds(seg).items():
            zone_ok = seg.zones_computed or zone == "Overall"
            for variant in _VARIANTS:
                if not zone_ok:
                    R = t = cm = direction = np.nan
                else:
                    sel = zone_selector(
                        dist_map, r_lo, r_hi, obj_mask, variant
                    )
                    R, t, cm = aggregate_orientation(phi, coh, grad, sel)
                    direction = _resultant_direction(phi, coh, sel)
                turning_degrees = float(np.degrees(t))
                per_zone[(variant, zone)] = (
                    R,
                    turning_degrees,
                    cm,
                    direction,
                )  # scalars only
                self._write_diagnostic(
                    row,
                    f"OrientZones_Concentration-{variant}-{zone}",
                    R,
                )
                self._write_diagnostic(
                    row,
                    f"OrientZones_Turning-{variant}-{zone}",
                    turning_degrees,
                )
                self._write_diagnostic(
                    row,
                    f"OrientZones_Coherence-{variant}-{zone}",
                    cm,
                )
            valid_bounds = (
                np.isfinite(r_lo) and np.isfinite(r_hi) and r_hi > r_lo
            )
            if not zone_ok or not valid_bounds:
                radial_tilt = radial_turning = radial_support = np.nan
            else:
                structure_selector = zone_selector(
                    dist_map,
                    r_lo,
                    r_hi,
                    obj_mask,
                    "Mask",
                )
                radial_tilt, radial_turning, radial_support = (
                    aggregate_radial_relative(
                        absolute_tilt,
                        outward_turning,
                        polar_angle,
                        coh,
                        dist_map,
                        structure_selector,
                        _RADIAL_RELATIVE_N_SECTORS,
                    )
                )
            radial_tilt_degrees = float(np.degrees(radial_tilt))
            radial_turning_degrees = float(np.degrees(radial_turning))
            radial_relative[zone] = (
                radial_tilt_degrees,
                radial_turning_degrees,
                radial_support,
            )
            self._write_diagnostic(
                row,
                f"OrientZones_RadialTilt-Mask-{zone}",
                radial_tilt_degrees,
            )
            self._write_diagnostic(
                row,
                f"OrientZones_OutwardTurning-Mask-{zone}",
                radial_turning_degrees,
            )
            self._write_diagnostic(
                row,
                f"OrientZones_RadialSectorSupport-Mask-{zone}",
                radial_support,
            )
        long_range, ring_profile = self._fill_long_range_metrics(
            row,
            seg,
            obj_mask,
            signed_tilt,
            polar_angle,
            coh,
            dist_map,
        )
        return per_zone, radial_relative, long_range, ring_profile

    def _fill_long_range_metrics(
        self,
        row,
        seg,
        obj_mask,
        signed_tilt,
        polar_angle,
        coherence,
        dist_map,
    ):
        """Write fixed-lag Sholl-style rotation metrics in public degrees.

        Complete annular bands start at the inferred inoculum boundary. Ring
        pairs are compared at ``long_range_lag`` and assigned to a radial zone
        by their midpoint. The separate Dense-to-Sparse result compares broad
        coherence-weighted sector means rather than individual ring pairs.

        Returns:
            ``(long_range, ring_profile)`` compact dictionaries for figures.
        """
        long_range = {zone: (np.nan, np.nan, np.nan) for zone in _ZONES}
        long_range["DenseToSparse"] = (np.nan, np.nan, np.nan)
        empty = np.empty(0, dtype=np.float64)
        ring_profile = {
            "radii": empty,
            "mean_absolute_tilt": empty.copy(),
            "mean_signed_tilt": empty.copy(),
            "support": empty.copy(),
            "pair_midpoints": empty.copy(),
            "mean_absolute_rotation": empty.copy(),
            "mean_signed_rotation": empty.copy(),
            "pair_support": empty.copy(),
        }

        metric_names = (
            "LongRangeRotation",
            "SignedLongRangeRotation",
            "LongRangeRotationSupport",
        )

        def _write_result(
            name: str, result: tuple[float, float, float]
        ) -> None:
            magnitude, signed, support = result
            magnitude_degrees = float(np.degrees(magnitude))
            signed_degrees = float(np.degrees(signed))
            public = (magnitude_degrees, signed_degrees, support)
            long_range[name] = public
            values = (magnitude_degrees, signed_degrees, support)
            for metric, value in zip(metric_names, values):
                self._write_diagnostic(
                    row,
                    f"OrientZones_{metric}-Mask-{name}",
                    value,
                )

        if not seg.zones_computed:
            for name in (*_ZONES, "DenseToSparse"):
                _write_result(name, (np.nan, np.nan, np.nan))
            return long_range, ring_profile

        inner_radius = float(seg.core_end_radius)
        outer_radius = min(
            float(seg.sparse_end_radius),
            float(seg.symmetric_radius),
        )
        structure_selector = zone_selector(
            dist_map,
            inner_radius,
            outer_radius,
            obj_mask,
            "Mask",
        )
        ring_centres, ring_sector_tilt, _ring_resultant = (
            radial_ring_orientation_profile(
                signed_tilt,
                polar_angle,
                coherence,
                dist_map,
                structure_selector,
                inner_radius,
                outer_radius,
                self.radial_ring_width,
                _RADIAL_RELATIVE_N_SECTORS,
            )
        )
        pair_midpoints, signed_rotation = long_range_ring_rotation_profile(
            ring_centres,
            ring_sector_tilt,
            self.long_range_lag,
        )
        long_range_bounds = {
            "Overall": (inner_radius, outer_radius),
            "Dense": (
                float(seg.core_end_radius),
                min(float(seg.dense_end_radius), outer_radius),
            ),
            "Sparse": (
                float(seg.dense_end_radius),
                outer_radius,
            ),
        }
        for zone, (lower, upper) in long_range_bounds.items():
            result = aggregate_long_range_rotation(
                pair_midpoints,
                signed_rotation,
                lower,
                upper,
            )
            _write_result(zone, result)

        dense_selector = zone_selector(
            dist_map,
            float(seg.core_end_radius),
            float(seg.dense_end_radius),
            obj_mask,
            "Mask",
        )
        sparse_selector = zone_selector(
            dist_map,
            float(seg.dense_end_radius),
            outer_radius,
            obj_mask,
            "Mask",
        )
        dense_sector_tilt, _dense_resultant = _axial_sector_means(
            signed_tilt,
            polar_angle,
            coherence,
            dense_selector,
            _RADIAL_RELATIVE_N_SECTORS,
        )
        sparse_sector_tilt, _sparse_resultant = _axial_sector_means(
            signed_tilt,
            polar_angle,
            coherence,
            sparse_selector,
            _RADIAL_RELATIVE_N_SECTORS,
        )
        transition = aggregate_paired_zone_rotation(
            dense_sector_tilt,
            sparse_sector_tilt,
        )
        _write_result("DenseToSparse", transition)

        def _summarize_cells(
            cells: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            absolute = np.full(cells.shape[0], np.nan, dtype=np.float64)
            signed = np.full(cells.shape[0], np.nan, dtype=np.float64)
            support = np.zeros(cells.shape[0], dtype=np.float64)
            for index, values in enumerate(cells):
                finite = np.isfinite(values)
                support[index] = float(finite.sum()) / float(values.size)
                if finite.any():
                    absolute[index] = float(np.mean(np.abs(values[finite])))
                    signed[index] = float(np.mean(values[finite]))
            return absolute, signed, support

        ring_absolute, ring_signed, ring_support = _summarize_cells(
            ring_sector_tilt
        )
        pair_absolute, pair_signed, pair_support = _summarize_cells(
            signed_rotation
        )
        ring_profile = {
            "radii": ring_centres,
            "mean_absolute_tilt": np.degrees(ring_absolute),
            "mean_signed_tilt": np.degrees(ring_signed),
            "support": ring_support,
            "pair_midpoints": pair_midpoints,
            "mean_absolute_rotation": np.degrees(pair_absolute),
            "mean_signed_rotation": np.degrees(pair_signed),
            "pair_support": pair_support,
        }
        return long_range, ring_profile

    def _coherence_canvas(self, image, downsample: int = 4):
        """Recompute per-object coherence and composite onto a plate canvas.

        Used only by dashboard()'s heatmap. Full-res fields are recomputed via
        _iter_object_fields and discarded here — the heatmap costs compute, not
        persistent memory. Returned canvas is downsampled for a light figure.
        """
        props, label2section = self._prep(image)
        canvas = np.full(image.gray[:].shape[:2], np.nan)
        for (
            _prop,
            seg,
            _mask,
            _phi,
            coh,
            _grad,
            _dist,
            centre,
        ) in self._iter_object_fields(image, props, label2section):
            r0 = int(round(seg.centroid_global[0] - centre[0]))
            c0 = int(round(seg.centroid_global[1] - centre[1]))
            h, w = coh.shape
            r1, c1 = min(r0 + h, canvas.shape[0]), min(c0 + w, canvas.shape[1])
            canvas[max(r0, 0) : r1, max(c0, 0) : c1] = coh[
                : r1 - max(r0, 0), : c1 - max(c0, 0)
            ]
        return canvas[::downsample, ::downsample]

    # ── figure surfaces ──────────────────────────────────────────────

    def _require_cache_image(self):
        """Return the cached image or raise if :meth:`measure` has not run."""
        if self._cache_image is None:
            raise RuntimeError(
                "MeasureOrientationZones: diagnostic cache is empty. "
                "Call .measure(image) before .inspect()/.dashboard()."
            )
        return self._cache_image

    @figure(
        title="Orientation-field overlay",
        primary=True,
        controls={"base_layer": BASE_LAYER},
    )
    def inspect(
        self,
        image=None,
        base_layer: Literal["rgb", "gray", "detect_mat"] = "detect_mat",
        *,
        for_save: bool = False,
    ):
        """Plate overview of the pixels and local fiber axes being aggregated.

        Uses the compact per-object cache populated by the most recent
        :meth:`measure` call for axes, rings, means, and hover values. The signed
        outward-turning overlay is recomputed on demand and discarded after the
        figure is assembled. Local axes are clipped to the overall radial
        selector, rotated from the structure-tensor gradient normal to the fiber
        axis, and confidence-coded by coherence. Zone metrics are available by
        hovering the colony centre.

        Args:
            image: Detected Image with objmap. If *None*, the image cached by the
                most recent :meth:`measure` call is reused.
            base_layer: Which image array to render behind the overlay
                (``"rgb"``, ``"gray"`` or ``"detect_mat"``).
            for_save: When *True*, every legend-only overlay trace is force-shown
                so the figure renders meaningfully as a static raster (the CLI's
                ``--save-inspect`` flag passes this). Defaults to *False*.

        Returns:
            A ``plotly.graph_objects.Figure`` with toggleable overlay layers.

        Examples:
            >>> from phenotypic.data import load_synth_filamentous_plate
            >>> from phenotypic.measure import MeasureOrientationZones
            >>> op = MeasureOrientationZones()
            >>> _ = op.measure(load_synth_filamentous_plate())
            >>> fig = op.inspect()
            >>> len(fig.data) > 0
            True
        """
        from phenotypic.sdk_._plotly_helpers import (
            _require_plotly,
            plotly_imshow,
        )

        _require_plotly()

        valid_base_layers = BASE_LAYER.options or ()
        if base_layer not in valid_base_layers:
            allowed = ", ".join(repr(value) for value in valid_base_layers)
            raise ValueError(
                f"base_layer must be one of {allowed}; got {base_layer!r}"
            )

        if image is None:
            image = self._require_cache_image()
        if (
            not self._cache
            or self._cache_image is not image
            or self._cache_signature != self.model_dump_json()
        ):
            self.measure(image)

        base = getattr(image, base_layer)[:]
        h, w = base.shape[:2]
        display_w = 900
        display_h = int(display_w * h / w)
        fig = plotly_imshow(
            base,
            title="Orientation zones: local fiber axes and measurement regions",
            figsize=(display_w // 100, display_h // 100),
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(legend=dict(groupclick="togglegroup"))

        self._add_signed_outward_turning_trace(fig, image, (h, w))
        self._add_long_range_ring_traces(fig)
        self._add_mask_selector_trace(fig, image)
        self._add_quiver_trace(fig)
        self._add_zone_ring_traces(fig)
        self._add_mean_axis_traces(fig)
        self._add_metric_hover_trace(fig)
        fig.add_annotation(
            text=(
                "<b>Field:</b> short blue bars are local fiber axes "
                "(tensor normal + 90°); length and opacity encode local "
                "coherence C.<br>"
                "<b>Signed outward turning:</b> the Spectral overlay shows "
                "the full observed directional range in deg/px, centred at "
                "zero.<br>Negative is counterclockwise and positive is "
                "clockwise as growth moves outward. The inferred inoculum core is "
                "excluded from this directional view.<br>"
                "<b>Long range:</b> thin concentric circles sample the "
                f"{self.radial_ring_width:g} px Sholl-style orientation bands; "
                f"matching sectors {self.long_range_lag:g} px apart are compared "
                "in degrees.<br>Hover the circles or colony centres for "
                "ring and zone-to-zone summaries.<br>"
                "<b>Selectors:</b> circles bound the radial zones; toggle mean "
                "axes and the green detected-mask boundary.<br>"
                "<b>Metrics:</b> hover colony centres for R, turning, and C. "
                "R is common parallel alignment, not radial alignment.<br>Turning "
                "is the coherence-weighted spatial change in the local axis. "
                "Radial tilt compares each detected local axis with its outward "
                "spoke; outward turning measures how that tilt changes radially."
            ),
            xref="paper",
            yref="paper",
            x=0.0,
            y=-0.19,
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            font=dict(color=_OI_NAVY, size=11),
        )
        fig.update_layout(margin=dict(b=350))
        fig.update_xaxes(range=[-0.5, w - 0.5], constrain="domain")
        fig.update_yaxes(
            range=[h - 0.5, -0.5],
            scaleanchor="x",
            scaleratio=1,
        )
        if for_save:
            for trace in fig.data:
                if getattr(trace, "visible", True) == "legendonly":
                    trace.visible = True
        return fig

    @figure(
        title="Cumulative radial rotation overlay",
        controls={"base_layer": BASE_LAYER},
    )
    def cumulative_rotation_overlay(
        self,
        image=None,
        base_layer: Literal["rgb", "gray", "detect_mat"] = "detect_mat",
    ):
        """Show accumulated ring-to-ring orientation change on the source layer.

        Each angular sector's first supported ring outside the inferred
        inoculum is its zero reference. Adjacent seam-safe axial changes are
        unwrapped and summed while radial support remains continuous. The
        result is painted only onto detected structure that contributed to the
        calculation. This is a visualization-only view; it does not add a
        branch-density measurement or alter exported metrics.

        Args:
            image: Detected Image with objmap. If *None*, reuse the image cached
                by the most recent :meth:`measure` call.
            base_layer: Image array shown beneath the Spectral overlay
                (``"rgb"``, ``"gray"`` or ``"detect_mat"``).

        Returns:
            A ``plotly.graph_objects.Figure`` with cumulative signed rotation in
            degrees, sampled ring boundaries, and optional local fiber axes.

        Examples:
            >>> from phenotypic.data import load_synth_filamentous_plate
            >>> from phenotypic.measure import MeasureOrientationZones
            >>> op = MeasureOrientationZones()
            >>> _ = op.measure(load_synth_filamentous_plate())
            >>> fig = op.cumulative_rotation_overlay()
            >>> len(fig.data) > 0
            True
        """
        from phenotypic.sdk_._plotly_helpers import (
            _require_plotly,
            plotly_imshow,
        )

        _require_plotly()
        valid_base_layers = BASE_LAYER.options or ()
        if base_layer not in valid_base_layers:
            allowed = ", ".join(repr(value) for value in valid_base_layers)
            raise ValueError(
                f"base_layer must be one of {allowed}; got {base_layer!r}"
            )
        if image is None:
            image = self._require_cache_image()
        if (
            not self._cache
            or self._cache_image is not image
            or self._cache_signature != self.model_dump_json()
        ):
            self.measure(image)

        base = getattr(image, base_layer)[:]
        height, width = base.shape[:2]
        display_width = 900
        display_height = int(display_width * height / width)
        fig = plotly_imshow(
            base,
            title=(
                "Cumulative radial rotation: accumulated change from the "
                "first supported ring in each sector"
            ),
            figsize=(display_width // 100, display_height // 100),
        )
        fig.update_coloraxes(showscale=False)
        self._add_cumulative_rotation_trace(fig, image, (height, width))
        self._add_long_range_ring_traces(fig)
        self._add_quiver_trace(fig)
        fig.add_annotation(
            text=(
                "<b>Cumulative rotation:</b> zero at each sector's first "
                f"supported {self.radial_ring_width:g} px ring outside the "
                "inoculum; "
                "then the signed, seam-safe change between adjacent rings is "
                "summed outward within each angular sector.<br>"
                "Positive and negative values are opposite turning senses. "
                "Signed unwrapping assumes adjacent-ring changes are less "
                "than 90°. "
                "Blank regions are the excluded inoculum, background, or a "
                "sector after continuous radial support was lost. Short blue "
                "bars show the local fiber axes and can be toggled in the "
                "legend."
            ),
            xref="paper",
            yref="paper",
            x=0.0,
            y=-0.19,
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            font=dict(color=_OI_NAVY, size=11),
        )
        fig.update_layout(margin=dict(b=300))
        fig.update_xaxes(range=[-0.5, width - 0.5], constrain="domain")
        fig.update_yaxes(
            range=[height - 0.5, -0.5],
            scaleanchor="x",
            scaleratio=1,
        )
        return fig

    @figure(
        title="Matched-ring cumulative fiber rotation overlay",
        controls={"base_layer": BASE_LAYER},
    )
    def matched_cumulative_rotation_overlay(
        self,
        image=None,
        base_layer: Literal["rgb", "gray", "detect_mat"] = "detect_mat",
        *,
        max_sector_shift: int = 2,
        allow_gap_bridging: bool = False,
        allow_restarts: bool = False,
    ):
        """Show fiber-axis rotation accumulated along nearby matched ring cells.

        Unlike :meth:`cumulative_rotation_overlay`, which remains in each fixed
        angular sector and accumulates radial-relative tilt changes, this
        diagnostic follows reliable fiber-axis means into nearby sectors on the
        next annular band. It accumulates seam-safe fiber-axis changes along the
        matched path. The inferred inoculum core and unsupported path segments
        remain blank. This view does not alter exported measurements.

        Args:
            image: Detected Image with objmap. If *None*, reuse the image cached
                by the most recent :meth:`measure` call.
            base_layer: Image array shown beneath the Spectral overlay
                (``"rgb"``, ``"gray"`` or ``"detect_mat"``).
            max_sector_shift: Maximum nearby 10-degree sector displacement
                allowed between adjacent rings.
            allow_gap_bridging: Whether a path may scan past rings with no
                reliable nearby candidate. Skipped rings remain blank.
            allow_restarts: Whether a terminated seed may start a new segment
                at its next reliable cell. Restarted segments reset cumulative
                rotation to zero and are not inoculum-path measurements.

        Returns:
            A ``plotly.graph_objects.Figure`` with cumulative signed fiber-axis
            rotation in degrees and the matched annular paths.

        Raises:
            ValueError: If ``base_layer`` is unsupported or
                ``max_sector_shift`` is not an integer greater than or equal to
                zero, or either continuity flag is not boolean.

        Examples:
            >>> from phenotypic.data import load_synth_filamentous_plate
            >>> from phenotypic.measure import MeasureOrientationZones
            >>> op = MeasureOrientationZones()
            >>> _ = op.measure(load_synth_filamentous_plate())
            >>> fig = op.matched_cumulative_rotation_overlay()
            >>> len(fig.data) > 0
            True
        """
        from phenotypic.sdk_._plotly_helpers import (
            _require_plotly,
            plotly_imshow,
        )

        _require_plotly()
        valid_base_layers = BASE_LAYER.options or ()
        if base_layer not in valid_base_layers:
            allowed = ", ".join(repr(value) for value in valid_base_layers)
            raise ValueError(
                f"base_layer must be one of {allowed}; got {base_layer!r}"
            )
        if isinstance(max_sector_shift, bool) or not isinstance(
            max_sector_shift, (int, np.integer)
        ):
            raise ValueError("max_sector_shift must be an integer >= 0")
        if max_sector_shift < 0:
            raise ValueError("max_sector_shift must be an integer >= 0")
        if not isinstance(allow_gap_bridging, (bool, np.bool_)):
            raise ValueError("allow_gap_bridging must be a boolean")
        if not isinstance(allow_restarts, (bool, np.bool_)):
            raise ValueError("allow_restarts must be a boolean")
        if image is None:
            image = self._require_cache_image()
        if (
            not self._cache
            or self._cache_image is not image
            or self._cache_signature != self.model_dump_json()
        ):
            self.measure(image)

        base = getattr(image, base_layer)[:]
        height, width = base.shape[:2]
        display_width = 900
        display_height = int(display_width * height / width)
        fig = plotly_imshow(
            base,
            title=(
                "Matched-ring cumulative fiber rotation: nearby annular "
                "orientation cells are connected outward"
            ),
            figsize=(display_width // 100, display_height // 100),
        )
        fig.update_coloraxes(showscale=False)
        self._add_matched_cumulative_rotation_trace(
            fig,
            image,
            (height, width),
            max_sector_shift=max_sector_shift,
            allow_gap_bridging=allow_gap_bridging,
            allow_restarts=allow_restarts,
        )
        self._add_long_range_ring_traces(fig)
        path_note = (
            "The signed, seam-safe fiber-axis changes are summed along the "
            "white path. "
            if not allow_restarts
            else "White path lines are hidden because restarted segment "
            "boundaries are not continuous trajectories. "
        )
        fig.add_annotation(
            text=(
                "<b>Matched-ring accumulation:</b> each geometric seed starts "
                f"at its first supported {self.radial_ring_width:g} px ring. "
                f"The next ring may move up to {max_sector_shift} nearby "
                "10° sectors, guided by outward radial-relative tilt, fiber-axis "
                "continuity, and orientation reliability.<br>"
                f"Gap bridging is {'enabled' if allow_gap_bridging else 'disabled'}; "
                f"segment restarts are {'enabled' if allow_restarts else 'disabled'}. "
                "Skipped rings remain blank. Restarted segments reset to 0° and "
                "are segment-relative rather than cumulative from the inoculum.<br>"
                f"{path_note}Spectral colors use the fixed full range from "
                "−180° to +180°. Blank regions are inoculum, background, or "
                "terminated/unsupported paths."
            ),
            xref="paper",
            yref="paper",
            x=0.0,
            y=-0.19,
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            font=dict(color=_OI_NAVY, size=11),
        )
        fig.update_layout(margin=dict(b=300))
        fig.update_xaxes(range=[-0.5, width - 0.5], constrain="domain")
        fig.update_yaxes(
            range=[height - 0.5, -0.5],
            scaleanchor="x",
            scaleratio=1,
        )
        return fig

    def fiber_bend_overlay(
        self,
        image=None,
        base_layer: Literal["rgb", "gray", "detect_mat"] = "detect_mat",
        scale_set: Literal["fine", "balanced", "broad"] = "balanced",
    ):
        """Compare director-line curvature at three Q-averaging scales.

        This prototype is diagnostic-only. It recomputes mask-aware fiber bend
        from the same structure-tensor field used by the existing orientation
        metrics, but it does not change measurements, schemas, caches, the
        primary :meth:`inspect` figure, or the cumulative radial reference.
        Bend is the nonnegative curvature magnitude of the local fiber-director
        integral curves, reported in degrees per pixel.

        Args:
            image: Detected Image with objmap. If *None*, reuse the image cached
                by the most recent :meth:`measure` call.
            base_layer: Image array shown under each bend panel (``"rgb"``,
                ``"gray"`` or ``"detect_mat"``).
            scale_set: Three Q-field Gaussian standard deviations in pixels:
                ``"fine"`` is 2/4/8, ``"balanced"`` is 4/8/16, and
                ``"broad"`` is 8/16/32.

        Returns:
            A three-panel ``plotly.graph_objects.Figure`` with each scale's
            complete observed bend-magnitude range.

        Examples:
            >>> from phenotypic.data import load_synth_filamentous_plate
            >>> from phenotypic.measure import MeasureOrientationZones
            >>> op = MeasureOrientationZones()
            >>> _ = op.measure(load_synth_filamentous_plate())
            >>> fig = op.fiber_bend_overlay(scale_set="balanced")
            >>> len(fig.data) >= 3
            True
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from phenotypic.sdk_._plotly_helpers import _require_plotly

        _require_plotly()
        valid_base_layers = BASE_LAYER.options or ()
        if base_layer not in valid_base_layers:
            allowed = ", ".join(repr(value) for value in valid_base_layers)
            raise ValueError(
                f"base_layer must be one of {allowed}; got {base_layer!r}"
            )
        if scale_set not in _BEND_SCALE_PRESETS:
            allowed = ", ".join(repr(value) for value in _BEND_SCALE_PRESETS)
            raise ValueError(
                f"scale_set must be one of {allowed}; got {scale_set!r}"
            )
        if image is None:
            image = self._require_cache_image()
        if (
            not self._cache
            or self._cache_image is not image
            or self._cache_signature != self.model_dump_json()
        ):
            self.measure(image)

        base = np.asarray(getattr(image, base_layer)[:])
        height, width = base.shape[:2]
        scales = _BEND_SCALE_PRESETS[scale_set]
        rasters, raw_peaks, stride = self._bend_scale_rasters(
            image,
            (height, width),
            scales,
        )
        subplot_titles = [
            (
                f"Q scale σ={scale:g} px"
                f"<br><sup>raw peak {peak:.3f} deg/px</sup>"
            )
            for scale, peak in zip(scales, raw_peaks)
        ]
        fig = make_subplots(
            rows=1,
            cols=3,
            shared_yaxes=True,
            horizontal_spacing=0.025,
            subplot_titles=subplot_titles,
        )
        raster_height, raster_width = rasters[0].shape
        raster_x = (np.arange(raster_width) + 0.5) * stride - 0.5
        raster_y = (np.arange(raster_height) + 0.5) * stride - 0.5
        display_base = base[::stride, ::stride]
        base_is_rgb = base.ndim == 3
        finite_base = display_base[np.isfinite(display_base)]
        if finite_base.size:
            base_min, base_max = np.percentile(finite_base, (1.0, 99.8))
            if base_max <= base_min:
                base_max = base_min + _EPS
        else:
            base_min, base_max = 0.0, 1.0
        if base_is_rgb:
            rgb = np.asarray(display_base, dtype=np.float64)
            rgb = (rgb - base_min) / (base_max - base_min)
            display_base = np.clip(
                np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
            display_base = np.round(display_base * 255.0).astype(np.uint8)
        for column, (scale, raster) in enumerate(
            zip(scales, rasters),
            start=1,
        ):
            coloraxis_name = (
                "coloraxis" if column == 1 else f"coloraxis{column}"
            )
            if base_is_rgb:
                fig.add_trace(
                    go.Image(
                        z=display_base,
                        x0=float(raster_x[0]),
                        y0=float(raster_y[0]),
                        dx=stride,
                        dy=stride,
                        hoverinfo="skip",
                        name=base_layer,
                    ),
                    row=1,
                    col=column,
                )
            else:
                fig.add_trace(
                    go.Heatmap(
                        x=raster_x,
                        y=raster_y,
                        z=display_base,
                        colorscale=((0.0, "black"), (1.0, "white")),
                        zmin=float(base_min),
                        zmax=float(base_max),
                        showscale=False,
                        hoverinfo="skip",
                        name=base_layer,
                        showlegend=False,
                    ),
                    row=1,
                    col=column,
                )
            fig.add_trace(
                go.Heatmap(
                    x=raster_x,
                    y=raster_y,
                    z=raster,
                    coloraxis=coloraxis_name,
                    opacity=0.72,
                    name=f"Fiber bend σ={scale:g} px",
                    hovertemplate=(
                        f"Q scale={scale:g} px<br>"
                        "Fiber bend=%{z:.3f} deg/px<extra></extra>"
                    ),
                    connectgaps=False,
                ),
                row=1,
                col=column,
            )
        coloraxis_layout: dict[str, dict] = {}
        for column, peak in enumerate(raw_peaks, start=1):
            coloraxis_name = (
                "coloraxis" if column == 1 else f"coloraxis{column}"
            )
            full_range = peak
            if not np.isfinite(full_range) or full_range <= _EPS:
                full_range = float(np.finfo(np.float32).eps)
            coloraxis_layout[coloraxis_name] = dict(
                colorscale="Viridis",
                cmin=0.0,
                cmax=full_range,
                colorbar=dict(
                    title=dict(text="Bend (deg/px)", side="top"),
                    orientation="h",
                    x=(column - 0.5) / 3.0,
                    xanchor="center",
                    y=-0.08,
                    yanchor="top",
                    len=0.27,
                    thickness=12,
                ),
            )
        fig.update_layout(
            title=(
                "Multiscale fiber bend: curvature along the local director "
                "field"
            ),
            width=1500,
            height=max(520, min(1050, int(500 * height / width + 280))),
            margin=dict(b=260),
            showlegend=False,
            **coloraxis_layout,
        )
        for index in range(1, 4):
            xaxis_name = "xaxis" if index == 1 else f"xaxis{index}"
            yaxis_name = "yaxis" if index == 1 else f"yaxis{index}"
            xref = "x" if index == 1 else f"x{index}"
            fig.layout[xaxis_name].update(
                range=[-0.5, width - 0.5],
                constrain="domain",
                showticklabels=False,
            )
            fig.layout[yaxis_name].update(
                range=[height - 0.5, -0.5],
                scaleanchor=xref,
                scaleratio=1,
                showticklabels=False,
            )
        fig.add_annotation(
            text=(
                "<b>Interpretation:</b> brighter values are stronger local "
                "curvature along the fiber-orientation field. Scale controls "
                "Q-field averaging, not the structure-tensor scales. The "
                "inoculum, background, low-coherence pixels, and mixed "
                "scale-local orientations are blank.<br>"
                "A feature that persists across scales is less likely to be a "
                "single-scale texture artifact. Each panel uses its own full "
                "raw range, so compare spatial persistence across panels, not "
                "color brightness.<br>Bend is unsigned because a fiber director "
                "has no intrinsic arrowhead. Existing radial and cumulative "
                "figures remain the signed references."
            ),
            xref="paper",
            yref="paper",
            x=0.0,
            y=-0.18,
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            font=dict(color=_OI_NAVY, size=11),
        )
        return fig

    def _bend_scale_rasters(
        self,
        image,
        image_shape: tuple[int, int],
        scales: tuple[float, ...],
    ) -> tuple[list[np.ndarray], list[float], int]:
        """Composite scale-local fiber bend into bounded plate rasters."""
        props, label2section = self._prep(image)
        height, width = image_shape
        stride = max(1, int(np.ceil(max(height, width) / 900.0)))
        raster_height = int(np.ceil(height / stride))
        raster_width = int(np.ceil(width / stride))
        maxima = [
            np.full(raster_height * raster_width, -np.inf, dtype=np.float32)
            for _scale in scales
        ]
        raw_peaks = [0.0 for _scale in scales]
        for (
            _prop,
            seg,
            obj_mask,
            phi,
            coherence,
            _gradient,
            dist_map,
            centre,
        ) in self._iter_object_fields(image, props, label2section):
            inner_radius = float(seg.core_end_radius)
            outer_radius = min(
                float(seg.sparse_end_radius),
                float(seg.symmetric_radius),
            )
            if inner_radius <= _EPS or outer_radius <= inner_radius:
                continue
            selector = zone_selector(
                dist_map,
                inner_radius,
                outer_radius,
                obj_mask,
                "Mask",
            )
            origin_row = int(round(seg.centroid_global[0] - centre[0]))
            origin_col = int(round(seg.centroid_global[1] - centre[1]))
            for scale_index, scale in enumerate(scales):
                bend, scale_resultant = fiber_bend_field(
                    phi,
                    coherence,
                    selector,
                    scale,
                )
                valid = (
                    selector
                    & np.isfinite(bend)
                    & np.isfinite(coherence)
                    & np.isfinite(scale_resultant)
                    & (coherence >= _RADIAL_RELATIVE_MIN_COHERENCE)
                    & (scale_resultant >= _RADIAL_RING_MIN_RESULTANT)
                )
                if not valid.any():
                    continue
                rows, cols = np.nonzero(valid)
                global_rows = rows + origin_row
                global_cols = cols + origin_col
                inside = (
                    (global_rows >= 0)
                    & (global_rows < height)
                    & (global_cols >= 0)
                    & (global_cols < width)
                )
                if not inside.any():
                    continue
                values = np.degrees(bend[valid]).astype(np.float32)[inside]
                raw_peaks[scale_index] = max(
                    raw_peaks[scale_index],
                    float(np.max(values)),
                )
                flat_ids = (
                    global_rows[inside] // stride
                ) * raster_width + global_cols[inside] // stride
                np.maximum.at(maxima[scale_index], flat_ids, values)
        rasters: list[np.ndarray] = []
        for scale_maxima in maxima:
            raster = scale_maxima.reshape(raster_height, raster_width)
            raster = np.where(np.isfinite(raster), raster, np.nan).astype(
                np.float32,
                copy=False,
            )
            rasters.append(raster)
        return rasters, raw_peaks, stride

    def _add_long_range_ring_traces(self, fig) -> None:
        """Overlay sampled orientation bands and fixed-lag pair midpoints.

        At most eight rings per object are drawn to keep plate-scale figures
        responsive. The complete fixed-width profile still drives the metrics;
        display subsampling does not change any calculation. Line traces skip
        hover; one compact marker per displayed circle carries the values.
        """
        import plotly.graph_objects as go

        ring_xs: list[float | None] = []
        ring_ys: list[float | None] = []
        ring_marker_x: list[float] = []
        ring_marker_y: list[float] = []
        ring_hover: list[str] = []
        pair_xs: list[float | None] = []
        pair_ys: list[float | None] = []
        pair_marker_x: list[float] = []
        pair_marker_y: list[float] = []
        pair_hover: list[str] = []
        for record in self._cache.values():
            profile = record.get("ring_profile", {})
            radii = np.asarray(profile.get("radii", []), dtype=float)
            absolute = np.asarray(
                profile.get("mean_absolute_tilt", []), dtype=float
            )
            signed = np.asarray(
                profile.get("mean_signed_tilt", []), dtype=float
            )
            support = np.asarray(profile.get("support", []), dtype=float)
            if radii.size == 0:
                continue
            step = max(1, int(np.ceil(radii.size / 8.0)))
            indices = list(range(0, radii.size, step))
            if indices[-1] != radii.size - 1:
                indices.append(radii.size - 1)
            cy, cx = record["centroid_global"]
            for index in indices:
                radius = float(radii[index])
                circle_x, circle_y = _circle_xy(cx, cy, radius)
                hover = (
                    "<b>Sholl-style orientation band</b><br>"
                    f"Radius={radius:.1f} px<br>"
                    f"Mean |radial tilt|={absolute[index]:.2f}°<br>"
                    f"Mean signed radial tilt={signed[index]:.2f}°<br>"
                    f"Sector support={support[index]:.3f}"
                )
                ring_xs.extend([*circle_x.tolist(), None])
                ring_ys.extend([*circle_y.tolist(), None])
                ring_marker_x.append(float(cx + radius))
                ring_marker_y.append(float(cy))
                ring_hover.append(hover)

            pair_midpoints = np.asarray(
                profile.get("pair_midpoints", []), dtype=float
            )
            pair_absolute = np.asarray(
                profile.get("mean_absolute_rotation", []), dtype=float
            )
            pair_signed = np.asarray(
                profile.get("mean_signed_rotation", []), dtype=float
            )
            pair_support = np.asarray(
                profile.get("pair_support", []), dtype=float
            )
            if pair_midpoints.size == 0:
                continue
            pair_step = max(1, int(np.ceil(pair_midpoints.size / 8.0)))
            pair_indices = list(range(0, pair_midpoints.size, pair_step))
            if pair_indices[-1] != pair_midpoints.size - 1:
                pair_indices.append(pair_midpoints.size - 1)
            for index in pair_indices:
                radius = float(pair_midpoints[index])
                circle_x, circle_y = _circle_xy(cx, cy, radius)
                hover = (
                    "<b>Fixed-lag ring comparison</b><br>"
                    f"Pair midpoint={radius:.1f} px<br>"
                    f"Radial lag={self.long_range_lag:g} px<br>"
                    f"Mean |rotation|={pair_absolute[index]:.2f}°<br>"
                    f"Mean signed rotation={pair_signed[index]:.2f}°<br>"
                    f"Paired-sector support={pair_support[index]:.3f}"
                )
                pair_xs.extend([*circle_x.tolist(), None])
                pair_ys.extend([*circle_y.tolist(), None])
                pair_marker_x.append(float(cx + radius))
                pair_marker_y.append(float(cy))
                pair_hover.append(hover)

        if ring_xs:
            fig.add_trace(
                go.Scattergl(
                    x=ring_xs,
                    y=ring_ys,
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.72)", width=0.8),
                    name=(
                        f"Orientation rings ({self.radial_ring_width:g} px bands)"
                    ),
                    legendgroup="long-range-rings",
                    legendgrouptitle_text="Long-range radial rotation",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ring_marker_x,
                    y=ring_marker_y,
                    text=ring_hover,
                    mode="markers",
                    marker=dict(
                        size=6,
                        color="white",
                        line=dict(color=_OI_NAVY, width=0.8),
                    ),
                    name="Ring orientation (hover)",
                    legendgroup="long-range-rings",
                    showlegend=False,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        if pair_xs:
            fig.add_trace(
                go.Scattergl(
                    x=pair_xs,
                    y=pair_ys,
                    mode="lines",
                    line=dict(color=_OI_ORANGE, width=1.2, dash="dot"),
                    name=f"{self.long_range_lag:g} px pair midpoints",
                    legendgroup="long-range-rings",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pair_marker_x,
                    y=pair_marker_y,
                    text=pair_hover,
                    mode="markers",
                    marker=dict(
                        size=7,
                        symbol="diamond",
                        color=_OI_ORANGE,
                        line=dict(color=_OI_NAVY, width=0.8),
                    ),
                    name="Long-range rotation (hover)",
                    legendgroup="long-range-rings",
                    showlegend=False,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    def _add_signed_outward_turning_trace(
        self,
        fig,
        image,
        image_shape: tuple[int, int],
    ) -> None:
        """Overlay reliable signed outward turning as a bounded raster.

        The field is recomputed from the same ``intensity_source`` and tensor
        scales used for measurement. Values are not averaged, clipped, or
        percentile-normalized. The Spectral colorscale spans the complete
        observed range symmetrically around zero so equal clockwise and
        counterclockwise rates receive equal color distance from the midpoint.
        The raster is capped at the inspect display resolution; each display
        cell retains the signed source pixel with the greatest absolute rate,
        preventing opposite directions from cancelling while avoiding one
        Plotly marker per source pixel.
        """
        import plotly.graph_objects as go

        props, label2section = self._prep(image)
        height, width = image_shape
        stride = max(1, int(np.ceil(max(height, width) / 900.0)))
        raster_height = int(np.ceil(height / stride))
        raster_width = int(np.ceil(width / stride))
        raster = np.full(
            (raster_height, raster_width),
            np.nan,
            dtype=np.float32,
        )
        positive_max = np.full(raster.size, -np.inf, dtype=np.float32)
        negative_abs_max = np.full(raster.size, -np.inf, dtype=np.float32)
        full_range = 0.0
        for (
            _prop,
            seg,
            obj_mask,
            phi,
            coherence,
            _gradient,
            dist_map,
            centre,
        ) in self._iter_object_fields(image, props, label2section):
            _tilt, signed_turning, _magnitude, _polar = (
                signed_radial_relative_field(phi, centre, dist_map)
            )
            # Match the Dense zone's actual inner boundary. The separate PELT
            # ``core_radius`` can equal the full symmetric radius when no early
            # density changepoint is found, which would erase the entire view.
            core_exclusion_radius = float(seg.core_end_radius)
            outer_radius = min(
                float(seg.sparse_end_radius),
                float(seg.symmetric_radius),
            )
            selector = zone_selector(
                dist_map,
                core_exclusion_radius,
                outer_radius,
                obj_mask,
                "Mask",
            )
            valid = (
                selector
                & (dist_map > _EPS)
                & np.isfinite(signed_turning)
                & np.isfinite(coherence)
                & (coherence >= _RADIAL_RELATIVE_MIN_COHERENCE)
            )
            if not valid.any():
                continue
            rows, cols = np.nonzero(valid)
            origin_row = int(round(seg.centroid_global[0] - centre[0]))
            origin_col = int(round(seg.centroid_global[1] - centre[1]))
            global_rows = rows + origin_row
            global_cols = cols + origin_col
            inside = (
                (global_rows >= 0)
                & (global_rows < height)
                & (global_cols >= 0)
                & (global_cols < width)
            )
            if not inside.any():
                continue
            values = np.degrees(signed_turning[valid]).astype(np.float32)
            values = values[inside]
            full_range = max(full_range, float(np.max(np.abs(values))))
            flat_ids = (
                global_rows[inside] // stride
            ) * raster_width + global_cols[inside] // stride
            positive = values >= 0.0
            np.maximum.at(
                positive_max,
                flat_ids[positive],
                values[positive],
            )
            negative = ~positive
            np.maximum.at(
                negative_abs_max,
                flat_ids[negative],
                -values[negative],
            )
        has_positive = np.isfinite(positive_max)
        has_negative = np.isfinite(negative_abs_max)
        choose_positive = has_positive & (
            ~has_negative | (positive_max >= negative_abs_max)
        )
        raster_flat = raster.ravel()
        raster_flat[choose_positive] = positive_max[choose_positive]
        choose_negative = has_negative & ~choose_positive
        raster_flat[choose_negative] = -negative_abs_max[choose_negative]
        if not np.isfinite(raster).any():
            return

        if not np.isfinite(full_range) or full_range <= _EPS:
            full_range = float(np.finfo(np.float32).eps)
        fig.add_trace(
            go.Heatmap(
                x=(np.arange(raster_width) + 0.5) * stride - 0.5,
                y=(np.arange(raster_height) + 0.5) * stride - 0.5,
                z=raster,
                colorscale="Spectral",
                zmin=-full_range,
                zmax=full_range,
                zmid=0.0,
                colorbar=dict(
                    title=dict(
                        text="Signed outward turning (deg/px)",
                        side="top",
                    ),
                    orientation="h",
                    x=0.45,
                    xanchor="center",
                    y=-0.075,
                    yanchor="top",
                    len=0.55,
                    thickness=14,
                ),
                opacity=0.55,
                name="Signed outward turning",
                legendgroup="directional-turning",
                legendgrouptitle_text="Directional diagnostic",
                hovertemplate=(
                    "Signed outward turning=%{z:.3f} deg/px<extra></extra>"
                ),
                connectgaps=False,
            )
        )

    def _add_cumulative_rotation_trace(
        self,
        fig,
        image,
        image_shape: tuple[int, int],
    ) -> None:
        """Overlay cumulative ring-to-ring axial rotation in degrees."""
        import plotly.graph_objects as go

        props, label2section = self._prep(image)
        height, width = image_shape
        stride = max(1, int(np.ceil(max(height, width) / 900.0)))
        raster_height = int(np.ceil(height / stride))
        raster_width = int(np.ceil(width / stride))
        raster = np.full(
            (raster_height, raster_width),
            np.nan,
            dtype=np.float32,
        )
        strongest = np.full(raster.size, -np.inf, dtype=np.float32)
        signed_value = np.full(raster.size, np.nan, dtype=np.float32)
        full_range = 0.0
        for (
            _prop,
            seg,
            obj_mask,
            phi,
            coherence,
            _gradient,
            dist_map,
            centre,
        ) in self._iter_object_fields(image, props, label2section):
            signed_tilt, _turning, _magnitude, polar_angle = (
                signed_radial_relative_field(phi, centre, dist_map)
            )
            inner_radius = float(seg.core_end_radius)
            outer_radius = min(
                float(seg.sparse_end_radius),
                float(seg.symmetric_radius),
            )
            structure_selector = zone_selector(
                dist_map,
                inner_radius,
                outer_radius,
                obj_mask,
                "Mask",
            )
            _radii, sector_tilt, _resultant = radial_ring_orientation_profile(
                signed_tilt,
                polar_angle,
                coherence,
                dist_map,
                structure_selector,
                inner_radius,
                outer_radius,
                self.radial_ring_width,
                _RADIAL_RELATIVE_N_SECTORS,
            )
            cumulative = cumulative_ring_rotation_profile(sector_tilt)
            reliable_structure = (
                structure_selector
                & np.isfinite(coherence)
                & (coherence >= _RADIAL_RELATIVE_MIN_COHERENCE)
            )
            local_field = radial_ring_sector_field(
                cumulative,
                polar_angle,
                dist_map,
                reliable_structure,
                inner_radius,
                self.radial_ring_width,
            )
            valid = np.isfinite(local_field)
            if not valid.any():
                continue
            rows, cols = np.nonzero(valid)
            origin_row = int(round(seg.centroid_global[0] - centre[0]))
            origin_col = int(round(seg.centroid_global[1] - centre[1]))
            global_rows = rows + origin_row
            global_cols = cols + origin_col
            inside = (
                (global_rows >= 0)
                & (global_rows < height)
                & (global_cols >= 0)
                & (global_cols < width)
            )
            if not inside.any():
                continue
            values = np.degrees(local_field[valid]).astype(np.float32)[inside]
            full_range = max(full_range, float(np.max(np.abs(values))))
            flat_ids = (
                global_rows[inside] // stride
            ) * raster_width + global_cols[inside] // stride
            magnitudes = np.abs(values)
            order = np.lexsort((magnitudes, flat_ids))
            ordered_ids = flat_ids[order]
            last_for_id = np.r_[
                ordered_ids[1:] != ordered_ids[:-1],
                True,
            ]
            chosen_ids = ordered_ids[last_for_id]
            chosen_values = values[order][last_for_id]
            chosen_magnitudes = magnitudes[order][last_for_id]
            stronger = chosen_magnitudes >= strongest[chosen_ids]
            strongest[chosen_ids[stronger]] = chosen_magnitudes[stronger]
            signed_value[chosen_ids[stronger]] = chosen_values[stronger]

        available = np.isfinite(strongest)
        raster.ravel()[available] = signed_value[available]
        if not np.isfinite(raster).any():
            return
        if not np.isfinite(full_range) or full_range <= _EPS:
            full_range = float(np.finfo(np.float32).eps)
        fig.add_trace(
            go.Heatmap(
                x=(np.arange(raster_width) + 0.5) * stride - 0.5,
                y=(np.arange(raster_height) + 0.5) * stride - 0.5,
                z=raster,
                colorscale="Spectral",
                zmin=-full_range,
                zmax=full_range,
                zmid=0.0,
                colorbar=dict(
                    title=dict(
                        text="Cumulative signed radial rotation (deg)",
                        side="top",
                    ),
                    orientation="h",
                    x=0.45,
                    xanchor="center",
                    y=-0.075,
                    yanchor="top",
                    len=0.55,
                    thickness=14,
                ),
                opacity=0.58,
                name="Cumulative radial rotation",
                legendgroup="cumulative-radial-rotation",
                legendgrouptitle_text="Cumulative directional diagnostic",
                hovertemplate=(
                    "Cumulative signed rotation=%{z:.2f}°<extra></extra>"
                ),
                connectgaps=False,
            )
        )

    def _add_matched_cumulative_rotation_trace(
        self,
        fig,
        image,
        image_shape: tuple[int, int],
        *,
        max_sector_shift: int,
        allow_gap_bridging: bool = False,
        allow_restarts: bool = False,
    ) -> None:
        """Overlay matched-ring cumulative fiber-axis rotation in degrees."""
        import plotly.graph_objects as go

        props, label2section = self._prep(image)
        height, width = image_shape
        stride = max(1, int(np.ceil(max(height, width) / 900.0)))
        raster_height = int(np.ceil(height / stride))
        raster_width = int(np.ceil(width / stride))
        raster = np.full(
            (raster_height, raster_width),
            np.nan,
            dtype=np.float32,
        )
        strongest = np.full(raster.size, -np.inf, dtype=np.float32)
        signed_value = np.full(raster.size, np.nan, dtype=np.float32)
        path_x: list[float | None] = []
        path_y: list[float | None] = []
        bridge_x: list[float | None] = []
        bridge_y: list[float | None] = []
        for (
            _prop,
            seg,
            obj_mask,
            phi,
            coherence,
            _gradient,
            dist_map,
            centre,
        ) in self._iter_object_fields(image, props, label2section):
            _signed_tilt, _turning, _magnitude, polar_angle = (
                signed_radial_relative_field(phi, centre, dist_map)
            )
            fiber_axis = phi + _FIBER_AXIS_OFFSET
            inner_radius = float(seg.core_end_radius)
            outer_radius = min(
                float(seg.sparse_end_radius),
                float(seg.symmetric_radius),
            )
            structure_selector = zone_selector(
                dist_map,
                inner_radius,
                outer_radius,
                obj_mask,
                "Mask",
            )
            radii, sector_orientation, sector_resultant = (
                radial_ring_orientation_profile(
                    fiber_axis,
                    polar_angle,
                    coherence,
                    dist_map,
                    structure_selector,
                    inner_radius,
                    outer_radius,
                    self.radial_ring_width,
                    _RADIAL_RELATIVE_N_SECTORS,
                )
            )
            cumulative, path_sectors = (
                matched_ring_cumulative_rotation_profile(
                    radii,
                    sector_orientation,
                    sector_resultant,
                    max_sector_shift=max_sector_shift,
                    allow_gap_bridging=allow_gap_bridging,
                    allow_restarts=allow_restarts,
                )
            )
            ring_sector_values = matched_tracks_to_ring_sector_values(
                cumulative,
                path_sectors,
            )
            reliable_structure = (
                structure_selector
                & np.isfinite(coherence)
                & (coherence >= _RADIAL_RELATIVE_MIN_COHERENCE)
            )
            local_field = radial_ring_sector_field(
                ring_sector_values,
                polar_angle,
                dist_map,
                reliable_structure,
                inner_radius,
                self.radial_ring_width,
            )
            valid = np.isfinite(local_field)
            origin_row = int(round(seg.centroid_global[0] - centre[0]))
            origin_col = int(round(seg.centroid_global[1] - centre[1]))
            if valid.any():
                rows, cols = np.nonzero(valid)
                global_rows = rows + origin_row
                global_cols = cols + origin_col
                inside = (
                    (global_rows >= 0)
                    & (global_rows < height)
                    & (global_cols >= 0)
                    & (global_cols < width)
                )
                if inside.any():
                    values = np.degrees(local_field[valid]).astype(np.float32)
                    values = values[inside]
                    flat_ids = (
                        global_rows[inside] // stride
                    ) * raster_width + global_cols[inside] // stride
                    magnitudes = np.abs(values)
                    order = np.lexsort((magnitudes, flat_ids))
                    ordered_ids = flat_ids[order]
                    last_for_id = np.r_[
                        ordered_ids[1:] != ordered_ids[:-1],
                        True,
                    ]
                    chosen_ids = ordered_ids[last_for_id]
                    chosen_values = values[order][last_for_id]
                    chosen_magnitudes = magnitudes[order][last_for_id]
                    stronger = chosen_magnitudes >= strongest[chosen_ids]
                    strongest[chosen_ids[stronger]] = chosen_magnitudes[
                        stronger
                    ]
                    signed_value[chosen_ids[stronger]] = chosen_values[
                        stronger
                    ]

            n_sectors = sector_orientation.shape[1]
            sector_angles = (np.arange(n_sectors, dtype=np.float64) + 0.5) * (
                2.0 * np.pi / float(n_sectors)
            )
            for seed_sector in range(n_sectors):
                supported = np.flatnonzero(
                    (path_sectors[:, seed_sector] >= 0)
                    & np.isfinite(cumulative[:, seed_sector])
                )
                if supported.size < 2:
                    continue
                matched_sectors = path_sectors[supported, seed_sector]
                angles = sector_angles[matched_sectors]
                local_rows = centre[0] + radii[supported] * np.sin(angles)
                local_cols = centre[1] + radii[supported] * np.cos(angles)
                global_rows = local_rows + origin_row
                global_cols = local_cols + origin_col
                inside = (
                    (global_rows >= 0.0)
                    & (global_rows < height)
                    & (global_cols >= 0.0)
                    & (global_cols < width)
                )
                if np.count_nonzero(inside) < 2:
                    continue
                inside_rings = supported[inside]
                inside_cols = global_cols[inside]
                inside_rows = global_rows[inside]
                for point_index in range(1, inside_rings.size):
                    is_bridge = (
                        inside_rings[point_index]
                        - inside_rings[point_index - 1]
                        > 1
                    )
                    target_x = bridge_x if is_bridge else path_x
                    target_y = bridge_y if is_bridge else path_y
                    target_x.extend(
                        [
                            float(inside_cols[point_index - 1]),
                            float(inside_cols[point_index]),
                            None,
                        ]
                    )
                    target_y.extend(
                        [
                            float(inside_rows[point_index - 1]),
                            float(inside_rows[point_index]),
                            None,
                        ]
                    )

        available = np.isfinite(strongest)
        raster.ravel()[available] = signed_value[available]
        if np.isfinite(raster).any():
            fig.add_trace(
                go.Heatmap(
                    x=(np.arange(raster_width) + 0.5) * stride - 0.5,
                    y=(np.arange(raster_height) + 0.5) * stride - 0.5,
                    z=raster,
                    colorscale="Spectral",
                    zmin=-180.0,
                    zmax=180.0,
                    zmid=0.0,
                    colorbar=dict(
                        title=dict(
                            text="Matched cumulative fiber rotation (deg)",
                            side="top",
                        ),
                        orientation="h",
                        x=0.45,
                        xanchor="center",
                        y=-0.075,
                        yanchor="top",
                        len=0.55,
                        thickness=14,
                    ),
                    opacity=0.68,
                    name="Matched cumulative fiber rotation",
                    legendgroup="matched-cumulative-rotation",
                    legendgrouptitle_text="Matched-ring directional diagnostic",
                    hovertemplate=(
                        "Matched cumulative rotation=%{z:.2f}°<extra></extra>"
                    ),
                    connectgaps=False,
                )
            )
        if path_x and not allow_restarts:
            fig.add_trace(
                go.Scattergl(
                    x=path_x,
                    y=path_y,
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.58)", width=1.0),
                    name="Matched outward ring paths",
                    legendgroup="matched-cumulative-rotation",
                    hoverinfo="skip",
                )
            )
        if bridge_x and not allow_restarts:
            fig.add_trace(
                go.Scattergl(
                    x=bridge_x,
                    y=bridge_y,
                    mode="lines",
                    line=dict(
                        color="rgba(255,255,255,0.48)",
                        width=1.0,
                        dash="dash",
                    ),
                    name="Bridged unsupported rings",
                    legendgroup="matched-cumulative-rotation",
                    hoverinfo="skip",
                )
            )

    @staticmethod
    def _add_mask_selector_trace(fig, image) -> None:
        """Add the detected-mask boundary used by the ``Mask`` variant.

        The contour is legend-only during interactive use. Static inspect
        export shows it so the mask-intersected selector can be compared with
        the concentric ``Radial`` selector without displaying object numbers.
        """
        import plotly.graph_objects as go

        mask = (image.objmap[:] > 0).astype(np.uint8)
        if not mask.any():
            return
        fig.add_trace(
            go.Contour(
                z=mask,
                autocontour=False,
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=1.0,
                    coloring="lines",
                    showlabels=False,
                ),
                line=dict(color=_OI_GREEN, width=1.5),
                showscale=False,
                showlegend=True,
                name="Detected-mask selector",
                legendgroup="selectors",
                legendgrouptitle_text="Selectors",
                visible="legendonly",
                hoverinfo="skip",
            )
        )

    @staticmethod
    def _tile_origin(record) -> tuple[float, float]:
        """Plate-frame (row, col) origin of a cached object's tile.

        The tile pixel ``(r_tile, c_tile)`` sits at plate coordinates
        ``(r_tile + origin_row, c_tile + origin_col)``; the inoculum centre lands
        on ``centroid_global`` by construction (``origin = centroid_global -
        centre``).
        """
        cg = record["centroid_global"]
        ctr = record["centre"]
        return (cg[0] - ctr[0], cg[1] - ctr[1])

    def _add_quiver_trace(self, fig) -> None:
        """Draw confidence-coded local fiber axes inside the overall selector.

        Reads only the pre-downsampled block quiver ``(rows, cols, phi_block,
        coh_block)`` from each cached record. Blocks outside
        ``symmetric_radius`` are excluded because those pixels do not contribute
        to the Overall metric. The stored gradient-normal axis is rotated 90°
        to show the local fiber axis. Segment half-length scales with coherence;
        three traces provide low/medium/high confidence opacity levels.
        """
        import plotly.graph_objects as go

        binned_xy: list[tuple[list[float | None], list[float | None]]] = [
            ([], []) for _ in _QUIVER_COHERENCE_BINS
        ]
        half = 0.5 * max(1, int(self.quiver_block))
        for record in self._cache.values():
            rows, cols, phi_block, coh_block = record["quiver"]
            origin_r, origin_c = self._tile_origin(record)
            centre_r, centre_c = record["centre"]
            symmetric_radius = record["radii"].get("symmetric", np.nan)
            if not np.isfinite(symmetric_radius) or symmetric_radius <= 0:
                continue
            for i in range(phi_block.shape[0]):
                for j in range(phi_block.shape[1]):
                    phi = phi_block[i, j]
                    coh = coh_block[i, j]
                    if not np.isfinite(phi) or not np.isfinite(coh):
                        continue
                    if (
                        np.hypot(
                            rows[i, j] - centre_r,
                            cols[i, j] - centre_c,
                        )
                        >= symmetric_radius
                    ):
                        continue
                    bin_idx = next(
                        (
                            idx
                            for idx, (_, lower, upper, _) in enumerate(
                                _QUIVER_COHERENCE_BINS
                            )
                            if lower <= coh < upper
                        ),
                        None,
                    )
                    if bin_idx is None:
                        continue
                    # Block centre in plate coords (x=col, y=row).
                    cx = cols[i, j] + origin_c
                    cy = rows[i, j] + origin_r
                    length = half * float(coh)
                    fiber_phi = phi + _FIBER_AXIS_OFFSET
                    dx = length * np.cos(fiber_phi)
                    dy = length * np.sin(fiber_phi)
                    xs, ys = binned_xy[bin_idx]
                    xs.extend([cx - dx, cx + dx, None])
                    ys.extend([cy - dy, cy + dy, None])
        for (name, _lower, _upper, opacity), (xs, ys) in zip(
            _QUIVER_COHERENCE_BINS,
            binned_xy,
        ):
            if not xs:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=_OI_SKY, width=1.6),
                    opacity=opacity,
                    name=f"Local fiber axis · {name}",
                    legendgroup="fiber-axes",
                    legendgrouptitle_text="Local fiber axes",
                    hoverinfo="skip",
                )
            )

    def _add_zone_ring_traces(self, fig) -> None:
        """Concentric zone-boundary circles centred at each object's inoculum.

        Draws the symmetric, core-end, dense-end and sparse-end radii (skipping
        non-finite radii) as legend-toggleable circle polygons read from the
        cached ``radii`` + ``centroid_global`` scalars.
        """
        import plotly.graph_objects as go

        ring_styles = (
            ("symmetric", "Overall selector limit", "#785EF0", "solid"),
            ("core_end", "Dense zone inner boundary", "#DC267F", "dot"),
            ("dense_end", "Dense / sparse boundary", _OI_NAVY, "dash"),
            ("sparse_end", "Sparse zone outer boundary", _OI_SKY, "dash"),
        )
        for key, name, color, dash in ring_styles:
            xs: list[float | None] = []
            ys: list[float | None] = []
            for record in self._cache.values():
                r = record["radii"].get(key, np.nan)
                if r is None or not np.isfinite(r) or r <= 0:
                    continue
                cy, cx = record["centroid_global"]
                cxs, cys = _circle_xy(cx, cy, float(r))
                xs.extend([*cxs.tolist(), None])
                ys.extend([*cys.tolist(), None])
            if not xs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=1.5, dash=dash),
                    name=name,
                    legendgroup="rings",
                    legendgrouptitle_text="Radial selectors",
                    hoverinfo="skip",
                )
            )

    def _add_mean_axis_traces(self, fig) -> None:
        """Add one centered, undirected mean fiber axis per Radial zone.

        Orientation is axial modulo 180°, so centered line segments are used
        instead of arrows. Each zone has its own color and trace. Segment length
        is proportional to concentration ``R``; Mask-variant values remain in
        the hover summary rather than being overplotted at the same centre.
        """
        import plotly.graph_objects as go

        zone_styles = (
            ("Overall", _OI_ORANGE),
            ("Dense", "#CC79A7"),
            ("Sparse", _OI_VERMILION),
        )
        for zone, color in zone_styles:
            axis_x: list[float | None] = []
            axis_y: list[float | None] = []
            for record in self._cache.values():
                cy, cx = record["centroid_global"]
                sym = record["radii"].get("symmetric", np.nan)
                scale = (
                    float(sym)
                    if sym is not None and np.isfinite(sym) and sym > 0
                    else 20.0
                )
                R, _turning, _coh, direction = record["per_zone"][
                    ("Radial", zone)
                ]
                if not np.isfinite(R) or not np.isfinite(direction):
                    continue
                half_length = 0.5 * scale * float(R)
                fiber_direction = direction + _FIBER_AXIS_OFFSET
                dx = half_length * np.cos(fiber_direction)
                dy = half_length * np.sin(fiber_direction)
                axis_x.extend([cx - dx, cx + dx, None])
                axis_y.extend([cy - dy, cy + dy, None])
            if not axis_x:
                continue
            fig.add_trace(
                go.Scatter(
                    x=axis_x,
                    y=axis_y,
                    mode="lines",
                    line=dict(color=color, width=3.0),
                    name=f"Mean fiber axis · {zone}",
                    legendgroup="mean-axes",
                    legendgrouptitle_text="Radial mean axes (length = R)",
                    visible="legendonly",
                    hoverinfo="skip",
                )
            )

    def _add_metric_hover_trace(self, fig) -> None:
        """Add invisible centroid hit targets containing all zone metrics."""
        import plotly.graph_objects as go

        def _value(value: float, digits: int) -> str:
            return f"{value:.{digits}f}" if np.isfinite(value) else "NaN"

        xs: list[float] = []
        ys: list[float] = []
        hover_text: list[str] = []
        for record in self._cache.values():
            cy, cx = record["centroid_global"]
            lines = [
                "<b>Zone metrics</b>",
                "R = parallel concentration; T = turning (deg/px); C = coherence",
                "RTilt = absolute tilt from radial (deg); "
                "OutT = outward radial turning (deg/px); "
                "Support = reliable sector fraction (QC)",
            ]
            for variant in _VARIANTS:
                lines.append(f"<b>{variant} selector</b>")
                for zone in _ZONES:
                    R, turning, coherence, _direction = record["per_zone"][
                        (variant, zone)
                    ]
                    lines.append(
                        f"{zone}: R={_value(R, 3)}, "
                        f"T={_value(turning, 4)}, C={_value(coherence, 3)}"
                    )
            lines.append("<b>Detected structure · radial-relative</b>")
            for zone in _ZONES:
                radial_tilt, radial_turning, radial_support = record[
                    "radial_relative"
                ][zone]
                lines.append(
                    f"{zone}: RTilt={_value(radial_tilt, 3)}, "
                    f"OutT={_value(radial_turning, 4)}, "
                    f"Support={_value(radial_support, 3)}"
                )
            lines.append(
                f"<b>Long range · {self.long_range_lag:g} px ring lag</b>"
            )
            for region in (*_ZONES, "DenseToSparse"):
                magnitude, signed, support = record["long_range"][region]
                lines.append(
                    f"{region}: |Δ|={_value(magnitude, 2)}°, "
                    f"signed Δ={_value(signed, 2)}°, "
                    f"Support={_value(support, 3)}"
                )
            xs.append(float(cx))
            ys.append(float(cy))
            hover_text.append("<br>".join(lines))
        if not xs:
            return
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=16, color="rgba(0, 0, 0, 0.01)"),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="Zone metrics (hover centres)",
                showlegend=False,
            )
        )

    def dashboard(self, image=None, show: bool = True):
        """Composed notebook diagnostic (returns a single ``go.Figure``).

        Stacks three vertically-arranged panels: the :meth:`inspect` overview,
        a recomputed coherence heatmap, and a per-zone concentration/turning
        summary table. Calls :meth:`measure` first when the compact cache is
        empty or was built for a different image.

        Args:
            image: Detected Image to render. If *None*, the image cached by the
                most recent :meth:`measure` call is reused.
            show: When *True*, call ``fig.show()`` before returning (best-effort;
                swallowed outside a display context). Defaults to *True*.

        Returns:
            A single composed ``plotly.graph_objects.Figure`` stacking the three
            panels vertically.

        Examples:
            >>> from phenotypic.data import load_synth_filamentous_plate
            >>> from phenotypic.measure import MeasureOrientationZones
            >>> op = MeasureOrientationZones()
            >>> fig = op.dashboard(load_synth_filamentous_plate(), show=False)
            >>> any(getattr(tr, "type", None) == "table" for tr in fig.data)
            True
        """
        if image is None:
            image = self._require_cache_image()
        if (
            not self._cache
            or self._cache_image is not image
            or self._cache_signature != self.model_dump_json()
        ):
            self.measure(image)
        report = _OrientationZonesReport(self, image, self._cache)
        fig = report.dash()
        if show:
            try:
                fig.show()
            except Exception:  # pragma: no cover - display-context dependent
                pass
        return fig


class _OrientationZonesReport(FigureProvider):
    """Transient control-free FigureProvider composing the orientation diagnostic.

    Holds a reference to the owning :class:`MeasureOrientationZones`, the subject
    image, and the operator's compact cache. Overrides :meth:`dash` (the
    ``GridFitReport`` pattern) because the base composer builds a uniform ``xy``
    subplot grid that cannot host the ``go.Table`` summary panel. Discard after
    rendering.
    """

    def __init__(
        self, op: "MeasureOrientationZones", image, cache: dict
    ) -> None:
        self._op = op
        self._image = image
        self._cache = cache

    @figure(title="Orientation-field overlay")
    def _panel_overview(self):
        """Panel A: the saveable inspect() overview (legend layers flattened)."""
        return self._op.inspect(self._image, for_save=True)

    @figure(title="Coherence map")
    def _panel_coherence(self):
        """Panel B: the coherence heatmap.

        Recomputed on demand via ``_coherence_canvas`` (the lean cache holds no
        full-resolution coherence) and discarded — costs compute, not memory.
        """
        import plotly.graph_objects as go

        canvas = self._op._coherence_canvas(self._image)
        fig = go.Figure(
            go.Heatmap(
                z=canvas,
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                colorbar=dict(title="C"),
            )
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    @figure(title="Per-zone absolute & radial-relative orientation")
    def _panel_summary(self):
        """Panel C: table of absolute and radial-relative zone measurements.

        One row per ``(Variant, Zone)``, aggregated across objects as
        ``np.nanmean`` over the cached per-zone scalars. Requires the custom
        :meth:`dash` override (the base composer cannot host a ``go.Table``).
        """
        import plotly.graph_objects as go

        rows: list[tuple[str, str, str, str, str, str]] = []
        for variant in _VARIANTS:
            for zone in _ZONES:
                if self._cache:
                    conc_vals = [
                        rec["per_zone"][(variant, zone)][0]
                        for rec in self._cache.values()
                    ]
                    turn_vals = [
                        rec["per_zone"][(variant, zone)][1]
                        for rec in self._cache.values()
                    ]
                    conc = _safe_nanmean(conc_vals)
                    turn = _safe_nanmean(turn_vals)
                    if variant == "Mask":
                        radial_tilt = _safe_nanmean(
                            [
                                rec["radial_relative"][zone][0]
                                for rec in self._cache.values()
                            ]
                        )
                        radial_turn = _safe_nanmean(
                            [
                                rec["radial_relative"][zone][1]
                                for rec in self._cache.values()
                            ]
                        )
                        radial_support = _safe_nanmean(
                            [
                                rec["radial_relative"][zone][2]
                                for rec in self._cache.values()
                            ]
                        )
                    else:
                        radial_tilt = radial_turn = radial_support = np.nan
                else:
                    conc = turn = radial_tilt = radial_turn = (
                        radial_support
                    ) = np.nan
                rows.append(
                    (
                        f"{variant} · {zone}",
                        f"{conc:.3f}",
                        f"{turn:.4f}",
                        f"{radial_tilt:.3f}"
                        if np.isfinite(radial_tilt)
                        else "",
                        f"{radial_turn:.4f}"
                        if np.isfinite(radial_turn)
                        else "",
                        f"{radial_support:.3f}"
                        if np.isfinite(radial_support)
                        else "",
                    )
                )
        header = [
            "Variant · Zone",
            "Concentration (R)",
            "Turning (deg/px)",
            "Radial tilt (deg)",
            "Outward turning (deg/px)",
            "Reliable-sector support (QC)",
        ]
        cols = list(zip(*rows)) if rows else [(), (), (), (), (), ()]
        return go.Figure(
            go.Table(
                header=dict(values=header),
                cells=dict(values=[list(c) for c in cols]),
            )
        )

    def dash(self, subject=None):
        """Compose the three panels into one stacked ``go.Figure``.

        Mirrors :meth:`GridFitReport.dash`: render each ``@figure`` spec, detect
        table vs xy panels, build ``make_subplots`` with matching per-row
        ``specs``, transfer traces, carry the overview panel's shapes/annotations
        (zone rings and R/turning badges are shapes/annotations the generic
        trace-copy would drop), and apply the house theme.

        Args:
            subject: Unused (this helper holds its own state); accepted only to
                match the :meth:`FigureProvider.dash` signature.

        Returns:
            A single themed ``plotly.graph_objects.Figure``.
        """
        from plotly.subplots import make_subplots

        from phenotypic.sdk_.viz.figures._theme import apply_theme

        specs = self.iter_figures()
        rendered = [self._render_spec(spec) for spec in specs]
        is_table = [
            bool(fig.data) and fig.data[0].type == "table" for fig in rendered
        ]
        row_specs = [
            [{"type": "table"}] if tbl else [{"type": "xy"}]
            for tbl in is_table
        ]
        composed = make_subplots(
            rows=len(specs),
            cols=1,
            subplot_titles=[s.title for s in specs],
            specs=row_specs,
            vertical_spacing=0.06,
        )
        # ``xy_row`` counts cartesian panels: a table cell creates no x/y axis,
        # so the Nth xy panel owns axis number N (mirrors GridFitReport.dash).
        xy_row = 0
        for row, (sub, tbl) in enumerate(zip(rendered, is_table), start=1):
            for trace in sub.data:
                composed.add_trace(trace, row=row, col=1)
            if tbl:
                continue
            xy_row += 1
            # Carry the standalone panel's shapes and explanatory annotations
            # onto this subplot.
            for shape in sub.layout.shapes:
                composed.add_shape(shape.to_plotly_json(), row=row, col=1)
            axis_suffix = "" if xy_row == 1 else str(xy_row)
            for ann in sub.layout.annotations:
                payload = ann.to_plotly_json()
                for key, axis in (("xref", "x"), ("yref", "y")):
                    ref = payload.get(key, "")
                    if ref == "paper":
                        payload[key] = f"{axis}{axis_suffix} domain"
                    elif ref.startswith(axis):
                        suffix = " domain" if ref.endswith(" domain") else ""
                        payload[key] = f"{axis}{axis_suffix}{suffix}"
                composed.add_annotation(payload)
        composed.update_layout(
            height=420 * len(specs),
            title_text="Orientation-Field Diagnostics",
        )
        return apply_theme(composed)


def _safe_nanmean(values) -> float:
    """``np.nanmean`` that returns NaN (not a warning) for an all-NaN input."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")
