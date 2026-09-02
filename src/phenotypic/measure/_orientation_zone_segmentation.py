"""Pure Method B change-point segmentation for branch-orientation zones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from phenotypic.sdk_.orientation_fields import (
    LiteralCrossingRingProfile,
    LiteralSkeletonRingCrossingTransform,
    literal_crossing_ring_profile,
    literal_skeleton_ring_crossings,
)
from phenotypic.util._orientation_field import orientation_field


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_RELIABLE_PIXEL_COHERENCE = 0.15
_CROSSING_HALF_WIDTH = 1.5
_CROSSING_RESULTANT = 0.15


@dataclass(frozen=True)
class OrientationChangePointParams:
    """Parameters for one center-origin Method B fit."""

    sigma_d: float = 1.5
    sigma_i: float = 4.0
    ring_width: float = 8.0
    outer_zone_percentile: float = 100.0
    minimum_segment: int = 4
    min_crossings: int = 3
    min_resultant: float = 0.15
    min_ring_coherence: float = 0.15
    support_weight: float = 4.0
    outer_support_margin: float = 0.0
    maximum_gap: int = 0


@dataclass(frozen=True)
class OrientationRadialProfile:
    """Method B center-origin ring features and orientation support."""

    radii: FloatArray
    continuous_features: FloatArray
    crossing_count: NDArray[np.int64]
    crossing_resultant: FloatArray
    ring_coherence: FloatArray
    raw_support: BoolArray
    bridged_support: BoolArray


@dataclass(frozen=True)
class OrientationZoneResult:
    """Resolved CoreZone, DenseZone, and SparseZone geometry."""

    core_zone_radius: float
    dense_radius: float
    outer_radius: float
    full_extent_radius: float
    requested_percentile: float
    retained_mask_fraction: float
    supported_fraction: float
    objective: float
    ring_count: int
    method_used: Literal["exact", "collapsed", "missing"]
    failure_reason: str

    @property
    def method_code(self) -> int:
        """Return the stable numeric status code used by diagnostics."""
        return {"exact": 1, "collapsed": 2, "missing": 4}[self.method_used]

    @property
    def zones_computed(self) -> bool:
        """Return whether usable zone boundaries were resolved."""
        return self.method_used != "missing"


@dataclass(frozen=True)
class OrientationAnalysisContext:
    """Ephemeral arrays reused by orientation-zone measurement consumers."""

    signal: FloatArray
    object_mask: BoolArray
    center: tuple[float, float]
    distance_map: FloatArray
    phi: FloatArray
    coherence: FloatArray
    gradient: FloatArray
    transform: LiteralSkeletonRingCrossingTransform
    zoning_profile: LiteralCrossingRingProfile
    measurement_profile: LiteralCrossingRingProfile
    radial_profile: OrientationRadialProfile


@dataclass(frozen=True)
class OrientationZoneFit:
    """Method B result paired with reusable orientation intermediates."""

    result: OrientationZoneResult
    context: OrientationAnalysisContext | None


def distance_from_center(
    shape: tuple[int, int], center: tuple[float, float]
) -> FloatArray:
    """Return Euclidean distance from one row-column center."""
    rows, cols = np.indices(shape, dtype=np.float64)
    return np.hypot(rows - center[0], cols - center[1])


def selected_outer_radius(
    mask: BoolArray,
    distance_map: FloatArray,
    percentile: float,
) -> tuple[float, float, float]:
    """Return configured radius, full extent, and retained mask fraction."""
    distances = np.asarray(distance_map, dtype=np.float64)[mask]
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return float("nan"), float("nan"), float("nan")
    full_extent = float(np.max(distances))
    outer = (
        full_extent
        if percentile == 100.0
        else float(np.percentile(distances, percentile, method="linear"))
    )
    if not np.isfinite(outer) or outer <= 0.0:
        return float("nan"), full_extent, float("nan")
    retained = float(
        np.count_nonzero(mask & (distance_map < np.nextafter(outer, np.inf)))
        / np.count_nonzero(mask)
    )
    return outer, full_extent, retained


def robust_scale_signal(
    signal: FloatArray, mask: BoolArray
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Return statistical scaling, derivative fill, and source validity."""
    source = np.asarray(signal, dtype=np.float64)
    validity = np.asarray(mask, dtype=bool) & np.isfinite(source)
    population = source[validity]
    if population.size == 0:
        zeros = np.zeros_like(source, dtype=np.float64)
        return zeros, zeros, validity
    low, high = np.percentile(population, [2.0, 98.0], method="linear")
    if not np.isfinite(high - low) or high <= low:
        zeros = np.zeros_like(source, dtype=np.float64)
        return zeros, zeros, validity
    scaled = np.clip((source - low) / (high - low), 0.0, 1.0)
    fill = float(np.median(scaled[validity]))
    derivative_scaled = np.where(np.isfinite(scaled), scaled, fill)
    return scaled, derivative_scaled, validity


def robust_standardize(matrix: FloatArray) -> FloatArray:
    """Median-impute and robust-standardize feature columns."""
    result = np.asarray(matrix, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        values = result[:, column]
        finite = np.isfinite(values)
        fill = float(np.median(values[finite])) if finite.any() else 0.0
        values[~finite] = fill
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = max(
            1.4826 * mad,
            float(np.std(values)),
            float(np.finfo(np.float64).eps),
        )
        result[:, column] = (values - median) / scale
    return result


def bridge_short_gaps(support: BoolArray, maximum_gap: int) -> BoolArray:
    """Bridge bounded interior unsupported runs without filling edge gaps."""
    result = np.asarray(support, dtype=bool).copy()
    if maximum_gap <= 0 or result.size == 0:
        return result
    padded = np.concatenate(([True], result, [True]))
    starts = np.flatnonzero(np.diff(padded.astype(np.int8)) == -1)
    stops = np.flatnonzero(np.diff(padded.astype(np.int8)) == 1)
    for start, stop in zip(starts, stops, strict=True):
        if start > 0 and stop < result.size and stop - start <= maximum_gap:
            result[start:stop] = True
    return result


def _segment_sse(matrix: FloatArray, start: int, stop: int) -> float:
    segment = matrix[start:stop]
    if segment.size == 0:
        return float("inf")
    return float(np.square(segment - segment.mean(axis=0)).sum())


def exact_two_change_points(
    matrix: FloatArray,
    support: BoolArray,
    minimum_segment: int,
    outer_support_margin: float,
) -> tuple[float, int, int] | None:
    """Return the deterministic minimum-SSE two-change solution."""
    ring_count = int(matrix.shape[0])
    if ring_count < 3 * minimum_segment:
        return None
    best: tuple[float, int, int] | None = None
    for first in range(
        minimum_segment, ring_count - 2 * minimum_segment + 1
    ):
        support_gain = float(support[first:].mean()) - float(
            support[:first].mean()
        )
        if support_gain < outer_support_margin:
            continue
        for second in range(
            first + minimum_segment, ring_count - minimum_segment + 1
        ):
            if not support[first:second].any() or not support[second:].any():
                continue
            candidate = (
                _segment_sse(matrix, 0, first)
                + _segment_sse(matrix, first, second)
                + _segment_sse(matrix, second, ring_count),
                first,
                second,
            )
            if best is None or candidate < best:
                best = candidate
    return best


def collapsed_one_change_point(
    support: BoolArray, minimum_segment: int
) -> tuple[float, int] | None:
    """Return the deterministic one-change fallback on unresolved evidence."""
    ring_count = int(support.size)
    if ring_count < 2 * minimum_segment:
        return None
    evidence = (1.0 - support.astype(np.float64))[:, None]
    best: tuple[float, int] | None = None
    for boundary in range(minimum_segment, ring_count - minimum_segment + 1):
        if not support[boundary:].any():
            continue
        candidate = (
            _segment_sse(evidence, 0, boundary)
            + _segment_sse(evidence, boundary, ring_count),
            boundary,
        )
        if best is None or candidate < best:
            best = candidate
    return best


def _missing_result(
    params: OrientationChangePointParams,
    *,
    reason: str,
    outer_radius: float = float("nan"),
    full_extent_radius: float = float("nan"),
    retained_mask_fraction: float = float("nan"),
    supported_fraction: float = float("nan"),
    ring_count: int = 0,
) -> OrientationZoneResult:
    return OrientationZoneResult(
        core_zone_radius=float("nan"),
        dense_radius=float("nan"),
        outer_radius=outer_radius,
        full_extent_radius=full_extent_radius,
        requested_percentile=params.outer_zone_percentile,
        retained_mask_fraction=retained_mask_fraction,
        supported_fraction=supported_fraction,
        objective=float("nan"),
        ring_count=ring_count,
        method_used="missing",
        failure_reason=reason,
    )


def _resolve_radial_profile(
    profile: OrientationRadialProfile,
    params: OrientationChangePointParams,
    *,
    outer_radius: float,
    full_extent_radius: float,
    retained_mask_fraction: float,
) -> tuple[OrientationZoneResult, OrientationRadialProfile]:
    """Resolve boundaries from precomputed radial evidence.

    The image-derived profile is independent of the support thresholds and
    change-point penalties. Keeping this partition step separate lets
    evaluation reuse one expensive tensor/skeleton profile across a declared
    parameter grid while production follows the identical solver path.
    """
    crossing_resultant = np.nan_to_num(
        profile.crossing_resultant, nan=-np.inf
    )
    raw_support = (
        (profile.crossing_count >= params.min_crossings)
        & (crossing_resultant >= params.min_resultant)
        & (
            np.nan_to_num(profile.ring_coherence, nan=-np.inf)
            >= params.min_ring_coherence
        )
    )
    support = bridge_short_gaps(raw_support, params.maximum_gap)
    matrix = np.column_stack(
        (
            robust_standardize(profile.continuous_features),
            support * params.support_weight,
        )
    )
    resolved_profile = OrientationRadialProfile(
        radii=profile.radii.copy(),
        continuous_features=profile.continuous_features.copy(),
        crossing_count=profile.crossing_count.copy(),
        crossing_resultant=profile.crossing_resultant.copy(),
        ring_coherence=profile.ring_coherence.copy(),
        raw_support=raw_support.copy(),
        bridged_support=support.copy(),
    )
    supported_fraction = float(np.mean(support)) if support.size else 0.0
    exact = exact_two_change_points(
        matrix,
        support,
        params.minimum_segment,
        params.outer_support_margin,
    )
    if exact is not None:
        objective, first, second = exact
        core_zone_radius = float(
            profile.radii[first] - params.ring_width / 2.0
        )
        dense_radius = float(
            profile.radii[second] - params.ring_width / 2.0
        )
        method: Literal["exact", "collapsed"] = "exact"
        reason = "none"
    else:
        collapsed = collapsed_one_change_point(support, params.minimum_segment)
        if collapsed is None:
            return (
                _missing_result(
                    params,
                    reason="no_supported_change_point",
                    outer_radius=outer_radius,
                    full_extent_radius=full_extent_radius,
                    retained_mask_fraction=retained_mask_fraction,
                    supported_fraction=supported_fraction,
                    ring_count=int(profile.radii.size),
                ),
                resolved_profile,
            )
        objective, boundary = collapsed
        core_zone_radius = dense_radius = float(
            profile.radii[boundary] - params.ring_width / 2.0
        )
        method = "collapsed"
        reason = "no_valid_two_change_candidate"

    core_zone_radius = float(np.clip(core_zone_radius, 0.0, outer_radius))
    dense_radius = float(
        np.clip(dense_radius, core_zone_radius, outer_radius)
    )
    return (
        OrientationZoneResult(
            core_zone_radius=core_zone_radius,
            dense_radius=dense_radius,
            outer_radius=outer_radius,
            full_extent_radius=full_extent_radius,
            requested_percentile=params.outer_zone_percentile,
            retained_mask_fraction=retained_mask_fraction,
            supported_fraction=supported_fraction,
            objective=float(objective),
            ring_count=int(profile.radii.size),
            method_used=method,
            failure_reason=reason,
        ),
        resolved_profile,
    )


def fit_orientation_zones(
    object_mask: BoolArray,
    signal: FloatArray,
    center: tuple[float, float],
    params: OrientationChangePointParams,
) -> OrientationZoneFit:
    """Fit canonical Method B zones and retain reusable orientation evidence."""
    mask = np.asarray(object_mask, dtype=bool)
    source = np.asarray(signal, dtype=np.float64)
    if mask.ndim != 2 or source.shape != mask.shape or not mask.any():
        return OrientationZoneFit(
            _missing_result(params, reason="invalid_object_mask"), None
        )
    distance_map = distance_from_center(mask.shape, center)
    outer, full_extent, retained = selected_outer_radius(
        mask, distance_map, params.outer_zone_percentile
    )
    if not np.isfinite(outer):
        return OrientationZoneFit(
            _missing_result(
                params,
                reason="invalid_outer_extent",
                full_extent_radius=full_extent,
            ),
            None,
        )

    inclusive_outer = np.nextafter(outer, np.inf)
    selected_mask = mask & (distance_map < inclusive_outer)
    ring_count = max(1, int(np.ceil(outer / params.ring_width)))
    radii = (
        np.arange(ring_count, dtype=np.float64) + 0.5
    ) * params.ring_width

    scaled, derivative_scaled, source_validity = robust_scale_signal(source, mask)
    gradient_y, gradient_x = np.gradient(derivative_scaled)
    edge_energy = np.hypot(gradient_x, gradient_y)
    phi, coherence, gradient = orientation_field(
        derivative_scaled, params.sigma_d, params.sigma_i
    )
    fiber_axis = (phi + np.pi / 2.0 + np.pi / 2.0) % np.pi - np.pi / 2.0
    rows, cols = np.indices(mask.shape, dtype=np.float64)
    azimuth = np.arctan2(rows - center[0], cols - center[1])
    radial_tilt = 0.5 * np.arctan2(
        np.sin(2.0 * (fiber_axis - azimuth)),
        np.cos(2.0 * (fiber_axis - azimuth)),
    )

    transform = literal_skeleton_ring_crossings(
        mask,
        fiber_axis,
        coherence,
        distance_map,
        center,
        radii,
        selector=selected_mask,
        minimum_coherence=_RELIABLE_PIXEL_COHERENCE,
        crossing_half_width=_CROSSING_HALF_WIDTH,
        minimum_crossing_resultant=_CROSSING_RESULTANT,
    )
    zoning_profile = literal_crossing_ring_profile(
        transform, minimum_points=1, minimum_resultant=0.0
    )
    measurement_profile = literal_crossing_ring_profile(
        transform,
        minimum_points=3,
        minimum_resultant=_CROSSING_RESULTANT,
    )

    features: list[list[float]] = []
    for index, radius in enumerate(radii):
        geometric_ring = (
            np.abs(distance_map - radius) <= params.ring_width / 2.0
        ) & (distance_map < inclusive_outer)
        selected = geometric_ring & mask
        valid_selected = selected & source_validity
        reliable = (
            valid_selected
            & np.isfinite(coherence)
            & (coherence >= _RELIABLE_PIXEL_COHERENCE)
        )
        if valid_selected.any():
            intensity_mean = float(np.mean(scaled[valid_selected]))
            intensity_variance = float(np.var(scaled[valid_selected]))
            mean_edge = float(np.mean(edge_energy[valid_selected]))
        else:
            intensity_mean = intensity_variance = mean_edge = float("nan")
        if reliable.any():
            mean_coherence = float(np.mean(coherence[reliable]))
            radial_tilt_resultant = float(
                abs(np.mean(np.exp(2j * radial_tilt[reliable])))
            )
        else:
            mean_coherence = radial_tilt_resultant = float("nan")
        occupancy = (
            float(selected.sum() / geometric_ring.sum())
            if geometric_ring.any()
            else float("nan")
        )
        features.append(
            [
                intensity_mean,
                intensity_variance,
                occupancy,
                mean_coherence,
                radial_tilt_resultant,
                mean_edge,
                float(zoning_profile.resultant[index]),
            ]
        )

    feature_array = np.asarray(features, dtype=np.float64)
    ring_coherence = feature_array[:, 3]
    radial_profile = OrientationRadialProfile(
        radii=radii.copy(),
        continuous_features=feature_array,
        crossing_count=zoning_profile.crossing_count.copy(),
        crossing_resultant=zoning_profile.resultant.copy(),
        ring_coherence=ring_coherence.copy(),
        raw_support=np.zeros(ring_count, dtype=bool),
        bridged_support=np.zeros(ring_count, dtype=bool),
    )
    result, radial_profile = _resolve_radial_profile(
        radial_profile,
        params,
        outer_radius=outer,
        full_extent_radius=full_extent,
        retained_mask_fraction=retained,
    )
    context = OrientationAnalysisContext(
        signal=source,
        object_mask=mask,
        center=(float(center[0]), float(center[1])),
        distance_map=distance_map,
        phi=np.asarray(phi, dtype=np.float64),
        coherence=np.asarray(coherence, dtype=np.float64),
        gradient=np.asarray(gradient, dtype=np.float64),
        transform=transform,
        zoning_profile=zoning_profile,
        measurement_profile=measurement_profile,
        radial_profile=radial_profile,
    )
    return OrientationZoneFit(result=result, context=context)
