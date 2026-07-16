"""Branch-tracking-free orientation sampling on concentric rings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label as connected_components
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class LiteralSkeletonRingCrossing:
    """One connected skeleton intersection with one sampled radial ring.

    Attributes:
        point_id: Stable zero-based identifier within one transform.
        ring_index: Zero-based index into the transform radii.
        radius: Ring-center radius in pixels.
        row: Coherence-weighted crossing row coordinate.
        col: Coherence-weighted crossing column coordinate.
        anchor_row: Skeleton-pixel row nearest the weighted crossing center.
        anchor_col: Skeleton-pixel column nearest the weighted crossing center.
        fiber_axis: Coherence-weighted axial fiber orientation in radians.
        radial_tilt: Signed radial-relative axial tilt in radians.
        coherence: Mean local coherence across crossing pixels.
        resultant: Doubled-angle axial resultant across crossing pixels.
        pixel_count: Number of skeleton pixels in the crossing component.
    """

    point_id: int
    ring_index: int
    radius: float
    row: float
    col: float
    anchor_row: int
    anchor_col: int
    fiber_axis: float
    radial_tilt: float
    coherence: float
    resultant: float
    pixel_count: int


@dataclass(frozen=True)
class LiteralSkeletonRingCrossingTransform:
    """Literal skeleton-ring crossing evidence for one object.

    Attributes:
        crossings: Ring-ordered crossing records.
        reliable_skeleton: Skeleton pixels accepted by the selector and
            coherence threshold.
        radii: Sampled ring-center radii in pixels.
        center: Inoculum center as ``(row, col)``.
        crossing_half_width: Radial half-width used to rasterize each ring.
    """

    crossings: tuple[LiteralSkeletonRingCrossing, ...]
    reliable_skeleton: NDArray[np.bool_]
    radii: NDArray[np.float64]
    center: tuple[float, float]
    crossing_half_width: float


@dataclass(frozen=True)
class LiteralCrossingRingProfile:
    """Equal-crossing outward orientation profile.

    Attributes:
        radii: Sampled ring-center radii in pixels.
        consensus_tilt: Equal-crossing axial consensus at each ring in
            radians. Unsupported rings contain ``NaN``.
        resultant: Doubled-angle resultant of crossing tilts at each eligible
            ring. Rings below ``minimum_points`` contain ``NaN``.
        crossing_count: Number of literal crossings observed at each ring.
        contiguous_change: Seam-safe accumulated consensus change in radians.
            Each contiguous supported run starts at zero. Gaps and exact
            90-degree changes break a run.
        run_id: Zero-based contiguous-run identifier. Unsupported rings use
            ``-1``.
    """

    radii: NDArray[np.float64]
    consensus_tilt: NDArray[np.float64]
    resultant: NDArray[np.float64]
    crossing_count: NDArray[np.int64]
    contiguous_change: NDArray[np.float64]
    run_id: NDArray[np.int64]

    @property
    def supported(self) -> NDArray[np.bool_]:
        """Return rings with a finite contiguous-change state."""
        return np.isfinite(self.contiguous_change)

    @property
    def raw_peak(self) -> float:
        """Return the largest absolute within-run change in radians."""
        supported = self.supported
        if not supported.any():
            return float("nan")
        return float(np.max(np.abs(self.contiguous_change[supported])))


def _axial_difference(outer: float, inner: float) -> float:
    """Return a signed axial difference in ``[-pi/2, pi/2]``."""
    difference = outer - inner
    return float(
        0.5
        * np.arctan2(
            np.sin(2.0 * difference),
            np.cos(2.0 * difference),
        )
    )


def _validated_field(
    value: NDArray[np.generic],
    *,
    name: str,
    shape: tuple[int, int] | None = None,
    boolean: bool = False,
) -> np.ndarray:
    """Return one validated two-dimensional field."""
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if boolean and array.dtype != np.bool_:
        array = array.astype(bool, copy=False)
    return array


def literal_skeleton_ring_crossings(
    object_mask: NDArray[np.bool_],
    fiber_axis: NDArray[np.floating],
    coherence: NDArray[np.floating],
    distance_map: NDArray[np.floating],
    center: tuple[float, float],
    radii: NDArray[np.floating],
    *,
    selector: NDArray[np.bool_] | None = None,
    minimum_coherence: float = 0.15,
    crossing_half_width: float = 1.5,
    minimum_crossing_resultant: float = 0.15,
) -> LiteralSkeletonRingCrossingTransform:
    """Collect orientation samples where a skeleton intersects radial rings.

    The object mask is skeletonized once. For every requested ring center,
    reliable skeleton pixels within ``crossing_half_width`` are grouped by
    8-connectivity. Each connected group contributes one coherence-weighted
    axial orientation sample. The function never infers correspondence between
    rings or identifies individual biological branches.

    Args:
        object_mask: Binary mask of one detected colony or object.
        fiber_axis: Local axial fiber orientation in radians. Values differing
            by ``pi`` represent the same axis.
        coherence: Local orientation coherence in ``[0, 1]``.
        distance_map: Pixel distance from ``center``.
        center: Inoculum center as ``(row, col)``.
        radii: Strictly increasing ring-center radii in pixels.
        selector: Optional binary eligibility mask, such as the object outside
            an inoculum exclusion radius. Defaults to ``object_mask``.
        minimum_coherence: Smallest accepted local coherence.
        crossing_half_width: Radial half-width of each rasterized ring.
        minimum_crossing_resultant: Smallest accepted within-crossing axial
            resultant.

    Returns:
        Literal crossing records and the reliable measurement skeleton.

    Raises:
        ValueError: If arrays, center, radii, or thresholds are invalid.
    """
    mask = _validated_field(object_mask, name="object_mask", boolean=True)
    shape = mask.shape
    axes = _validated_field(fiber_axis, name="fiber_axis", shape=shape)
    coherence_field = _validated_field(
        coherence, name="coherence", shape=shape
    )
    distances = _validated_field(
        distance_map, name="distance_map", shape=shape
    )
    eligible = (
        mask
        if selector is None
        else _validated_field(
            selector, name="selector", shape=shape, boolean=True
        )
    )
    center_array = np.asarray(center, dtype=float)
    if center_array.shape != (2,) or not np.isfinite(center_array).all():
        raise ValueError("center must contain two finite coordinates")
    sampled_radii = np.asarray(radii, dtype=float)
    if sampled_radii.ndim != 1 or sampled_radii.size < 1:
        raise ValueError("radii must be a non-empty 1-D array")
    if not np.isfinite(sampled_radii).all() or np.any(sampled_radii < 0.0):
        raise ValueError("radii must contain finite nonnegative values")
    if np.any(np.diff(sampled_radii) <= 0.0):
        raise ValueError("radii must be strictly increasing")
    finite_coherence = coherence_field[np.isfinite(coherence_field)]
    if finite_coherence.size and (
        np.min(finite_coherence) < 0.0 or np.max(finite_coherence) > 1.0
    ):
        raise ValueError("finite coherence values must be in [0, 1]")
    for name, value in (
        ("minimum_coherence", minimum_coherence),
        ("minimum_crossing_resultant", minimum_crossing_resultant),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not np.isfinite(crossing_half_width) or crossing_half_width <= 0.0:
        raise ValueError("crossing_half_width must be finite and > 0")

    full_skeleton = np.asarray(skeletonize(mask), dtype=bool)
    reliable_skeleton = (
        full_skeleton
        & eligible
        & np.isfinite(axes)
        & np.isfinite(coherence_field)
        & (coherence_field >= minimum_coherence)
        & np.isfinite(distances)
    )
    center_row, center_col = (float(value) for value in center_array)
    structure: NDArray[np.uint8] = np.ones((3, 3), dtype=np.uint8)
    crossings: list[LiteralSkeletonRingCrossing] = []
    for ring_index, radius in enumerate(sampled_radii):
        ring_pixels = reliable_skeleton & (
            np.abs(distances - radius) <= crossing_half_width
        )
        components = np.zeros(ring_pixels.shape, dtype=np.int32)
        count = cast(
            int,
            connected_components(
                ring_pixels,
                structure=structure,
                output=components,
            ),
        )
        for component in range(1, count + 1):
            rows, cols = np.nonzero(components == component)
            if rows.size == 0:
                continue
            weights = np.asarray(coherence_field[rows, cols], dtype=float)
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue
            component_axes = np.asarray(axes[rows, cols], dtype=float)
            mean_cosine = float(
                np.sum(weights * np.cos(2.0 * component_axes)) / weight_sum
            )
            mean_sine = float(
                np.sum(weights * np.sin(2.0 * component_axes)) / weight_sum
            )
            resultant = float(np.hypot(mean_cosine, mean_sine))
            if resultant < minimum_crossing_resultant:
                continue
            crossing_axis = 0.5 * np.arctan2(mean_sine, mean_cosine)
            row = float(np.sum(weights * rows) / weight_sum)
            col = float(np.sum(weights * cols) / weight_sum)
            anchor_index = int(
                np.argmin(np.square(rows - row) + np.square(cols - col))
            )
            polar_angle = float(np.arctan2(row - center_row, col - center_col))
            crossings.append(
                LiteralSkeletonRingCrossing(
                    point_id=len(crossings),
                    ring_index=ring_index,
                    radius=float(radius),
                    row=row,
                    col=col,
                    anchor_row=int(rows[anchor_index]),
                    anchor_col=int(cols[anchor_index]),
                    fiber_axis=float(crossing_axis),
                    radial_tilt=_axial_difference(crossing_axis, polar_angle),
                    coherence=float(np.mean(weights)),
                    resultant=resultant,
                    pixel_count=int(rows.size),
                )
            )

    return LiteralSkeletonRingCrossingTransform(
        crossings=tuple(crossings),
        reliable_skeleton=reliable_skeleton,
        radii=sampled_radii.copy(),
        center=(center_row, center_col),
        crossing_half_width=float(crossing_half_width),
    )


def literal_crossing_ring_profile(
    transform: LiteralSkeletonRingCrossingTransform,
    *,
    minimum_points: int = 3,
    minimum_resultant: float = 0.15,
    ambiguity_tolerance: float = 1e-12,
) -> LiteralCrossingRingProfile:
    """Summarize literal crossings without following individual branches.

    Every crossing contributes one equal vote to its ring. Consecutive
    supported consensuses accumulate seam-safe axial changes. Missing rings
    and exact 90-degree changes break the sequence; the next supported ring
    starts a new zero-relative run.

    Args:
        transform: Literal crossing transform to summarize.
        minimum_points: Smallest number of crossings accepted at one ring.
        minimum_resultant: Smallest accepted ring-level axial resultant.
        ambiguity_tolerance: Absolute radian tolerance for detecting an exact
            90-degree inter-ring change.

    Returns:
        Equal-crossing ring consensus and contiguous outward-change profile.

    Raises:
        ValueError: If profile thresholds are invalid.
    """
    if (
        isinstance(minimum_points, (bool, np.bool_))
        or not isinstance(minimum_points, (int, np.integer))
        or minimum_points < 1
    ):
        raise ValueError("minimum_points must be an integer >= 1")
    if (
        not np.isfinite(minimum_resultant)
        or not 0.0 <= minimum_resultant <= 1.0
    ):
        raise ValueError("minimum_resultant must be finite and in [0, 1]")
    if not np.isfinite(ambiguity_tolerance) or ambiguity_tolerance < 0.0:
        raise ValueError("ambiguity_tolerance must be finite and >= 0")

    n_rings = transform.radii.size
    consensus = np.full(n_rings, np.nan, dtype=float)
    resultants = np.full(n_rings, np.nan, dtype=float)
    counts = np.zeros(n_rings, dtype=np.int64)
    by_ring: dict[int, list[float]] = {}
    for crossing in transform.crossings:
        if not 0 <= crossing.ring_index < n_rings:
            raise ValueError("crossing ring_index is outside transform radii")
        by_ring.setdefault(crossing.ring_index, []).append(
            crossing.radial_tilt
        )
    for ring_index, values in by_ring.items():
        tilts = np.asarray(values, dtype=float)
        counts[ring_index] = tilts.size
        if tilts.size < minimum_points:
            continue
        mean_cosine = float(np.mean(np.cos(2.0 * tilts)))
        mean_sine = float(np.mean(np.sin(2.0 * tilts)))
        resultant = float(np.hypot(mean_cosine, mean_sine))
        resultants[ring_index] = resultant
        if resultant >= minimum_resultant:
            consensus[ring_index] = 0.5 * np.arctan2(mean_sine, mean_cosine)

    contiguous_change = np.full(n_rings, np.nan, dtype=float)
    run_id = np.full(n_rings, -1, dtype=np.int64)
    previous_ring: int | None = None
    next_run_id = 0
    for ring_index, angle in enumerate(consensus):
        if not np.isfinite(angle):
            previous_ring = None
            continue
        if previous_ring is None:
            contiguous_change[ring_index] = 0.0
            run_id[ring_index] = next_run_id
            next_run_id += 1
            previous_ring = ring_index
            continue
        step = _axial_difference(float(angle), float(consensus[previous_ring]))
        if np.isclose(
            abs(step),
            0.5 * np.pi,
            atol=ambiguity_tolerance,
            rtol=0.0,
        ):
            previous_ring = None
            continue
        contiguous_change[ring_index] = contiguous_change[previous_ring] + step
        run_id[ring_index] = run_id[previous_ring]
        previous_ring = ring_index

    return LiteralCrossingRingProfile(
        radii=transform.radii.copy(),
        consensus_tilt=consensus,
        resultant=resultants,
        crossing_count=counts,
        contiguous_change=contiguous_change,
        run_id=run_id,
    )
