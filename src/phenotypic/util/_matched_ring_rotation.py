"""Match nearby annular orientation cells and accumulate axial rotation."""

from __future__ import annotations

import numpy as np


def _axial_difference(outer: np.ndarray, inner: float) -> np.ndarray:
    """Return seam-safe signed axial differences in ``[-pi/2, pi/2]``."""
    difference = outer - inner
    return 0.5 * np.arctan2(
        np.sin(2.0 * difference),
        np.cos(2.0 * difference),
    )


def _wrapped_angular_difference(outer: np.ndarray, inner: float) -> np.ndarray:
    """Return signed circular differences in ``[-pi, pi]``."""
    return np.arctan2(np.sin(outer - inner), np.cos(outer - inner))


def matched_ring_cumulative_rotation_profile(
    ring_centres: np.ndarray,
    sector_orientation: np.ndarray,
    sector_resultant: np.ndarray,
    *,
    max_sector_shift: int = 2,
    reliability_weight: float = 0.25,
    max_abs_radial_tilt: float = np.deg2rad(75.0),
    allow_gap_bridging: bool = False,
    allow_restarts: bool = False,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate fiber-axis rotation along nearby matched annular cells.

    Each angular sector is a geometric seed. Its track starts at that sector's
    first reliable ring and proceeds only through the immediately following
    rings. At every step, candidates are restricted to nearby angular sectors.
    Their cost combines distance from the radial-relative orientation predictor,
    axial orientation continuity, and scale-local orientation reliability.

    The predictor follows the polar tangent relation
    ``d(alpha) / d(r) = tan(delta) / r``, where ``delta`` is the signed axial
    difference between the fiber orientation and the outward radial direction.
    Orientation rotation is accumulated with a doubled-angle difference, so a
    fiber axis and the same axis plus 180 degrees are equivalent.

    Args:
        ring_centres: Strictly increasing positive ring-centre radii, shaped
            ``(n_rings,)``.
        sector_orientation: Coherence-weighted fiber-axis means in radians,
            shaped ``(n_rings, n_sectors)``. Unsupported cells are ``NaN``.
        sector_resultant: Axial resultant in ``[0, 1]`` for each orientation
            cell, with the same shape. Unsupported cells are ``NaN``.
        max_sector_shift: Maximum circular sector-index displacement allowed
            between adjacent rings.
        reliability_weight: Nonnegative cost weight applied to ``1 - R``.
        max_abs_radial_tilt: Largest absolute fiber-to-radial tilt that may
            advance outward, in radians. Tracks terminate before the radial
            tangent predictor becomes ill-conditioned.
        allow_gap_bridging: If ``True``, scan outward past rings with no
            reliable nearby candidate. Skipped rings remain ``NaN`` and the
            predictor uses the complete radial interval to the resumed ring.
            Geometric and axial-ambiguity failures still terminate a segment.
        allow_restarts: If ``True``, a terminated seed starts a new segment at
            its next reliable cell in the original geometric seed sector. Each
            restarted segment resets cumulative rotation to zero because no
            defensible rotation can be assigned across the break.
        eps: Positive numerical tolerance for degenerate axial changes.

    Returns:
        ``(cumulative_rotation, path_sectors)``. Both arrays have shape
        ``(n_rings, n_sectors)`` where columns identify geometric seed sectors.
        Cumulative rotation is in radians and is ``NaN`` before a track starts
        or where no segment exists. With restarts enabled, one seed column may
        contain multiple zero-based segments. ``path_sectors`` contains the
        matched sector at every supported step and ``-1`` elsewhere.

    Raises:
        ValueError: If array shapes, radii, resultants, or parameters are
            invalid.
    """
    radii = np.asarray(ring_centres, dtype=np.float64)
    orientation = np.asarray(sector_orientation, dtype=np.float64)
    resultant = np.asarray(sector_resultant, dtype=np.float64)
    if radii.ndim != 1:
        raise ValueError("ring_centres must be one-dimensional")
    if orientation.ndim != 2 or orientation.shape[0] != radii.size:
        raise ValueError("sector_orientation rows must match ring_centres")
    if resultant.shape != orientation.shape:
        raise ValueError("sector_resultant must match sector_orientation")
    if radii.size and (
        not np.isfinite(radii).all()
        or np.any(radii <= 0.0)
        or np.any(np.diff(radii) <= 0.0)
    ):
        raise ValueError(
            "ring_centres must be finite, positive, and increasing"
        )
    if isinstance(max_sector_shift, bool) or not isinstance(
        max_sector_shift, (int, np.integer)
    ):
        raise ValueError("max_sector_shift must be an integer >= 0")
    if max_sector_shift < 0:
        raise ValueError("max_sector_shift must be an integer >= 0")
    if not np.isfinite(reliability_weight) or reliability_weight < 0.0:
        raise ValueError("reliability_weight must be finite and >= 0")
    if (
        not np.isfinite(max_abs_radial_tilt)
        or max_abs_radial_tilt <= 0.0
        or max_abs_radial_tilt >= np.pi / 2.0
    ):
        raise ValueError("max_abs_radial_tilt must be finite and in (0, pi/2)")
    if not isinstance(allow_gap_bridging, (bool, np.bool_)):
        raise ValueError("allow_gap_bridging must be a boolean")
    if not isinstance(allow_restarts, (bool, np.bool_)):
        raise ValueError("allow_restarts must be a boolean")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and > 0")
    finite_resultant = resultant[np.isfinite(resultant)]
    if np.any((finite_resultant < 0.0) | (finite_resultant > 1.0)):
        raise ValueError("finite sector_resultant values must be in [0, 1]")

    n_rings, n_sectors = orientation.shape
    cumulative = np.full_like(orientation, np.nan)
    path_sectors = np.full((n_rings, n_sectors), -1, dtype=np.int64)
    if n_rings == 0 or n_sectors == 0:
        return cumulative, path_sectors

    reliable = np.isfinite(orientation) & np.isfinite(resultant)
    sector_width = 2.0 * np.pi / float(n_sectors)
    sector_angles = (
        np.arange(n_sectors, dtype=np.float64) + 0.5
    ) * sector_width
    offsets = np.arange(
        -max_sector_shift, max_sector_shift + 1, dtype=np.int64
    )

    for seed_sector in range(n_sectors):
        starts = np.flatnonzero(reliable[:, seed_sector])
        if starts.size == 0:
            continue
        segment_start = int(starts[0])

        while segment_start < n_rings:
            current_ring = segment_start
            current_sector = seed_sector
            cumulative[current_ring, seed_sector] = 0.0
            path_sectors[current_ring, seed_sector] = current_sector
            failure_ring: int | None = None

            while current_ring + 1 < n_rings:
                search_start = current_ring + 1
                candidate_ring = search_start
                matched = False
                while candidate_ring < n_rings:
                    previous_orientation = orientation[
                        current_ring,
                        current_sector,
                    ]
                    if not np.isfinite(previous_orientation):
                        failure_ring = candidate_ring
                        break

                    candidate_sectors = np.unique(
                        np.mod(current_sector + offsets, n_sectors)
                    )
                    candidate_sectors = candidate_sectors[
                        reliable[candidate_ring, candidate_sectors]
                    ]
                    if candidate_sectors.size == 0:
                        if allow_gap_bridging:
                            candidate_ring += 1
                            continue
                        failure_ring = candidate_ring
                        break

                    current_alpha = sector_angles[current_sector]
                    radial_relative = float(
                        _axial_difference(
                            np.asarray([previous_orientation]),
                            current_alpha,
                        )[0]
                    )
                    if abs(radial_relative) > max_abs_radial_tilt:
                        failure_ring = candidate_ring
                        break
                    radial_ratio = radii[candidate_ring] / radii[current_ring]
                    predicted_step = np.tan(radial_relative) * np.log(
                        radial_ratio
                    )
                    maximum_predicted_step = (
                        max_sector_shift + 0.5
                    ) * sector_width
                    if abs(predicted_step) > maximum_predicted_step:
                        failure_ring = candidate_ring
                        break
                    predicted_alpha = current_alpha + predicted_step
                    candidate_alpha = sector_angles[candidate_sectors]
                    position_residual = _wrapped_angular_difference(
                        candidate_alpha,
                        predicted_alpha,
                    )
                    orientation_change = _axial_difference(
                        orientation[candidate_ring, candidate_sectors],
                        previous_orientation,
                    )
                    unambiguous = ~np.isclose(
                        np.abs(orientation_change),
                        np.pi / 2.0,
                        atol=eps,
                        rtol=0.0,
                    )
                    if not unambiguous.any():
                        failure_ring = candidate_ring
                        break
                    candidate_sectors = candidate_sectors[unambiguous]
                    position_residual = position_residual[unambiguous]
                    orientation_change = orientation_change[unambiguous]

                    position_cost = np.square(position_residual / sector_width)
                    orientation_cost = np.square(
                        orientation_change / sector_width
                    )
                    reliability_cost = reliability_weight * (
                        1.0 - resultant[candidate_ring, candidate_sectors]
                    )
                    total_cost = (
                        position_cost + orientation_cost + reliability_cost
                    )
                    chosen_index = int(np.argmin(total_cost))
                    next_sector = int(candidate_sectors[chosen_index])
                    step_rotation = float(orientation_change[chosen_index])

                    cumulative[candidate_ring, seed_sector] = (
                        cumulative[current_ring, seed_sector] + step_rotation
                    )
                    path_sectors[candidate_ring, seed_sector] = next_sector
                    current_ring = candidate_ring
                    current_sector = next_sector
                    matched = True
                    break

                if matched:
                    continue
                if failure_ring is None:
                    failure_ring = search_start
                break

            if not allow_restarts or failure_ring is None:
                break
            restart_offsets = np.flatnonzero(
                reliable[failure_ring:, seed_sector]
            )
            if restart_offsets.size == 0:
                break
            segment_start = failure_ring + int(restart_offsets[0])

    return cumulative, path_sectors


def matched_tracks_to_ring_sector_values(
    cumulative_rotation: np.ndarray,
    path_sectors: np.ndarray,
) -> np.ndarray:
    """Project seed-indexed matched tracks onto a ring-sector value lattice.

    When tracks converge on the same ring-sector cell, the signed value with the
    largest absolute accumulated rotation is retained. This rule is diagnostic
    only and does not define a colony-level phenotype.

    Args:
        cumulative_rotation: Seed-indexed cumulative values shaped
            ``(n_rings, n_sectors)``.
        path_sectors: Matched sector indices with the same shape and ``-1`` for
            unsupported steps.

    Returns:
        Ring-sector lattice with the same shape and ``NaN`` for cells unused by
        any matched track.

    Raises:
        ValueError: If arrays are not equally shaped two-dimensional matrices or
            contain out-of-range path indices.
    """
    cumulative = np.asarray(cumulative_rotation, dtype=np.float64)
    paths = np.asarray(path_sectors)
    if cumulative.ndim != 2 or paths.shape != cumulative.shape:
        raise ValueError(
            "matched track arrays must share one two-dimensional shape"
        )
    if not np.issubdtype(paths.dtype, np.integer):
        raise ValueError("path sector indices must use an integer dtype")
    n_rings, n_sectors = cumulative.shape
    if np.any(paths < -1) or np.any(paths >= n_sectors):
        raise ValueError(
            "path sector indices must be -1 or valid sector indices"
        )

    field = np.full_like(cumulative, np.nan)
    for ring_index in range(n_rings):
        for seed_sector in range(n_sectors):
            value = cumulative[ring_index, seed_sector]
            target_sector = int(paths[ring_index, seed_sector])
            if target_sector < 0 or not np.isfinite(value):
                continue
            existing = field[ring_index, target_sector]
            if not np.isfinite(existing) or abs(value) > abs(existing):
                field[ring_index, target_sector] = value
    return field
