"""Re-derive matched-ring cumulative-rotation invariants from NumPy only."""

from __future__ import annotations

import numpy as np


def axial_difference(outer: np.ndarray, inner: float) -> np.ndarray:
    """Return unoriented-axis differences without a 180-degree seam."""
    delta = outer - inner
    return 0.5 * np.arctan2(np.sin(2.0 * delta), np.cos(2.0 * delta))


def circular_difference(outer: np.ndarray, inner: float) -> np.ndarray:
    """Return signed circular differences."""
    return np.arctan2(np.sin(outer - inner), np.cos(outer - inner))


def derive_matched_rotation(
    radii: np.ndarray,
    orientation: np.ndarray,
    resultant: np.ndarray,
    max_shift: int = 2,
    max_abs_radial_tilt: float = np.deg2rad(75.0),
    allow_gap_bridging: bool = False,
    allow_restarts: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Independently derive the prototype's local matched-ring recurrence."""
    n_rings, n_sectors = orientation.shape
    sector_width = 2.0 * np.pi / n_sectors
    sector_angles = (np.arange(n_sectors) + 0.5) * sector_width
    cumulative = np.full_like(orientation, np.nan, dtype=float)
    paths = np.full(orientation.shape, -1, dtype=int)
    reliable = np.isfinite(orientation) & np.isfinite(resultant)
    offsets = np.arange(-max_shift, max_shift + 1)

    for seed in range(n_sectors):
        starts = np.flatnonzero(reliable[:, seed])
        if starts.size == 0:
            continue
        segment_start = int(starts[0])
        while segment_start < n_rings:
            current_ring = segment_start
            sector = seed
            cumulative[current_ring, seed] = 0.0
            paths[current_ring, seed] = sector
            failure_ring: int | None = None
            while current_ring + 1 < n_rings:
                search_start = current_ring + 1
                ring = search_start
                matched = False
                while ring < n_rings:
                    previous = orientation[current_ring, sector]
                    candidates = np.unique((sector + offsets) % n_sectors)
                    candidates = candidates[reliable[ring, candidates]]
                    if candidates.size == 0:
                        if allow_gap_bridging:
                            ring += 1
                            continue
                        failure_ring = ring
                        break
                    alpha = sector_angles[sector]
                    radial_relative = axial_difference(
                        np.array([previous]), alpha
                    )[0]
                    if abs(radial_relative) > max_abs_radial_tilt:
                        failure_ring = ring
                        break
                    predicted_step = np.tan(radial_relative) * np.log(
                        radii[ring] / radii[current_ring]
                    )
                    if abs(predicted_step) > (max_shift + 0.5) * sector_width:
                        failure_ring = ring
                        break
                    predicted = alpha + predicted_step
                    position_delta = circular_difference(
                        sector_angles[candidates], predicted
                    )
                    orientation_delta = axial_difference(
                        orientation[ring, candidates], previous
                    )
                    usable = ~np.isclose(
                        np.abs(orientation_delta),
                        np.pi / 2.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                    if not usable.any():
                        failure_ring = ring
                        break
                    candidates = candidates[usable]
                    position_delta = position_delta[usable]
                    orientation_delta = orientation_delta[usable]
                    cost = (
                        np.square(position_delta / sector_width)
                        + np.square(orientation_delta / sector_width)
                        + 0.25 * (1.0 - resultant[ring, candidates])
                    )
                    choice = int(np.argmin(cost))
                    sector = int(candidates[choice])
                    cumulative[ring, seed] = (
                        cumulative[current_ring, seed]
                        + orientation_delta[choice]
                    )
                    paths[ring, seed] = sector
                    current_ring = ring
                    matched = True
                    break
                if matched:
                    continue
                if failure_ring is None:
                    failure_ring = search_start
                break
            if not allow_restarts or failure_ring is None:
                break
            restart_offsets = np.flatnonzero(reliable[failure_ring:, seed])
            if restart_offsets.size == 0:
                break
            segment_start = failure_ring + int(restart_offsets[0])
    return cumulative, paths


def fixed_sector_rotation(orientation: np.ndarray) -> np.ndarray:
    """Derive the same-sector reference without importing phenotypic."""
    output = np.full_like(orientation, np.nan, dtype=float)
    for sector in range(orientation.shape[1]):
        starts = np.flatnonzero(np.isfinite(orientation[:, sector]))
        if starts.size == 0:
            continue
        start = int(starts[0])
        output[start, sector] = 0.0
        for ring in range(start + 1, orientation.shape[0]):
            if not (
                np.isfinite(orientation[ring - 1, sector])
                and np.isfinite(orientation[ring, sector])
            ):
                break
            step = axial_difference(
                np.array([orientation[ring, sector]]),
                orientation[ring - 1, sector],
            )[0]
            if np.isclose(abs(step), np.pi / 2.0, atol=1e-12):
                break
            output[ring, sector] = output[ring - 1, sector] + step
    return output


def validate_moving_corridor() -> None:
    """A moving branch corridor must be followed while the fixed sector stops."""
    radii = np.array([20.0, 28.0, 36.0, 44.0])
    orientation = np.full((4, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    path = np.array([0, 1, 2, 3])
    orientation[np.arange(4), path] = np.deg2rad([0.0, 10.0, 20.0, 30.0])
    resultant[np.arange(4), path] = 1.0

    matched, sectors = derive_matched_rotation(radii, orientation, resultant)
    fixed = fixed_sector_rotation(orientation)
    assert np.array_equal(sectors[:, 0], path)
    assert np.allclose(np.degrees(matched[:, 0]), [0.0, 10.0, 20.0, 30.0])
    assert np.isfinite(fixed[0, 0]) and np.isnan(fixed[1:, 0]).all()


def validate_axis_seam() -> None:
    """The recurrence must cross the unoriented fiber-axis seam."""
    radii = np.array([20.0, 28.0, 36.0, 44.0])
    orientation = np.full((4, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    path = np.array([8, 9, 10, 11])
    orientation[np.arange(4), path] = np.deg2rad([80.0, -89.0, -78.0, -67.0])
    resultant[np.arange(4), path] = 1.0
    matched, sectors = derive_matched_rotation(radii, orientation, resultant)
    assert np.array_equal(sectors[:, 8], path)
    assert np.allclose(np.degrees(matched[:, 8]), [0.0, 11.0, 22.0, 33.0])


def validate_competing_candidate_costs() -> None:
    """Position, orientation continuity, and reliability must affect matches."""
    radii = np.array([20.0, 28.0])
    sector_width = 2.0 * np.pi / 36.0

    orientation = np.full((2, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    radial_tilt = np.arctan(sector_width / np.log(radii[1] / radii[0]))
    previous = sector_width / 2.0 + radial_tilt
    orientation[0, 0] = previous
    orientation[1, 0] = previous
    orientation[1, 1] = previous + sector_width / 2.0
    resultant[0, 0] = 1.0
    resultant[1, :2] = 1.0
    matched, paths = derive_matched_rotation(radii, orientation, resultant)
    assert paths[1, 0] == 1
    assert np.isclose(np.degrees(matched[1, 0]), 5.0)

    orientation = np.full((2, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    previous = np.deg2rad(5.0)
    orientation[0, 0] = previous
    orientation[1, 0] = previous + np.deg2rad(20.0)
    orientation[1, 1] = previous
    resultant[0, 0] = 1.0
    resultant[1, :2] = 1.0
    matched, paths = derive_matched_rotation(radii, orientation, resultant)
    assert paths[1, 0] == 1
    assert np.isclose(matched[1, 0], 0.0)

    orientation = np.full((2, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    half_sector = sector_width / 2.0
    radial_tilt = np.arctan(half_sector / np.log(radii[1] / radii[0]))
    previous = sector_width / 2.0 + radial_tilt
    orientation[0, 0] = previous
    orientation[1, :2] = previous
    resultant[0, 0] = 1.0
    resultant[1, 0] = 0.2
    resultant[1, 1] = 1.0
    _matched, paths = derive_matched_rotation(radii, orientation, resultant)
    assert paths[1, 0] == 1


def validate_ill_conditioned_prediction_terminates() -> None:
    """Tangential and out-of-neighborhood predictions must terminate."""
    radii = np.array([20.0, 28.0])
    orientation = np.full((2, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[0, 0] = np.deg2rad(85.0)
    orientation[1, 1] = np.deg2rad(85.0)
    resultant[0, 0] = 1.0
    resultant[1, 1] = 1.0
    matched, paths = derive_matched_rotation(radii, orientation, resultant)
    assert np.isnan(matched[1, 0]) and paths[1, 0] == -1

    orientation[0, 0] = np.deg2rad(75.0)
    orientation[1] = np.nan
    resultant[1] = np.nan
    orientation[1, 2] = np.deg2rad(75.0)
    resultant[1, 2] = 1.0
    matched, paths = derive_matched_rotation(radii, orientation, resultant)
    assert np.isnan(matched[1, 0]) and paths[1, 0] == -1


def validate_support_gap_terminates() -> None:
    """A missing adjacent ring must terminate rather than silently bridge."""
    radii = np.array([20.0, 28.0, 36.0, 44.0])
    orientation = np.full((4, 8), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[0, 0] = 0.0
    resultant[0, 0] = 1.0
    orientation[2, 1] = np.deg2rad(10.0)
    resultant[2, 1] = 1.0
    matched, paths = derive_matched_rotation(radii, orientation, resultant)
    assert matched[0, 0] == 0.0
    assert np.isnan(matched[1:, 0]).all()
    assert np.array_equal(paths[:, 0], [0, -1, -1, -1])


def validate_gap_and_restart_relaxations() -> None:
    """The four rule combinations must preserve their distinct semantics."""
    radii = np.array([20.0, 28.0, 36.0])
    orientation = np.full((3, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[0, 0] = np.deg2rad(5.0)
    orientation[2, 0] = np.deg2rad(25.0)
    resultant[0, 0] = 1.0
    resultant[2, 0] = 1.0

    strict, _ = derive_matched_rotation(radii, orientation, resultant)
    bridged, _ = derive_matched_rotation(
        radii,
        orientation,
        resultant,
        allow_gap_bridging=True,
    )
    restarted, _ = derive_matched_rotation(
        radii,
        orientation,
        resultant,
        allow_restarts=True,
    )
    both, _ = derive_matched_rotation(
        radii,
        orientation,
        resultant,
        allow_gap_bridging=True,
        allow_restarts=True,
    )
    assert np.isnan(strict[2, 0])
    assert np.isclose(np.degrees(bridged[2, 0]), 20.0)
    assert np.isclose(restarted[2, 0], 0.0)
    assert np.isclose(np.degrees(both[2, 0]), 20.0)

    orientation[:, 0] = np.deg2rad([85.0, 5.0, 15.0])
    resultant[:, 0] = 1.0
    restarted, _ = derive_matched_rotation(
        radii,
        orientation,
        resultant,
        allow_restarts=True,
    )
    assert np.allclose(np.degrees(restarted[:, 0]), [0.0, 0.0, 10.0])


def validate_gap_uses_full_interval_and_not_hard_failures() -> None:
    """Bridges use full radial distance and never skip axial ambiguity."""
    radii = np.array([20.0, 28.0, 44.0])
    orientation = np.full((3, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    sector_width = 2.0 * np.pi / 36.0
    radial_tilt = np.arctan(np.deg2rad(8.0) / np.log(radii[2] / radii[0]))
    previous = sector_width / 2.0 + radial_tilt
    orientation[0, 0] = previous
    orientation[2, :2] = previous
    resultant[0, 0] = 1.0
    resultant[2, :2] = 1.0
    bridged, paths = derive_matched_rotation(
        radii,
        orientation,
        resultant,
        allow_gap_bridging=True,
    )
    assert paths[2, 0] == 1 and np.isclose(bridged[2, 0], 0.0)

    radii = np.array([20.0, 28.0, 36.0])
    orientation = np.full((3, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[:, 0] = np.deg2rad([5.0, -85.0, 15.0])
    resultant[:, 0] = 1.0
    bridged, paths = derive_matched_rotation(
        radii,
        orientation,
        resultant,
        allow_gap_bridging=True,
    )
    assert np.array_equal(paths[:, 0], [0, -1, -1])
    assert np.isnan(bridged[1:, 0]).all()


def validate_matched_ring_rotation() -> None:
    """Run all independent numeric claims and fail loudly on drift."""
    validate_moving_corridor()
    validate_axis_seam()
    validate_competing_candidate_costs()
    validate_ill_conditioned_prediction_terminates()
    validate_support_gap_terminates()
    validate_gap_and_restart_relaxations()
    validate_gap_uses_full_interval_and_not_hard_failures()
    print("matched-ring cumulative rotation invariants: PASS")


if __name__ == "__main__":
    validate_matched_ring_rotation()
