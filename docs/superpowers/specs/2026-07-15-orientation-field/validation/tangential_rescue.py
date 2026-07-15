"""Re-derive bounded tangential-rescue invariants with NumPy only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def axial_difference(outer: np.ndarray | float, inner: float) -> np.ndarray:
    """Return signed unoriented-axis differences."""
    difference = np.asarray(outer, dtype=float) - inner
    return 0.5 * np.arctan2(
        np.sin(2.0 * difference),
        np.cos(2.0 * difference),
    )


def circular_difference(outer: np.ndarray, inner: float) -> np.ndarray:
    """Return signed circular differences."""
    return np.arctan2(np.sin(outer - inner), np.cos(outer - inner))


@dataclass(frozen=True)
class Transition:
    """One independently selected ring transition."""

    kind: str
    next_sector: int
    rotation: float
    route: tuple[int, ...]


def derive_transition(
    radii: np.ndarray,
    orientation: np.ndarray,
    resultant: np.ndarray,
    ring: int,
    sector: int,
    *,
    max_outward_shift: int = 2,
    max_tangential_steps: int = 2,
    max_abs_radial_tilt: float = np.deg2rad(75.0),
    max_axis_mismatch: float = np.deg2rad(35.0),
) -> Transition | None:
    """Independently enumerate one direct-first tangential rescue step."""
    n_sectors = orientation.shape[1]
    sector_width = 2.0 * np.pi / n_sectors
    sector_angles = (np.arange(n_sectors) + 0.5) * sector_width
    reliable = np.isfinite(orientation) & np.isfinite(resultant)
    offsets = np.arange(-max_outward_shift, max_outward_shift + 1)
    direct: list[tuple[float, int, float, tuple[int, ...]]] = []
    lateral: list[tuple[float, int, float, tuple[int, ...]]] = []

    for direction in (0, -1, 1):
        counts = (0,) if direction == 0 else range(1, max_tangential_steps + 1)
        previous_sector = sector
        lateral_rotation = 0.0
        route = [sector]
        for count in counts:
            if count:
                current_sector = (sector + direction * count) % n_sectors
                if not reliable[ring, current_sector]:
                    break
                change = float(
                    axial_difference(
                        orientation[ring, current_sector],
                        orientation[ring, previous_sector],
                    )
                )
                if np.isclose(
                    abs(change),
                    np.pi / 2.0,
                    atol=1e-12,
                    rtol=0.0,
                ):
                    break
                chord_bearing = sector_angles[previous_sector] + direction * (
                    np.pi / 2.0 + sector_width / 2.0
                )
                mismatch = max(
                    abs(
                        float(
                            axial_difference(
                                orientation[ring, previous_sector],
                                chord_bearing,
                            )
                        )
                    ),
                    abs(
                        float(
                            axial_difference(
                                orientation[ring, current_sector],
                                chord_bearing,
                            )
                        )
                    ),
                )
                if mismatch > max_axis_mismatch:
                    break
                lateral_rotation += change
                previous_sector = current_sector
                route.append(current_sector)
            else:
                current_sector = sector

            current_angle = sector_angles[current_sector]
            current_orientation = orientation[ring, current_sector]
            radial_tilt = float(
                axial_difference(current_orientation, current_angle)
            )
            if abs(radial_tilt) > max_abs_radial_tilt:
                continue
            predicted_step = np.tan(radial_tilt) * np.log(
                radii[ring + 1] / radii[ring]
            )
            if abs(predicted_step) > (
                max_outward_shift + 0.5
            ) * sector_width:
                continue
            candidates = np.unique((current_sector + offsets) % n_sectors)
            candidates = candidates[reliable[ring + 1, candidates]]
            if candidates.size == 0:
                continue
            outward_change = axial_difference(
                orientation[ring + 1, candidates],
                current_orientation,
            )
            usable = ~np.isclose(
                np.abs(outward_change),
                np.pi / 2.0,
                atol=1e-12,
                rtol=0.0,
            )
            if not usable.any():
                continue
            candidates = candidates[usable]
            outward_change = outward_change[usable]
            residual = circular_difference(
                sector_angles[candidates],
                current_angle + predicted_step,
            )
            cost = np.square(residual / sector_width)
            cost += np.square(outward_change / sector_width)
            cost += 0.25 * (1.0 - resultant[ring + 1, candidates])
            cost += 0.20 * count
            chosen = int(np.argmin(cost))
            item = (
                float(cost[chosen]),
                int(candidates[chosen]),
                lateral_rotation + float(outward_change[chosen]),
                (*route, int(candidates[chosen])),
            )
            (direct if count == 0 else lateral).append(item)

    pool = direct or lateral
    if not pool:
        return None
    _cost, next_sector, rotation, route = min(pool, key=lambda item: item[0])
    return Transition(
        "direct" if direct else "tangential",
        next_sector,
        rotation,
        route,
    )


def axial_mean(values: np.ndarray) -> tuple[float, float]:
    """Return the equal-weight doubled-angle mean and resultant."""
    cosine = float(np.mean(np.cos(2.0 * values)))
    sine = float(np.mean(np.sin(2.0 * values)))
    return 0.5 * float(np.arctan2(sine, cosine)), float(np.hypot(cosine, sine))


def empty_lattice(
    n_rings: int = 2,
    n_sectors: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unsupported orientation and resultant lattices."""
    orientation = np.full((n_rings, n_sectors), np.nan)
    return orientation, np.full_like(orientation, np.nan)


def validate_direct_path_has_priority() -> None:
    """A valid direct edge cannot be displaced by an available lateral edge."""
    radii = np.array([20.0, 22.0])
    orientation, resultant = empty_lattice()
    orientation[0, 0] = np.deg2rad(65.0)
    orientation[0, 1] = np.deg2rad(100.0)
    orientation[1, 1] = np.deg2rad(65.0)
    orientation[1, 2] = np.deg2rad(80.0)
    resultant[np.isfinite(orientation)] = 1.0
    transition = derive_transition(radii, orientation, resultant, 0, 0)
    assert transition is not None and transition.kind == "direct"
    assert transition.next_sector == 1


def validate_two_cell_tangential_rescue() -> None:
    """Two bearing-aligned cells can rescue a failed tangent continuation."""
    radii = np.array([20.0, 22.0])
    orientation, resultant = empty_lattice()
    orientation[0, :3] = np.deg2rad([100.0, 100.0, 75.0])
    orientation[1, 3] = np.deg2rad(80.0)
    resultant[np.isfinite(orientation)] = 1.0
    transition = derive_transition(radii, orientation, resultant, 0, 0)
    assert transition is not None and transition.kind == "tangential"
    assert transition.route == (0, 1, 2, 3)
    assert np.isclose(np.degrees(transition.rotation), -20.0)


def validate_chord_bearing_and_wrapping() -> None:
    """The half-sector chord correction must work across sector 35 to 0."""
    radii = np.array([20.0, 22.0])
    orientation, resultant = empty_lattice()
    orientation[0, 35] = np.deg2rad(90.0)
    orientation[0, 0] = np.deg2rad(65.0)
    orientation[1, 1] = np.deg2rad(65.0)
    resultant[np.isfinite(orientation)] = 1.0
    transition = derive_transition(radii, orientation, resultant, 0, 35)
    assert transition is not None and transition.kind == "tangential"
    assert transition.route[:2] == (35, 0)


def validate_three_cell_rescue_is_rejected() -> None:
    """A route requiring a third same-ring step must remain unsupported."""
    radii = np.array([20.0, 22.0])
    orientation, resultant = empty_lattice()
    orientation[0, :4] = np.deg2rad([100.0, 100.0, 100.0, 85.0])
    orientation[1, 5] = np.deg2rad(85.0)
    resultant[np.isfinite(orientation)] = 1.0
    transition = derive_transition(radii, orientation, resultant, 0, 0)
    assert transition is None


def validate_axial_seam_is_invariant() -> None:
    """Equivalent axes on opposite sides of the seam have zero change."""
    difference = axial_difference(np.deg2rad(-91.0), np.deg2rad(89.0))
    assert np.isclose(difference, 0.0)


def validate_same_cell_duplication_control() -> None:
    """Repeating identical evidence cannot change cell angle or resultant."""
    sparse = np.deg2rad(np.array([30.0]))
    dense = np.repeat(sparse, 40)
    sparse_mean, sparse_resultant = axial_mean(sparse)
    dense_mean, dense_resultant = axial_mean(dense)
    assert np.isclose(sparse_mean, dense_mean)
    assert np.isclose(sparse_resultant, dense_resultant)


def validate_tangent_sign_is_ambiguous() -> None:
    """An axial tangent has two equally non-outward Cartesian hypotheses."""
    radial = np.array([0.0, 1.0])
    tangent = np.array([1.0, 0.0])
    assert np.isclose(np.dot(tangent, radial), 0.0)
    assert np.isclose(np.dot(-tangent, radial), 0.0)


def validate_all() -> None:
    """Run every independent tangential-rescue invariant."""
    validate_direct_path_has_priority()
    validate_two_cell_tangential_rescue()
    validate_chord_bearing_and_wrapping()
    validate_three_cell_rescue_is_rejected()
    validate_axial_seam_is_invariant()
    validate_same_cell_duplication_control()
    validate_tangent_sign_is_ambiguous()
    print("tangential rescue invariants: PASS")


if __name__ == "__main__":
    validate_all()
