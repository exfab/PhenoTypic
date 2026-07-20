"""Validate equal-sector Sholl ring compounding with NumPy only."""

from __future__ import annotations

import numpy as np


def axial_difference(outer: np.ndarray | float, inner: float) -> np.ndarray:
    """Return signed axial differences in [-pi/2, pi/2]."""
    difference = np.asarray(outer, dtype=float) - inner
    return 0.5 * np.arctan2(
        np.sin(2.0 * difference),
        np.cos(2.0 * difference),
    )


def axial_mean(values: np.ndarray) -> tuple[float, float]:
    """Return an equal-sector axial mean and resultant."""
    cosine = float(np.mean(np.cos(2.0 * values)))
    sine = float(np.mean(np.sin(2.0 * values)))
    return 0.5 * float(np.arctan2(sine, cosine)), float(np.hypot(cosine, sine))


def axial_median(values: np.ndarray, mean_angle: float) -> float:
    """Return a sample axial median with mean-proximity tie breaking."""
    distances = np.abs(
        axial_difference(values[:, np.newaxis], values[np.newaxis, :])
    )
    costs = distances.sum(axis=1)
    best = np.flatnonzero(np.isclose(costs, costs.min(), atol=1e-12, rtol=0.0))
    if best.size == 1:
        return float(values[best[0]])
    tie_distance = np.abs(axial_difference(values[best], mean_angle))
    return float(values[best[int(np.argmin(tie_distance))]])


def ring_consensus(
    sector_tilt: np.ndarray,
    sector_resultant: np.ndarray,
    minimum_sectors: int = 3,
    minimum_ring_resultant: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Independently derive equal-reliable-sector mean and median profiles."""
    means = np.full(sector_tilt.shape[0], np.nan)
    medians = np.full(sector_tilt.shape[0], np.nan)
    reliable = np.isfinite(sector_tilt) & np.isfinite(sector_resultant)
    for ring in range(sector_tilt.shape[0]):
        values = sector_tilt[ring, reliable[ring]]
        if values.size < minimum_sectors:
            continue
        mean_angle, _resultant = axial_mean(values)
        if _resultant < minimum_ring_resultant:
            continue
        means[ring] = mean_angle
        medians[ring] = axial_median(values, mean_angle)
    return means, medians


def compound(
    radii: np.ndarray,
    ring_tilt: np.ndarray,
    maximum_abs_tilt: float = np.deg2rad(75.0),
) -> np.ndarray:
    """Integrate the constant-tilt polar predictor across adjacent rings."""
    output = np.full_like(ring_tilt, np.nan)
    starts = np.flatnonzero(np.isfinite(ring_tilt))
    if starts.size == 0:
        return output
    start = int(starts[0])
    output[start] = 0.0
    for ring in range(start, ring_tilt.size - 1):
        if not (
            np.isfinite(output[ring])
            and np.isfinite(ring_tilt[ring])
            and np.isfinite(ring_tilt[ring + 1])
        ):
            break
        if abs(ring_tilt[ring]) > maximum_abs_tilt:
            break
        output[ring + 1] = output[ring] + np.tan(ring_tilt[ring]) * np.log(
            radii[ring + 1] / radii[ring]
        )
    return output


def validate_straight_radial_is_zero() -> None:
    """Zero radial-relative tilt must remain zero at every radius."""
    radii = np.array([20.0, 28.0, 36.0, 44.0])
    result = compound(radii, np.zeros(radii.size))
    assert np.array_equal(result, np.zeros(radii.size))


def validate_constant_tilt_matches_analytic_spiral() -> None:
    """Constant tilt must telescope to tan(delta) log(r/r0)."""
    radii = np.array([20.0, 24.0, 31.0, 44.0, 60.0])
    tilt = np.full(radii.size, np.deg2rad(25.0))
    result = compound(radii, tilt)
    expected = np.tan(tilt[0]) * np.log(radii / radii[0])
    assert np.allclose(result, expected)

    refined_radii = np.linspace(20.0, 60.0, 101)
    refined = compound(
        refined_radii,
        np.full(refined_radii.size, tilt[0]),
    )
    assert np.isclose(refined[-1], expected[-1])


def validate_turning_sign_reverses() -> None:
    """Clockwise and counterclockwise tilts must have opposite accumulation."""
    radii = np.array([20.0, 28.0, 36.0])
    positive = compound(radii, np.full(3, np.deg2rad(20.0)))
    negative = compound(radii, np.full(3, np.deg2rad(-20.0)))
    assert np.allclose(positive, -negative)


def validate_mean_median_and_resultant_weights() -> None:
    """Reliable sectors count once regardless of their resultant magnitude."""
    sector_tilt = np.deg2rad(
        np.array(
            [
                [10.0, 20.0, 30.0, np.nan],
                [10.0, 20.0, 30.0, np.nan],
            ]
        )
    )
    resultants = np.array(
        [
            [0.2, 0.4, 1.0, np.nan],
            [1.0, 0.2, 0.4, np.nan],
        ]
    )
    means, medians = ring_consensus(sector_tilt, resultants)
    assert np.isclose(means[0], means[1])
    assert np.isclose(medians[0], medians[1])
    assert np.isclose(np.degrees(medians[0]), 20.0)


def validate_axial_median_crosses_seam() -> None:
    """The median must not jump at the axial plus/minus 90-degree seam."""
    values = np.deg2rad(np.array([88.0, -89.0, -87.0]))
    mean_angle, _resultant = axial_mean(values)
    median = axial_median(values, mean_angle)
    assert abs(float(axial_difference(median, np.deg2rad(-89.0)))) < 1e-12


def validate_balanced_distribution_is_ambiguous() -> None:
    """Orthogonal sector families cannot define one ring consensus axis."""
    sector_tilt = np.deg2rad(np.array([[0.0, 0.0, 90.0, 90.0]]))
    resultants = np.ones_like(sector_tilt)
    means, medians = ring_consensus(sector_tilt, resultants)
    assert np.isnan(means[0]) and np.isnan(medians[0])


def validate_gaps_and_tangency_terminate() -> None:
    """Unsupported or near-tangent rings cannot be silently integrated."""
    radii = np.array([20.0, 28.0, 36.0, 44.0])
    gap = compound(radii, np.array([0.1, np.nan, 0.1, 0.1]))
    assert gap[0] == 0.0 and np.isnan(gap[1:]).all()

    tangent = compound(
        radii,
        np.deg2rad(np.array([20.0, 80.0, 20.0, 20.0])),
    )
    assert np.isfinite(tangent[:2]).all()
    assert np.isnan(tangent[2:]).all()


def validate_all() -> None:
    """Run every independent ring-compounding invariant."""
    validate_straight_radial_is_zero()
    validate_constant_tilt_matches_analytic_spiral()
    validate_turning_sign_reverses()
    validate_mean_median_and_resultant_weights()
    validate_axial_median_crosses_seam()
    validate_balanced_distribution_is_ambiguous()
    validate_gaps_and_tangency_terminate()
    print("ring compounded rotation invariants: PASS")


if __name__ == "__main__":
    validate_all()
