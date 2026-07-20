"""Analytic controls for the multiscale fiber-bend field."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.util._nematic_bend import fiber_bend_field


def _polar_grid(
    size: int = 181,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    centre = ((size - 1) / 2.0, (size - 1) / 2.0)
    rows, cols = np.indices((size, size), dtype=np.float64)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    return distance, polar, centre


def _gradient_normal_from_fiber(fiber_axis: np.ndarray) -> np.ndarray:
    return fiber_axis - np.pi / 2.0


def test_uniform_fiber_director_has_zero_bend():
    phi = np.full((81, 81), -np.pi / 2.0)
    coherence = np.ones_like(phi)
    selector = np.ones_like(phi, dtype=bool)

    bend, resultant = fiber_bend_field(phi, coherence, selector, 4.0)

    assert np.nanmax(bend) < 1e-12
    assert np.nanmin(resultant) > 1.0 - 1e-12


def test_radial_spokes_have_zero_bend_away_from_mask_boundaries():
    distance, polar, _centre = _polar_grid()
    selector = (distance >= 20.0) & (distance < 80.0)
    interior = (distance >= 36.0) & (distance < 64.0)
    phi = _gradient_normal_from_fiber(polar)

    bend, _resultant = fiber_bend_field(
        phi,
        np.ones_like(phi),
        selector,
        4.0,
    )

    assert np.nanmedian(bend[interior]) < 2e-4


def test_constant_pitch_spiral_recovers_director_line_curvature():
    distance, polar, _centre = _polar_grid()
    beta = np.deg2rad(30.0)
    selector = (distance >= 20.0) & (distance < 80.0)
    interior = (distance >= 36.0) & (distance < 64.0)
    phi = _gradient_normal_from_fiber(polar + beta)

    bend, _resultant = fiber_bend_field(
        phi,
        np.ones_like(phi),
        selector,
        4.0,
    )
    expected = np.abs(np.sin(beta)) / distance[interior]

    assert np.nanmedian(bend[interior] / expected) == pytest.approx(
        1.0,
        rel=0.06,
    )


def test_fiber_bend_is_invariant_to_axial_pi_representation():
    distance, polar, _centre = _polar_grid(121)
    selector = (distance >= 16.0) & (distance < 52.0)
    phi = _gradient_normal_from_fiber(polar + np.deg2rad(25.0))

    bend_a, resultant_a = fiber_bend_field(
        phi,
        np.ones_like(phi),
        selector,
        3.0,
    )
    seam_phi = phi.copy()
    seam_phi[:, seam_phi.shape[1] // 2 :] += np.pi
    bend_b, resultant_b = fiber_bend_field(
        seam_phi,
        np.ones_like(phi),
        selector,
        3.0,
    )

    assert np.allclose(bend_a, bend_b, equal_nan=True, atol=1e-12)
    assert np.allclose(resultant_a, resultant_b, equal_nan=True, atol=1e-12)


def test_fiber_quarter_turn_is_applied_before_bend():
    distance, polar, _centre = _polar_grid()
    selector = (distance >= 20.0) & (distance < 80.0)
    interior = (distance >= 36.0) & (distance < 64.0)
    # A circular fiber director has gradient-normal axis equal to ``polar``.
    phi = polar

    bend, _resultant = fiber_bend_field(
        phi,
        np.ones_like(phi),
        selector,
        4.0,
    )
    expected = 1.0 / distance[interior]

    assert np.nanmedian(bend[interior] / expected) == pytest.approx(
        1.0,
        rel=0.06,
    )


def test_mask_aware_smoothing_ignores_background_orientation():
    rng = np.random.default_rng(0)
    distance, _polar, _centre = _polar_grid(101)
    selector = distance < 35.0
    reference_phi = np.full(distance.shape, -np.pi / 2.0)
    changed_phi = reference_phi.copy()
    changed_phi[~selector] = rng.uniform(
        -np.pi / 2.0, np.pi / 2.0, (~selector).sum()
    )
    coherence = np.ones_like(distance)

    reference_bend, reference_resultant = fiber_bend_field(
        reference_phi,
        coherence,
        selector,
        6.0,
    )
    changed_bend, changed_resultant = fiber_bend_field(
        changed_phi,
        coherence,
        selector,
        6.0,
    )

    assert np.allclose(reference_bend, changed_bend, equal_nan=True)
    assert np.allclose(reference_resultant, changed_resultant, equal_nan=True)


def test_orthogonal_fiber_families_reduce_scale_resultant():
    phi = np.full((101, 101), -np.pi / 2.0)
    phi[:, 51:] = 0.0

    _bend, resultant = fiber_bend_field(
        phi,
        np.ones_like(phi),
        np.ones_like(phi, dtype=bool),
        8.0,
    )

    assert resultant[50, 50] < 0.1
    assert resultant[50, 20] > 0.99


def test_coherence_weights_conflicting_fiber_families():
    phi = np.full((101, 101), -np.pi / 2.0)
    phi[:, 51:] = 0.0
    selector = np.ones_like(phi, dtype=bool)
    uniform_coherence = np.ones_like(phi)
    weighted_coherence = uniform_coherence.copy()
    weighted_coherence[:, 51:] = 0.05

    _bend, uniform_resultant = fiber_bend_field(
        phi,
        uniform_coherence,
        selector,
        8.0,
    )
    _bend, weighted_resultant = fiber_bend_field(
        phi,
        weighted_coherence,
        selector,
        8.0,
    )

    assert uniform_resultant[50, 50] < 0.1
    assert weighted_resultant[50, 50] > 0.9


def test_broad_scale_attenuates_localized_director_oscillation():
    _rows, cols = np.indices((129, 129), dtype=np.float64)
    fiber_axis = 0.3 * np.sin(2.0 * np.pi * cols / 20.0)
    phi = _gradient_normal_from_fiber(fiber_axis)
    coherence = np.ones_like(phi)
    selector = np.ones_like(phi, dtype=bool)
    interior = (slice(20, -20), slice(20, -20))

    fine_bend, _resultant = fiber_bend_field(
        phi,
        coherence,
        selector,
        1.5,
    )
    broad_bend, _resultant = fiber_bend_field(
        phi,
        coherence,
        selector,
        8.0,
    )

    fine_p90 = np.nanpercentile(fine_bend[interior], 90.0)
    broad_p90 = np.nanpercentile(broad_bend[interior], 90.0)
    assert broad_p90 < 0.1 * fine_p90


@pytest.mark.parametrize("sigma_q", [0.0, -1.0, np.nan, np.inf])
def test_fiber_bend_rejects_invalid_scale(sigma_q):
    array = np.ones((8, 8))

    with pytest.raises(ValueError, match="sigma_q"):
        fiber_bend_field(array, array, array.astype(bool), sigma_q)


def test_fiber_bend_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="share one shape"):
        fiber_bend_field(
            np.ones((8, 8)),
            np.ones((7, 8)),
            np.ones((8, 8), dtype=bool),
            2.0,
        )
