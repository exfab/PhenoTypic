"""Independently verify multiscale nematic fiber-bend invariants.

This script deliberately does not import ``phenotypic``. It re-derives the
load-bearing doubled-angle geometry using only NumPy and SciPy.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def independent_fiber_bend(
    gradient_normal: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mask-aware fiber bend and scale-local axial resultant."""
    valid = selector & np.isfinite(gradient_normal) & (coherence > 0.0)
    weights = np.where(valid, coherence, 0.0)
    # A fiber is perpendicular to the supplied gradient normal, so both
    # doubled-angle components change sign.
    cosine = -np.cos(2.0 * np.where(valid, gradient_normal, 0.0))
    sine = -np.sin(2.0 * np.where(valid, gradient_normal, 0.0))
    smooth_weight = gaussian_filter(weights, sigma, mode="constant", cval=0.0)
    qx = gaussian_filter(weights * cosine, sigma, mode="constant", cval=0.0)
    qy = gaussian_filter(weights * sine, sigma, mode="constant", cval=0.0)
    norm_squared = qx * qx + qy * qy
    norm = np.sqrt(norm_squared)
    resultant = np.divide(
        norm,
        smooth_weight,
        out=np.full_like(norm, np.nan),
        where=smooth_weight > 1e-12,
    )
    qx_y, qx_x = np.gradient(qx)
    qy_y, qy_x = np.gradient(qy)
    denominator = norm_squared + 1e-12
    angle_x = 0.5 * (qx * qy_x - qy * qx_x) / denominator
    angle_y = 0.5 * (qx * qy_y - qy * qx_y) / denominator
    angle = 0.5 * np.arctan2(qy, qx)
    bend = np.abs(np.cos(angle) * angle_x + np.sin(angle) * angle_y)
    supported = valid & (smooth_weight > 1e-12) & (norm_squared > 1e-12)
    return np.where(supported, bend, np.nan), np.where(
        supported,
        np.clip(resultant, 0.0, 1.0),
        np.nan,
    )


def verify_nematic_fiber_bend_invariants() -> None:
    """Assert straight, radial, spiral, circular, and axial invariants."""
    size = 181
    centre = ((size - 1) / 2.0, (size - 1) / 2.0)
    rows, cols = np.indices((size, size), dtype=float)
    distance = np.hypot(rows - centre[0], cols - centre[1])
    polar = np.arctan2(rows - centre[0], cols - centre[1])
    annulus = (distance >= 20.0) & (distance < 80.0)
    interior = (distance >= 36.0) & (distance < 64.0)
    coherence = np.ones_like(distance)

    straight_phi = np.full_like(distance, -np.pi / 2.0)
    straight, _ = independent_fiber_bend(
        straight_phi,
        coherence,
        annulus,
        4.0,
    )
    assert np.nanmax(straight[interior]) < 1e-12

    radial_phi = polar - np.pi / 2.0
    radial, _ = independent_fiber_bend(
        radial_phi,
        coherence,
        annulus,
        4.0,
    )
    assert np.nanmedian(radial[interior]) < 2e-4

    pitch = np.deg2rad(30.0)
    spiral_phi = polar + pitch - np.pi / 2.0
    spiral, spiral_resultant = independent_fiber_bend(
        spiral_phi,
        coherence,
        annulus,
        4.0,
    )
    expected_spiral = abs(np.sin(pitch)) / distance[interior]
    assert np.isclose(
        np.nanmedian(spiral[interior] / expected_spiral),
        1.0,
        rtol=0.06,
    )
    assert np.nanmedian(spiral_resultant[interior]) > 0.95

    # A circular fiber field has gradient normal equal to the radial direction.
    circular, _ = independent_fiber_bend(
        polar,
        coherence,
        annulus,
        4.0,
    )
    expected_circular = 1.0 / distance[interior]
    assert np.isclose(
        np.nanmedian(circular[interior] / expected_circular),
        1.0,
        rtol=0.06,
    )

    seam_phi = spiral_phi.copy()
    seam_phi[:, seam_phi.shape[1] // 2 :] += np.pi
    axial_copy, axial_resultant = independent_fiber_bend(
        seam_phi,
        coherence,
        annulus,
        4.0,
    )
    assert np.allclose(spiral, axial_copy, equal_nan=True, atol=1e-12)
    assert np.allclose(
        spiral_resultant,
        axial_resultant,
        equal_nan=True,
        atol=1e-12,
    )

    orthogonal = np.full((101, 101), -np.pi / 2.0)
    orthogonal[:, 51:] = 0.0
    _, mixed_resultant = independent_fiber_bend(
        orthogonal,
        np.ones_like(orthogonal),
        np.ones_like(orthogonal, dtype=bool),
        8.0,
    )
    assert mixed_resultant[50, 50] < 0.1
    assert mixed_resultant[50, 20] > 0.99

    weighted_coherence = np.ones_like(orthogonal)
    weighted_coherence[:, 51:] = 0.05
    _, weighted_resultant = independent_fiber_bend(
        orthogonal,
        weighted_coherence,
        np.ones_like(orthogonal, dtype=bool),
        8.0,
    )
    assert weighted_resultant[50, 50] > 0.9

    _, columns = np.indices((129, 129), dtype=float)
    oscillating_fiber = 0.3 * np.sin(2.0 * np.pi * columns / 20.0)
    oscillating_phi = oscillating_fiber - np.pi / 2.0
    oscillating_selector = np.ones_like(oscillating_phi, dtype=bool)
    oscillating_coherence = np.ones_like(oscillating_phi)
    fine_bend, _ = independent_fiber_bend(
        oscillating_phi,
        oscillating_coherence,
        oscillating_selector,
        1.5,
    )
    broad_bend, _ = independent_fiber_bend(
        oscillating_phi,
        oscillating_coherence,
        oscillating_selector,
        8.0,
    )
    oscillating_interior = (slice(20, -20), slice(20, -20))
    fine_p90 = np.nanpercentile(fine_bend[oscillating_interior], 90.0)
    broad_p90 = np.nanpercentile(broad_bend[oscillating_interior], 90.0)
    assert broad_p90 < 0.1 * fine_p90


if __name__ == "__main__":
    verify_nematic_fiber_bend_invariants()
