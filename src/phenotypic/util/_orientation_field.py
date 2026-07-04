"""Structure-tensor orientation field on a 2-D intensity tile.

Derivation: docs/superpowers/explain/2026-07-03-gradient-to-orientation-field-metrics.md
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def orientation_field(
    intensity: np.ndarray,
    sigma_d: float = 1.5,
    sigma_i: float = 4.0,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the structure-tensor orientation field of an intensity tile.

    Args:
        intensity: 2-D intensity array (e.g. ``image.detect_mat[:]`` crop).
        sigma_d: Gaussian-derivative (gradient) scale in pixels, ~ hypha width.
        sigma_i: Structure-tensor integration scale in pixels.
        eps: Numerical floor for the coherence denominator.

    Returns:
        ``(phi, coherence, grad_phi)`` — orientation in radians in
        ``(-pi/2, pi/2]``, coherence in ``[0, 1]``, and the doubled-angle
        (pi-safe) orientation-gradient magnitude ``|grad phi|`` in rad/px.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    # Gaussian-derivative gradients at the derivative scale. scipy `order` is
    # per-axis (axis0=rows=y, axis1=cols=x): (0,1) -> d/dx, (1,0) -> d/dy.
    Ix = gaussian_filter(intensity, sigma_d, order=(0, 1))
    Iy = gaussian_filter(intensity, sigma_d, order=(1, 0))
    # Structure-tensor components smoothed at the integration scale.
    Jxx = gaussian_filter(Ix * Ix, sigma_i)
    Jyy = gaussian_filter(Iy * Iy, sigma_i)
    Jxy = gaussian_filter(Ix * Iy, sigma_i)
    # Dominant orientation via the doubled angle.
    phi = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    # Coherence (anisotropy) in [0, 1].
    coherence = np.sqrt((Jyy - Jxx) ** 2 + 4.0 * Jxy ** 2) / (Jxx + Jyy + eps)
    coherence = np.clip(coherence, 0.0, 1.0)
    # |grad phi| via the doubled-angle representation (pi-safe): the field is
    # 2phi, and |grad phi| = 1/2 * |grad(2phi)| recovered from cos2phi/sin2phi.
    c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
    gc_y, gc_x = np.gradient(c2)
    gs_y, gs_x = np.gradient(s2)
    grad_phi = 0.5 * np.sqrt(gc_x ** 2 + gc_y ** 2 + gs_x ** 2 + gs_y ** 2)
    return phi, coherence, grad_phi
