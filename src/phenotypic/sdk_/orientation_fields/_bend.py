"""Mask-aware multiscale bend of a two-dimensional fiber director field."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def fiber_bend_field(
    phi: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
    sigma_q: float,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate director-invariant fiber bend at one spatial scale.

    ``phi`` is the gradient-normal axis returned by the structure tensor. The
    local fiber axis is perpendicular, so its doubled-angle components are the
    negatives of the gradient-normal components. Coherence-weighted components
    are Gaussian-averaged only over ``selector`` and differentiated through
    their doubled-angle phase. This avoids the axial ``theta``/``theta + pi``
    seam.

    Bend is the curvature magnitude of the director-field integral curves,
    ``|(n . grad)n| = |n . grad(theta)|``. It is nonnegative and has units of
    radians per pixel. The returned scale resultant reports how consistently
    one axial direction survives averaging at ``sigma_q``.

    Args:
        phi: Gradient-normal axial orientation in radians, shaped ``(H, W)``.
        coherence: Structure-tensor coherence in ``[0, 1]``, shaped ``(H, W)``.
        selector: Boolean pixels allowed to influence the scale-local field.
        sigma_q: Gaussian standard deviation for averaging the fiber
            doubled-angle field, in pixels.
        eps: Positive numerical floor for normalized convolution and phase
            derivatives.

    Returns:
        ``(bend_magnitude, scale_resultant)`` arrays with the input shape.
        Values outside ``selector`` or without finite weighted support are
        ``NaN``.

    Raises:
        ValueError: If inputs are not equally shaped two-dimensional arrays,
            ``sigma_q`` is not finite and positive, or ``eps`` is invalid.
    """
    phi = np.asarray(phi, dtype=np.float64)
    coherence = np.asarray(coherence, dtype=np.float64)
    selector = np.asarray(selector, dtype=bool)
    if phi.ndim != 2:
        raise ValueError("fiber-bend arrays must be two-dimensional")
    if coherence.shape != phi.shape or selector.shape != phi.shape:
        raise ValueError("fiber-bend arrays must share one shape")
    if not np.isfinite(sigma_q) or sigma_q <= 0:
        raise ValueError("sigma_q must be finite and > 0")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0")

    reliable = (
        selector
        & np.isfinite(phi)
        & np.isfinite(coherence)
        & (coherence > 0.0)
    )
    weights = np.where(reliable, coherence, 0.0)
    fiber_cosine = -np.cos(2.0 * np.where(reliable, phi, 0.0))
    fiber_sine = -np.sin(2.0 * np.where(reliable, phi, 0.0))
    smooth_weight = gaussian_filter(
        weights,
        sigma_q,
        mode="constant",
        cval=0.0,
    )
    q_cosine = gaussian_filter(
        weights * fiber_cosine,
        sigma_q,
        mode="constant",
        cval=0.0,
    )
    q_sine = gaussian_filter(
        weights * fiber_sine,
        sigma_q,
        mode="constant",
        cval=0.0,
    )
    q_norm_squared = q_cosine * q_cosine + q_sine * q_sine
    q_norm = np.sqrt(q_norm_squared)
    scale_resultant = np.divide(
        q_norm,
        smooth_weight,
        out=np.full_like(q_norm, np.nan),
        where=smooth_weight > eps,
    )
    scale_resultant = np.clip(scale_resultant, 0.0, 1.0)

    cosine_y, cosine_x = np.gradient(q_cosine)
    sine_y, sine_x = np.gradient(q_sine)
    phase_denominator = q_norm_squared + eps
    theta_x = 0.5 * (q_cosine * sine_x - q_sine * cosine_x) / phase_denominator
    theta_y = 0.5 * (q_cosine * sine_y - q_sine * cosine_y) / phase_denominator
    theta = 0.5 * np.arctan2(q_sine, q_cosine)
    director_x = np.cos(theta)
    director_y = np.sin(theta)
    bend_magnitude = np.abs(director_x * theta_x + director_y * theta_y)

    valid = reliable & (smooth_weight > eps) & (q_norm_squared > eps)
    bend_magnitude = np.where(valid, bend_magnitude, np.nan)
    scale_resultant = np.where(valid, scale_resultant, np.nan)
    return bend_magnitude, scale_resultant
