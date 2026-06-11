"""Pure, unit-testable robust colorimetric estimators used by MeasureColor.

Free of Image/accessor dependencies so they can be tested in isolation. The
robust center reuses the verified ``phenotypic.util.geometric_median`` (the
``cohen`` method is unimplemented, so we always pin ``method='weiszfeld'``).
See docs/superpowers/specs/2026-06-10-robust-lab-color-measures-design.md.
"""
from __future__ import annotations

import colour
import numpy as np

from phenotypic.util._geometric_median import geometric_median as _geometric_median


def robust_color_center(
    points: np.ndarray, max_iter: int = 50, tol: float = 1e-4
) -> np.ndarray:
    """Euclidean geometric median of ``points`` (N, D), as a bare (D,) array.

    Reuses ``phenotypic.util.geometric_median`` (Weiszfeld). Returns all-NaN for
    empty input and the sole point for ``N == 1`` (the underlying solver requires
    ``N >= 1`` and a defined centroid).

    Args:
        points: (N, D) coordinates (Lab pixels, or HSV cone coordinates).
        max_iter: Weiszfeld iteration cap.
        tol: Convergence tolerance (forwarded as ``eps``).

    Returns:
        (D,) geometric-median coordinate.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be 2-D (N, D)")
    n, d = points.shape
    if n == 0:
        return np.full(d, np.nan)
    if n == 1:
        return points[0].copy()
    center, _info = _geometric_median(
        points, method="weiszfeld", eps=tol, max_iter=max_iter, verbose=False
    )
    return np.asarray(center, dtype=np.float64)


def medoid_ciede2000(
    lab_points: np.ndarray, max_pixels: int = 1000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """ΔE2000 medoid center and per-pixel ΔE2000 distances to it.

    The medoid (real pixel minimizing total ΔE2000) is selected from a seeded
    subsample of at most ``max_pixels`` (the selection is O(m^2)); the returned
    distances are computed from the chosen medoid to **all** input pixels.

    Args:
        lab_points: (N, 3) CIE L*a*b* pixel vectors.
        max_pixels: Subsample cap for medoid selection.
        seed: RNG seed for reproducible subsampling.

    Returns:
        (center (3,), all_deltas (N,)). center is all-NaN and all_deltas empty
        when ``lab_points`` is empty.
    """
    lab = np.asarray(lab_points, dtype=np.float64)
    n = lab.shape[0]
    if n == 0:
        return np.full(3, np.nan), np.empty(0)
    if n == 1:
        return lab[0].copy(), np.zeros(1)

    if n > max_pixels:
        rng = np.random.default_rng(seed)
        sample = lab[rng.choice(n, size=max_pixels, replace=False)]
    else:
        sample = lab

    pairwise = colour.difference.delta_E_CIE2000(
        sample[:, None, :], sample[None, :, :]
    )
    medoid = sample[pairwise.sum(axis=1).argmin()]
    all_deltas = np.asarray(colour.difference.delta_E_CIE2000(lab, medoid))
    return medoid, all_deltas


def delta_e2000_spread(deltas: np.ndarray) -> tuple[float, float, float]:
    """Return (median, mean, P95) of a ΔE2000 distance array; NaNs if empty."""
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(deltas)),
        float(np.mean(deltas)),
        float(np.percentile(deltas, 95)),
    )
