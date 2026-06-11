"""Pure, unit-testable robust colorimetric estimators used by MeasureColor.

Free of Image/accessor dependencies so they can be tested in isolation. The
robust center reuses the verified ``phenotypic.util.geometric_median`` (the
``cohen`` method is unimplemented, so we always pin ``method='weiszfeld'``).
See docs/superpowers/specs/2026-06-10-robust-lab-color-measures-design.md.
"""
from __future__ import annotations

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
