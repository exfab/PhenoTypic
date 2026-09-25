"""Seam-safe differences between axial (period-pi) angles.

An axial angle describes a line, not a direction, so ``theta`` and
``theta + pi`` are the same fiber axis. Differences are therefore wrapped with
the doubled-angle form ``0.5 * arctan2(sin 2d, cos 2d)`` into
``[-pi/2, pi/2]``.

An exactly 90-degree change lands on the ``+-pi`` branch cut of ``arctan2``,
where the sign of ``sin 2d`` is a few-ulp rounding residue that differs between
CPUs. Callers that keep a signed change must decide what an orthogonal change
means. :func:`axial_change` stores it canonically as ``+pi/2`` and
:func:`signed_axial_mean` counts it as directionless; other callers instead
drop the orthogonal step (``literal_crossing_ring_profile``,
``matched_ring_cumulative_rotation_profile``) or use it only through
sign-invariant doubled-angle sums (the Method B radial-tilt resultant).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Numerical floor shared by the orientation-field helpers in this package.
_EPS = 1e-9


def axial_difference(outer: Any, inner: Any) -> Any:
    """Return the signed axial difference ``outer - inner`` in ``[-pi/2, pi/2]``.

    The value at exactly ``+-pi/2`` has no reliable sign. Use
    :func:`axial_change` when the signed result is kept.

    Args:
        outer: Axial angle(s) in radians. Scalars and arrays broadcast.
        inner: Axial angle(s) in radians subtracted from ``outer``.

    Returns:
        The wrapped difference with the broadcast shape of the inputs; a NumPy
        scalar for scalar inputs.
    """
    difference = outer - inner
    return 0.5 * np.arctan2(
        np.sin(2.0 * difference),
        np.cos(2.0 * difference),
    )


def is_orthogonal_change(change: Any, atol: float = _EPS) -> Any:
    """Return where a wrapped axial change is exactly 90 degrees within ``atol``.

    Args:
        change: Wrapped axial change(s) in radians.
        atol: Absolute radian tolerance around ``pi/2``.

    Returns:
        Boolean array (or NumPy bool for a scalar input).
    """
    return np.isclose(np.abs(change), np.pi / 2.0, atol=atol, rtol=0.0)


def axial_change(outer: Any, inner: Any) -> np.ndarray:
    """Return the platform-independent signed axial change ``outer - inner``.

    The doubled-angle wrap puts an exactly 90-degree change on the ``+-pi``
    branch cut of ``arctan2``, where the sign is decided by floating-point
    noise of a few ulps and so differs between CPUs. Such a change has no
    turning direction; it is stored canonically as ``+pi / 2`` so every
    intermediate array is platform-independent, and signed summaries treat it
    as directionless via :func:`signed_axial_mean`.

    Args:
        outer: Axial angle(s) in radians.
        inner: Axial angle(s) in radians subtracted from ``outer``.

    Returns:
        Float64 array of changes in ``(-pi/2, pi/2]``.
    """
    change = axial_difference(
        np.asarray(outer, dtype=np.float64),
        np.asarray(inner, dtype=np.float64),
    )
    return np.where(is_orthogonal_change(change), np.pi / 2.0, change)


def signed_axial_mean(changes: Any) -> float:
    """Mean signed axial change, counting a 90-degree change as directionless.

    An orthogonal change contributes zero, exactly as a pair of opposing
    changes cancels, while still counting toward the mean's denominator; its
    magnitude remains in the absolute summaries.

    Args:
        changes: Signed axial changes in radians.

    Returns:
        The mean as a Python float.
    """
    changes = np.asarray(changes, dtype=np.float64)
    return float(
        np.mean(np.where(is_orthogonal_change(changes), 0.0, changes))
    )
