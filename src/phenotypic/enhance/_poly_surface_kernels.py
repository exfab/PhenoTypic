"""Private float64 kernels for :class:`SubtractPolySurface`.

Implements the source-free contract in ``design.md`` section 4 (coordinates, term
sets, Legendre design matrix, least-squares solve, surface evaluation). This is a
clean-room implementation: it was written from the specification and the published
method, with no Gwyddion source consulted. Drift-register rows D1 (Legendre basis
spans the same space as monomials) and D2 (rank-deficient design raises) apply.

References:
    Necas, D. and Klapetek, P. (2012). Gwyddion: an open-source software for SPM
    data analysis. Central European Journal of Physics 10, 181-188.
    doi:10.2478/s11534-011-0096-2
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.polynomial import legendre as _legendre

#: Upper bound on the number of pixels used in a single fit.
MAX_FIT_POINTS: Final[int] = 262_144


def normalized_axis(n: int) -> np.ndarray:
    """Return ``n`` samples spanning ``[-1, 1]`` as ``2k / (n - 1) - 1``.

    Args:
        n: Number of samples along the axis; must be at least 2.

    Returns:
        float64 array of length ``n`` from ``-1`` to ``+1`` inclusive.

    Raises:
        ValueError: If ``n`` is less than 2.
    """
    if n < 2:
        raise ValueError(f"axis needs at least 2 samples to normalize, got {n}")
    return 2.0 * np.arange(n, dtype=np.float64) / (n - 1) - 1.0


def term_powers(degree: int, independent: bool) -> tuple[tuple[int, int], ...]:
    """List the ``(p, q)`` powers of ``(u, v)`` in the fitted term set.

    Args:
        degree: Polynomial degree.
        independent: If True, every ``p`` and ``q`` in ``0..degree`` (tensor
            product); otherwise only ``p + q <= degree`` (total degree).

    Returns:
        Tuples ordered p-major, then q ascending.
    """
    return tuple(
        (p, q)
        for p in range(degree + 1)
        for q in range(degree + 1)
        if independent or p + q <= degree
    )


def legendre_design(
    u: np.ndarray,
    v: np.ndarray,
    degree: int,
    terms: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Build the design matrix whose column ``k`` is ``L_p(u) * L_q(v)``.

    Args:
        u: 1-D normalized x coordinates, one per point.
        v: 1-D normalized y coordinates, one per point.
        degree: Highest Legendre order required by ``terms``.
        terms: ``(p, q)`` pairs, as returned by :func:`term_powers`.

    Returns:
        float64 array of shape ``(n_points, n_terms)``.
    """
    vander_u = _legendre.legvander(u, degree)
    vander_v = _legendre.legvander(v, degree)
    p_index = np.fromiter((p for p, _ in terms), dtype=np.intp, count=len(terms))
    q_index = np.fromiter((q for _, q in terms), dtype=np.intp, count=len(terms))
    return vander_u[:, p_index] * vander_v[:, q_index]


def solve_least_squares(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Solve ``a @ x ~= z`` in the least-squares sense.

    Args:
        a: Design matrix, shape ``(n_points, n_terms)``.
        z: Observations, shape ``(n_points,)``.

    Returns:
        Coefficient vector of length ``n_terms``.

    Raises:
        ValueError: If ``a`` is rank deficient (drift D2).
    """
    coef, _, rank, _ = np.linalg.lstsq(a, z, rcond=None)
    if rank < a.shape[1]:
        raise ValueError(
            f"design matrix is rank deficient (rank {rank} < {a.shape[1]} terms); "
            "lower the polynomial degree or supply more distinct sample points"
        )
    return coef


def evaluate_surface(
    coef: np.ndarray,
    terms: Sequence[tuple[int, int]],
    degree: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Evaluate the fitted surface on the full ``height`` by ``width`` grid.

    Args:
        coef: Coefficients aligned with ``terms``.
        terms: ``(p, q)`` pairs, as returned by :func:`term_powers`.
        degree: Polynomial degree used for the fit.
        height: Output rows (the ``v`` axis).
        width: Output columns (the ``u`` axis).

    Returns:
        float64 array of shape ``(height, width)``.
    """
    coef_matrix = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    for value, (p, q) in zip(coef, terms):
        coef_matrix[q, p] = value
    return _legendre.leggrid2d(
        normalized_axis(height), normalized_axis(width), coef_matrix
    )
