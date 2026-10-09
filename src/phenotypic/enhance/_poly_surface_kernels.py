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

    Rousseeuw, P. J. and Croux, C. (1993). Alternatives to the median absolute
    deviation. Journal of the American Statistical Association 88, 1273-1283.
    doi:10.1080/01621459.1993.10476408
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, NamedTuple

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


class RobustFit(NamedTuple):
    """Result of :func:`robust_least_squares`.

    Attributes:
        coef: Coefficient vector, shape ``(n_terms,)``.
        kept: Boolean mask, shape ``(n_points,)``, of the final inlier set.
        rounds: Refits performed after the initial fit, in ``0..max_iter``.
    """

    coef: np.ndarray
    kept: np.ndarray
    rounds: int


def robust_least_squares(
    a: np.ndarray,
    z: np.ndarray,
    *,
    clip_sigma: float,
    max_iter: int,
) -> RobustFit:
    """Fit ``a @ x ~= z`` by iterative sigma-clipped least squares (spec 4.3).

    The first fit uses every point. Each round computes residuals at all points,
    then the median and the normal-consistent median absolute deviation (Rousseeuw
    and Croux 1993) of the residuals of the current inliers, and removes inliers
    farther than ``clip_sigma`` scales from that median. A clipped point never
    returns. The loop stops, keeping the latest accepted fit, when the scale is
    zero, when a round would leave fewer inliers than terms (drift D3), when no
    point is clipped, or after ``max_iter`` refits.

    Args:
        a: Design matrix, shape ``(n_points, n_terms)``.
        z: Observations, shape ``(n_points,)``.
        clip_sigma: Clipping threshold in units of the MAD-derived scale.
        max_iter: Maximum number of refits after the initial fit.

    Returns:
        The coefficients, final inlier mask and number of refits performed.
    """
    from scipy.stats import median_abs_deviation

    n_terms = a.shape[1]
    kept = np.ones(a.shape[0], dtype=bool)
    coef = solve_least_squares(a, z)
    rounds = 0
    while rounds < max_iter:
        residual = z - a @ coef
        inlier_residual = residual[kept]
        center = np.median(inlier_residual)
        scale = median_abs_deviation(inlier_residual, scale="normal")
        if scale == 0:
            break
        new = kept & (np.abs(residual - center) <= clip_sigma * scale)
        count = int(new.sum())
        if count < n_terms or count == int(kept.sum()):
            break
        kept = new
        coef = solve_least_squares(a[kept], z[kept])
        rounds += 1
    return RobustFit(coef=coef, kept=kept, rounds=rounds)


def fit_surface_coefficients(
    z: np.ndarray,
    *,
    degree: int,
    independent: bool,
    fit: str,
    clip_sigma: float,
    max_iter: int,
    max_fit_points: int = MAX_FIT_POINTS,
) -> np.ndarray:
    """Fit the polynomial surface to ``z``, subsampling large images (spec 4.8).

    When ``z.size`` exceeds ``max_fit_points`` the fit uses a strided subsample
    with stride ``s = ceil(sqrt(z.size / max_fit_points))``, clamped per axis to
    ``max(1, (n_axis - 1) // max(degree, 1))`` so that each axis keeps at least
    ``degree + 1`` distinct coordinates. Subsampled points keep their full-grid
    normalized coordinates (drift D4).

    Args:
        z: 2-D image, shape ``(height, width)``.
        degree: Polynomial degree.
        independent: Tensor-product (True) or total-degree (False) term set.
        fit: ``"lstsq"`` or ``"robust"``.
        clip_sigma: Clipping threshold for ``fit="robust"``.
        max_iter: Maximum refits for ``fit="robust"``.
        max_fit_points: Pixel count above which the image is subsampled.

    Returns:
        Coefficients in :func:`term_powers` order.
    """
    height, width = z.shape
    stride = 1
    if z.size > max_fit_points:
        stride = math.ceil(math.sqrt(z.size / max_fit_points))
    reach = max(degree, 1)
    stride_v = min(stride, max(1, (height - 1) // reach))
    stride_u = min(stride, max(1, (width - 1) // reach))
    v = normalized_axis(height)[::stride_v]
    u = normalized_axis(width)[::stride_u]
    sample = np.asarray(z[::stride_v, ::stride_u], dtype=np.float64)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    terms = term_powers(degree, independent)
    a = legendre_design(uu.ravel(), vv.ravel(), degree, terms)
    if fit == "robust":
        return robust_least_squares(
            a, sample.ravel(), clip_sigma=clip_sigma, max_iter=max_iter
        ).coef
    return solve_least_squares(a, sample.ravel())


#: Rows per batch in the vectorised per-line robust fit (spec 4.8).
_LINE_BLOCK: Final[int] = 256

#: Divisor making the median absolute deviation consistent with a normal sigma.
_MAD_NORMAL: Final[float] = 1.482602218505602


def _robust_line_block(
    block: np.ndarray,
    a: np.ndarray,
    outer: np.ndarray,
    *,
    clip_sigma: float,
    max_iter: int,
) -> np.ndarray:
    """Return fitted values for a block of rows, each fitted by spec 4.3 on its own.

    Every row keeps its own inlier mask, stopping state and fit; a row that stops
    keeps its last accepted fit while the others continue. Refits solve the
    stacked weighted normal equations, which stay well conditioned for Legendre
    columns on ``[-1, 1]`` at the supported line orders.

    Args:
        block: Rows to fit, shape ``(rows, width)``.
        a: Line design matrix, shape ``(width, n_terms)``.
        outer: Row-wise outer products of ``a``, shape ``(width, n_terms ** 2)``.
        clip_sigma: Clipping threshold in units of the MAD-derived scale.
        max_iter: Maximum number of refits after the initial fit.

    Returns:
        float64 array of fitted values, shape ``(rows, width)``.
    """
    n_terms = a.shape[1]
    coef = solve_least_squares(a, block.T).T
    kept = np.ones(block.shape, dtype=bool)
    active = np.arange(block.shape[0])
    for _ in range(max_iter):
        if active.size == 0:
            break
        z = block[active]
        mask = kept[active]
        residual = z - coef[active] @ a.T
        masked = np.where(mask, residual, np.nan)
        center = np.nanmedian(masked, axis=1, keepdims=True)
        scale = _MAD_NORMAL * np.nanmedian(np.abs(masked - center), axis=1)
        new = mask & (np.abs(residual - center) <= clip_sigma * scale[:, None])
        count = new.sum(axis=1)
        refit = (scale > 0) & (count >= n_terms) & (count != mask.sum(axis=1))
        active = active[refit]
        if active.size == 0:
            break
        new = new[refit]
        kept[active] = new
        weight = new.astype(np.float64)
        gram = (weight @ outer).reshape(-1, n_terms, n_terms)
        rhs = (weight * block[active]) @ a
        coef[active] = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
    return coef @ a.T


def level_lines(
    z: np.ndarray,
    *,
    line_order: int,
    fit: str,
    clip_sigma: float,
    max_iter: int,
) -> np.ndarray:
    """Level every row of ``z`` by its own polynomial, restoring the global mean.

    Spec 4.5. The global mean is taken before leveling; each row is fitted with a
    Legendre polynomial of order ``line_order`` over the normalized column axis,
    and the output is ``z - fit + mean``. ``fit="lstsq"`` solves all rows in one
    call; ``fit="robust"`` applies the spec 4.3 sigma-clipped loop independently
    to each row, in blocks of rows (lines are never subsampled, spec 4.8). To
    level columns, pass the transpose.

    Args:
        z: 2-D image, shape ``(height, width)``.
        line_order: Polynomial order along each row.
        fit: ``"lstsq"`` or ``"robust"``.
        clip_sigma: Clipping threshold for ``fit="robust"``.
        max_iter: Maximum refits per row for ``fit="robust"``.

    Returns:
        New float64 array of shape ``(height, width)``.
    """
    z = np.asarray(z, dtype=np.float64)
    avg = z.mean()
    a = _legendre.legvander(normalized_axis(z.shape[1]), line_order)
    if fit == "robust":
        n_terms = a.shape[1]
        outer = (a[:, :, None] * a[:, None, :]).reshape(a.shape[0], n_terms * n_terms)
        fitted = np.empty_like(z)
        for start in range(0, z.shape[0], _LINE_BLOCK):
            stop = start + _LINE_BLOCK
            fitted[start:stop] = _robust_line_block(
                z[start:stop], a, outer, clip_sigma=clip_sigma, max_iter=max_iter
            )
    else:
        fitted = (a @ solve_least_squares(a, z.T)).T
    return z - fitted + avg


_METHODS: Final[frozenset[str]] = frozenset({"offset", "plane", "polynomial", "line"})
_FITS: Final[frozenset[str]] = frozenset({"lstsq", "robust"})
_LINE_AXES: Final[frozenset[str]] = frozenset({"row", "column"})


def _validate_flatten_request(
    z: np.ndarray,
    *,
    method: str,
    order: int,
    line_order: int,
    line_axis: str,
    fit: str,
) -> None:
    """Reject unknown option strings and geometries too small for the request (spec 4.7).

    Raises:
        ValueError: On an unknown ``method``, ``fit`` or ``line_axis``; on an image
            with fewer than 2 rows or columns; on a polynomial ``order`` needing more
            samples than an axis has; or on a ``line_order`` needing more samples
            than a line has.
    """
    for name, value, allowed in (
        ("method", method, _METHODS),
        ("fit", fit, _FITS),
        ("line_axis", line_axis, _LINE_AXES),
    ):
        if value not in allowed:
            raise ValueError(
                f"unknown {name} {value!r}; expected one of {sorted(allowed)}"
            )
    height, width = z.shape
    if height < 2 or width < 2:
        raise ValueError(
            f"image must have at least 2 rows and 2 columns, got {z.shape}"
        )
    if method == "polynomial" and min(height, width) < order + 1:
        raise ValueError(
            f"polynomial order {order} needs at least {order + 1} samples per axis, "
            f"got {height} x {width}"
        )
    if method == "line":
        length = width if line_axis == "row" else height
        if length < line_order + 1:
            raise ValueError(
                f"line_order {line_order} needs at least {line_order + 1} samples "
                f"per line, got {length} along the {line_axis} axis"
            )


def flatten_surface(
    z: np.ndarray,
    *,
    method: str,
    order: int,
    independent: bool,
    line_order: int,
    line_axis: str,
    fit: str,
    clip_sigma: float,
    max_iter: int,
    max_fit_points: int = MAX_FIT_POINTS,
) -> np.ndarray:
    """Flatten ``z`` by one of the four background-removal methods (spec 4.4-4.7).

    Output levels follow spec 4.6: ``offset`` and ``polynomial`` land the background
    at zero, ``line`` restores the input mean, and ``plane`` removes the fitted tilt
    about the pivot ``(W / 2, H / 2)`` (drift D7), discarding the fitted constant.

    Args:
        z: 2-D image, shape ``(height, width)``; never modified.
        method: ``"offset"``, ``"plane"``, ``"polynomial"`` or ``"line"``.
        order: Polynomial degree for ``method="polynomial"``.
        independent: Tensor-product (True) or total-degree (False) term set.
        line_order: Per-line polynomial order for ``method="line"``.
        line_axis: ``"row"`` or ``"column"``: which lines are leveled.
        fit: ``"lstsq"`` or ``"robust"``.
        clip_sigma: Clipping threshold for ``fit="robust"``.
        max_iter: Maximum refits for ``fit="robust"``.
        max_fit_points: Pixel count above which a surface fit is subsampled.

    Returns:
        New float64 array of shape ``(height, width)``.

    Raises:
        ValueError: On unknown option strings or a geometry too small for the
            request (see :func:`_validate_flatten_request`).
    """
    z = np.asarray(z, dtype=np.float64)
    _validate_flatten_request(
        z,
        method=method,
        order=order,
        line_order=line_order,
        line_axis=line_axis,
        fit=fit,
    )
    height, width = z.shape

    def surface_coefficients(degree: int, tensor: bool) -> np.ndarray:
        return fit_surface_coefficients(
            z,
            degree=degree,
            independent=tensor,
            fit=fit,
            clip_sigma=clip_sigma,
            max_iter=max_iter,
            max_fit_points=max_fit_points,
        )

    def level(image: np.ndarray) -> np.ndarray:
        return level_lines(
            image,
            line_order=line_order,
            fit=fit,
            clip_sigma=clip_sigma,
            max_iter=max_iter,
        )

    if method == "offset":
        if fit == "lstsq":
            return z - z.mean()
        coef = surface_coefficients(0, True)
        return z - coef[0]
    if method == "plane":
        coef = surface_coefficients(1, False)
        slope_x = 2.0 * coef[2] / (width - 1)
        slope_y = 2.0 * coef[1] / (height - 1)
        rows = np.arange(height, dtype=np.float64)[:, None] - height / 2
        cols = np.arange(width, dtype=np.float64)[None, :] - width / 2
        return z - slope_x * cols - slope_y * rows
    if method == "polynomial":
        coef = surface_coefficients(order, independent)
        terms = term_powers(order, independent)
        return z - evaluate_surface(coef, terms, order, height, width)
    if line_axis == "column":
        return level(z.T).T.copy()
    return level(z)
