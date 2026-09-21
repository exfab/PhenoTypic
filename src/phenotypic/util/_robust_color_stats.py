"""Pure, unit-testable robust colorimetric estimators used by MeasureColor.

Free of Image/accessor dependencies so they can be tested in isolation. The
robust center reuses the verified ``phenotypic.util.geometric_median`` (the
``cohen`` method is unimplemented, so we always pin ``method='weiszfeld'``).
See docs/superpowers/specs/2026-06-10-robust-lab-color-measures-design.md.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from phenotypic.util._geometric_median import geometric_median as _geometric_median

_EPS = 1e-12

#: Candidates scored against every pixel by :func:`candidate_medoid`.  256 is
#: ~60x cheaper than the exhaustive form and returned the identical pixel on
#: every tile-like cloud tested; see the spec's Stage E.
DEFAULT_MEDOID_CANDIDATES = 256

#: Weiszfeld settings for the seed geometric median, matching the constants
#: ``ColorCheckerProfile`` uses so the two paths agree.
GEOMEDIAN_MAX_ITER = 200
GEOMEDIAN_TOL = 1e-6

#: A winner this deep into the candidate set means the cloud is not unimodal
#: around its geometric median, so the true medoid may lie outside the set and
#: the search is re-run once against a four-times-larger set.
#:
#: The guard is a safety net at the default candidate count, not at any count.
#: Measured on a deliberately bimodal cloud (900 pixels at one colour, 300 at
#: another 30 a\* away): at ``k >= 256`` it fires and recovers the exhaustive
#: medoid, but at ``k <= 64`` the winner sits at rank 3 -- comfortably inside
#: the set -- while being the wrong pixel, so nothing fires. Do not lower
#: ``candidates`` below the default expecting the guard to compensate.
CANDIDATE_EDGE_FRACTION = 0.8

#: Byte budget for one block of :func:`candidate_medoid`'s ΔE2000 scoring.
#: Sizing the block from it, rather than from a fixed candidate count, keeps a
#: very large object (a merged lawn, a whole-plate mis-detection) from
#: allocating gigabytes of CIEDE2000 temporaries.
MEDOID_MEMORY_BUDGET_BYTES = 256 * 2**20

#: Peak bytes allocated per candidate-pixel pair while scoring one block.
#: colour's CIEDE2000 materialises many ``(block, N)`` float64 temporaries.
#: Measured with ``tracemalloc`` on a 20 000-pixel unimodal L*a*b* cloud
#: (macOS arm64, numpy 2, colour-science 0.4): peak over ``chunk_size * N`` is
#: 264.6 B at ``chunk_size=64`` and 266.5 B at 16, the difference between the
#: two giving a slope of 264.0 B/pair plus ~0.8 MB of per-call overhead.
#: Rounded up to 265 so the block errs small.
MEDOID_BYTES_PER_PAIR = 265


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
    lab_points: np.ndarray,
    max_pixels: int = 1000,
    seed: int = 0,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """ΔE2000 medoid center and per-pixel ΔE2000 distances to it.

    The medoid (real pixel minimizing total ΔE2000) is selected from a seeded
    subsample of at most ``max_pixels``; the returned distances are computed from
    the chosen medoid to **all** input pixels.

    The total-distance ("row sum") used to pick the medoid is accumulated in
    candidate blocks of ``chunk_size`` rows rather than materializing the full
    ``m x m`` pairwise matrix. Peak memory is therefore ``O(chunk_size * m)``
    instead of ``O(m^2)`` (CIEDE2000 allocates many intermediate arrays the size
    of its broadcast grid). The result is bit-identical to the full-matrix form
    for any ``chunk_size`` -- chunking bounds memory, not accuracy.

    Args:
        lab_points: (N, 3) CIE L*a*b* pixel vectors.
        max_pixels: Subsample cap for medoid selection.
        seed: RNG seed for reproducible subsampling.
        chunk_size: Number of candidate rows scored per block; caps peak memory.

    Returns:
        (center (3,), all_deltas (N,)). center is all-NaN and all_deltas empty
        when ``lab_points`` is empty.
    """
    import colour

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

    # Accumulate each candidate's total ΔE2000 to all sample points in blocks,
    # so we never allocate the full (m, m) pairwise matrix at once.
    m = sample.shape[0]
    row_sums = np.empty(m, dtype=np.float64)
    for start in range(0, m, chunk_size):
        block = sample[start : start + chunk_size]  # (b, 3)
        block_dists = colour.difference.delta_E_CIE2000(
            block[:, None, :], sample[None, :, :]
        )  # (b, m)
        row_sums[start : start + chunk_size] = np.asarray(block_dists).sum(axis=1)

    medoid = sample[row_sums.argmin()]
    all_deltas = np.asarray(colour.difference.delta_E_CIE2000(lab, medoid))
    return medoid, all_deltas


def _delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """delta-E 2000 between broadcastable ``(..., 3)`` Lab arrays."""
    import colour

    return np.asarray(colour.difference.delta_E_CIE2000(lab_a, lab_b))


class MedoidResult(NamedTuple):
    """Outcome of a candidate-restricted medoid search.

    Attributes:
        index: Row of the winning pixel in the input array.
        lab: The winning pixel's Lab coordinate.
        rank: The winner's position within the candidate set, 0 being the
            pixel closest to the geometric median.
        total_delta_e: The winner's total delta-E 2000 to every input pixel.
        widened: ``True`` when the first search put the winner near the edge
            of the candidate set and the search was re-run with a larger one.
    """

    index: int
    lab: np.ndarray
    rank: int
    total_delta_e: float
    widened: bool


def candidate_medoid(
        lab_points: np.ndarray,
        k: int = DEFAULT_MEDOID_CANDIDATES,
        chunk_size: int | None = None,
) -> MedoidResult:
    """The delta-E 2000 medoid of *lab_points*, found deterministically.

    Scores only the *k* pixels nearest the cloud's Lab geometric median, each
    against **every** pixel, and returns the best.  Cost is ``O(k * n)`` rather
    than the exhaustive ``O(n^2)``, and no pixel is ever sampled at random.

    Why not :func:`~phenotypic.util.medoid_ciede2000`: it selects the medoid
    from a seeded subsample of 1000 pixels, and re-drawing that seed moves the
    answer by ~0.12 delta-E 2000 (worst 0.81) -- larger than the difference
    between the medoid and the geometric median it is chosen over.  Raising the
    cap does not fix it affordably: selection is quadratic, costing ~41 s per
    24-tile frame at a 4000-pixel cap, while the seed spread only falls from
    ~0.32 to ~0.25 delta-E 2000.  Restricting the candidates instead removes
    the randomness entirely and reproduces the exhaustive medoid exactly on
    unimodal tile clouds.

    Args:
        lab_points: ``(N, 3)`` CIE Lab pixels.
        k: Number of candidates to score.  Widened once, automatically, if the
            winner lands past :data:`CANDIDATE_EDGE_FRACTION` of the set.
        chunk_size: Candidates scored per block, bounding peak memory to
            ``O(chunk_size * N)`` instead of ``O(k * N)``.  ``None`` (the
            default) sizes the block from :data:`MEDOID_MEMORY_BUDGET_BYTES`:
            ``max(1, min(64, budget // (N * MEDOID_BYTES_PER_PAIR)))``.  The
            result is bit-identical for every block size -- each candidate's
            total is summed over the same full row -- so this bounds memory and
            nothing else.  Below one candidate per block the floor is one
            ``(1, N)`` row, ~265 B per pixel.

    Returns:
        A :class:`MedoidResult`.  For an empty input the index is ``-1`` and
        the coordinate is all-NaN; for a single pixel it is that pixel.

    Raises:
        ValueError: If *lab_points* is not 2-D, or *k* or *chunk_size* is
            not positive.
    """
    points = np.asarray(lab_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"lab_points must be (N, 3); got {points.shape}.")
    if k < 1:
        raise ValueError(f"k must be a positive candidate count; got {k}.")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}.")

    n = points.shape[0]
    if n == 0:
        return MedoidResult(-1, np.full(3, np.nan), -1, float("nan"), False)
    if n == 1:
        return MedoidResult(0, points[0].copy(), 0, 0.0, False)

    seed = robust_color_center(
            points, max_iter=GEOMEDIAN_MAX_ITER, tol=GEOMEDIAN_TOL
    )
    order = np.argsort(np.linalg.norm(points - seed, axis=1), kind="stable")
    if chunk_size is None:
        chunk_size = max(
                1, min(64, MEDOID_MEMORY_BUDGET_BYTES // (n * MEDOID_BYTES_PER_PAIR))
        )

    attempts = (min(k, n), min(max(4 * k, k), n))
    widened = False
    for attempt_no, attempt_k in enumerate(attempts):
        candidate_idx = order[:attempt_k]
        totals = np.empty(attempt_k, dtype=np.float64)
        for start in range(0, attempt_k, chunk_size):
            block = points[candidate_idx[start : start + chunk_size]]
            totals[start : start + chunk_size] = _delta_e(
                    block[:, None, :], points[None, :, :]
            ).sum(axis=1)
        rank = int(totals.argmin())
        at_edge = rank >= CANDIDATE_EDGE_FRACTION * attempt_k
        last_attempt = attempt_no == len(attempts) - 1
        # Always return on the final attempt: a cloud that is still at the
        # edge of a widened set has no better answer available here, and
        # falling through would be a crash rather than a degraded result.
        if not at_edge or attempt_k >= n or last_attempt:
            winner = int(candidate_idx[rank])
            return MedoidResult(
                    winner, points[winner].copy(), rank, float(totals[rank]), widened
            )
        widened = True

    raise AssertionError("unreachable")  # pragma: no cover


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


def hsv_to_cone(hsv: np.ndarray) -> np.ndarray:
    """Embed HSV (H,S,V in [0,1]) into Cartesian cone coords (S*V*cosθ, S*V*sinθ, V)."""
    hsv = np.asarray(hsv, dtype=np.float64)
    theta = 2.0 * np.pi * hsv[..., 0]
    chroma = hsv[..., 1] * hsv[..., 2]
    x = chroma * np.cos(theta)
    y = chroma * np.sin(theta)
    z = hsv[..., 2]
    return np.stack([x, y, z], axis=-1)


def cone_to_hsv(cone: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hsv_to_cone`; returns H,S,V in [0,1]."""
    cone = np.asarray(cone, dtype=np.float64)
    x, y, z = cone[..., 0], cone[..., 1], cone[..., 2]
    hue = (np.arctan2(y, x) / (2.0 * np.pi)) % 1.0
    chroma = np.sqrt(x * x + y * y)
    value = z
    sat = np.where(value > _EPS, np.clip(chroma / np.where(value > _EPS, value, 1.0), 0.0, 1.0), 0.0)
    return np.stack([hue, sat, value], axis=-1)


def lab_to_srgb_hex(lab: np.ndarray) -> str:
    """Convert a single CIE L*a*b* (D65) color to an sRGB ``#RRGGBB`` string.

    Returns ``""`` if any coordinate is NaN (e.g. an empty object).
    """
    import colour

    lab = np.asarray(lab, dtype=np.float64)
    if np.isnan(lab).any():
        return ""
    xyz = colour.Lab_to_XYZ(lab)
    srgb = np.clip(colour.XYZ_to_sRGB(xyz), 0.0, 1.0)
    r, g, b = (np.round(srgb * 255.0).astype(int))
    return f"#{r:02x}{g:02x}{b:02x}"
