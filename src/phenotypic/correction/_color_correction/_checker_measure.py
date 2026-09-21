"""Per-tile colour measurement for in-frame colour checkers.

Every statistic here consumes **CIE L\\*a\\*b\\*** that the caller obtained from
``Image.color.Lab``.  The prototype these functions are ported from fed linear
RawTherapee RGB into ``skimage.rgb2lab``, which expects sRGB encoding; its Lab
values -- and every threshold expressed in them -- were therefore a consistent
internal contrast measure rather than calibrated CIE quantities.  Converting
once, correctly, upstream is what makes the thresholds here mean what they say.

The patch colour itself is a **candidate-restricted delta-E 2000 medoid**: the
real pixel minimising total delta-E 2000 to every other pixel in the tile,
found by scoring only the pixels nearest the tile's Lab geometric median.  See
:func:`candidate_medoid` for why that replaces
:func:`~phenotypic.util.medoid_ciede2000` on this path.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from ...util._robust_color_stats import robust_color_center

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

#: delta-E 2000 beyond which a pixel counts as contamination rather than noise.
IMPURITY_DELTA_E = 6.0

#: Spatial median-filter width applied before the contamination statistics, so
#: they reflect an occluder crossing the tile rather than sensor noise.
CONTAMINATION_SMOOTH = 9


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
        chunk_size: int = 64,
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
            ``O(chunk_size * N)`` instead of ``O(k * N)``.

    Returns:
        A :class:`MedoidResult`.  For an empty input the index is ``-1`` and
        the coordinate is all-NaN; for a single pixel it is that pixel.

    Raises:
        ValueError: If *lab_points* is not 2-D, or *k* is not positive.
    """
    points = np.asarray(lab_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"lab_points must be (N, 3); got {points.shape}.")
    if k < 1:
        raise ValueError(f"k must be a positive candidate count; got {k}.")

    n = points.shape[0]
    if n == 0:
        return MedoidResult(-1, np.full(3, np.nan), -1, float("nan"), False)
    if n == 1:
        return MedoidResult(0, points[0].copy(), 0, 0.0, False)

    seed = robust_color_center(
            points, max_iter=GEOMEDIAN_MAX_ITER, tol=GEOMEDIAN_TOL
    )
    order = np.argsort(np.linalg.norm(points - seed, axis=1), kind="stable")

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


def extract_patch(array: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Slice ``(y0, y1, x0, x1)`` out of *array*, clipped to its bounds.

    Args:
        array: ``(H, W, C)`` image data.
        box: Float pixel bounds, rounded inward-safely with ``int()``.

    Returns:
        The ``(h, w, C)`` sub-array, possibly empty when the box misses.
    """
    height, width = array.shape[:2]
    y0, y1, x0, x1 = box
    ys, ye = max(0, int(y0)), min(height, int(y1))
    xs, xe = max(0, int(x0)), min(width, int(x1))
    if ye <= ys or xe <= xs:
        return array[0:0, 0:0]
    return array[ys:ye, xs:xe]


def _smoothed(patch: np.ndarray, size: int) -> np.ndarray:
    """Per-channel spatial median filter, returned flattened to ``(N, C)``."""
    from scipy.ndimage import median_filter

    if size <= 1:
        return patch.reshape(-1, patch.shape[-1])
    stack = np.dstack(
            [median_filter(patch[..., c], size=size) for c in range(patch.shape[-1])]
    )
    return stack.reshape(-1, patch.shape[-1])


def impurity(
        lab_patch: np.ndarray,
        threshold: float = IMPURITY_DELTA_E,
        smooth: int = CONTAMINATION_SMOOTH,
) -> float:
    """Fraction of a tile's pixels far from its own median colour.

    Answers *is something covering the card* -- an occluder, a border, a
    reflection crossing the tile.  Spatially smoothed first so sensor noise
    does not register.

    Args:
        lab_patch: ``(h, w, 3)`` Lab pixels of one tile's core box.
        threshold: delta-E 2000 beyond which a pixel counts as contamination.
        smooth: Median-filter width; ``<= 1`` disables smoothing.

    Returns:
        The contaminated fraction in ``[0, 1]``, or NaN for an empty patch.
    """
    if lab_patch.size == 0:
        return float("nan")
    flat = _smoothed(lab_patch, smooth)
    centre = np.median(flat, axis=0)
    return float((_delta_e(np.broadcast_to(centre, flat.shape), flat) > threshold).mean())


def robust_shift(
        lab_patch: np.ndarray,
        keep: float = 0.6,
        smooth: int = CONTAMINATION_SMOOTH,
) -> float:
    """How far a tile's median colour moves when atypical pixels are dropped.

    Answers *and did it move the answer*.  A per-channel median already absorbs
    a minority of outliers, so a tile can be visibly contaminated and still
    report the right colour; only this statistic is grounds to reject a band.

    Args:
        lab_patch: ``(h, w, 3)`` Lab pixels of one tile's core box.
        keep: Fraction of pixels closest to the tile centre retained for the
            comparison median.
        smooth: Median-filter width; ``<= 1`` disables smoothing.

    Returns:
        delta-E 2000 between the all-pixel median and the trimmed median, or
        NaN when the patch holds fewer than 20 pixels.
    """
    if lab_patch.size == 0:
        return float("nan")
    flat = _smoothed(lab_patch, smooth)
    if flat.shape[0] < 20:
        return float("nan")
    centre = np.median(flat, axis=0)
    distances = _delta_e(np.broadcast_to(centre, flat.shape), flat)
    selected = flat[distances <= np.quantile(distances, keep)]
    return float(_delta_e(centre, np.median(selected, axis=0)))


def clipped_fraction(
        rgb_patch: np.ndarray,
        low: float = 1.0 / 65535.0,
        high: float = 0.999,
) -> float:
    """Fraction of a tile's pixels with any channel pinned at the sensor limit.

    A third, independent fault: clipping is uniform rather than an outlier, so
    the median sits squarely on the clipped value and neither :func:`impurity`
    nor :func:`robust_shift` sees it.  No estimator recovers a value that was
    never recorded.

    Args:
        rgb_patch: ``(h, w, 3)`` RGB normalised to ``[0, 1]``.
        low: Floor at or below which a channel counts as clipped.
        high: Ceiling at or above which a channel counts as clipped.

    Returns:
        The clipped fraction in ``[0, 1]``, or NaN for an empty patch.
    """
    if rgb_patch.size == 0:
        return float("nan")
    pinned = (rgb_patch <= low) | (rgb_patch >= high)
    return float(pinned.any(axis=-1).mean())


class TileMeasurement(BaseModel):
    """One tile's measured colour and quality statistics.

    Attributes:
        row: Tile row within its ROI lattice.
        col: Tile column within its ROI lattice.
        roi_index: Index of the ROI this tile came from.
        n_pixels: Core-box pixel count the measurement used.
        lab: Medoid Lab colour.
        srgb: The medoid pixel's own sRGB value, in ``[0, 1]`` -- this is what
            the fit consumes, so the existing fit path needs no Lab entry point.
        spread_delta_e: Median delta-E 2000 from the medoid to all core pixels,
            a within-tile consistency measure.
        medoid_rank: Winner's rank within the candidate set.
        medoid_widened: Whether the candidate set had to be widened.
        impurity: See :func:`impurity`.
        robust_shift: See :func:`robust_shift`.
        clipped: See :func:`clipped_fraction`.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    row: int
    col: int
    roi_index: int
    n_pixels: int
    lab: tuple[float, float, float]
    srgb: tuple[float, float, float]
    spread_delta_e: float
    medoid_rank: int
    medoid_widened: bool
    impurity: float
    robust_shift: float
    clipped: float


def measure_tile(
        lab_patch: np.ndarray,
        srgb_patch: np.ndarray,
        *,
        row: int,
        col: int,
        roi_index: int,
        candidates: int = DEFAULT_MEDOID_CANDIDATES,
) -> TileMeasurement:
    """Measure one tile from its core-box Lab and sRGB pixels.

    Args:
        lab_patch: ``(h, w, 3)`` Lab pixels from ``Image.color.Lab``.
        srgb_patch: The same pixels as sRGB in ``[0, 1]``, same shape.
        row: Tile row within its ROI lattice.
        col: Tile column within its ROI lattice.
        roi_index: Index of the ROI this tile came from.
        candidates: Candidate-set size for :func:`candidate_medoid`.

    Returns:
        A :class:`TileMeasurement`.

    Raises:
        ValueError: If the two patches disagree in shape.
    """
    if lab_patch.shape != srgb_patch.shape:
        raise ValueError(
                f"lab_patch {lab_patch.shape} and srgb_patch {srgb_patch.shape} "
                "must describe the same pixels."
        )
    flat_lab = lab_patch.reshape(-1, 3)
    flat_srgb = srgb_patch.reshape(-1, 3)
    medoid = candidate_medoid(flat_lab, k=candidates)
    if medoid.index < 0:
        nan3 = (float("nan"),) * 3
        return TileMeasurement(
                row=row, col=col, roi_index=roi_index, n_pixels=0, lab=nan3,
                srgb=nan3, spread_delta_e=float("nan"), medoid_rank=-1,
                medoid_widened=False, impurity=float("nan"),
                robust_shift=float("nan"), clipped=float("nan"),
        )
    spread = _delta_e(np.broadcast_to(medoid.lab, flat_lab.shape), flat_lab)
    return TileMeasurement(
            row=row,
            col=col,
            roi_index=roi_index,
            n_pixels=int(flat_lab.shape[0]),
            lab=tuple(float(v) for v in medoid.lab),
            srgb=tuple(float(v) for v in flat_srgb[medoid.index]),
            spread_delta_e=float(np.median(spread)),
            medoid_rank=medoid.rank,
            medoid_widened=medoid.widened,
            impurity=impurity(lab_patch),
            robust_shift=robust_shift(lab_patch),
            clipped=clipped_fraction(srgb_patch),
    )
