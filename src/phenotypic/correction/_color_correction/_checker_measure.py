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

import numpy as np
from pydantic import BaseModel, ConfigDict

from ...util._robust_color_stats import (
    DEFAULT_MEDOID_CANDIDATES,
    MedoidResult as MedoidResult,
    _delta_e,
    candidate_medoid,
)

#: delta-E 2000 beyond which a pixel counts as contamination rather than noise.
IMPURITY_DELTA_E = 6.0

#: Spatial median-filter width applied before the contamination statistics, so
#: they reflect an occluder crossing the tile rather than sensor noise.
CONTAMINATION_SMOOTH = 9


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
