"""Region-overlap metrics — the minimal Dice/IoU pair (supervised-scorers §A).

The two region-overlap statistics the supervised scorer needs, and **only** these
two — one metric *pair*, never a redundant panel (metrics v1 is minimal, plan §0b).
Both operate on boolean masks (any array is coerced to truthy), are pure NumPy, and
carry no ground-truth-loader coupling — they are the bare overlap primitives the
matcher and scorer build on.

The two are monotonically related (``Dice = 2·IoU / (1 + IoU)``), so they rank
candidates identically; both ship because Dice is the conventional segmentation report
(F1 over pixels) while IoU (Jaccard) is the matcher's per-pair acceptance statistic
(``τ = 0.5``, see ``_matching``).

The **empty-mask convention** (supervised-scorers §A.5): two empty masks *perfectly
agree* — there is nothing to over- or under-segment — so both metrics return ``1.0``.
A non-empty mask against an empty one has no overlap and scores ``0.0``.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _as_bool(mask: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Coerce any array-like to a boolean mask (truthy → ``True``).

    Args:
        mask: A 2-D array-like — boolean, integer label/count, or float — whose
            non-zero entries are the region's foreground.

    Returns:
        The mask as a contiguous boolean array.
    """
    return np.asarray(mask).astype(bool)


def dice(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """The Dice–Sørensen overlap coefficient of two boolean masks.

    ``Dice = 2·|A ∩ B| / (|A| + |B|)`` — the pixel-wise F1 score. ``1.0`` for an
    exact match, ``0.0`` for disjoint masks. Per the §A.5 empty-mask convention,
    **two empty masks score ``1.0``** (perfect agreement: nothing to segment); a
    non-empty mask against an empty one scores ``0.0``.

    Args:
        a: The first mask (any array-like; non-zero entries are foreground).
        b: The second mask, broadcastable to ``a``'s shape.

    Returns:
        The Dice coefficient in ``[0, 1]`` (higher = better).

    Examples:
        A real plate's foreground compared to itself is a perfect match:

        >>> import numpy as np
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> mask = np.asarray(load_synth_yeast_plate().objmap[:]) > 0
        >>> float(dice(mask, mask))
        1.0

        Eroding the mask by one column lowers the overlap below ``1``:

        >>> shifted = np.zeros_like(mask)
        >>> shifted[:, 1:] = mask[:, :-1]
        >>> 0.0 < dice(mask, shifted) < 1.0
        True
    """
    am = _as_bool(a)
    bm = _as_bool(b)
    size_a = int(am.sum())
    size_b = int(bm.sum())
    denom = size_a + size_b
    if denom == 0:
        return 1.0  # §A.5: both empty → perfect agreement.
    intersection = int(np.logical_and(am, bm).sum())
    return 2.0 * intersection / denom


def iou(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """The intersection-over-union (Jaccard) overlap of two boolean masks.

    ``IoU = |A ∩ B| / |A ∪ B|``. ``1.0`` for an exact match, ``0.0`` for disjoint
    masks. Per the §A.5 empty-mask convention, **two empty masks score ``1.0``**;
    a non-empty mask against an empty one scores ``0.0``. IoU is the matcher's
    per-pair acceptance statistic (``τ = 0.5``, see :mod:`._matching`).

    Args:
        a: The first mask (any array-like; non-zero entries are foreground).
        b: The second mask, broadcastable to ``a``'s shape.

    Returns:
        The IoU in ``[0, 1]`` (higher = better).

    Examples:
        Identical foregrounds give an IoU of ``1``; the metric relates to Dice by
        ``Dice = 2·IoU / (1 + IoU)``:

        >>> import numpy as np
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> mask = np.asarray(load_synth_yeast_plate().objmap[:]) > 0
        >>> float(iou(mask, mask))
        1.0
        >>> shifted = np.zeros_like(mask)
        >>> shifted[:, 1:] = mask[:, :-1]
        >>> j = iou(mask, shifted)
        >>> bool(np.isclose(dice(mask, shifted), 2 * j / (1 + j)))
        True
    """
    am = _as_bool(a)
    bm = _as_bool(b)
    union = int(np.logical_or(am, bm).sum())
    if union == 0:
        return 1.0  # §A.5: both empty → perfect agreement.
    intersection = int(np.logical_and(am, bm).sum())
    return intersection / union
