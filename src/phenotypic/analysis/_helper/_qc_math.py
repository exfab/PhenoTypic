"""Shared robust-statistics primitives for QC checks and outlier removers.

This module isolates the *numerics* used by the analysis-layer outlier
removers (:class:`~phenotypic.analysis.filter._mad_outlier.MADOutlierRemover` and
:class:`~phenotypic.analysis.filter._tukey_outlier.TukeyOutlierRemover`) so that
smart-QC checks can reuse the exact same formulas. Every function is a pure,
NumPy-based helper with no pydantic, no pandas, and no I/O.

The two families of statistics provided are:

- **Modified Z-score (MAD).** The Iglewicz-Hoaglin robust analogue of the
  standard Z-score, built on the median and median absolute deviation. Has a
  50% breakdown point, so it stays accurate even when up to half the values in
  a group are contaminated.
- **Tukey's fences.** The classic interquartile-range outlier rule,
  ``Q1 - k*IQR`` / ``Q3 + k*IQR``.

NaN handling matches the removers: MAD helpers use nan-aware reductions
(``np.nanmedian`` / ``np.nanmean``) and propagate NaN scores for NaN inputs,
while Tukey helpers use ``np.nanpercentile`` (linear interpolation) so that a
NaN measurement neither shifts the quartiles nor destroys the fences, exactly
as :class:`TukeyOutlierRemover` does.
"""

from __future__ import annotations

import numpy as np

# Iglewicz & Hoaglin (1993) consistency constant: for a normal distribution
# sigma ~= 1.4826 * MAD, and 0.6745 ~= 1 / 1.4826. Multiplying the absolute
# deviation by 0.6745 / MAD therefore estimates the deviation in units of
# sigma, putting the modified Z-score on the same scale as a standard Z-score.
# Reference: Iglewicz, B., & Hoaglin, D. C. (1993). How to Detect and Handle
# Outliers. ASQC Quality Press.
MAD_CONSISTENCY: float = 0.6745


def median_abs_deviation(values: np.ndarray) -> float:
    """Compute the median absolute deviation (MAD) about the median.

    The MAD is ``median(|value - median(values)|)``, a robust estimate of
    spread with a 50% breakdown point. NaN entries are ignored via nan-aware
    reductions (``np.nanmedian``), matching the outlier removers.

    Args:
        values: Array of numeric values. NaN entries are ignored.

    Returns:
        The median absolute deviation as a ``float``. Returns ``0.0`` when all
        (non-NaN) values are identical, and ``nan`` when the input is empty or
        entirely NaN.

    Examples:
        >>> import numpy as np
        >>> from phenotypic.analysis._helper._qc_math import median_abs_deviation
        >>> float(median_abs_deviation(np.array([1.0, 2.0, 3.0, 4.0, 5.0])))
        1.0
        >>> float(median_abs_deviation(np.array([7.0, 7.0, 7.0])))
        0.0
    """
    values = np.asarray(values, dtype=float)
    median = float(np.nanmedian(values))
    return float(np.nanmedian(np.abs(values - median)))


def modified_z_scores(values: np.ndarray) -> np.ndarray:
    """Compute Iglewicz-Hoaglin modified Z-scores for an array of values.

    Implements ``MAD_CONSISTENCY * |value - median| / MAD``. When the MAD is
    zero (all values identical, or a tie spanning more than half the values),
    the test falls back to the raw absolute deviation scaled by the *mean*
    absolute deviation, preserving the breakdown point while avoiding division
    by zero. If every (non-NaN) value is identical the mean absolute deviation
    is also zero and an all-zeros score array is returned (no outliers).

    This replicates :meth:`MADOutlierRemover._modified_z_scores` byte-for-byte;
    NaN inputs propagate to NaN scores (a NaN score never exceeds a positive
    threshold, so NaN rows are never flagged as outliers).

    Args:
        values: Array of numeric values. NaN entries are ignored when computing
            the median and deviations, and produce NaN scores.

    Returns:
        A float array of per-value modified Z-scores aligned with ``values``,
        with NaN where the input was NaN. An empty input yields an empty array.

    Examples:
        >>> import numpy as np
        >>> from phenotypic.analysis._helper._qc_math import modified_z_scores
        >>> scores = modified_z_scores(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
        >>> bool(scores[-1] > scores[0])
        True
        >>> # All-identical values produce zero scores (no outliers).
        >>> modified_z_scores(np.array([5.0, 5.0, 5.0])).tolist()
        [0.0, 0.0, 0.0]
        >>> # NaN inputs propagate to NaN scores.
        >>> scores = modified_z_scores(np.array([1.0, 2.0, np.nan, 3.0]))
        >>> bool(np.isnan(scores[2]))
        True
    """
    values = np.asarray(values, dtype=float)
    median = float(np.nanmedian(values))
    abs_dev = np.abs(values - median)
    mad = float(np.nanmedian(abs_dev))

    if mad == 0.0:
        mean_ad = float(np.nanmean(abs_dev))
        if mean_ad == 0.0:
            # All values identical -- no outliers possible.
            return np.zeros_like(values, dtype=float)
        return abs_dev / mean_ad

    return MAD_CONSISTENCY * abs_dev / mad


def tukey_fences(values: np.ndarray, k: float = 1.5) -> tuple[float, float]:
    """Compute Tukey's lower and upper outlier fences for an array of values.

    The fences are ``Q1 - k*IQR`` and ``Q3 + k*IQR`` where ``IQR = Q3 - Q1``.
    Quartiles use ``np.nanpercentile`` with its default linear interpolation
    (matching :class:`TukeyOutlierRemover`): NaN entries are ignored, so the
    fences of a group are unaffected by a missing measurement. A group whose
    values are *all* NaN (or empty) has no quartiles to compute and yields
    ``(nan, nan)``.

    Args:
        values: Array of numeric values for one group. NaN entries are ignored.
        k: IQR multiplier for the fences. ``1.5`` flags standard outliers;
            ``3.0`` flags only extreme outliers. Default is ``1.5``.

    Returns:
        A ``(lower_fence, upper_fence)`` tuple of floats, computed from the
        non-NaN values. Returns ``(nan, nan)`` when ``values`` is empty or
        entirely NaN.

    Examples:
        >>> import numpy as np
        >>> from phenotypic.analysis._helper._qc_math import tukey_fences
        >>> lower, upper = tukey_fences(np.arange(1.0, 11.0))
        >>> round(lower, 4), round(upper, 4)
        (-3.5, 14.5)
        >>> # A NaN colony measurement does not disturb the fences.
        >>> with_nan = np.append(np.arange(1.0, 11.0), np.nan)
        >>> lower, upper = tukey_fences(with_nan)
        >>> round(lower, 4), round(upper, 4)
        (-3.5, 14.5)
        >>> # An all-NaN group has no quartiles.
        >>> tukey_fences(np.array([np.nan, np.nan]))
        (nan, nan)
    """
    values = np.asarray(values, dtype=float)
    # np.nanpercentile emits "RuntimeWarning: All-NaN slice encountered" and
    # returns nan for a fully-NaN (or empty) input; return the nan fences
    # directly instead of warning.
    if values.size == 0 or bool(np.all(np.isnan(values))):
        return float("nan"), float("nan")
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    iqr = q3 - q1
    lower_fence = q1 - (iqr * k)
    upper_fence = q3 + (iqr * k)
    return float(lower_fence), float(upper_fence)


def tukey_outlier_mask(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Flag values that fall outside Tukey's fences.

    A value is an outlier when it is strictly below the lower fence or strictly
    above the upper fence (``value < lower or value > upper``), matching the
    visualization path of :class:`TukeyOutlierRemover`. Equivalently, inliers
    satisfy ``lower <= value <= upper`` -- the same boundary the remover uses
    when filtering rows.

    Args:
        values: Array of numeric values for one group.
        k: IQR multiplier for the fences. Default is ``1.5``.

    Returns:
        A boolean array aligned with ``values``, ``True`` where the value is an
        outlier. NaN entries are never flagged: they are ignored when computing
        the fences, and every comparison against a NaN value is ``False``.

    Examples:
        >>> import numpy as np
        >>> from phenotypic.analysis._helper._qc_math import tukey_outlier_mask
        >>> data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 100.0])
        >>> tukey_outlier_mask(data).tolist()
        [False, False, False, False, False, True]
    """
    values = np.asarray(values, dtype=float)
    lower_fence, upper_fence = tukey_fences(values, k)
    return (values < lower_fence) | (values > upper_fence)


def tukey_outlier_fraction(values: np.ndarray, k: float = 1.5) -> float:
    """Compute the fraction of values that fall outside Tukey's fences.

    Args:
        values: Array of numeric values for one group.
        k: IQR multiplier for the fences. Default is ``1.5``.

    Returns:
        The fraction of values flagged as outliers, in ``[0.0, 1.0]``. Returns
        ``0.0`` for an empty input.

    Examples:
        >>> import numpy as np
        >>> from phenotypic.analysis._helper._qc_math import tukey_outlier_fraction
        >>> data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 100.0])
        >>> round(tukey_outlier_fraction(data), 4)
        0.1667
        >>> tukey_outlier_fraction(np.array([]))
        0.0
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    mask = tukey_outlier_mask(values, k)
    return float(np.count_nonzero(mask) / values.size)
