"""Tukey outlier-fraction detection quality check.

Flags ``(group, time)`` bins in which an unusually large share of members
fall outside Tukey's fences (``Q1 - k*IQR`` … ``Q3 + k*IQR``). For each
timepoint the check computes the fraction of members beyond the fences and
uses it as the per-bin metric, then broadcasts the per-bin scalars back to
every replicate row in the bin so downstream curation can pick up the flag
from any row.

Where :class:`~phenotypic.analysis.qc._max_modz.MaxModifiedZScore` targets the
single worst member, this check measures *how many* members are robust
outliers — a high fraction signals a systematically noisy group (e.g. a
contaminated or mis-imaged plate) rather than one stray colony.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from phenotypic.analysis._helper._qc_math import tukey_fences, tukey_outlier_mask
from phenotypic.analysis.abc_._quality_check import QualityCheck
from phenotypic.schema import CULTURE_METADATA, QUALITY_TUKEY
from phenotypic.sdk_ import ColumnRef

_TIME = str(CULTURE_METADATA.TIME)


class TukeyOutlierFraction(QualityCheck):
    """Flag ``(group, time)`` bins with a high fraction of Tukey outliers.

    For each combination of ``self.groupby`` columns, this check splits
    the group by ``self.time_label`` and computes Tukey's fences
    ``Q1 - k*IQR`` / ``Q3 + k*IQR`` at every timepoint. The per-bin metric
    is the fraction of members that fall strictly outside the fences; bins
    whose metric exceeds the warn/fail thresholds are surfaced for
    curation. The per-bin scalars are broadcast back to every replicate
    row in the bin so the GUI can pick up the flag from any row.

    ``_HIGHER_IS_BAD`` is ``True``: a larger outlier fraction means a
    noisier group, so the base class flags rows whose metric meets or
    exceeds ``fail_threshold``.

    One guard path short-circuits to ``metric = NaN`` so under-powered
    bins never gate curation (the base class treats ``NaN`` metric as
    ``Status="pass"``):

    1. **``n < min_replicates``** — quartiles and the IQR are not
       meaningful for tiny bins, and a single member would otherwise read
       as a 0% or 100% outlier fraction. Defaults to ``min_replicates=4``
       (the smallest bin where Tukey's quartile rule is informative);
       raising it lets callers demand more statistical power.

    When ``self.time_label`` is absent from the input data, the entire
    group is treated as a single timepoint bin so the check remains
    usable on snapshot (non-time-course) measurement frames.

    The check does **not** aggregate measurement values — it builds the
    fence/outlier summary statistics inside :meth:`_compute` — so
    :attr:`_exposes_agg_func` is ``False`` and the GUI parameter-form
    rendering driver hides the ``agg_func`` field. The base
    ``SetAnalyzer.agg_func`` is preserved on the signature for parity
    only.

    Attributes:
        time_label: Column name carrying the timepoint within each
            group. Defaults to ``"Metadata_Time"``.
        k: IQR multiplier for the fences. ``1.5`` flags standard outliers;
            ``3.0`` flags only extreme outliers. Defaults to ``1.5``.
        min_replicates: Minimum member count required before the outlier
            fraction is considered meaningful. Bins below this threshold
            receive ``metric = NaN``. Defaults to ``4``.
        warn_threshold: Outlier fraction at which ``Status`` becomes
            ``"warn"``. Defaults to ``0.10``.
        fail_threshold: Outlier fraction at which ``Status`` becomes
            ``"fail"`` and ``Flag=True``. Defaults to ``0.25``.

    Examples:
        Basic — ten members per timepoint with one extreme outlier; the
        check adds ``QC_Tukey_Metric`` plus the per-bin summary columns:

        >>> import pandas as pd
        >>> from phenotypic.analysis.qc import (
        ...     TukeyOutlierFraction,
        ... )
        >>> data = pd.DataFrame({
        ...     "Plate": ["P1"] * 10,
        ...     "Metadata_Time": [0] * 10,
        ...     "Size_Area": [
        ...         10.0, 11.0, 12.0, 13.0, 14.0,
        ...         10.5, 11.5, 12.5, 13.5, 200.0,
        ...     ],
        ... })
        >>> chk = TukeyOutlierFraction(
        ...     on="Size_Area",
        ...     groupby=["Plate"],
        ...     time_label="Metadata_Time",
        ... )
        >>> result = chk.analyze(data)
        >>> "QC_Tukey_Metric" in result.columns
        True

        Advanced — only three members per ``(group, time)`` bin with the
        default ``min_replicates=4`` triggers the under-powered guard:

        >>> sparse = pd.DataFrame({
        ...     "Plate": ["P1", "P1", "P1"],
        ...     "Metadata_Time": [0, 0, 0],
        ...     "Size_Area": [10.0, 11.0, 12.0],
        ... })
        >>> chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        >>> result = chk.analyze(sparse)
        >>> bool(result["QC_Tukey_Metric"].isna().all())
        True
    """

    name: ClassVar[str] = "Tukey"
    _HIGHER_IS_BAD: ClassVar[bool] = True
    _exposes_agg_func: ClassVar[bool] = False
    _measurement_infoclass = QUALITY_TUKEY

    warn_threshold: float = 0.10
    fail_threshold: float = 0.25

    time_label: ColumnRef = _TIME
    k: float = 1.5
    min_replicates: int = 4

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute per-``(group, time)`` Tukey statistics and broadcast back.

        Args:
            group: One group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (a copy) with five new columns appended:
            ``QC_Tukey_LowerFence``, ``QC_Tukey_UpperFence``,
            ``QC_Tukey_NumOutliers``, ``QC_Tukey_NumMembers``,
            ``QC_Tukey_Metric``. The metric column is ``NaN`` for bins
            that hit the under-powered guard documented on the class.
        """
        out = group.copy()

        lower_col = str(QUALITY_TUKEY.LOWER_FENCE)
        upper_col = str(QUALITY_TUKEY.UPPER_FENCE)
        outliers_col = str(QUALITY_TUKEY.NUM_OUTLIERS)
        n_col = str(QUALITY_TUKEY.NUM_MEMBERS)
        metric_col = self.metric_col()

        # Initialize emitted columns so partial bins still produce a
        # consistent column set on return.
        out[lower_col] = np.nan
        out[upper_col] = np.nan
        out[outliers_col] = 0
        out[n_col] = 0
        out[metric_col] = np.nan

        if len(out) == 0:
            return out

        if self.time_label in out.columns:
            time_iter = out.groupby(self.time_label, dropna=False)
        else:
            # Single-bin fallback for snapshot data.
            time_iter = [(None, out)]

        for _, bin_frame in time_iter:
            idx = bin_frame.index
            values = bin_frame[self.on].dropna().to_numpy(dtype=float)
            n = int(len(values))

            if n > 0:
                lower_fence, upper_fence = tukey_fences(values, self.k)
                outlier_mask = tukey_outlier_mask(values, self.k)
                num_outliers = int(np.count_nonzero(outlier_mask))
            else:
                lower_fence = float("nan")
                upper_fence = float("nan")
                num_outliers = 0

            under_powered = n < self.min_replicates
            if under_powered:
                metric = float("nan")
            else:
                metric = num_outliers / n

            out.loc[idx, lower_col] = lower_fence
            out.loc[idx, upper_col] = upper_fence
            out.loc[idx, outliers_col] = num_outliers
            out.loc[idx, n_col] = n
            out.loc[idx, metric_col] = metric

        out[outliers_col] = out[outliers_col].astype(int)
        out[n_col] = out[n_col].astype(int)
        return out
