"""Maximum modified Z-score replicate-agreement quality check.

Flags ``(group, time)`` bins that contain a single member disagreeing
sharply with the rest of its replicates or detection runs. For each
timepoint the check computes the Iglewicz-Hoaglin modified Z-score of
every member relative to the bin median (scaled by the MAD) and takes the
maximum as the per-bin metric, then broadcasts the per-bin scalars back to
every replicate row in the bin so downstream curation can pick up the flag
from any row.

Where :class:`~phenotypic.analysis._relative_mad.RelativeMAD` measures the
overall spread of a bin, this check targets the *worst single member*: a
large maximum modified Z-score marks the most-deviating colony, which often
reflects contamination, an edge artifact, or a segmentation error rather
than real biology.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from phenotypic.analysis._qc_math import median_abs_deviation, modified_z_scores
from phenotypic.analysis.abc_._quality_check import QualityCheck
from phenotypic.schema import QUALITY_ZMAX
from phenotypic.tools_ import ColumnRef


class MaxModifiedZScore(QualityCheck):
    """Flag ``(group, time)`` bins whose worst member is a robust outlier.

    For each combination of ``self.groupby`` columns, this check splits
    the group by ``self.time_label`` and computes the Iglewicz-Hoaglin
    modified Z-score ``0.6745 * |x - median| / MAD`` of every member at
    each timepoint. The per-bin metric is the maximum of those scores —
    the deviation of the single most-disagreeing member — so a bin fails
    as soon as one colony is far enough from the others. The per-bin
    scalars are broadcast back to every replicate row in the bin so the
    GUI can pick up the flag from any row.

    ``_HIGHER_IS_BAD`` is ``True``: a larger maximum modified Z-score
    means a worse outlier, so the base class flags rows whose metric meets
    or exceeds ``fail_threshold``.

    Two guard paths short-circuit to ``metric = NaN`` so under-powered or
    degenerate bins never gate curation (the base class treats ``NaN``
    metric as ``Status="pass"``):

    1. **``n < min_replicates``** — too few members for a meaningful
       robust Z-score. Defaults to ``min_replicates=2``; raising it lets
       callers demand more statistical power.
    2. **All members identical** — the bin median equals every value, so
       the MAD is zero and every modified Z-score is zero. A maximum of
       zero is reported as ``metric = NaN`` (perfect agreement is not an
       outlier and should never gate curation), matching the
       "no outliers" semantics of
       :func:`~phenotypic.analysis._qc_math.modified_z_scores`.

    When ``self.time_label`` is absent from the input data, the entire
    group is treated as a single timepoint bin so the check remains
    usable on snapshot (non-time-course) measurement frames.

    The check does **not** aggregate measurement values — it builds the
    median/MAD summary statistics inside :meth:`_compute` — so
    :attr:`_exposes_agg_func` is ``False`` and the GUI parameter-form
    rendering driver hides the ``agg_func`` field. The base
    ``SetAnalyzer.agg_func`` is preserved on the signature for parity
    only.

    Attributes:
        time_label: Column name carrying the timepoint within each
            group. Defaults to ``"Metadata_Time"``.
        min_replicates: Minimum member count required before the modified
            Z-score is considered meaningful. Bins below this threshold
            receive ``metric = NaN``.
        warn_threshold: Maximum modified Z-score at which ``Status``
            becomes ``"warn"``. Defaults to ``3.5``.
        fail_threshold: Maximum modified Z-score at which ``Status``
            becomes ``"fail"`` and ``Flag=True``. Defaults to ``5.0``.

    Examples:
        Basic — four members per timepoint, the check adds
        ``QC_ZMax_Metric`` plus the per-bin summary columns:

        >>> import pandas as pd
        >>> from phenotypic.analysis._max_modz import MaxModifiedZScore
        >>> data = pd.DataFrame({
        ...     "Plate": ["P1"] * 8,
        ...     "Metadata_Time": [0, 0, 0, 0, 1, 1, 1, 1],
        ...     "Size_Area": [
        ...         10.0, 10.1, 9.9, 10.2,
        ...         20.0, 20.1, 19.9, 60.0,
        ...     ],
        ... })
        >>> chk = MaxModifiedZScore(
        ...     on="Size_Area",
        ...     groupby=["Plate"],
        ...     time_label="Metadata_Time",
        ... )
        >>> result = chk.analyze(data)
        >>> "QC_ZMax_Metric" in result.columns
        True

        Advanced — only one member per ``(group, time)`` bin with
        ``min_replicates=2`` triggers the under-powered guard:

        >>> singleton = pd.DataFrame({
        ...     "Plate": ["P1", "P1"],
        ...     "Metadata_Time": [0, 1],
        ...     "Size_Area": [10.0, 20.0],
        ... })
        >>> chk = MaxModifiedZScore(
        ...     on="Size_Area",
        ...     groupby=["Plate"],
        ...     min_replicates=2,
        ... )
        >>> result = chk.analyze(singleton)
        >>> bool(result["QC_ZMax_Metric"].isna().all())
        True
    """

    name: ClassVar[str] = "ZMax"
    _HIGHER_IS_BAD: ClassVar[bool] = True
    _exposes_agg_func: ClassVar[bool] = False
    _measurement_infoclass = QUALITY_ZMAX

    warn_threshold: float = 3.5
    fail_threshold: float = 5.0

    time_label: ColumnRef = "Metadata_Time"
    min_replicates: int = 2

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute per-``(group, time)`` modified-Z statistics and broadcast.

        Args:
            group: One group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (a copy) with four new columns appended:
            ``QC_ZMax_Median``, ``QC_ZMax_MAD``, ``QC_ZMax_NumMembers``,
            ``QC_ZMax_Metric``. The metric column is ``NaN`` for bins that
            hit either guard path documented on the class.
        """
        out = group.copy()

        median_col = str(QUALITY_ZMAX.MEDIAN)
        mad_col = str(QUALITY_ZMAX.MAD)
        n_col = str(QUALITY_ZMAX.NUM_MEMBERS)
        metric_col = self.metric_col()

        # Initialize emitted columns so partial bins still produce a
        # consistent column set on return.
        out[median_col] = np.nan
        out[mad_col] = np.nan
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

            median_val = float(np.median(values)) if n > 0 else float("nan")
            mad_val = median_abs_deviation(values) if n > 0 else float("nan")

            under_powered = n < self.min_replicates
            if under_powered:
                metric = float("nan")
            else:
                scores = modified_z_scores(values)
                max_score = float(np.nanmax(scores)) if scores.size else 0.0
                # A maximum score of zero means perfect agreement (all
                # members identical); never treat that as an outlier.
                metric = float("nan") if max_score == 0.0 else max_score

            out.loc[idx, median_col] = median_val
            out.loc[idx, mad_col] = mad_val
            out.loc[idx, n_col] = n
            out.loc[idx, metric_col] = metric

        out[n_col] = out[n_col].astype(int)
        return out
