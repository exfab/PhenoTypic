"""Robust replicate-agreement quality check based on the relative MAD.

Flags ``(group, time)`` bins whose biological replicates disagree on a
phenotype, using a robust spread estimate that resists single
contaminated or mis-segmented colonies. For each timepoint the check
computes the median absolute deviation (MAD) of the measurement across
replicates, normalizes by the absolute median to produce a
relative-MAD metric, and broadcasts the per-bin scalars back to every
replicate row in the bin so downstream curation can pick up the flag
from any row.

This is the robust analogue of
:class:`~phenotypic.analysis.qc._replicate_agreement.ReplicateAgreement`'s
relative standard error: where the SE check is sensitive to a single
outlying replicate, the MAD's 50% breakdown point keeps the metric
stable until more than half the replicates disagree.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from phenotypic.analysis._qc_math import median_abs_deviation
from phenotypic.analysis.abc_._quality_check import QualityCheck
from phenotypic.schema import QUALITY_MAD
from phenotypic.tools_ import ColumnRef


class RelativeMAD(QualityCheck):
    """Flag ``(group, time)`` bins with poor robust agreement across replicates.

    For each combination of ``self.groupby`` columns, this check splits
    the group by ``self.time_label`` and computes the median absolute
    deviation (MAD) of the measurement across replicates at every
    timepoint. The relative MAD ``metric = MAD / |median|`` is the
    per-bin metric; bins whose metric exceeds the warn/fail thresholds
    are surfaced for curation. The per-bin scalars are broadcast back to
    every replicate row in the bin so the GUI can pick up the flag from
    any row.

    Because the MAD has a 50% breakdown point, the metric stays accurate
    even when up to half the replicates in a bin are contaminated — a
    single mis-segmented or contaminated colony will not inflate it the
    way it inflates the relative standard error. It is therefore the
    robust counterpart to
    :class:`~phenotypic.analysis.qc._replicate_agreement.ReplicateAgreement`.

    ``_HIGHER_IS_BAD`` is ``True``: a larger relative MAD means worse
    replicate agreement, so the base class flags rows whose metric meets
    or exceeds ``fail_threshold``.

    Three guard paths short-circuit to ``metric = NaN`` so under-powered
    or degenerate bins never gate curation (the base class treats
    ``NaN`` metric as ``Status="pass"``):

    1. **``n < min_replicates``** — too few replicates for a meaningful
       spread estimate. Defaults to ``min_replicates=2``; raising it lets
       callers demand more statistical power.
    2. **``|median| < eps``** — the relative-MAD ratio blows up at zero
       median, so near-zero baseline measurements (t=0 wells, blank
       wells, true-zero conditions) would otherwise flag every row. The
       default ``eps=1e-9`` catches sensor-zero readouts without losing
       genuinely-above-noise-floor measurements.
    3. **``MAD == 0`` and ``median == 0``** — degenerate bin (all
       replicates exactly zero); mathematically undefined. Treated as
       pass.

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
        min_replicates: Minimum replicate count required before the MAD
            is considered meaningful. Bins below this threshold receive
            ``metric = NaN``.
        eps: Floor on ``|median|`` below which the relative-MAD ratio is
            considered undefined. Bins below this floor receive
            ``metric = NaN``.
        warn_threshold: Relative MAD at which ``Status`` becomes
            ``"warn"``. Defaults to ``0.10``.
        fail_threshold: Relative MAD at which ``Status`` becomes
            ``"fail"`` and ``Flag=True``. Defaults to ``0.20``.

    Examples:
        Basic — three-replicate, four-timepoint synthetic frame; the
        check adds ``QC_MAD_Metric`` plus the per-bin summary columns:

        >>> import pandas as pd
        >>> from phenotypic.analysis.qc import RelativeMAD
        >>> times = [0, 1, 2, 3]
        >>> data = pd.DataFrame({
        ...     "Plate": ["P1"] * 12,
        ...     "Metadata_Time": [t for t in times for _ in range(3)],
        ...     "Replicate": [1, 2, 3] * 4,
        ...     "Size_Area": [
        ...         10.0, 10.1, 9.9,
        ...         20.0, 20.2, 19.8,
        ...         40.0, 40.4, 39.6,
        ...         80.0, 80.8, 79.2,
        ...     ],
        ... })
        >>> chk = RelativeMAD(
        ...     on="Size_Area",
        ...     groupby=["Plate"],
        ...     time_label="Metadata_Time",
        ... )
        >>> result = chk.analyze(data)  # doctest: +SKIP
        >>> "QC_MAD_Metric" in result.columns  # doctest: +SKIP
        True

        Advanced — only one replicate per ``(group, time)`` bin with
        ``min_replicates=2`` triggers the under-powered guard:

        >>> singleton = pd.DataFrame({
        ...     "Plate": ["P1", "P1"],
        ...     "Metadata_Time": [0, 1],
        ...     "Size_Area": [10.0, 20.0],
        ... })
        >>> chk = RelativeMAD(
        ...     on="Size_Area",
        ...     groupby=["Plate"],
        ...     min_replicates=2,
        ... )
        >>> result = chk.analyze(singleton)  # doctest: +SKIP
        >>> bool(result["QC_MAD_Metric"].isna().all())  # doctest: +SKIP
        True
    """

    name: ClassVar[str] = "MAD"
    _HIGHER_IS_BAD: ClassVar[bool] = True
    _exposes_agg_func: ClassVar[bool] = False
    _measurement_infoclass = QUALITY_MAD

    warn_threshold: float = 0.10
    fail_threshold: float = 0.20

    time_label: ColumnRef = "Metadata_Time"
    min_replicates: int = 2
    eps: float = 1e-9

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute per-``(group, time)`` MAD statistics and broadcast back.

        Args:
            group: One group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (a copy) with four new columns appended:
            ``QC_MAD_Median``, ``QC_MAD_MAD``, ``QC_MAD_NumMembers``,
            ``QC_MAD_Metric``. The metric column is ``NaN`` for bins
            that hit any of the three guard paths documented on the
            class.
        """
        out = group.copy()

        median_col = str(QUALITY_MAD.MEDIAN)
        mad_col = str(QUALITY_MAD.MAD)
        n_col = str(QUALITY_MAD.NUM_MEMBERS)
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

            median_val = float(np.nanmedian(values)) if n > 0 else float("nan")
            mad_val = median_abs_deviation(values) if n > 0 else float("nan")

            under_powered = n < self.min_replicates
            near_zero_median = abs(median_val) < self.eps
            degenerate = mad_val == 0 and median_val == 0

            if under_powered or near_zero_median or degenerate:
                metric = float("nan")
            else:
                metric = mad_val / abs(median_val)

            out.loc[idx, median_col] = median_val
            out.loc[idx, mad_col] = mad_val
            out.loc[idx, n_col] = n
            out.loc[idx, metric_col] = metric

        out[n_col] = out[n_col].astype(int)
        return out
