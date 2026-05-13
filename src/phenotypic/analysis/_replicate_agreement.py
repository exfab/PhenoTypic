"""Replicate-agreement standard-error quality check.

Flags ``(group, time)`` bins whose biological replicates disagree on a
phenotype. Computes the standard error of the mean (``SE = stddev /
sqrt(n)``) across replicates per timepoint, normalizes by the mean to
produce a relative-SE severity, and broadcasts the per-bin scalars back
to every replicate row in the bin so downstream curation can pick up the
flag from any row.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from phenotypic.analysis.abc_._quality_check import QualityCheck
from phenotypic.tools_ import ColumnRef, ColumnRefList
from phenotypic.tools_.measurement_info import QUALITY_SE


class ReplicateAgreement(QualityCheck):
    """Flag ``(group, time)`` bins with poor agreement across replicates.

    For each combination of ``self.groupby`` columns, this check splits
    the group by ``self.time_label`` and computes the standard error of
    the measurement across replicates at every timepoint. The relative
    standard error ``severity = |SE| / |mean|`` is the per-bin severity;
    bins whose severity exceeds the warn/fail thresholds are surfaced
    for curation. The per-bin scalars are broadcast back to every
    replicate row in the bin so the GUI can pick up the flag from any
    row.

    Three guard paths short-circuit to ``severity = NaN`` so
    under-powered or degenerate bins never gate curation (the base class
    treats ``NaN`` severity as ``Status="pass"``):

    1. **``n < min_replicates``** — too few replicates for a meaningful
       standard error. Defaults to ``min_replicates=2``; raising it lets
       callers demand more statistical power.
    2. **``|mean| < eps``** — the relative-SE ratio blows up at zero
       mean, so near-zero baseline measurements (t=0 wells, blank wells,
       true-zero conditions) would otherwise flag every row. The default
       ``eps=1e-9`` catches sensor-zero readouts without losing
       genuinely-above-noise-floor measurements.
    3. **``stddev == 0`` and ``mean == 0``** — degenerate bin (all
       replicates exactly zero); mathematically undefined. Treated as
       pass.

    When ``self.time_label`` is absent from the input data, the entire
    group is treated as a single timepoint bin so the check remains
    usable on snapshot (non-time-course) measurement frames.

    The check does **not** aggregate measurement values — it builds the
    SE/Mean/CV summary statistics inside :meth:`_compute` — so
    :attr:`_exposes_agg_func` is ``False`` and the GUI parameter-form
    rendering driver hides the ``agg_func`` field. The base
    ``SetAnalyzer.agg_func`` is preserved on the signature for parity
    only.

    Attributes:
        time_label: Column name carrying the timepoint within each
            group. Defaults to ``"Metadata_Time"``.
        min_replicates: Minimum replicate count required before SE is
            considered meaningful. Bins below this threshold receive
            ``severity = NaN``.
        eps: Floor on ``|mean|`` below which the relative-SE ratio is
            considered undefined. Bins below this floor receive
            ``severity = NaN``.

    Examples:
        Basic — three-replicate, four-timepoint synthetic frame; the
        check adds ``QC_SE_Severity`` plus the per-bin summary columns:

        >>> import pandas as pd
        >>> from phenotypic.analysis._replicate_agreement import (
        ...     ReplicateAgreement,
        ... )
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
        >>> chk = ReplicateAgreement(
        ...     on="Size_Area",
        ...     groupby=["Plate"],
        ...     time_label="Metadata_Time",
        ... )
        >>> result = chk.analyze(data)  # doctest: +SKIP
        >>> "QC_SE_Severity" in result.columns  # doctest: +SKIP
        True

        Advanced — only one replicate per ``(group, time)`` bin with
        ``min_replicates=2`` triggers the under-powered guard:

        >>> singleton = pd.DataFrame({
        ...     "Plate": ["P1", "P1"],
        ...     "Metadata_Time": [0, 1],
        ...     "Size_Area": [10.0, 20.0],
        ... })
        >>> chk = ReplicateAgreement(
        ...     on="Size_Area",
        ...     groupby=["Plate"],
        ...     min_replicates=2,
        ... )
        >>> result = chk.analyze(singleton)  # doctest: +SKIP
        >>> bool(result["QC_SE_Severity"].isna().all())  # doctest: +SKIP
        True
    """

    name: ClassVar[str] = "SE"
    severity_warn: ClassVar[float] = 0.10
    severity_fail: ClassVar[float] = 0.20
    _exposes_agg_func: ClassVar[bool] = False
    _measurement_infoclass = QUALITY_SE

    def __init__(
        self,
        on: ColumnRef,
        groupby: ColumnRefList,
        time_label: ColumnRef = "Metadata_Time",
        *,
        severity_warn: float | None = None,
        severity_fail: float | None = None,
        min_replicates: int = 2,
        eps: float = 1e-9,
        agg_func: str = "mean",
        n_jobs: int = 1,
    ) -> None:
        """Construct a replicate-agreement check.

        Args:
            on: Measurement column whose replicate dispersion is
                evaluated.
            groupby: Columns that define a comparison unit. Replicates
                are pooled within each ``(group, time)`` bin.
            time_label: Column carrying the timepoint within each
                group. Defaults to ``"Metadata_Time"``. When absent
                from the data, the whole group is treated as one bin.
            severity_warn: Per-instance override for ``severity_warn``.
                ``None`` falls back to the class default (``0.10``).
            severity_fail: Per-instance override for ``severity_fail``.
                ``None`` falls back to the class default (``0.20``).
            min_replicates: Minimum replicate count per bin before SE
                is reported as a number. Bins with fewer replicates
                receive ``severity = NaN``. Defaults to ``2``.
            eps: Floor on ``|mean|`` below which the relative-SE ratio
                is considered undefined. Bins whose mean falls below
                this floor receive ``severity = NaN``. Defaults to
                ``1e-9``.
            agg_func: Forwarded to :class:`SetAnalyzer`. The check does
                not aggregate measurement values itself; the parameter
                is preserved for signature parity.
            n_jobs: Worker count. Currently unused by the base
                ``analyze`` loop.
        """
        super().__init__(
            on=on,
            groupby=groupby,
            severity_warn=severity_warn,
            severity_fail=severity_fail,
            agg_func=agg_func,
            n_jobs=n_jobs,
        )
        self.time_label = time_label
        self.min_replicates = min_replicates
        self.eps = eps

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute per-``(group, time)`` SE statistics and broadcast back.

        Args:
            group: One group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (a copy) with five new columns appended:
            ``QC_SE_Value``, ``QC_SE_Mean``, ``QC_SE_CV``,
            ``QC_SE_NumReplicates``, ``QC_SE_Severity``. The severity
            column is ``NaN`` for bins that hit any of the three guard
            paths documented on the class.
        """
        out = group.copy()

        value_col = f"{QUALITY_SE.VALUE}"
        mean_col = f"{QUALITY_SE.MEAN}"
        cv_col = f"{QUALITY_SE.CV}"
        n_col = f"{QUALITY_SE.NUM_REPLICATES}"
        severity_col = self.severity_col()

        # Initialize emitted columns so partial bins still produce a
        # consistent column set on return.
        out[value_col] = np.nan
        out[mean_col] = np.nan
        out[cv_col] = np.nan
        out[n_col] = 0
        out[severity_col] = np.nan

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

            mean_val = float(values.mean()) if n > 0 else float("nan")
            std_val = float(values.std(ddof=1)) if n > 1 else 0.0
            se_val = std_val / sqrt(n) if n > 0 else float("nan")
            cv_val = (
                std_val / abs(mean_val)
                if abs(mean_val) > self.eps
                else float("nan")
            )

            under_powered = n < self.min_replicates
            near_zero_mean = abs(mean_val) < self.eps
            degenerate = std_val == 0 and mean_val == 0

            if under_powered or near_zero_mean or degenerate:
                severity = float("nan")
            else:
                severity = abs(se_val) / abs(mean_val)

            out.loc[idx, value_col] = se_val
            out.loc[idx, mean_col] = mean_val
            out.loc[idx, cv_col] = cv_val
            out.loc[idx, n_col] = n
            out.loc[idx, severity_col] = severity

        out[n_col] = out[n_col].astype(int)
        return out

    def dash(self, **kwargs: Any) -> go.Figure:
        """Render mean ± SE bands per group across time.

        For each ``self.groupby`` combination the plot draws the
        per-timepoint mean as a connected line with vertical
        error-bars sized to the per-bin SE. The line's color is the
        worst status observed across that group's timepoints:
        ``"pass"`` is green, ``"warn"`` is gold, ``"fail"`` is red.

        Args:
            **kwargs: Passed through to :class:`plotly.graph_objects.Figure`
                / ``Figure.update_layout``. Accepted keys are ``title``
                and ``height``.

        Returns:
            A :class:`plotly.graph_objects.Figure` with one line + error
            bar trace per group.

        Raises:
            RuntimeError: If :meth:`analyze` has not been called yet.
        """
        df = self._latest_measurements
        if df.empty:
            raise RuntimeError("call analyze() first")

        value_col = f"{QUALITY_SE.VALUE}"
        mean_col = f"{QUALITY_SE.MEAN}"
        status_col = self.status_col()

        status_colors = {
            "pass": "#2E86AB",
            "warn": "#F4A261",
            "fail": "#E63946",
        }
        status_rank = {"pass": 0, "warn": 1, "fail": 2}
        inv_rank = {v: k for k, v in status_rank.items()}

        fig = go.Figure()

        has_time = self.time_label in df.columns
        groupby_cols = list(self.groupby)

        for key, group_frame in df.groupby(groupby_cols, dropna=False):
            if has_time:
                per_bin = (
                    group_frame.groupby(self.time_label, dropna=False)
                    .agg({
                        mean_col: "first",
                        value_col: "first",
                        status_col: "first",
                    })
                    .reset_index()
                    .sort_values(self.time_label)
                )
                t_vals = per_bin[self.time_label].to_numpy()
            else:
                row = group_frame.iloc[0]
                per_bin = pd.DataFrame({
                    mean_col: [row[mean_col]],
                    value_col: [row[value_col]],
                    status_col: [row[status_col]],
                })
                t_vals = np.array([0])

            mean_vals = per_bin[mean_col].astype(float).to_numpy()
            se_vals = np.nan_to_num(
                per_bin[value_col].astype(float).to_numpy(), nan=0.0
            )
            statuses = per_bin[status_col].astype(str).tolist()
            worst_rank = max(
                (status_rank.get(s, 0) for s in statuses), default=0
            )
            worst_status = inv_rank[worst_rank]
            color = status_colors.get(worst_status, "#888888")

            if isinstance(key, tuple):
                label = " | ".join(str(k) for k in key)
            else:
                label = str(key)

            fig.add_trace(
                go.Scatter(
                    x=t_vals,
                    y=mean_vals,
                    mode="lines+markers",
                    name=label,
                    line={"color": color, "width": 2},
                    marker={"color": color, "size": 7},
                    error_y={
                        "type": "data",
                        "array": se_vals,
                        "visible": True,
                        "color": color,
                        "thickness": 1,
                    },
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        f"{self.time_label}: %{{x}}<br>"
                        "Mean: %{y:.4f}<br>"
                        f"Status: {worst_status}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=kwargs.get(
                "title", "Replicate Agreement (mean ± SE)"
            ),
            xaxis_title=self.time_label if has_time else "",
            yaxis_title=self.on,
            height=kwargs.get("height", 360),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#f5f7fa",
            hovermode="closest",
        )
        return fig
