"""Intraclass-correlation-coefficient replicate-reliability quality check.

Flags ``groupby`` groups whose replicates fail to reproduce one another's
measurements. Each group is modeled as a two-way random-effects design —
**subjects are the repeated-measure axis** (``Metadata_Time`` by default) and
**raters are the replicates that should agree** (``Metadata_Replicate`` by
default) — and the ICC(2,1) (single-measurement, absolute-agreement)
coefficient quantifies how consistently the replicates reproduce each subject.
A single scalar is computed per group and broadcast back to every member row
so downstream curation can pick up the flag from any row.

It treats ``Metadata_Time`` as the repeated-measure subject axis and replicates
as raters; with growth over time, between-timepoint variance dominates, so the
ICC primarily flags replicate disagreement that is large relative to the growth
signal. The ICC is computed with NumPy only (no ``pingouin`` dependency) from
the classic two-way mean-square decomposition. The subject/rater axes are
configurable, so the same check serves snapshot designs (e.g.
subject=``Metadata_StrainID``) by overriding ``subject_label``.

``_HIGHER_IS_BAD`` is ``False``: the ICC is an agreement *score* in roughly
``[0, 1]`` where higher is better, so a group fails when its ICC falls to or
below ``fail_threshold``. This is the project's reference implementation of
the lower-is-bad direction.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from phenotypic.analysis.abc_._quality_check import QualityCheck
from phenotypic.schema import QUALITY_ICC
from phenotypic.tools_ import ColumnRef


class ICC(QualityCheck):
    """Flag ``groupby`` groups whose replicates have low ICC(2,1) agreement.

    For each combination of ``self.groupby`` columns, this check builds a
    complete ``subjects × raters`` matrix — one row per ``subject_label``
    value (the repeated-measure axis, by default ``"Metadata_Time"``) and
    one column per ``rater_label`` value (the replicates that *should*
    agree, by default ``"Metadata_Replicate"``) — and computes the
    ICC(2,1) two-way random, absolute-agreement coefficient over it. With
    ``Metadata_Time`` as the subject axis the between-timepoint (growth)
    variance dominates, so the ICC primarily flags replicate disagreement
    that is large relative to the growth signal. The single per-group ICC
    is the metric, broadcast to every member row so the GUI can pick up the
    flag from any row.

    The estimator is the classic two-way mean-square decomposition::

        ICC = (MSR - MSE)
              / (MSR + (k - 1) * MSE + (k / n) * (MSC - MSE))

    where ``n`` is the subject count, ``k`` the rater count, ``MSR`` the
    between-subjects mean square, ``MSC`` the between-raters mean square,
    and ``MSE`` the residual mean square. Computed with NumPy only — no
    ``pingouin`` dependency.

    ``_HIGHER_IS_BAD`` is ``False``: the ICC is an agreement score where a
    *smaller* value is worse, so the base class flags rows whose metric is
    less than or equal to ``fail_threshold`` and warns at or below
    ``warn_threshold`` (with ``fail_threshold <= warn_threshold``). This
    check is the reference implementation of the lower-is-bad direction.
    A **negative ICC is a valid result**, not an error: it signals
    agreement worse than chance (e.g. raters that anti-correlate across
    subjects) and correctly lands well inside the ``"fail"`` band.

    Several guard paths short-circuit to ``metric = NaN`` so under-powered
    or degenerate groups never gate curation. ``NaN`` here means
    **"insufficient data to estimate agreement"** — it is *not* a passing
    grade of good agreement. The base class maps ``NaN`` to
    ``Status="pass"`` only so degenerate groups never *gate* curation; a
    reviewer reading the metric should treat ``NaN`` as "could not be
    computed", never as "agreement is fine". The guards are:

    1. **Missing axis column (LOUD)** — ``subject_label`` or
       ``rater_label`` is absent from the input frame, so the two-way
       model cannot be built. The metric is ``NaN``, but the group key is
       **also recorded in** :attr:`unmatched_groups` (mirroring
       :class:`ExpectedVsDetectedCount`) so a not-evaluated check is
       visibly "could not run", never a silent green pass. The other
       guards below are genuine "insufficient data" cases and do *not*
       populate :attr:`unmatched_groups`.
    2. **Incomplete matrix** — at least one ``(subject, rater)`` cell is
       missing or duplicated after pivoting; a balanced two-way ANOVA
       requires exactly one observation per cell. A single missing cell
       NaNs the **whole group** — subjects and raters are never silently
       dropped to complete the design, and no rows are removed.
    3. **``n < 2`` subjects or ``k < 2`` raters** — at least two of each
       are required for the between-source mean squares.
    4. **Zero variance** — the total mean square is zero (all values
       identical), so the ICC is mathematically undefined. This is
       insufficient signal, explicitly **not** a perfect ``1.0``.

    The check does **not** aggregate measurement values — it builds the
    two-way matrix inside :meth:`_compute` — so :attr:`_exposes_agg_func`
    is ``False`` and the GUI parameter-form rendering driver hides the
    ``agg_func`` field. The base ``SetAnalyzer.agg_func`` is preserved on
    the signature for parity only.

    Attributes:
        subject_label: Column whose distinct values index the *subject*
            (row) axis of the two-way model — the repeated-measure axis.
            Defaults to ``"Metadata_Time"`` so each timepoint is a subject
            and the ICC flags replicates that disagree relative to the
            growth trend. Override (e.g. ``"Metadata_StrainID"``) for a
            snapshot reliability design.
        rater_label: Column whose distinct values index the *rater*
            (column) axis of the two-way model — the replicates that
            should agree. Defaults to ``"Metadata_Replicate"``.
        warn_threshold: ICC at or below which ``Status`` becomes
            ``"warn"``. Defaults to ``0.75``.
        fail_threshold: ICC at or below which ``Status`` becomes
            ``"fail"`` and ``Flag=True``. Defaults to ``0.50``.
        unmatched_groups: Group keys whose ``subject_label`` or
            ``rater_label`` axis column was absent, so the check could not
            be evaluated. Reset at the top of each :meth:`analyze`.

    Examples:
        Basic — three timepoints (subjects) × three replicates (raters)
        with tight replicate agreement at each timepoint; the check adds
        ``QC_ICC_Metric`` plus the per-group summary columns:

        >>> import pandas as pd
        >>> from phenotypic.analysis._icc import ICC
        >>> data = pd.DataFrame({
        ...     "Plate": ["P1"] * 9,
        ...     "Metadata_Time": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...     "Metadata_Replicate": [1, 2, 3] * 3,
        ...     "Size_Area": [
        ...         10.0, 10.1, 9.9,
        ...         20.0, 20.2, 19.8,
        ...         40.0, 40.1, 39.9,
        ...     ],
        ... })
        >>> chk = ICC(on="Size_Area", groupby=["Plate"])
        >>> result = chk.analyze(data)
        >>> "QC_ICC_Metric" in result.columns
        True

        Advanced — when the rater axis column is absent the two-way model
        cannot be built: the metric is NaN *and* the group is recorded as
        unmatched so the not-evaluated check is loud, not a silent pass:

        >>> no_rater = pd.DataFrame({
        ...     "Plate": ["P1"] * 3,
        ...     "Metadata_Time": [0, 1, 2],
        ...     "Size_Area": [10.0, 20.0, 40.0],
        ... })
        >>> chk = ICC(on="Size_Area", groupby=["Plate"])
        >>> result = chk.analyze(no_rater)
        >>> bool(result["QC_ICC_Metric"].isna().all())
        True
        >>> chk.unmatched_groups
        [('P1',)]
    """

    name: ClassVar[str] = "ICC"
    _HIGHER_IS_BAD: ClassVar[bool] = False
    _exposes_agg_func: ClassVar[bool] = False
    _measurement_infoclass = QUALITY_ICC

    warn_threshold: float = 0.75
    fail_threshold: float = 0.50

    subject_label: ColumnRef = "Metadata_Time"
    rater_label: ColumnRef = "Metadata_Replicate"

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reset :attr:`unmatched_groups` and run the base ``analyze``.

        Re-running the check on a different measurement frame must not
        carry over groups flagged as unmatched (missing axis column) from
        a previous run, so the list is cleared before delegating to the
        base class.

        Args:
            data: Measurement frame to evaluate.

        Returns:
            The augmented frame from :meth:`QualityCheck.analyze`.
        """
        self.unmatched_groups = []
        return super().analyze(data)

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute the per-group ICC(2,1) and broadcast it to every row.

        Args:
            group: One group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (a copy) with four new columns appended:
            ``QC_ICC_NumSubjects``, ``QC_ICC_NumRaters``,
            ``QC_ICC_NumMembers``, ``QC_ICC_Metric``. The metric column is
            ``NaN`` for groups that hit any of the guard paths documented
            on the class. When an axis column is absent the group key is
            additionally recorded in :attr:`unmatched_groups`.
        """
        out = group.copy()

        n_subjects_col = str(QUALITY_ICC.NUM_SUBJECTS)
        n_raters_col = str(QUALITY_ICC.NUM_RATERS)
        n_members_col = str(QUALITY_ICC.NUM_MEMBERS)
        metric_col = self.metric_col()

        out[n_subjects_col] = 0
        out[n_raters_col] = 0
        out[n_members_col] = int(len(out))
        out[metric_col] = np.nan

        if len(out) == 0:
            return out

        # LOUD missing-axis guard: a not-evaluated check must be visible,
        # not a silent green pass. Record the group key so the GUI/CLI can
        # show "could not run", then fall through to the NaN metric.
        if (
            self.subject_label not in out.columns
            or self.rater_label not in out.columns
        ):
            self.unmatched_groups.append(self._group_key(out))
            return out

        matrix = self._build_matrix(out)
        if matrix is None:
            return out

        n_subjects, n_raters = matrix.shape
        out[n_subjects_col] = int(n_subjects)
        out[n_raters_col] = int(n_raters)
        out[metric_col] = self._icc_2_1(matrix)

        return out

    def _group_key(self, group: pd.DataFrame) -> tuple:
        """Extract the ``groupby`` key for a single group as a tuple.

        Args:
            group: One group frame. The ``self.groupby`` columns are
                constant within the group, so the first row suffices.

        Returns:
            A tuple of the group's ``groupby`` values, regardless of
            whether ``groupby`` has one or many columns.
        """
        row = group.iloc[0]
        return tuple(row[col] for col in self.groupby)

    def _build_matrix(self, group: pd.DataFrame) -> np.ndarray | None:
        """Pivot the group into a complete ``subjects × raters`` matrix.

        The missing-axis-column case is handled by :meth:`_compute` (which
        records the group as unmatched); by the time this runs both axis
        columns are present, so a ``None`` here means a genuine
        insufficient-data case (empty, duplicated/incomplete cell, or
        ``n_subjects < 2`` / ``n_raters < 2``).

        Args:
            group: One group frame whose axis columns are both present.

        Returns:
            A ``(n_subjects, n_raters)`` float matrix with one observation
            per cell, or ``None`` when the matrix is empty,
            incomplete/duplicated, or too small for a two-way ANOVA
            (``n_subjects < 2`` or ``n_raters < 2``).
        """
        sub = group[[self.subject_label, self.rater_label, self.on]].dropna()
        if sub.empty:
            return None

        # A balanced two-way design needs exactly one observation per
        # (subject, rater) cell; a duplicate makes the pivot ambiguous.
        if sub.duplicated([self.subject_label, self.rater_label]).any():
            return None

        pivot = sub.pivot(
            index=self.subject_label,
            columns=self.rater_label,
            values=self.on,
        )

        # Any NaN cell means the (subject, rater) combination was missing.
        if pivot.isna().to_numpy().any():
            return None

        matrix = pivot.to_numpy(dtype=float)
        n_subjects, n_raters = matrix.shape
        if n_subjects < 2 or n_raters < 2:
            return None

        return matrix

    @staticmethod
    def _icc_2_1(matrix: np.ndarray) -> float:
        """Compute ICC(2,1) two-way random, absolute agreement.

        Args:
            matrix: A complete ``(n_subjects, n_raters)`` observation
                matrix with at least two rows and two columns.

        Returns:
            The ICC(2,1) coefficient as a float, or ``NaN`` when the total
            variance is zero (all values identical).
        """
        n = matrix.shape[0]  # subjects
        k = matrix.shape[1]  # raters

        grand_mean = float(matrix.mean())
        row_means = matrix.mean(axis=1)
        col_means = matrix.mean(axis=0)

        ss_total = float(((matrix - grand_mean) ** 2).sum())
        if ss_total == 0.0:
            return float("nan")

        # Two-way ANOVA sums of squares.
        ss_rows = float(k * ((row_means - grand_mean) ** 2).sum())
        ss_cols = float(n * ((col_means - grand_mean) ** 2).sum())
        ss_error = ss_total - ss_rows - ss_cols

        ms_rows = ss_rows / (n - 1)
        ms_cols = ss_cols / (k - 1)
        ms_error = ss_error / ((n - 1) * (k - 1))

        denominator = (
            ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
        )
        if denominator == 0.0:
            return float("nan")

        return float((ms_rows - ms_error) / denominator)
