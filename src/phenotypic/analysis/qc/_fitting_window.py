"""The fitting-window quality check — Rule D, wired to a measurement frame.

The algorithm lives in :mod:`._fitting_window_rule`, a numpy-only transcription
of a polars reference with its own golden fixture. This module is the
*pipeline* half: it turns a measurement frame into one mean-CV series per
condition, judges each against one pooled bar, and reports the verdict per
condition.

Nothing here spells a measurement or metadata column. ``on``, ``groupby``,
``replicate_by`` and ``time_col`` are all user parameters, and the only columns
the module names for itself are the ``QC_Window_*`` headers it emits (asked of
:class:`~phenotypic.schema.QUALITY_WINDOW`). A source-guard test pins that.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Sequence

import numpy as np
import pandas as pd
from pydantic import PrivateAttr, field_validator, model_validator

from phenotypic.abc_.plotting import PlotOutput, PlotPage, PlotQc, canonical_group_key
from phenotypic.analysis.abc_._quality_check import QualityCheck
from phenotypic.analysis.abc_._set_analyzer import (
    normalize_measurement_metadata_columns,
    normalize_metadata_column_reference,
    normalize_metadata_column_references,
)
from phenotypic.schema import QUALITY_WINDOW
from phenotypic.sdk_ import ColumnRef, ColumnRefList

from ._expected_positions import numeric_order_key
from ._fitting_window_rule import (
    ABRUPT_K,
    EDGE_FRAMES,
    MIN_BAR_OVER_NOISE,
    MIN_REPLICATES,
    MIN_SERIES_FRAMES,
    NEAR_MIN_TOLERANCE,
    PERSIST,
    ROUGH_K,
    ROUGH_LAG,
    ROUGH_ROLL,
    ROUGH_SPAN,
    SMOOTH,
    MeanCvSeries,
    RuleDResult,
    WindowOutcome,
    abrupt_bar,
    derivatives,
    mean_cv_series,
    rule_d,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import plotly.graph_objects as go

logger = logging.getLogger(__name__)

__all__ = ["FittingWindow"]

#: CV band edges the figure draws, which are also this check's default
#: ``warn_threshold``/``fail_threshold`` — so the status and the figure agree by
#: construction rather than by coincidence.
_CV_BAND_EDGES = (0.10, 0.30)

#: Characters a condition slug may not carry (it names a figure file).
_UNSAFE = re.compile(r"[^A-Za-z0-9.]+")


def _condition_slug(groupby: Sequence[str], values: Sequence[Any]) -> str:
    """Return the stable page key for one condition.

    The readable part is the condition's values, sanitized and joined with
    ``_``; the suffix is six hex digits of a hash over the *typed* values, so
    two conditions whose values sanitize alike (``"a b"`` and ``"a-b"``), or
    differ only in type (``1`` and ``"1"``), still get different keys. The key
    depends on this condition alone, never on which others are present.

    Args:
        groupby: The condition column names.
        values: This condition's values, in ``groupby`` order.

    Returns:
        A filesystem-safe name, e.g. ``"Condition.A-3f9a1c"``.

    Raises:
        ValueError: If ``groupby`` and ``values`` differ in length.

    Examples:
        >>> slug = _condition_slug(["condition"], ["Condition A"])
        >>> slug.startswith("Condition-A-")
        True
        >>> slug == _condition_slug(["condition"], ["Condition A"])
        True
        >>> _condition_slug(["plate"], [1]) == _condition_slug(["plate"], ["1"])
        False
    """
    if len(groupby) != len(values):
        raise ValueError(
            f"groupby has {len(groupby)} columns but {len(values)} values were given"
        )
    readable = "_".join(
        _UNSAFE.sub("-", str(value)).strip("-.") or "na" for value in values
    )
    key = canonical_group_key(list(zip(groupby, values)))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:6]
    return f"{readable}-{digest}"


def _plain_value(value: Any) -> Any:
    """Return a hashable, slug-safe Python scalar for one condition value.

    numpy scalars become Python scalars, and every flavour of missing (``NaN``,
    ``None``, ``pd.NA``, ``NaT``) becomes ``None``. Both matter: the condition
    key is a dict key, and ``NaN != NaN`` would make a null condition
    unreachable.

    Args:
        value: One ``groupby`` value, as read off a group's first row.

    Returns:
        A plain Python scalar, or ``None`` for any missing value.
    """
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError):  # pragma: no cover - defensive
            pass
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):  # pragma: no cover - never a scalar key
        pass
    return value


class _PendingCondition(NamedTuple):
    """One condition's inputs, held between the series pass and the bar."""

    values: tuple
    series: MeanCvSeries


class _ConditionOutcome(NamedTuple):
    """Everything the check knows about one condition after ``analyze``.

    Held on a ``PrivateAttr`` because :meth:`QualityCheck.analyze` hands
    ``_compute`` one group at a time and the bar those results rest on is
    pooled over all of them.
    """

    #: The ``groupby`` values, missing values normalized to ``None``.
    values: tuple
    #: The condition's stable key — the name of its figure page.
    slug: str
    #: Counted frames, their mean replicate CV, and the groups behind each.
    series: MeanCvSeries
    #: Rule D's verdict, against the run-wide pooled bar.
    result: RuleDResult
    #: The centred rolling-mean trend, aligned with ``series.frames``.
    trend: np.ndarray


class FittingWindow(QualityCheck, PlotQc):
    """Propose the frames over which a condition's replicates agree well enough to fit.

    For each ``groupby`` combination — one **condition** — the check builds the
    per-frame mean replicate coefficient of variation of ``on``, then asks Rule D
    where that curve is flat and low: the span a growth model can be fitted over
    without the fit being driven by replicate disagreement. A window is the run
    of frames where replicates agree, not "where it is growing". A condition it
    can answer for is graded by the mean CV inside that span; a condition it
    cannot is failed, and ``QC_Window_Outcome`` says why.

    **The bar is pooled over the whole run, and that is deliberate.** Rule D
    judges every condition against **one** absolute abrupt-change threshold,
    computed from every condition's curvature at once, because a per-condition
    threshold is exactly the unequal treatment the rule exists to remove. Three
    consequences a reader should know:

    * ``QC_Window_Bar`` is a run-level constant — the same value on every row of
      every condition.
    * **Adding or removing a condition can move another condition's window.**
      The verdict is about a condition *within this run*, not an intrinsic
      property of it.
    * **A condition analyzed alone is never too noisy, at the defaults.** With
      one series in the pool, the bar is that series' quiet level plus
      ``abrupt_k`` of its own noise units, and ``abrupt_k`` (10) exceeds
      ``min_bar_over_noise`` (3). So ``too_noisy`` means "too noisy *next to the
      other conditions in this run*". A condition alone can still go without a
      window for the other reasons (too few frames, or its best agreement in
      its last frames), but never for noise. Setting ``abrupt_k`` below
      ``min_bar_over_noise`` removes that guarantee.

    ``QualityCheck.analyze`` hands ``_compute`` one group, so the bar cannot be
    computed there; :meth:`analyze` is overridden to compute every series, the
    pooled bar and every Rule D result once, before delegating upward.

    **What a window is good for.** Fitting a growth model only inside the
    window gives a trustworthy **rate** ``v``, because the frames where
    replicates disagree are left out. It does **not** give a trustworthy lag or
    starting size: a window usually opens after the lag, so a windowed fit has
    no frames that constrain ``lambda`` or ``s0``. **Take lag and starting size
    from a fit over all frames**, and the rate from the windowed fit.

    **The time axis is a frame index, not elapsed hours.** Images of different
    plates never share an hour value, so hours cannot line plates up across a
    condition; :class:`~phenotypic.post.AddFrameIndex` numbers each plate's
    images in capture order, 1-based. Rule D compares frames with each other and
    never with an absolute number, so shifting the axis by one shifts every
    reported window by exactly one and changes no outcome. A window is
    therefore a span of frame indices: to fit on hours, filter the table on the
    frame index first, then fit with the hours column as the time axis (see the
    example). Rule D needs at least 14 counted frames at the default settings.

    **The metric.** Mean measured CV over the proposed window when there is one,
    ``+inf`` when there is not — so the base class's ``metric >=
    fail_threshold`` fails every windowless condition with no override of its
    status logic. Windowless conditions therefore tie on ``+inf`` and are not
    ordered among themselves; ``QC_Window_BestCV`` (the series minimum) is
    carried so a reader can still see how close each came. Using that minimum
    *as* the metric would pass a condition Rule D rejected as too noisy, on one
    good frame.

    **Diagnostic only.** ``supports_object_curation`` is ``False``: the verdict
    is about a condition, so ``QC_Window_Flag`` never offers colonies for
    removal, and :meth:`to_table` collapses to one row per condition rather than
    writing one identical row per colony into ``qc.duckdb``.

    Args:
        on: **The input metric column** — the measurement whose replicate CV is
            tracked. Required, inherited; nothing about this check is specific
            to one column.
        groupby: **The condition columns.** One group is one condition.
            Required, inherited.
        replicate_by: Columns defining one replicate group (for example, a
            plate or a strain). A group counts at a frame only when it has at
            least ``min_replicates`` detected colonies *and* a mean ``on`` above
            the floor.
        time_col: The per-image frame index that lines plates up across a
            condition. Must hold whole numbers.
        min_replicates: Detected colonies a replicate group needs at a frame to
            be counted into that frame's mean CV.
        smooth: Trend window, in frames, for the centred rolling mean.
        abrupt_k: How far above the pooled quiet curvature level an acceleration
            must go to count as abrupt.
        persist: Frames an excursion must persist to count as an event.
        rough_roll: Window of the roughness rolling mean.
        rough_k: Multiple of a condition's own recent roughness that ends a
            window.
        rough_lag: How far back the trailing roughness baseline starts.
        rough_span: How wide that trailing baseline window is.
        near_min_tolerance: How close to its own minimum the smoothed CV must
            come for a frame to belong to the minimum's basin, where the window
            opens.
        min_bar_over_noise: Multiples of a condition's own curvature noise the
            pooled bar must exceed for the test to discriminate at all. Below
            it the rule abstains.
        edge_frames: Frames at each end excluded from evaluation.
        min_series_frames: Shortest series the rule will assess. A plain
            integer, deliberately not re-derived from ``smooth`` and
            ``edge_frames`` — in the reference it comes from a different,
            narrower smoother, and re-deriving it here would reject series the
            reference accepts.
        warn_threshold: Window mean CV at which ``Status`` becomes ``"warn"``.
        fail_threshold: Window mean CV at which ``Status`` becomes ``"fail"``.

    Raises:
        KeyError: From :meth:`analyze`, if ``on``, ``time_col`` or any
            ``groupby``/``replicate_by`` column is absent from the frame.
        ValueError: From :meth:`analyze`, if ``time_col`` holds a value that is
            not a whole number. At construction, if ``replicate_by`` is empty or
            the thresholds are ordered against the metric's bad direction
            (surfaced by pydantic as a ``ValidationError``).

    Examples:
        A labelled synthetic series: four conditions of three replicate plates,
        thirty images six hours apart (see
        :func:`~phenotypic.data.make_synthetic_window_series`). Number each
        plate's images, then judge every condition in one run:

        >>> from phenotypic.analysis import LinearLagModel
        >>> from phenotypic.analysis.qc import FittingWindow
        >>> from phenotypic.data import make_synthetic_window_series
        >>> from phenotypic.post import AddFrameIndex
        >>> from phenotypic.schema import CULTURE, SAMPLE, SIZE
        >>> condition, replicate = "Metadata_Condition", str(SAMPLE.BIO_REPLICATE)
        >>> frame, hours = str(CULTURE.FRAME_INDEX), str(CULTURE.TIME)
        >>> radius = str(SIZE.ROBUST_MEAN_RADIUS)
        >>> table = AddFrameIndex(plate_by=["PlateID"]).apply(
        ...     make_synthetic_window_series()
        ... )
        >>> window = FittingWindow(
        ...     on=radius, groupby=[condition], replicate_by=[replicate], time_col=frame
        ... )
        >>> _ = window.analyze(table)
        >>> verdict = window.to_table().set_index(condition)
        >>> for name, row in verdict.iterrows():
        ...     print(name, row["QC_Window_Outcome"],
        ...           row["QC_Window_Start"], row["QC_Window_End"])
        Condition A found 9 26
        Condition B found 19 26
        Condition C found 9 17
        Condition D too_noisy <NA> <NA>

        A and B grow identically; B's replicates take longer to agree, so its
        window opens later. D gets no window in this run, and says why:

        >>> verdict.loc["Condition D", "QC_Window_Reason"]
        "bar is 1.1x this curve's own noise (<3): not assessable"

        The handoff to a growth fit is a pandas filter on the **frame index**,
        then a fit on **hours**. The windowed fit gives the rate (the series
        was built with 0.25 px/h):

        >>> a = table[table[condition] == "Condition A"]
        >>> start = verdict.loc["Condition A", "QC_Window_Start"]
        >>> end = verdict.loc["Condition A", "QC_Window_End"]
        >>> in_window = a[a[frame].between(start, end)]
        >>> fit = LinearLagModel(on=radius, groupby=[replicate], time_label=hours)
        >>> windowed = fit.analyze(in_window)
        >>> windowed["LinearLagModel_RobustMeanRadius_v"].round(2).tolist()
        [0.25, 0.25, 0.25]

        Lag and starting size come from a fit over **all** frames (the series
        was built with a 30 h lag and an 8 px start):

        >>> full = fit.analyze(a)
        >>> full["LinearLagModel_RobustMeanRadius_lambda"].round().tolist()
        [30.0, 30.0, 30.0]
        >>> full["LinearLagModel_RobustMeanRadius_s0"].round().tolist()
        [8.0, 8.0, 8.0]

        The windowed fit's own lag and starting size are not those: its window
        opens after the lag, so no frame in it constrains either. On the three
        replicates it puts the lag at 36 to 42 h and the start at 9.5 to 11 px:

        >>> lag = windowed["LinearLagModel_RobustMeanRadius_lambda"]
        >>> start_size = windowed["LinearLagModel_RobustMeanRadius_s0"]
        >>> bool(lag.between(35, 43).all()), bool(start_size.between(9, 11.5).all())
        (True, True)
    """

    name: ClassVar[str] = "Window"
    _HIGHER_IS_BAD: ClassVar[bool] = True
    _exposes_agg_func: ClassVar[bool] = False
    _measurement_infoclass: ClassVar[type | None] = QUALITY_WINDOW
    #: A verdict about a condition, not about a colony.
    supports_object_curation: ClassVar[bool] = False
    #: No per-object key: there is nothing here to curate.
    member_key_cols: ClassVar[tuple[str, ...]] = ()

    # The figure's own CV band edges, so the status and the figure grade a
    # condition identically.
    warn_threshold: float = _CV_BAND_EDGES[0]
    fail_threshold: float = _CV_BAND_EDGES[1]

    replicate_by: ColumnRefList
    time_col: ColumnRef

    min_replicates: int = MIN_REPLICATES
    smooth: int = SMOOTH
    abrupt_k: float = ABRUPT_K
    persist: int = PERSIST
    rough_roll: int = ROUGH_ROLL
    rough_k: float = ROUGH_K
    rough_lag: int = ROUGH_LAG
    rough_span: int = ROUGH_SPAN
    near_min_tolerance: float = NEAR_MIN_TOLERANCE
    min_bar_over_noise: float = MIN_BAR_OVER_NOISE
    edge_frames: int = EDGE_FRAMES
    min_series_frames: int = MIN_SERIES_FRAMES

    #: The run-wide pooled abrupt-change bar. NaN until :meth:`analyze` runs.
    _bar: float = PrivateAttr(default=float("nan"))
    #: Per-condition state, keyed by the normalized ``groupby`` value tuple.
    _conditions: dict = PrivateAttr(default_factory=dict)

    @field_validator("replicate_by", mode="before")
    @classmethod
    def _normalize_replicate_by(cls, value: Any) -> Any:
        """Resolve known metadata references; leave anything else alone.

        The same normalization ``on`` and ``groupby`` get, so a bare label or a
        historical per-topic header resolves while a non-metadata axis (a
        computed frame index, say) survives untouched.
        """
        if value is None:
            return None
        return normalize_metadata_column_references(value)

    @field_validator("time_col", mode="before")
    @classmethod
    def _normalize_time_col(cls, value: Any) -> str:
        """Resolve a known metadata reference for the frame-index column."""
        if not isinstance(value, str):
            raise ValueError("time_col must be a string column reference")
        return normalize_metadata_column_reference(value)

    @model_validator(mode="after")
    def _validate_replicate_by_is_populated(self) -> "FittingWindow":
        """Reject an empty ``replicate_by``.

        A CV needs replicates. With no grouping columns every colony in a
        condition would be one "group", and the check would silently measure
        spread across the whole condition instead of within replicate sets.

        Returns:
            The validated instance.

        Raises:
            ValueError: If ``replicate_by`` names no columns.
        """
        if not self.replicate_by:
            raise ValueError("replicate_by must name at least one column")
        return self

    @property
    def time_label(self) -> str:
        """The frame-index column, under the name the base class asks for.

        ``QualityCheck.table_spec`` records a check's time axis by reading
        ``time_label``, which is what the other time-course checks call the
        parameter. This check names it ``time_col``, so without this alias
        every ``FittingWindow`` row in the ``qc_modules`` catalog would record
        a null ``time_col`` for a check whose whole subject is a time series.
        """
        return self.time_col

    # --- analysis ---------------------------------------------------------------

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build every condition's series, pool the bar once, then delegate upward.

        The order is forced by the algorithm, not by taste. Rule D tests every
        condition against **one** bar pooled over all of them, and
        :meth:`QualityCheck.analyze` hands ``_compute`` a single group — so the
        bar has to exist before the first group is computed. This method makes
        three passes:

        1. per condition, the mean-CV series;
        2. once, :func:`abrupt_bar` over **every** series;
        3. per condition, :func:`rule_d` against that bar.

        The results are stashed on a private attribute and ``_compute`` only
        looks its own condition up. A per-condition bar would type-check, run,
        and silently reclassify conditions.

        Args:
            data: The measurement frame. On the CLI path the post-applied,
                metadata-joined mirror.

        Returns:
            The augmented frame from :meth:`QualityCheck.analyze`. Its
            ``time_col`` is the caller's own column, unchanged: a frame index
            stored as text is read as numbers for the series, and comes back
            as text.

        Raises:
            KeyError: If ``on``, ``time_col`` or any ``groupby`` /
                ``replicate_by`` column is absent.
            ValueError: If ``time_col`` holds a value that is not a whole
                number.
        """
        data = normalize_measurement_metadata_columns(data)
        missing = [
            column
            for column in (
                self.on,
                self.time_col,
                *self.groupby,
                *self.replicate_by,
            )
            if column not in data.columns
        ]
        if missing:
            raise KeyError(
                f"Missing required columns for FittingWindow: {missing}"
            )

        # Coerce the time axis once, at the entrance, rather than at each of
        # the places that order by it: the pre-aggregation sort inside
        # :func:`mean_cv_series` and that function's per-frame ``groupby``,
        # whose order *is* the series axis Rule D's rolling windows run over.
        # A ``time_col`` stored as text sorts frame 10 between 1 and 2, which
        # scrambles the series and every window bound read off it while every
        # published number still looks plausible. The coerced copy feeds the
        # series only: the frame handed back keeps the caller's own column.
        series_input = data
        ordered_time = numeric_order_key(data[self.time_col])
        if ordered_time is not data[self.time_col]:
            series_input = data.assign(**{self.time_col: ordered_time})

        pending: dict[tuple, _PendingCondition] = {}
        pool: list[tuple[np.ndarray, np.ndarray]] = []
        for _, group in series_input.groupby(self.groupby, dropna=False):
            values = self._condition_values(group)
            series = mean_cv_series(
                group,
                on=self.on,
                replicate_by=list(self.replicate_by),
                time_col=self.time_col,
                min_replicates=self.min_replicates,
            )
            pending[values] = _PendingCondition(values=values, series=series)
            pool.append((series.frames, series.mean_cv))

        bar = float(
            abrupt_bar(
                pool,
                min_series_frames=self.min_series_frames,
                abrupt_k=self.abrupt_k,
                smooth=self.smooth,
                rough_roll=self.rough_roll,
                edge_frames=self.edge_frames,
            )
        )

        conditions: dict[tuple, _ConditionOutcome] = {}
        for values, item in pending.items():
            result = rule_d(
                item.series.frames,
                item.series.mean_cv,
                bar,
                smooth=self.smooth,
                persist=self.persist,
                rough_roll=self.rough_roll,
                rough_k=self.rough_k,
                rough_lag=self.rough_lag,
                rough_span=self.rough_span,
                near_min_tolerance=self.near_min_tolerance,
                min_bar_over_noise=self.min_bar_over_noise,
                edge_frames=self.edge_frames,
                min_series_frames=self.min_series_frames,
            )
            conditions[values] = _ConditionOutcome(
                values=values,
                slug=_condition_slug(list(self.groupby), list(values)),
                series=item.series,
                result=result,
                trend=self._trend(item.series),
            )

        # ``_compute`` reads both during ``super().analyze``, so they have to be
        # on the instance before the call — but a failure inside it (the metric
        # coercion, the concat) must not leave run N's verdicts beside run
        # N-1's frame. Staged, and rolled back if the call does not return.
        previous = (self._bar, self._conditions)
        self._bar, self._conditions = bar, conditions
        try:
            return super().analyze(data)
        except BaseException:
            self._bar, self._conditions = previous
            raise

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Broadcast this condition's stashed verdict across its rows.

        Looks the condition up rather than computing anything: the series, the
        pooled bar and Rule D's answer were all settled in :meth:`analyze`.

        Args:
            group: One group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (a copy) with the metric and every
            ``QC_Window_*`` extra column appended.
        """
        state: _ConditionOutcome = self._conditions[self._condition_values(group)]
        result = state.result
        start, end = result.window if result.window is not None else (None, None)
        series = state.series

        out = group.copy()
        out[str(QUALITY_WINDOW.OUTCOME)] = str(result.outcome)
        out[str(QUALITY_WINDOW.START)] = pd.array(
            [start] * len(out), dtype="Int64"
        )
        out[str(QUALITY_WINDOW.END)] = pd.array([end] * len(out), dtype="Int64")
        out[str(QUALITY_WINDOW.REASON)] = result.detail
        out[str(QUALITY_WINDOW.BEST_CV)] = self._best_cv(series)
        out[str(QUALITY_WINDOW.BAR)] = self._bar
        out[str(QUALITY_WINDOW.N_GROUPS)] = pd.array(
            [self._n_groups(series)] * len(out), dtype="Int64"
        )
        out[str(QUALITY_WINDOW.N_FRAMES_COUNTED)] = int(len(series.frames))
        out[self.metric_col()] = self._metric(state)
        return out

    def to_table(self) -> pd.DataFrame:
        """Return one row per condition — the grain this check's verdict has.

        The inherited :meth:`QualityCheck.to_table` is member-level, which for a
        per-condition verdict would write one identical row per colony into
        ``qc.duckdb`` and make ``summary()``'s ``n_members`` a colony count.
        This collapses to the condition grain the same way
        :meth:`GridOccupancy.to_table` does: the constant ``QC_Window_*``
        extras, merged with the metric and status from :meth:`summary`.

        Returns:
            ``[*groupby, QC_Window_Outcome, _Start, _End, _Reason, _BestCV,
            _Bar, _NGroups, _NFramesCounted, QC_Window_Metric,
            QC_Window_Status, QC_Window_Flag]``.
        """
        df = self._latest_measurements
        extras = [
            str(member)
            for member in QUALITY_WINDOW
            if str(member) in df.columns
        ]
        first = (
            df.groupby(self.groupby, dropna=False)[extras]
            .first()
            .reset_index()
        )
        summary = self.summary().rename(
            columns={
                "qc_worst_metric": self.metric_col(),
                "qc_status": self.status_col(),
            }
        )
        merged = first.merge(
            summary[[*self.groupby, self.metric_col(), self.status_col()]],
            on=list(self.groupby),
            how="left",
        )
        # Every row of a condition carries the same metric, so the group-level
        # status is the row-level one; the flag follows it, as it does on the
        # base class.
        merged[self.flag_col()] = merged[self.status_col()] == "fail"
        return merged

    # --- the figure -------------------------------------------------------------

    def inspect(
        self,
        subject: Any = None,
        *,
        for_save: bool = False,
        **kwargs: Any,
    ) -> PlotOutput:
        """Return one page per condition: measured mean CV, trend, bands, window.

        Measured mean CV dashed, the smoothed trend solid, the three CV bands
        shaded, the proposed window banded, and the outcome in the title
        whenever there is no window (which is the only case where the title has
        something to add).

        Pages are keyed by a stable per-condition slug, so a figure file and a
        condition always name each other.

        Args:
            subject: Ignored — the check reads its own analyzed state.
            for_save: Accepted for the plotting contract; the geometry is the
                same either way.
            **kwargs: ``height`` overrides the per-page figure height.

        Returns:
            A :class:`PlotOutput` with one :class:`PlotPage` per condition, in
            slug order.

        Raises:
            RuntimeError: If :meth:`analyze` has not run.
        """
        import plotly.graph_objects as go

        del subject, for_save
        if not self._conditions:
            raise RuntimeError("call analyze() first")

        height = kwargs.get("height", 420)
        pages: list[PlotPage] = []
        for state in sorted(self._conditions.values(), key=lambda s: s.slug):
            pages.append(
                PlotPage(
                    key=state.slug,
                    label=self._condition_label(state),
                    metadata={
                        column: _metadata_scalar(value)
                        for column, value in zip(self.groupby, state.values)
                    },
                    figure=self._condition_figure(state, plotly_go=go, height=height),
                )
            )
        return PlotOutput(pages=tuple(pages))

    def report(self, subject: Any = None, **kwargs: Any) -> PlotOutput:
        """Return the complete per-condition report (every page)."""
        return self.inspect(subject, **kwargs)

    def _condition_figure(
        self, state: _ConditionOutcome, *, plotly_go: Any, height: int
    ) -> "go.Figure":
        """Draw one condition's mean-CV chart.

        Args:
            state: The condition to draw.
            plotly_go: ``plotly.graph_objects``, imported by the caller so the
                lazy-import contract is visible at one place.
            height: Figure height in pixels.

        Returns:
            The figure.
        """
        warn, fail = self.warn_threshold, self.fail_threshold
        frames = state.series.frames
        figure = plotly_go.Figure()
        for lower, upper, colour in (
            (0.0, warn, "rgba(0, 158, 115, 0.10)"),
            (warn, fail, "rgba(230, 159, 0, 0.10)"),
            (fail, max(fail * 2, 1.0), "rgba(213, 94, 0, 0.10)"),
        ):
            figure.add_hrect(
                y0=lower, y1=upper, line_width=0, fillcolor=colour, layer="below"
            )
        if state.result.window is not None:
            start, end = state.result.window
            figure.add_vrect(
                x0=start,
                x1=end,
                line_width=0,
                fillcolor="rgba(0, 54, 96, 0.12)",
                layer="below",
            )
        figure.add_trace(
            plotly_go.Scatter(
                x=frames,
                y=state.series.mean_cv,
                mode="lines",
                name="measured",
                line={"dash": "dash", "color": "#56B4E9"},
            )
        )
        figure.add_trace(
            plotly_go.Scatter(
                x=frames,
                y=state.trend,
                mode="lines",
                name="trend",
                line={"color": "#003660"},
            )
        )
        title = self._condition_label(state)
        if state.result.window is None:
            title = f"{title} — no window ({state.result.outcome})"
        figure.update_layout(
            title=title,
            xaxis_title=self.time_col,
            yaxis_title=f"mean replicate CV of {self.on}",
            height=height,
        )
        return figure

    def _condition_label(self, state: _ConditionOutcome) -> str:
        """Return a readable name for one condition."""
        if len(self.groupby) == 1:
            return str(state.values[0])
        return " | ".join(
            f"{column}={value}"
            for column, value in zip(self.groupby, state.values)
        )

    # --- internals --------------------------------------------------------------

    def _condition_values(self, group: pd.DataFrame) -> tuple:
        """Return one group's ``groupby`` values as a normalized tuple key.

        **Read column-wise, never row-wise.** ``group.iloc[0]`` returns a Series
        with the common dtype of *every* column in the frame, so on an
        all-numeric frame an integer condition value ``7`` comes back as
        ``7.0`` — and :func:`_condition_slug` hashes ``int`` and ``float``
        distinctly, so the page key would depend on which other columns the
        frame happens to carry. ``group[column].iloc[0]`` keeps each value in
        its own column's dtype.
        """
        return tuple(
            _plain_value(group[column].iloc[0]) for column in self.groupby
        )

    def _trend(self, series: MeanCvSeries) -> np.ndarray:
        """Return the centred rolling-mean trend aligned with ``series.frames``."""
        if len(series.frames) == 0:
            return np.empty(0, dtype=float)
        return derivatives(
            series.frames,
            series.mean_cv,
            smooth=self.smooth,
            rough_roll=self.rough_roll,
            edge_frames=self.edge_frames,
        ).smooth_v

    def _metric(self, state: _ConditionOutcome) -> float:
        """Return the condition's metric: window mean CV, or ``+inf``.

        ``+inf`` rather than a fallback value is what makes a windowless
        condition fail through the base class's own comparison, with no
        override of its status logic.
        """
        result = state.result
        if result.outcome is not WindowOutcome.FOUND or result.window is None:
            return math.inf
        start, end = result.window
        frames = state.series.frames
        inside = (frames >= start) & (frames <= end)
        if not bool(inside.any()):  # pragma: no cover - a found window has frames
            return math.inf
        return float(np.mean(state.series.mean_cv[inside]))

    @staticmethod
    def _best_cv(series: MeanCvSeries) -> float:
        """Return the series minimum, or NaN when no frame counted."""
        if len(series.mean_cv) == 0:
            return float("nan")
        return float(np.min(series.mean_cv))

    @staticmethod
    def _n_groups(series: MeanCvSeries) -> int | None:
        """Return the median replicate-group count, or ``None`` when empty.

        Rounded to the nearest integer, halves to even — Python's own ``round``
        rule, and the one the ``NGroups`` schema entry states. Truncating
        instead reported 3 for counted-group counts of ``[3, 4]``, which
        understates the sample size the verdict rests on.
        """
        if len(series.n_counted) == 0:
            return None
        return int(round(float(np.median(series.n_counted))))


def _metadata_scalar(value: Any) -> str | int | float | bool | None:
    """Coerce a condition value to something a :class:`PlotPage` may carry."""
    plain = _plain_value(value)
    if plain is None or isinstance(plain, (str, int, float, bool)):
        return plain
    return str(plain)
