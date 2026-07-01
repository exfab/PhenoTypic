"""Base ABC for metric-driven quality-control checks on measurement frames."""

from __future__ import annotations

import abc
from abc import ABC
from typing import Any, ClassVar

import pandas as pd
from pydantic import Field, model_validator

from phenotypic.schema import EXPERIMENT_METADATA, METADATA, OBJECT, QUALITY_CHECK

from ._qc_table_spec import QcTableSpec
from ._set_analyzer import SetAnalyzer


class QualityCheck(SetAnalyzer, ABC):
    """Detect quality-control issues in measurement frames.

    ``QualityCheck`` is a thin layer over :class:`SetAnalyzer` that
    standardizes how subclasses surface flagged rows for downstream
    curation. Subclasses implement a single ``_compute(group)`` hook
    that augments one group with at minimum a raw, directional
    ``QC_<name>_Metric`` column (the check's headline value in its own
    units). The base class then derives two companion columns from the
    metric, using the check's ``_HIGHER_IS_BAD`` direction flag to decide
    which side of each threshold is "bad":

    * ``QC_<name>_Flag`` (``bool``): ``True`` when the metric crosses
      ``fail_threshold`` in the bad direction. When ``_HIGHER_IS_BAD`` is
      ``True`` that means ``metric >= fail_threshold``; when ``False``
      (e.g. an agreement score where lower is worse) it means
      ``metric <= fail_threshold``. Rows with ``Flag=True`` are the ones
      the results-viewer GUI offers to mark for curation removal.
    * ``QC_<name>_Status`` (``str``): tri-state label derived from the
      same metric column. With ``_HIGHER_IS_BAD=True``: ``"pass"`` until
      ``metric >= warn_threshold`` (``"warn"``), then
      ``metric >= fail_threshold`` (``"fail"``), with
      ``warn_threshold <= fail_threshold``. With ``_HIGHER_IS_BAD=False``
      the comparisons invert to ``<=`` with
      ``fail_threshold <= warn_threshold``. Only ``"fail"`` triggers
      ``Flag=True``; the ``"warn"`` tier is informational.

    NaN metrics (e.g. under-powered replicate bins in
    :class:`ReplicateAgreement`) are treated as ``"pass"`` with
    ``Flag=False`` so degenerate groups never gate curation.

    Subclasses set class-level attributes that drive column naming,
    threshold direction, and docstring autogeneration:

    * ``name`` — short identifier composed into output column names
      (``QC_<name>_Flag`` and friends). Must be set on every concrete
      subclass.
    * ``_HIGHER_IS_BAD`` — ``True`` when a larger metric value is worse
      (the common case), ``False`` when a smaller value is worse (e.g. an
      agreement score). Intrinsic to the metric, not user-tunable; it has
      **no default on the base** and must be set on every concrete
      subclass.
    * ``_measurement_infoclass`` — optional per-subclass
      :class:`MeasurementInfo` enum documenting any check-specific
      columns the subclass emits beyond the generic
      ``Flag``/``Metric``/``Status`` trio. When set,
      ``__init_subclass__`` appends its RST table to the subclass
      docstring.

    The base class drives group iteration directly in :meth:`analyze`,
    so the abstract :meth:`SetAnalyzer._apply2group_func` is overridden
    to raise — subclasses implement ``_compute`` instead.

    Attributes:
        name: Short identifier composed into output column names. Set on
            each concrete subclass (e.g. ``"Count"``, ``"SE"``).
        warn_threshold: Metric value (in the check's own units) at which
            ``Status`` becomes ``"warn"`` in the bad direction. A pydantic
            instance field; subclasses override the default by
            re-declaring it, and callers may override it per instance.
        fail_threshold: Metric value at which ``Status`` becomes
            ``"fail"`` and ``Flag=True`` in the bad direction. A pydantic
            instance field; subclasses override the default by
            re-declaring it, and callers may override it per instance.
        unmatched_groups: Groups that the check could not evaluate (for
            example, expected counts whose group key never appeared in
            the data). Populated by subclasses that need to report
            missing combinations; empty by default.
    """

    name: ClassVar[str]
    _HIGHER_IS_BAD: ClassVar[bool]
    _exposes_agg_func: ClassVar[bool] = False
    _measurement_infoclass: ClassVar[type | None] = None

    #: Whether this check's rows map to curatable detected objects. False for
    #: diagnostic-only checks (e.g. GridOccupancy) — the Review tab hides the
    #: curation radial + tile gallery and verified-good skips them.
    supports_object_curation: ClassVar[bool] = True

    #: Per-object curation-key columns. Empty tuple when the check has no
    #: per-object key. Subclasses may narrow this.
    member_key_cols: ClassVar[tuple[str, ...]] = (
        str(METADATA.IMAGE_NAME),
        str(OBJECT.LABEL),
    )

    warn_threshold: float = 0.05
    fail_threshold: float = 0.10
    unmatched_groups: list = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_threshold_order(self) -> QualityCheck:
        """Reject thresholds ordered against the check's bad direction.

        The ``warn`` tier must always sit between ``"pass"`` and ``"fail"``
        in the metric's bad direction, so the two thresholds have a fixed
        ordering that depends only on ``_HIGHER_IS_BAD``:

        * ``_HIGHER_IS_BAD=True``  → ``warn_threshold <= fail_threshold``
          (a larger metric is worse, so ``fail`` is the higher boundary).
        * ``_HIGHER_IS_BAD=False`` → ``warn_threshold >= fail_threshold``
          (a smaller metric is worse, so ``fail`` is the lower boundary).

        Equality is allowed — it collapses the ``warn`` band so the check
        is purely pass/fail. A mis-ordered pair would silently invert the
        tri-state (e.g. a lower-is-bad check that fails *above* its warn
        line), so it raises ``ValueError`` (surfaced by pydantic as a
        ``ValidationError``) at construction.

        Intermediate ABCs that have not yet bound ``_HIGHER_IS_BAD`` skip
        the check so the base class stays instantiable for test doubles.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the thresholds are ordered against the bad
                direction implied by ``_HIGHER_IS_BAD``.
        """
        higher_is_bad = getattr(type(self), "_HIGHER_IS_BAD", None)
        if higher_is_bad is None:
            return self
        if higher_is_bad and self.warn_threshold > self.fail_threshold:
            raise ValueError(
                "warn_threshold must be <= fail_threshold for a "
                "higher-is-bad check "
                f"(got warn={self.warn_threshold}, "
                f"fail={self.fail_threshold})"
            )
        if not higher_is_bad and self.warn_threshold < self.fail_threshold:
            raise ValueError(
                "warn_threshold must be >= fail_threshold for a "
                "lower-is-bad check "
                f"(got warn={self.warn_threshold}, "
                f"fail={self.fail_threshold})"
            )
        return self

    @abc.abstractmethod
    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Add the check's metric columns to one group.

        Must add at minimum the metric column (``QC_<name>_Metric``), the
        raw headline value in the check's own units. May add
        check-specific columns documented by ``_measurement_infoclass``.
        ``Flag`` and ``Status`` are derived by the base class from the
        metric and ``_HIGHER_IS_BAD``, so subclasses must not set them
        directly.

        Args:
            group: A single group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (typically a copy) augmented with the
            ``QC_<name>_Metric`` column and any check-specific metric
            columns.
        """

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the check on every group and return the augmented frame.

        Iterates over ``data.groupby(self.groupby, dropna=False)``,
        delegates per-group computation to :meth:`_compute`, and adds
        three generic columns derived from the metric:

        * ``QC_<name>_Metric`` (carry-through from ``_compute``)
        * ``QC_<name>_Flag`` (``bool``)
        * ``QC_<name>_Status`` (``"pass"`` / ``"warn"`` / ``"fail"``)

        ``Flag`` and ``Status`` are directional. With
        ``_HIGHER_IS_BAD=True`` a row fails when ``metric >=
        fail_threshold`` and warns when ``metric >= warn_threshold``;
        with ``_HIGHER_IS_BAD=False`` the comparisons invert to ``<=``.
        A ``NaN`` metric always yields ``Status="pass"`` and
        ``Flag=False``.

        Rows are never dropped. The augmented frame is stored on
        :attr:`_latest_measurements` and returned.

        Args:
            data: Input measurement frame. Must contain ``self.on`` and
                every column in ``self.groupby``.

        Returns:
            The input frame with the three generic QC columns appended
            plus whatever ``_compute`` contributed.

        Raises:
            KeyError: If ``self.on`` or any column in ``self.groupby`` is
                missing from ``data``.
        """
        missing = [
            col for col in [self.on, *self.groupby] if col not in data.columns
        ]
        if missing:
            raise KeyError(
                f"Missing required columns for QualityCheck: {missing}"
            )

        metric_col = self.metric_col()
        flag_col = self.flag_col()
        status_col = self.status_col()

        pieces: list[pd.DataFrame] = []
        for _, group in data.groupby(self.groupby, dropna=False):
            pieces.append(self._compute(group))

        if pieces:
            result = pd.concat(pieces, axis=0)
        else:
            result = data.iloc[0:0].copy()
            result[metric_col] = pd.Series(dtype=float)

        metric = pd.to_numeric(result[metric_col], errors="coerce")
        status = pd.Series("pass", index=result.index, dtype=object)
        if self._HIGHER_IS_BAD:
            flag = metric.ge(self.fail_threshold)
            status = status.mask(metric.ge(self.warn_threshold), "warn")
            status = status.mask(metric.ge(self.fail_threshold), "fail")
        else:
            flag = metric.le(self.fail_threshold)
            status = status.mask(metric.le(self.warn_threshold), "warn")
            status = status.mask(metric.le(self.fail_threshold), "fail")
        flag = flag.fillna(False).astype(bool)
        status = status.where(metric.notna(), "pass")

        result[metric_col] = metric
        result[flag_col] = flag
        result[status_col] = status

        self._latest_measurements = result
        return result

    def summary(self) -> pd.DataFrame:
        """Return a one-row-per-group summary of the most recent analyze.

        The aggregate columns are **prefixed with ``qc_``** so they can
        never collide with a ``groupby`` column on ``reset_index`` — a
        plate-layout column literally named ``status`` or ``num_rows``
        would otherwise raise. The summary therefore always carries the
        group key columns *plus* the four prefixed aggregates.

        Returns:
            DataFrame with columns ``[*self.groupby, "qc_n_members",
            "qc_n_flagged", "qc_worst_metric", "qc_status"]``.
            ``qc_worst_metric`` is the extreme metric value in the bad
            direction across the group: ``group[metric_col].max()`` when
            ``_HIGHER_IS_BAD`` is ``True``, else ``group[metric_col].min()``.
            ``qc_status`` is the worst status across the group: ``"fail"``
            wins over ``"warn"`` which wins over ``"pass"``.
        """
        rank = {"pass": 0, "warn": 1, "fail": 2}
        inv_rank = {v: k for k, v in rank.items()}

        df = self._latest_measurements
        metric_col = self.metric_col()
        flag_col = self.flag_col()
        status_col = self.status_col()
        higher_is_bad = self._HIGHER_IS_BAD

        def _summarize(group: pd.DataFrame) -> pd.Series:
            worst = int(group[status_col].map(rank).max())
            metric = group[metric_col]
            worst_metric = metric.max() if higher_is_bad else metric.min()
            return pd.Series({
                "qc_n_members": int(len(group)),
                "qc_n_flagged": int(group[flag_col].sum()),
                "qc_worst_metric": float(worst_metric),
                "qc_status": inv_rank[worst],
            })

        grouped = df.groupby(self.groupby, dropna=False)
        summary = grouped.apply(_summarize, include_groups=False).reset_index()
        return summary

    def flagged_keys(self) -> list[tuple[str, int]]:
        """Return (``Metadata_ImageName``, ``Object_Label``) pairs to curate.

        Used by the GUI "Mark all flagged for removal" button. Requires
        the analyzed frame to carry both ``Metadata_ImageName`` and
        ``Object_Label`` columns (the curation key used by
        ``STORE_REMOVED_KEYS``). Returns an empty list when those
        columns are absent or when no rows were flagged.

        Returns:
            De-duplicated list of ``(image_file, object_label)`` tuples
            for rows where ``Flag=True``.
        """
        df = self._latest_measurements
        flag_col = self.flag_col()
        label_col = str(OBJECT.LABEL)
        if flag_col not in df.columns:
            return []
        image_name_col = str(METADATA.IMAGE_NAME)
        if image_name_col not in df.columns or label_col not in df.columns:
            return []
        flagged = df.loc[df[flag_col].fillna(False).astype(bool),
                         [image_name_col, label_col]].dropna()
        if flagged.empty:
            return []
        flagged = flagged.drop_duplicates()
        return [
            (str(getattr(row, image_name_col)), int(row.Object_Label))
            for row in flagged.itertuples(index=False)
        ]

    def group_members(self) -> dict[tuple, list[tuple[str, int, Any]]]:
        """Map each group key to its member rows for worklists/galleries.

        Walks the most recent analyzed frame and, for every group key
        produced by ``data.groupby(self.groupby, dropna=False)``, collects
        the rows that belong to it as ``(Metadata_ImageName, Object_Label,
        member_value)`` tuples, where ``member_value`` is the row's
        ``self.on`` value (the column the check operates on). The mapping
        preserves group iteration order.

        Mirrors :meth:`flagged_keys`'s guard: if the analyzed frame lacks
        either ``Metadata_ImageName`` or the object-label column, an empty
        mapping is returned rather than raising.

        Returns:
            Ordered mapping of group key (always a tuple, even for a
            single ``groupby`` column) to a list of
            ``(image_file, object_label, member_value)`` tuples. Empty
            when the curation key columns are absent.
        """
        df = self._latest_measurements
        label_col = str(OBJECT.LABEL)
        image_name_col = str(METADATA.IMAGE_NAME)
        if image_name_col not in df.columns or label_col not in df.columns:
            return {}

        members: dict[tuple, list[tuple[str, int, Any]]] = {}
        for key, group in df.groupby(self.groupby, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            image_files = group[image_name_col].tolist()
            labels = group[label_col].tolist()
            values = group[self.on].tolist()
            members[key_tuple] = [
                (str(image_file), int(label), value)
                for image_file, label, value in zip(
                    image_files, labels, values
                )
            ]
        return members

    @classmethod
    def metric_col(cls) -> str:
        """Return the metric column name for this check."""
        return f"QC_{cls.name}_Metric"

    @classmethod
    def flag_col(cls) -> str:
        """Return the flag column name for this check."""
        return f"QC_{cls.name}_Flag"

    @classmethod
    def status_col(cls) -> str:
        """Return the status column name for this check."""
        return f"QC_{cls.name}_Status"

    def results(self) -> pd.DataFrame:
        """Return the augmented frame stored by the most recent analyze()."""
        return self._latest_measurements

    def to_table(self) -> pd.DataFrame:
        """Return the module's self-describing frame to persist to DuckDB.

        Precondition: :meth:`analyze` has run (this reads
        :attr:`_latest_measurements`). The default is member-level: the
        augmented frame projected to group-key + member-key + ``on`` + every
        ``QC_<name>_*`` column (metric/flag/status AND check-specific extras)
        + context columns (``Metadata_Dataset`` and the column named by
        ``self.time_label``) when those columns are present.

        Diagnostic-only checks override to return a group-level frame.

        Returns:
            The projected DataFrame; columns vary per check (self-describing).
        """
        df = self._latest_measurements
        qc_cols = [c for c in df.columns if c.startswith(f"QC_{self.name}_")]
        context = [
            c
            for c in (str(EXPERIMENT_METADATA.DATASET), getattr(self, "time_label", None))
            if c and c in df.columns
        ]
        keep: list[str] = []
        for col in [
            *self.groupby,
            *self.member_key_cols,
            self.on,
            *context,
            *qc_cols,
        ]:
            if col in df.columns and col not in keep:
                keep.append(col)
        return df[keep].copy()

    def table_spec(self, instance_id: str) -> QcTableSpec:
        """Return the catalog descriptor for this analyzed check.

        Precondition: :meth:`analyze` has run. Reads column roles from the
        class + instance config and derives ``extra_cols`` from the augmented
        frame.

        Args:
            instance_id: The recipe entry id this check was built from.

        Returns:
            A populated :class:`QcTableSpec`.
        """
        df = self._latest_measurements
        generic = {self.metric_col(), self.flag_col(), self.status_col()}
        extra = [
            c
            for c in df.columns
            if c.startswith(f"QC_{self.name}_") and c not in generic
        ]
        time_col = getattr(self, "time_label", None)
        return QcTableSpec(
            instance_id=instance_id,
            cls_name=type(self).__name__,
            name=self.name,
            groupby_cols=list(self.groupby),
            metric_col=self.metric_col(),
            status_col=self.status_col(),
            flag_col=self.flag_col(),
            on_col=self.on,
            member_key_cols=list(self.member_key_cols),
            supports_object_curation=self.supports_object_curation,
            time_col=time_col if (time_col and time_col in df.columns) else None,
            higher_is_bad=self._HIGHER_IS_BAD,
            extra_cols=extra,
            warn_threshold=float(self.warn_threshold),
            fail_threshold=float(self.fail_threshold),
        )

    @staticmethod
    def _apply2group_func(group: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Not used by QualityCheck — implement ``_compute`` on the subclass.

        :meth:`QualityCheck.analyze` drives group iteration directly via
        :meth:`_compute`; the abstract ``_apply2group_func`` from
        :class:`SetAnalyzer` is satisfied here purely to keep the class
        instantiable. Raises ``NotImplementedError`` so accidental
        external calls fail loudly.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "QualityCheck subclasses implement _compute(group), not "
            "_apply2group_func. analyze() drives the iteration."
        )

    def show(self, *args: Any, **kwargs: Any) -> Any:
        """QualityCheck plots are Plotly-only — see :meth:`dash`.

        :class:`SetAnalyzer`'s matplotlib ``show()`` is not implemented
        for QC because the QC tab is Plotly-driven. Raising rather than
        falling back to a placeholder so notebook users discover the
        right method.

        Raises:
            NotImplementedError: Always; use :meth:`dash` instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement matplotlib "
            f"show(); use dash() for interactive output."
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Append QC and per-check RST tables to the subclass docstring.

        Skips intermediate ABCs that have not yet bound ``name``. When
        the subclass declares both a docstring and a ``name``, the
        generic :class:`QUALITY_CHECK` table is appended (substituting
        ``name`` into the column headers). If
        ``_measurement_infoclass`` is also set, its table is appended
        as well so check-specific columns are documented alongside the
        generic trio.
        """
        super().__init_subclass__(**kwargs)
        if cls.__doc__ and getattr(cls, "name", None):
            cls.__doc__ = QUALITY_CHECK.append_rst_to_doc(
                cls.__doc__, check_name=cls.name
            )
            mi = getattr(cls, "_measurement_infoclass", None)
            if mi is not None:
                cls.__doc__ = mi.append_rst_to_doc(cls.__doc__)
