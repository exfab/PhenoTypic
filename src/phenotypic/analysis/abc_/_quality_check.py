"""Base ABC for severity-driven quality-control checks on measurement frames."""

from __future__ import annotations

import abc
from abc import ABC
from typing import Any, ClassVar

import pandas as pd
from pydantic import Field

from phenotypic.schema import QUALITY_CHECK

from ._set_analyzer import SetAnalyzer


class QualityCheck(SetAnalyzer, ABC):
    """Detect quality-control issues in measurement frames.

    ``QualityCheck`` is a thin layer over :class:`SetAnalyzer` that
    standardizes how subclasses surface flagged rows for downstream
    curation. Subclasses implement a single ``_compute(group)`` hook
    that augments one group with at minimum a ``QC_<name>_Severity``
    column. The base class then derives two companion columns from
    severity:

    * ``QC_<name>_Flag`` (``bool``): ``True`` when severity is at or
      above ``severity_fail``. Rows with ``Flag=True`` are the ones the
      results-viewer GUI offers to mark for curation removal.
    * ``QC_<name>_Status`` (``str``): tri-state label derived from the
      same severity column. ``"pass"`` for ``severity < severity_warn``,
      ``"warn"`` for ``severity_warn <= severity < severity_fail``,
      ``"fail"`` for ``severity >= severity_fail``. Only ``"fail"``
      triggers ``Flag=True``; the ``"warn"`` tier is informational.

    NaN severities (e.g. under-powered replicate bins in
    :class:`ReplicateAgreement`) are treated as ``"pass"`` with
    ``Flag=False`` so degenerate groups never gate curation.

    Subclasses set two class-level attributes that drive column naming
    and docstring autogeneration:

    * ``name`` — short identifier composed into output column names
      (``QC_<name>_Flag`` and friends). Must be set on every concrete
      subclass.
    * ``_measurement_infoclass`` — optional per-subclass
      :class:`MeasurementInfo` enum documenting any check-specific
      columns the subclass emits beyond the generic
      ``Flag``/``Severity``/``Status`` trio. When set,
      ``__init_subclass__`` appends its RST table to the subclass
      docstring.

    The base class drives group iteration directly in :meth:`analyze`,
    so the abstract :meth:`SetAnalyzer._apply2group_func` is overridden
    to raise — subclasses implement ``_compute`` instead.

    Attributes:
        name: Short identifier composed into output column names. Set on
            each concrete subclass (e.g. ``"Count"``, ``"SE"``).
        severity_warn: Severity at/above which ``Status="warn"``.
            A class-level constant; subclasses may override it by
            re-declaring the ``ClassVar`` in their class body.
        severity_fail: Severity at/above which ``Status="fail"`` and
            ``Flag=True``. A class-level constant; subclasses may
            override it by re-declaring the ``ClassVar`` in their
            class body.
        unmatched_groups: Groups that the check could not evaluate (for
            example, expected counts whose group key never appeared in
            the data). Populated by subclasses that need to report
            missing combinations; empty by default.
    """

    name: ClassVar[str]
    _exposes_agg_func: ClassVar[bool] = False
    _measurement_infoclass: ClassVar[type | None] = None

    severity_warn: ClassVar[float] = 0.05
    severity_fail: ClassVar[float] = 0.10
    unmatched_groups: list = Field(default_factory=list)

    @abc.abstractmethod
    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Add the check's metric columns to one group.

        Must add at minimum the severity column (``QC_<name>_Severity``).
        May add check-specific columns documented by
        ``_measurement_infoclass``. ``Flag`` and ``Status`` are computed
        by the base class from severity, so subclasses must not set them
        directly.

        Args:
            group: A single group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (typically a copy) augmented with the
            severity column and any check-specific metric columns.
        """

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the check on every group and return the augmented frame.

        Iterates over ``data.groupby(self.groupby, dropna=False)``,
        delegates per-group computation to :meth:`_compute`, and adds
        three generic columns derived from severity:

        * ``QC_<name>_Severity`` (carry-through from ``_compute``)
        * ``QC_<name>_Flag`` (``bool``)
        * ``QC_<name>_Status`` (``"pass"`` / ``"warn"`` / ``"fail"``)

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

        severity_col = self.severity_col()
        flag_col = self.flag_col()
        status_col = self.status_col()

        pieces: list[pd.DataFrame] = []
        for _, group in data.groupby(self.groupby, dropna=False):
            pieces.append(self._compute(group))

        if pieces:
            result = pd.concat(pieces, axis=0)
        else:
            result = data.iloc[0:0].copy()
            result[severity_col] = pd.Series(dtype=float)

        severity = pd.to_numeric(result[severity_col], errors="coerce")
        flag = severity.ge(self.severity_fail).fillna(False).astype(bool)
        status = pd.Series("pass", index=result.index, dtype=object)
        status = status.mask(severity.ge(self.severity_warn), "warn")
        status = status.mask(severity.ge(self.severity_fail), "fail")
        status = status.where(severity.notna(), "pass")

        result[severity_col] = severity
        result[flag_col] = flag
        result[status_col] = status

        self._latest_measurements = result
        return result

    def summary(self) -> pd.DataFrame:
        """Return a one-row-per-group summary of the most recent analyze.

        Returns:
            DataFrame with columns ``[*self.groupby, "num_rows",
            "num_flagged", "max_severity", "status"]``. The ``status``
            column is the worst status across the group: ``"fail"``
            wins over ``"warn"`` which wins over ``"pass"``.
        """
        rank = {"pass": 0, "warn": 1, "fail": 2}
        inv_rank = {v: k for k, v in rank.items()}

        df = self._latest_measurements
        severity_col = self.severity_col()
        flag_col = self.flag_col()
        status_col = self.status_col()

        def _summarize(group: pd.DataFrame) -> pd.Series:
            worst = int(group[status_col].map(rank).max())
            return pd.Series({
                "num_rows": int(len(group)),
                "num_flagged": int(group[flag_col].sum()),
                "max_severity": float(group[severity_col].max()),
                "status": inv_rank[worst],
            })

        grouped = df.groupby(self.groupby, dropna=False)
        summary = grouped.apply(_summarize, include_groups=False).reset_index()
        return summary

    def flagged_keys(self) -> list[tuple[str, int]]:
        """Return (``Metadata_ImageFile``, ``ObjectLabel``) pairs to curate.

        Used by the GUI "Mark all flagged for removal" button. Requires
        the analyzed frame to carry both ``Metadata_ImageFile`` and
        ``ObjectLabel`` columns (the curation key used by
        ``STORE_REMOVED_KEYS``). Returns an empty list when those
        columns are absent or when no rows were flagged.

        Returns:
            De-duplicated list of ``(image_file, object_label)`` tuples
            for rows where ``Flag=True``.
        """
        df = self._latest_measurements
        flag_col = self.flag_col()
        if flag_col not in df.columns:
            return []
        if "Metadata_ImageFile" not in df.columns or "ObjectLabel" not in df.columns:
            return []
        flagged = df.loc[df[flag_col].fillna(False).astype(bool),
                         ["Metadata_ImageFile", "ObjectLabel"]].dropna()
        if flagged.empty:
            return []
        flagged = flagged.drop_duplicates()
        return [
            (str(row.Metadata_ImageFile), int(row.ObjectLabel))
            for row in flagged.itertuples(index=False)
        ]

    @classmethod
    def severity_col(cls) -> str:
        """Return the severity column name for this check."""
        return f"QC_{cls.name}_Severity"

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
