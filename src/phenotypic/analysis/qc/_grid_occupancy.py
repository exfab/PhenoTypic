"""Grid-occupancy quality check.

A metadata-driven variant of :class:`ExpectedVsDetectedCount`. Instead of
comparing the raw detected row count against the expected count, it compares
the number of **distinct filled grid cells** (doublets collapse to one)
against the expected cell count from the user-provided layout frame, and
flags plates whose occupancy falls below threshold.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import plotly.graph_objects as go

from phenotypic.analysis.qc._expected_vs_detected import ExpectedVsDetectedCount
from phenotypic.schema import GRID, QUALITY_OCCUPANCY
from phenotypic.sdk_ import ColumnRef


class GridOccupancy(ExpectedVsDetectedCount):
    """Flag groups whose grid occupancy (filled cells / expected) is low.

    Inherits the entire metadata-form surface of
    :class:`ExpectedVsDetectedCount` — the single ``metadata`` field
    (an in-memory DataFrame *or* a ``.csv``/``.parquet`` path), its
    store-verbatim coercion, the ``pipeline.json`` serialization round-trip
    (the *source path* persists under the ``metadata`` key and the frame
    re-reads on load), and the per-key expected-count precompute. The
    expected cell count for a group is the number of metadata rows for that
    ``groupby`` key (one row per expected pin position).

    Where the parent counts ``len(group)`` (raw detections, doublets
    included), this check counts ``group[cell_label].nunique()`` distinct
    occupied cells, so a doublet (two colonies sharing one grid cell) still
    counts once. The two columns play distinct roles:

    * ``on`` (inherited default ``"Object_Label"``) is the base-class
      required/guard column and the curation member value — it is unique
      per colony, so it is **not** what occupancy counts over.
    * ``cell_label`` (default ``"Grid_RowMajorIdx"``) is the grid-cell id
      the occupancy ``nunique`` collapses doublets over. Counting distinct
      *labels* would count colonies, not cells; counting distinct *cells*
      is what makes the metric doublet-insensitive.

    The metric is ``filled / expected``. ``_HIGHER_IS_BAD`` is ``False``: a
    *lower* occupancy is worse, so a row fails when
    ``metric <= fail_threshold`` and warns at ``metric <= warn_threshold``
    (hence ``warn_threshold >= fail_threshold``).

    A group present in the measurements but absent from the metadata frame
    (``expected == 0``) is recorded in :attr:`unmatched_groups` and given
    ``metric = 0.0`` so it fails — mirroring the parent's "force a flag on a
    metadata mismatch" behavior, adapted to the lower-is-bad direction. The
    ``QC_Occupancy_Expected = 0`` column distinguishes such a mismatch from
    a genuinely empty plate.

    Args:
        metadata: Layout (in-memory DataFrame or ``.csv``/``.parquet``
            path) whose row count per ``groupby`` key is the expected cell
            count. Same semantics, coercion, and serialization as
            :class:`ExpectedVsDetectedCount` (the path form round-trips
            through JSON).
        groupby: Columns that define one plate. Usually
            ``["Metadata_ImageFile"]``. Must be present in both the
            metadata frame and the measurement frame.
        on: Base-class required column and curation member value. Defaults
            to ``"Object_Label"``; occupancy does not count over it.
        cell_label: Grid-cell id column whose distinct count is the filled
            cell count. Defaults to ``"Grid_RowMajorIdx"``. Must be present
            in the measurement frame passed to :meth:`analyze`.
        warn_threshold: Occupancy at/below which ``Status`` becomes
            ``"warn"``. Defaults to ``0.95``.
        fail_threshold: Occupancy at/below which ``Status`` becomes
            ``"fail"`` and ``Flag=True``. Defaults to ``0.90``.

    Raises:
        KeyError: If ``cell_label`` is absent from the measurement frame, or
            (inherited) if any ``groupby`` column is absent from the
            metadata frame.
        FileNotFoundError: (inherited) If ``metadata`` is a path that does
            not exist.
        ValueError: (inherited) If ``metadata`` is a path with an
            unsupported suffix, or if it is ``None`` (a check serialized
            from an in-memory frame, which cannot round-trip).

    Examples:
        Basic — 96-cell metadata vs. a measurement frame with 92 colonies
        but only 90 distinct filled cells (two doublets). Occupancy reads
        the filled-cell count, not the colony count:

        >>> import pandas as pd
        >>> from phenotypic.analysis.qc import GridOccupancy
        >>> metadata = pd.DataFrame({
        ...     "Metadata_ImageFile": ["p1.png"] * 96,
        ...     "Object_Label": list(range(96)),
        ... })
        >>> measurements = pd.DataFrame({
        ...     "Metadata_ImageFile": ["p1.png"] * 92,
        ...     "Object_Label": list(range(92)),
        ...     "Grid_RowMajorIdx": list(range(90)) + [5, 17],
        ... })
        >>> chk = GridOccupancy(
        ...     metadata=metadata, groupby=["Metadata_ImageFile"]
        ... )
        >>> out = chk.analyze(measurements)
        >>> int(out["QC_Occupancy_Filled"].iloc[0])
        90
        >>> round(float(out["QC_Occupancy_Metric"].iloc[0]), 4)
        0.9375
    """

    name: ClassVar[str] = "Occupancy"
    _HIGHER_IS_BAD: ClassVar[bool] = False
    _exposes_agg_func: ClassVar[bool] = False
    _measurement_infoclass: ClassVar[type | None] = QUALITY_OCCUPANCY
    supports_object_curation: ClassVar[bool] = False

    warn_threshold: float = 0.95
    fail_threshold: float = 0.90
    cell_label: ColumnRef = str(GRID.ROW_MAJOR_IDX)

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Guard the cell-id column, then run the inherited ``analyze``.

        The base ``QualityCheck.analyze`` only guards ``self.on`` and
        ``self.groupby``; occupancy additionally needs ``cell_label`` so it
        can collapse doublets, so its absence is surfaced here before the
        per-group loop runs.

        Args:
            data: Measurement frame to evaluate.

        Returns:
            The augmented frame from the inherited ``analyze`` (which also
            resets :attr:`unmatched_groups`).

        Raises:
            KeyError: If ``cell_label`` is missing from ``data``.
        """
        if self.cell_label not in data.columns:
            raise KeyError(
                "GridOccupancy requires the cell-id column "
                f"{self.cell_label!r} in the measurement frame"
            )
        return super().analyze(data)

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute occupancy for one group and broadcast across its rows.

        Counts distinct ``cell_label`` values (doublet-insensitive), looks
        up the group's expected cell count, and records the key tuple in
        :attr:`unmatched_groups` when no metadata counterpart was found.

        Args:
            group: One group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (a copy) with four new columns appended:
            ``QC_Occupancy_Filled``, ``QC_Occupancy_Expected``,
            ``QC_Occupancy_Vacant``, ``QC_Occupancy_Metric``.
        """
        filled = int(group[self.cell_label].nunique())
        key = self._group_key(group)
        expected = self._lookup_expected(key)

        if expected == 0:
            self.unmatched_groups.append(key)
            metric = 0.0  # lower-is-bad → forces a fail on metadata mismatch
        else:
            metric = filled / expected

        out = group.copy()
        out[str(QUALITY_OCCUPANCY.FILLED)] = filled
        out[str(QUALITY_OCCUPANCY.EXPECTED)] = expected
        out[str(QUALITY_OCCUPANCY.VACANT)] = max(expected - filled, 0)
        out[self.metric_col()] = float(metric)
        return out

    def to_table(self) -> pd.DataFrame:
        """Return one group-level row per group (occupancy is per-plate).

        Occupancy reports filled-vs-expected counts broadcast across a
        group's rows, so per-colony rows carry no extra signal. Collapse to
        one row per group: the base ``summary()`` (renamed to the generic
        QC columns) plus the occupancy-specific counts.

        Returns:
            A group-level frame: ``[*groupby, QC_Occupancy_Filled,
            QC_Occupancy_Expected, QC_Occupancy_Vacant, QC_Occupancy_Metric,
            QC_Occupancy_Status, QC_Occupancy_Flag]``.
        """
        df = self._latest_measurements
        occ_cols = [
            str(QUALITY_OCCUPANCY.FILLED),
            str(QUALITY_OCCUPANCY.EXPECTED),
            str(QUALITY_OCCUPANCY.VACANT),
        ]
        first = (
            df.groupby(self.groupby, dropna=False)[
                [c for c in occ_cols if c in df.columns]
            ]
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
        # Group-level flag: any member flagged → fail-status drives the flag.
        merged[self.flag_col()] = merged[self.status_col()] == "fail"
        return merged

    def dash(self, **kwargs: Any) -> go.Figure:
        """Render a horizontal bar of per-group occupancy, colored by status.

        Args:
            **kwargs: Passed through to ``Figure.update_layout`` — accepted
                keys are ``title`` and ``height``.

        Returns:
            A :class:`plotly.graph_objects.Figure` with one bar trace and a
            dashed reference line at ``fail_threshold``.

        Raises:
            RuntimeError: If :meth:`analyze` has not been called yet.
        """
        df = self._latest_measurements
        if df.empty:
            raise RuntimeError("call analyze() first")

        metric_col = self.metric_col()
        status_col = self.status_col()
        per = (
            df.groupby(self.groupby, dropna=False)
            .agg({metric_col: "first", status_col: "first"})
            .reset_index()
        )
        labels = per[self.groupby].astype(str).agg(" | ".join, axis=1)
        status_colors = {
            "pass": "#2E86AB",
            "warn": "#F4A261",
            "fail": "#E63946",
        }
        colors = per[status_col].map(status_colors).fillna("#888888")

        fig = go.Figure(
            go.Bar(
                x=per[metric_col].astype(float),
                y=labels,
                orientation="h",
                marker={"color": colors.tolist()},
            )
        )
        fig.add_vline(
            x=self.fail_threshold, line={"color": "#E63946", "dash": "dash"}
        )
        fig.update_layout(
            title=kwargs.get(
                "title", "Grid Occupancy (filled / expected cells)"
            ),
            xaxis_title="Occupancy",
            xaxis={"range": [0, 1]},
            yaxis_title=" | ".join(self.groupby),
            height=kwargs.get("height", max(240, 24 * len(labels) + 80)),
        )
        return fig
