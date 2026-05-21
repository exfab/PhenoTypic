"""Expected-vs-detected colony count quality check.

Compares the detected colony count per group in a measurement frame against
the expected count derived from a separately-provided metadata frame
(usually the plate's layout CSV). Surfaces groups where colonies are
missing or over-detected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Callable, ClassVar

import pandas as pd
import plotly.graph_objects as go
from pydantic import PrivateAttr, WithJsonSchema, field_validator

from phenotypic.analysis.abc_._quality_check import QualityCheck
from phenotypic.tools_ import ColumnRef
from phenotypic.tools_.measurement_info import QUALITY_COUNT

# The metadata layout frame is an ``arbitrary_types_allowed`` field: a
# raw ``pandas.DataFrame`` has no JSON schema, so attach an object-typed
# placeholder so ``model_json_schema()`` succeeds. The frame is supplied
# (and resolved from a CSV/Parquet path) at construction time and is not
# part of the JSON-serializable parameter surface.
_MetadataFrame = Annotated[
    pd.DataFrame,
    WithJsonSchema({"type": "object"}),
]


class ExpectedVsDetectedCount(QualityCheck):
    """Flag groups whose detected colony count diverges from metadata.

    For each ``groupby`` combination the check compares the number of rows
    in the measurement frame (``detected``) against the number of rows in
    the externally-provided ``metadata`` frame for the same key
    (``expected``). The signed difference and its normalized magnitude
    drive a tri-state pass/warn/fail label:

    * ``QC_Count_Severity = |detected - expected| / expected``
    * ``QC_Count_Severity = numpy.inf`` when ``expected == 0`` (i.e. the
      measurement group has no metadata counterpart). This always exceeds
      ``severity_fail`` so the status becomes ``"fail"`` and the rows are
      flagged. The offending key tuple is recorded in
      :attr:`unmatched_groups` so the GUI can distinguish a real biology
      fail from a metadata-mismatch fail.

    The check does **not** aggregate measurement values — it counts rows
    — so :attr:`_exposes_agg_func` is ``False`` and the GUI
    parameter-form rendering driver hides the ``agg_func`` field. The
    base ``SetAnalyzer.agg_func`` is pinned to ``"first"`` internally.

    The ``metadata`` argument can be either a ready-made
    :class:`pandas.DataFrame` or a path (``Path`` or ``str``) to a
    ``.csv``/``.parquet`` file. The file is read once at construction
    time and the resolved frame is stored on the instance. Every column
    named in ``groupby`` must be present in the metadata frame;
    otherwise :class:`KeyError` is raised at ``__init__`` so the failure
    surfaces before ``analyze`` runs.

    Args:
        metadata: Layout frame whose row count per ``groupby`` key is the
            expected colony count. Either a DataFrame or a path to a CSV
            or Parquet file.
        groupby: Columns that define a comparison unit. Must be present
            in both the metadata frame and the measurement frame passed
            to :meth:`analyze`.
        on: Measurement column the check operates on. Defaults to
            ``"ObjectLabel"`` since "detected" means "a measurement row
            exists".
        severity_warn: Per-instance override for ``severity_warn``.
            ``None`` falls back to the class default (``0.05``).
        severity_fail: Per-instance override for ``severity_fail``.
            ``None`` falls back to the class default (``0.10``).
        n_jobs: Worker count. Currently unused by the base ``analyze``
            loop; kept on the signature for parity with
            :class:`SetAnalyzer`.

    Raises:
        FileNotFoundError: If ``metadata`` is a path that does not exist.
        KeyError: If any column in ``groupby`` is absent from the
            resolved metadata frame.
        ValueError: If ``metadata`` is a path with an unsupported suffix.

    Attributes:
        unmatched_groups: List of group-key tuples that appeared in the
            measurement frame but had no counterpart in the metadata
            frame during the most recent :meth:`analyze` call. Reset at
            the top of each ``analyze`` so re-runs do not accumulate.

    Examples:
        Basic match — 96-well metadata vs. a measurement frame missing
        one well:

        >>> import pandas as pd
        >>> from phenotypic.analysis._expected_vs_detected import (
        ...     ExpectedVsDetectedCount,
        ... )
        >>> metadata = pd.DataFrame({
        ...     "Metadata_ImageFile": ["plate1.png"] * 96,
        ...     "ObjectLabel": list(range(96)),
        ... })
        >>> measurements = pd.DataFrame({
        ...     "Metadata_ImageFile": ["plate1.png"] * 95,
        ...     "ObjectLabel": list(range(95)),
        ... })
        >>> chk = ExpectedVsDetectedCount(
        ...     metadata=metadata,
        ...     groupby=["Metadata_ImageFile"],
        ... )
        >>> result = chk.analyze(measurements)  # doctest: +SKIP
        >>> "QC_Count_Severity" in result.columns  # doctest: +SKIP
        True

        Advanced — a measurement group has no metadata counterpart, so
        severity is infinite and the key is recorded:

        >>> metadata = pd.DataFrame({
        ...     "Metadata_ImageFile": ["plate1.png"] * 96,
        ...     "ObjectLabel": list(range(96)),
        ... })
        >>> measurements = pd.DataFrame({
        ...     "Metadata_ImageFile": ["plate2.png"] * 10,
        ...     "ObjectLabel": list(range(10)),
        ... })
        >>> chk = ExpectedVsDetectedCount(
        ...     metadata=metadata,
        ...     groupby=["Metadata_ImageFile"],
        ... )
        >>> _ = chk.analyze(measurements)  # doctest: +SKIP
        >>> chk.unmatched_groups  # doctest: +SKIP
        [('plate2.png',)]
    """

    name: ClassVar[str] = "Count"
    severity_warn: ClassVar[float] = 0.05
    severity_fail: ClassVar[float] = 0.10
    _exposes_agg_func: ClassVar[bool] = False
    _measurement_infoclass = QUALITY_COUNT

    on: ColumnRef = "ObjectLabel"
    agg_func: Callable | str | list | dict | None = "first"
    metadata: _MetadataFrame

    _metadata: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)
    _expected_counts: pd.Series = PrivateAttr(default_factory=pd.Series)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(
        cls, value: pd.DataFrame | Path | str
    ) -> pd.DataFrame:
        """Resolve a DataFrame-or-path ``metadata`` argument to a frame.

        Args:
            value: Either an in-memory DataFrame or a path (``Path`` or
                ``str``) to a ``.csv``/``.parquet`` file.

        Returns:
            The resolved DataFrame.

        Raises:
            FileNotFoundError: If ``value`` is a path that does not exist.
            ValueError: If the path has an unsupported suffix.
        """
        return cls._resolve_metadata(value)

    def model_post_init(self, __context: Any) -> None:
        """Validate metadata columns and pre-compute expected counts.

        Runs after pydantic has validated every field. Mirrors the
        resolved ``metadata`` frame onto the private ``_metadata`` slot,
        verifies every ``groupby`` column is present, and caches the
        per-key expected colony counts.

        Args:
            __context: Pydantic post-init context (unused).

        Raises:
            KeyError: If any column in ``groupby`` is absent from the
                resolved metadata frame.
        """
        missing = [
            col for col in self.groupby if col not in self.metadata.columns
        ]
        if missing:
            raise KeyError(
                "metadata frame is missing required groupby column(s): "
                f"{missing}"
            )
        self._metadata = self.metadata
        self._expected_counts = self.metadata.groupby(
            self.groupby, dropna=False
        ).size()

    @staticmethod
    def _resolve_metadata(
        metadata: pd.DataFrame | Path | str,
    ) -> pd.DataFrame:
        """Coerce a DataFrame-or-path metadata argument into a DataFrame.

        Args:
            metadata: Either an in-memory DataFrame or a path (``Path``
                or ``str``) to a ``.csv``/``.parquet`` file.

        Returns:
            The resolved DataFrame. If ``metadata`` is already a
            DataFrame it is returned as-is (no copy).

        Raises:
            FileNotFoundError: If ``metadata`` is a path that does not
                exist.
            ValueError: If the path has an unsupported suffix.
        """
        if isinstance(metadata, pd.DataFrame):
            return metadata

        path = Path(metadata)
        if not path.exists():
            raise FileNotFoundError(
                f"metadata path does not exist: {path}"
            )

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(
            "metadata path must be a .csv or .parquet file; got "
            f"suffix {suffix!r}"
        )

    def _lookup_expected(self, key: Any) -> int:
        """Return the expected row count for one ``groupby`` key.

        Args:
            key: The group key as produced by
                ``DataFrame.groupby(groupby).__iter__`` — a scalar when
                ``groupby`` is a single column, otherwise a tuple.

        Returns:
            The expected row count, or ``0`` if the key is not present
            in the metadata's index.
        """
        try:
            value = self._expected_counts.loc[key]
        except KeyError:
            return 0
        if isinstance(value, pd.Series):
            return int(value.sum())
        return int(value)

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute count-divergence metrics for one group.

        Looks up the group's expected count, broadcasts the detected /
        expected / delta / severity scalars across every row, and
        records the key tuple in :attr:`unmatched_groups` when no
        metadata counterpart was found.

        Args:
            group: One group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (a copy) with four new columns appended:
            ``QC_Count_Detected``, ``QC_Count_Expected``,
            ``QC_Count_Delta``, ``QC_Count_Severity``.
        """
        detected = int(len(group))
        key = self._group_key(group)
        expected = self._lookup_expected(key)

        if expected == 0:
            self.unmatched_groups.append(key)
            severity = float("inf")
        else:
            severity = abs(detected - expected) / expected

        delta = detected - expected

        out = group.copy()
        out[str(QUALITY_COUNT.DETECTED)] = detected
        out[str(QUALITY_COUNT.EXPECTED)] = expected
        out[str(QUALITY_COUNT.DELTA)] = delta
        out[self.severity_col()] = float(severity)
        return out

    def _group_key(self, group: pd.DataFrame) -> tuple:
        """Extract the ``groupby`` key for a single group as a tuple.

        Args:
            group: One group frame. The values in ``self.groupby``
                columns are constant within the group, so the first row
                suffices.

        Returns:
            A tuple of the group's ``groupby`` values, regardless of
            whether ``groupby`` has one or many columns. Tuples are used
            uniformly so the per-key index lookup is independent of
            ``groupby`` arity.
        """
        row = group.iloc[0]
        return tuple(row[col] for col in self.groupby)

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reset :attr:`unmatched_groups` and run the base ``analyze``.

        Re-running the check on a different measurement frame must not
        carry over unmatched groups from a previous run, so the list is
        cleared before delegating to the base class.

        Args:
            data: Measurement frame to evaluate.

        Returns:
            The augmented frame from :meth:`QualityCheck.analyze`.
        """
        self.unmatched_groups = []
        return super().analyze(data)

    def dash(self, **kwargs: Any) -> go.Figure:
        """Render a horizontal lollipop chart of ``Delta`` per group.

        Each group's signed ``Delta`` is drawn as a horizontal stem from
        zero to ``Delta``, with a marker at the tip colored by
        ``Status``. The hover label exposes detected, expected, and
        severity for the group.

        Args:
            **kwargs: Passed through to :func:`plotly.graph_objects.Figure`
                / ``Figure.update_layout`` — accepted keys are ``title``
                and ``height``.

        Returns:
            A :class:`plotly.graph_objects.Figure` with one stem trace
            and one marker trace.

        Raises:
            RuntimeError: If :meth:`analyze` has not been called yet.
        """
        df = self._latest_measurements
        if df.empty:
            raise RuntimeError("call analyze() first")

        severity_col = self.severity_col()
        status_col = self.status_col()
        delta_col = str(QUALITY_COUNT.DELTA)
        detected_col = str(QUALITY_COUNT.DETECTED)
        expected_col = str(QUALITY_COUNT.EXPECTED)

        per_group = (
            df.groupby(self.groupby, dropna=False)
            .agg({
                delta_col: "first",
                detected_col: "first",
                expected_col: "first",
                severity_col: "first",
                status_col: "first",
            })
            .reset_index()
        )

        labels = per_group[self.groupby].astype(str).agg(" | ".join, axis=1)
        deltas = per_group[delta_col].astype(float)
        statuses = per_group[status_col].astype(str)
        status_colors = {
            "pass": "#2E86AB",
            "warn": "#F4A261",
            "fail": "#E63946",
        }
        marker_colors = statuses.map(status_colors).fillna("#888888")

        hover = [
            (
                f"Detected: {int(d)}<br>"
                f"Expected: {int(e)}<br>"
                f"Delta: {int(dl)}<br>"
                f"Severity: {sv:.4f}<br>"
                f"Status: {st}"
            )
            for d, e, dl, sv, st in zip(
                per_group[detected_col],
                per_group[expected_col],
                deltas,
                per_group[severity_col].astype(float),
                statuses,
            )
        ]

        fig = go.Figure()
        for label, delta in zip(labels, deltas):
            fig.add_trace(
                go.Scatter(
                    x=[0, delta],
                    y=[label, label],
                    mode="lines",
                    line={"color": "#888888", "width": 2},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=deltas,
                y=labels,
                mode="markers",
                marker={"color": marker_colors.tolist(), "size": 12},
                text=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
        fig.add_vline(x=0, line={"color": "black", "width": 1})

        fig.update_layout(
            title=kwargs.get(
                "title", "Expected vs. Detected Colony Count"
            ),
            xaxis_title="Detected − Expected",
            yaxis_title=" | ".join(self.groupby),
            height=kwargs.get("height", max(240, 24 * len(labels) + 80)),
        )
        return fig
