"""Expected-vs-detected colony count quality check.

Compares the detected colony count per group in a measurement frame against
the expected count derived from a separately-provided metadata frame
(usually the plate's layout CSV). Surfaces groups where colonies are
missing or over-detected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Callable, ClassVar, Optional

import pandas as pd
import plotly.graph_objects as go
from pydantic import (
    Field,
    PrivateAttr,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from phenotypic.analysis.abc_._quality_check import QualityCheck
from phenotypic.sdk_ import ColumnRef
from phenotypic.schema import OBJECT, QUALITY_COUNT

# The metadata layout frame is an ``arbitrary_types_allowed`` field: a
# raw ``pandas.DataFrame`` has no JSON schema, so attach an object-typed
# placeholder so ``model_json_schema()`` succeeds. The frame is excluded
# from ``model_dump`` (``Field(exclude=True)``) — a DataFrame is not
# JSON-native — and the serializable surface carries only the
# ``metadata_source`` path string, so ``pipeline.json`` round-trips the
# layout *source* and re-reads the frame on load.
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

    * ``QC_Count_Metric = |detected - expected| / expected``
    * ``QC_Count_Metric = numpy.inf`` when ``expected == 0`` (i.e. the
      measurement group has no metadata counterpart). This always exceeds
      ``fail_threshold`` so the status becomes ``"fail"`` and the rows are
      flagged. The offending key tuple is recorded in
      :attr:`unmatched_groups` so the GUI can distinguish a real biology
      fail from a metadata-mismatch fail.

    ``_HIGHER_IS_BAD`` is ``True``: a larger normalized count divergence
    is worse, so the base class flags rows whose metric meets or exceeds
    ``fail_threshold`` (including the infinite metric of an unmatched
    group).

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

    **Serialization:** the resolved frame is *not* part of the
    JSON-serializable parameter surface (a DataFrame is not JSON-native).
    When ``metadata`` is supplied as a path, that path string is captured
    in the serializable :attr:`metadata_source` field, so
    ``model_dump`` / ``pipeline.json`` round-trip the layout *source* and
    a reloaded instance re-reads the file. When ``metadata`` is supplied
    as an in-memory DataFrame there is no source path to persist —
    :attr:`metadata_source` stays ``None`` and the check cannot be
    rebuilt from JSON alone (it will fail to instantiate with a clear
    error, surfaced as a skip-with-warning by the lazy QC instantiation
    path). Configure QC checks from a metadata *path* whenever the
    pipeline is meant to round-trip.

    Args:
        metadata: Layout frame whose row count per ``groupby`` key is the
            expected colony count. Either a DataFrame or a path to a CSV
            or Parquet file. Excluded from serialization — supply
            ``metadata_source`` instead when rebuilding from JSON.
        metadata_source: Path to the layout CSV/Parquet, captured
            automatically when ``metadata`` is given as a path. This is
            the JSON-serializable handle to the layout: on
            reconstruction from ``pipeline.json`` the frame is re-read
            from here. Usually set implicitly; pass it explicitly only
            when reconstructing without a ``metadata`` frame.
        groupby: Columns that define a comparison unit. Must be present
            in both the metadata frame and the measurement frame passed
            to :meth:`analyze`.
        on: Measurement column the check operates on. Defaults to
            ``"Object_Label"`` since "detected" means "a measurement row
            exists".
        warn_threshold: Normalized count divergence at which ``Status``
            becomes ``"warn"``. Defaults to ``0.05``.
        fail_threshold: Normalized count divergence at which ``Status``
            becomes ``"fail"`` and ``Flag=True``. Defaults to ``0.10``.
        n_jobs: Worker count. Currently unused by the base ``analyze``
            loop; kept on the signature for parity with
            :class:`SetAnalyzer`.

    Raises:
        FileNotFoundError: If ``metadata`` (or ``metadata_source``) is a
            path that does not exist.
        KeyError: If any column in ``groupby`` is absent from the
            resolved metadata frame.
        ValueError: If ``metadata`` is a path with an unsupported suffix,
            or if neither ``metadata`` nor ``metadata_source`` is
            supplied (e.g. reconstructing from JSON that was built from
            an in-memory frame, which has no source path to persist).

    Attributes:
        unmatched_groups: List of group-key tuples that appeared in the
            measurement frame but had no counterpart in the metadata
            frame during the most recent :meth:`analyze` call. Reset at
            the top of each ``analyze`` so re-runs do not accumulate.

    Examples:
        Basic match — 96-well metadata vs. a measurement frame missing
        one well:

        >>> import pandas as pd
        >>> from phenotypic.analysis.qc import (
        ...     ExpectedVsDetectedCount,
        ... )
        >>> metadata = pd.DataFrame({
        ...     "Metadata_ImageFile": ["plate1.png"] * 96,
        ...     "Object_Label": list(range(96)),
        ... })
        >>> measurements = pd.DataFrame({
        ...     "Metadata_ImageFile": ["plate1.png"] * 95,
        ...     "Object_Label": list(range(95)),
        ... })
        >>> chk = ExpectedVsDetectedCount(
        ...     metadata=metadata,
        ...     groupby=["Metadata_ImageFile"],
        ... )
        >>> result = chk.analyze(measurements)  # doctest: +SKIP
        >>> "QC_Count_Metric" in result.columns  # doctest: +SKIP
        True

        Advanced — a measurement group has no metadata counterpart, so
        the metric is infinite and the key is recorded:

        >>> metadata = pd.DataFrame({
        ...     "Metadata_ImageFile": ["plate1.png"] * 96,
        ...     "Object_Label": list(range(96)),
        ... })
        >>> measurements = pd.DataFrame({
        ...     "Metadata_ImageFile": ["plate2.png"] * 10,
        ...     "Object_Label": list(range(10)),
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
    _HIGHER_IS_BAD: ClassVar[bool] = True
    _exposes_agg_func: ClassVar[bool] = False
    # Annotate with the base ClassVar type (not a bare assignment) so
    # subclasses may override it with their own MeasurementInfo enum.
    _measurement_infoclass: ClassVar[type | None] = QUALITY_COUNT

    warn_threshold: float = 0.05
    fail_threshold: float = 0.10
    on: ColumnRef = str(OBJECT.LABEL)
    agg_func: Callable | str | list | dict | None = "first"
    # ``metadata`` carries the resolved frame at runtime but is excluded
    # from ``model_dump`` (a DataFrame is not JSON-native). The
    # serializable handle is ``metadata_source``.
    metadata: _MetadataFrame = Field(exclude=True)
    metadata_source: Optional[str] = None

    _metadata: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)
    _expected_counts: pd.Series = PrivateAttr(default_factory=pd.Series)

    @model_validator(mode="before")
    @classmethod
    def _capture_metadata_source(cls, data: Any) -> Any:
        """Capture a metadata *path* into ``metadata_source`` and resolve it.

        Runs before field validation on the raw input mapping. Two
        construction shapes feed this check:

        * **Direct / GUI construction** — the caller passes
          ``metadata=<path-or-frame>``. When it is a path, the path
          string is recorded in ``metadata_source`` (unless the caller
          already supplied one) so the value survives a later
          ``model_dump`` even though the resolved ``metadata`` frame does
          not.
        * **Reconstruction from JSON** — ``model_dump`` excluded the
          frame, so the input has ``metadata_source`` but no
          ``metadata``. Here the frame is resolved *from*
          ``metadata_source`` and injected as ``metadata`` so the
          downstream field validator and ``model_post_init`` see a real
          frame.

        Args:
            data: The raw input. Normally the constructor kwargs mapping;
                pydantic may also hand this validator a non-dict (e.g. an
                already-built model on revalidation), which passes
                through untouched.

        Returns:
            The input mapping with ``metadata`` resolved to a frame and
            ``metadata_source`` populated when a source path is known.

        Raises:
            ValueError: If neither ``metadata`` nor ``metadata_source``
                is provided.
        """
        if not isinstance(data, dict):
            return data

        raw_metadata = data.get("metadata")
        raw_source = data.get("metadata_source")

        if raw_metadata is None and raw_source is None:
            raise ValueError(
                "ExpectedVsDetectedCount requires either 'metadata' "
                "(a DataFrame or path) or 'metadata_source' (a path). "
                "Both are missing — a check serialized from an in-memory "
                "DataFrame cannot be rebuilt from JSON; configure it from "
                "a metadata CSV/Parquet path so the source round-trips."
            )

        # Reconstruction path: only the source survived serialization.
        if raw_metadata is None and raw_source is not None:
            data = dict(data)
            data["metadata"] = raw_source
            return data

        # Direct construction: capture the path (if it is one) so a later
        # dump preserves it. An in-memory frame has no path to record.
        if raw_source is None and isinstance(raw_metadata, (str, Path)):
            data = dict(data)
            data["metadata_source"] = str(raw_metadata)

        return data

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
        super().model_post_init(__context)
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
        expected / delta / metric scalars across every row, and
        records the key tuple in :attr:`unmatched_groups` when no
        metadata counterpart was found.

        Args:
            group: One group as produced by
                ``data.groupby(self.groupby, dropna=False)``.

        Returns:
            The group frame (a copy) with four new columns appended:
            ``QC_Count_Detected``, ``QC_Count_Expected``,
            ``QC_Count_Delta``, ``QC_Count_Metric``.
        """
        detected = int(len(group))
        key = self._group_key(group)
        expected = self._lookup_expected(key)

        if expected == 0:
            self.unmatched_groups.append(key)
            metric = float("inf")
        else:
            metric = abs(detected - expected) / expected

        delta = detected - expected

        out = group.copy()
        out[str(QUALITY_COUNT.DETECTED)] = detected
        out[str(QUALITY_COUNT.EXPECTED)] = expected
        out[str(QUALITY_COUNT.DELTA)] = delta
        out[self.metric_col()] = float(metric)
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
        ``Status``. The hover label exposes detected, expected, and the
        metric for the group.

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

        metric_col = self.metric_col()
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
                metric_col: "first",
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
                f"Metric: {sv:.4f}<br>"
                f"Status: {st}"
            )
            for d, e, dl, sv, st in zip(
                per_group[detected_col],
                per_group[expected_col],
                deltas,
                per_group[metric_col].astype(float),
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
