"""Expected-vs-detected colony count quality check.

Compares the detected colony count per group in a measurement frame against
the expected count derived from a separately-provided metadata frame
(usually the plate's layout CSV). Surfaces groups where colonies are
missing or over-detected.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Annotated, Any, Callable, ClassVar

import pandas as pd
import plotly.graph_objects as go
from pydantic import (
    PrivateAttr,
    WithJsonSchema,
    field_serializer,
    field_validator,
)

from phenotypic.analysis.abc_._quality_check import QualityCheck
from phenotypic.abc_.plotting import PlotQc
from phenotypic.sdk_ import (
    ColumnRef,
    ensure_metadata_prefix,
    is_metadata_header,
    metadata_member_for_header,
    metadata_member_for_label,
    normalize_metadata_columns,
)
from phenotypic.schema import MeasurementInfo, MetadataInfo, OBJECT, QUALITY_COUNT
import phenotypic.schema as schema

# ``metadata`` is a single, unified field that accepts **either** an
# in-memory layout :class:`pandas.DataFrame` (an "array") **or** a path
# string to a ``.csv``/``.parquet`` layout file. The value is stored
# verbatim — ``self.metadata`` echoes exactly what the caller passed —
# while the *resolved* frame is precomputed onto the private ``_metadata``
# slot.
#
# Only a path string is JSON-native, so that is the form that round-trips
# through ``model_dump`` / ``pipeline.json``: ``_serialize_metadata``
# emits the path when ``metadata`` is a string and ``None`` when it is an
# in-memory frame (which therefore cannot be rebuilt from JSON — configure
# the check from a path whenever it must round-trip). A raw
# ``pandas.DataFrame`` has no JSON schema, so the object branch attaches a
# placeholder so ``model_json_schema()`` succeeds.
_MetadataField = Annotated[
    pd.DataFrame | str,
    WithJsonSchema({
        "oneOf": [
            {
                "type": "string",
                "description": (
                    "Path to a .csv/.parquet layout file (the form that "
                    "round-trips through JSON)."
                ),
            },
            {
                "type": "object",
                "description": (
                    "In-memory pandas DataFrame layout (runtime-only; not "
                    "JSON-serializable)."
                ),
            },
        ]
    }),
]


_EXTERNAL_QUALIFIED_HEADER = re.compile(r"^[A-Z][A-Za-z0-9]*_[A-Z][A-Za-z0-9]*$")


def _is_known_nonmetadata_header(column: str) -> bool:
    """Return whether *column* belongs to an explicit non-metadata schema enum."""
    for name in schema.__all__:
        candidate = getattr(schema, name)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, MeasurementInfo)
            and not issubclass(candidate, MetadataInfo)
            and candidate.owns_header(column)
        ):
            return True
    return False


def _is_layout_metadata_column(column: str) -> bool:
    """Return whether a layout column belongs to the metadata namespace.

    Known metadata headers and labels always normalize. Explicit known
    non-metadata schema headers remain unchanged. Unknown external headers must
    use two PascalCase segments (for example, ``ExternalMeasure_BatchKey``);
    lower snake-case labels such as ``batch_id`` remain bare layout metadata.
    """
    if (
        is_metadata_header(column)
        or metadata_member_for_header(column) is not None
        or metadata_member_for_label(column) is not None
    ):
        return True
    return not (
        _is_known_nonmetadata_header(column)
        or _EXTERNAL_QUALIFIED_HEADER.fullmatch(column) is not None
    )


def _normalize_layout_column_reference(column: str) -> str:
    """Resolve a layout-context column reference to its emitted spelling."""
    return ensure_metadata_prefix(column) if _is_layout_metadata_column(column) else column


def _normalize_layout_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize layout metadata while preserving qualified non-metadata fields.

    Unlike a metadata-only sidecar, a layout frame can carry qualified object,
    measurement, or arbitrary external columns. Those are retained unchanged.
    Every bare and lower snake-case column is metadata in this file-oriented
    context, so generic labels such as ``Foo`` become ``Metadata_Foo`` through
    the SDK normalizer.
    """
    source_names = [str(column) for column in frame.columns]
    metadata_positions = [
        position
        for position, name in enumerate(source_names)
        if _is_layout_metadata_column(name)
    ]
    if not metadata_positions:
        return frame.copy(deep=True)

    metadata = normalize_metadata_columns(frame.iloc[:, metadata_positions])
    groups: dict[str, list[int]] = {}
    for position in metadata_positions:
        groups.setdefault(
            _normalize_layout_column_reference(source_names[position]), []
        ).append(position)
    anchors = {
        target: next(
            (position for position in positions if source_names[position] == target),
            positions[0],
        )
        for target, positions in groups.items()
    }
    normalized_by_anchor = {
        anchor: metadata.iloc[:, index]
        for index, anchor in enumerate(sorted(anchors.values()))
    }
    consumed = {position for positions in groups.values() for position in positions}
    columns = [
        normalized_by_anchor[position]
        if position in normalized_by_anchor
        else frame.iloc[:, position].copy(deep=True)
        for position in range(len(source_names))
        if position not in consumed or position in normalized_by_anchor
    ]
    return pd.concat(columns, axis=1)


def _normalize_measurement_join_columns(
    frame: pd.DataFrame, groupby: list[str]
) -> pd.DataFrame:
    """Normalize only join keys and known metadata aliases in measurement data.

    Measurements are mixed tables, unlike layout files. Known metadata aliases
    always normalize; otherwise, layout-style normalization applies only to a
    column that supplies one of this check's configured grouping keys. This
    keeps unrelated custom measures such as ``custom_score`` and ``note_text``
    exactly as callers supplied them.
    """
    source_names = [str(column) for column in frame.columns]
    metadata_positions = [
        position
        for position, name in enumerate(source_names)
        if (
            is_metadata_header(name)
            or metadata_member_for_header(name) is not None
            or metadata_member_for_label(name) is not None
            or (
                _is_layout_metadata_column(name)
                and _normalize_layout_column_reference(name) in groupby
            )
        )
    ]
    if not metadata_positions:
        return frame.copy(deep=True)

    metadata = normalize_metadata_columns(frame.iloc[:, metadata_positions])
    groups: dict[str, list[int]] = {}
    for position in metadata_positions:
        groups.setdefault(
            _normalize_layout_column_reference(source_names[position]), []
        ).append(position)
    anchors = {
        target: next(
            (position for position in positions if source_names[position] == target),
            positions[0],
        )
        for target, positions in groups.items()
    }
    normalized_by_anchor = {
        anchor: metadata.iloc[:, index]
        for index, anchor in enumerate(sorted(anchors.values()))
    }
    consumed = {position for positions in groups.values() for position in positions}
    columns = [
        normalized_by_anchor[position]
        if position in normalized_by_anchor
        else frame.iloc[:, position].copy(deep=True)
        for position in range(len(source_names))
        if position not in consumed or position in normalized_by_anchor
    ]
    return pd.concat(columns, axis=1)


class ExpectedVsDetectedCount(QualityCheck, PlotQc):
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

    The single ``metadata`` argument accepts **either** an in-memory
    :class:`pandas.DataFrame` (an "array") **or** a path (``Path`` or
    ``str``) to a ``.csv``/``.parquet`` file. The value is stored verbatim
    — ``self.metadata`` echoes exactly what was passed — and the *resolved*
    frame is read once at construction time onto the private ``_metadata``
    slot. Every column named in ``groupby`` must be present in the resolved
    frame; otherwise :class:`KeyError` is raised at ``__init__`` so the
    failure surfaces before ``analyze`` runs.

    **Serialization:** only a path is JSON-native, so the path is the form
    that round-trips. When ``metadata`` is a path, ``model_dump`` /
    ``pipeline.json`` persist that path string under the same ``metadata``
    key and a reloaded instance re-reads the file. When ``metadata`` is an
    in-memory DataFrame there is no source path to persist — the JSON form
    is ``None`` and the check cannot be rebuilt from JSON alone (it fails
    to instantiate with a clear error, surfaced as a skip-with-warning by
    the lazy QC instantiation path). Configure QC checks from a metadata
    *path* whenever the pipeline is meant to round-trip.

    Args:
        metadata: Layout whose row count per ``groupby`` key is the
            expected colony count. Either an in-memory DataFrame or a path
            (``Path``/``str``) to a ``.csv``/``.parquet`` file. The path
            form is what serializes and round-trips through JSON; an
            in-memory frame is runtime-only.
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
        FileNotFoundError: If ``metadata`` is a path that does not exist.
        KeyError: If any column in ``groupby`` is absent from the
            resolved metadata frame.
        ValueError: If ``metadata`` is a path with an unsupported suffix,
            or if it is ``None`` — i.e. reconstructing from JSON that was
            built from an in-memory frame, which has no source path to
            persist.

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
        >>> from phenotypic.schema import IMAGE
        >>> image_name = str(IMAGE.IMAGE_NAME)
        >>> metadata = pd.DataFrame({
        ...     image_name: ["plate1.png"] * 96,
        ...     "Object_Label": list(range(96)),
        ... })
        >>> measurements = pd.DataFrame({
        ...     image_name: ["plate1.png"] * 95,
        ...     "Object_Label": list(range(95)),
        ... })
        >>> chk = ExpectedVsDetectedCount(
        ...     metadata=metadata,
        ...     groupby=[image_name],
        ... )
        >>> result = chk.analyze(measurements)  # doctest: +SKIP
        >>> "QC_Count_Metric" in result.columns  # doctest: +SKIP
        True

        Advanced — a measurement group has no metadata counterpart, so
        the metric is infinite and the key is recorded:

        >>> metadata = pd.DataFrame({
        ...     image_name: ["plate1.png"] * 96,
        ...     "Object_Label": list(range(96)),
        ... })
        >>> measurements = pd.DataFrame({
        ...     image_name: ["plate2.png"] * 10,
        ...     "Object_Label": list(range(10)),
        ... })
        >>> chk = ExpectedVsDetectedCount(
        ...     metadata=metadata,
        ...     groupby=[image_name],
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
    # Single unified field: an in-memory layout DataFrame *or* a path to a
    # ``.csv``/``.parquet`` layout file. Stored verbatim; the resolved
    # frame is precomputed onto the private ``_metadata`` slot. Only a path
    # round-trips through JSON (see ``_serialize_metadata``). NOTE:
    # reassigning ``metadata`` post-construction does not re-resolve
    # ``_metadata``/``_expected_counts`` (``model_post_init`` runs once);
    # construct a new instance to change the layout.
    metadata: _MetadataField

    _metadata: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)
    _expected_counts: pd.Series = PrivateAttr(default_factory=pd.Series)

    @field_validator("groupby")
    @classmethod
    def _normalize_layout_groupby_references(cls, value: list[str]) -> list[str]:
        """Apply the layout namespace rule to every configured grouping key."""
        return [_normalize_layout_column_reference(column) for column in value]

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(
        cls, value: pd.DataFrame | Path | str | None
    ) -> pd.DataFrame | str:
        """Normalize the raw ``metadata`` input to a frame or a path string.

        Keeps the value in its caller-supplied form — a frame stays a
        frame, a path becomes a plain ``str`` — so ``self.metadata`` echoes
        what was passed and a path round-trips through JSON. The file is
        *not* read here; resolution into the working frame happens in
        :meth:`model_post_init`.

        Args:
            value: An in-memory DataFrame, a path (``Path`` or ``str``) to
                a ``.csv``/``.parquet`` layout file, or ``None`` — the
                sentinel a JSON dump of an in-memory-frame check emits,
                which cannot be rebuilt.

        Returns:
            The DataFrame unchanged, or the path coerced to ``str``.

        Raises:
            ValueError: If ``value`` is ``None`` — i.e. the check was
                serialized from an in-memory frame (which has no source
                path) and cannot be reconstructed from JSON. Configure it
                from a ``.csv``/``.parquet`` path so the source round-trips.
        """
        if value is None:
            raise ValueError(
                f"{cls.__name__} requires 'metadata' as a layout DataFrame "
                "or a .csv/.parquet path. Got None — a check serialized "
                "from an in-memory DataFrame has no source path to "
                "round-trip; configure it from a metadata file path so it "
                "can be rebuilt from JSON."
            )
        if isinstance(value, Path):
            return str(value)
        return value

    @field_serializer("metadata")
    def _serialize_metadata(
        self, value: pd.DataFrame | str, info: Any
    ) -> str | pd.DataFrame | None:
        """Serialize ``metadata`` as its source *path*, never the frame.

        A path string is JSON-native and round-trips; an in-memory frame is
        not, so the JSON form is ``None`` (a reload then fails fast in
        :meth:`_normalize_metadata` with a clear error). In Python mode the
        frame passes through unchanged.

        Args:
            value: The stored ``metadata`` value (a frame or a path str).
            info: Pydantic serialization info; ``info.mode`` is ``"json"``
                or ``"python"``.

        Returns:
            The path string when ``metadata`` is a path; otherwise the
            frame in Python mode, or ``None`` in JSON mode.
        """
        if isinstance(value, str):
            return value
        if info.mode == "json":
            return None
        return value

    def model_post_init(self, __context: Any) -> None:
        """Resolve the layout frame, validate columns, pre-compute counts.

        Runs after pydantic has validated every field. Resolves
        ``self.metadata`` (a frame or a path) into the working frame on the
        private ``_metadata`` slot, verifies every ``groupby`` column is
        present, and caches the per-key expected colony counts.

        Args:
            __context: Pydantic post-init context (unused).

        Raises:
            FileNotFoundError: If ``metadata`` is a path that does not
                exist.
            ValueError: If ``metadata`` is a path with an unsupported
                suffix.
            KeyError: If any column in ``groupby`` is absent from the
                resolved metadata frame.
        """
        super().model_post_init(__context)
        self._metadata = self._resolve_metadata(self.metadata)
        missing = [
            col for col in self.groupby if col not in self._metadata.columns
        ]
        if missing:
            raise KeyError(
                "metadata frame is missing required groupby column(s): "
                f"{missing}"
            )
        self._expected_counts = self._metadata.groupby(
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
            A copied resolved DataFrame with layout metadata normalized. Known
            object, grid, bounding-box, and measurement headers are retained.

        Raises:
            FileNotFoundError: If ``metadata`` is a path that does not
                exist.
            ValueError: If the path has an unsupported suffix.
        """
        if isinstance(metadata, pd.DataFrame):
            return _normalize_layout_columns(metadata)

        path = Path(metadata)
        if not path.exists():
            raise FileNotFoundError(
                f"metadata path does not exist: {path}"
            )

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return _normalize_layout_columns(pd.read_csv(path))
        if suffix == ".parquet":
            return _normalize_layout_columns(pd.read_parquet(path))
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

    @staticmethod
    def _detected_count(group: pd.DataFrame) -> int:
        """Number of rows in *group* that are a real detected object.

        Counts rows carrying an ``Object_Label`` rather than simply taking
        ``len(group)``. A row with no label is not a detection: the CLI's
        ``--metadata`` left join seeds the measurements mirror with rows for
        strains that matched no measured object, and every measurement/info
        column on such a row — ``Object_Label`` included — is null. Counting
        them would report ``detected == expected`` for a strain detected
        *nowhere*, i.e. a metric of 0.0 and a passing check, inverting the very
        divergence this class measures.

        No flag column is consulted, so this stays correct for a frame that
        never went through the CLI. On any frame without null labels — every
        frame today, and every notebook ``analyze()`` call — this equals
        ``len(group)`` exactly, so existing results do not move.

        Args:
            group: One group of the analyzed frame.

        Returns:
            Count of rows with a non-null ``Object_Label``, or ``len(group)``
            when the frame carries no label column at all.
        """
        label = str(OBJECT.LABEL)
        if label not in group.columns:
            return int(len(group))
        return int(group[label].notna().sum())

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
        detected = self._detected_count(group)
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
        return super().analyze(_normalize_measurement_join_columns(data, self.groupby))

    def inspect(
        self,
        subject: Any = None,
        *,
        for_save: bool = False,
        **kwargs: Any,
    ) -> go.Figure:
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
        del subject, for_save
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

    def report(self, subject: Any = None, **kwargs: Any) -> go.Figure:
        """Return the complete expected-count report."""
        return self.inspect(subject, **kwargs)
