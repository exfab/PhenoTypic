"""Multi-page replicate-preserving measurement time-series plot."""

from __future__ import annotations

import inspect
import json
import math
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from phenotypic.abc_.plotting import (
    PlotMeas,
    PlotOutput,
    PlotPage,
    canonical_group_key,
)
from phenotypic.sdk_ import ColumnRef, ColumnRefList, is_metadata_header


class PlotMeasTimeSeries(BaseModel, PlotMeas):
    """Plot replicate scatter time series across grouped environments.

    Each page is one genetic grouping, each row is a measurement, and each
    column is one environmental grouping. Replicates remain separate traces;
    values are never averaged or otherwise aggregated.

    Args:
        page_by: Columns defining separate figure pages.
        environment_by: Columns defining subplot columns within a page.
        replicate_by: Columns identifying biological or technical replicates.
        time: Numeric or ordered time column used on the x-axis.
        measurements: Numeric measurement columns. An empty list selects
            eligible columns automatically.
        connect: Whether to connect points within each replicate trace.
    """

    model_config = ConfigDict(extra="forbid")

    page_by: ColumnRefList = Field(
            default_factory=lambda: ["MetadataGenetic_Strain"]
    )
    environment_by: ColumnRefList
    replicate_by: ColumnRefList
    time: ColumnRef = "MetadataCulture_Time"
    measurements: ColumnRefList = Field(default_factory=list)
    connect: bool = True

    @model_validator(mode="after")
    def _validate_roles(self) -> "PlotMeasTimeSeries":
        roles = {
            "page_by"       : self.page_by,
            "environment_by": self.environment_by,
            "replicate_by"  : self.replicate_by,
        }
        for name, columns in roles.items():
            if not columns:
                raise ValueError(f"{name} must contain at least one column")
            duplicates = sorted(
                    {column for column in columns if columns.count(column) > 1}
            )
            if duplicates:
                raise ValueError(f"{name} contains duplicate columns: {duplicates}")

        claimed: dict[str, str] = {self.time: "time"}
        for role, columns in roles.items():
            for column in columns:
                previous = claimed.get(column)
                if previous is not None:
                    raise ValueError(
                            f"column {column!r} cannot be used for both {previous} and {role}"
                    )
                claimed[column] = role
        return self

    def inspect(
            self,
            subject: Any = None,
            *,
            for_save: bool = False,
            **overrides: Any,
    ) -> PlotOutput:
        """Build one Plotly page per configured page grouping.

        Args:
            subject: Current measurement mirror as a pandas DataFrame.
            for_save: Accepted for the common plotting contract. Plot geometry
                is identical for interactive and saved output.
            **overrides: Optional field-value overrides for this invocation.

        Returns:
            Ordered multi-page output. Empty input returns no pages.
        """
        del for_save
        configured = self.model_copy(update=overrides) if overrides else self
        if not isinstance(subject, pd.DataFrame):
            raise TypeError(
                    "PlotMeasTimeSeries.inspect requires a pandas DataFrame subject"
            )
        if subject.empty:
            return PlotOutput(pages=())
        configured._validate_input_columns(subject)
        measurements = configured._measurement_columns(subject)
        if not measurements:
            raise ValueError("no eligible numeric measurement columns were found")

        pages: list[PlotPage] = []
        page_groups = list(_group_rows(subject, configured.page_by))
        page_groups.sort(key=lambda item: _typed_group_key(configured.page_by, item[0]))
        for page_values, page_frame in page_groups:
            page_pairs = _group_pairs(configured.page_by, page_values)
            page_key = _canonical_group_key(page_pairs)
            pages.append(
                    PlotPage(
                            key=page_key,
                            label=_display_pairs(
                                    page_pairs, values_only=len(page_pairs) == 1
                            ),
                            metadata={
                                column: _metadata_group_value(value)
                                for column, value in page_pairs
                            },
                            figure=configured._build_page(page_frame, measurements),
                    )
            )
        return PlotOutput(pages=tuple(pages))

    def report(self, subject: Any = None, **overrides: Any) -> PlotOutput:
        """Return the complete multi-page report."""
        return self.inspect(subject, **overrides)

    def _validate_input_columns(self, frame: pd.DataFrame) -> None:
        requested = [
            *self.page_by,
            *self.environment_by,
            *self.replicate_by,
            self.time,
            *self.measurements,
        ]
        missing = [column for column in requested if column not in frame.columns]
        if missing:
            raise ValueError(f"measurement table is missing columns: {missing}")
        for role, columns in (
                ("page_by", self.page_by),
                ("environment_by", self.environment_by),
                ("replicate_by", self.replicate_by),
        ):
            for column in columns:
                _validate_group_values(frame[column], role=role, column=column)
        if self.measurements:
            nonnumeric = [
                column
                for column in self.measurements
                if not pd.api.types.is_numeric_dtype(frame[column])
            ]
            if nonnumeric:
                raise ValueError(
                        f"explicit measurements must be numeric: {nonnumeric}"
                )

    def _measurement_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.measurements:
            return list(self.measurements)
        excluded = {
            *self.page_by,
            *self.environment_by,
            *self.replicate_by,
            self.time,
            *_nonmeasurement_schema_headers(),
            *_known_analysis_headers(),
        }
        return [
            column
            for column in frame.columns
            if column not in excluded
               and pd.api.types.is_numeric_dtype(frame[column])
               and not is_metadata_header(column)
               and not column.startswith(("Object_", "Grid_", "Quality", "QC_"))
        ]

    def _build_page(
            self,
            frame: pd.DataFrame,
            measurements: list[str],
    ) -> Any:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        environments = list(_group_rows(frame, self.environment_by))
        environments.sort(
                key=lambda item: _typed_group_key(self.environment_by, item[0])
        )
        columns = len(environments)
        subplot_titles = []
        for row_index in range(len(measurements)):
            for values, _ in environments:
                pairs = _group_pairs(self.environment_by, values)
                subplot_titles.append(_display_pairs(pairs) if row_index == 0 else "")
        figure = make_subplots(
                rows=len(measurements),
                cols=columns,
                subplot_titles=subplot_titles,
                shared_xaxes=False,
                vertical_spacing=min(0.12, 0.35 / max(len(measurements), 1)),
        )

        shown_legends: set[str] = set()
        for column_index, (_, environment_frame) in enumerate(environments, start=1):
            replicates = list(_group_rows(environment_frame, self.replicate_by))
            replicates.sort(
                    key=lambda item: _typed_group_key(self.replicate_by, item[0])
            )
            for replicate_values, replicate_frame in replicates:
                pairs = _group_pairs(self.replicate_by, replicate_values)
                replicate_key = _canonical_group_key(pairs)
                replicate_label = _display_pairs(pairs)
                ordered = replicate_frame.sort_values(
                        self.time, kind="mergesort"
                )
                for row_index, measurement in enumerate(measurements, start=1):
                    showlegend = replicate_key not in shown_legends
                    figure.add_trace(
                            go.Scatter(
                                    x=ordered[self.time].tolist(),
                                    y=ordered[measurement].tolist(),
                                    mode="lines+markers" if self.connect else "markers",
                                    name=replicate_label,
                                    legendgroup=replicate_key,
                                    showlegend=showlegend,
                            ),
                            row=row_index,
                            col=column_index,
                    )
                    if showlegend:
                        shown_legends.add(replicate_key)
            for row_index, measurement in enumerate(measurements, start=1):
                figure.update_xaxes(title_text=self.time, row=row_index,
                                    col=column_index)
                if column_index == 1:
                    figure.update_yaxes(
                            title_text=measurement,
                            row=row_index,
                            col=column_index,
                    )
        figure.update_layout(
                height=max(360, 300 * len(measurements)),
                width=max(650, 450 * columns),
                legend_title_text="Replicate",
        )
        return figure


def _group_rows(
        frame: pd.DataFrame,
        columns: list[str],
) -> list[tuple[Any, pd.DataFrame]]:
    """Group rows without pandas' null-category or mixed-type coercion.

    The caller's frame remains untouched and first-seen row order is retained
    inside each group. Callers sort the returned groups by their canonical
    typed key when a deterministic presentation order is required.
    """
    grouped: dict[str, tuple[Any, list[int]]] = {}
    values_frame = frame.loc[:, columns]
    for position, row in enumerate(
            values_frame.itertuples(index=False, name=None)
    ):
        normalized = tuple(_normalize_group_value(value) for value in row)
        raw_values: Any = normalized[0] if len(normalized) == 1 else normalized
        key = _canonical_group_key(list(zip(columns, normalized)))
        if key not in grouped:
            grouped[key] = (raw_values, [])
        grouped[key][1].append(position)
    return [
        (values, frame.iloc[positions])
        for values, positions in grouped.values()
    ]


def _group_pairs(
        columns: list[str],
        raw_values: Any,
) -> list[tuple[str, Any]]:
    values = raw_values if isinstance(raw_values, tuple) else (raw_values,)
    return [
        (column, _normalize_group_value(value))
        for column, value in zip(columns, values)
    ]


def _typed_group_key(columns: list[str], values: Any) -> str:
    return _canonical_group_key(_group_pairs(columns, values))


def _canonical_group_key(pairs: list[tuple[str, Any]]) -> str:
    """Return a typed group key without losing pandas nanoseconds."""
    encoded: list[list[str | None]] = []
    for column, value in pairs:
        if isinstance(value, pd.Timedelta):
            encoded.append([column, "timedelta_ns", str(value.value)])
        elif isinstance(value, pd.Timestamp):
            encoded.append([column, "datetime_ns", value.isoformat()])
        else:
            encoded.extend(json.loads(canonical_group_key([(column, value)])))
    return json.dumps(encoded, ensure_ascii=False, separators=(",", ":"))


def _normalize_group_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool, datetime, date, timedelta)):
        return value
    raise TypeError(f"unsupported grouping value {value!r} ({type(value).__name__})")


def _metadata_group_value(value: Any) -> str | int | float | bool | None:
    """Return a JSON-native selector value for a plot manifest."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timedelta):
        return f"{value.value} ns"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        nanoseconds = (
                (value.days * 86_400 + value.seconds) * 1_000_000_000
                + value.microseconds * 1_000
        )
        return f"{nanoseconds} ns"
    raise TypeError(
            f"unsupported metadata grouping value {value!r} ({type(value).__name__})"
    )


def _display_pairs(
        pairs: list[tuple[str, Any]],
        *,
        values_only: bool = False,
) -> str:
    if values_only:
        value = pairs[0][1]
        return "<null>" if value is None else str(value)
    return ", ".join(
            f"{column}={'<null>' if value is None else value}" for column, value in
            pairs
    )


def _validate_group_values(
        series: pd.Series,
        *,
        role: str,
        column: str,
) -> None:
    for value in series.drop_duplicates().tolist():
        try:
            normalized = _normalize_group_value(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                    f"{role} column {column!r} contains unsupported grouping value {value!r}"
            ) from exc
        if isinstance(normalized, float) and not math.isfinite(normalized):
            raise ValueError(
                    f"{role} column {column!r} contains infinite grouping value {value!r}"
            )


def _nonmeasurement_schema_headers() -> set[str]:
    import phenotypic.schema as schema
    from phenotypic.schema import MeasurementInfo

    headers: set[str] = set()
    for _, candidate in inspect.getmembers(schema, inspect.isclass):
        if not issubclass(candidate, MeasurementInfo) or candidate is MeasurementInfo:
            continue
        kind = candidate.kind()
        if kind in {"identity", "quality"}:
            headers.update(candidate.get_headers())
    return headers


def _known_analysis_headers() -> set[str]:
    import phenotypic.schema as schema
    from phenotypic.schema import MeasurementInfo

    headers: set[str] = set()
    for name, candidate in inspect.getmembers(schema, inspect.isclass):
        if not issubclass(candidate, MeasurementInfo) or candidate is MeasurementInfo:
            continue
        if "MODEL" in name or name == "EDGE_CORRECTION":
            headers.update(candidate.get_headers())
    return headers


__all__ = ["PlotMeasTimeSeries"]
