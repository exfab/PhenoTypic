"""Ready-to-use multi-page colony metric time-series plot."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from phenotypic.abc_.plotting import PlotMeas, PlotOutput
from phenotypic.schema import (
    CONDITION,
    CULTURE,
    GENETIC,
    SAMPLE,
)
from phenotypic.sdk_ import ColumnRef, ColumnRefList

from ._plot_meas_time_series import PlotMeasTimeSeries
from ._metadata import (
    normalize_metadata_column_reference,
    normalize_metadata_column_references,
)


class PlotColonyMetricOverTime(BaseModel, PlotMeas):
    """Plot one colony metric over time by strain and environment.

    The plot emits one page per strain, one subplot column per environmental
    group, and one trace per replicate. Values are never aggregated. ``on``
    accepts any numeric measurement column, so the same class can visualize
    radius, area, intensity, or a custom numeric colony metric.

    Args:
        on: Numeric measurement column plotted on the y-axis.
        strain_label: Column defining the strain on each figure page.
        groupby: Columns defining the subplot groups on each page. Defaults
            to growth medium.
        replicate_label: Column identifying replicate traces. Defaults to the
            biological replicate identifier.
        time: Numeric or ordered time column used on the x-axis.
        connect: Whether to connect points within each replicate trace.
    """

    model_config = ConfigDict(extra="forbid")

    on: ColumnRef
    strain_label: ColumnRef = str(GENETIC.STRAIN)
    groupby: ColumnRefList = Field(
            default_factory=lambda: [str(CONDITION.MEDIA)]
    )
    replicate_label: ColumnRef = str(SAMPLE.BIO_REPLICATE)
    time: ColumnRef = str(CULTURE.TIME)
    connect: bool = True

    @field_validator("on", "strain_label", "replicate_label", "time", mode="before")
    @classmethod
    def _normalize_column_reference(cls, value: str) -> str:
        """Accept current and flat metadata references in plot settings."""
        if not isinstance(value, str):
            raise ValueError("column reference must be a string")
        return normalize_metadata_column_reference(value)

    @field_validator("groupby", mode="before")
    @classmethod
    def _normalize_groupby_references(cls, value: Any) -> list[str]:
        """Accept current and flat metadata grouping references."""
        return normalize_metadata_column_references(value)

    def inspect(
            self,
            subject: Any = None,
            *,
            for_save: bool = False,
            **overrides: Any,
    ) -> PlotOutput:
        """Build one page per configured colony grouping.

        Args:
            subject: Current measurement mirror as a pandas DataFrame.
            for_save: Accepted for the common plotting contract.
            **overrides: Optional field-value overrides for this invocation.

        Returns:
            Ordered multi-page time-series output.
        """
        configured = (
            type(self).model_validate({**self.model_dump(), **overrides})
            if overrides
            else self
        )
        delegate = PlotMeasTimeSeries(
                page_by=[configured.strain_label],
                environment_by=configured.groupby,
                replicate_by=[configured.replicate_label],
                time=configured.time,
                measurements=[configured.on],
                connect=configured.connect,
        )
        return delegate.inspect(subject, for_save=for_save)

    def report(self, subject: Any = None, **overrides: Any) -> PlotOutput:
        """Return the complete multi-page metric report."""
        return self.inspect(subject, **overrides)


__all__ = ["PlotColonyMetricOverTime"]
