"""Ready-to-use multi-page colony metric time-series plot."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from phenotypic.abc_.plotting import PlotMeas
from phenotypic.schema import (
    CONDITION_METADATA,
    CULTURE_METADATA,
    GENETIC_METADATA,
    SAMPLE_METADATA,
)
from phenotypic.sdk_ import ColumnRef, ColumnRefList

from ._output import PlotOutput
from ._plot_meas_time_series import PlotMeasTimeSeries


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
    strain_label: ColumnRef = str(GENETIC_METADATA.STRAIN)
    groupby: ColumnRefList = Field(
        default_factory=lambda: [str(CONDITION_METADATA.MEDIA)]
    )
    replicate_label: ColumnRef = str(SAMPLE_METADATA.BIO_REPLICATE)
    time: ColumnRef = str(CULTURE_METADATA.TIME)
    connect: bool = True

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
