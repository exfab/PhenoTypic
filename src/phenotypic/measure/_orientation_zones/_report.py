"""Composed notebook report for :class:`MeasureOrientationZones`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from phenotypic.abc_.plotting import PlotImage

from ._common import _ZONES

if TYPE_CHECKING:
    from ._operation import MeasureOrientationZones


class _OrientationZonesReport(PlotImage):
    """Stateless image consumer composing the orientation diagnostic.

    The image and producing operation are call-time inputs only. The report
    instance retains neither, avoiding a second object graph that keeps image
    data or measurement caches alive after composition.
    """

    @staticmethod
    def _panel_overview(image, producer: "MeasureOrientationZones"):
        """Panel A: the saveable inspect() overview (legend layers flattened)."""
        return producer.inspect(image, for_save=True)

    @staticmethod
    def _panel_coherence(image, producer: "MeasureOrientationZones"):
        """Panel B: the coherence heatmap.

        Recomputed on demand via ``_coherence_canvas`` (the lean cache holds no
        full-resolution coherence) and discarded — costs compute, not memory.
        """
        import plotly.graph_objects as go

        canvas = producer._coherence_canvas(image)
        fig = go.Figure(
                go.Heatmap(
                        z=canvas,
                        colorscale="Viridis",
                        zmin=0,
                        zmax=1,
                        colorbar=dict(title="C"),
                )
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    @staticmethod
    def _panel_summary(producer: "MeasureOrientationZones"):
        """Panel C: table of primary outward-rotation measurements.

        One row per radial zone, aggregated across objects as ``np.nanmean``
        over the compact primary-metric cache. Requires the custom
        :meth:`report` override because the base composer cannot host a table.
        """
        import plotly.graph_objects as go

        def _mean(zone: str, metric: str) -> float:
            return _OrientationZonesReport._safe_nanmean(
                    [
                        record["outward_rotation"][zone][metric]
                        for record in producer._cache.values()
                    ]
            )

        def _format(value: float, digits: int) -> str:
            return f"{value:.{digits}f}" if np.isfinite(value) else ""

        rows = []
        for zone in _ZONES:
            rows.append(
                    (
                        zone,
                        _format(_mean(zone, "OutwardRotationSustainedPeak"), 2),
                        _format(_mean(zone, "OutwardRotationNet"), 2),
                        _format(_mean(zone, "OutwardRotationRate"), 4),
                        _format(_mean(zone, "OutwardRotationConsistency"), 3),
                    )
            )
        header = [
            "Zone",
            "Sustained peak (deg)",
            "Net rotation (deg)",
            "Rotation rate (deg/px)",
            "Consistency",
        ]
        cols = list(zip(*rows)) if rows else [(), (), (), (), ()]
        return go.Figure(
                go.Table(
                        header=dict(values=header),
                        cells=dict(values=[list(c) for c in cols]),
                )
        )

    def report(
            self,
            subject=None,
            *,
            producer: "MeasureOrientationZones | None" = None,
            **overrides,
    ):
        """Compose the three panels into one stacked ``go.Figure``.

        Renders the three panels from call-time inputs, detects table versus xy
        panels, builds matching subplots, transfers traces, carries the overview
        shapes and annotations, and applies the house theme.

        Args:
            subject: Image consumed by this ``PlotImage`` report.
            producer: The :class:`MeasureOrientationZones` operation whose
                settings and compact cache supply the summaries. It is not
                stored on this object.
            **overrides: Plot-specific overrides are unsupported.

        Returns:
            A single themed ``plotly.graph_objects.Figure``.
        """
        from ._operation import MeasureOrientationZones

        if not isinstance(producer, MeasureOrientationZones):
            raise ValueError(
                    "report(): a MeasureOrientationZones producer is required"
            )
        if overrides:
            raise ValueError(
                    f"report(): unsupported override(s) {sorted(overrides)}"
            )
        if subject is None:
            raise ValueError("report(): an image subject is required")

        from plotly.subplots import make_subplots

        from phenotypic.sdk_.viz.figures._theme import apply_theme

        titles = (
            "Orientation-field overlay",
            "Coherence map",
            "Primary outward-rotation metrics",
        )
        rendered = [
            self._panel_overview(subject, producer),
            self._panel_coherence(subject, producer),
            self._panel_summary(producer),
        ]
        is_table = [
            bool(fig.data) and fig.data[0].type == "table" for fig in rendered
        ]
        row_specs = [
            [{"type": "table"}] if tbl else [{"type": "xy"}]
            for tbl in is_table
        ]
        composed = make_subplots(
                rows=len(rendered),
                cols=1,
                subplot_titles=titles,
                specs=row_specs,
                vertical_spacing=0.06,
        )
        # ``xy_row`` counts cartesian panels: a table cell creates no x/y axis,
        # so the Nth xy panel owns axis number N (mirrors GridFitReport.report).
        xy_row = 0
        for row, (sub, tbl) in enumerate(zip(rendered, is_table), start=1):
            for trace in sub.data:
                composed.add_trace(trace, row=row, col=1)
            if tbl:
                continue
            xy_row += 1
            # Carry the standalone panel's shapes and explanatory annotations
            # onto this subplot.
            for shape in sub.layout.shapes:
                composed.add_shape(shape.to_plotly_json(), row=row, col=1)
            axis_suffix = "" if xy_row == 1 else str(xy_row)
            for ann in sub.layout.annotations:
                payload = ann.to_plotly_json()
                for key, axis in (("xref", "x"), ("yref", "y")):
                    ref = payload.get(key, "")
                    if ref == "paper":
                        payload[key] = f"{axis}{axis_suffix} domain"
                    elif ref.startswith(axis):
                        suffix = " domain" if ref.endswith(" domain") else ""
                        payload[key] = f"{axis}{axis_suffix}{suffix}"
                composed.add_annotation(payload)
        composed.update_layout(
                height=420 * len(rendered),
                title_text="Orientation-Field Diagnostics",
        )
        return apply_theme(composed)

    @staticmethod
    def _safe_nanmean(values) -> float:
        """``np.nanmean`` that returns NaN (not a warning) for an all-NaN input."""
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        return float(finite.mean()) if finite.size else float("nan")
