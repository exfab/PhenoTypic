"""Transient Plotly figure provider for :class:`AutoGridFinder` diagnostics.

:class:`GridFitReport` is the notebook-only successor to the old Panel
grid-fitting dashboard. It is a plain (non-pydantic, non-registered)
helper that holds an already-computed timed-pipeline result plus the
per-axis dashboard stats, and exposes a set of control-free ``@figure``
methods that port the former matplotlib ``_plot_*`` panels to raw
``plotly.graph_objects`` figures.

Because every ``@figure`` here is control-free, :meth:`AutoGridFinder.dashboard`
calls ``GridFitReport(...).dash()`` and gets back a single composed,
vertically-stacked ``go.Figure`` (the repo-wide ``.dash() -> go.Figure``
contract for control-free providers). No Panel, no Dash, no ipywidgets.

The report is a *helper* provider: it overrides :meth:`_figure_subject`
to return ``None`` (the figures read the stored data on ``self`` and take
no subject parameter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go

from phenotypic.abc_ import FigureProvider, figure
from phenotypic.schema import BBOX, GRID
from phenotypic.viz.figures._theme import MUTED, NAVY, OKABE_ITO

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

# Okabe-Ito data colors carried over from the matplotlib dashboard. These
# are explicit per-trace data colors (not theme styling), so they are kept
# here rather than delegated to the "phenotypic" template.
_NAVY = NAVY  # series 1 / primary bars
_ORANGE = OKABE_ITO[1]  # fitted-pitch / secondary highlight
_GREEN = OKABE_ITO[3]  # fitted pitch reference line
_VERMILION = OKABE_ITO[6]  # oversized / empty-cell / image-pitch alerts
_GREY = "#BBBBBB"  # neutral reference markers
_MUTED = MUTED  # empty-state placeholder text (theme muted-label token)

# Section tags grouping the @figure panels into logical cards.
_SECTION_TIMING = "timing"
_SECTION_OBJECTS = "objects"
_SECTION_GRID = "grid"
_SECTION_AXIS = "axis"
_SECTION_SUMMARY = "summary"


class GridFitReport(FigureProvider):
    """Notebook Plotly report for one :class:`AutoGridFinder` grid fit.

    Holds the output of :meth:`AutoGridFinder._run_timed_pipeline` together
    with the per-axis dashboard stats produced by
    :meth:`AutoGridFinder._compute_axis_dashboard_stats`, and renders them
    as control-free ``@figure`` panels. Construct it once per fit and
    discard it; it carries no model state and is not registered as a
    plotter.

    Args:
        result: The dict returned by ``AutoGridFinder._run_timed_pipeline``,
            with keys ``timings``, ``info_table``, ``row_edges``,
            ``col_edges``, ``grid_df``, ``pipeline_path``.
        row_stats: Per-row dashboard stats dict from
            ``AutoGridFinder._compute_axis_dashboard_stats(axis=0, ...)``.
        col_stats: Per-column dashboard stats dict from
            ``AutoGridFinder._compute_axis_dashboard_stats(axis=1, ...)``.
        nrows: Expected number of grid rows.
        ncols: Expected number of grid columns.
        image_shape: ``(height, width[, channels])`` of the fitted image.
        num_objects: Number of detected objects in the fitted image.

    Examples:
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.grid import AutoGridFinder
        >>> import phenotypic
        >>> image = phenotypic.GridImage(load_synth_yeast_plate())
        >>> image = OtsuDetector().apply(image, inplace=False)
        >>> finder = AutoGridFinder(nrows=8, ncols=12)
        >>> fig = finder.dashboard(image, show_progress=False)
        >>> type(fig).__name__
        'Figure'
    """

    def __init__(
        self,
        result: dict[str, Any],
        *,
        row_stats: dict[str, Any],
        col_stats: dict[str, Any],
        nrows: int,
        ncols: int,
        image_shape: tuple[int, ...],
        num_objects: int,
    ) -> None:
        self._result = result
        self._row_stats = row_stats
        self._col_stats = col_stats
        self._nrows = nrows
        self._ncols = ncols
        self._image_shape = image_shape
        self._num_objects = num_objects
        self._info_table: pd.DataFrame = result["info_table"]
        self._info_table_empty = bool(self._info_table.empty)

    # This report is a *helper* provider: every ``@figure`` method reads the
    # stored result/stats on ``self`` directly and takes no subject parameter,
    # so the inherited ``_figure_subject() -> None`` is left unoverridden.

    # -- composed dashboard -------------------------------------------------

    def dash(self, subject: Any = None) -> go.Figure:
        """Compose every control-free panel into one stacked ``go.Figure``.

        Overrides :meth:`FigureProvider.dash`. The base mixin composes via
        ``make_subplots`` with uniform ``xy`` cells, which cannot host the
        summary ``go.Table``. This override builds the subplot grid with
        per-row ``specs`` — ``type="table"`` for the summary row, ``type="xy"``
        for the chart rows — and carries each panel's axis titles and shapes
        (the grid-edge / pitch reference lines) over to the composed figure,
        which the generic trace-only composition would otherwise drop.

        Args:
            subject: Unused (this helper holds its own data); accepted only
                to match the :meth:`FigureProvider.dash` signature.

        Returns:
            A single themed ``plotly.graph_objects.Figure`` stacking all
            panels vertically.
        """
        from plotly.subplots import make_subplots

        from phenotypic.viz.figures._theme import apply_theme

        specs = self.iter_figures()
        rendered = [self._render_spec(spec) for spec in specs]
        is_table = [
            bool(fig.data) and fig.data[0].type == "table" for fig in rendered
        ]
        row_specs = [
            [{"type": "table"}] if tbl else [{"type": "xy"}]
            for tbl in is_table
        ]
        composed = make_subplots(
            rows=len(specs),
            cols=1,
            subplot_titles=[s.title for s in specs],
            specs=row_specs,
            vertical_spacing=0.05,
        )
        for row, (sub, tbl) in enumerate(zip(rendered, is_table), start=1):
            for trace in sub.data:
                composed.add_trace(trace, row=row, col=1)
            if tbl:
                continue
            # Carry axis titles / ranges for the chart (xy) rows.
            composed.update_xaxes(
                title_text=sub.layout.xaxis.title.text,
                range=sub.layout.xaxis.range,
                row=row,
                col=1,
            )
            composed.update_yaxes(
                title_text=sub.layout.yaxis.title.text,
                range=sub.layout.yaxis.range,
                row=row,
                col=1,
            )
            # Re-add each panel's reference lines (grid edges, pitch markers)
            # onto this subplot's axes. The standalone shapes use the default
            # ``x``/``y`` data axis (plus a paper-domain perpendicular ref);
            # passing ``row``/``col`` remaps them to this subplot's axes.
            for shape in sub.layout.shapes:
                composed.add_shape(shape.to_plotly_json(), row=row, col=1)
            # Carry annotations too. add_vline(annotation_text=...) labels are
            # annotations (not shapes); the empty-state placeholder is a
            # paper-anchored annotation. Retarget each ref to THIS subplot's
            # axes, preserving any " domain" suffix and re-centering paper ones.
            axis_suffix = "" if row == 1 else str(row)
            for ann in sub.layout.annotations:
                payload = ann.to_plotly_json()
                for key, axis in (("xref", "x"), ("yref", "y")):
                    ref = payload.get(key, "")
                    if ref == "paper":
                        payload[key] = f"{axis}{axis_suffix} domain"
                        payload["x" if axis == "x" else "y"] = 0.5
                    elif ref.startswith(axis):
                        suffix = " domain" if ref.endswith(" domain") else ""
                        payload[key] = f"{axis}{axis_suffix}{suffix}"
                composed.add_annotation(payload)

        composed.update_layout(
            height=320 * len(specs),
            showlegend=False,
            title_text="Grid Fitting Diagnostics",
        )
        return apply_theme(composed)

    # -- empty-state helper -------------------------------------------------

    @staticmethod
    def _empty_figure(
        title: str, message: str = "No objects detected"
    ) -> go.Figure:
        """Return a titled figure with a centered placeholder annotation."""
        fig = go.Figure()
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font=dict(color=_MUTED, size=12),
        )
        fig.update_layout(title=title)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    # -- bbox-area helpers --------------------------------------------------

    def _bbox_areas(self) -> np.ndarray:
        """Per-object bounding-box areas (height x width) from the info table."""
        info_table = self._info_table
        heights = np.asarray(info_table[str(BBOX.MAX_RR)].values) - np.asarray(
            info_table[str(BBOX.MIN_RR)].values
        )
        widths = np.asarray(info_table[str(BBOX.MAX_CC)].values) - np.asarray(
            info_table[str(BBOX.MIN_CC)].values
        )
        return heights * widths

    def _expected_cell_area(self) -> float:
        """Expected per-cell bbox area from the image shape and grid geometry."""
        return (self._image_shape[0] / self._nrows) * (
            self._image_shape[1] / self._ncols
        )

    # -- section "timing" ---------------------------------------------------

    @figure(title="Step Timing", section=_SECTION_TIMING)
    def fig_timing_waterfall(self) -> go.Figure:
        """Per-step pipeline timing as a horizontal bar chart.

        Ports :meth:`AutoGridFinder._plot_timing_waterfall`: one navy bar
        per pipeline step (``regionprops``, ``fit rows``, ``fit cols``,
        ``grid assignment``) with the per-step seconds annotated and the
        cumulative total in the title.

        Returns:
            A ``go.Figure`` whose primary trace is a horizontal ``go.Bar``.
        """
        timings: dict[str, float] = self._result["timings"]
        steps = list(timings.keys())
        times = [timings[s] for s in steps]
        total = sum(times)

        fig = go.Figure(
            go.Bar(
                x=times,
                y=steps,
                orientation="h",
                marker=dict(color=_NAVY),
                text=[f"{t:.3f}s" for t in times],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"Step Timing (total: {total:.3f}s)",
            xaxis_title="Time (s)",
            showlegend=False,
        )
        # Match the matplotlib invert_yaxis so the first step is on top.
        fig.update_yaxes(autorange="reversed")
        return fig

    # -- section "objects" --------------------------------------------------

    @figure(title="Object Size Distribution", section=_SECTION_OBJECTS)
    def fig_object_size_dist(self) -> go.Figure:
        """Histogram of object bounding-box areas with the expected cell size.

        Ports :meth:`AutoGridFinder._plot_object_size_dist`: bbox areas
        split into a normal (navy) series and an oversized (vermilion)
        series exceeding the expected cell area, with a dashed grey
        reference line at the expected cell area.

        Returns:
            A ``go.Figure`` with one or two ``go.Histogram`` traces plus a
            vertical reference line; an empty-state placeholder when no
            objects were detected.
        """
        if self._info_table_empty:
            return self._empty_figure("Object Size Distribution")

        areas = self._bbox_areas()
        expected_cell_area = self._expected_cell_area()
        oversized_mask = areas > expected_cell_area

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=areas[~oversized_mask],
                marker=dict(color=_NAVY),
                opacity=0.8,
                name="Normal",
            )
        )
        if oversized_mask.any():
            fig.add_trace(
                go.Histogram(
                    x=areas[oversized_mask],
                    marker=dict(color=_VERMILION),
                    opacity=0.8,
                    name=f"Oversized ({int(oversized_mask.sum())})",
                )
            )
        fig.add_vline(
            x=expected_cell_area,
            line=dict(color=_GREY, width=1.5, dash="dash"),
            annotation_text="Expected cell area",
            annotation_position="top",
        )
        fig.update_layout(
            title="Object Size Distribution",
            xaxis_title="Bbox Area (px²)",
            yaxis_title="Count",
            barmode="overlay",
        )
        return fig

    # -- section "grid" -----------------------------------------------------

    @figure(title="Centroids with Grid Overlay", section=_SECTION_GRID)
    def fig_center_scatter(self) -> go.Figure:
        """Scatter of weighted object centroids overlaid with grid edges.

        Ports :meth:`AutoGridFinder._plot_center_scatter`: the
        distance-weighted centroids as navy markers, with vermilion
        horizontal/vertical lines at the fitted row/column edges. The
        y-axis is reversed so it matches image (row-down) orientation.

        Returns:
            A ``go.Figure`` whose primary trace is a ``go.Scatter`` with
            grid-edge shapes; an empty-state placeholder (with edges still
            drawn) when no objects were detected.
        """
        row_edges = self._result["row_edges"]
        col_edges = self._result["col_edges"]

        if self._info_table_empty:
            fig = self._empty_figure("Centroids with Grid Overlay")
        else:
            info_table = self._info_table
            cc = info_table[str(BBOX.DIST_WEIGHTED_CENTER_CC)].values
            rr = info_table[str(BBOX.DIST_WEIGHTED_CENTER_RR)].values
            fig = go.Figure(
                go.Scatter(
                    x=cc,
                    y=rr,
                    mode="markers",
                    marker=dict(color=_NAVY, size=4, opacity=0.5),
                    name="Centroids",
                )
            )
            fig.update_layout(title="Centroids with Grid Overlay")

        for edge in row_edges:
            fig.add_hline(
                y=float(edge), line=dict(color=_VERMILION, width=0.8)
            )
        for edge in col_edges:
            fig.add_vline(
                x=float(edge), line=dict(color=_VERMILION, width=0.8)
            )

        fig.update_layout(
            xaxis_title="Column (px)",
            yaxis_title="Row (px)",
        )
        fig.update_xaxes(range=[0, self._image_shape[1]], visible=True)
        fig.update_yaxes(range=[self._image_shape[0], 0], visible=True)
        return fig

    # -- section "axis" -----------------------------------------------------

    @figure(title="Adjacent-Center Spacing", section=_SECTION_AXIS)
    def fig_successive_diffs(self) -> go.Figure:
        """Distribution of successive center diffs per axis with pitch markers.

        Ports :meth:`AutoGridFinder._plot_successive_diffs`: positive
        differences between adjacent sorted centers, one overlaid
        ``go.Histogram`` per axis (row = navy, col = orange), with a green
        fitted-pitch reference line, a vermilion 1x image-pitch line, and
        grey 2x/3x image-pitch markers for spotting sparse-coverage peaks.

        Returns:
            A ``go.Figure`` with up to two ``go.Histogram`` traces plus
            pitch reference lines; an empty-state placeholder when no
            objects were detected.
        """
        if self._info_table_empty:
            return self._empty_figure("Adjacent-Center Spacing")

        fig = go.Figure()
        any_data = False
        max_pitch = 0.0
        for stats, color in (
            (self._row_stats, _NAVY),
            (self._col_stats, _ORANGE),
        ):
            label = stats["label"]
            centers = stats["centers"]
            if len(centers) < 2:
                continue
            diffs = np.diff(centers)
            diffs = diffs[diffs > 0]
            if len(diffs) == 0:
                continue
            any_data = True
            image_pitch = stats["image_pitch"]
            fit_pitch = stats["fit_pitch"]
            fig.add_trace(
                go.Histogram(
                    x=diffs,
                    marker=dict(color=color),
                    opacity=0.7,
                    name=f"{label} diffs",
                )
            )
            fig.add_vline(
                x=fit_pitch,
                line=dict(color=_GREEN, width=1.5),
                annotation_text=f"{label} fit ({fit_pitch:.0f})",
            )
            fig.add_vline(
                x=image_pitch,
                line=dict(color=_VERMILION, width=1.2, dash="dash"),
                annotation_text=f"{label} 1x ip ({image_pitch:.0f})",
            )
            # Grey 2x/3x image-pitch markers help spot sparse-coverage peaks
            # (adjacent-center diffs that skipped one or two grid cells).
            max_pitch = max(max_pitch, image_pitch)
            fig.add_vline(
                x=2 * image_pitch,
                line=dict(color=_GREY, width=1.0, dash="dot"),
                annotation_text=f"{label} 2x ip ({2 * image_pitch:.0f})",
            )
            fig.add_vline(
                x=3 * image_pitch,
                line=dict(color=_GREY, width=1.0, dash="dot"),
                annotation_text=f"{label} 3x ip ({3 * image_pitch:.0f})",
            )

        if not any_data:
            return self._empty_figure(
                "Adjacent-Center Spacing", message="<2 usable centers"
            )

        fig.update_layout(
            title="Adjacent-Center Spacing",
            xaxis_title="Δ between adjacent centers (px)",
            yaxis_title="Count",
            barmode="overlay",
        )
        if max_pitch > 0:
            # Keep the 3x marker in view even when no diff reaches it.
            fig.update_xaxes(range=[0, 3.5 * max_pitch])
        return fig

    @figure(title="Axis Occupancy", section=_SECTION_AXIS)
    def fig_axis_occupancy(self) -> go.Figure:
        """Per-cell detection counts per axis with empty-cell highlighting.

        Adapted from :meth:`AutoGridFinder._plot_axis_occupancy`: the fitted
        per-index detection counts as grouped bars (row vs. col); cells
        with zero detections are colored vermilion to flag gaps. The title
        reports occupied/expected cells per axis. The image-pitch-count
        overlay the legacy plot showed on fit/image-pitch disagreement is
        intentionally dropped here — that comparison is reported numerically
        in the summary table (see DEFERRED.md).

        Returns:
            A ``go.Figure`` with one ``go.Bar`` trace per axis; an
            empty-state placeholder when no objects were detected.
        """
        if self._info_table_empty:
            return self._empty_figure("Axis Occupancy")

        fig = go.Figure()
        for stats in (self._row_stats, self._col_stats):
            label = stats["label"]
            n_expected = stats["n_expected"]
            fit_counts = stats["fit_counts"]
            colors = [_VERMILION if c == 0 else _NAVY for c in fit_counts]
            fig.add_trace(
                go.Bar(
                    x=list(range(n_expected)),
                    y=list(fit_counts),
                    marker=dict(color=colors),
                    opacity=0.85,
                    name=f"{label} ({stats['fit_occupied']}/{n_expected})",
                    text=[str(int(c)) if c > 0 else "" for c in fit_counts],
                    textposition="outside",
                )
            )
        fig.update_layout(
            title="Axis Occupancy (fitted counts per cell)",
            xaxis_title="Cell index",
            yaxis_title="# detections",
            barmode="group",
        )
        return fig

    # -- section "summary" --------------------------------------------------

    @figure(title="Summary", section=_SECTION_SUMMARY)
    def fig_summary_table(self) -> go.Figure:
        """Grid-fit summary as a two-column ``go.Table``.

        Ports :meth:`AutoGridFinder._build_inspect_summary` (formerly a
        ``pn.pane.Markdown``): object count, grid geometry, occupied cells,
        objects-per-cell spread, oversized count, per-axis pitch,
        pipeline path, occupancy/span coverage (fit vs. image-pitch), and
        total time. Disagreements between the fitted and image-pitch
        priors are marked with a warning glyph.

        Returns:
            A ``go.Figure`` whose single trace is a ``go.Table``.
        """
        result = self._result
        row_stats = self._row_stats
        col_stats = self._col_stats
        info_table = self._info_table
        timings = result["timings"]
        grid_df = result["grid_df"]
        nrows = self._nrows
        ncols = self._ncols

        n_objects = len(info_table)
        total_time = sum(timings.values())

        # Objects per cell stats (from the assigned grid frame).
        if not grid_df.empty and str(GRID.ROW_MAJOR_IDX) in grid_df.columns:
            counts = grid_df[str(GRID.ROW_MAJOR_IDX)].value_counts()
            min_per_cell = int(counts.min()) if len(counts) > 0 else 0
            med_per_cell = float(counts.median()) if len(counts) > 0 else 0.0
            max_per_cell = int(counts.max()) if len(counts) > 0 else 0
            occupied = len(counts)
        else:
            min_per_cell = max_per_cell = occupied = 0
            med_per_cell = 0.0

        # Oversized objects.
        if not info_table.empty:
            n_oversized = int(
                (self._bbox_areas() > self._expected_cell_area()).sum()
            )
        else:
            n_oversized = 0

        def _pair(fit: int, ip: int, total: int) -> str:
            """Render a 'fit / ip' coverage cell, marking disagreements."""
            if fit == ip:
                return f"{fit}/{total} ({fit / total:.0%})"
            return (
                f"⚠ fit: {fit}/{total} ({fit / total:.0%}), "
                f"ip: {ip}/{total} ({ip / total:.0%})"
            )

        metrics = [
            "Objects",
            "Grid",
            "Occupied cells",
            "Obj/cell (min / med / max)",
            "Oversized objects",
            "Row pitch",
            "Col pitch",
            "Pipeline path",
            "Row occupancy",
            "Col occupancy",
            "Row span coverage",
            "Col span coverage",
            "Total time",
        ]
        values = [
            f"{n_objects}",
            f"{nrows} x {ncols} ({nrows * ncols} cells)",
            f"{occupied}",
            f"{min_per_cell} / {med_per_cell:.1f} / {max_per_cell}",
            f"{n_oversized}",
            f"{row_stats['fit_pitch']:.1f} px",
            f"{col_stats['fit_pitch']:.1f} px",
            f"{result['pipeline_path']}",
            _pair(row_stats["fit_occupied"], row_stats["ip_occupied"], nrows),
            _pair(col_stats["fit_occupied"], col_stats["ip_occupied"], ncols),
            _pair(row_stats["fit_span"], row_stats["ip_span"], nrows),
            _pair(col_stats["fit_span"], col_stats["ip_span"], ncols),
            f"{total_time:.3f} s",
        ]

        fig = go.Figure(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color=_NAVY,
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(
                    values=[metrics, values],
                    align="left",
                ),
            )
        )
        fig.update_layout(
            title=(
                f"Grid Fitting Diagnostics — {self._num_objects} objects, "
                f"{nrows}x{ncols} grid"
            )
        )
        return fig


__all__ = ["GridFitReport"]
