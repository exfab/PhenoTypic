"""Tests for the notebook Plotly ``GridFitReport`` and the rewired
``AutoGridFinder.dashboard`` (Panel -> Plotly migration, Phase 5)."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

import phenotypic
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.grid import AutoGridFinder
from phenotypic.grid._grid_fit_report import GridFitReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detected_image():
    """A gridded synth yeast plate with objects detected (8x12 layout)."""
    image = phenotypic.GridImage(load_synth_yeast_plate())
    return OtsuDetector().apply(image, inplace=False)


@pytest.fixture(scope="module")
def report(detected_image):
    """A built ``GridFitReport`` for the detected synth plate."""
    finder = AutoGridFinder(nrows=8, ncols=12)
    result = finder._run_timed_pipeline(detected_image, show_progress=False)
    row_stats = finder._compute_axis_dashboard_stats(
        result["info_table"], axis=0, n_expected=finder.nrows,
        image_dim=detected_image.shape[0], edges=result["row_edges"],
    )
    col_stats = finder._compute_axis_dashboard_stats(
        result["info_table"], axis=1, n_expected=finder.ncols,
        image_dim=detected_image.shape[1], edges=result["col_edges"],
    )
    return GridFitReport(
        result,
        row_stats=row_stats,
        col_stats=col_stats,
        nrows=finder.nrows,
        ncols=finder.ncols,
        image_shape=detected_image.shape,
        num_objects=detected_image.num_objects,
    )


# ---------------------------------------------------------------------------
# AutoGridFinder.dashboard()
# ---------------------------------------------------------------------------


class TestDashboardReturnsGoFigure:

    def test_dashboard_returns_go_figure(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        fig = finder.dashboard(detected_image, show_progress=False)
        assert isinstance(fig, go.Figure)

    def test_dashboard_is_composed_multiple_traces(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        fig = finder.dashboard(detected_image, show_progress=False)
        # Composed: timing bar + size histograms + scatter + diffs + occupancy
        # bars + summary table => well more than one trace.
        assert len(fig.data) > 1

    def test_dashboard_carries_all_panel_trace_types(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        fig = finder.dashboard(detected_image, show_progress=False)
        trace_types = {type(t).__name__ for t in fig.data}
        # The composed figure mixes bar, histogram, scatter, and table panels.
        assert {"Bar", "Histogram", "Scatter", "Table"} <= trace_types

    def test_dashboard_preserves_grid_edge_shapes(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        fig = finder.dashboard(detected_image, show_progress=False)
        # Grid-edge + pitch reference lines survive composition as shapes.
        assert len(fig.layout.shapes) > 0


# ---------------------------------------------------------------------------
# GridFitReport.iter_figures() + per-figure spot checks
# ---------------------------------------------------------------------------


class TestGridFitReportFigures:

    EXPECTED_FIGURES = [
        "fig_timing_waterfall",
        "fig_object_size_dist",
        "fig_center_scatter",
        "fig_successive_diffs",
        "fig_axis_occupancy",
        "fig_summary_table",
    ]

    def test_iter_figures_count(self, report):
        specs = report.iter_figures()
        assert len(specs) == len(self.EXPECTED_FIGURES)

    def test_iter_figures_are_control_free(self, report):
        specs = report.iter_figures()
        assert all(not spec.controls for spec in specs)

    def test_iter_figures_names_in_definition_order(self, report):
        names = [spec.name for spec in report.iter_figures()]
        assert names == self.EXPECTED_FIGURES

    def test_timing_waterfall_is_bar(self, report):
        fig = report.fig_timing_waterfall()
        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "bar"
        assert fig.data[0].orientation == "h"

    def test_object_size_dist_is_histogram(self, report):
        fig = report.fig_object_size_dist()
        assert isinstance(fig, go.Figure)
        assert any(t.type == "histogram" for t in fig.data)

    def test_center_scatter_is_scatter_with_edges(self, report):
        fig = report.fig_center_scatter()
        assert isinstance(fig, go.Figure)
        assert any(t.type == "scatter" for t in fig.data)
        # Row + column grid edges drawn as line shapes.
        assert len(fig.layout.shapes) > 0

    def test_successive_diffs_is_histogram(self, report):
        fig = report.fig_successive_diffs()
        assert isinstance(fig, go.Figure)
        assert any(t.type == "histogram" for t in fig.data)

    def test_axis_occupancy_is_bar(self, report):
        fig = report.fig_axis_occupancy()
        assert isinstance(fig, go.Figure)
        assert all(t.type == "bar" for t in fig.data)

    def test_summary_is_table(self, report):
        fig = report.fig_summary_table()
        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "table"

    def test_dash_composes_to_single_figure(self, report):
        fig = report.dash()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 1


# ---------------------------------------------------------------------------
# Empty-state handling (no detected objects)
# ---------------------------------------------------------------------------


class TestGridFitReportEmptyState:

    @pytest.fixture
    def empty_report(self):
        """A report over an image with zero detected objects."""
        image = phenotypic.GridImage(load_synth_yeast_plate())
        # No detector applied -> num_objects == 0 -> empty info_table path.
        finder = AutoGridFinder(nrows=8, ncols=12)
        result = finder._run_timed_pipeline(image, show_progress=False)
        row_stats = finder._compute_axis_dashboard_stats(
            result["info_table"], axis=0, n_expected=finder.nrows,
            image_dim=image.shape[0], edges=result["row_edges"],
        )
        col_stats = finder._compute_axis_dashboard_stats(
            result["info_table"], axis=1, n_expected=finder.ncols,
            image_dim=image.shape[1], edges=result["col_edges"],
        )
        return GridFitReport(
            result,
            row_stats=row_stats,
            col_stats=col_stats,
            nrows=finder.nrows,
            ncols=finder.ncols,
            image_shape=image.shape,
            num_objects=image.num_objects,
        )

    def test_empty_dash_returns_figure(self, empty_report):
        fig = empty_report.dash()
        assert isinstance(fig, go.Figure)

    def test_empty_summary_table_still_renders(self, empty_report):
        fig = empty_report.fig_summary_table()
        assert fig.data[0].type == "table"
