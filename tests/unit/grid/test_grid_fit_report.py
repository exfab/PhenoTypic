"""Tests for the notebook Plotly ``GridFitReport`` and the rewired
``AutoGridFinder.report`` (Panel -> Plotly migration, Phase 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import phenotypic
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.grid import AutoGridFinder
from phenotypic.grid._grid_fit_report import GridFitReport
from phenotypic.abc_.plotting import PhtPlot


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
        result["info_table"],
        axis=0,
        n_expected=finder.nrows,
        image_dim=detected_image.shape[0],
        edges=result["row_edges"],
    )
    col_stats = finder._compute_axis_dashboard_stats(
        result["info_table"],
        axis=1,
        n_expected=finder.ncols,
        image_dim=detected_image.shape[1],
        edges=result["col_edges"],
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
# AutoGridFinder.report()
# ---------------------------------------------------------------------------


class TestReportReturnsGoFigure:
    def test_report_returns_go_figure(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        fig = finder.report(detected_image, show_progress=False)
        assert isinstance(fig, go.Figure)

    def test_report_is_composed_multiple_traces(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        fig = finder.report(detected_image, show_progress=False)
        # Composed: timing bar + size histograms + scatter + diffs + occupancy
        # bars + summary table => well more than one trace.
        assert len(fig.data) > 1

    def test_report_carries_all_panel_trace_types(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        fig = finder.report(detected_image, show_progress=False)
        trace_types = {type(t).__name__ for t in fig.data}
        # The composed figure mixes bar, histogram, scatter, and table panels.
        assert {"Bar", "Histogram", "Scatter", "Table"} <= trace_types

    def test_report_preserves_grid_edge_shapes(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        fig = finder.report(detected_image, show_progress=False)
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
        assert isinstance(report, PhtPlot)
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

    def test_report_composes_to_single_figure(self, report):
        fig = report.report()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 1

    def test_report_carries_reference_line_labels(self, report):
        """add_vline annotation labels survive composition (F5)."""
        fig = report.report()
        labels = [a.text for a in fig.layout.annotations if a.text]
        assert any("fit (" in t for t in labels), (
            "pitch-fit label lost in compose"
        )

    def test_successive_diffs_has_2x_3x_pitch_markers(self, report):
        """fig_successive_diffs draws 2x and 3x image-pitch reference lines (F6)."""
        fig = report.fig_successive_diffs()
        labels = [a.text for a in fig.layout.annotations if a.text]
        assert any("2x ip" in t for t in labels)
        assert any("3x ip" in t for t in labels)

    def test_successive_diffs_range_includes_observed_sparse_peak(self):
        """Observed diffs beyond the 3x marker should remain visible."""
        result = {
            "info_table": pd.DataFrame({"label": [1]}),
        }
        row_stats = {
            "label": "row",
            "centers": np.asarray([0.0, 10.0, 100.0]),
            "image_pitch": 10.0,
            "fit_pitch": 10.0,
        }
        col_stats = {
            "label": "col",
            "centers": np.asarray([0.0]),
            "image_pitch": 10.0,
            "fit_pitch": 10.0,
        }
        sparse_report = GridFitReport(
            result,
            row_stats=row_stats,
            col_stats=col_stats,
            nrows=8,
            ncols=12,
            image_shape=(120, 180),
            num_objects=1,
        )

        fig = sparse_report.fig_successive_diffs()

        assert fig.layout.xaxis.range[1] >= 90.0


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
            result["info_table"],
            axis=0,
            n_expected=finder.nrows,
            image_dim=image.shape[0],
            edges=result["row_edges"],
        )
        col_stats = finder._compute_axis_dashboard_stats(
            result["info_table"],
            axis=1,
            n_expected=finder.ncols,
            image_dim=image.shape[1],
            edges=result["col_edges"],
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

    def test_empty_report_returns_figure(self, empty_report):
        fig = empty_report.report()
        assert isinstance(fig, go.Figure)

    def test_empty_summary_table_still_renders(self, empty_report):
        fig = empty_report.fig_summary_table()
        assert fig.data[0].type == "table"


def test_plotting_objects_have_no_legacy_report_aliases(report):
    assert not hasattr(report, "dash")
    assert not hasattr(report, "dashboard")
    assert not hasattr(AutoGridFinder, "dashboard")
