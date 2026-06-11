"""Tests for the Error-analysis tab's distribution figure builder."""

from __future__ import annotations

import numpy as np

from phenotypic.gui._design import category_color
from phenotypic.gui.results_viewer._error_tab._figure import build_distribution_figure


def _arrays():
    good = np.array([1.0, 2.0, 3.0, 4.0])
    error = np.array([8.0, 9.0, 10.0, 11.0])
    return good, error


def test_figure_has_two_box_traces():
    good, error = _arrays()
    fig = build_distribution_figure(good, error, "Size_Area", "debris", 5.5)
    box_traces = [t for t in fig.data if t.type == "box"]
    assert len(box_traces) == 2


def test_figure_y_axis_title_is_measurement():
    good, error = _arrays()
    fig = build_distribution_figure(good, error, "Size_Area", "debris", 5.5)
    assert fig.layout.yaxis.title.text == "Size_Area"


def test_figure_has_single_horizontal_cutoff_line_at_cutoff():
    good, error = _arrays()
    fig = build_distribution_figure(good, error, "Size_Area", "debris", 5.5)
    line_shapes = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(line_shapes) == 1
    shape = line_shapes[0]
    assert shape.y0 == 5.5
    assert shape.y1 == 5.5


def test_figure_error_trace_uses_category_color():
    good, error = _arrays()
    fig = build_distribution_figure(good, error, "Size_Area", "debris", 5.5)
    error_trace = next(t for t in fig.data if t.type == "box" and t.name == "debris")
    assert error_trace.marker.color == category_color("debris")


def test_figure_cutoff_shape_is_editable():
    good, error = _arrays()
    fig = build_distribution_figure(good, error, "Size_Area", "debris", 5.5)
    shape = next(s for s in fig.layout.shapes if s.type == "line")
    assert shape.editable is True
