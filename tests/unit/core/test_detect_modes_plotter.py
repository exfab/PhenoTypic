"""Tests for the standalone ``PlotDetectModes`` image plot."""

from __future__ import annotations

import plotly.graph_objects as go

from phenotypic._core._image_parts.detection_modes import (
    available_modes,
    get_detection_mode,
)
from phenotypic.data import load_synth_yeast_plate
from phenotypic.plotting import PlotDetectModes


def _num_computable_modes(image) -> int:
    """Count registered modes computable for *image* (plus the detect_mat panel)."""
    has_rgb = not image.rgb.isempty()
    count = 0
    for name in available_modes():
        mode = get_detection_mode(name)
        if mode.requires_rgb and not has_rgb:
            continue
        count += 1
    return count + 1  # + the appended current detect_mat panel


def test_plot_detect_modes_is_fieldless_serializable_model() -> None:
    assert PlotDetectModes().model_dump() == {}


def test_plot_detect_modes_returns_faceted_figure() -> None:
    """``inspect(image)`` returns one panel per mode plus detect_mat."""
    image = load_synth_yeast_plate()
    fig = PlotDetectModes().inspect(image)

    assert isinstance(fig, go.Figure)

    expected_panels = _num_computable_modes(image)
    assert expected_panels > 1

    # One Image trace is added per panel.
    assert len(fig.data) == expected_panels

    # ``make_subplots`` allocates one (xaxis, yaxis) pair per panel.
    xaxes = [k for k in fig.layout if k.startswith("xaxis")]
    assert len(xaxes) == expected_panels


def test_report_detect_modes_returns_figure_directly() -> None:
    """The full report is the same single faceted figure."""
    image = load_synth_yeast_plate()
    fig = PlotDetectModes().report(image)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == _num_computable_modes(image)
