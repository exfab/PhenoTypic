"""Tests for the Plotly ``DetectModesPlotter`` faceted detection-mode figure."""

from __future__ import annotations

import plotly.graph_objects as go

from phenotypic._core._image_parts.detection_modes import (
    available_modes,
    get_detection_mode,
)
from phenotypic._core._image_parts.plot_accessor._detect_modes_plotter import (
    DetectModesPlotter,
)
from phenotypic.data import load_synth_yeast_plate


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


def test_iter_figures_single_control_free_spec() -> None:
    """The provider declares exactly one control-free figure named ``detect_modes``."""
    image = load_synth_yeast_plate()
    specs = DetectModesPlotter(image).iter_figures()

    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "detect_modes"
    assert spec.controls == {}


def test_plot_detect_modes_returns_faceted_figure() -> None:
    """``image.plot.detect_modes()`` returns a faceted ``go.Figure`` with one
    subplot (and image trace) per detection mode plus the current detect_mat."""
    image = load_synth_yeast_plate()
    fig = image.plot.detect_modes()

    assert isinstance(fig, go.Figure)

    expected_panels = _num_computable_modes(image)
    assert expected_panels > 1

    # One Image trace is added per panel.
    assert len(fig.data) == expected_panels

    # ``make_subplots`` allocates one (xaxis, yaxis) pair per panel.
    xaxes = [k for k in fig.layout if k.startswith("xaxis")]
    assert len(xaxes) == expected_panels


def test_dash_detect_modes_returns_figure_directly() -> None:
    """A single control-free ``@figure`` → ``dash()`` returns the ``go.Figure``."""
    image = load_synth_yeast_plate()
    fig = image.plot.dash.detect_modes()

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == _num_computable_modes(image)
