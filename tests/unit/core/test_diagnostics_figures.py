"""Tests for the Plotly ``@figure`` dual renderer on :class:`DiagnosticsPlotter`.

These cover the interactive Plotly surface added alongside the static
matplotlib ``diagnostics()`` (the "dual renderer" design): figure discovery,
control declaration, per-figure trace types, control responsiveness, and a
regression that the original matplotlib path is unchanged.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest

from phenotypic import Image
from phenotypic._core._image_parts.plot_accessor._diagnostics_plotter import (
    DiagnosticsPlotter,
)
from phenotypic.data import load_synth_yeast_plate

# Expected figure name → (section, set of control kwargs).
EXPECTED_FIGURES: dict[str, tuple[str, set[str]]] = {
    "fig_intensity_histogram": ("noise", set()),
    "fig_noise_autocorrelation": ("noise", set()),
    "fig_power_spectral_density": ("noise", set()),
    "fig_detection_matrix": ("contrast", set()),
    "fig_local_contrast_map": ("contrast", set()),
    "fig_contrast_metrics": ("contrast", set()),
    "fig_gradient_magnitude": ("structure", set()),
    "fig_orientation_coherence": ("structure", {"sigma"}),
    "fig_ridge_response": ("structure", {"sigma", "ridge_method"}),
    "fig_background_estimate": ("background", {"bg_sigma"}),
    "fig_local_variance": ("background", set()),
    "fig_quality_summary": ("summary", {"sigma", "ridge_method", "bg_sigma"}),
}

# figure name → expected primary trace class.
EXPECTED_TRACE_TYPE: dict[str, type] = {
    "fig_intensity_histogram": go.Scatter,
    "fig_noise_autocorrelation": go.Heatmap,
    "fig_power_spectral_density": go.Scatter,
    "fig_detection_matrix": go.Heatmap,
    "fig_local_contrast_map": go.Heatmap,
    "fig_contrast_metrics": go.Bar,
    "fig_gradient_magnitude": go.Heatmap,
    "fig_orientation_coherence": go.Heatmap,
    "fig_ridge_response": go.Scatter,
    "fig_background_estimate": go.Heatmap,
    "fig_local_variance": go.Heatmap,
    "fig_quality_summary": go.Scatterpolar,
}


@pytest.fixture(scope="module")
def diagnostics_image() -> Image:
    """Keep the call-time image alive while its plotter is in use."""
    return load_synth_yeast_plate()


@pytest.fixture(scope="module")
def plotter(diagnostics_image: Image) -> DiagnosticsPlotter:
    """A DiagnosticsPlotter weakly bound to the synthetic yeast plate."""
    image = diagnostics_image
    return DiagnosticsPlotter(image)


def _control_defaults(plotter: DiagnosticsPlotter, name: str) -> dict[str, object]:
    """Return ``{kwarg: control.default}`` for the named figure spec."""
    spec = next(s for s in plotter.iter_figures() if s.name == name)
    return {kw: ctrl.default for kw, ctrl in spec.controls.items()}


def test_iter_figures_names_and_order(plotter: DiagnosticsPlotter) -> None:
    """iter_figures() exposes exactly the expected fig_* methods, in order."""
    names = [s.name for s in plotter.iter_figures()]
    assert names == list(EXPECTED_FIGURES)


def test_figure_sections(plotter: DiagnosticsPlotter) -> None:
    """Each figure declares its expected section grouping."""
    sections = {s.name: s.section for s in plotter.iter_figures()}
    for name, (section, _controls) in EXPECTED_FIGURES.items():
        assert sections[name] == section


def test_control_declarations(plotter: DiagnosticsPlotter) -> None:
    """Only the designated figures declare controls, with the right kwargs."""
    declared = {s.name: set(s.controls) for s in plotter.iter_figures()}
    for name, (_section, controls) in EXPECTED_FIGURES.items():
        assert declared[name] == controls


def test_controls_shared_by_identity(plotter: DiagnosticsPlotter) -> None:
    """A control reused across figures is the SAME instance (shared by identity)."""
    specs = {s.name: s for s in plotter.iter_figures()}
    sigma_coherence = specs["fig_orientation_coherence"].controls["sigma"]
    sigma_ridge = specs["fig_ridge_response"].controls["sigma"]
    sigma_summary = specs["fig_quality_summary"].controls["sigma"]
    assert sigma_coherence is sigma_ridge is sigma_summary

    bg_estimate = specs["fig_background_estimate"].controls["bg_sigma"]
    bg_summary = specs["fig_quality_summary"].controls["bg_sigma"]
    assert bg_estimate is bg_summary


def test_quality_summary_is_sole_primary(plotter: DiagnosticsPlotter) -> None:
    """fig_quality_summary is the only primary figure → it is the inspect() target."""
    primaries = [s.name for s in plotter.iter_figures() if s.primary]
    assert primaries == ["fig_quality_summary"]


@pytest.mark.parametrize("name", list(EXPECTED_FIGURES))
def test_each_figure_returns_go_figure_with_expected_trace(
    plotter: DiagnosticsPlotter, name: str
) -> None:
    """Each fig_* called with control defaults returns a non-empty go.Figure."""
    method = getattr(plotter, name)
    fig = method(**_control_defaults(plotter, name))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert isinstance(fig.data[0], EXPECTED_TRACE_TYPE[name])


@pytest.mark.parametrize("name", list(EXPECTED_FIGURES))
def test_each_figure_has_non_empty_primary_trace(
    plotter: DiagnosticsPlotter, name: str
) -> None:
    """The primary trace of each figure carries actual data."""
    method = getattr(plotter, name)
    fig = method(**_control_defaults(plotter, name))
    trace = fig.data[0]
    # Heatmaps carry z; everything else carries x/y (or r/theta for polar).
    payload = (
        getattr(trace, "z", None)
        if isinstance(trace, go.Heatmap)
        else getattr(trace, "r", None)
        if isinstance(trace, go.Scatterpolar)
        else getattr(trace, "x", None)
    )
    assert payload is not None
    assert np.asarray(payload).size > 0


def test_theme_applied_by_decorator(plotter: DiagnosticsPlotter) -> None:
    """The @figure decorator applies the composed house template."""
    fig = plotter.fig_intensity_histogram()
    assert fig.layout.template is not None


def test_background_estimate_control_changes_data(
    plotter: DiagnosticsPlotter,
) -> None:
    """A control-bearing figure produces different data for different control values."""
    low = plotter.fig_background_estimate(bg_sigma=20)
    high = plotter.fig_background_estimate(bg_sigma=120)
    z_low = np.asarray(low.data[0].z)
    z_high = np.asarray(high.data[0].z)
    assert z_low.shape == z_high.shape
    assert not np.allclose(z_low, z_high)


def test_ridge_response_method_control_changes_data(
    plotter: DiagnosticsPlotter,
) -> None:
    """Switching ridge_method changes the ridge-response curve."""
    meij = plotter.fig_ridge_response(sigma=1.5, ridge_method="meijering")
    fran = plotter.fig_ridge_response(sigma=1.5, ridge_method="frangi")
    y_meij = np.asarray(meij.data[0].y)
    y_fran = np.asarray(fran.data[0].y)
    assert not np.allclose(y_meij, y_fran)


def test_inspect_returns_radar_figure(plotter: DiagnosticsPlotter) -> None:
    """inspect() picks the primary figure → a Scatterpolar radar chart."""
    fig = plotter.inspect()
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Scatterpolar)
    r = np.asarray(fig.data[0].r)
    assert r.size > 0
    # Quality scores are normalized to [0, 1].
    assert np.all((r >= 0) & (r <= 1))


def test_inspect_override_changes_radar(plotter: DiagnosticsPlotter) -> None:
    """inspect() honors control overrides (different bg_sigma → different radar)."""
    default = plotter.inspect()
    overridden = plotter.inspect(bg_sigma=120)
    r_default = np.asarray(default.data[0].r)
    r_over = np.asarray(overridden.data[0].r)
    # Uniformity score depends on background nonuniformity, which depends on
    # bg_sigma, so at least one component should move.
    assert not np.allclose(r_default, r_over)


def test_flat_image_histogram_has_only_finite_trace_values() -> None:
    """Flat images should not render a non-finite Gaussian overlay."""
    image = Image(np.full((64, 64, 3), 128, dtype=np.uint8))
    flat_plotter = DiagnosticsPlotter(image)

    fig = flat_plotter.fig_intensity_histogram()

    for trace in fig.data:
        y = getattr(trace, "y", None)
        if y is not None:
            assert np.all(np.isfinite(np.asarray(y, dtype=float)))
    labels = [annotation.text for annotation in fig.layout.annotations]
    assert any("Gaussian fit omitted" in str(label) for label in labels)


@pytest.mark.parametrize("side", [1, 2, 3, 4])
def test_tiny_flat_image_histogram_returns_finite_figure(side: int) -> None:
    """Tiny valid images should not crash while computing noise annotations."""
    image = Image(np.zeros((side, side, 3), dtype=np.uint8))
    flat_plotter = DiagnosticsPlotter(image)

    fig = flat_plotter.fig_intensity_histogram()

    assert isinstance(fig, go.Figure)
    for trace in fig.data:
        y = getattr(trace, "y", None)
        if y is not None:
            assert np.all(np.isfinite(np.asarray(y, dtype=float)))


def test_flat_image_psd_returns_annotated_empty_state() -> None:
    """A zero-power PSD should explain the empty state instead of going blank."""
    image = Image(np.full((64, 64, 3), 128, dtype=np.uint8))
    flat_plotter = DiagnosticsPlotter(image)

    fig = flat_plotter.fig_power_spectral_density()

    assert len(fig.data) == 0
    labels = [annotation.text for annotation in fig.layout.annotations]
    assert any("No positive PSD" in str(label) for label in labels)


def test_matplotlib_diagnostics_regression() -> None:
    """The retained private builder still returns its Matplotlib payload."""
    image = load_synth_yeast_plate()
    result = DiagnosticsPlotter(image).diagnostics()
    assert isinstance(result, tuple)
    assert len(result) == 2
    fig, metrics = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(metrics, dict)
    assert "noise" in metrics
    assert "quality_scores" in metrics
    plt.close(fig)
