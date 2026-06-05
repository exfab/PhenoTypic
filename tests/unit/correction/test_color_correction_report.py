"""Tests for the Plotly :class:`ColorCorrectionReport` (Panel→Plotly Phase 5).

Covers the notebook-only ``FigureProvider`` that replaces the legacy Panel
``ColorCorrectionDashboard``: figure discovery, per-figure trace types,
image-dependent section gating, and the ``profile.dashboard`` / corrector
delegation returning a composed ``go.Figure`` (control-free → not a widget).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from phenotypic import Image
from phenotypic.correction import ColorCheckerProfile, ColorCorrector
from phenotypic.correction._color_correction._color_correction_report import (
    ColorCorrectionReport,
)

# Reuse the synthetic-checker builders proven in the corrector test suite.
from .test_color_corrector import (
    BLACK_PATCH_NAME,
    make_synthetic_checker,
    make_synthetic_framed_checker_image,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fitted_profile() -> ColorCheckerProfile:
    """A ColorCheckerProfile fitted from synthetic patch colors (no image)."""
    measured_rgb, patch_names = make_synthetic_checker()
    profile = ColorCheckerProfile(degree=2)
    profile._fit_from_patch_colors(measured_rgb, patch_names=patch_names)
    return profile


@pytest.fixture()
def fitted_profile_with_image() -> ColorCheckerProfile:
    """A ColorCheckerProfile fitted from a framed-checker image (image + ROIs)."""
    img = make_synthetic_framed_checker_image()
    return ColorCheckerProfile(degree=2).fit(Image(arr=img, bit_depth=8))


# ---------------------------------------------------------------------------
# TestReportFigures
# ---------------------------------------------------------------------------


class TestReportFigures:
    """Figure discovery, sectioning, and trace types on the report."""

    def test_iter_figures_patch_fit_excludes_image_sections(self, fitted_profile):
        """A patch-color fit (no image) exposes only the delta_e + patches figures."""
        report = ColorCorrectionReport(fitted_profile)
        names = [spec.name for spec in report.iter_figures()]
        assert names == ["fig_delta_e", "fig_patch_swatches"]

    def test_iter_figures_image_fit_includes_all_sections(
        self, fitted_profile_with_image
    ):
        """An image fit exposes all four figures including pipeline/segmentation."""
        report = ColorCorrectionReport(
            fitted_profile_with_image,
            image=fitted_profile_with_image._image,
            rois=fitted_profile_with_image.rois,
        )
        names = [spec.name for spec in report.iter_figures()]
        assert names == [
            "fig_delta_e",
            "fig_patch_swatches",
            "fig_pipeline_steps",
            "fig_segmentation",
        ]

    def test_figure_sections(self, fitted_profile_with_image):
        """Each figure carries its expected section tag (D12 collapsible cards)."""
        report = ColorCorrectionReport(
            fitted_profile_with_image,
            image=fitted_profile_with_image._image,
            rois=fitted_profile_with_image.rois,
        )
        sections = {spec.name: spec.section for spec in report.iter_figures()}
        assert sections == {
            "fig_delta_e": "delta_e",
            "fig_patch_swatches": "patches",
            "fig_pipeline_steps": "pipeline",
            "fig_segmentation": "segmentation",
        }

    def test_no_controls_declared(self, fitted_profile_with_image):
        """No figure declares a Control → dash() yields a composed go.Figure."""
        report = ColorCorrectionReport(
            fitted_profile_with_image,
            image=fitted_profile_with_image._image,
            rois=fitted_profile_with_image.rois,
        )
        assert all(not spec.controls for spec in report.iter_figures())

    def test_delta_e_is_bar(self, fitted_profile):
        """The Delta-E figure is a grouped go.Bar (before/after)."""
        report = ColorCorrectionReport(fitted_profile)
        fig = report.fig_delta_e()
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Bar)
        assert {trace.name for trace in fig.data} == {"Before", "After"}

    def test_patch_swatches_is_image(self, fitted_profile):
        """The matched-patch swatch grid is a go.Image with shape (N, 3, 3)."""
        report = ColorCorrectionReport(fitted_profile)
        fig = report.fig_patch_swatches()
        assert isinstance(fig.data[0], go.Image)
        z = np.asarray(fig.data[0].z)
        n_patches = len(fitted_profile.diagnostics["patches"])
        assert z.shape == (n_patches, 3, 3)

    def test_pipeline_and_segmentation_are_images(self, fitted_profile_with_image):
        """Pipeline and segmentation figures are faceted go.Image grids."""
        report = ColorCorrectionReport(
            fitted_profile_with_image,
            image=fitted_profile_with_image._image,
            rois=fitted_profile_with_image.rois,
        )
        pipeline = report.fig_pipeline_steps()
        assert isinstance(pipeline.data[0], go.Image)
        # One ROI x five stages.
        n_rois = len(fitted_profile_with_image.rois)
        assert len(pipeline.data) == 5 * n_rois

        segmentation = report.fig_segmentation()
        assert isinstance(segmentation.data[0], go.Image)
        # One ROI x two panels (preprocessed + overlay).
        assert len(segmentation.data) == 2 * n_rois

    def test_inspect_returns_delta_e(self, fitted_profile):
        """inspect() selects the primary (delta_e) figure."""
        report = ColorCorrectionReport(fitted_profile)
        fig = report.inspect()
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Bar)


# ---------------------------------------------------------------------------
# TestDashboardEntryPoints
# ---------------------------------------------------------------------------


class TestDashboardEntryPoints:
    """profile.dashboard / corrector.dashboard return a control-free go.Figure."""

    def test_profile_dashboard_returns_figure(self, fitted_profile):
        """profile.dashboard(show=False) returns a composed go.Figure, not a widget."""
        result = fitted_profile.dashboard(show=False)
        assert isinstance(result, go.Figure)

    def test_profile_dashboard_with_image_returns_figure(
        self, fitted_profile_with_image
    ):
        """An image-backed profile dashboard also returns a composed go.Figure."""
        result = fitted_profile_with_image.dashboard(show=False)
        assert isinstance(result, go.Figure)

    def test_corrector_dashboard_delegates(self, fitted_profile):
        """ColorCorrector.dashboard delegates to the profile and returns a go.Figure."""
        corrector = ColorCorrector(profile=fitted_profile)
        result = corrector.dashboard(show=False)
        assert isinstance(result, go.Figure)

    def test_unfitted_profile_raises(self):
        """An unfitted profile cannot build a dashboard."""
        with pytest.raises(RuntimeError, match="unfitted"):
            ColorCheckerProfile().dashboard(show=False)


# ---------------------------------------------------------------------------
# TestReportData
# ---------------------------------------------------------------------------


class TestReportData:
    """Spot-check that the figures reflect the diagnostics data."""

    def test_delta_e_bars_sorted_worst_first(self, fitted_profile):
        """Delta-E 'before' bars are sorted descending (worst patch first)."""
        report = ColorCorrectionReport(fitted_profile)
        fig = report.fig_delta_e()
        before = next(t for t in fig.data if t.name == "Before")
        values = list(before.y)
        assert values == sorted(values, reverse=True)

    def test_rejected_patch_flagged_in_swatch_labels(self):
        """A rejected patch is flagged ``[REJ]`` in the swatch row labels."""
        measured_rgb, patch_names = make_synthetic_checker(noise_sigma=0.005)
        measured_rgb[0] = [1.0, 0.0, 1.0]  # magenta — far from 'dark skin'
        profile = ColorCheckerProfile(degree=2, outlier_sigma=1.5)
        profile._fit_from_patch_colors(measured_rgb, patch_names=patch_names)
        assert profile.diagnostics["rejected_patches"]  # precondition

        report = ColorCorrectionReport(profile)
        fig = report.fig_patch_swatches()
        ticktext = fig.layout.yaxis.ticktext
        assert any("[REJ]" in str(label) for label in ticktext)

    def test_segmentation_uses_black_patch(self, fitted_profile_with_image):
        """The image-backed segmentation figure renders for a known framed grid."""
        report = ColorCorrectionReport(
            fitted_profile_with_image,
            image=fitted_profile_with_image._image,
            rois=fitted_profile_with_image.rois,
        )
        # The black F4 patch must be present in the fitted diagnostics so the
        # segmentation overlay has something meaningful to draw.
        assert BLACK_PATCH_NAME in fitted_profile_with_image.diagnostics["patches"]
        fig = report.fig_segmentation()
        assert len(fig.data) >= 2
