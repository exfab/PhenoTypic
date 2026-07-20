"""Tests for the Plotly :class:`ColorCorrectionReport` (Panel→Plotly Phase 5).

Covers the notebook-only ``PhtPlot`` that replaces the legacy Panel
``ColorCorrectionDashboard``: figure discovery, per-figure trace types,
image-dependent section gating, and the ``profile.report`` / corrector
delegation returning a composed ``go.Figure`` (control-free → not a widget).
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import plotly.graph_objects as go
import pytest

from phenotypic import Image
from phenotypic.correction import ColorCheckerProfile, ColorCorrector
from phenotypic.correction._color_correction._color_correction_report import (
    ColorCorrectionReport,
)
from phenotypic.abc_.plotting import PhtPlot

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
def fitted_profile_with_image(request) -> ColorCheckerProfile:
    """A ColorCheckerProfile fitted from a framed-checker image (image + ROIs)."""
    image = Image(arr=make_synthetic_framed_checker_image(), bit_depth=8)
    request.node._color_checker_image = image
    return ColorCheckerProfile(degree=2).fit(image)


# ---------------------------------------------------------------------------
# TestReportFigures
# ---------------------------------------------------------------------------


class TestReportFigures:
    """Figure discovery, sectioning, and trace types on the report."""

    def test_iter_figures_patch_fit_excludes_image_sections(self, fitted_profile):
        """A patch-color fit (no image) exposes only the delta_e + patches figures."""
        report = ColorCorrectionReport(fitted_profile)
        assert isinstance(report, PhtPlot)
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
        """No figure declares a Control, so report() yields a composed figure."""
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


class TestReportEntryPoints:
    """Profile and corrector report methods return a control-free go.Figure."""

    def test_profile_report_returns_figure(self, fitted_profile):
        """profile.report(show=False) returns a composed figure, not a widget."""
        result = fitted_profile.report(show=False)
        assert isinstance(result, go.Figure)

    def test_profile_and_report_do_not_retain_source_image(self):
        image = Image(
            arr=make_synthetic_framed_checker_image(),
            bit_depth=8,
        )
        image_ref = weakref.ref(image)
        profile = ColorCheckerProfile(degree=2).fit(image)
        report = ColorCorrectionReport(profile, image=image, rois=profile.rois)

        del image
        gc.collect()

        assert image_ref() is None
        assert profile._image is None
        assert [spec.name for spec in report.iter_figures()] == [
            "fig_delta_e",
            "fig_patch_swatches",
        ]

    def test_profile_report_with_image_returns_figure(
        self, fitted_profile_with_image
    ):
        """An image-backed profile report also returns a composed go.Figure."""
        result = fitted_profile_with_image.report(show=False)
        assert isinstance(result, go.Figure)

    def test_image_report_preserves_nested_subplot_axes(
        self, fitted_profile_with_image
    ):
        """Pipeline stage panels stay on separate axes in the composed report."""
        result = fitted_profile_with_image.report(show=False)
        layout = result.layout.to_plotly_json()
        image_axes = []
        for trace in result.data:
            if trace.type != "image":
                continue
            xref = trace.xaxis or "x"
            yref = trace.yaxis or "y"
            image_axes.append(
                (
                    xref,
                    yref,
                    layout[f"xaxis{xref[1:]}"]["domain"],
                    layout[f"yaxis{yref[1:]}"]["domain"],
                )
            )

        stage_labels = [
            "1. Original Crop",
            "2. Background Trimmed",
            "3. Median Filtered",
            "4. Centered & Padded",
            "5. Border Mask",
        ]
        stage_axes = set()
        for label in stage_labels:
            annotation = next(
                item for item in result.layout.annotations if item.text == label
            )
            matches = [
                (xref, yref, ydomain[1])
                for xref, yref, xdomain, ydomain in image_axes
                if xdomain[0] <= annotation.x <= xdomain[1]
                and ydomain[1] <= annotation.y
            ]
            assert matches
            xref, yref, _ = max(matches, key=lambda item: item[2])
            stage_axes.add((xref, yref))

        assert len(stage_axes) == len(stage_labels)

    def test_patch_color_report_preserves_shapes_and_swatch_hover(
        self, fitted_profile
    ):
        """No-image reports keep Delta-E refs and swatch hover metadata."""
        result = fitted_profile.report(show=False)

        assert len(result.layout.shapes) >= 3
        labels = [annotation.text for annotation in result.layout.annotations]
        assert "Just noticeable" in labels
        assert "Perceptible" in labels
        assert "Significant" in labels

        swatch = next(trace for trace in result.data if trace.type == "image")
        assert swatch.hovertemplate == "%{customdata}<extra></extra>"
        assert swatch.customdata is not None

    def test_image_report_handles_multiple_rois(self):
        """Composed report domains stay valid for multiple image ROIs."""
        card = make_synthetic_framed_checker_image()
        gap = np.zeros((card.shape[0], 20, 3), dtype=card.dtype)
        combined = np.concatenate([card, gap, card], axis=1)
        profile = ColorCheckerProfile(degree=2).fit(Image(arr=card, bit_depth=8))
        left = (slice(0, card.shape[0]), slice(0, card.shape[1]))
        right_start = card.shape[1] + gap.shape[1]
        right = (
            slice(0, card.shape[0]),
            slice(right_start, right_start + card.shape[1]),
        )
        source_image = Image(arr=combined, bit_depth=8)
        report = ColorCorrectionReport(
            profile,
            image=source_image,
            rois=[left, right],
        )

        result = report.report()

        layout = result.layout.to_plotly_json()
        y_domains = [
            axis["domain"]
            for name, axis in layout.items()
            if name.startswith("yaxis") and "domain" in axis
        ]
        assert y_domains
        for domain in y_domains:
            assert 0.0 <= domain[0] <= domain[1] <= 1.0

    def test_report_composer_supports_domain_traces(self):
        """Domain-based traces should stay domain-based when remapped."""
        composed = go.Figure()
        source = go.Figure(
            go.Table(
                header=dict(values=["metric"]),
                cells=dict(values=[["value"]]),
            )
        )

        ColorCorrectionReport._append_figure_to_domain(
            composed,
            source,
            y0=0.2,
            y1=0.6,
            axis_counts={"x": 0, "y": 0},
        )

        assert composed.data[0].type == "table"
        assert tuple(composed.data[0].domain.y) == (0.2, 0.6)

    def test_corrector_report_delegates(self, fitted_profile):
        """ColorCorrector.report delegates and returns a go.Figure."""
        corrector = ColorCorrector(profile=fitted_profile)
        result = corrector.report(show=False)
        assert isinstance(result, go.Figure)

    def test_unfitted_profile_raises(self):
        """An unfitted profile cannot build a report."""
        with pytest.raises(RuntimeError, match="unfitted"):
            ColorCheckerProfile().report(show=False)

    def test_legacy_report_aliases_are_absent(self, fitted_profile):
        report = ColorCorrectionReport(fitted_profile)
        assert not hasattr(report, "dash")
        assert not hasattr(report, "dashboard")
        assert not hasattr(fitted_profile, "dashboard")
        assert not hasattr(ColorCorrector, "dashboard")


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
