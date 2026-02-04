"""Tests for the interactive DiagnosticsDashboard (Panel-based)."""

import matplotlib
matplotlib.use("Agg")

import pytest
import matplotlib.pyplot as plt

from phenotypic._core._image_parts.plot_accessor._diagnostics_dashboard import (
    PANEL_AVAILABLE,
)

pytestmark = pytest.mark.skipif(
    not PANEL_AVAILABLE, reason="Panel/param not installed"
)

if PANEL_AVAILABLE:
    import panel as pn
    from phenotypic._core._image_parts.plot_accessor._diagnostics_dashboard import (
        DiagnosticsDashboard,
    )

from phenotypic.data import load_synth_yeast_plate


@pytest.fixture(scope="module")
def sample_image():
    """Load a sample image once for the module."""
    return load_synth_yeast_plate()


@pytest.fixture
def dashboard(sample_image):
    """Create a DiagnosticsDashboard from the sample image."""
    result, _metrics = sample_image.plot.diagnostics()
    return result


class TestCreateDashboard:
    """Test dashboard creation."""

    def test_create_dashboard(self, dashboard):
        assert isinstance(dashboard, DiagnosticsDashboard)

    def test_diagnostics_returns_two_tuple(self, sample_image):
        result = sample_image.plot.diagnostics()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_diagnostics_first_element_is_dashboard(self, sample_image):
        dashboard, _ = sample_image.plot.diagnostics()
        assert isinstance(dashboard, DiagnosticsDashboard)
        assert hasattr(dashboard, "panel")

    def test_dashboard_with_custom_params(self, sample_image):
        dashboard, _ = sample_image.plot.diagnostics(
            structure_sigma=3.0,
            ridge_method="frangi",
            ridge_scales=[1, 2, 3],
            background_sigma=80.0,
        )
        assert dashboard.structure_sigma == 3.0
        assert dashboard.ridge_method == "frangi"
        assert dashboard.background_sigma == 80.0
        assert dashboard.ridge_scales_str == "1, 2, 3"


class TestPanelReturnsViewable:
    """Test that .panel() returns a Panel viewable."""

    def test_panel_returns_viewable(self, dashboard):
        layout = dashboard.panel()
        assert isinstance(layout, pn.viewable.Viewable)

    def test_panel_returns_column(self, dashboard):
        layout = dashboard.panel()
        assert isinstance(layout, pn.Column)


class TestMetricsDictStructure:
    """Test that the metrics dict has all expected keys."""

    EXPECTED_TOP_KEYS = {
        "bit_depth",
        "noise",
        "contrast",
        "structure",
        "background",
        "quality_scores",
        "interpretations",
        "recommendations",
    }

    EXPECTED_NOISE_KEYS = {"snr", "sigma_mad", "correlation_length"}
    EXPECTED_CONTRAST_KEYS = {"rms_contrast", "michelson", "dynamic_range", "p1", "p99"}
    EXPECTED_STRUCTURE_KEYS = {
        "mean_coherence",
        "optimal_scale",
        "peak_response",
        "ridge_responses",
        "scales",
        "ridge_method",
    }
    EXPECTED_BACKGROUND_KEYS = {"nonuniformity_ratio", "mean_gradient"}
    EXPECTED_QUALITY_KEYS = {"SNR", "Contrast", "Coherence", "Uniformity", "Sharpness"}

    def test_top_level_keys(self, dashboard):
        metrics = dashboard.metrics
        assert self.EXPECTED_TOP_KEYS == set(metrics.keys())

    def test_noise_keys(self, dashboard):
        metrics = dashboard.metrics
        assert self.EXPECTED_NOISE_KEYS <= set(metrics["noise"].keys())

    def test_contrast_keys(self, dashboard):
        metrics = dashboard.metrics
        assert self.EXPECTED_CONTRAST_KEYS <= set(metrics["contrast"].keys())

    def test_structure_keys(self, dashboard):
        metrics = dashboard.metrics
        assert self.EXPECTED_STRUCTURE_KEYS <= set(metrics["structure"].keys())

    def test_background_keys(self, dashboard):
        metrics = dashboard.metrics
        assert self.EXPECTED_BACKGROUND_KEYS <= set(metrics["background"].keys())

    def test_quality_scores_keys(self, dashboard):
        metrics = dashboard.metrics
        assert self.EXPECTED_QUALITY_KEYS == set(metrics["quality_scores"].keys())

    def test_interpretations_are_strings(self, dashboard):
        metrics = dashboard.metrics
        for section in ("noise", "contrast", "structure", "background"):
            assert isinstance(metrics["interpretations"][section], str)
            assert len(metrics["interpretations"][section]) > 0

    def test_recommendations_are_list_of_strings(self, dashboard):
        metrics = dashboard.metrics
        assert isinstance(metrics["recommendations"], list)
        for rec in metrics["recommendations"]:
            assert isinstance(rec, str)


class TestMetricsMatchOriginal:
    """Verify noise/contrast metrics match between dashboard and direct computation."""

    def test_noise_metrics_match(self, sample_image):
        dashboard, metrics = sample_image.plot.diagnostics()
        # Compute directly via plotter for comparison
        detect_mat = sample_image.detect_mat[:]
        direct_noise = sample_image.plot._compute_noise_metrics(detect_mat)

        assert metrics["noise"]["snr"] == pytest.approx(direct_noise["snr"])
        assert metrics["noise"]["sigma_mad"] == pytest.approx(direct_noise["sigma_mad"])

    def test_contrast_metrics_match(self, sample_image):
        dashboard, metrics = sample_image.plot.diagnostics()
        detect_mat = sample_image.detect_mat[:]
        direct_contrast = sample_image.plot._compute_contrast_metrics(detect_mat)

        assert metrics["contrast"]["rms_contrast"] == pytest.approx(
            direct_contrast["rms_contrast"]
        )
        assert metrics["contrast"]["michelson"] == pytest.approx(
            direct_contrast["michelson"]
        )


class TestParameterChangeUpdatesMetrics:
    """Test that changing interactive params changes parameter-dependent metrics."""

    def test_structure_sigma_change(self, dashboard):
        metrics_before = dashboard.metrics
        dashboard.structure_sigma = 5.0
        metrics_after = dashboard.metrics

        # Coherence changes with sigma
        assert (
            metrics_before["structure"]["mean_coherence"]
            != metrics_after["structure"]["mean_coherence"]
        )

    def test_background_sigma_change(self, dashboard):
        metrics_before = dashboard.metrics
        dashboard.background_sigma = 150.0
        metrics_after = dashboard.metrics

        assert (
            metrics_before["background"]["nonuniformity_ratio"]
            != metrics_after["background"]["nonuniformity_ratio"]
        )

    def test_ridge_method_change(self, dashboard):
        dashboard.ridge_method = "meijering"
        metrics_meijering = dashboard.metrics

        dashboard.ridge_method = "frangi"
        metrics_frangi = dashboard.metrics

        # Different methods produce different responses
        assert (
            metrics_meijering["structure"]["ridge_responses"]
            != metrics_frangi["structure"]["ridge_responses"]
        )

    def test_noise_metrics_stable_across_param_changes(self, dashboard):
        """Noise metrics are parameter-free and should stay constant."""
        noise_before = dashboard.metrics["noise"].copy()
        dashboard.structure_sigma = 8.0
        dashboard.background_sigma = 100.0
        noise_after = dashboard.metrics["noise"]

        assert noise_before["snr"] == pytest.approx(noise_after["snr"])
        assert noise_before["sigma_mad"] == pytest.approx(noise_after["sigma_mad"])


class TestSectionToggles:
    """Test section toggle parameters."""

    def test_show_noise_toggle(self, dashboard):
        dashboard.show_noise = True
        section = dashboard._noise_section()
        assert isinstance(section, pn.Card)

        dashboard.show_noise = False
        section = dashboard._noise_section()
        assert isinstance(section, pn.Column)
        assert len(section) == 0

    def test_show_contrast_toggle(self, dashboard):
        dashboard.show_contrast = True
        section = dashboard._contrast_section()
        assert isinstance(section, pn.Card)

        dashboard.show_contrast = False
        section = dashboard._contrast_section()
        assert isinstance(section, pn.Column)
        assert len(section) == 0

    def test_show_structure_toggle(self, dashboard):
        dashboard.show_structure = True
        section = dashboard._structure_section()
        assert isinstance(section, pn.Card)

        dashboard.show_structure = False
        section = dashboard._structure_section()
        assert isinstance(section, pn.Column)
        assert len(section) == 0

    def test_show_background_toggle(self, dashboard):
        dashboard.show_background = True
        section = dashboard._background_section()
        assert isinstance(section, pn.Card)

        dashboard.show_background = False
        section = dashboard._background_section()
        assert isinstance(section, pn.Column)
        assert len(section) == 0
