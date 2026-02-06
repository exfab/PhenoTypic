"""Tests for the ImageMetricsCalculator module."""

import numpy as np
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.util.image_metrics import (
    BackgroundMetrics,
    ContrastMetrics,
    ImageMetricsCalculator,
    NoiseMetrics,
    QualityScores,
    StructureMetrics,
    THRESHOLDS,
)


@pytest.fixture(scope="module")
def sample_image():
    """Load a sample image once for the module."""
    return load_synth_yeast_plate()


@pytest.fixture(scope="module")
def calculator(sample_image):
    """Create an ImageMetricsCalculator from the sample image."""
    return ImageMetricsCalculator(sample_image.detect_mat[:])


@pytest.fixture
def synthetic_uniform_image():
    """Create a uniform synthetic image for edge case testing."""
    return np.ones((100, 100), dtype=np.uint8) * 128


@pytest.fixture
def synthetic_high_contrast_image():
    """Create a high contrast checkerboard pattern."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    return img


class TestImageMetricsCalculatorInit:
    """Test ImageMetricsCalculator initialization."""

    def test_creates_calculator(self, calculator):
        assert isinstance(calculator, ImageMetricsCalculator)

    def test_bit_depth_8bit(self, calculator):
        # Sample image is 8-bit
        assert calculator.bit_depth == 8

    def test_bit_depth_16bit(self):
        img_16bit = np.ones((100, 100), dtype=np.uint16) * 1000
        calc = ImageMetricsCalculator(img_16bit)
        assert calc.bit_depth == 16

    def test_max_intensity_8bit(self, calculator):
        assert calculator.max_intensity == 255.0

    def test_max_intensity_16bit(self):
        img_16bit = np.ones((100, 100), dtype=np.uint16) * 1000
        calc = ImageMetricsCalculator(img_16bit)
        assert calc.max_intensity == 65535.0


class TestNoiseMetrics:
    """Test noise metric computation."""

    def test_returns_noise_metrics_type(self, calculator):
        metrics = calculator.compute_noise_metrics()
        # Should have all expected keys
        assert "snr" in metrics
        assert "sigma_mad" in metrics
        assert "correlation_length" in metrics

    def test_snr_positive(self, calculator):
        metrics = calculator.compute_noise_metrics()
        assert metrics["snr"] > 0

    def test_sigma_mad_positive(self, calculator):
        metrics = calculator.compute_noise_metrics()
        assert metrics["sigma_mad"] > 0

    def test_correlation_length_positive(self, calculator):
        metrics = calculator.compute_noise_metrics()
        assert metrics["correlation_length"] > 0

    def test_uniform_image_high_snr(self, synthetic_uniform_image):
        """Uniform image should have very high SNR (low noise)."""
        calc = ImageMetricsCalculator(synthetic_uniform_image)
        metrics = calc.compute_noise_metrics()
        # Uniform images have very little noise variance
        assert metrics["snr"] > 10  # High SNR for uniform image


class TestContrastMetrics:
    """Test contrast metric computation."""

    def test_returns_contrast_metrics_type(self, calculator):
        metrics = calculator.compute_contrast_metrics()
        assert "rms_contrast" in metrics
        assert "michelson" in metrics
        assert "dynamic_range" in metrics
        assert "p1" in metrics
        assert "p99" in metrics

    def test_rms_contrast_range(self, calculator):
        metrics = calculator.compute_contrast_metrics()
        assert 0 <= metrics["rms_contrast"]

    def test_michelson_range(self, calculator):
        metrics = calculator.compute_contrast_metrics()
        assert 0 <= metrics["michelson"] <= 1

    def test_dynamic_range_range(self, calculator):
        metrics = calculator.compute_contrast_metrics()
        assert 0 <= metrics["dynamic_range"] <= 1

    def test_percentiles_ordered(self, calculator):
        metrics = calculator.compute_contrast_metrics()
        assert metrics["p1"] <= metrics["p99"]

    def test_high_contrast_image_high_michelson(self, synthetic_high_contrast_image):
        """High contrast checkerboard should have high Michelson contrast."""
        calc = ImageMetricsCalculator(synthetic_high_contrast_image)
        metrics = calc.compute_contrast_metrics()
        assert metrics["michelson"] > 0.9  # Near 1.0 for black/white

    def test_uniform_image_low_contrast(self, synthetic_uniform_image):
        """Uniform image should have very low contrast."""
        calc = ImageMetricsCalculator(synthetic_uniform_image)
        metrics = calc.compute_contrast_metrics()
        assert metrics["rms_contrast"] < 0.01


class TestStructureMetrics:
    """Test structure metric computation."""

    def test_returns_structure_metrics_type(self, calculator):
        metrics = calculator.compute_structure_metrics()
        assert "mean_coherence" in metrics
        assert "optimal_scale" in metrics
        assert "peak_response" in metrics
        assert "ridge_responses" in metrics
        assert "scales" in metrics
        assert "ridge_method" in metrics
        assert "coherence_map" in metrics

    def test_coherence_range(self, calculator):
        metrics = calculator.compute_structure_metrics()
        assert 0 <= metrics["mean_coherence"] <= 1

    def test_optimal_scale_in_scales(self, calculator):
        metrics = calculator.compute_structure_metrics()
        assert metrics["optimal_scale"] in metrics["scales"]

    def test_ridge_responses_count_matches_scales(self, calculator):
        metrics = calculator.compute_structure_metrics()
        assert len(metrics["ridge_responses"]) == len(metrics["scales"])

    def test_coherence_map_is_array(self, calculator):
        metrics = calculator.compute_structure_metrics()
        assert isinstance(metrics["coherence_map"], np.ndarray)

    def test_custom_scales(self, calculator):
        custom_scales = [1.0, 2.0, 4.0]
        metrics = calculator.compute_structure_metrics(scales=custom_scales)
        assert metrics["scales"] == custom_scales
        assert len(metrics["ridge_responses"]) == 3

    def test_different_ridge_methods(self, calculator):
        for method in ["meijering", "frangi", "hessian"]:
            metrics = calculator.compute_structure_metrics(ridge_method=method)
            assert metrics["ridge_method"] == method

    def test_sigma_affects_coherence(self, calculator):
        metrics_low_sigma = calculator.compute_structure_metrics(sigma=0.5)
        metrics_high_sigma = calculator.compute_structure_metrics(sigma=5.0)
        # Different sigmas should produce different coherence values
        assert metrics_low_sigma["mean_coherence"] != metrics_high_sigma["mean_coherence"]


class TestBackgroundMetrics:
    """Test background metric computation."""

    def test_returns_background_metrics_type(self, calculator):
        metrics = calculator.compute_background_metrics()
        assert "nonuniformity_ratio" in metrics
        assert "mean_gradient" in metrics
        assert "background_estimate" in metrics

    def test_nonuniformity_non_negative(self, calculator):
        metrics = calculator.compute_background_metrics()
        assert metrics["nonuniformity_ratio"] >= 0

    def test_mean_gradient_non_negative(self, calculator):
        metrics = calculator.compute_background_metrics()
        assert metrics["mean_gradient"] >= 0

    def test_background_estimate_is_array(self, calculator):
        metrics = calculator.compute_background_metrics()
        assert isinstance(metrics["background_estimate"], np.ndarray)

    def test_background_estimate_same_shape(self, calculator):
        metrics = calculator.compute_background_metrics()
        assert metrics["background_estimate"].shape == calculator._detect_mat.shape

    def test_uniform_image_low_nonuniformity(self, synthetic_uniform_image):
        """Uniform image should have very low nonuniformity."""
        calc = ImageMetricsCalculator(synthetic_uniform_image)
        metrics = calc.compute_background_metrics()
        assert metrics["nonuniformity_ratio"] < 0.01


class TestQualityScores:
    """Test quality score computation."""

    def test_returns_quality_scores_type(self, calculator):
        noise = calculator.compute_noise_metrics()
        contrast = calculator.compute_contrast_metrics()
        structure = calculator.compute_structure_metrics()
        background = calculator.compute_background_metrics()

        scores = calculator.compute_quality_scores(noise, contrast, structure, background)

        assert "SNR" in scores
        assert "Contrast" in scores
        assert "Coherence" in scores
        assert "Uniformity" in scores
        assert "Sharpness" in scores

    def test_scores_normalized_to_01(self, calculator):
        noise = calculator.compute_noise_metrics()
        contrast = calculator.compute_contrast_metrics()
        structure = calculator.compute_structure_metrics()
        background = calculator.compute_background_metrics()

        scores = calculator.compute_quality_scores(noise, contrast, structure, background)

        for key, val in scores.items():
            assert 0 <= val <= 1, f"{key} score {val} not in [0, 1]"


class TestInterpretation:
    """Test interpretation text generation."""

    def test_noise_interpretation(self, calculator):
        noise = calculator.compute_noise_metrics()
        text = calculator.generate_interpretation("noise", noise)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "SNR" in text

    def test_contrast_interpretation(self, calculator):
        contrast = calculator.compute_contrast_metrics()
        text = calculator.generate_interpretation("contrast", contrast)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "contrast" in text.lower()

    def test_structure_interpretation(self, calculator):
        structure = calculator.compute_structure_metrics()
        text = calculator.generate_interpretation("structure", structure)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "coherence" in text.lower()

    def test_background_interpretation(self, calculator):
        background = calculator.compute_background_metrics()
        text = calculator.generate_interpretation("background", background)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "background" in text.lower() or "uniformity" in text.lower()


class TestRecommendations:
    """Test recommendation generation."""

    def test_returns_list(self, calculator):
        noise = calculator.compute_noise_metrics()
        contrast = calculator.compute_contrast_metrics()
        structure = calculator.compute_structure_metrics()
        background = calculator.compute_background_metrics()

        recs = calculator.generate_recommendations(noise, contrast, structure, background)
        assert isinstance(recs, list)

    def test_recommendations_are_strings(self, calculator):
        noise = calculator.compute_noise_metrics()
        contrast = calculator.compute_contrast_metrics()
        structure = calculator.compute_structure_metrics()
        background = calculator.compute_background_metrics()

        recs = calculator.generate_recommendations(noise, contrast, structure, background)
        for rec in recs:
            assert isinstance(rec, str)

    def test_always_includes_scale_recommendation(self, calculator):
        """Should always include optimal scale recommendation."""
        noise = calculator.compute_noise_metrics()
        contrast = calculator.compute_contrast_metrics()
        structure = calculator.compute_structure_metrics()
        background = calculator.compute_background_metrics()

        recs = calculator.generate_recommendations(noise, contrast, structure, background)
        scale_rec = [r for r in recs if "sigma_range" in r]
        assert len(scale_rec) >= 1


class TestComputeAll:
    """Test the convenience compute_all method."""

    def test_returns_dict(self, calculator):
        result = calculator.compute_all()
        assert isinstance(result, dict)

    def test_has_all_keys(self, calculator):
        result = calculator.compute_all()
        expected_keys = {
            "bit_depth",
            "noise",
            "contrast",
            "structure",
            "background",
            "quality_scores",
            "interpretations",
            "recommendations",
        }
        assert expected_keys == set(result.keys())

    def test_excludes_non_serializable_by_default(self, calculator):
        result = calculator.compute_all()
        assert "coherence_map" not in result["structure"]
        assert "background_estimate" not in result["background"]

    def test_includes_non_serializable_when_requested(self, calculator):
        result = calculator.compute_all(include_non_serializable=True)
        assert "coherence_map" in result["structure"]
        assert "background_estimate" in result["background"]
        assert isinstance(result["structure"]["coherence_map"], np.ndarray)

    def test_custom_parameters(self, calculator):
        result = calculator.compute_all(
            structure_sigma=3.0,
            ridge_method="frangi",
            ridge_scales=[1.0, 2.0, 3.0],
            background_sigma=100.0,
        )
        # Verify structure reflects custom ridge method
        # (scales are cleaned out of non-serializable output but method is kept)
        assert result["structure"]["ridge_method"] == "frangi"


class TestThresholds:
    """Test threshold constants."""

    def test_thresholds_exist(self):
        assert "snr" in THRESHOLDS
        assert "rms_contrast" in THRESHOLDS
        assert "coherence" in THRESHOLDS
        assert "nonuniformity" in THRESHOLDS

    def test_thresholds_have_critical_and_marginal(self):
        for key in THRESHOLDS:
            assert "critical" in THRESHOLDS[key]
            assert "marginal" in THRESHOLDS[key]


class TestHelperMethods:
    """Test helper methods for visualization data."""

    def test_compute_psd_returns_tuple(self, calculator):
        freqs, psd = calculator.compute_psd()
        assert isinstance(freqs, np.ndarray)
        assert isinstance(psd, np.ndarray)
        assert len(freqs) == len(psd)

    def test_compute_local_contrast_returns_array(self, calculator):
        contrast = calculator.compute_local_contrast()
        assert isinstance(contrast, np.ndarray)
        assert contrast.shape == calculator._detect_mat.shape

    def test_compute_local_variance_returns_array(self, calculator):
        variance = calculator.compute_local_variance()
        assert isinstance(variance, np.ndarray)
        assert variance.shape == calculator._detect_mat.shape

    def test_compute_local_variance_non_negative(self, calculator):
        variance = calculator.compute_local_variance()
        assert np.all(variance >= 0)


class TestMetricsConsistency:
    """Test that metrics are consistent across calls."""

    def test_noise_metrics_deterministic(self, calculator):
        metrics1 = calculator.compute_noise_metrics()
        metrics2 = calculator.compute_noise_metrics()
        assert metrics1["snr"] == pytest.approx(metrics2["snr"])
        assert metrics1["sigma_mad"] == pytest.approx(metrics2["sigma_mad"])

    def test_contrast_metrics_deterministic(self, calculator):
        metrics1 = calculator.compute_contrast_metrics()
        metrics2 = calculator.compute_contrast_metrics()
        assert metrics1["rms_contrast"] == pytest.approx(metrics2["rms_contrast"])

    def test_structure_metrics_deterministic(self, calculator):
        metrics1 = calculator.compute_structure_metrics(sigma=1.5)
        metrics2 = calculator.compute_structure_metrics(sigma=1.5)
        assert metrics1["mean_coherence"] == pytest.approx(metrics2["mean_coherence"])

    def test_background_metrics_deterministic(self, calculator):
        metrics1 = calculator.compute_background_metrics(sigma=50.0)
        metrics2 = calculator.compute_background_metrics(sigma=50.0)
        assert metrics1["nonuniformity_ratio"] == pytest.approx(
            metrics2["nonuniformity_ratio"]
        )
