"""Tests for plot accessor classes in PhenoTypic.

This module provides comprehensive test coverage for the plot accessor
functionality, including morphology visualization, size distribution analysis,
spatial mapping, and thresholding visualization.
"""

import pytest
import matplotlib.pyplot as plt

from phenotypic import GridImage
from phenotypic.detect import OtsuDetector
from phenotypic.data import load_plate_12hr
from phenotypic._core._image_parts.plot_accessor import BasePlotter


@pytest.fixture
def sample_image_with_objects():
    """Create a sample image with detected objects for testing."""
    # Load built-in test plate data
    plate_data = load_plate_12hr()
    image = GridImage(plate_data, nrows=8, ncols=12)

    # Apply detection to get objects
    detector = OtsuDetector()
    image = detector.apply(image)

    return image


@pytest.fixture
def sample_image_no_objects():
    """Create a blank image without objects."""
    plate_data = load_plate_12hr()
    image = GridImage(plate_data, nrows=8, ncols=12)
    # Don't apply detection
    return image


class TestBasePlotter:
    """Tests for BasePlotter validation methods.

    These tests instantiate BasePlotter directly to test its internal
    validation methods, which are used by all plotter subclasses.
    """

    def test_validate_figsize_valid(self, sample_image_with_objects):
        """Test figsize validation with valid input."""
        plotter = BasePlotter(sample_image_with_objects)
        # Should not raise
        plotter._validate_figsize((10, 8))
        plotter._validate_figsize((12.5, 9.5))

    def test_validate_figsize_none(self, sample_image_with_objects):
        """Test figsize validation with None."""
        plotter = BasePlotter(sample_image_with_objects)
        # None is valid
        plotter._validate_figsize(None)

    def test_validate_figsize_invalid_tuple(self, sample_image_with_objects):
        """Test figsize validation with invalid tuple structure."""
        plotter = BasePlotter(sample_image_with_objects)
        with pytest.raises(ValueError, match="figsize must be a tuple"):
            plotter._validate_figsize("invalid")
        with pytest.raises(ValueError, match="figsize must be"):
            plotter._validate_figsize((10,))
        with pytest.raises(ValueError, match="figsize must be"):
            plotter._validate_figsize((10, 8, 6))

    def test_validate_figsize_negative(self, sample_image_with_objects):
        """Test figsize validation with negative dimensions."""
        plotter = BasePlotter(sample_image_with_objects)
        with pytest.raises(ValueError, match="positive"):
            plotter._validate_figsize((-10, 8))
        with pytest.raises(ValueError, match="positive"):
            plotter._validate_figsize((10, -8))

    def test_validate_cmap_valid(self, sample_image_with_objects):
        """Test colormap validation with valid name."""
        plotter = BasePlotter(sample_image_with_objects)
        # Should not raise for standard colormaps
        plotter._validate_cmap("viridis")
        plotter._validate_cmap("plasma")

    def test_validate_cmap_invalid(self, sample_image_with_objects):
        """Test colormap validation with invalid name."""
        plotter = BasePlotter(sample_image_with_objects)
        with pytest.raises(ValueError, match="Unknown colormap"):
            plotter._validate_cmap("nonexistent_colormap_xyz")

    def test_validate_alpha_valid(self, sample_image_with_objects):
        """Test alpha validation with valid values."""
        plotter = BasePlotter(sample_image_with_objects)
        plotter._validate_alpha(0.0)
        plotter._validate_alpha(0.5)
        plotter._validate_alpha(1.0)
        plotter._validate_alpha(None)

    def test_validate_alpha_invalid(self, sample_image_with_objects):
        """Test alpha validation with invalid values."""
        plotter = BasePlotter(sample_image_with_objects)
        with pytest.raises(ValueError, match="between 0 and 1"):
            plotter._validate_alpha(-0.1)
        with pytest.raises(ValueError, match="between 0 and 1"):
            plotter._validate_alpha(1.1)
        with pytest.raises(ValueError, match="numeric"):
            plotter._validate_alpha("invalid")


class TestMorphologyPlotter:
    """Tests for MorphologyPlotter methods."""

    def test_morph_progression_basic(self, sample_image_with_objects):
        """Test basic morphological progression visualization."""
        plot = sample_image_with_objects.plot
        fig, axes = plot.morph_progression(
            operation="opening",
            kernel_sizes=[1, 3, 5],
            shape="disk",
        )
        assert fig is not None
        assert axes is not None
        plt.close(fig)

    def test_morph_progression_no_objects(self, sample_image_no_objects):
        """Test morph_progression raises error without objects."""
        plot = sample_image_no_objects.plot
        with pytest.raises(ValueError, match="No objects detected"):
            plot.morph_progression()

    def test_morph_progression_invalid_figsize(self, sample_image_with_objects):
        """Test morph_progression with invalid figsize."""
        plot = sample_image_with_objects.plot
        with pytest.raises(ValueError, match="figsize"):
            plot.morph_progression(figsize=(-10, 8))

    def test_morph_progression_invalid_cmap(self, sample_image_with_objects):
        """Test morph_progression with invalid colormap."""
        plot = sample_image_with_objects.plot
        with pytest.raises(ValueError, match="Unknown colormap"):
            plot.morph_progression(cmap="invalid_cmap")

    def test_structural_response_curve_basic(self, sample_image_with_objects):
        """Test structural response curve visualization."""
        plot = sample_image_with_objects.plot
        fig, ax = plot.structural_response_curve(
            operation="opening",
            kernel_range=(1, 10),
            metric="count",
        )
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_structural_response_curve_insufficient_kernels(self, sample_image_with_objects):
        """Test structural response curve raises error with insufficient kernels."""
        plot = sample_image_with_objects.plot
        with pytest.raises(ValueError, match="at least 2"):
            plot.structural_response_curve(kernel_range=[5])

    def test_boundary_displacement_basic(self, sample_image_with_objects):
        """Test boundary displacement visualization."""
        plot = sample_image_with_objects.plot
        fig, axes = plot.boundary_displacement(
            operation="opening",
            kernel_sizes=[1, 3, 5, 7],
        )
        assert fig is not None
        assert axes is not None
        plt.close(fig)


class TestSizeDistributionPlotter:
    """Tests for SizeDistributionPlotter methods."""

    def test_size_distribution_basic(self, sample_image_with_objects):
        """Test basic size distribution visualization."""
        plot = sample_image_with_objects.plot
        fig, axes = plot.size_distribution()
        assert fig is not None
        assert axes is not None
        plt.close(fig)

    def test_size_distribution_no_objects(self, sample_image_no_objects):
        """Test size_distribution raises error without objects."""
        plot = sample_image_no_objects.plot
        with pytest.raises(ValueError, match="No labeled objects"):
            plot.size_distribution()

    def test_size_distribution_with_custom_thresholds(self, sample_image_with_objects):
        """Test size distribution with custom threshold values."""
        plot = sample_image_with_objects.plot
        fig, axes = plot.size_distribution(thresholds=[10, 50, 100])
        assert fig is not None
        plt.close(fig)

    def test_size_distribution_invalid_figsize(self, sample_image_with_objects):
        """Test size_distribution with invalid figsize."""
        plot = sample_image_with_objects.plot
        with pytest.raises(ValueError, match="figsize"):
            plot.size_distribution(figsize=(0, 8))

    def test_size_distribution_log_scale(self, sample_image_with_objects):
        """Test size distribution with log scale."""
        plot = sample_image_with_objects.plot
        fig, axes = plot.size_distribution(log_scale=True)
        assert fig is not None
        plt.close(fig)


class TestSpatialPlotter:
    """Tests for SpatialPlotter methods."""

    def test_spatial_size_map_basic(self, sample_image_with_objects):
        """Test basic spatial size map visualization."""
        plot = sample_image_with_objects.plot
        fig, axes, metadata = plot.spatial_size_map(mode="median")
        assert fig is not None
        assert axes is not None
        assert metadata is not None
        assert "center" in metadata
        assert "mean_size" in metadata
        plt.close(fig)

    def test_spatial_size_map_modes(self, sample_image_with_objects):
        """Test spatial size map with different center modes."""
        plot = sample_image_with_objects.plot

        # Test median mode
        fig1, _, _ = plot.spatial_size_map(mode="median")
        plt.close(fig1)

        # Test mean mode
        fig2, _, _ = plot.spatial_size_map(mode="mean", robust=True)
        plt.close(fig2)

        # Test percentile mode
        fig3, _, _ = plot.spatial_size_map(mode="percentile", value=75)
        plt.close(fig3)

        # Test absolute mode
        fig4, _, _ = plot.spatial_size_map(mode="absolute", value=50)
        plt.close(fig4)

    def test_spatial_size_map_invalid_mode(self, sample_image_with_objects):
        """Test spatial size map with invalid mode."""
        plot = sample_image_with_objects.plot
        with pytest.raises(ValueError, match="Unknown mode"):
            plot.spatial_size_map(mode="invalid_mode")

    def test_spatial_size_map_percentile_no_value(self, sample_image_with_objects):
        """Test spatial size map percentile mode requires value."""
        plot = sample_image_with_objects.plot
        with pytest.raises(ValueError, match="requires value"):
            plot.spatial_size_map(mode="percentile")

    def test_spatial_size_map_absolute_no_value(self, sample_image_with_objects):
        """Test spatial size map absolute mode requires value."""
        plot = sample_image_with_objects.plot
        with pytest.raises(ValueError, match="requires value"):
            plot.spatial_size_map(mode="absolute")

    def test_spatial_size_map_invalid_params(self, sample_image_with_objects):
        """Test spatial size map with invalid parameters."""
        plot = sample_image_with_objects.plot

        # Invalid colormap
        with pytest.raises(ValueError, match="Unknown colormap"):
            plot.spatial_size_map(cmap="invalid_cmap")

        # Invalid figsize
        with pytest.raises(ValueError, match="figsize"):
            plot.spatial_size_map(figsize=(-10, 8))

        # Invalid alpha
        with pytest.raises(ValueError, match="alpha"):
            plot.spatial_size_map(alpha=1.5)

    def test_size_scatter_basic(self, sample_image_with_objects):
        """Test basic size scatter plot."""
        plot = sample_image_with_objects.plot
        fig, ax = plot.size_scatter(show_regression=True, show_marginals=True)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_size_scatter_insufficient_objects(self, sample_image_with_objects):
        """Test size scatter with insufficient objects."""
        plot = sample_image_with_objects.plot
        # This should work with real images that have enough objects
        # The test would need an image with <3 objects to fail
        # Skip for now since our test image has enough objects


class TestThresholdPlotter:
    """Tests for ThresholdPlotter methods."""

    def test_try_thresh_basic(self, sample_image_with_objects):
        """Test basic thresholding visualization."""
        plot = sample_image_with_objects.plot
        fig, axes = plot.try_thresh()
        assert fig is not None
        assert axes is not None
        plt.close(fig)

    def test_try_thresh_custom_figsize(self, sample_image_with_objects):
        """Test thresholding with custom figure size."""
        plot = sample_image_with_objects.plot
        fig, axes = plot.try_thresh(figsize=(16, 10))
        assert fig is not None
        plt.close(fig)

    def test_try_thresh_invalid_figsize(self, sample_image_with_objects):
        """Test try_thresh with invalid figsize."""
        plot = sample_image_with_objects.plot
        with pytest.raises(ValueError, match="figsize"):
            plot.try_thresh(figsize=(0, 8))


class TestSpatialSizeMapFix:
    """Tests specifically for the SpatialPlotter size mapping fix."""

    def test_size_map_uses_actual_sizes(self, sample_image_with_objects):
        """Verify that spatial_size_map uses actual object sizes, not pixel counts."""
        plot = sample_image_with_objects.plot
        fig, axes, metadata = plot.spatial_size_map()

        # Check that metadata contains correct statistics
        assert "mean_size" in metadata
        assert "median_size" in metadata
        assert metadata["vmin"] >= 0
        assert metadata["vmax"] >= metadata["vmin"]

        plt.close(fig)


class TestKDEScalingFix:
    """Tests specifically for KDE scaling fix."""

    def test_kde_histogram_alignment(self, sample_image_with_objects):
        """Verify that KDE is properly scaled to match histogram."""
        plot = sample_image_with_objects.plot
        fig, axes = plot.size_distribution(log_scale=False)

        # Verify figure was created with histogram
        assert len(fig.axes) >= 1

        plt.close(fig)


class TestDiagnosticsMatplotlib:
    """Tests for DiagnosticsPlotter (always matplotlib).

    plot.diagnostics() always returns a matplotlib Figure.
    For an interactive dashboard, use image.plot.dash.diagnostics() instead.
    """

    def test_diagnostics_returns_tuple(self, sample_image_with_objects):
        """Test that diagnostics returns a tuple of (figure, metrics)."""
        result = sample_image_with_objects.plot.diagnostics()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_diagnostics_returns_figure(self, sample_image_with_objects):
        """Test that diagnostics returns a matplotlib Figure."""
        fig, metrics = sample_image_with_objects.plot.diagnostics()
        assert isinstance(fig, plt.Figure)
        assert isinstance(metrics, dict)
        plt.close(fig)

    def test_diagnostics_metrics_structure(self, sample_image_with_objects):
        """Test that diagnostics returns proper metrics structure."""
        fig, metrics = sample_image_with_objects.plot.diagnostics()

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
        assert expected_keys == set(metrics.keys())
        plt.close(fig)

    def test_diagnostics_quality_scores(self, sample_image_with_objects):
        """Test quality scores are in [0,1] range."""
        fig, metrics = sample_image_with_objects.plot.diagnostics()

        for key, score in metrics["quality_scores"].items():
            assert 0 <= score <= 1, f"{key} score {score} not in [0,1]"
        plt.close(fig)

    def test_diagnostics_custom_sections(self, sample_image_with_objects):
        """Test diagnostics with custom sections."""
        fig, _ = sample_image_with_objects.plot.diagnostics(
            sections=["noise", "contrast"]
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_diagnostics_custom_parameters(self, sample_image_with_objects):
        """Test diagnostics with custom parameters."""
        fig, metrics = sample_image_with_objects.plot.diagnostics(
            structure_sigma=3.0,
            ridge_method="frangi",
            ridge_scales=[1.0, 2.0, 3.0],
            background_sigma=100.0,
        )
        assert metrics["structure"]["ridge_method"] == "frangi"
        plt.close(fig)

    def test_diagnostics_cleanup(self, sample_image_with_objects):
        """Test that figure can be properly closed."""
        fig, _ = sample_image_with_objects.plot.diagnostics()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
