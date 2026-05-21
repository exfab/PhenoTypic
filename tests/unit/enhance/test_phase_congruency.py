"""
Tests for PhaseCongruencyEnhancer.

Tests parameter validation, output properties, and basic functionality.
"""

import pytest
import numpy as np

from phenotypic import Image
from phenotypic.enhance import PhaseCongruencyEnhancer


class TestPhaseCongruencyEnhancerParameterValidation:
    """Test PhaseCongruencyEnhancer parameter validation."""

    def test_n_scale_less_than_one_raises_error(self):
        """Test that n_scale < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_scale must be >= 1"):
            PhaseCongruencyEnhancer(n_scale=0)

    def test_n_orient_less_than_one_raises_error(self):
        """Test that n_orient < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_orient must be >= 1"):
            PhaseCongruencyEnhancer(n_orient=0)

    def test_min_wavelength_less_than_two_raises_error(self):
        """Test that min_wavelength < 2 raises ValueError."""
        with pytest.raises(ValueError, match="min_wavelength must be >= 2"):
            PhaseCongruencyEnhancer(min_wavelength=1.5)

    def test_mult_less_than_or_equal_one_raises_error(self):
        """Test that mult <= 1 raises ValueError."""
        with pytest.raises(ValueError, match="mult must be > 1"):
            PhaseCongruencyEnhancer(mult=1.0)
        with pytest.raises(ValueError, match="mult must be > 1"):
            PhaseCongruencyEnhancer(mult=0.5)

    def test_sigma_onf_out_of_range_raises_error(self):
        """Test that sigma_onf outside [0.1, 1.0] raises ValueError."""
        with pytest.raises(ValueError, match="sigma_onf must be in"):
            PhaseCongruencyEnhancer(sigma_onf=0.05)
        with pytest.raises(ValueError, match="sigma_onf must be in"):
            PhaseCongruencyEnhancer(sigma_onf=1.5)

    def test_negative_k_raises_error(self):
        """Test that k < 0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 0"):
            PhaseCongruencyEnhancer(k=-1.0)

    def test_cutoff_out_of_range_raises_error(self):
        """Test that cutoff outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="cutoff must be in"):
            PhaseCongruencyEnhancer(cutoff=0.0)
        with pytest.raises(ValueError, match="cutoff must be in"):
            PhaseCongruencyEnhancer(cutoff=1.0)

    def test_g_non_positive_raises_error(self):
        """Test that g <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="g must be > 0"):
            PhaseCongruencyEnhancer(g=0.0)
        with pytest.raises(ValueError, match="g must be > 0"):
            PhaseCongruencyEnhancer(g=-5.0)

    def test_invalid_output_raises_error(self):
        """Test that invalid output mode raises ValueError.

        ``output`` is a ``Literal`` field, so an out-of-set value is
        rejected by pydantic with a ``literal_error`` (a subclass of
        ``ValueError``) rather than the legacy hand-rolled message.
        """
        with pytest.raises(ValueError, match="Input should be"):
            PhaseCongruencyEnhancer(output="invalid")

    def test_valid_parameters_accepted(self):
        """Test that valid parameters are accepted."""
        enhancer = PhaseCongruencyEnhancer(
            n_scale=4,
            n_orient=6,
            min_wavelength=3.0,
            mult=2.1,
            sigma_onf=0.55,
            k=2.0,
            cutoff=0.5,
            g=10.0,
            noise_method=-1,
            output="pc_sum",
        )
        assert enhancer.n_scale == 4
        assert enhancer.n_orient == 6
        assert enhancer.min_wavelength == 3.0
        assert enhancer.mult == 2.1
        assert enhancer.sigma_onf == 0.55
        assert enhancer.k == 2.0
        assert enhancer.cutoff == 0.5
        assert enhancer.g == 10.0
        assert enhancer.noise_method == -1
        assert enhancer.output == "pc_sum"


class TestPhaseCongruencyEnhancerOutputProperties:
    """Test PhaseCongruencyEnhancer output properties."""

    @pytest.fixture
    def synthetic_image(self):
        """Create a synthetic test image with edges."""
        # Create 128x128 image with a vertical edge in the middle
        arr = np.zeros((128, 128), dtype=np.float64)
        arr[:, 64:] = 1.0  # Right half is bright
        return Image(arr=arr)

    @pytest.fixture
    def uniform_image(self):
        """Create a uniform (featureless) test image."""
        arr = np.ones((64, 64), dtype=np.float64) * 0.5
        return Image(arr=arr)

    def test_output_shape_preserved(self, synthetic_image):
        """Test that output has same shape as input."""
        enhancer = PhaseCongruencyEnhancer(n_scale=3, n_orient=4)
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_output_range_clipped_to_unit_interval(self, synthetic_image):
        """Test that output is in [0, 1] range."""
        enhancer = PhaseCongruencyEnhancer()
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0

    def test_pc_sum_output_mode(self, synthetic_image):
        """Test pc_sum output mode works."""
        enhancer = PhaseCongruencyEnhancer(output="pc_sum")
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_M_output_mode(self, synthetic_image):
        """Test M (edge strength) output mode works."""
        enhancer = PhaseCongruencyEnhancer(output="M")
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_m_output_mode(self, synthetic_image):
        """Test m (corner strength) output mode works."""
        enhancer = PhaseCongruencyEnhancer(output="m")
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_uniform_image_low_response(self, uniform_image):
        """Test that uniform image produces low phase congruency."""
        enhancer = PhaseCongruencyEnhancer(n_scale=3, n_orient=4)
        result = enhancer.apply(uniform_image)
        # Uniform regions should have low PC values
        assert result.detect_mat[:].mean() < 0.3


class TestPhaseCongruencyEnhancerEdgeDetection:
    """Test PhaseCongruencyEnhancer edge detection capabilities."""

    def test_vertical_edge_detected(self):
        """Test that vertical edges are detected with high M values."""
        # Create image with sharp vertical edge
        arr = np.zeros((128, 128), dtype=np.float64)
        arr[:, 64:] = 1.0
        image = Image(arr=arr)

        enhancer = PhaseCongruencyEnhancer(output="M", n_scale=3, n_orient=4)
        result = enhancer.apply(image)

        # Edge region (columns ~60-68) should have higher values than uniform regions
        edge_region = result.detect_mat[:, 60:68]
        left_uniform = result.detect_mat[:, 10:30]
        right_uniform = result.detect_mat[:, 90:110]

        assert edge_region.mean() > left_uniform.mean()
        assert edge_region.mean() > right_uniform.mean()

    def test_horizontal_edge_detected(self):
        """Test that horizontal edges are detected with high M values."""
        # Create image with sharp horizontal edge
        arr = np.zeros((128, 128), dtype=np.float64)
        arr[64:, :] = 1.0
        image = Image(arr=arr)

        enhancer = PhaseCongruencyEnhancer(output="M", n_scale=3, n_orient=4)
        result = enhancer.apply(image)

        # Edge region (rows ~60-68) should have higher values than uniform regions
        edge_region = result.detect_mat[60:68, :]
        top_uniform = result.detect_mat[10:30, :]
        bottom_uniform = result.detect_mat[90:110, :]

        assert edge_region.mean() > top_uniform.mean()
        assert edge_region.mean() > bottom_uniform.mean()


class TestPhaseCongruencyEnhancerNoiseHandling:
    """Test noise estimation methods."""

    @pytest.fixture
    def noisy_image(self):
        """Create image with step edge and noise."""
        arr = np.zeros((128, 128), dtype=np.float64)
        arr[:, 64:] = 0.8
        # Add some noise
        np.random.seed(42)
        arr += np.random.normal(0, 0.05, arr.shape)
        arr = np.clip(arr, 0, 1)
        return Image(arr=arr)

    def test_noise_method_median(self, noisy_image):
        """Test median noise estimation method (-1)."""
        enhancer = PhaseCongruencyEnhancer(noise_method=-1, n_scale=3, n_orient=4)
        result = enhancer.apply(noisy_image)
        assert result.detect_mat[:].shape == noisy_image.detect_mat[:].shape

    def test_noise_method_mode(self, noisy_image):
        """Test mode noise estimation method (-2)."""
        enhancer = PhaseCongruencyEnhancer(noise_method=-2, n_scale=3, n_orient=4)
        result = enhancer.apply(noisy_image)
        assert result.detect_mat[:].shape == noisy_image.detect_mat[:].shape

    def test_noise_method_fixed(self, noisy_image):
        """Test fixed noise threshold (>= 0)."""
        enhancer = PhaseCongruencyEnhancer(noise_method=0.1, n_scale=3, n_orient=4)
        result = enhancer.apply(noisy_image)
        assert result.detect_mat[:].shape == noisy_image.detect_mat[:].shape

    def test_higher_k_reduces_response(self, noisy_image):
        """Test that higher k (more noise rejection) reduces overall response."""
        enhancer_low_k = PhaseCongruencyEnhancer(k=2.0, n_scale=3, n_orient=4)
        enhancer_high_k = PhaseCongruencyEnhancer(k=10.0, n_scale=3, n_orient=4)

        result_low_k = enhancer_low_k.apply(noisy_image)
        result_high_k = enhancer_high_k.apply(noisy_image)

        # Higher k should produce lower overall response (more aggressive thresholding)
        assert result_high_k.detect_mat[:].mean() <= result_low_k.detect_mat[:].mean()


class TestPhaseCongruencyEnhancerIntegration:
    """Integration tests with phenotypic data and Image class."""

    def test_apply_preserves_image_rgb(self):
        """Test that apply() does not modify image.rgb (immutability)."""
        arr = np.random.rand(64, 64, 3).astype(np.float64)
        image = Image(arr=arr)
        original_rgb = image.rgb[:].copy()

        enhancer = PhaseCongruencyEnhancer(n_scale=3, n_orient=4)
        enhancer.apply(image)

        # Original image rgb should be unchanged
        assert np.array_equal(image.rgb[:], original_rgb)

    def test_apply_preserves_image_gray(self):
        """Test that apply() does not modify image.gray (immutability)."""
        arr = np.random.rand(64, 64).astype(np.float64)
        image = Image(arr=arr)
        original_gray = image.gray[:].copy()

        enhancer = PhaseCongruencyEnhancer(n_scale=3, n_orient=4)
        enhancer.apply(image)

        # Original image gray should be unchanged
        assert np.array_equal(image.gray[:], original_gray)

    def test_inplace_modifies_original(self):
        """Test that inplace=True modifies the original image."""
        arr = np.random.rand(64, 64).astype(np.float64)
        image = Image(arr=arr)
        original_detect_mat = image.detect_mat[:].copy()

        enhancer = PhaseCongruencyEnhancer(n_scale=3, n_orient=4)
        result = enhancer.apply(image, inplace=True)

        # Result should be the same object
        assert result is image
        # detect_mat should be modified (not equal to original)
        assert not np.array_equal(image.detect_mat[:], original_detect_mat)


# Run all tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
