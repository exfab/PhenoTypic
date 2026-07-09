"""
Tests for FocusEdgePhase.

Tests parameter validation, output properties, and basic functionality.
"""

import pytest
import numpy as np
from pydantic import ValidationError

from phenotypic import Image
from phenotypic.enhance import FocusEdgePhase


class TestPhaseCongruencyEnhancerParameterValidation:
    """Test FocusEdgePhase parameter validation.

    The bare-scalar bounds migrated from ``field_validator``s to
    ``Field(ge=, le=, gt=, lt=)`` (the annotations workstream), so these assert
    on the ``ValidationError`` *type* (a ``ValueError`` subclass) rather than the
    old hand-rolled messages — the rejection contract is unchanged.
    """

    def test_n_scale_zero_is_rejected(self):
        """Test that n_scale=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(n_scale=0)

    def test_n_scale_one_is_rejected(self):
        """n_scale=1 divides by (n_scale - 1) and returns an all-zero detect_mat
        (max=0 versus 0.971004 at n_scale=4). Rejected at construction rather
        than producing garbage. See drift-register M3."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(n_scale=1)

    def test_n_scale_two_is_accepted(self):
        assert FocusEdgePhase(n_scale=2).n_scale == 2

    def test_n_orient_less_than_one_raises_error(self):
        """Test that n_orient < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(n_orient=0)

    def test_min_wavelength_less_than_two_raises_error(self):
        """Test that min_wavelength < 2 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(min_wavelength=1.5)

    def test_mult_less_than_or_equal_one_raises_error(self):
        """Test that mult <= 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(mult=1.0)
        with pytest.raises(ValidationError):
            FocusEdgePhase(mult=0.5)

    def test_sigma_onf_out_of_range_raises_error(self):
        """Test that sigma_onf outside [0.1, 1.0] raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(sigma_onf=0.05)
        with pytest.raises(ValidationError):
            FocusEdgePhase(sigma_onf=1.5)

    def test_negative_k_raises_error(self):
        """Test that k < 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(k=-1.0)

    def test_cutoff_out_of_range_raises_error(self):
        """Test that cutoff outside (0, 1) raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(cutoff=0.0)
        with pytest.raises(ValidationError):
            FocusEdgePhase(cutoff=1.0)

    def test_g_non_positive_raises_error(self):
        """Test that g <= 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(g=0.0)
        with pytest.raises(ValidationError):
            FocusEdgePhase(g=-5.0)

    def test_invalid_output_raises_error(self):
        """Test that invalid output mode raises ValueError.

        ``output`` is a ``Literal`` field, so an out-of-set value is
        rejected by pydantic with a ``literal_error`` (a subclass of
        ``ValueError``) rather than the legacy hand-rolled message.
        """
        with pytest.raises(ValueError, match="Input should be"):
            FocusEdgePhase(output="invalid")

    def test_valid_parameters_accepted(self):
        """Test that valid parameters are accepted."""
        enhancer = FocusEdgePhase(
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


class TestTheEpsilonSeamIsLocked:
    """`_phasecong3` must hand `spread_weight` phasecong3's 1e-5, not the module's 1e-4.

    Before the kernels refactor, `epsilon = 1e-5` was a local literal inside `_phasecong3`.
    It is now an argument to a shared function whose module constant `EPSILON_MONOGENIC` is
    `1e-4` -- `phasecongmono`'s value, not `phasecong3`'s. That is a seam, and nothing in
    the repository locked it: substituting `1e-4` leaves `tests/unit/enhance` and the
    filamentous detector's suite entirely green while shifting `pc_sum` by 7.48%
    (`max|d| = 6.026e-02`, 469165 / 480000 pixels changed on `load_synth_yeast_plate`).
    """

    def test_phasecong3_passes_phasecong3s_epsilon_to_spread_weight(self, monkeypatch):
        """Capture the value at the call boundary rather than inferring it from output."""
        import phenotypic.enhance._focus_edge_phase as fep

        seen: list[float] = []
        real = fep.spread_weight

        def spy(sum_amplitude, max_amplitude, n_scale, cutoff, g, epsilon):
            seen.append(epsilon)
            return real(sum_amplitude, max_amplitude, n_scale, cutoff, g, epsilon)

        monkeypatch.setattr(fep, "spread_weight", spy)
        FocusEdgePhase(n_scale=2, n_orient=2)._phasecong3(np.zeros((32, 32)))

        assert seen, "spread_weight was never called"
        assert set(seen) == {1e-5}, f"expected phasecong3's 1e-5, got {sorted(set(seen))}"

    def test_the_two_epsilons_are_actually_different(self):
        """Guard the guard: if these ever unify, the test above becomes vacuous."""
        from phenotypic.enhance._monogenic_kernels import EPSILON_MONOGENIC

        assert EPSILON_MONOGENIC == 1e-4
        assert EPSILON_MONOGENIC != 1e-5


class TestPhaseCongruencyEnhancerOutputProperties:
    """Test FocusEdgePhase output properties."""

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
        enhancer = FocusEdgePhase(n_scale=3, n_orient=4)
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_output_range_clipped_to_unit_interval(self, synthetic_image):
        """Test that output is in [0, 1] range."""
        enhancer = FocusEdgePhase()
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0

    def test_pc_sum_output_mode(self, synthetic_image):
        """Test pc_sum output mode works."""
        enhancer = FocusEdgePhase(output="pc_sum")
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_M_output_mode(self, synthetic_image):
        """Test M (edge strength) output mode works."""
        enhancer = FocusEdgePhase(output="M")
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_m_output_mode(self, synthetic_image):
        """Test m (corner strength) output mode works."""
        enhancer = FocusEdgePhase(output="m")
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_uniform_image_low_response(self, uniform_image):
        """Test that uniform image produces low phase congruency."""
        enhancer = FocusEdgePhase(n_scale=3, n_orient=4)
        result = enhancer.apply(uniform_image)
        # Uniform regions should have low PC values
        assert result.detect_mat[:].mean() < 0.3


class TestPhaseCongruencyEnhancerEdgeDetection:
    """Test FocusEdgePhase edge detection capabilities."""

    def test_vertical_edge_detected(self):
        """Test that vertical edges are detected with high M values."""
        # Create image with sharp vertical edge
        arr = np.zeros((128, 128), dtype=np.float64)
        arr[:, 64:] = 1.0
        image = Image(arr=arr)

        enhancer = FocusEdgePhase(output="M", n_scale=3, n_orient=4)
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

        enhancer = FocusEdgePhase(output="M", n_scale=3, n_orient=4)
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
        enhancer = FocusEdgePhase(noise_method=-1, n_scale=3, n_orient=4)
        result = enhancer.apply(noisy_image)
        assert result.detect_mat[:].shape == noisy_image.detect_mat[:].shape

    def test_noise_method_mode(self, noisy_image):
        """Test mode noise estimation method (-2)."""
        enhancer = FocusEdgePhase(noise_method=-2, n_scale=3, n_orient=4)
        result = enhancer.apply(noisy_image)
        assert result.detect_mat[:].shape == noisy_image.detect_mat[:].shape

    def test_noise_method_fixed(self, noisy_image):
        """Test fixed noise threshold (>= 0)."""
        enhancer = FocusEdgePhase(noise_method=0.1, n_scale=3, n_orient=4)
        result = enhancer.apply(noisy_image)
        assert result.detect_mat[:].shape == noisy_image.detect_mat[:].shape

    def test_higher_k_reduces_response(self, noisy_image):
        """Test that higher k (more noise rejection) reduces overall response."""
        enhancer_low_k = FocusEdgePhase(k=2.0, n_scale=3, n_orient=4)
        enhancer_high_k = FocusEdgePhase(k=10.0, n_scale=3, n_orient=4)

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

        enhancer = FocusEdgePhase(n_scale=3, n_orient=4)
        enhancer.apply(image)

        # Original image rgb should be unchanged
        assert np.array_equal(image.rgb[:], original_rgb)

    def test_apply_preserves_image_gray(self):
        """Test that apply() does not modify image.gray (immutability)."""
        arr = np.random.rand(64, 64).astype(np.float64)
        image = Image(arr=arr)
        original_gray = image.gray[:].copy()

        enhancer = FocusEdgePhase(n_scale=3, n_orient=4)
        enhancer.apply(image)

        # Original image gray should be unchanged
        assert np.array_equal(image.gray[:], original_gray)

    def test_inplace_modifies_original(self):
        """Test that inplace=True modifies the original image."""
        arr = np.random.rand(64, 64).astype(np.float64)
        image = Image(arr=arr)
        original_detect_mat = image.detect_mat[:].copy()

        enhancer = FocusEdgePhase(n_scale=3, n_orient=4)
        result = enhancer.apply(image, inplace=True)

        # Result should be the same object
        assert result is image
        # detect_mat should be modified (not equal to original)
        assert not np.array_equal(image.detect_mat[:], original_detect_mat)


# Run all tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
