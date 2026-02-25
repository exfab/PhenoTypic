"""Tests for CoherenceEnhancingDiffusion.

Tests parameter validation, default values, output invariants, anisotropic
diffusion correctness, isotropic region stability, pipeline integration,
and serialization roundtrip.
"""

import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.enhance import CoherenceEnhancingDiffusion


# -- Parameter validation ----------------------------------------------------


class TestParameterValidation:
    """Reject invalid constructor arguments."""

    def test_num_iter_zero_raises(self):
        with pytest.raises(ValueError, match="num_iter must be >= 1"):
            CoherenceEnhancingDiffusion(num_iter=0)

    def test_num_iter_negative_raises(self):
        with pytest.raises(ValueError, match="num_iter must be >= 1"):
            CoherenceEnhancingDiffusion(num_iter=-5)

    def test_dt_zero_raises(self):
        with pytest.raises(ValueError, match="dt must be > 0"):
            CoherenceEnhancingDiffusion(dt=0)

    def test_dt_negative_raises(self):
        with pytest.raises(ValueError, match="dt must be > 0"):
            CoherenceEnhancingDiffusion(dt=-0.1)

    def test_dt_too_large_raises(self):
        with pytest.raises(ValueError, match="dt > 0.125"):
            CoherenceEnhancingDiffusion(dt=0.15)

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            CoherenceEnhancingDiffusion(sigma=0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            CoherenceEnhancingDiffusion(sigma=-1.0)

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            CoherenceEnhancingDiffusion(alpha=0)

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            CoherenceEnhancingDiffusion(alpha=1.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            CoherenceEnhancingDiffusion(alpha=-0.1)

    def test_C_zero_raises(self):
        with pytest.raises(ValueError, match="C must be > 0"):
            CoherenceEnhancingDiffusion(C=0)

    def test_C_negative_raises(self):
        with pytest.raises(ValueError, match="C must be > 0"):
            CoherenceEnhancingDiffusion(C=-1.0)


# -- Default values ----------------------------------------------------------


class TestDefaults:
    """Verify default parameter values stored on instance."""

    def test_default_values(self):
        ced = CoherenceEnhancingDiffusion()
        assert ced.num_iterations == 20
        assert ced.sigma == 1.5
        assert ced.dt == 0.1
        assert ced.alpha == 0.001
        assert ced.C == 1.0

    def test_custom_values(self):
        ced = CoherenceEnhancingDiffusion(
            num_iter=10, sigma=2.0, dt=0.05, alpha=0.01, C=5.0,
        )
        assert ced.num_iterations == 10
        assert ced.sigma == 2.0
        assert ced.dt == 0.05
        assert ced.alpha == 0.01
        assert ced.C == 5.0


# -- Output shape/dtype preserved -------------------------------------------


class TestOutputInvariants:
    """Output detect_mat has same shape and dtype; rgb/gray unchanged."""

    @pytest.fixture
    def gray_image(self):
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float64) * 0.5 + 0.25
        return Image(arr=arr)

    @pytest.fixture
    def rgb_image(self):
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64, 3)).astype(np.float64)
        return Image(arr=arr)

    def test_shape_preserved(self, gray_image):
        ced = CoherenceEnhancingDiffusion(num_iter=3, sigma=1.0)
        result = ced.apply(gray_image)
        assert result.detect_mat[:].shape == gray_image.detect_mat[:].shape

    def test_dtype_preserved(self, gray_image):
        original_dtype = gray_image.detect_mat[:].dtype
        ced = CoherenceEnhancingDiffusion(num_iter=3, sigma=1.0)
        result = ced.apply(gray_image)
        assert result.detect_mat[:].dtype == original_dtype

    def test_rgb_immutability(self, rgb_image):
        original_rgb = rgb_image.rgb[:].copy()
        ced = CoherenceEnhancingDiffusion(num_iter=3, sigma=1.0)
        ced.apply(rgb_image)
        np.testing.assert_array_equal(rgb_image.rgb[:], original_rgb)

    def test_gray_immutability(self, gray_image):
        original_gray = gray_image.gray[:].copy()
        ced = CoherenceEnhancingDiffusion(num_iter=3, sigma=1.0)
        ced.apply(gray_image)
        np.testing.assert_array_equal(gray_image.gray[:], original_gray)

    def test_output_clipped_to_valid_range(self, gray_image):
        ced = CoherenceEnhancingDiffusion(num_iter=5, sigma=1.0)
        result = ced.apply(gray_image)
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0


# -- Anisotropic diffusion correctness --------------------------------------


class TestAnisotropy:
    """CED should smooth along edges more than across them."""

    def test_horizontal_edge_anisotropy(self):
        """Sharp horizontal edge: smooth along edge, preserve across edge."""
        # Create image with a sharp horizontal edge at the midpoint
        img = np.zeros((100, 100), dtype=np.float64)
        img[50:, :] = 1.0
        image = Image(arr=img)

        ced = CoherenceEnhancingDiffusion(
            num_iter=10, sigma=2.0, dt=0.1, alpha=0.001, C=1.0,
        )
        result = ced.apply(image)
        result_mat = result.detect_mat[:]

        # Measure edge sharpness: variance along vertical profile through center
        # A preserved edge has high variance in vertical cross-sections
        vertical_profile = result_mat[:, 50]
        vertical_gradient = np.diff(vertical_profile)
        edge_sharpness = np.max(np.abs(vertical_gradient))

        # Measure along-edge uniformity: variance of the edge row values
        # If smoothed along the edge, each row should be nearly constant
        edge_row = result_mat[50, :]
        along_edge_variance = np.var(edge_row)

        # The edge should remain sharp (strong gradient across the edge)
        assert edge_sharpness > 0.05, (
            f"Edge was over-smoothed: max gradient = {edge_sharpness}"
        )

        # Along the edge should be very uniform (low variance)
        assert along_edge_variance < 0.01, (
            f"Along-edge variance too high: {along_edge_variance}"
        )


# -- Isotropic region test --------------------------------------------------


class TestIsotropicRegion:
    """Uniform image should stay approximately uniform."""

    def test_uniform_stays_uniform(self):
        img = np.full((64, 64), 0.5, dtype=np.float64)
        image = Image(arr=img)

        ced = CoherenceEnhancingDiffusion(num_iter=10, sigma=1.5)
        result = ced.apply(image)
        result_mat = result.detect_mat[:]

        np.testing.assert_allclose(result_mat, 0.5, atol=1e-6)


# -- Pipeline integration ---------------------------------------------------


class TestPipelineIntegration:
    """CED works in a pipeline with other operations."""

    def test_ced_in_pipeline(self):
        from phenotypic.enhance import GaussianBlur

        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float64) * 0.5 + 0.25
        image = Image(arr=arr)

        pipeline = ImagePipeline([
            CoherenceEnhancingDiffusion(num_iter=3, sigma=1.0),
            GaussianBlur(sigma=1.0),
        ])
        result = pipeline.apply(image)
        assert result.detect_mat[:].shape == image.detect_mat[:].shape
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0


# -- Serialization roundtrip ------------------------------------------------


class TestSerialization:
    """to_json/from_json preserves all parameters including C."""

    def test_roundtrip_preserves_params(self):
        pipeline = ImagePipeline([
            CoherenceEnhancingDiffusion(
                num_iter=15, sigma=2.5, dt=0.08, alpha=0.01, C=3.0,
            ),
        ])
        json_str = pipeline.to_json()
        loaded = ImagePipeline.from_json(json_str)

        ops = list(loaded._ops.values())
        assert len(ops) == 1
        ced = ops[0]
        assert isinstance(ced, CoherenceEnhancingDiffusion)
        assert ced.num_iterations == 15
        assert ced.sigma == 2.5
        assert ced.dt == 0.08
        assert ced.alpha == 0.01
        assert ced.C == 3.0

    def test_default_params_roundtrip(self):
        pipeline = ImagePipeline([CoherenceEnhancingDiffusion()])
        json_str = pipeline.to_json()
        loaded = ImagePipeline.from_json(json_str)

        ced = list(loaded._ops.values())[0]
        assert ced.num_iterations == 20
        assert ced.C == 1.0
