"""
Tests for AnscombeForward and AnscombeInverse.

Tests parameter validation, mathematical transforms, forward/inverse
roundtrip, pipeline composition, scale factor auto-detection, serialization,
and immutability guarantees.
"""

import pytest
import numpy as np

from phenotypic import Image, ImagePipeline
from phenotypic.enhance import (
    AnscombeForward,
    AnscombeInverse,
    GaussianBlur,
    BilateralDenoise,
)


# -- Parameter validation (both classes) ----------------------------------


class TestAnscombeForwardParameterValidation:
    """Test AnscombeForward parameter validation."""

    def test_gain_zero_raises_error(self):
        with pytest.raises(ValueError, match="gain must be > 0"):
            AnscombeForward(gain=0)

    def test_gain_negative_raises_error(self):
        with pytest.raises(ValueError, match="gain must be > 0"):
            AnscombeForward(gain=-1.0)

    def test_sigma_negative_raises_error(self):
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            AnscombeForward(sigma=-0.1)

    def test_scale_factor_zero_raises_error(self):
        with pytest.raises(ValueError, match="scale_factor must be > 0"):
            AnscombeForward(scale_factor=0)

    def test_scale_factor_negative_raises_error(self):
        with pytest.raises(ValueError, match="scale_factor must be > 0"):
            AnscombeForward(scale_factor=-255)

    def test_valid_parameters_accepted(self):
        fwd = AnscombeForward(
            gain=2.0, mu=0.5, sigma=1.0, scale_factor=255.0
        )
        assert fwd.gain == 2.0
        assert fwd.mu == 0.5
        assert fwd.sigma == 1.0
        assert fwd.scale_factor == 255.0

    def test_defaults(self):
        fwd = AnscombeForward()
        assert fwd.gain == 1.0
        assert fwd.mu == 0.0
        assert fwd.sigma == 0.0
        assert fwd.scale_factor is None


class TestAnscombeInverseParameterValidation:
    """Test AnscombeInverse parameter validation."""

    def test_gain_zero_raises_error(self):
        with pytest.raises(ValueError, match="gain must be > 0"):
            AnscombeInverse(gain=0)

    def test_gain_negative_raises_error(self):
        with pytest.raises(ValueError, match="gain must be > 0"):
            AnscombeInverse(gain=-1.0)

    def test_sigma_negative_raises_error(self):
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            AnscombeInverse(sigma=-0.1)

    def test_scale_factor_zero_raises_error(self):
        with pytest.raises(ValueError, match="scale_factor must be > 0"):
            AnscombeInverse(scale_factor=0)

    def test_scale_factor_negative_raises_error(self):
        with pytest.raises(ValueError, match="scale_factor must be > 0"):
            AnscombeInverse(scale_factor=-255)

    def test_valid_parameters_accepted(self):
        inv = AnscombeInverse(
            gain=2.0, mu=0.5, sigma=1.0, scale_factor=255.0
        )
        assert inv.gain == 2.0
        assert inv.mu == 0.5
        assert inv.sigma == 1.0
        assert inv.scale_factor == 255.0


# -- Forward transform mathematics ----------------------------------------


class TestForwardTransformMathematics:
    """Test mathematical correctness of the forward Anscombe transform."""

    def test_sqrt_scaling_for_large_counts(self):
        """Forward transform approximates 2*sqrt(x) for large counts."""
        x = np.array([100.0, 400.0, 900.0])
        result = AnscombeForward._generalized_anscombe(
            x, mu=0, sigma=0, gain=1.0
        )
        expected_approx = 2 * np.sqrt(x)
        np.testing.assert_allclose(result, expected_approx, rtol=0.05)

    def test_handles_zero_values(self):
        """Forward transform handles zero counts without NaN."""
        x = np.array([0.0, 0.0, 0.0])
        result = AnscombeForward._generalized_anscombe(
            x, mu=0, sigma=0, gain=1.0
        )
        assert not np.any(np.isnan(result))
        assert np.all(result >= 0)

    def test_with_read_noise(self):
        """Forward transform with non-zero read noise produces valid output."""
        x = np.array([100.0, 200.0, 300.0])
        result = AnscombeForward._generalized_anscombe(
            x, mu=5.0, sigma=10.0, gain=2.0
        )
        assert not np.any(np.isnan(result))
        assert np.all(result > 0)


# -- Inverse transform mathematics ----------------------------------------


class TestInverseTransformMathematics:
    """Test mathematical correctness of the inverse Anscombe transform."""

    def test_handles_small_values(self):
        """Inverse transform handles small transformed values."""
        x = np.array([0.5, 0.8, 1.0, 1.5])
        result = AnscombeInverse._inverse_generalized_anscombe(
            x, mu=0, sigma=0, gain=1.0
        )
        assert not np.any(np.isnan(result))
        assert np.all(result >= 0)

    def test_handles_nan(self):
        """Inverse transform replaces NaN with 0."""
        x = np.array([np.nan, 10.0, 20.0])
        result = AnscombeInverse._inverse_generalized_anscombe(
            x, mu=0, sigma=0, gain=1.0
        )
        assert result[0] == 0.0
        assert not np.any(np.isnan(result))


# -- Forward/Inverse roundtrip --------------------------------------------


class TestForwardInverseRoundtrip:
    """Test that forward then inverse approximately recovers original."""

    def test_roundtrip_large_counts(self):
        x = np.array([50.0, 100.0, 200.0, 500.0, 1000.0])
        forward = AnscombeForward._generalized_anscombe(
            x, mu=0, sigma=0, gain=1.0
        )
        inverse = AnscombeInverse._inverse_generalized_anscombe(
            forward, mu=0, sigma=0, gain=1.0
        )
        np.testing.assert_allclose(inverse, x, rtol=0.1)

    def test_roundtrip_with_read_noise(self):
        x = np.array([100.0, 200.0, 500.0, 1000.0])
        gain, mu, sigma = 2.0, 1.0, 3.0
        forward = AnscombeForward._generalized_anscombe(
            x, mu=mu, sigma=sigma, gain=gain
        )
        inverse = AnscombeInverse._inverse_generalized_anscombe(
            forward, mu=mu, sigma=sigma, gain=gain
        )
        np.testing.assert_allclose(inverse, x, rtol=0.15)

    def test_roundtrip_on_image(self):
        """Forward + Inverse pipeline recovers detect_mat approximately."""
        np.random.seed(42)
        arr = np.random.rand(64, 64).astype(np.float64) * 0.5 + 0.25
        image = Image(arr=arr)
        original = image.detect_mat[:].copy()

        pipeline = ImagePipeline([
            AnscombeForward(gain=1.0, sigma=0.0, scale_factor=255.0),
            AnscombeInverse(gain=1.0, sigma=0.0, scale_factor=255.0),
        ])
        result = pipeline.apply(image)

        np.testing.assert_allclose(
            result.detect_mat[:], original, atol=0.02
        )


# -- Pipeline integration -------------------------------------------------


class TestPipelineIntegration:
    """Test AnscombeForward/Inverse composed with denoisers in a pipeline."""

    @pytest.fixture
    def synthetic_image(self):
        np.random.seed(42)
        arr = np.random.rand(64, 64).astype(np.float64) * 0.5 + 0.25
        return Image(arr=arr)

    @pytest.fixture
    def rgb_image(self):
        np.random.seed(42)
        arr = np.random.rand(64, 64, 3).astype(np.float64)
        return Image(arr=arr)

    def test_forward_denoise_inverse_pipeline(self, synthetic_image):
        """Full pipeline: Forward -> GaussianBlur -> Inverse."""
        pipeline = ImagePipeline([
            AnscombeForward(
                gain=1.0, sigma=0.0, scale_factor=255.0
            ),
            GaussianBlur(sigma=1.0),
            AnscombeInverse(
                gain=1.0, sigma=0.0, scale_factor=255.0
            ),
        ])
        result = pipeline.apply(synthetic_image)

        assert result.detect_mat[:].shape == (
            synthetic_image.detect_mat[:].shape
        )
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0

    def test_pipeline_with_clip_false_enhancer(self, synthetic_image):
        """Pipeline with clip=False on intermediate BilateralDenoise."""
        pipeline = ImagePipeline([
            AnscombeForward(
                gain=1.0, sigma=0.0, scale_factor=255.0
            ),
            BilateralDenoise(sigma_spatial=5, clip=False),
            AnscombeInverse(
                gain=1.0, sigma=0.0, scale_factor=255.0
            ),
        ])
        result = pipeline.apply(synthetic_image)

        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0
        assert result.detect_mat[:].max() > 0.1

    def test_preserves_rgb(self, rgb_image):
        """Pipeline does not modify image.rgb (immutability)."""
        original_rgb = rgb_image.rgb[:].copy()

        pipeline = ImagePipeline([
            AnscombeForward(scale_factor=255.0),
            GaussianBlur(sigma=1.0),
            AnscombeInverse(scale_factor=255.0),
        ])
        pipeline.apply(rgb_image)

        np.testing.assert_array_equal(rgb_image.rgb[:], original_rgb)

    def test_preserves_gray(self, synthetic_image):
        """Pipeline does not modify image.gray (immutability)."""
        original_gray = synthetic_image.gray[:].copy()

        pipeline = ImagePipeline([
            AnscombeForward(scale_factor=255.0),
            GaussianBlur(sigma=1.0),
            AnscombeInverse(scale_factor=255.0),
        ])
        pipeline.apply(synthetic_image)

        np.testing.assert_array_equal(
            synthetic_image.gray[:], original_gray
        )

    def test_forward_produces_gat_scale_values(self, synthetic_image):
        """AnscombeForward produces values > 1 (GAT domain)."""
        fwd = AnscombeForward(scale_factor=255.0)
        result = fwd.apply(synthetic_image)
        # GAT values should be well above 1 for typical [0.25, 0.75] data
        assert result.detect_mat[:].max() > 1.0

    def test_nested_in_outer_pipeline(self, synthetic_image):
        """Forward/Inverse pair nested inside a larger pipeline."""
        from phenotypic.enhance import CLAHE

        pipeline = ImagePipeline([
            AnscombeForward(scale_factor=255.0),
            GaussianBlur(sigma=1.0),
            AnscombeInverse(scale_factor=255.0),
            CLAHE(clip_limit=0.02),
        ])
        result = pipeline.apply(synthetic_image)

        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0


# -- Scale factor auto-detection ------------------------------------------


class TestScaleFactorAutoDetection:
    """Test scale factor auto-detection from image metadata."""

    def test_manual_override_forward(self):
        arr = np.random.rand(32, 32).astype(np.float64)
        image = Image(arr=arr)
        fwd = AnscombeForward(scale_factor=65535.0)
        assert fwd._get_scale_factor(image) == 65535.0

    def test_manual_override_inverse(self):
        arr = np.random.rand(32, 32).astype(np.float64)
        image = Image(arr=arr)
        inv = AnscombeInverse(scale_factor=65535.0)
        assert inv._get_scale_factor(image) == 65535.0

    def test_default_when_no_metadata_forward(self):
        arr = np.random.rand(32, 32).astype(np.float64)
        image = Image(arr=arr)
        fwd = AnscombeForward(scale_factor=None)
        assert fwd._get_scale_factor(image) == 255.0

    def test_default_when_no_metadata_inverse(self):
        arr = np.random.rand(32, 32).astype(np.float64)
        image = Image(arr=arr)
        inv = AnscombeInverse(scale_factor=None)
        assert inv._get_scale_factor(image) == 255.0


# -- Serialization ---------------------------------------------------------


class TestSerialization:
    """Test that both classes serialize and deserialize in pipelines."""

    def test_forward_serializes_to_json(self):
        pipeline = ImagePipeline([
            AnscombeForward(
                gain=2.0, mu=1.0, sigma=0.5, scale_factor=255.0
            )
        ])
        json_str = pipeline.to_json()
        assert "AnscombeForward" in json_str

    def test_inverse_serializes_to_json(self):
        pipeline = ImagePipeline([
            AnscombeInverse(
                gain=2.0, mu=1.0, sigma=0.5, scale_factor=255.0
            )
        ])
        json_str = pipeline.to_json()
        assert "AnscombeInverse" in json_str

    def test_full_pipeline_serializes(self):
        pipeline = ImagePipeline([
            AnscombeForward(
                gain=1.0, sigma=0.0, scale_factor=255.0
            ),
            GaussianBlur(sigma=1.0),
            AnscombeInverse(
                gain=1.0, sigma=0.0, scale_factor=255.0
            ),
        ])
        json_str = pipeline.to_json()
        assert "AnscombeForward" in json_str
        assert "AnscombeInverse" in json_str
        assert "GaussianBlur" in json_str

    def test_roundtrip_deserialization(self):
        """Both classes have no required params so deserialization works."""
        pipeline = ImagePipeline([
            AnscombeForward(
                gain=2.0, mu=1.0, sigma=0.5, scale_factor=255.0
            ),
            GaussianBlur(sigma=1.0),
            AnscombeInverse(
                gain=2.0, mu=1.0, sigma=0.5, scale_factor=255.0
            ),
        ])
        json_str = pipeline.to_json()
        loaded = ImagePipeline.from_json(json_str)

        ops = list(loaded._ops.values())
        assert len(ops) == 3
        assert isinstance(ops[0], AnscombeForward)
        assert isinstance(ops[2], AnscombeInverse)
        assert ops[0].gain == 2.0
        assert ops[0].mu == 1.0
        assert ops[0].sigma == 0.5
        assert ops[0].scale_factor == 255.0
        assert ops[2].gain == 2.0


# -- Old class is removed -------------------------------------------------


class TestOldClassRemoved:
    """Verify that AnscombeTransformDenoise is no longer importable."""

    def test_old_class_not_in_enhance(self):
        import phenotypic.enhance as enhance_mod
        assert not hasattr(enhance_mod, "AnscombeTransformDenoise")
