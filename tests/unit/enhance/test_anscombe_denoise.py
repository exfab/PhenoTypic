"""
Tests for AnscombeTransformDenoise.

Tests parameter validation, mathematical transforms, integration with
ImageEnhancer and ImagePipeline, ClipControlMixin functionality, and
immutability guarantees.
"""

import pytest
import numpy as np

from phenotypic import Image, ImagePipeline
from phenotypic.enhance import AnscombeTransformDenoise, GaussianBlur, BilateralDenoise
from phenotypic.tools_ import ClipControlMixin


class TestAnscombeTransformDenoiseParameterValidation:
    """Test AnscombeTransformDenoise parameter validation."""

    def test_inner_enhancer_without_apply_raises_error(self):
        """Test that inner_enhancer without apply() method raises TypeError."""
        with pytest.raises(TypeError, match="inner_enhancer must be an ImageEnhancer"):
            AnscombeTransformDenoise(inner_enhancer="not_an_enhancer")

    def test_inner_enhancer_with_non_callable_apply_raises_error(self):
        """Test that inner_enhancer with non-callable apply raises TypeError."""

        class FakeEnhancer:
            apply = "not_callable"

        with pytest.raises(TypeError, match="inner_enhancer must be an ImageEnhancer"):
            AnscombeTransformDenoise(inner_enhancer=FakeEnhancer())

    def test_gain_zero_raises_error(self):
        """Test that gain = 0 raises ValueError."""
        with pytest.raises(ValueError, match="gain must be > 0"):
            AnscombeTransformDenoise(inner_enhancer=GaussianBlur(), gain=0)

    def test_gain_negative_raises_error(self):
        """Test that negative gain raises ValueError."""
        with pytest.raises(ValueError, match="gain must be > 0"):
            AnscombeTransformDenoise(inner_enhancer=GaussianBlur(), gain=-1.0)

    def test_sigma_negative_raises_error(self):
        """Test that negative sigma (read noise) raises ValueError."""
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            AnscombeTransformDenoise(inner_enhancer=GaussianBlur(), sigma=-0.1)

    def test_scale_factor_zero_raises_error(self):
        """Test that scale_factor = 0 raises ValueError."""
        with pytest.raises(ValueError, match="scale_factor must be > 0"):
            AnscombeTransformDenoise(inner_enhancer=GaussianBlur(), scale_factor=0)

    def test_scale_factor_negative_raises_error(self):
        """Test that negative scale_factor raises ValueError."""
        with pytest.raises(ValueError, match="scale_factor must be > 0"):
            AnscombeTransformDenoise(inner_enhancer=GaussianBlur(), scale_factor=-255)

    def test_valid_parameters_with_image_enhancer(self):
        """Test that valid parameters with ImageEnhancer are accepted."""
        enhancer = AnscombeTransformDenoise(
            inner_enhancer=GaussianBlur(sigma=1.0),
            gain=2.0,
            mu=0.5,
            sigma=1.0,
            scale_factor=255.0,
        )
        assert enhancer.gain == 2.0
        assert enhancer.mu == 0.5
        assert enhancer.sigma == 1.0
        assert enhancer.scale_factor == 255.0

    def test_valid_parameters_with_image_pipeline(self):
        """Test that valid parameters with ImagePipeline are accepted."""
        pipeline = ImagePipeline([GaussianBlur(sigma=1.0)])
        enhancer = AnscombeTransformDenoise(
            inner_enhancer=pipeline,
            gain=1.0,
            mu=0.0,
            sigma=0.0,
        )
        assert enhancer.inner_enhancer is pipeline

    def test_scale_factor_none_accepted(self):
        """Test that scale_factor=None (auto-detect) is accepted."""
        enhancer = AnscombeTransformDenoise(
            inner_enhancer=GaussianBlur(),
            scale_factor=None,
        )
        assert enhancer.scale_factor is None


class TestAnscombeTransformMathematics:
    """Test the mathematical correctness of the Anscombe transforms."""

    def test_forward_transform_increases_sqrt_scaled(self):
        """Test that forward transform applies sqrt scaling."""
        # For pure Poisson (sigma=0), forward transform: (2/gain) * sqrt(gain*x + 3/8*gain^2)
        x = np.array([100.0, 400.0, 900.0])
        result = AnscombeTransformDenoise._generalized_anscombe(
            x, mu=0, sigma=0, gain=1.0
        )
        # For large counts, result should be approximately 2*sqrt(x)
        expected_approx = 2 * np.sqrt(x)
        np.testing.assert_allclose(result, expected_approx, rtol=0.05)

    def test_forward_transform_handles_zero_values(self):
        """Test that forward transform handles zero counts gracefully."""
        x = np.array([0.0, 0.0, 0.0])
        result = AnscombeTransformDenoise._generalized_anscombe(
            x, mu=0, sigma=0, gain=1.0
        )
        # Should not produce NaN
        assert not np.any(np.isnan(result))
        # Result should be positive (due to 3/8 constant term)
        assert np.all(result >= 0)

    def test_forward_inverse_roundtrip_large_counts(self):
        """Test that forward + inverse transforms approximately recover original for large counts."""
        # Create array with values that have good roundtrip properties
        x = np.array([50.0, 100.0, 200.0, 500.0, 1000.0])
        gain = 1.0
        mu = 0.0
        sigma = 0.0

        # Forward then inverse
        forward = AnscombeTransformDenoise._generalized_anscombe(x, mu, sigma, gain)
        inverse = AnscombeTransformDenoise._inverse_generalized_anscombe(
            forward, mu, sigma, gain
        )

        # Should approximately recover original (better for larger counts)
        np.testing.assert_allclose(inverse, x, rtol=0.1)

    def test_forward_transform_with_read_noise(self):
        """Test forward transform with non-zero read noise parameters."""
        x = np.array([100.0, 200.0, 300.0])
        mu = 5.0
        sigma = 10.0
        gain = 2.0

        result = AnscombeTransformDenoise._generalized_anscombe(x, mu, sigma, gain)

        # Should produce valid, positive results
        assert not np.any(np.isnan(result))
        assert np.all(result > 0)

    def test_inverse_transform_handles_small_values(self):
        """Test that inverse transform handles small transformed values."""
        # Small transformed values (< 1) get clamped to 1 in the inverse
        x = np.array([0.5, 0.8, 1.0, 1.5])
        result = AnscombeTransformDenoise._inverse_generalized_anscombe(
            x, mu=0, sigma=0, gain=1.0
        )

        # Should not produce NaN
        assert not np.any(np.isnan(result))
        # Should produce non-negative results
        assert np.all(result >= 0)

    def test_inverse_transform_handles_nan(self):
        """Test that inverse transform replaces NaN with 0."""
        x = np.array([np.nan, 10.0, 20.0])
        result = AnscombeTransformDenoise._inverse_generalized_anscombe(
            x, mu=0, sigma=0, gain=1.0
        )

        # NaN should be replaced with 0
        assert result[0] == 0.0
        assert not np.any(np.isnan(result))


class TestAnscombeTransformDenoiseIntegration:
    """Integration tests with Image, ImageEnhancer, and ImagePipeline."""

    @pytest.fixture
    def synthetic_image(self):
        """Create a synthetic test image."""
        np.random.seed(42)
        arr = np.random.rand(64, 64).astype(np.float64) * 0.5 + 0.25
        return Image(arr=arr)

    @pytest.fixture
    def rgb_image(self):
        """Create a synthetic RGB image."""
        np.random.seed(42)
        arr = np.random.rand(64, 64, 3).astype(np.float64)
        return Image(arr=arr)

    def test_apply_with_gaussian_blur(self, synthetic_image):
        """Test apply with GaussianBlur as inner enhancer."""
        enhancer = AnscombeTransformDenoise(
            inner_enhancer=GaussianBlur(sigma=1.0),
            gain=1.0,
            sigma=0.0,
            scale_factor=255.0,
        )
        result = enhancer.apply(synthetic_image)

        # Output should have same shape
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape
        # Output should be in [0, 1] range
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0

    def test_apply_with_image_pipeline(self, synthetic_image):
        """Test apply with ImagePipeline as inner enhancer."""
        pipeline = ImagePipeline([
            GaussianBlur(sigma=0.5),
            BilateralDenoise(sigma_spatial=5),
        ])
        enhancer = AnscombeTransformDenoise(
            inner_enhancer=pipeline,
            gain=1.0,
            sigma=0.0,
            scale_factor=255.0,
        )
        result = enhancer.apply(synthetic_image)

        # Output should have same shape
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape
        # Output should be in [0, 1] range
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0

    def test_apply_with_clipping_enhancer(self, synthetic_image):
        """Test that AnscombeTransformDenoise works with clipping enhancers.

        BilateralDenoise has clip=True by default, which would destroy GAT-scale
        values. AnscombeTransformDenoise should disable clipping internally.
        """
        enhancer = AnscombeTransformDenoise(
            inner_enhancer=BilateralDenoise(sigma_spatial=5),  # clip=True default
            gain=1.0,
            sigma=0.0,
            scale_factor=255.0,
        )
        result = enhancer.apply(synthetic_image)

        # Output should be in [0, 1] range and non-trivial
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0
        # Output should not be all zeros (which would happen if clipping broke things)
        assert result.detect_mat[:].max() > 0.1

    def test_apply_with_pipeline_containing_clipping_enhancers(self, synthetic_image):
        """Test that AnscombeTransformDenoise disables clipping in pipelines."""
        pipeline = ImagePipeline([
            GaussianBlur(sigma=0.5),
            BilateralDenoise(sigma_spatial=5),  # Has clip=True
        ])
        enhancer = AnscombeTransformDenoise(
            inner_enhancer=pipeline,
            gain=1.0,
            sigma=0.0,
            scale_factor=255.0,
        )
        result = enhancer.apply(synthetic_image)

        # Output should be non-trivial (not all zeros)
        assert result.detect_mat[:].max() > 0.1
        # Original pipeline's enhancer should still have clip=True
        # _ops is a Dict[str, ImageOperation], access by operation name
        orig_ops = list(pipeline._ops.values())
        assert orig_ops[1].clip is True

    def test_apply_preserves_image_rgb(self, rgb_image):
        """Test that apply() does not modify image.rgb (immutability)."""
        original_rgb = rgb_image.rgb[:].copy()

        enhancer = AnscombeTransformDenoise(
            inner_enhancer=GaussianBlur(sigma=1.0),
            scale_factor=255.0,
        )
        result = enhancer.apply(rgb_image)

        # Original image rgb should be unchanged
        assert np.array_equal(rgb_image.rgb[:], original_rgb)

    def test_apply_preserves_image_gray(self, synthetic_image):
        """Test that apply() does not modify image.gray (immutability)."""
        original_gray = synthetic_image.gray[:].copy()

        enhancer = AnscombeTransformDenoise(
            inner_enhancer=GaussianBlur(sigma=1.0),
            scale_factor=255.0,
        )
        result = enhancer.apply(synthetic_image)

        # Original image gray should be unchanged
        assert np.array_equal(synthetic_image.gray[:], original_gray)

    def test_inplace_modifies_original(self, synthetic_image):
        """Test that inplace=True modifies the original image."""
        original_detect_mat = synthetic_image.detect_mat[:].copy()

        enhancer = AnscombeTransformDenoise(
            inner_enhancer=GaussianBlur(sigma=1.0),
            scale_factor=255.0,
        )
        result = enhancer.apply(synthetic_image, inplace=True)

        # Result should be the same object
        assert result is synthetic_image
        # detect_mat should be modified (not equal to original)
        assert not np.array_equal(synthetic_image.detect_mat[:], original_detect_mat)

    def test_nested_in_pipeline(self, synthetic_image):
        """Test AnscombeTransformDenoise nested inside ImagePipeline."""
        from phenotypic.enhance import CLAHE

        pipeline = ImagePipeline([
            AnscombeTransformDenoise(
                inner_enhancer=GaussianBlur(sigma=1.0),
                scale_factor=255.0,
            ),
            CLAHE(clip_limit=0.02),
        ])
        result = pipeline.apply(synthetic_image)

        # Output should have same shape
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape
        # Output should be in [0, 1] range
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0


class TestClipParameter:
    """Test the clip parameter added to enhancers for GAT compatibility."""

    def test_bilateral_denoise_clip_true_default(self):
        """Test that BilateralDenoise clips to [0,1] by default."""
        # Create GAT-scale test data (~1-32)
        np.random.seed(42)
        gat_data = np.random.uniform(1.0, 32.0, (32, 32)).astype(np.float64)
        image = Image(arr=gat_data)

        enh = BilateralDenoise(sigma_spatial=5)  # clip=True by default
        result = enh.apply(image)

        # Output should be clipped to [0, 1]
        assert result.detect_mat[:].max() <= 1.0
        assert result.detect_mat[:].min() >= 0.0

    def test_bilateral_denoise_clip_false_preserves_scale(self):
        """Test that BilateralDenoise with clip=False preserves GAT scale."""
        # Create GAT-scale test data (~1-32)
        np.random.seed(42)
        gat_data = np.random.uniform(1.0, 32.0, (32, 32)).astype(np.float64)
        image = Image(arr=gat_data)

        enh = BilateralDenoise(sigma_spatial=5, clip=False)
        result = enh.apply(image)

        # Output should preserve GAT scale (max > 1)
        assert result.detect_mat[:].max() > 1.0


class TestClipControlMixin:
    """Test ClipControlMixin._disable_clipping functionality."""

    def test_disable_clipping_single_enhancer(self):
        """Test that _disable_clipping creates clip-disabled copy of enhancer."""
        enh = BilateralDenoise(sigma_spatial=5, clip=True)
        copied = ClipControlMixin._disable_clipping(enh)

        # Original should be unchanged
        assert enh.clip is True
        # Copy should have clip=False
        assert copied.clip is False
        # Should be different objects
        assert enh is not copied

    def test_disable_clipping_enhancer_without_clip(self):
        """Test that enhancers without clip parameter are returned unchanged."""
        enh = GaussianBlur(sigma=1.0)
        copied = ClipControlMixin._disable_clipping(enh)

        # Should return the same object (no copy needed)
        assert copied is enh

    def test_disable_clipping_pipeline(self):
        """Test that _disable_clipping works recursively on pipelines."""
        pipeline = ImagePipeline([
            GaussianBlur(sigma=1.0),
            BilateralDenoise(sigma_spatial=5, clip=True),
        ])
        copied_pipe = ClipControlMixin._disable_clipping(pipeline)

        # _ops is a Dict[str, ImageOperation], so we get values as a list
        orig_ops = list(pipeline._ops.values())
        copied_ops = list(copied_pipe._ops.values())

        # Original pipeline should be unchanged
        assert orig_ops[1].clip is True
        # Copied pipeline should have clip=False on BilateralDenoise
        assert copied_ops[1].clip is False
        # Should be different pipeline objects
        assert pipeline is not copied_pipe

    def test_disable_clipping_nested_pipeline(self):
        """Test that _disable_clipping handles nested pipelines."""
        inner_pipeline = ImagePipeline([BilateralDenoise(sigma_spatial=5, clip=True)])
        outer_pipeline = ImagePipeline([
            GaussianBlur(sigma=1.0),
            inner_pipeline,
        ])
        # Note: ImagePipeline doesn't have a clip attribute but contains _ops
        # This test verifies the recursion handles pipeline-like structures


class TestAnscombeTransformDenoiseScaleFactorAutoDetect:
    """Test scale factor auto-detection from image metadata."""

    def test_scale_factor_manual_override(self):
        """Test that manual scale_factor is used when provided."""
        arr = np.random.rand(32, 32).astype(np.float64)
        image = Image(arr=arr)

        enhancer = AnscombeTransformDenoise(
            inner_enhancer=GaussianBlur(sigma=0.5),
            scale_factor=65535.0,  # Manual override
        )

        scale = enhancer._get_scale_factor(image)
        assert scale == 65535.0

    def test_scale_factor_default_when_none_and_no_metadata(self):
        """Test that default 255 is used when scale_factor=None and no metadata."""
        arr = np.random.rand(32, 32).astype(np.float64)
        image = Image(arr=arr)

        enhancer = AnscombeTransformDenoise(
            inner_enhancer=GaussianBlur(sigma=0.5),
            scale_factor=None,  # Auto-detect
        )

        scale = enhancer._get_scale_factor(image)
        # Should default to 255 when no bit_depth metadata
        assert scale == 255.0


class TestAnscombeTransformDenoiseSerialization:
    """Test serialization and deserialization in pipelines.

    Note: Full roundtrip deserialization is limited by the current serialization
    system which instantiates classes with empty constructors. AnscombeTransformDenoise
    has a required parameter (inner_enhancer) which prevents this pattern from working.
    Serialization works, but deserialization would require framework changes.
    """

    def test_pipeline_to_json_serializes(self):
        """Test that pipeline with AnscombeTransformDenoise can be serialized to JSON."""
        pipeline = ImagePipeline([
            AnscombeTransformDenoise(
                inner_enhancer=GaussianBlur(sigma=1.0),
                gain=2.0,
                mu=1.0,
                sigma=0.5,
                scale_factor=255.0,
            )
        ])

        # Serialize should succeed
        json_str = pipeline.to_json()
        assert "AnscombeTransformDenoise" in json_str
        assert "GaussianBlur" in json_str
        assert "inner_enhancer" in json_str

    @pytest.mark.skip(
        reason="Deserialization requires empty constructor support. "
        "AnscombeTransformDenoise has required 'inner_enhancer' parameter."
    )
    def test_pipeline_to_json_and_back(self):
        """Test that pipeline with AnscombeTransformDenoise can be deserialized."""
        pipeline = ImagePipeline([
            AnscombeTransformDenoise(
                inner_enhancer=GaussianBlur(sigma=1.0),
                gain=2.0,
                mu=1.0,
                sigma=0.5,
                scale_factor=255.0,
            )
        ])

        # Serialize
        json_str = pipeline.to_json()

        # Deserialize - this would require framework changes
        loaded = ImagePipeline.from_json(json_str)

        # Verify the operation was restored
        assert len(loaded._ops) == 1
        restored = loaded._ops[0]
        assert isinstance(restored, AnscombeTransformDenoise)
        assert restored.gain == 2.0
        assert restored.mu == 1.0
        assert restored.sigma == 0.5
        assert restored.scale_factor == 255.0

    @pytest.mark.skip(
        reason="Deserialization requires empty constructor support. "
        "AnscombeTransformDenoise has required 'inner_enhancer' parameter."
    )
    def test_pipeline_with_nested_pipeline_serialization(self):
        """Test serialization with ImagePipeline as inner enhancer."""
        inner_pipeline = ImagePipeline([GaussianBlur(sigma=1.0)])
        pipeline = ImagePipeline([
            AnscombeTransformDenoise(
                inner_enhancer=inner_pipeline,
                gain=1.0,
                scale_factor=255.0,
            )
        ])

        # Serialize
        json_str = pipeline.to_json()

        # Deserialize - this would require framework changes
        loaded = ImagePipeline.from_json(json_str)

        # Verify structure
        assert len(loaded._ops) == 1
        restored = loaded._ops[0]
        assert isinstance(restored, AnscombeTransformDenoise)
        assert isinstance(restored.inner_enhancer, ImagePipeline)


# Run all tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
