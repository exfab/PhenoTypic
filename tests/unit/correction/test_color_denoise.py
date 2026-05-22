"""Tests for the ColorDenoise CBM3D color-denoising corrector."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.correction import ColorDenoise


class TestColorDenoiseParameterValidation:
    """Parameter validation and error handling."""

    def test_negative_sigma_psd_raises(self):
        with pytest.raises(ValueError, match="sigma_psd"):
            ColorDenoise(sigma_psd=-0.1)

    def test_zero_block_size_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            ColorDenoise(block_size=0)

    def test_negative_block_size_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            ColorDenoise(block_size=-8)

    def test_zero_gat_gain_raises(self):
        with pytest.raises(ValueError, match="gat_gain"):
            ColorDenoise(gat_gain=0.0)

    def test_negative_gat_read_sigma_raises(self):
        with pytest.raises(ValueError, match="gat_read_sigma"):
            ColorDenoise(gat_read_sigma=-1.0)

    def test_zero_gat_scale_factor_raises(self):
        with pytest.raises(ValueError, match="gat_scale_factor"):
            ColorDenoise(gat_scale_factor=0.0)

    def test_valid_defaults(self):
        c = ColorDenoise()
        assert c.sigma_psd == 0.02
        assert c.block_size == 8
        assert c.clip is True
        assert c.use_gat is False
        assert c.gat_gain == 1.0
        assert c.gat_mu == 0.0
        assert c.gat_read_sigma == 0.0
        assert c.gat_scale_factor is None

    def test_valid_custom_params(self):
        # Pydantic models are keyword-only constructed.
        c = ColorDenoise(
                sigma_psd=0.05,
                block_size=16,
                clip=False,
                use_gat=True,
                gat_gain=2.0,
                gat_scale_factor=65535.0,
        )
        assert c.sigma_psd == 0.05
        assert c.block_size == 16
        assert c.clip is False
        assert c.use_gat is True
        assert c.gat_gain == 2.0
        assert c.gat_scale_factor == 65535.0

    def test_gat_scale_factor_none_allowed(self):
        assert ColorDenoise(gat_scale_factor=None).gat_scale_factor is None


@pytest.fixture()
def noisy_rgb_image():
    """A small synthetic RGB image with additive Gaussian noise."""
    rng = np.random.default_rng(42)
    clean = np.full((64, 64, 3), 128, dtype=np.float64)
    noisy = (clean + rng.normal(0, 12, clean.shape)).clip(0, 255)
    return Image(arr=noisy.astype(np.uint8))


class TestColorDenoiseBehavior:
    """End-to-end denoising behavior on RGB images."""

    def test_plain_changes_rgb(self, noisy_rgb_image):
        original = noisy_rgb_image.rgb[:].copy()
        result = ColorDenoise(sigma_psd=0.05).apply(noisy_rgb_image)
        assert not np.array_equal(result.rgb[:], original)

    def test_gat_changes_rgb(self, noisy_rgb_image):
        original = noisy_rgb_image.rgb[:].copy()
        result = ColorDenoise(use_gat=True, gat_gain=2.0).apply(noisy_rgb_image)
        assert not np.array_equal(result.rgb[:], original)

    def test_shape_and_dtype_preserved(self, noisy_rgb_image):
        original = noisy_rgb_image.rgb[:]
        result = ColorDenoise(sigma_psd=0.05).apply(noisy_rgb_image)
        assert result.rgb[:].shape == original.shape
        assert result.rgb[:].dtype == original.dtype

    def test_cascade_rebuilds_gray_and_detect_mat(self, noisy_rgb_image):
        """Writing denoised RGB rebuilds gray and detect_mat via the cascade."""
        original_gray = noisy_rgb_image.gray[:].copy()
        original_dm = noisy_rgb_image.detect_mat[:].copy()
        result = ColorDenoise(sigma_psd=0.05).apply(noisy_rgb_image)
        assert not np.array_equal(result.gray[:], original_gray)
        assert not np.array_equal(result.detect_mat[:], original_dm)

    def test_input_not_mutated(self, noisy_rgb_image):
        """apply() defaults to inplace=False, leaving the input untouched."""
        original = noisy_rgb_image.rgb[:].copy()
        ColorDenoise(sigma_psd=0.05).apply(noisy_rgb_image)
        np.testing.assert_array_equal(noisy_rgb_image.rgb[:], original)

    def test_output_in_dtype_range(self, noisy_rgb_image):
        result = ColorDenoise(sigma_psd=0.05).apply(noisy_rgb_image)
        assert result.rgb[:].min() >= 0
        assert result.rgb[:].max() <= 255

    def test_mse_decreases_vs_clean(self):
        """CBM3D reduces mean squared error against the clean signal."""
        rng = np.random.default_rng(123)
        clean = np.full((64, 64, 3), 128, dtype=np.float64)
        noisy = (clean + rng.normal(0, 15, clean.shape)).clip(0, 255)
        image = Image(arr=noisy.astype(np.uint8))

        result = ColorDenoise(sigma_psd=0.06).apply(image)

        noisy_mse = np.mean((noisy - clean) ** 2)
        denoised_mse = np.mean((result.rgb[:].astype(np.float64) - clean) ** 2)
        assert denoised_mse < noisy_mse


class TestColorDenoiseGuards:
    """Guard conditions for unsupported input."""

    def test_grayscale_image_raises(self):
        """CBM3D requires 3-channel RGB; a grayscale image is rejected."""
        gray = np.linspace(0, 255, 64 * 64).reshape(64, 64).astype(np.uint8)
        with pytest.raises(ValueError, match="requires a 3-channel RGB"):
            # Call _operate directly to surface the raw guard ValueError;
            # apply() would wrap it in a generic Exception.
            ColorDenoise()._operate(Image(arr=gray))


class TestColorDenoiseSerialization:
    """Serialization round-trips for reproducible pipelines."""

    def test_model_roundtrip(self):
        c = ColorDenoise(
                sigma_psd=0.03, block_size=16, use_gat=True, gat_gain=2.0
        )
        restored = ColorDenoise.model_validate(c.model_dump())
        assert restored.sigma_psd == c.sigma_psd
        assert restored.block_size == c.block_size
        assert restored.use_gat == c.use_gat
        assert restored.gat_gain == c.gat_gain

    def test_pipeline_json_roundtrip(self):
        pipe = ImagePipeline(ops=[ColorDenoise(sigma_psd=0.03, use_gat=True)])
        loaded = ImagePipeline.from_json(pipe.to_json())
        op = list(loaded._ops.values())[0]
        assert type(op).__name__ == "ColorDenoise"
        assert op.sigma_psd == 0.03
        assert op.use_gat is True
