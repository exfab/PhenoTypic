from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.correction import StableDenoise


class TestBM3DCorrectorParameterValidation:
    """Test parameter validation and error handling."""

    def test_gain_zero_raises(self):
        with pytest.raises(ValueError, match="gain must be > 0"):
            StableDenoise(gain=0.0)

    def test_gain_negative_raises(self):
        with pytest.raises(ValueError, match="gain must be > 0"):
            StableDenoise(gain=-1.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            StableDenoise(sigma=-0.1)

    def test_scale_factor_zero_raises(self):
        with pytest.raises(ValueError, match="scale_factor must be > 0"):
            StableDenoise(scale_factor=0.0)

    def test_scale_factor_negative_raises(self):
        with pytest.raises(ValueError, match="scale_factor must be > 0"):
            StableDenoise(scale_factor=-10.0)

    def test_invalid_stage_arg_raises(self):
        # ``stage_arg`` is a ``Literal`` field post-pydantic-migration; an
        # out-of-set value raises ``ValidationError`` (a ``ValueError``).
        with pytest.raises(ValueError, match="stage_arg"):
            StableDenoise(stage_arg="invalid")

    def test_valid_defaults(self):
        c = StableDenoise()
        assert c.block_size == 8
        assert c.stage_arg == "all_stages"
        assert c.gain == 1.0
        assert c.mu == 0.0
        assert c.sigma == 0.0
        assert c.scale_factor is None

    def test_valid_custom_params(self):
        # Pydantic models are keyword-only constructed.
        c = StableDenoise(
                block_size=16,
                stage_arg="hard_thresholding",
                gain=2.0,
                mu=1.0,
                sigma=0.5,
                scale_factor=65535.0,
        )
        assert c.block_size == 16
        assert c.stage_arg == "hard_thresholding"
        assert c.gain == 2.0
        assert c.mu == 1.0
        assert c.sigma == 0.5
        assert c.scale_factor == 65535.0


class TestBM3DCorrectorComponentModification:
    """Test that gray and detect_mat are modified, RGB unchanged."""

    @pytest.fixture()
    def noisy_image(self):
        rng = np.random.default_rng(42)
        clean = np.full((64, 64), 0.5, dtype=np.float64)
        noisy = (clean + rng.normal(0, 0.05, clean.shape)).clip(0.0, 1.0)
        return Image(arr=noisy)

    def test_gray_modified(self, noisy_image):
        original_gray = noisy_image.gray[:].copy()
        result = StableDenoise(
                stage_arg="hard_thresholding", scale_factor=255.0
        ).apply(noisy_image)
        assert not np.array_equal(result.gray[:], original_gray)

    def test_detect_mat_modified(self, noisy_image):
        original_dm = noisy_image.detect_mat[:].copy()
        result = StableDenoise(
                stage_arg="hard_thresholding", scale_factor=255.0
        ).apply(noisy_image)
        assert not np.array_equal(result.detect_mat[:], original_dm)

    def test_rgb_unchanged(self):
        """RGB data is not modified by StableDenoise."""
        rng = np.random.default_rng(99)
        rgb = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        image = Image(arr=rgb)
        original_rgb = image.rgb[:].copy()
        result = StableDenoise(
                stage_arg="hard_thresholding", scale_factor=255.0
        ).apply(image)
        np.testing.assert_array_equal(result.rgb[:], original_rgb)

    def test_output_in_unit_range(self, noisy_image):
        result = StableDenoise(
                stage_arg="hard_thresholding", scale_factor=255.0
        ).apply(noisy_image)
        assert result.gray[:].min() >= 0.0
        assert result.gray[:].max() <= 1.0
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0

    def test_shape_preserved(self, noisy_image):
        original_shape = noisy_image.gray[:].shape
        result = StableDenoise(
                stage_arg="hard_thresholding", scale_factor=255.0
        ).apply(noisy_image)
        assert result.gray[:].shape == original_shape
        assert result.detect_mat[:].shape == original_shape


class TestBM3DCorrectorDenoisingQuality:
    """Test that denoising actually reduces noise (MSE decreases)."""

    def test_mse_decreases_vs_clean(self):
        rng = np.random.default_rng(123)
        clean = np.full((64, 64), 0.5, dtype=np.float64)
        noisy = (clean + rng.normal(0, 0.08, clean.shape)).clip(0.0, 1.0)
        image = Image(arr=noisy)

        result = StableDenoise(
                stage_arg="hard_thresholding", scale_factor=255.0
        ).apply(image)

        mse_before = np.mean((noisy - clean) ** 2)
        mse_after = np.mean((result.gray[:] - clean) ** 2)
        assert mse_after < mse_before


class TestBM3DCorrectorScaleFactor:
    """Test scale factor auto-detection and manual override."""

    def test_manual_scale_factor_used(self):
        rng = np.random.default_rng(42)
        arr = (rng.uniform(0.3, 0.7, (64, 64))).astype(np.float64)
        image = Image(arr=arr)

        c = StableDenoise(stage_arg="hard_thresholding", scale_factor=1000.0)
        assert c._get_scale_factor(image) == 1000.0

    def test_default_falls_back_to_255(self):
        rng = np.random.default_rng(42)
        arr = (rng.uniform(0.3, 0.7, (64, 64))).astype(np.float64)
        image = Image(arr=arr)

        c = StableDenoise()
        assert c._get_scale_factor(image) == 255.0


class TestBM3DCorrectorLinearization:
    """Test the sRGB linearization wrapped around the GAT denoise."""

    def test_flat_field_preserved_through_linearization(self):
        """A noise-free flat field survives sRGB-decode -> GAT -> BM3D ->
        inverse GAT -> sRGB-encode without a meaningful intensity shift.

        Guards the linearization round-trip: a broken decode/encode pair
        would bias the recovered intensity off the input level.
        """
        flat = np.full((64, 64), 0.5, dtype=np.float64)
        image = Image(arr=flat)

        result = StableDenoise(scale_factor=255.0).apply(image)

        np.testing.assert_allclose(result.gray[:], 0.5, atol=0.01)
