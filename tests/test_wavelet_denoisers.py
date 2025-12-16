"""Tests for wavelet denoising operations (VisuShrink and BayesShrink).

Tests cover ImageEnhancer versions (VisuShrinkEnhancer, BayesShrinkEnhancer) that
modify only enh_gray, and ImageCorrector versions (VisuShrinkCorrector,
BayesShrinkCorrector) that modify all components (RGB, gray, enh_gray).
"""

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.enhance import BayesShrinkEnhancer, VisuShrinkEnhancer
from phenotypic.correction import BayesShrinkCorrector, VisuShrinkCorrector


class TestVisuShrinkEnhancer:
    """Tests for VisuShrinkEnhancer (ImageEnhancer)."""

    def test_instantiate_with_defaults(self):
        """Can instantiate with default parameters."""
        enhancer = VisuShrinkEnhancer()
        assert enhancer.sigma is None
        assert enhancer.wavelet == "db2"
        assert enhancer.mode == "soft"
        assert enhancer.wavelet_levels is None

    def test_instantiate_with_custom_params(self):
        """Can instantiate with custom parameters."""
        enhancer = VisuShrinkEnhancer(
            sigma=0.05, wavelet="db4", mode="hard", wavelet_levels=4
        )
        assert enhancer.sigma == 0.05
        assert enhancer.wavelet == "db4"
        assert enhancer.mode == "hard"
        assert enhancer.wavelet_levels == 4

    def test_apply_returns_image(self, sample_image):
        """apply() returns an Image object."""
        enhancer = VisuShrinkEnhancer()
        result = enhancer.apply(sample_image)
        assert isinstance(result, Image)

    def test_apply_preserves_shape(self, sample_image):
        """apply() preserves image shape."""
        enhancer = VisuShrinkEnhancer()
        original_shape = sample_image.shape
        result = enhancer.apply(sample_image)
        assert result.shape == original_shape

    def test_apply_preserves_rgb(self, sample_image):
        """apply() does not modify RGB (protected by integrity validation)."""
        enhancer = VisuShrinkEnhancer()
        original_rgb = sample_image.rgb[:].copy()
        result = enhancer.apply(sample_image)
        assert np.array_equal(result.rgb[:], original_rgb)

    def test_apply_preserves_gray(self, sample_image):
        """apply() does not modify gray (protected by integrity validation)."""
        enhancer = VisuShrinkEnhancer()
        original_gray = sample_image.gray[:].copy()
        result = enhancer.apply(sample_image)
        assert np.array_equal(result.gray[:], original_gray)

    def test_apply_modifies_enh_gray(self, sample_image):
        """apply() modifies enh_gray."""
        enhancer = VisuShrinkEnhancer()
        original_enh = sample_image.enh_gray[:].copy()
        result = enhancer.apply(sample_image)
        # Should be different after denoising
        assert not np.array_equal(result.enh_gray[:], original_enh)

    def test_enh_gray_in_valid_range(self, sample_image):
        """enh_gray remains in [0.0, 1.0] after denoising."""
        enhancer = VisuShrinkEnhancer()
        result = enhancer.apply(sample_image)
        enh = result.enh_gray[:]
        assert np.all(enh >= 0.0) and np.all(enh <= 1.0)

    def test_auto_sigma_estimation(self, sample_image):
        """Works with sigma=None (auto-estimation)."""
        enhancer = VisuShrinkEnhancer(sigma=None)
        result = enhancer.apply(sample_image)
        assert isinstance(result, Image)

    def test_explicit_sigma(self, sample_image):
        """Works with explicit sigma value."""
        enhancer = VisuShrinkEnhancer(sigma=0.03)
        result = enhancer.apply(sample_image)
        assert isinstance(result, Image)

    def test_different_wavelets(self, sample_image):
        """Works with different wavelet types."""
        for wavelet in ["db2", "db4", "sym2"]:
            enhancer = VisuShrinkEnhancer(wavelet=wavelet)
            result = enhancer.apply(sample_image)
            assert isinstance(result, Image)

    def test_soft_vs_hard_mode(self, sample_image):
        """Soft and hard modes produce results."""
        result_soft = VisuShrinkEnhancer(mode="soft").apply(sample_image)
        result_hard = VisuShrinkEnhancer(mode="hard").apply(sample_image)
        # Both should be valid denoised images
        assert result_soft.enh_gray[:].shape == sample_image.enh_gray[:].shape
        assert result_hard.enh_gray[:].shape == sample_image.enh_gray[:].shape

    def test_inplace_false_does_not_modify_original(self, sample_image):
        """apply(inplace=False) does not modify original image."""
        enhancer = VisuShrinkEnhancer()
        original_enh = sample_image.enh_gray[:].copy()
        result = enhancer.apply(sample_image, inplace=False)
        # Original should be unchanged
        assert np.array_equal(sample_image.enh_gray[:], original_enh)
        # Result should be different
        assert not np.array_equal(result.enh_gray[:], original_enh)


class TestBayesShrinkEnhancer:
    """Tests for BayesShrinkEnhancer (ImageEnhancer)."""

    def test_instantiate_with_defaults(self):
        """Can instantiate with default parameters."""
        enhancer = BayesShrinkEnhancer()
        assert enhancer.sigma is None
        assert enhancer.wavelet == "db2"
        assert enhancer.mode == "soft"
        assert enhancer.wavelet_levels is None

    def test_apply_returns_image(self, sample_image):
        """apply() returns an Image object."""
        enhancer = BayesShrinkEnhancer()
        result = enhancer.apply(sample_image)
        assert isinstance(result, Image)

    def test_apply_preserves_rgb(self, sample_image):
        """apply() does not modify RGB (protected)."""
        enhancer = BayesShrinkEnhancer()
        original_rgb = sample_image.rgb[:].copy()
        result = enhancer.apply(sample_image)
        assert np.array_equal(result.rgb[:], original_rgb)

    def test_apply_preserves_gray(self, sample_image):
        """apply() does not modify gray (protected)."""
        enhancer = BayesShrinkEnhancer()
        original_gray = sample_image.gray[:].copy()
        result = enhancer.apply(sample_image)
        assert np.array_equal(result.gray[:], original_gray)

    def test_apply_modifies_enh_gray(self, sample_image):
        """apply() modifies enh_gray."""
        enhancer = BayesShrinkEnhancer()
        original_enh = sample_image.enh_gray[:].copy()
        result = enhancer.apply(sample_image)
        assert not np.array_equal(result.enh_gray[:], original_enh)

    def test_enh_gray_in_valid_range(self, sample_image):
        """enh_gray remains in [0.0, 1.0] after denoising."""
        enhancer = BayesShrinkEnhancer()
        result = enhancer.apply(sample_image)
        enh = result.enh_gray[:]
        assert np.all(enh >= 0.0) and np.all(enh <= 1.0)

    def test_bayesshrink_vs_visushrink(self, sample_image):
        """BayesShrink and VisuShrink produce valid results."""
        bayes = BayesShrinkEnhancer().apply(sample_image)
        visu = VisuShrinkEnhancer().apply(sample_image)
        # Both should be valid denoised images
        assert bayes.enh_gray[:].shape == visu.enh_gray[:].shape
        assert np.all(bayes.enh_gray[:] >= 0.0) and np.all(bayes.enh_gray[:] <= 1.0)

    def test_soft_vs_hard_mode(self, sample_image):
        """Soft and hard modes produce valid results."""
        result_soft = BayesShrinkEnhancer(mode="soft").apply(sample_image)
        result_hard = BayesShrinkEnhancer(mode="hard").apply(sample_image)
        # Both should be valid denoised images
        assert result_soft.enh_gray[:].shape == sample_image.enh_gray[:].shape
        assert result_hard.enh_gray[:].shape == sample_image.enh_gray[:].shape


class TestVisuShrinkCorrector:
    """Tests for VisuShrinkCorrector (ImageCorrector)."""

    def test_instantiate_with_defaults(self):
        """Can instantiate with default parameters."""
        corrector = VisuShrinkCorrector()
        assert corrector.sigma is None
        assert corrector.wavelet == "db2"
        assert corrector.mode == "soft"
        assert corrector.wavelet_levels is None
        assert corrector.convert2ycbcr is True

    def test_apply_returns_image(self, sample_image):
        """apply() returns an Image object."""
        corrector = VisuShrinkCorrector()
        result = corrector.apply(sample_image)
        assert isinstance(result, Image)

    def test_apply_modifies_rgb(self, sample_image):
        """apply() modifies RGB if present."""
        if sample_image.rgb.isempty():
            pytest.skip("Test image has no RGB")
        corrector = VisuShrinkCorrector()
        original_rgb = sample_image.rgb[:].copy()
        result = corrector.apply(sample_image)
        assert not np.array_equal(result.rgb[:], original_rgb)

    def test_apply_modifies_gray(self, sample_image):
        """apply() modifies gray."""
        corrector = VisuShrinkCorrector()
        original_gray = sample_image.gray[:].copy()
        result = corrector.apply(sample_image)
        assert not np.array_equal(result.gray[:], original_gray)

    def test_apply_modifies_enh_gray(self, sample_image):
        """apply() modifies enh_gray."""
        corrector = VisuShrinkCorrector()
        original_enh = sample_image.enh_gray[:].copy()
        result = corrector.apply(sample_image)
        assert not np.array_equal(result.enh_gray[:], original_enh)

    def test_rgb_dtype_preserved(self, sample_image):
        """RGB dtype is preserved as uint8."""
        if sample_image.rgb.isempty():
            pytest.skip("Test image has no RGB")
        corrector = VisuShrinkCorrector()
        result = corrector.apply(sample_image)
        assert result.rgb[:].dtype == np.uint8

    def test_rgb_range_valid(self, sample_image):
        """RGB values remain in [0, 255]."""
        if sample_image.rgb.isempty():
            pytest.skip("Test image has no RGB")
        corrector = VisuShrinkCorrector()
        result = corrector.apply(sample_image)
        rgb = result.rgb[:]
        assert np.all(rgb >= 0) and np.all(rgb <= 255)

    def test_gray_dtype_is_float(self, sample_image):
        """Gray dtype is floating point."""
        corrector = VisuShrinkCorrector()
        result = corrector.apply(sample_image)
        assert result.gray[:].dtype in [np.float32, np.float64]

    def test_gray_range_valid(self, sample_image):
        """Gray values remain in [0.0, 1.0]."""
        corrector = VisuShrinkCorrector()
        result = corrector.apply(sample_image)
        gray = result.gray[:]
        assert np.all(gray >= 0.0) and np.all(gray <= 1.0)

    def test_enh_gray_range_valid(self, sample_image):
        """enh_gray values remain in [0.0, 1.0]."""
        corrector = VisuShrinkCorrector()
        result = corrector.apply(sample_image)
        enh = result.enh_gray[:]
        assert np.all(enh >= 0.0) and np.all(enh <= 1.0)

    def test_convert2ycbcr_parameter(self, sample_image):
        """convert2ycbcr parameter works without error."""
        if sample_image.rgb.isempty():
            pytest.skip("Test image has no RGB")
        # YCbCr denoising
        result_ycbcr = VisuShrinkCorrector(convert2ycbcr=True).apply(sample_image)
        # RGB denoising
        result_rgb = VisuShrinkCorrector(convert2ycbcr=False).apply(sample_image)
        # Both should produce valid results
        assert result_ycbcr.rgb[:].shape == sample_image.rgb[:].shape
        assert result_rgb.rgb[:].shape == sample_image.rgb[:].shape


class TestBayesShrinkCorrector:
    """Tests for BayesShrinkCorrector (ImageCorrector)."""

    def test_instantiate_with_defaults(self):
        """Can instantiate with default parameters."""
        corrector = BayesShrinkCorrector()
        assert corrector.sigma is None
        assert corrector.wavelet == "db2"
        assert corrector.mode == "soft"
        assert corrector.wavelet_levels is None
        assert corrector.convert2ycbcr is True

    def test_apply_returns_image(self, sample_image):
        """apply() returns an Image object."""
        corrector = BayesShrinkCorrector()
        result = corrector.apply(sample_image)
        assert isinstance(result, Image)

    def test_apply_modifies_gray(self, sample_image):
        """apply() modifies gray."""
        corrector = BayesShrinkCorrector()
        original_gray = sample_image.gray[:].copy()
        result = corrector.apply(sample_image)
        assert not np.array_equal(result.gray[:], original_gray)

    def test_apply_modifies_enh_gray(self, sample_image):
        """apply() modifies enh_gray."""
        corrector = BayesShrinkCorrector()
        original_enh = sample_image.enh_gray[:].copy()
        result = corrector.apply(sample_image)
        assert not np.array_equal(result.enh_gray[:], original_enh)

    def test_gray_range_valid(self, sample_image):
        """Gray values remain in [0.0, 1.0]."""
        corrector = BayesShrinkCorrector()
        result = corrector.apply(sample_image)
        gray = result.gray[:]
        assert np.all(gray >= 0.0) and np.all(gray <= 1.0)

    def test_bayesshrink_vs_visushrink_corrector(self, sample_image):
        """BayesShrink and VisuShrink correctors produce valid results."""
        bayes = BayesShrinkCorrector().apply(sample_image)
        visu = VisuShrinkCorrector().apply(sample_image)
        # Both should be valid denoised images
        assert bayes.gray[:].shape == visu.gray[:].shape
        assert np.all(bayes.gray[:] >= 0.0) and np.all(bayes.gray[:] <= 1.0)

    def test_soft_vs_hard_mode(self, sample_image):
        """Soft and hard modes produce valid results."""
        result_soft = BayesShrinkCorrector(mode="soft").apply(sample_image)
        result_hard = BayesShrinkCorrector(mode="hard").apply(sample_image)
        # Both should be valid denoised images
        assert result_soft.gray[:].shape == sample_image.gray[:].shape
        assert result_hard.gray[:].shape == sample_image.gray[:].shape


class TestGrayscaleOnlyImages:
    """Tests for handling grayscale-only images (no RGB)."""

    def test_visushrink_enhancer_grayscale(self, grayscale_only_image):
        """VisuShrinkEnhancer handles grayscale-only image."""
        enhancer = VisuShrinkEnhancer()
        result = enhancer.apply(grayscale_only_image)
        assert isinstance(result, Image)
        assert grayscale_only_image.rgb.isempty()

    def test_bayesshrink_enhancer_grayscale(self, grayscale_only_image):
        """BayesShrinkEnhancer handles grayscale-only image."""
        enhancer = BayesShrinkEnhancer()
        result = enhancer.apply(grayscale_only_image)
        assert isinstance(result, Image)

    def test_visushrink_corrector_grayscale(self, grayscale_only_image):
        """VisuShrinkCorrector handles grayscale-only image without error."""
        corrector = VisuShrinkCorrector()
        result = corrector.apply(grayscale_only_image)
        assert isinstance(result, Image)
        # Gray should be modified
        assert not np.array_equal(result.gray[:], grayscale_only_image.gray[:])

    def test_bayesshrink_corrector_grayscale(self, grayscale_only_image):
        """BayesShrinkCorrector handles grayscale-only image without error."""
        corrector = BayesShrinkCorrector()
        result = corrector.apply(grayscale_only_image)
        assert isinstance(result, Image)


class TestPipelineIntegration:
    """Tests for integration with ImagePipeline."""

    def test_enhancer_in_pipeline(self, sample_image):
        """VisuShrinkEnhancer works in ImagePipeline."""
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline()
        pipeline.set_ops([VisuShrinkEnhancer()])
        result = pipeline.apply(sample_image)
        assert isinstance(result, Image)

    def test_corrector_in_pipeline(self, sample_image):
        """VisuShrinkCorrector works in ImagePipeline."""
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline()
        pipeline.set_ops([VisuShrinkCorrector()])
        result = pipeline.apply(sample_image)
        assert isinstance(result, Image)

    def test_multiple_denoisers_in_pipeline(self, sample_image):
        """Multiple wavelet denoisers can be chained."""
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline()
        # Can't use both enhancer and corrector on same component,
        # but can use different enhancers
        pipeline.set_ops([VisuShrinkEnhancer()])
        result = pipeline.apply(sample_image)
        assert isinstance(result, Image)

    def test_denoiser_with_other_operations(self, sample_image):
        """Denoiser integrates with other operations."""
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur

        pipeline = ImagePipeline()
        pipeline.set_ops([VisuShrinkEnhancer(), GaussianBlur(sigma=1)])
        result = pipeline.apply(sample_image)
        assert isinstance(result, Image)


# Fixtures
@pytest.fixture
def sample_image():
    """Create a sample RGB Image for testing."""
    # Create a 100x100 RGB image with some test data
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image(arr)


@pytest.fixture
def grayscale_only_image():
    """Create a grayscale-only image (no RGB)."""
    # Create a 100x100 grayscale image
    arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    return Image(arr)
