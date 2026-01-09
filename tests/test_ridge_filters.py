"""Tests for Sato and Meijering ridge detection filters.

This module tests ridge detection enhancers that use the Hessian matrix
eigenvalues to detect continuous ridge-like structures in agar plate images.
Both filters are useful for detecting filamentous colonies, mycelial networks,
and other elongated structures.
"""

import pytest
import numpy as np

from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synthetic_detection_image
from phenotypic.detect import RoundPeaksDetector
from phenotypic.enhance import SatoRidgeFilter, MeijeringRidgeFilter


@pytest.fixture
def test_image():
    """Load a test image for ridge filter testing."""
    array = load_synthetic_detection_image()
    image = Image(array)
    # Pre-detect to have valid image state
    RoundPeaksDetector().apply(image, inplace=True)
    return image


class TestSatoRidgeFilter:
    """Tests for SatoRidgeFilter enhancer."""

    def test_sato_instantiation_with_defaults(self):
        """SatoRidgeFilter should instantiate with default parameters."""
        filter_obj = SatoRidgeFilter()
        assert filter_obj.sigmas == (1, 2, 3)
        assert filter_obj.black_ridges is True
        assert filter_obj.mode == 'reflect'
        assert filter_obj.cval == 0

    def test_sato_custom_parameters(self):
        """SatoRidgeFilter should accept custom parameters."""
        sigmas = (0.5, 1, 1.5, 2)
        filter_obj = SatoRidgeFilter(
            sigmas=sigmas,
            black_ridges=False,
            mode='constant',
            cval=1.0
        )
        assert filter_obj.sigmas == sigmas
        assert filter_obj.black_ridges is False
        assert filter_obj.mode == 'constant'
        assert filter_obj.cval == 1.0

    def test_sato_apply_returns_image(self, test_image):
        """SatoRidgeFilter.apply() should return an Image object."""
        filter_obj = SatoRidgeFilter()
        result = filter_obj.apply(test_image)
        assert isinstance(result, Image)

    def test_sato_modifies_only_enh_gray(self, test_image):
        """SatoRidgeFilter should only modify enh_gray, not rgb or gray."""
        filter_obj = SatoRidgeFilter()

        # Store original values
        original_rgb = test_image.rgb[:].copy()
        original_gray = test_image.gray[:].copy()

        # Apply filter
        result = filter_obj.apply(test_image)

        # Check that RGB and gray are unchanged
        assert np.array_equal(result.rgb[:], original_rgb), \
            "SatoRidgeFilter modified RGB data"
        assert np.array_equal(result.gray[:], original_gray), \
            "SatoRidgeFilter modified gray data"

    def test_sato_output_shape(self, test_image):
        """SatoRidgeFilter output should have same shape as input."""
        filter_obj = SatoRidgeFilter()
        result = filter_obj.apply(test_image)
        assert result.enh_gray[:].shape == test_image.enh_gray[:].shape

    def test_sato_output_is_float(self, test_image):
        """SatoRidgeFilter output should be float (probability map)."""
        filter_obj = SatoRidgeFilter()
        result = filter_obj.apply(test_image)
        # scikit-image filters return float values
        assert np.issubdtype(result.enh_gray[:].dtype, np.floating)

    def test_sato_black_ridges_true(self, test_image):
        """SatoRidgeFilter with black_ridges=True detects dark structures."""
        filter_true = SatoRidgeFilter(black_ridges=True)
        result_true = filter_true.apply(test_image)
        # Just verify it runs without error and produces output
        assert result_true.enh_gray[:].size > 0

    def test_sato_black_ridges_false(self, test_image):
        """SatoRidgeFilter with black_ridges=False detects bright structures."""
        filter_false = SatoRidgeFilter(black_ridges=False)
        result_false = filter_false.apply(test_image)
        # Just verify it runs without error and produces output
        assert result_false.enh_gray[:].size > 0

    def test_sato_different_sigma_ranges(self, test_image):
        """SatoRidgeFilter should work with different sigma ranges."""
        sigmas_list = [
            (1,),
            (1, 2, 3),
            (0.5, 1, 1.5, 2, 2.5),
            range(1, 5),
            [1.0, 2.0, 3.0],
        ]
        for sigmas in sigmas_list:
            filter_obj = SatoRidgeFilter(sigmas=sigmas)
            result = filter_obj.apply(test_image)
            assert result.enh_gray[:].size > 0, \
                f"SatoRidgeFilter failed with sigmas={sigmas}"

    def test_sato_different_modes(self, test_image):
        """SatoRidgeFilter should work with different boundary modes."""
        modes = ['reflect', 'constant', 'nearest', 'wrap', 'mirror']
        for mode in modes:
            filter_obj = SatoRidgeFilter(mode=mode)
            result = filter_obj.apply(test_image)
            assert result.enh_gray[:].size > 0, \
                f"SatoRidgeFilter failed with mode={mode}"

    def test_sato_inplace_modification(self, test_image):
        """SatoRidgeFilter.apply() should support inplace=True."""
        filter_obj = SatoRidgeFilter()
        original_id = id(test_image)
        result = filter_obj.apply(test_image, inplace=True)
        assert id(result) == original_id, \
            "inplace=True should return the same Image object"

    def test_sato_reproducibility(self, test_image):
        """SatoRidgeFilter should produce identical results on repeated calls."""
        filter_obj = SatoRidgeFilter()
        result1 = filter_obj.apply(test_image)
        result2 = filter_obj.apply(test_image)
        assert np.allclose(result1.enh_gray[:], result2.enh_gray[:]), \
            "SatoRidgeFilter is not reproducible"

    def test_sato_in_pipeline(self, test_image):
        """SatoRidgeFilter should work in ImagePipeline."""
        pipeline = ImagePipeline([
            SatoRidgeFilter(sigmas=(1, 2))
        ])
        result = pipeline.apply(test_image)
        assert isinstance(result, Image)
        assert result.enh_gray[:].size > 0


class TestMeijeringRidgeFilter:
    """Tests for MeijeringRidgeFilter enhancer."""

    def test_meijering_instantiation_with_defaults(self):
        """MeijeringRidgeFilter should instantiate with default parameters."""
        filter_obj = MeijeringRidgeFilter()
        assert filter_obj.sigmas == (1, 2, 3)
        assert filter_obj.alpha is None
        assert filter_obj.black_ridges is True
        assert filter_obj.mode == 'reflect'
        assert filter_obj.cval == 0

    def test_meijering_custom_parameters(self):
        """MeijeringRidgeFilter should accept custom parameters."""
        sigmas = (0.5, 1, 1.5, 2)
        filter_obj = MeijeringRidgeFilter(
            sigmas=sigmas,
            alpha=-0.5,
            black_ridges=False,
            mode='constant',
            cval=1.0
        )
        assert filter_obj.sigmas == sigmas
        assert filter_obj.alpha == -0.5
        assert filter_obj.black_ridges is False
        assert filter_obj.mode == 'constant'
        assert filter_obj.cval == 1.0

    def test_meijering_apply_returns_image(self, test_image):
        """MeijeringRidgeFilter.apply() should return an Image object."""
        filter_obj = MeijeringRidgeFilter()
        result = filter_obj.apply(test_image)
        assert isinstance(result, Image)

    def test_meijering_modifies_only_enh_gray(self, test_image):
        """MeijeringRidgeFilter should only modify enh_gray, not rgb or gray."""
        filter_obj = MeijeringRidgeFilter()

        # Store original values
        original_rgb = test_image.rgb[:].copy()
        original_gray = test_image.gray[:].copy()

        # Apply filter
        result = filter_obj.apply(test_image)

        # Check that RGB and gray are unchanged
        assert np.array_equal(result.rgb[:], original_rgb), \
            "MeijeringRidgeFilter modified RGB data"
        assert np.array_equal(result.gray[:], original_gray), \
            "MeijeringRidgeFilter modified gray data"

    def test_meijering_output_shape(self, test_image):
        """MeijeringRidgeFilter output should have same shape as input."""
        filter_obj = MeijeringRidgeFilter()
        result = filter_obj.apply(test_image)
        assert result.enh_gray[:].shape == test_image.enh_gray[:].shape

    def test_meijering_output_is_float(self, test_image):
        """MeijeringRidgeFilter output should be float (probability map)."""
        filter_obj = MeijeringRidgeFilter()
        result = filter_obj.apply(test_image)
        # scikit-image filters return float values
        assert np.issubdtype(result.enh_gray[:].dtype, np.floating)

    def test_meijering_black_ridges_true(self, test_image):
        """MeijeringRidgeFilter with black_ridges=True detects dark structures."""
        filter_true = MeijeringRidgeFilter(black_ridges=True)
        result_true = filter_true.apply(test_image)
        # Just verify it runs without error and produces output
        assert result_true.enh_gray[:].size > 0

    def test_meijering_black_ridges_false(self, test_image):
        """MeijeringRidgeFilter with black_ridges=False detects bright structures."""
        filter_false = MeijeringRidgeFilter(black_ridges=False)
        result_false = filter_false.apply(test_image)
        # Just verify it runs without error and produces output
        assert result_false.enh_gray[:].size > 0

    def test_meijering_alpha_none_vs_specified(self, test_image):
        """MeijeringRidgeFilter should work with alpha=None and specific values."""
        filter_none = MeijeringRidgeFilter(alpha=None)
        result_none = filter_none.apply(test_image)

        filter_specified = MeijeringRidgeFilter(alpha=-0.5)
        result_specified = filter_specified.apply(test_image)

        # Both should produce valid output
        assert result_none.enh_gray[:].size > 0
        assert result_specified.enh_gray[:].size > 0

    def test_meijering_different_sigma_ranges(self, test_image):
        """MeijeringRidgeFilter should work with different sigma ranges."""
        sigmas_list = [
            (1,),
            (1, 2, 3),
            (0.5, 1, 1.5, 2, 2.5),
            range(1, 5),
            [1.0, 2.0, 3.0],
        ]
        for sigmas in sigmas_list:
            filter_obj = MeijeringRidgeFilter(sigmas=sigmas)
            result = filter_obj.apply(test_image)
            assert result.enh_gray[:].size > 0, \
                f"MeijeringRidgeFilter failed with sigmas={sigmas}"

    def test_meijering_different_modes(self, test_image):
        """MeijeringRidgeFilter should work with different boundary modes."""
        modes = ['reflect', 'constant', 'nearest', 'wrap', 'mirror']
        for mode in modes:
            filter_obj = MeijeringRidgeFilter(mode=mode)
            result = filter_obj.apply(test_image)
            assert result.enh_gray[:].size > 0, \
                f"MeijeringRidgeFilter failed with mode={mode}"

    def test_meijering_inplace_modification(self, test_image):
        """MeijeringRidgeFilter.apply() should support inplace=True."""
        filter_obj = MeijeringRidgeFilter()
        original_id = id(test_image)
        result = filter_obj.apply(test_image, inplace=True)
        assert id(result) == original_id, \
            "inplace=True should return the same Image object"

    def test_meijering_reproducibility(self, test_image):
        """MeijeringRidgeFilter should produce identical results on repeated calls."""
        filter_obj = MeijeringRidgeFilter()
        result1 = filter_obj.apply(test_image)
        result2 = filter_obj.apply(test_image)
        assert np.allclose(result1.enh_gray[:], result2.enh_gray[:]), \
            "MeijeringRidgeFilter is not reproducible"

    def test_meijering_in_pipeline(self, test_image):
        """MeijeringRidgeFilter should work in ImagePipeline."""
        pipeline = ImagePipeline([
            MeijeringRidgeFilter(sigmas=(1, 2))
        ])
        result = pipeline.apply(test_image)
        assert isinstance(result, Image)
        assert result.enh_gray[:].size > 0


class TestRidgeFiltersComparison:
    """Tests comparing Sato and Meijering ridge filters."""

    def test_both_filters_produce_different_results(self, test_image):
        """Sato and Meijering should produce different ridge maps."""
        sato = SatoRidgeFilter(sigmas=(1, 2, 3))
        meijering = MeijeringRidgeFilter(sigmas=(1, 2, 3))

        result_sato = sato.apply(test_image)
        result_meijering = meijering.apply(test_image)

        # Results should be different (different algorithms)
        assert not np.allclose(result_sato.enh_gray[:], result_meijering.enh_gray[:]), \
            "Sato and Meijering should produce different results"

    def test_filters_in_combined_pipeline(self, test_image):
        """Both filters should work together in a pipeline."""
        from phenotypic.enhance import GaussianBlur

        pipeline = ImagePipeline([
            GaussianBlur(sigma=1),
            SatoRidgeFilter(sigmas=(1, 2)),
            MeijeringRidgeFilter(sigmas=(1, 2)),
        ])
        result = pipeline.apply(test_image)
        assert isinstance(result, Image)
        assert result.enh_gray[:].size > 0

    def test_chaining_both_filters(self, test_image):
        """Chaining both filters sequentially should work."""
        sato = SatoRidgeFilter(sigmas=(1, 2))
        meijering = MeijeringRidgeFilter(sigmas=(1, 2))

        result = sato.apply(test_image)
        result = meijering.apply(result)

        assert isinstance(result, Image)
        assert result.enh_gray[:].size > 0


class TestRidgeFiltersEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_sato_with_single_sigma(self, test_image):
        """SatoRidgeFilter should work with a single sigma value."""
        filter_obj = SatoRidgeFilter(sigmas=(2,))
        result = filter_obj.apply(test_image)
        assert result.enh_gray[:].size > 0

    def test_meijering_with_single_sigma(self, test_image):
        """MeijeringRidgeFilter should work with a single sigma value."""
        filter_obj = MeijeringRidgeFilter(sigmas=(2,))
        result = filter_obj.apply(test_image)
        assert result.enh_gray[:].size > 0

    def test_sato_with_very_small_sigmas(self, test_image):
        """SatoRidgeFilter should work with very small sigmas."""
        filter_obj = SatoRidgeFilter(sigmas=(0.5, 0.7))
        result = filter_obj.apply(test_image)
        assert result.enh_gray[:].size > 0

    def test_meijering_with_very_small_sigmas(self, test_image):
        """MeijeringRidgeFilter should work with very small sigmas."""
        filter_obj = MeijeringRidgeFilter(sigmas=(0.5, 0.7))
        result = filter_obj.apply(test_image)
        assert result.enh_gray[:].size > 0

    def test_sato_with_large_sigmas(self, test_image):
        """SatoRidgeFilter should work with large sigmas."""
        filter_obj = SatoRidgeFilter(sigmas=(5, 10, 15))
        result = filter_obj.apply(test_image)
        assert result.enh_gray[:].size > 0

    def test_meijering_with_large_sigmas(self, test_image):
        """MeijeringRidgeFilter should work with large sigmas."""
        filter_obj = MeijeringRidgeFilter(sigmas=(5, 10, 15))
        result = filter_obj.apply(test_image)
        assert result.enh_gray[:].size > 0
