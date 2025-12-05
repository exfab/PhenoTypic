"""
Comprehensive test suite for morphological enhancer classes.

Tests cover initialization, enhancement on different image types,
parameter variations, edge cases, and reproducibility for all morphological
enhancers: MorphologicalOpening, MorphologicalClosing, MorphologicalErosion,
MorphologicalDilation, MorphologicalGradient, and BlackTophatEnhancer.
"""

import pytest
import numpy as np
import phenotypic
from phenotypic.enhance import (
    MorphologicalOpening,
    MorphologicalClosing,
    MorphologicalErosion,
    MorphologicalDilation,
    MorphologicalGradient,
    BlackTophatEnhancer,
)
from phenotypic.data import load_plate_12hr, load_plate_72hr

from .resources.TestHelper import timeit


# List of all morphological enhancer classes for parametric testing
MORPHOLOGICAL_ENHANCERS = [
    MorphologicalOpening,
    MorphologicalClosing,
    MorphologicalErosion,
    MorphologicalDilation,
    MorphologicalGradient,
    BlackTophatEnhancer,
]


class TestMorphologicalEnhancersInitialization:
    """Test initialization and parameter validation for all morphological enhancers."""

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_default_initialization(self, enhancer_class):
        """Test that all enhancers instantiate with default parameters."""
        enhancer = enhancer_class()
        assert enhancer is not None
        assert isinstance(enhancer, enhancer_class)
        # Check that shape and radius are set
        if enhancer_class != BlackTophatEnhancer:
            assert hasattr(enhancer, "shape")
            assert hasattr(enhancer, "radius")
            assert enhancer.shape == "disk"
            assert enhancer.radius >= 1

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    @pytest.mark.parametrize("shape", ["disk", "square", "diamond"])
    def test_initialization_with_different_shapes(self, enhancer_class, shape):
        """Test initialization with all supported footprint shapes."""
        enhancer = enhancer_class(shape=shape)
        assert enhancer.shape == shape

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    @pytest.mark.parametrize("radius", [1, 2, 3, 5])
    def test_initialization_with_different_radii(self, enhancer_class, radius):
        """Test initialization with various radius values."""
        if enhancer_class == BlackTophatEnhancer:
            enhancer = enhancer_class(radius=radius)
        else:
            enhancer = enhancer_class(radius=radius)

        # Check that radius is set correctly
        if enhancer_class == BlackTophatEnhancer:
            assert enhancer.radius == radius
        else:
            assert enhancer.radius == radius

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_invalid_shape_raises_error(self, enhancer_class):
        """Test that invalid shape raises ValueError."""
        with pytest.raises(ValueError, match="shape must be one of"):
            enhancer_class(shape="invalid_shape")

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_invalid_radius_raises_error(self, enhancer_class):
        """Test that invalid radius raises ValueError."""
        with pytest.raises(ValueError, match="radius must be"):
            enhancer_class(radius=0)

        with pytest.raises(ValueError, match="radius must be"):
            enhancer_class(radius=-1)

        with pytest.raises(ValueError, match="radius must be"):
            enhancer_class(radius=1.5)


class TestMorphologicalEnhancersOnImage:
    """Test enhancers on various image types."""

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_enhancement_on_plate_12hr(self, enhancer_class):
        """Test enhancement on 12-hour plate."""
        image = phenotypic.Image(load_plate_12hr())
        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        assert result is not None
        assert not result.isempty()
        # Enhanced grayscale should be modified
        assert result.enh_gray[:].shape == image.gray[:].shape

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_enhancement_on_plate_72hr(self, enhancer_class):
        """Test enhancement on 72-hour plate."""
        image = phenotypic.Image(load_plate_72hr())
        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        assert result is not None
        assert not result.isempty()
        assert result.enh_gray[:].shape == image.gray[:].shape

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_inplace_enhancement(self, enhancer_class):
        """Test in-place enhancement modifies original."""
        image = phenotypic.Image(load_plate_12hr())
        original_enh = image.enh_gray[:].copy()

        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=True)

        assert result is image
        # Enhanced grayscale should be different
        assert not np.array_equal(image.enh_gray[:], original_enh)

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_copy_enhancement_preserves_original(self, enhancer_class):
        """Test that copy mode (default) doesn't modify original."""
        image = phenotypic.Image(load_plate_12hr())
        original_enh = image.enh_gray[:].copy()
        original_rgb = image.rgb[:].copy()
        original_gray = image.gray[:].copy()

        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        # Original should be unchanged
        assert np.array_equal(image.enh_gray[:], original_enh)
        assert np.array_equal(image.rgb[:], original_rgb)
        assert np.array_equal(image.gray[:], original_gray)
        # Result should be different Image object
        assert result is not image


class TestMorphologicalEnhancersProtectImageData:
    """Test that enhancers only modify enh_gray and protect RGB/gray."""

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_rgb_unchanged(self, enhancer_class):
        """Test that RGB data is protected."""
        image = phenotypic.Image(load_plate_12hr())
        original_rgb = image.rgb[:].copy()

        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        assert np.array_equal(result.rgb[:], original_rgb)

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_gray_unchanged(self, enhancer_class):
        """Test that grayscale data is protected."""
        image = phenotypic.Image(load_plate_12hr())
        original_gray = image.gray[:].copy()

        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        assert np.array_equal(result.gray[:], original_gray)

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_enh_gray_modified(self, enhancer_class):
        """Test that enhanced grayscale is actually modified."""
        image = phenotypic.Image(load_plate_12hr())
        original_enh = image.enh_gray[:].copy()

        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        # Enhanced grayscale should be different (operation should modify it)
        assert not np.array_equal(result.enh_gray[:], original_enh)


class TestMorphologicalEnhancersWithDifferentParameters:
    """Test enhancers with various parameter combinations."""

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    @pytest.mark.parametrize("radius", [1, 3, 5, 7])
    def test_different_radii_affect_output(self, enhancer_class, radius):
        """Test that different radii produce different results."""
        image1 = phenotypic.Image(load_plate_12hr())
        image2 = phenotypic.Image(load_plate_12hr())

        enhancer1 = enhancer_class(radius=1)
        enhancer2 = enhancer_class(radius=radius)

        result1 = enhancer1.apply(image1, inplace=False)
        result2 = enhancer2.apply(image2, inplace=False)

        # Different radii should produce different results
        if radius != 1:
            assert not np.array_equal(result1.enh_gray[:], result2.enh_gray[:])

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    @pytest.mark.parametrize("shape", ["disk", "square", "diamond"])
    def test_different_shapes_valid_application(self, enhancer_class, shape):
        """Test that different shapes can be applied successfully."""
        image = phenotypic.Image(load_plate_12hr())

        enhancer = enhancer_class(shape=shape)
        result = enhancer.apply(image, inplace=False)

        # Should complete without error and produce valid result
        assert result is not None
        assert result.enh_gray[:].shape == image.gray[:].shape


class TestMorphologicalEnhancersEdgeCases:
    """Test edge cases and unusual inputs."""

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_blank_image(self, enhancer_class):
        """Test on blank/uniform image."""
        blank_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        image = phenotypic.Image(blank_array)
        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        # Should complete without error
        assert result is not None
        assert not result.isempty()

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_small_image(self, enhancer_class):
        """Test on very small image."""
        small_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        image = phenotypic.Image(small_array)
        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        assert result is not None
        assert not result.isempty()

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_large_radius_on_small_image(self, enhancer_class):
        """Test with large radius on small image."""
        small_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        image = phenotypic.Image(small_array)

        # Large radius relative to image size
        enhancer = enhancer_class(radius=20)
        result = enhancer.apply(image, inplace=False)

        assert result is not None

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_black_image(self, enhancer_class):
        """Test on all-black image."""
        black_array = np.zeros((100, 100, 3), dtype=np.uint8)
        image = phenotypic.Image(black_array)
        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        assert result is not None


class TestMorphologicalEnhancersReproducibility:
    """Test output consistency and reproducibility."""

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_reproducibility_same_enhancer(self, enhancer_class):
        """Test that enhancement is reproducible with same enhancer."""
        image1 = phenotypic.Image(load_plate_12hr())
        image2 = phenotypic.Image(load_plate_12hr())

        enhancer = enhancer_class(radius=3, shape="disk")

        result1 = enhancer.apply(image1, inplace=False)
        result2 = enhancer.apply(image2, inplace=False)

        # Results should be identical
        assert np.array_equal(result1.enh_gray[:], result2.enh_gray[:])

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_reproducibility_multiple_applications(self, enhancer_class):
        """Test reproducibility when applying multiple times."""
        image = phenotypic.Image(load_plate_12hr())

        enhancer = enhancer_class(radius=3, shape="disk")

        result1 = enhancer.apply(image, inplace=False)
        result2 = enhancer.apply(image, inplace=False)

        # Multiple applications to same source should give same result
        assert np.array_equal(result1.enh_gray[:], result2.enh_gray[:])

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_reproducibility_different_instances(self, enhancer_class):
        """Test that different instances with same parameters are reproducible."""
        image1 = phenotypic.Image(load_plate_12hr())
        image2 = phenotypic.Image(load_plate_12hr())

        enhancer1 = enhancer_class(radius=3, shape="disk")
        enhancer2 = enhancer_class(radius=3, shape="disk")

        result1 = enhancer1.apply(image1, inplace=False)
        result2 = enhancer2.apply(image2, inplace=False)

        # Different instances with same params should give same result
        assert np.array_equal(result1.enh_gray[:], result2.enh_gray[:])


class TestBlackTophatEnhancerSpecific:
    """Tests specific to BlackTophatEnhancer (auto-radius feature)."""

    @timeit
    def test_black_tophat_auto_radius(self):
        """Test that BlackTophatEnhancer auto-calculates radius when None."""
        image = phenotypic.Image(load_plate_12hr())
        enhancer = BlackTophatEnhancer(radius=None)
        result = enhancer.apply(image, inplace=False)

        assert result is not None
        assert not result.isempty()

    @timeit
    def test_black_tophat_explicit_vs_auto_radius(self):
        """Test that explicit radius differs from auto-radius in effects."""
        image1 = phenotypic.Image(load_plate_12hr())
        image2 = phenotypic.Image(load_plate_12hr())

        enhancer_auto = BlackTophatEnhancer(radius=None)
        enhancer_explicit = BlackTophatEnhancer(radius=3)

        result_auto = enhancer_auto.apply(image1, inplace=False)
        result_explicit = enhancer_explicit.apply(image2, inplace=False)

        # Should produce different results
        assert not np.array_equal(result_auto.enh_gray[:], result_explicit.enh_gray[:])


class TestMorphologicalEnhancersOutputShape:
    """Test that output shapes are preserved."""

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_output_shape_matches_input(self, enhancer_class):
        """Test that enh_gray output shape matches input grayscale shape."""
        image = phenotypic.Image(load_plate_12hr())
        input_shape = image.gray[:].shape

        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        assert result.enh_gray[:].shape == input_shape

    @timeit
    @pytest.mark.parametrize("enhancer_class", MORPHOLOGICAL_ENHANCERS)
    def test_output_dtype_preserved(self, enhancer_class):
        """Test that output dtype matches input dtype."""
        image = phenotypic.Image(load_plate_12hr())
        input_dtype = image.enh_gray[:].dtype

        enhancer = enhancer_class()
        result = enhancer.apply(image, inplace=False)

        assert result.enh_gray[:].dtype == input_dtype
