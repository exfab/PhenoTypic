"""Tests for ObjectMask boolean operators (&, |, ^, ~, &=, |=, ^=).

This test suite verifies that bitwise boolean operators on ObjectMask work correctly
with NumPy-style semantics for mask manipulation.
"""

import numpy as np
import pytest
from phenotypic import Image
from skimage.measure import label


@pytest.fixture
def sample_image_with_mask():
    """Create sample image with object mask containing two distinct objects."""
    arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    img = Image(arr)

    # Create simple binary mask with two objects
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 20:40] = True  # First object
    mask[60:80, 60:80] = True  # Second object

    img.objmap[:] = label(mask)
    return img


class TestBasicBooleanOperators:
    """Test basic boolean operators that return np.ndarray."""

    def test_and_with_array(self, sample_image_with_mask):
        """Test AND operator with numpy array."""
        img = sample_image_with_mask
        other = np.ones((100, 100), dtype=bool)
        other[30:50, 30:50] = False

        result = img.objmask & other

        assert isinstance(result, np.ndarray)
        assert result.dtype == int
        assert result.shape == (100, 100)
        # Check intersection: region [30:50, 30:50] should be 0
        assert np.sum(result[30:50, 30:50]) == 0

    def test_and_with_objmask(self, sample_image_with_mask):
        """Test AND operator with another ObjectMask."""
        img1 = sample_image_with_mask

        # Create second image with overlapping mask
        arr2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = Image(arr2)
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[25:45, 25:45] = True  # Overlaps first object
        img2.objmap[:] = label(mask2)

        result = img1.objmask & img2.objmask

        assert isinstance(result, np.ndarray)
        assert result.dtype == int
        # Check intersection exists
        assert np.sum(result) > 0

    def test_and_with_scalar_true(self, sample_image_with_mask):
        """Test AND with True scalar."""
        img = sample_image_with_mask
        result = img.objmask & True

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, img.objmask[:])

    def test_and_with_scalar_false(self, sample_image_with_mask):
        """Test AND with False scalar."""
        img = sample_image_with_mask
        result = img.objmask & False

        assert isinstance(result, np.ndarray)
        assert np.all(result == 0)

    def test_or_with_array(self, sample_image_with_mask):
        """Test OR operator with numpy array."""
        img = sample_image_with_mask
        additional = np.zeros((100, 100), dtype=bool)
        additional[50:60, 50:60] = True

        result = img.objmask | additional

        assert isinstance(result, np.ndarray)
        # Should have original + additional region
        assert np.sum(result) > np.sum(img.objmask[:])

    def test_or_with_objmask(self, sample_image_with_mask):
        """Test OR operator with another ObjectMask."""
        img1 = sample_image_with_mask

        # Create second image with different mask
        arr2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = Image(arr2)
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[45:65, 45:65] = True  # Non-overlapping region
        img2.objmap[:] = label(mask2)

        result = img1.objmask | img2.objmask

        assert isinstance(result, np.ndarray)
        assert np.sum(result) >= np.sum(img1.objmask[:])

    def test_or_with_scalar_true(self, sample_image_with_mask):
        """Test OR with True scalar."""
        img = sample_image_with_mask
        result = img.objmask | True

        assert isinstance(result, np.ndarray)
        assert np.all(result == 1)

    def test_or_with_scalar_false(self, sample_image_with_mask):
        """Test OR with False scalar."""
        img = sample_image_with_mask
        result = img.objmask | False

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, img.objmask[:])

    def test_xor_with_array(self, sample_image_with_mask):
        """Test XOR operator with numpy array."""
        img = sample_image_with_mask
        overlap = np.zeros((100, 100), dtype=bool)
        overlap[25:35, 25:35] = True  # Partially overlaps first object

        result = img.objmask ^ overlap

        assert isinstance(result, np.ndarray)
        # XOR should exclude overlap
        original_overlap_sum = np.sum(img.objmask[25:35, 25:35])
        result_overlap_sum = np.sum(result[25:35, 25:35])
        assert result_overlap_sum < original_overlap_sum

    def test_xor_with_objmask(self, sample_image_with_mask):
        """Test XOR operator with another ObjectMask."""
        img1 = sample_image_with_mask

        # Create second image with overlapping mask
        arr2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = Image(arr2)
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[25:45, 25:45] = True  # Overlaps first object
        img2.objmap[:] = label(mask2)

        result = img1.objmask ^ img2.objmask

        assert isinstance(result, np.ndarray)
        # XOR should find symmetric difference

    def test_invert_operator(self, sample_image_with_mask):
        """Test NOT operator."""
        img = sample_image_with_mask
        result = ~img.objmask

        assert isinstance(result, np.ndarray)
        # Inverted should have opposite values
        assert np.all((result == 0) == (img.objmask[:] == 1))
        assert np.all((result == 1) == (img.objmask[:] == 0))

    def test_shape_mismatch_raises_error(self, sample_image_with_mask):
        """Test that shape mismatch raises ValueError."""
        img = sample_image_with_mask
        wrong_shape = np.ones((50, 50), dtype=bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            _ = img.objmask & wrong_shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            _ = img.objmask | wrong_shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            _ = img.objmask ^ wrong_shape

    def test_unsupported_type_returns_notimplemented(self, sample_image_with_mask):
        """Test that unsupported types return NotImplemented."""
        img = sample_image_with_mask
        # AND with string should return NotImplemented
        result = img.objmask.__and__("invalid")
        assert result is NotImplemented

        # OR with float should return NotImplemented
        result = img.objmask.__or__(3.14)
        assert result is NotImplemented

        # XOR with dict should return NotImplemented
        result = img.objmask.__xor__({"a": 1})
        assert result is NotImplemented


class TestInPlaceBooleanOperators:
    """Test in-place boolean operators that modify the mask."""

    def test_iand_modifies_mask(self, sample_image_with_mask):
        """Test &= operator modifies mask in place."""
        img = sample_image_with_mask
        original_sum = np.sum(img.objmask[:])

        # Restrict to smaller region
        roi = np.zeros((100, 100), dtype=bool)
        roi[20:35, 20:35] = True

        result = (img.objmask.__iand__(roi))

        assert result is img.objmask  # Returns self
        assert np.sum(img.objmask[:]) < original_sum  # Mask reduced

    def test_iand_with_objmask(self, sample_image_with_mask):
        """Test &= operator with another ObjectMask."""
        img1 = sample_image_with_mask
        original_sum = np.sum(img1.objmask[:])

        # Create second image with smaller mask
        arr2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = Image(arr2)
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[25:35, 25:35] = True
        img2.objmap[:] = label(mask2)

        result = (img1.objmask.__iand__(img2.objmask))

        assert result is img1.objmask
        assert np.sum(img1.objmask[:]) <= original_sum

    def test_iand_with_scalar_false(self, sample_image_with_mask):
        """Test &= with False scalar."""
        img = sample_image_with_mask

        result = (img.objmask.__iand__(False))

        assert result is img.objmask
        assert np.all(img.objmask[:] == 0)

    def test_ior_modifies_mask(self, sample_image_with_mask):
        """Test |= operator modifies mask in place."""
        img = sample_image_with_mask
        original_sum = np.sum(img.objmask[:])

        # Add new region
        addition = np.zeros((100, 100), dtype=bool)
        addition[50:60, 50:60] = True

        result = (img.objmask.__ior__(addition))

        assert result is img.objmask
        assert np.sum(img.objmask[:]) > original_sum  # Mask expanded

    def test_ior_with_objmask(self, sample_image_with_mask):
        """Test |= operator with another ObjectMask."""
        img1 = sample_image_with_mask
        original_sum = np.sum(img1.objmask[:])

        # Create second image with non-overlapping mask
        arr2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = Image(arr2)
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[10:15, 10:15] = True  # Non-overlapping
        img2.objmap[:] = label(mask2)

        result = (img1.objmask.__ior__(img2.objmask))

        assert result is img1.objmask
        assert np.sum(img1.objmask[:]) > original_sum

    def test_ior_with_scalar_true(self, sample_image_with_mask):
        """Test |= with True scalar."""
        img = sample_image_with_mask

        result = (img.objmask.__ior__(True))

        assert result is img.objmask
        assert np.all(img.objmask[:] == 1)

    def test_ixor_modifies_mask(self, sample_image_with_mask):
        """Test ^= operator modifies mask in place."""
        img = sample_image_with_mask
        original = img.objmask[:].copy()

        # Toggle a region
        toggle = np.zeros((100, 100), dtype=bool)
        toggle[25:30, 25:30] = True

        result = (img.objmask.__ixor__(toggle))

        assert result is img.objmask
        # Toggled region should have opposite values
        assert not np.array_equal(img.objmask[:], original)

    def test_ixor_with_objmask(self, sample_image_with_mask):
        """Test ^= operator with another ObjectMask."""
        img1 = sample_image_with_mask
        original = img1.objmask[:].copy()

        # Create second image
        arr2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = Image(arr2)
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[25:35, 25:35] = True
        img2.objmap[:] = label(mask2)

        result = (img1.objmask.__ixor__(img2.objmask))

        assert result is img1.objmask
        assert not np.array_equal(img1.objmask[:], original)

    def test_inplace_triggers_relabeling(self, sample_image_with_mask):
        """Test that in-place operators trigger relabeling."""
        img = sample_image_with_mask
        original_num_objects = len(np.unique(img.objmap[:]))

        # Remove one object entirely
        removal = np.ones((100, 100), dtype=bool)
        removal[60:80, 60:80] = False

        img.objmask.__iand__(removal)

        # Should have fewer objects after relabeling
        final_num_objects = len(np.unique(img.objmap[:]))
        assert final_num_objects < original_num_objects

    def test_iand_shape_validation(self, sample_image_with_mask):
        """Test that in-place AND validates shape."""
        img = sample_image_with_mask
        wrong_shape = np.ones((50, 50), dtype=bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            img.objmask.__iand__(wrong_shape)


class TestOperatorChaining:
    """Test chaining of boolean operators."""

    def test_chain_basic_operators(self, sample_image_with_mask):
        """Test chaining multiple basic operators."""
        img = sample_image_with_mask
        mask1 = np.ones((100, 100), dtype=bool)
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[30:70, 30:70] = True

        # Chain: (objmask & mask1) | mask2
        result = (img.objmask & mask1) | mask2

        assert isinstance(result, np.ndarray)
        assert result.dtype == int

    def test_chain_with_inversion(self, sample_image_with_mask):
        """Test chaining with NOT operator."""
        img = sample_image_with_mask

        # Create background mask: ~objmask
        background = ~img.objmask

        # Combine with new detections
        new_region = np.zeros((100, 100), dtype=bool)
        new_region[10:15, 10:15] = True

        result = background & new_region
        assert isinstance(result, np.ndarray)
        # New region should be in background
        assert np.sum(result[10:15, 10:15]) > 0

    def test_chain_complex_expression(self, sample_image_with_mask):
        """Test complex chaining of operators."""
        img = sample_image_with_mask
        roi = np.ones((100, 100), dtype=bool)
        roi[5:95, 5:95] = False  # Exclude edges

        # Complex expression: (objmask & roi) | additional_region
        additional = np.zeros((100, 100), dtype=bool)
        additional[10:12, 10:12] = True

        result = (img.objmask & roi) | additional

        assert isinstance(result, np.ndarray)


class TestRealWorldUseCases:
    """Test realistic microbiology use cases."""

    def test_remove_edge_artifacts(self, sample_image_with_mask):
        """Test removing colonies touching image edges."""
        img = sample_image_with_mask

        # Create edge exclusion mask
        edge_buffer = 5
        roi = np.ones((100, 100), dtype=bool)
        roi[:edge_buffer, :] = False
        roi[-edge_buffer:, :] = False
        roi[:, :edge_buffer] = False
        roi[:, -edge_buffer:] = False

        # Remove edge colonies
        cleaned = img.objmask & roi

        assert isinstance(cleaned, np.ndarray)
        # Edges should be clear
        assert np.sum(cleaned[:edge_buffer, :]) == 0
        assert np.sum(cleaned[-edge_buffer:, :]) == 0
        assert np.sum(cleaned[:, :edge_buffer]) == 0
        assert np.sum(cleaned[:, -edge_buffer:]) == 0

    def test_combine_detection_methods(self, sample_image_with_mask):
        """Test combining results from multiple detectors."""
        img = sample_image_with_mask

        # Simulate second detector result
        alt_mask = np.zeros((100, 100), dtype=bool)
        alt_mask[45:65, 45:65] = True

        # Union of detections
        combined = img.objmask | alt_mask

        assert isinstance(combined, np.ndarray)
        assert np.sum(combined) >= np.sum(img.objmask[:])

    def test_find_misdetections(self, sample_image_with_mask):
        """Test finding regions detected by one method but not another."""
        img = sample_image_with_mask

        # Simulate another detection
        alt_mask = np.zeros((100, 100), dtype=bool)
        alt_mask[25:40, 25:40] = True

        # Find unique regions
        unique_to_original = img.objmask & ~alt_mask

        assert isinstance(unique_to_original, np.ndarray)
        # Original should have more in the first object region
        assert np.sum(unique_to_original[20:40, 20:40]) > 0

    def test_background_analysis(self, sample_image_with_mask):
        """Test analyzing background regions."""
        img = sample_image_with_mask

        # Get background mask
        background = ~img.objmask

        # Should be complementary to object mask
        assert np.sum(img.objmask[:]) + np.sum(background) == 100 * 100

    def test_inplace_refining_with_filter(self, sample_image_with_mask):
        """Test in-place refinement of detections."""
        img = sample_image_with_mask
        original_sum = np.sum(img.objmask[:])

        # Apply size filter (keep large components)
        size_filter = np.ones((100, 100), dtype=bool)
        size_filter[70:100, :] = False  # Remove small region

        img.objmask.__iand__(size_filter)

        # Mask should be smaller or same
        assert np.sum(img.objmask[:]) <= original_sum


class TestOperatorReturnTypes:
    """Test that operators return correct types."""

    def test_basic_and_returns_ndarray(self, sample_image_with_mask):
        """Test that & returns np.ndarray."""
        img = sample_image_with_mask
        result = img.objmask & np.ones((100, 100), dtype=bool)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, type(img.objmask))

    def test_basic_or_returns_ndarray(self, sample_image_with_mask):
        """Test that | returns np.ndarray."""
        img = sample_image_with_mask
        result = img.objmask | np.zeros((100, 100), dtype=bool)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, type(img.objmask))

    def test_basic_xor_returns_ndarray(self, sample_image_with_mask):
        """Test that ^ returns np.ndarray."""
        img = sample_image_with_mask
        result = img.objmask ^ np.zeros((100, 100), dtype=bool)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, type(img.objmask))

    def test_invert_returns_ndarray(self, sample_image_with_mask):
        """Test that ~ returns np.ndarray."""
        img = sample_image_with_mask
        result = ~img.objmask
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, type(img.objmask))

    def test_inplace_and_returns_self(self, sample_image_with_mask):
        """Test that &= returns ObjectMask (self)."""
        img = sample_image_with_mask
        result = (img.objmask.__iand__(np.ones((100, 100), dtype=bool)))
        assert result is img.objmask

    def test_inplace_or_returns_self(self, sample_image_with_mask):
        """Test that |= returns ObjectMask (self)."""
        img = sample_image_with_mask
        result = (img.objmask.__ior__(np.zeros((100, 100), dtype=bool)))
        assert result is img.objmask

    def test_inplace_xor_returns_self(self, sample_image_with_mask):
        """Test that ^= returns ObjectMask (self)."""
        img = sample_image_with_mask
        result = (img.objmask.__ixor__(np.zeros((100, 100), dtype=bool)))
        assert result is img.objmask


class TestOperatorDtypeHandling:
    """Test dtype handling in operators."""

    def test_and_with_different_dtypes(self, sample_image_with_mask):
        """Test AND with different numpy dtypes."""
        img = sample_image_with_mask

        # Test with different dtypes
        for dtype in [np.uint8, np.float32, np.int32, bool]:
            other = np.ones((100, 100), dtype=dtype)
            result = img.objmask & other
            assert result.dtype == int

    def test_or_with_float_array(self, sample_image_with_mask):
        """Test OR with float array."""
        img = sample_image_with_mask
        other = np.random.rand(100, 100)
        result = img.objmask | other
        assert result.dtype == int

    def test_xor_with_uint8_array(self, sample_image_with_mask):
        """Test XOR with uint8 array."""
        img = sample_image_with_mask
        other = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = img.objmask ^ other
        assert result.dtype == int


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
