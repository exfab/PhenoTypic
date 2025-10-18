"""Tests for numpy array interface (__array__) on all accessor classes.

This test suite verifies that numpy functions (e.g., np.sum, np.mean, np.max)
can be applied directly on accessor objects through the __array__ dunder method.
"""
import numpy as np
import pytest
from phenotypic import Image


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    # Create a 100x100 RGB image with some test data
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image(arr)


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    return Image(arr)


@pytest.fixture
def sample_image_with_objects(sample_grayscale_image):
    """Create a sample image with detected objects."""
    img = sample_grayscale_image
    # Create a simple binary mask with objects
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 20:40] = True
    mask[60:80, 60:80] = True
    
    # Apply detector using the mask
    from skimage.measure import label
    objmap = label(mask)
    img.objmap[:] = objmap
    
    return img


class TestImageArrayAccessor:
    """Tests for ImageArray accessor (RGB/multichannel images)."""
    
    def test_numpy_sum(self, sample_rgb_image):
        """Test np.sum on array accessor."""
        result = np.sum(sample_rgb_image.array)
        expected = np.sum(sample_rgb_image.array[:])
        assert result == expected
    
    def test_numpy_mean(self, sample_rgb_image):
        """Test np.mean on array accessor."""
        result = np.mean(sample_rgb_image.array)
        expected = np.mean(sample_rgb_image.array[:])
        assert np.isclose(result, expected)
    
    def test_numpy_max(self, sample_rgb_image):
        """Test np.max on array accessor."""
        result = np.max(sample_rgb_image.array)
        expected = np.max(sample_rgb_image.array[:])
        assert result == expected
    
    def test_numpy_min(self, sample_rgb_image):
        """Test np.min on array accessor."""
        result = np.min(sample_rgb_image.array)
        expected = np.min(sample_rgb_image.array[:])
        assert result == expected
    
    def test_numpy_std(self, sample_rgb_image):
        """Test np.std on array accessor."""
        result = np.std(sample_rgb_image.array)
        expected = np.std(sample_rgb_image.array[:])
        assert np.isclose(result, expected)
    
    def test_dtype_parameter(self, sample_rgb_image):
        """Test that dtype parameter works correctly."""
        result = np.array(sample_rgb_image.array, dtype=np.float32)
        assert result.dtype == np.float32


class TestImageMatrixAccessor:
    """Tests for ImageMatrix accessor (grayscale representation)."""
    
    def test_numpy_sum(self, sample_rgb_image):
        """Test np.sum on matrix accessor."""
        result = np.sum(sample_rgb_image.matrix)
        expected = np.sum(sample_rgb_image.matrix[:])
        assert np.isclose(result, expected)
    
    def test_numpy_mean(self, sample_rgb_image):
        """Test np.mean on matrix accessor."""
        result = np.mean(sample_rgb_image.matrix)
        expected = np.mean(sample_rgb_image.matrix[:])
        assert np.isclose(result, expected)
    
    def test_numpy_operations(self, sample_rgb_image):
        """Test various numpy operations."""
        matrix = sample_rgb_image.matrix
        
        assert np.min(matrix) >= 0
        assert np.max(matrix) <= 1
        assert matrix.shape == sample_rgb_image.matrix.shape


class TestEnhancedMatrixAccessor:
    """Tests for ImageEnhancedMatrix accessor."""
    
    def test_numpy_sum(self, sample_rgb_image):
        """Test np.sum on enhanced matrix accessor."""
        result = np.sum(sample_rgb_image.enh_matrix)
        expected = np.sum(sample_rgb_image.enh_matrix[:])
        assert np.isclose(result, expected)
    
    def test_numpy_mean(self, sample_rgb_image):
        """Test np.mean on enhanced matrix accessor."""
        result = np.mean(sample_rgb_image.enh_matrix)
        expected = np.mean(sample_rgb_image.enh_matrix[:])
        assert np.isclose(result, expected)


class TestObjectMapAccessor:
    """Tests for ObjectMap accessor (already had __array__)."""
    
    def test_numpy_sum(self, sample_image_with_objects):
        """Test np.sum on objmap accessor."""
        result = np.sum(sample_image_with_objects.objmap)
        expected = np.sum(sample_image_with_objects.objmap[:])
        assert result == expected
    
    def test_numpy_max(self, sample_image_with_objects):
        """Test np.max on objmap accessor."""
        result = np.max(sample_image_with_objects.objmap)
        expected = np.max(sample_image_with_objects.objmap[:])
        assert result == expected
    
    def test_numpy_unique(self, sample_image_with_objects):
        """Test np.unique on objmap accessor."""
        result = np.unique(sample_image_with_objects.objmap)
        expected = np.unique(sample_image_with_objects.objmap[:])
        assert np.array_equal(result, expected)


class TestObjectMaskAccessor:
    """Tests for ObjectMask accessor (already had __array__)."""
    
    def test_numpy_sum(self, sample_image_with_objects):
        """Test np.sum on objmask accessor (counts pixels)."""
        result = np.sum(sample_image_with_objects.objmask)
        expected = np.sum(sample_image_with_objects.objmask[:])
        assert result == expected
    
    def test_numpy_count_nonzero(self, sample_image_with_objects):
        """Test np.count_nonzero on objmask accessor."""
        result = np.count_nonzero(sample_image_with_objects.objmask)
        expected = np.count_nonzero(sample_image_with_objects.objmask[:])
        assert result == expected


class TestHsvAccessor:
    """Tests for HSV accessor."""
    
    def test_numpy_mean(self, sample_rgb_image):
        """Test np.mean on HSV accessor."""
        result = np.mean(sample_rgb_image.hsv)
        expected = np.mean(sample_rgb_image.hsv[:])
        assert np.isclose(result, expected)
    
    def test_numpy_std(self, sample_rgb_image):
        """Test np.std on HSV accessor."""
        result = np.std(sample_rgb_image.hsv)
        expected = np.std(sample_rgb_image.hsv[:])
        assert np.isclose(result, expected)
    
    def test_hsv_shape(self, sample_rgb_image):
        """Test that HSV has correct shape."""
        hsv_arr = np.array(sample_rgb_image.hsv)
        assert hsv_arr.shape[-1] == 3  # Should have 3 channels


class TestColorSpaceAccessors:
    """Tests for color space accessors (XYZ, CIELAB, etc.)."""
    
    def test_xyz_numpy_operations(self, sample_rgb_image):
        """Test numpy operations on XYZ accessor."""
        result = np.mean(sample_rgb_image.CieXYZ)
        expected = np.mean(sample_rgb_image.CieXYZ[:])
        assert np.isclose(result, expected)
    
    def test_cielab_numpy_operations(self, sample_rgb_image):
        """Test numpy operations on CIELAB accessor."""
        result = np.std(sample_rgb_image.CieLab)
        expected = np.std(sample_rgb_image.CieLab[:])
        assert np.isclose(result, expected)
    
    def test_chromaticity_numpy_operations(self, sample_rgb_image):
        """Test numpy operations on chromaticity accessor."""
        result = np.max(sample_rgb_image.Cie_xy)
        expected = np.max(sample_rgb_image.Cie_xy[:])
        assert np.isclose(result, expected)
    
    def test_xyz_d65_numpy_operations(self, sample_rgb_image):
        """Test numpy operations on XYZ_D65 accessor."""
        result = np.min(sample_rgb_image.CieXYZ_D65)
        expected = np.min(sample_rgb_image.CieXYZ_D65[:])
        assert np.isclose(result, expected)


class TestArrayInterfaceConsistency:
    """Tests to ensure __array__ returns consistent results."""
    
    def test_multiple_calls_same_result(self, sample_rgb_image):
        """Test that multiple calls to __array__ give the same result."""
        arr1 = np.array(sample_rgb_image.matrix)
        arr2 = np.array(sample_rgb_image.matrix)
        assert np.array_equal(arr1, arr2)
    
    def test_direct_access_matches_array_interface(self, sample_rgb_image):
        """Test that __array__ matches direct array access."""
        via_interface = np.array(sample_rgb_image.array)
        via_getitem = sample_rgb_image.array[:]
        assert np.array_equal(via_interface, via_getitem)
    
    def test_copy_parameter(self, sample_rgb_image):
        """Test that copy parameter works correctly."""
        arr1 = np.array(sample_rgb_image.matrix, copy=True)
        arr2 = np.array(sample_rgb_image.matrix, copy=True)
        
        # Modify arr1 and ensure arr2 is unchanged
        arr1[0, 0] = 999
        assert not np.array_equal(arr1, arr2)


class TestNumpyFunctionCompatibility:
    """Test compatibility with various numpy functions."""
    
    def test_reshape_operations(self, sample_rgb_image):
        """Test that reshape works after converting to array."""
        flat = np.reshape(sample_rgb_image.matrix, -1)
        assert flat.ndim == 1
    
    def test_where_operations(self, sample_rgb_image):
        """Test np.where with accessor."""
        matrix = sample_rgb_image.matrix
        threshold = 0.5
        # Convert to array explicitly for comparison operations
        matrix_arr = np.array(matrix)
        result = np.where(matrix_arr > threshold, 1, 0)
        assert result.shape == matrix.shape
    
    def test_clip_operations(self, sample_rgb_image):
        """Test np.clip with accessor."""
        result = np.clip(sample_rgb_image.matrix, 0.2, 0.8)
        assert np.all(result >= 0.2) and np.all(result <= 0.8)
    
    def test_concatenate_operations(self, sample_rgb_image):
        """Test np.concatenate with multiple accessors."""
        matrix = sample_rgb_image.matrix
        result = np.concatenate([matrix, matrix], axis=0)
        assert result.shape[0] == matrix.shape[0] * 2
    
    def test_percentile_operations(self, sample_rgb_image):
        """Test np.percentile with accessor."""
        result = np.percentile(sample_rgb_image.matrix, 50)
        expected = np.percentile(sample_rgb_image.matrix[:], 50)
        assert np.isclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
