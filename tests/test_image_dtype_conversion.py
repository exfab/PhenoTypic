"""Tests for Image class dtype conversion, bit depth inference, and format detection.

This module provides comprehensive testing of the dtype conversion and bit depth
inference logic in ImageDataManager, which is critical for proper image initialization
and metadata tracking in microbe colony phenotyping workflows.
"""

import warnings
import numpy as np
import pytest
from skimage.color import rgba2rgb

from phenotypic import Image
from phenotypic.core._image_parts._image_data_manager import ImageDataManager
from phenotypic.tools.constants_ import IMAGE_MODE
from .resources.TestHelper import timeit


# ============================================================================================
# Fixtures
# ============================================================================================


@pytest.fixture
def uint8_rgb_array():
    """Create a sample uint8 RGB array (0-255 range)."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def uint16_rgb_array():
    """Create a sample uint16 RGB array (0-65535 range)."""
    return np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)


@pytest.fixture
def float32_rgb_array():
    """Create a sample float32 RGB array (normalized to [0, 1])."""
    return np.random.rand(100, 100, 3).astype(np.float32)


@pytest.fixture
def float64_rgb_array():
    """Create a sample float64 RGB array (normalized to [0, 1])."""
    return np.random.rand(100, 100, 3).astype(np.float64)


@pytest.fixture
def uint8_gray_array():
    """Create a sample uint8 grayscale array."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def uint16_gray_array():
    """Create a sample uint16 grayscale array."""
    return np.random.randint(0, 65535, (100, 100), dtype=np.uint16)


@pytest.fixture
def float32_gray_array():
    """Create a sample float32 grayscale array (normalized to [0, 1])."""
    return np.random.rand(100, 100).astype(np.float32)


@pytest.fixture
def float64_gray_array():
    """Create a sample float64 grayscale array (normalized to [0, 1])."""
    return np.random.rand(100, 100).astype(np.float64)


@pytest.fixture
def rgba_array():
    """Create a sample RGBA array (4 channels)."""
    return np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)


@pytest.fixture
def single_channel_3d_array():
    """Create a 3D array with single channel (H, W, 1)."""
    return np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)


# ============================================================================================
# Test Bit Depth Inference (ImageDataManager._infer_bit_depth)
# ============================================================================================


class TestBitDepthInference:
    """Tests for bit depth inference from array dtype."""

    @timeit
    def test_infer_uint8_returns_8(self, uint8_rgb_array):
        """Test that uint8 array returns bit_depth=8."""
        bit_depth = ImageDataManager._infer_bit_depth(uint8_rgb_array)
        assert bit_depth == 8

    @timeit
    def test_infer_uint16_returns_16(self, uint16_rgb_array):
        """Test that uint16 array returns bit_depth=16."""
        bit_depth = ImageDataManager._infer_bit_depth(uint16_rgb_array)
        assert bit_depth == 16

    @timeit
    def test_infer_float32_returns_16(self, float32_rgb_array):
        """Test that float32 array returns bit_depth=16."""
        bit_depth = ImageDataManager._infer_bit_depth(float32_rgb_array)
        assert bit_depth == 16

    @timeit
    def test_infer_float64_returns_16(self, float64_rgb_array):
        """Test that float64 array returns bit_depth=16."""
        bit_depth = ImageDataManager._infer_bit_depth(float64_rgb_array)
        assert bit_depth == 16

    @timeit
    def test_infer_unknown_dtype_warns_and_returns_16(self):
        """Test that unknown dtype (int32) warns and returns bit_depth=16."""
        int32_array = np.random.randint(0, 255, (100, 100), dtype=np.int32)

        with pytest.warns(UserWarning, match="unknown dtype"):
            bit_depth = ImageDataManager._infer_bit_depth(int32_array)

        assert bit_depth == 16

    @timeit
    def test_infer_int64_warns_and_returns_16(self):
        """Test that int64 dtype warns and returns bit_depth=16."""
        int64_array = np.array([[1, 2], [3, 4]], dtype=np.int64)

        with pytest.warns(UserWarning, match="unknown dtype"):
            bit_depth = ImageDataManager._infer_bit_depth(int64_array)

        assert bit_depth == 16


# ============================================================================================
# Test Float Array Conversion (ImageDataManager._convert_float_array_to_int)
# ============================================================================================


class TestFloatArrayConversion:
    """Tests for conversion of float arrays to integer arrays."""

    @timeit
    def test_convert_float_to_uint8(self):
        """Test conversion of float [0, 1] array to uint8 [0, 255]."""
        float_array = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)

        result = ImageDataManager._convert_float_array_to_int(float_array, bit_depth=8)

        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 0  # 0.0 * 255 = 0
        assert result[0, 0, 2] == 255  # 1.0 * 255 = 255
        # 0.5 should map to ~127-128 (127.5 rounded)

    @timeit
    def test_convert_float_to_uint16(self):
        """Test conversion of float [0, 1] array to uint16 [0, 65535]."""
        float_array = np.array([[[0.0, 1.0]]], dtype=np.float32)

        result = ImageDataManager._convert_float_array_to_int(float_array, bit_depth=16)

        assert result.dtype == np.uint16
        assert result[0, 0, 0] == 0  # 0.0 * 65535 = 0
        assert result[0, 0, 1] == 65535  # 1.0 * 65535 = 65535

    @timeit
    def test_convert_preserves_array_shape(self):
        """Test that conversion preserves array shape."""
        float_array = np.random.rand(50, 75, 3).astype(np.float32)

        result = ImageDataManager._convert_float_array_to_int(float_array, bit_depth=8)

        assert result.shape == float_array.shape

    @timeit
    def test_convert_float_below_zero_raises_valueerror(self):
        """Test that float array with values < 0 raises ValueError."""
        float_array = np.array([[[-0.1, 0.5, 1.0]]], dtype=np.float32)

        with pytest.raises(ValueError, match="outside.*range"):
            ImageDataManager._convert_float_array_to_int(float_array, bit_depth=8)

    @timeit
    def test_convert_float_above_one_raises_valueerror(self):
        """Test that float array with values > 1 raises ValueError."""
        float_array = np.array([[[0.0, 0.5, 1.1]]], dtype=np.float32)

        with pytest.raises(ValueError, match="outside.*range"):
            ImageDataManager._convert_float_array_to_int(float_array, bit_depth=8)

    @timeit
    def test_convert_invalid_bit_depth_32_raises_valueerror(self):
        """Test that bit_depth=32 raises ValueError."""
        float_array = np.random.rand(10, 10, 3).astype(np.float32)

        with pytest.raises(ValueError, match="bit_depth must be 8 or 16"):
            ImageDataManager._convert_float_array_to_int(float_array, bit_depth=32)

    @timeit
    def test_convert_invalid_bit_depth_12_raises_valueerror(self):
        """Test that bit_depth=12 raises ValueError."""
        float_array = np.random.rand(10, 10, 3).astype(np.float32)

        with pytest.raises(ValueError, match="bit_depth must be 8 or 16"):
            ImageDataManager._convert_float_array_to_int(float_array, bit_depth=12)

    @timeit
    def test_convert_edge_case_very_small_float(self):
        """Test conversion of very small float values (near 0)."""
        float_array = np.array([[[1e-6, 0.0]]], dtype=np.float32)

        result = ImageDataManager._convert_float_array_to_int(float_array, bit_depth=8)

        # Very small value should convert to 0 or 1
        assert result[0, 0, 0] in (0, 1)


# ============================================================================================
# Test Image Format Detection (ImageDataManager._guess_image_format)
# ============================================================================================


class TestImageFormatDetection:
    """Tests for image format detection from array shape."""

    @timeit
    def test_detect_2d_array_as_grayscale(self, uint8_gray_array):
        """Test that 2D arrays are detected as GRAYSCALE."""
        format_enum = ImageDataManager._guess_image_format(uint8_gray_array)
        assert format_enum == IMAGE_MODE.GRAYSCALE

    @timeit
    def test_detect_3d_single_channel_as_grayscale(self, single_channel_3d_array):
        """Test that (H, W, 1) arrays are detected as GRAYSCALE_SINGLE_CHANNEL."""
        format_enum = ImageDataManager._guess_image_format(single_channel_3d_array)
        assert format_enum == IMAGE_MODE.GRAYSCALE_SINGLE_CHANNEL

    @timeit
    def test_detect_3d_three_channel_as_rgb(self, uint8_rgb_array):
        """Test that (H, W, 3) arrays are detected as RGB."""
        format_enum = ImageDataManager._guess_image_format(uint8_rgb_array)
        assert format_enum == IMAGE_MODE.RGB

    @timeit
    def test_detect_3d_four_channel_as_rgba(self, rgba_array):
        """Test that (H, W, 4) arrays are detected as RGBA."""
        format_enum = ImageDataManager._guess_image_format(rgba_array)
        assert format_enum == IMAGE_MODE.RGBA

    @timeit
    def test_detect_unsupported_5_channels_raises_valueerror(self):
        """Test that 5-channel arrays raise ValueError."""
        array_5ch = np.random.randint(0, 255, (100, 100, 5), dtype=np.uint8)

        with pytest.raises(ValueError, match="channels.*unknown format"):
            ImageDataManager._guess_image_format(array_5ch)

    @timeit
    def test_detect_unsupported_2_channels_raises_valueerror(self):
        """Test that 2-channel arrays raise ValueError."""
        array_2ch = np.random.randint(0, 255, (100, 100, 2), dtype=np.uint8)

        with pytest.raises(ValueError, match="channels.*unknown format"):
            ImageDataManager._guess_image_format(array_2ch)

    @timeit
    def test_detect_1d_array_raises_valueerror(self):
        """Test that 1D arrays raise ValueError."""
        array_1d = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="unsupported number of dimensions"):
            ImageDataManager._guess_image_format(array_1d)

    @timeit
    def test_detect_4d_array_raises_valueerror(self):
        """Test that 4D arrays raise ValueError."""
        array_4d = np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="unsupported number of dimensions"):
            ImageDataManager._guess_image_format(array_4d)

    @timeit
    def test_detect_non_numpy_raises_typeerror(self):
        """Test that non-numpy input raises TypeError."""
        python_list = [[1, 2], [3, 4]]

        with pytest.raises(TypeError, match="must be a numpy array"):
            ImageDataManager._guess_image_format(python_list)

    @timeit
    def test_detect_tuple_raises_typeerror(self):
        """Test that tuple input raises TypeError."""
        python_tuple = ((1, 2), (3, 4))

        with pytest.raises(TypeError, match="must be a numpy array"):
            ImageDataManager._guess_image_format(python_tuple)


# ============================================================================================
# Test Image Initialization with Various Dtypes
# ============================================================================================


class TestImageInitializationDtypes:
    """Integration tests for Image initialization with various dtypes."""

    @timeit
    def test_image_from_uint8_rgb(self, uint8_rgb_array):
        """Test Image initialization with uint8 RGB array."""
        img = Image(arr=uint8_rgb_array)

        assert img.bit_depth == 8
        assert not img.rgb.isempty()
        assert np.array_equal(img.rgb[:], uint8_rgb_array)
        assert img.isempty() is False

    @timeit
    def test_image_from_uint16_rgb(self, uint16_rgb_array):
        """Test Image initialization with uint16 RGB array."""
        img = Image(arr=uint16_rgb_array)

        assert img.bit_depth == 16
        assert not img.rgb.isempty()
        assert np.array_equal(img.rgb[:], uint16_rgb_array)

    @timeit
    def test_image_from_float32_rgb_converts(self, float32_rgb_array):
        """Test that float32 RGB array is converted to uint16."""
        img = Image(arr=float32_rgb_array)

        assert img.bit_depth == 16
        assert img.rgb[:].dtype == np.uint16  # Converted float → uint16 (bit_depth=16)
        # Verify scaling occurred (not all same value)
        assert len(np.unique(img.rgb[:])) > 1

    @timeit
    def test_image_from_float64_rgb_converts(self, float64_rgb_array):
        """Test that float64 RGB array is converted to uint16."""
        img = Image(arr=float64_rgb_array)

        assert img.bit_depth == 16
        assert img.rgb[:].dtype == np.uint16  # Converted float → uint16 (bit_depth=16)

    @timeit
    def test_image_from_uint8_grayscale(self, uint8_gray_array):
        """Test Image initialization with uint8 grayscale array."""
        img = Image(arr=uint8_gray_array)

        assert img.bit_depth == 8
        assert img.rgb.isempty()  # No RGB for grayscale input
        assert np.array_equal(img.gray[:], uint8_gray_array)

    @timeit
    def test_image_from_uint16_grayscale(self, uint16_gray_array):
        """Test Image initialization with uint16 grayscale array."""
        img = Image(arr=uint16_gray_array)

        assert img.bit_depth == 16
        assert img.rgb.isempty()
        assert np.array_equal(img.gray[:], uint16_gray_array)

    @timeit
    def test_image_from_float32_grayscale_no_conversion(self, float32_gray_array):
        """Test that float32 grayscale array is NOT converted (2D arrays)."""
        img = Image(arr=float32_gray_array)

        assert img.bit_depth == 16
        # Grayscale float arrays are NOT converted (only RGB floats are)
        assert np.array_equal(img.gray[:], float32_gray_array)

    @timeit
    def test_image_from_float64_grayscale_no_conversion(self, float64_gray_array):
        """Test that float64 grayscale array is NOT converted."""
        img = Image(arr=float64_gray_array)

        assert img.bit_depth == 16
        assert np.array_equal(img.gray[:], float64_gray_array)

    @timeit
    def test_image_from_rgba_converts_to_rgb(self, rgba_array):
        """Test that RGBA array is converted to RGB."""
        img = Image(arr=rgba_array)

        # Image should have RGB data with alpha dropped
        assert not img.rgb.isempty()
        assert img.rgb[:].shape == (100, 100, 3)
        # Verify it's the same as skimage's RGBA→RGB conversion
        expected_rgb = rgba2rgb(rgba_array)
        assert np.array_equal(img.rgb[:], expected_rgb)

    @timeit
    def test_image_from_single_channel_3d(self, single_channel_3d_array):
        """Test Image initialization with (H, W, 1) array (single channel)."""
        img = Image(arr=single_channel_3d_array)

        assert img.rgb.isempty()  # Treated as grayscale
        assert img.gray.shape == (100, 100)  # Squeezed to 2D

    @timeit
    def test_explicit_bit_depth_not_overridden(self, uint8_rgb_array):
        """Test that explicit bit_depth parameter is not overridden."""
        img = Image(arr=uint8_rgb_array, bit_depth=16)

        # Explicit bit_depth=16 should be preserved despite uint8 input
        assert img.bit_depth == 16

    @timeit
    def test_image_name_and_bit_depth_together(self, uint16_rgb_array):
        """Test Image initialization with both name and bit_depth."""
        img = Image(arr=uint16_rgb_array, name="test_colony", bit_depth=16)

        assert img.name == "test_colony"
        assert img.bit_depth == 16


# ============================================================================================
# Test Array Input Handling Edge Cases
# ============================================================================================


class TestArrayInputHandling:
    """Tests for _handle_array_input method and related logic."""

    @timeit
    def test_float_rgb_array_converted_through_handler(self, float32_rgb_array):
        """Test that float RGB array is converted through _handle_array_input."""
        img = Image()
        img.set_image(float32_rgb_array)

        assert img.bit_depth == 16
        assert img.rgb[:].dtype == np.uint16  # Converted to uint16 (bit_depth=16)
        # Verify conversion happened (not original float values)
        assert img.rgb[:].max() <= 65535

    @timeit
    def test_uint8_sets_bit_depth_automatically(self, uint8_gray_array):
        """Test that uint8 input automatically sets bit_depth=8."""
        img = Image()
        assert img.bit_depth is None  # Initially unset

        img.set_image(uint8_gray_array)

        assert img.bit_depth == 8

    @timeit
    def test_uint16_sets_bit_depth_automatically(self, uint16_gray_array):
        """Test that uint16 input automatically sets bit_depth=16."""
        img = Image()
        assert img.bit_depth is None

        img.set_image(uint16_gray_array)

        assert img.bit_depth == 16

    @timeit
    def test_explicit_bit_depth_prevents_inference(self, uint8_gray_array):
        """Test that explicit bit_depth prevents automatic inference."""
        img = Image(bit_depth=16)
        assert img.bit_depth == 16  # Set before

        img.set_image(uint8_gray_array)

        assert img.bit_depth == 16  # Should not change to 8


# ============================================================================================
# Test Error Handling
# ============================================================================================


class TestErrorHandling:
    """Tests for error handling in dtype conversion and input validation."""

    @timeit
    def test_set_image_with_list_raises_valueerror(self):
        """Test that setting image with a list raises ValueError."""
        img = Image()

        with pytest.raises(ValueError, match="must be a NumPy array"):
            img.set_image([1, 2, 3])

    @timeit
    def test_set_image_with_string_raises_valueerror(self):
        """Test that setting image with a string raises ValueError."""
        img = Image()

        with pytest.raises(ValueError, match="must be a NumPy array"):
            img.set_image("not_an_image")

    @timeit
    def test_set_image_with_dict_raises_valueerror(self):
        """Test that setting image with a dict raises ValueError."""
        img = Image()

        with pytest.raises(ValueError, match="must be a NumPy array"):
            img.set_image({"data": np.zeros((10, 10))})

    @timeit
    def test_float_rgb_out_of_range_negative(self):
        """Test that float RGB array with negative values raises ValueError."""
        float_rgb = np.array([[[-0.5, 0.5, 1.0]]], dtype=np.float32)

        with pytest.raises(ValueError, match="outside.*range"):
            Image(arr=float_rgb)

    @timeit
    def test_float_rgb_out_of_range_positive(self):
        """Test that float RGB array with values > 1 raises ValueError."""
        float_rgb = np.array([[[0.0, 0.5, 1.5]]], dtype=np.float32)

        with pytest.raises(ValueError, match="outside.*range"):
            Image(arr=float_rgb)

    @timeit
    def test_4d_array_raises_valueerror(self):
        """Test that 4D array raises ValueError during format detection."""
        array_4d = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            Image(arr=array_4d)

    @timeit
    def test_explicit_bit_depth_32_allows_but_unused(self):
        """Test that Image stores explicit bit_depth even if not 8/16.

        Note: Image class doesn't strictly validate bit_depth at init time,
        it accepts any value. Validation may occur elsewhere if needed.
        """
        uint8_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Image allows explicit bit_depth to be set
        img = Image(arr=uint8_array, bit_depth=32)
        assert img.bit_depth == 32

    @timeit
    def test_invalid_gamma_encoding_raises_valueerror(self):
        """Test that invalid gamma_encoding raises ValueError."""
        uint8_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            Image(arr=uint8_array, gamma_encoding="InvalidGamma")

    @timeit
    def test_invalid_illuminant_raises_valueerror(self):
        """Test that invalid illuminant raises ValueError."""
        uint8_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            Image(arr=uint8_array, illuminant="InvalidLight")


# ============================================================================================
# Test Gray Array Derivation from RGB
# ============================================================================================


class TestGrayArrayDerivation:
    """Tests for grayscale array derivation from RGB inputs."""

    @timeit
    def test_gray_from_uint8_rgb(self, uint8_rgb_array):
        """Test that grayscale is properly derived from uint8 RGB."""
        img = Image(arr=uint8_rgb_array)

        assert not img.gray.isempty()
        # Grayscale should be different from original (averaged)
        assert img.gray.shape == (100, 100)

    @timeit
    def test_gray_from_uint16_rgb(self, uint16_rgb_array):
        """Test that grayscale is properly derived from uint16 RGB."""
        img = Image(arr=uint16_rgb_array)

        assert not img.gray.isempty()
        assert img.gray.shape == (100, 100)

    @timeit
    def test_enh_gray_initialized_equal_to_gray(self, uint8_rgb_array):
        """Test that enhanced grayscale is initially equal to grayscale."""
        img = Image(arr=uint8_rgb_array)

        assert np.array_equal(img.enh_gray[:], img.gray[:])
