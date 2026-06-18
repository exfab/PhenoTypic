"""Tests for Image class dtype conversion, bit depth inference, and format detection.

This module provides comprehensive testing of the dtype conversion and bit depth
inference logic in ImageDataManager, which is critical for proper image initialization
and metadata tracking in microbe colony phenotyping workflows.
"""

import numpy as np
import pytest
from skimage.color import rgba2rgb

from phenotypic import Image
from phenotypic._core._image_parts._image_data_manager import ImageDataManager
from phenotypic.sdk_.constants_ import IMAGE_MODE
from ..resources.TestHelper import timeit


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
    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_infer_unknown_dtype_warns_and_returns_16(self, dtype):
        """Test that unknown dtypes warn and return bit_depth=16."""
        array = np.array([[1, 2], [3, 4]], dtype=dtype)

        with pytest.warns(UserWarning, match="unknown dtype"):
            bit_depth = ImageDataManager._infer_bit_depth(array)

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
    @pytest.mark.parametrize("bit_depth", [12, 32])
    def test_convert_invalid_bit_depth_raises_valueerror(self, bit_depth):
        """Test that unsupported bit_depth values raise ValueError."""
        float_array = np.random.rand(10, 10, 3).astype(np.float32)

        with pytest.raises(ValueError, match="bit_depth must be 8 or 16"):
            ImageDataManager._convert_float_array_to_int(float_array, bit_depth=bit_depth)

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
    @pytest.mark.parametrize("channels", [2, 5])
    def test_detect_unsupported_channels_raises_valueerror(self, channels):
        """Test that unsupported channel counts raise ValueError."""
        array = np.random.randint(0, 255, (100, 100, channels), dtype=np.uint8)

        with pytest.raises(ValueError, match="channels.*unknown format"):
            ImageDataManager._guess_image_format(array)

    @timeit
    @pytest.mark.parametrize("array", [
        np.array([1, 2, 3]),
        np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8),
    ], ids=["1d", "4d"])
    def test_detect_unsupported_dimensions_raises_valueerror(self, array):
        """Test that arrays with unsupported dimensionality raise ValueError."""
        with pytest.raises(ValueError, match="unsupported number of dimensions"):
            ImageDataManager._guess_image_format(array)

    @timeit
    @pytest.mark.parametrize("non_array", [
        [[1, 2], [3, 4]],
        ((1, 2), (3, 4)),
    ], ids=["list", "tuple"])
    def test_detect_non_numpy_raises_typeerror(self, non_array):
        """Test that non-numpy inputs raise TypeError."""
        with pytest.raises(TypeError, match="must be a numpy array"):
            ImageDataManager._guess_image_format(non_array)


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
    def test_image_from_float64_grayscale_converts_to_float32(self, float64_gray_array):
        """float64 grayscale is downcast to the float32 luminance-layer contract.

        ``gray``/``detect_mat`` are stored as float32 (enforced by
        ``ImageData.__setattr__``); a float64 input is preserved to float32
        precision while halving its footprint. ``bit_depth`` inference runs on
        the *input* dtype and is unaffected.
        """
        img = Image(arr=float64_gray_array)

        assert img.bit_depth == 16
        assert img.gray[:].dtype == np.float32
        assert np.allclose(img.gray[:], float64_gray_array, atol=1e-6)

    @timeit
    def test_image_from_rgba_converts_to_rgb(self, rgba_array):
        """Test that RGBA array is converted to RGB."""
        img = Image(arr=rgba_array)

        # Image should have RGB data with beta dropped
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
    @pytest.mark.parametrize("invalid_input", [
        [1, 2, 3],
        "not_an_image",
        {"data": "value"},
    ], ids=["list", "string", "dict"])
    def test_set_image_with_non_array_raises_valueerror(self, invalid_input):
        """Test that setting image with non-array types raises ValueError."""
        img = Image()

        with pytest.raises(ValueError, match="must be a NumPy array"):
            img.set_image(invalid_input)

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
        """Test that invalid gamma raises ValueError."""
        uint8_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            Image(arr=uint8_array, gamma="InvalidGamma")

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
    def test_detect_mat_initialized_equal_to_gray(self, uint8_rgb_array):
        """Test that enhanced grayscale is initially equal to grayscale."""
        img = Image(arr=uint8_rgb_array)

        assert np.array_equal(img.detect_mat[:], img.gray[:])


class TestFloat32LuminanceLayerContract:
    """``gray`` / ``detect_mat`` are stored as float32 (enforced contract).

    ``ImageData.__setattr__`` coerces any floating assignment to these luminance
    layers to float32 — skimage's ``rgb2gray`` returns float64, so without this
    the declared float32 contract was never actually enforced. This halves the
    in-memory and on-disk footprint of the two largest layers at negligible
    precision cost (float32 ≈ 7 significant digits, ~500× finer than the uint16
    quantization step). Integer inputs are left untouched.
    """

    @timeit
    def test_uint8_rgb_produces_float32_layers(self, uint8_rgb_array):
        """RGB input derives float32 gray + detect_mat (rgb2gray float64 downcast)."""
        img = Image(arr=uint8_rgb_array)
        assert img.gray[:].dtype == np.float32
        assert img.detect_mat[:].dtype == np.float32

    @timeit
    def test_uint16_rgb_produces_float32_layers(self, uint16_rgb_array):
        """16-bit RGB input also derives float32 luminance layers."""
        img = Image(arr=uint16_rgb_array)
        assert img.gray[:].dtype == np.float32
        assert img.detect_mat[:].dtype == np.float32

    @timeit
    def test_float64_grayscale_input_stored_as_float32(self, float64_gray_array):
        """A float64 grayscale input is downcast to float32 on assignment."""
        img = Image(arr=float64_gray_array)
        assert img.gray[:].dtype == np.float32
        assert np.allclose(img.gray[:], float64_gray_array, atol=1e-6)

    @timeit
    def test_accessor_write_coerces_float64_to_float32(self, uint8_rgb_array):
        """Writing a float64 array through the detect_mat accessor stays float32."""
        img = Image(arr=uint8_rgb_array)
        img.detect_mat[:] = img.detect_mat[:].astype(np.float64)
        assert img.detect_mat[:].dtype == np.float32

    @timeit
    def test_float32_layers_survive_hdf5_roundtrip(self, uint8_rgb_array, tmp_path):
        """Saving then loading an HDF5 image preserves the float32 layer contract."""
        img = Image(arr=uint8_rgb_array)
        path = tmp_path / "f32_layers.h5"
        img.save2hdf5(path)
        loaded = Image.load_hdf5(path)
        assert loaded.gray[:].dtype == np.float32
        assert loaded.detect_mat[:].dtype == np.float32
        assert np.allclose(loaded.gray[:], img.gray[:], atol=1e-6)
