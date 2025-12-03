import numpy as np
import pytest
from phenotypic import Image, GridImage
from phenotypic.grid import AutoGridFinder
from phenotypic.detect import OtsuDetector
from phenotypic.tools.exceptions_ import IllegalAssignmentError

from .resources.TestHelper import timeit

from .test_fixtures import plate_grid_images_with_detection, sample_image_array


@timeit
def test_blank_gridimage_initialization():
    # Test default initialization
    grid_image = GridImage()
    assert grid_image is not None
    assert isinstance(grid_image.grid_finder, AutoGridFinder)


@timeit
def test_gridimage_initialization(sample_image_array):
    # Test custom initialization with _root_image and grid setter
    input_image = sample_image_array
    grid_image = GridImage(arr=input_image)
    assert grid_image.isempty() is False

    grid_setter = AutoGridFinder(nrows=10, ncols=10)
    grid_image = GridImage(arr=input_image, grid_finder=grid_setter)
    assert grid_image.grid_finder == grid_setter


@timeit
def test_grid_accessor_default_property():
    grid_image = GridImage()
    grid_accessor = grid_image.grid
    assert grid_accessor is not None
    assert grid_accessor.nrows == 8
    assert grid_accessor.ncols == 12


@timeit
def test_grid_property_assignment_error():
    grid_image = GridImage()
    with pytest.raises(IllegalAssignmentError):
        grid_image.grid = "some other_image"


@timeit
def test_image_grid_section_retrieval(plate_grid_images_with_detection):
    grid_image = plate_grid_images_with_detection
    sub_image = grid_image[10:20, 10:30]
    assert isinstance(sub_image, Image)
    assert sub_image.shape[:2] == (10, 20)


@timeit
def test_grid_show_overlay(plate_grid_images_with_detection):
    grid_image = plate_grid_images_with_detection
    fig, ax = grid_image.show_overlay(show_labels=False)
    assert fig is not None
    assert ax is not None


@timeit
def test_optimal_grid_setter_defaults():
    grid_image = GridImage()
    grid_setter = grid_image.grid_finder
    assert isinstance(grid_setter, AutoGridFinder)
    assert grid_setter.nrows == 8
    assert grid_setter.ncols == 12


# ============================================================================================
# Test GridImage with Various Dtypes
# ============================================================================================


class TestGridImageDtypeHandling:
    """Tests for GridImage initialization with various dtypes."""

    @timeit
    def test_gridimage_uint8_rgb_initialization(self):
        """Test GridImage initialization with uint8 RGB plate array."""
        uint8_rgb = np.random.randint(0, 255, (512, 768, 3), dtype=np.uint8)
        grid_image = GridImage(arr=uint8_rgb, nrows=8, ncols=12)

        assert grid_image.isempty() is False
        assert grid_image.bit_depth == 8
        assert not grid_image.rgb.isempty()
        assert np.array_equal(grid_image.rgb[:], uint8_rgb)

    @timeit
    def test_gridimage_uint16_rgb_initialization(self):
        """Test GridImage initialization with uint16 RGB plate array."""
        uint16_rgb = np.random.randint(0, 65535, (512, 768, 3), dtype=np.uint16)
        grid_image = GridImage(arr=uint16_rgb, nrows=8, ncols=12)

        assert grid_image.isempty() is False
        assert grid_image.bit_depth == 16
        assert not grid_image.rgb.isempty()
        assert np.array_equal(grid_image.rgb[:], uint16_rgb)

    @timeit
    def test_gridimage_float32_rgb_initialization(self):
        """Test GridImage initialization with float32 RGB plate array."""
        float32_rgb = np.random.rand(512, 768, 3).astype(np.float32)
        grid_image = GridImage(arr=float32_rgb, nrows=8, ncols=12)

        assert grid_image.isempty() is False
        assert grid_image.bit_depth == 16
        # Float arrays are converted to uint based on bit_depth
        assert grid_image.rgb[:].dtype == np.uint16

    @timeit
    def test_gridimage_uint8_grayscale_initialization(self):
        """Test GridImage initialization with uint8 grayscale plate array."""
        uint8_gray = np.random.randint(0, 255, (512, 768), dtype=np.uint8)
        grid_image = GridImage(arr=uint8_gray, nrows=8, ncols=12)

        assert grid_image.isempty() is False
        assert grid_image.bit_depth == 8
        assert grid_image.rgb.isempty()  # No RGB for grayscale input
        assert np.array_equal(grid_image.gray[:], uint8_gray)

    @timeit
    def test_gridimage_float64_grayscale_initialization(self):
        """Test GridImage initialization with float64 grayscale plate array."""
        float64_gray = np.random.rand(512, 768).astype(np.float64)
        grid_image = GridImage(arr=float64_gray, nrows=8, ncols=12)

        assert grid_image.isempty() is False
        assert grid_image.bit_depth == 16
        # Grayscale float arrays are NOT converted (only RGB floats are)
        assert np.array_equal(grid_image.gray[:], float64_gray)

    @timeit
    def test_gridimage_bit_depth_preserved_with_grid_finder(self):
        """Test that bit_depth is preserved when using custom GridFinder."""
        uint16_rgb = np.random.randint(0, 65535, (512, 768, 3), dtype=np.uint16)
        finder = AutoGridFinder(nrows=8, ncols=12)
        grid_image = GridImage(arr=uint16_rgb, grid_finder=finder)

        assert grid_image.bit_depth == 16
        assert grid_image.grid_finder is finder

    @timeit
    def test_gridimage_explicit_bit_depth_respected(self):
        """Test that explicit bit_depth parameter is respected."""
        uint8_rgb = np.random.randint(0, 255, (512, 768, 3), dtype=np.uint8)
        grid_image = GridImage(arr=uint8_rgb, bit_depth=16)

        # Explicit bit_depth should override inferred bit_depth
        assert grid_image.bit_depth == 16


class TestGridImageBitDepthInheritance:
    """Tests for bit_depth inheritance from Image parent class."""

    @timeit
    def test_gridimage_inherits_image_bit_depth_property(self):
        """Test that GridImage inherits bit_depth property from Image."""
        uint8_array = np.random.randint(0, 255, (512, 768, 3), dtype=np.uint8)
        grid_image = GridImage(arr=uint8_array)

        # GridImage should have bit_depth property from Image parent
        assert hasattr(grid_image, "bit_depth")
        assert grid_image.bit_depth == 8

    @timeit
    def test_gridimage_bit_depth_consistency_across_operations(self):
        """Test that bit_depth remains consistent after grid operations."""
        uint16_array = np.random.randint(0, 65535, (512, 768, 3), dtype=np.uint16)
        grid_image = GridImage(arr=uint16_array, nrows=8, ncols=12)
        original_bit_depth = grid_image.bit_depth

        # Perform grid operations
        fig, ax = grid_image.show_overlay(show_labels=False)

        # Bit depth should remain unchanged
        assert grid_image.bit_depth == original_bit_depth

    @timeit
    def test_gridimage_sliced_image_inherits_bit_depth(self):
        """Test that sliced image from GridImage inherits bit_depth."""
        uint16_array = np.random.randint(0, 65535, (512, 768, 3), dtype=np.uint16)
        grid_image = GridImage(arr=uint16_array, nrows=8, ncols=12)

        # Slice a region from grid_image
        sliced = grid_image[100:200, 100:200]

        # Sliced image should be Image type and have correct bit_depth
        assert isinstance(sliced, Image)
        assert sliced.bit_depth == 16

    @timeit
    def test_gridimage_detector_preserves_bit_depth(self):
        """Test that detector operations preserve GridImage bit_depth."""
        uint8_array = np.random.randint(0, 255, (512, 768, 3), dtype=np.uint8)
        grid_image = GridImage(arr=uint8_array, nrows=8, ncols=12)
        original_bit_depth = grid_image.bit_depth

        # Apply detector
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        # Bit depth should be preserved
        assert detected.bit_depth == original_bit_depth

    @timeit
    def test_gridimage_different_dtypes_have_consistent_interface(self):
        """Test that GridImage interface is consistent across dtypes."""
        uint8_rgb = np.random.randint(0, 255, (512, 768, 3), dtype=np.uint8)
        uint16_rgb = np.random.randint(0, 65535, (512, 768, 3), dtype=np.uint16)
        float32_rgb = np.random.rand(512, 768, 3).astype(np.float32)

        grid_uint8 = GridImage(arr=uint8_rgb)
        grid_uint16 = GridImage(arr=uint16_rgb)
        grid_float32 = GridImage(arr=float32_rgb)

        # All should have grid property
        assert grid_uint8.grid is not None
        assert grid_uint16.grid is not None
        assert grid_float32.grid is not None

        # All should have same grid dimensions (defaults)
        assert (
            grid_uint8.grid.nrows
            == grid_uint16.grid.nrows
            == grid_float32.grid.nrows
            == 8
        )
        assert (
            grid_uint8.grid.ncols
            == grid_uint16.grid.ncols
            == grid_float32.grid.ncols
            == 12
        )

        # All should have correct bit_depth
        assert grid_uint8.bit_depth == 8
        assert grid_uint16.bit_depth == 16
        assert grid_float32.bit_depth == 16
