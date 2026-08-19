import numpy as np
import pytest
from phenotypic import Image, GridImage
from phenotypic.grid import AutoGridFinder, CenteredAutoGridFinder
from phenotypic.detect import OtsuDetector
from phenotypic.sdk_.exceptions_ import IllegalAssignmentError

from ..resources.TestHelper import timeit


@timeit
def test_blank_gridimage_initialization():
    # Test default initialization
    grid_image = GridImage()
    assert grid_image is not None
    assert isinstance(grid_image.grid_finder, CenteredAutoGridFinder)


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
def test_grid_plot_overlay(plate_grid_images_with_detection):
    grid_image = plate_grid_images_with_detection
    fig, ax = grid_image.show(overlay=True, show_labels=False)
    assert fig is not None


@timeit
def test_optimal_grid_setter_defaults():
    grid_image = GridImage()
    grid_setter = grid_image.grid_finder
    assert isinstance(grid_setter, CenteredAutoGridFinder)
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
        """GridImage downcasts a float64 grayscale plate to the float32 contract."""
        float64_gray = np.random.rand(512, 768).astype(np.float64)
        grid_image = GridImage(arr=float64_gray, nrows=8, ncols=12)

        assert grid_image.isempty() is False
        assert grid_image.bit_depth == 16
        # gray/detect_mat are stored float32 (ImageData enforces it); the float64
        # input is preserved to float32 precision, not bit-exact.
        assert grid_image.gray[:].dtype == np.float32
        assert np.allclose(grid_image.gray[:], float64_gray, atol=1e-6)

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


# Tests for GridAccessor slicing support
class TestGridAccessorSlicing:
    """Test suite for GridAccessor slicing functionality."""

    @timeit
    def test_grid_accessor_flattened_slice(self, plate_grid_images_with_detection):
        """Test grid[start:stop] flattened index slicing."""
        grid_image = plate_grid_images_with_detection
        # First row for 8x12 grid (sections 0-11)
        first_row = grid_image.grid[0:12]

        assert isinstance(first_row, Image)
        assert first_row.shape[0] > 0  # Has pixels
        assert first_row.shape[1] > 0

    @timeit
    def test_grid_accessor_row_slice_all_cols(self, plate_grid_images_with_detection):
        """Test grid[row, :] pattern."""
        grid_image = plate_grid_images_with_detection
        row_2 = grid_image.grid[2, :]

        assert isinstance(row_2, Image)
        assert row_2.shape[0] > 0
        assert row_2.shape[1] > 0

    @timeit
    def test_grid_accessor_col_slice_all_rows(self, plate_grid_images_with_detection):
        """Test grid[:, col] pattern."""
        grid_image = plate_grid_images_with_detection
        col_5 = grid_image.grid[:, 5]

        assert isinstance(col_5, Image)
        assert col_5.shape[0] > 0
        assert col_5.shape[1] > 0

    @timeit
    def test_grid_accessor_row_range_single_col(self, plate_grid_images_with_detection):
        """Test grid[row_start:row_stop, col] pattern."""
        grid_image = plate_grid_images_with_detection
        subset = grid_image.grid[0:4, 3]

        assert isinstance(subset, Image)
        assert subset.shape[0] > 0
        assert subset.shape[1] > 0

    @timeit
    def test_grid_accessor_single_row_col_range(self, plate_grid_images_with_detection):
        """Test grid[row, col_start:col_stop] pattern."""
        grid_image = plate_grid_images_with_detection
        subset = grid_image.grid[2, 0:6]

        assert isinstance(subset, Image)
        assert subset.shape[0] > 0
        assert subset.shape[1] > 0

    @timeit
    def test_grid_accessor_2d_slice_raises_error(self,
                                                 plate_grid_images_with_detection):
        """Test that grid[rows, cols] raises ValueError."""
        grid_image = plate_grid_images_with_detection

        with pytest.raises(ValueError, match="both dimensions"):
            _ = grid_image.grid[0:5, 2:7]

    @timeit
    def test_grid_accessor_empty_slice(self, plate_grid_images_with_detection):
        """Test empty slice returns valid image."""
        grid_image = plate_grid_images_with_detection

        # Out of bounds slice should return empty
        empty = grid_image.grid[100:200]
        assert isinstance(empty, Image)

    @timeit
    def test_grid_accessor_negative_indices(self, plate_grid_images_with_detection):
        """Test negative index support."""
        grid_image = plate_grid_images_with_detection

        # Last row for 8x12 grid: sections 84-95
        last_row = grid_image.grid[-12:]
        assert isinstance(last_row, Image)

    @timeit
    def test_grid_accessor_step_slicing(self, plate_grid_images_with_detection):
        """Test step slicing support."""
        grid_image = plate_grid_images_with_detection

        every_other = grid_image.grid[::2]
        assert isinstance(every_other, Image)

    @timeit
    def test_grid_accessor_full_grid_slice(self, plate_grid_images_with_detection):
        """Test grid[:] and grid[:, :] return entire grid."""
        grid_image = plate_grid_images_with_detection

        # Full flattened slice
        all_flat = grid_image.grid[:]
        assert isinstance(all_flat, Image)
        assert all_flat.shape[0] > 0

        # Full 2D slice
        all_2d = grid_image.grid[:, :]
        assert isinstance(all_2d, Image)
        assert all_2d.shape[0] > 0

    @timeit
    def test_grid_accessor_single_section_backward_compat(self,
                                                          plate_grid_images_with_detection):
        """Test backward compatibility: single section access unchanged."""
        grid_image = plate_grid_images_with_detection

        # Single index (int)
        single_int = grid_image.grid[0]
        assert isinstance(single_int, Image)

        # Single index (tuple)
        single_tuple = grid_image.grid[0, 0]
        assert isinstance(single_tuple, Image)

    @timeit
    def test_grid_accessor_slice_return_type_is_image(self,
                                                      plate_grid_images_with_detection):
        """Test that slices return Image, not GridImage."""
        grid_image = plate_grid_images_with_detection

        sliced = grid_image.grid[0:12]
        assert type(sliced).__name__ == 'Image'
        assert not isinstance(sliced, GridImage)

    @timeit
    def test_grid_accessor_slice_metadata_marked_as_grid_section(self,
                                                                 plate_grid_images_with_detection):
        """Test that sliced images have GRID_SECTION metadata."""
        from phenotypic.schema import IMAGE
        from phenotypic.sdk_.constants_ import IMAGE_TYPES

        grid_image = plate_grid_images_with_detection
        sliced = grid_image.grid[0:12]

        # Check metadata marks it as GRID_SECTION
        image_type = sliced.metadata.get(IMAGE.IMAGE_TYPE)
        assert image_type == IMAGE_TYPES.GRID_SECTION.value

    @timeit
    def test_grid_accessor_slice_object_filtering(self,
                                                  plate_grid_images_with_detection):
        """Test that sliced images have filtered objects."""
        grid_image = plate_grid_images_with_detection

        # Get first row
        first_row = grid_image.grid[0:12]

        # Should have objects (or be empty if no objects in first row)
        assert first_row.objects is not None
        # Object map should be valid (unsigned integer dtype)
        assert np.issubdtype(first_row.objmap[:].dtype, np.integer)

    @timeit
    def test_grid_accessor_invalid_tuple_length_raises_error(self,
                                                             plate_grid_images_with_detection):
        """Test that invalid tuple length raises IndexError."""
        grid_image = plate_grid_images_with_detection

        with pytest.raises(IndexError, match="length 2"):
            _ = grid_image.grid[0, 1, 2]

    @timeit
    def test_grid_accessor_invalid_type_raises_error(self,
                                                     plate_grid_images_with_detection):
        """Test that invalid index type raises TypeError."""
        grid_image = plate_grid_images_with_detection

        with pytest.raises(TypeError):
            _ = grid_image.grid["invalid"]


# ============================================================================================
# Test COL_MAJOR_IDX in GridFinder
# ============================================================================================


class TestColMajorIdx:
    """Tests for column-major index calculation in grid info."""

    COL = "Grid_ColMajorIdx"
    ROW_MAJ = "Grid_RowMajorIdx"
    ROW_NUM = "Grid_RowNum"
    COL_NUM = "Grid_ColNum"

    @timeit
    def test_col_major_idx_column_exists(
            self, plate_grid_images_with_detection
    ):
        """COL_MAJOR_IDX column should be present in grid info."""
        grid_image = plate_grid_images_with_detection
        info = grid_image.grid.info()
        assert self.COL in info.columns

    @timeit
    def test_col_major_idx_ordering(
            self, plate_grid_images_with_detection
    ):
        """COL_MAJOR_IDX should follow col * nrows + row ordering."""
        grid_image = plate_grid_images_with_detection
        info = grid_image.grid.info()
        nrows = grid_image.nrows

        valid = info.dropna(
                subset=[self.ROW_NUM, self.COL_NUM, self.COL]
        )
        if valid.empty:
            pytest.skip("No valid grid assignments")

        row_nums = valid[self.ROW_NUM].astype(int).values
        col_nums = valid[self.COL_NUM].astype(int).values
        expected = col_nums * nrows + row_nums

        actual = valid[self.COL].astype(int).values
        np.testing.assert_array_equal(actual, expected)

    @timeit
    def test_col_major_idx_nan_matches_row_major(
            self, plate_grid_images_with_detection
    ):
        """NaN positions in COL_MAJOR_IDX should match ROW_MAJOR_IDX."""
        grid_image = plate_grid_images_with_detection
        info = grid_image.grid.info()

        row_major_na = info[self.ROW_MAJ].isna()
        col_major_na = info[self.COL].isna()
        assert row_major_na.equals(col_major_na)
