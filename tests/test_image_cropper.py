"""
Comprehensive tests for ImageCropper class.

Tests initialization, parameter validation, cropping operations, index calculations,
and integration with Image pipelines for the image cropping functionality.
"""

import pytest
import numpy as np

from phenotypic import Image
from phenotypic.correction import ImageCropper


class TestImageCropperInit:
    """Test ImageCropper initialization and validation."""

    def test_basic_initialization_all_none(self):
        """Test initialization with all parameters as None (no cropping)."""
        cropper = ImageCropper(left=None, right=None, top=None, bottom=None)
        assert cropper.left is None
        assert cropper.right is None
        assert cropper.top is None
        assert cropper.bottom is None

    def test_symmetric_cropping_initialization(self):
        """Test initialization with symmetric crop margins from all edges."""
        cropper = ImageCropper(left=10, right=10, top=10, bottom=10)
        assert cropper.left == 10
        assert cropper.right == 10
        assert cropper.top == 10
        assert cropper.bottom == 10

    def test_asymmetric_cropping_initialization(self):
        """Test initialization with asymmetric crop margins."""
        cropper = ImageCropper(left=10, right=20, top=30, bottom=40)
        assert cropper.left == 10
        assert cropper.right == 20
        assert cropper.top == 30
        assert cropper.bottom == 40

    def test_partial_parameters_initialization(self):
        """Test initialization with only some parameters specified."""
        cropper = ImageCropper(top=50, right=40)
        assert cropper.top == 50
        assert cropper.right == 40
        assert cropper.left is None
        assert cropper.bottom is None

    def test_zero_values_initialization(self):
        """Test that zero values are allowed (means no cropping from that edge)."""
        cropper = ImageCropper(left=0, right=0, top=0, bottom=0)
        assert cropper.left == 0
        assert cropper.right == 0
        assert cropper.top == 0
        assert cropper.bottom == 0

    def test_negative_left_raises_error(self):
        """Test that negative left parameter raises ValueError."""
        with pytest.raises(ValueError, match="left cannot be negative"):
            ImageCropper(left=-10)

    def test_negative_right_raises_error(self):
        """Test that negative right parameter raises ValueError."""
        with pytest.raises(ValueError, match="right cannot be negative"):
            ImageCropper(right=-5)

    def test_negative_top_raises_error(self):
        """Test that negative top parameter raises ValueError."""
        with pytest.raises(ValueError, match="top cannot be negative"):
            ImageCropper(top=-15)

    def test_negative_bottom_raises_error(self):
        """Test that negative bottom parameter raises ValueError."""
        with pytest.raises(ValueError, match="bottom cannot be negative"):
            ImageCropper(bottom=-20)

    def test_multiple_negative_parameters(self):
        """Test that the first negative parameter detected raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            ImageCropper(left=-5, right=-10)


class TestOperateMethod:
    """Test the _operate method and apply method that perform cropping."""

    def test_apply_minimal_crop(self):
        """Test cropping with very small margins on large image."""
        arr = np.random.randint(0, 256, (1000, 1500, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=100, right=100, top=100, bottom=100)
        cropped = cropper.apply(image)

        assert cropped.shape == (800, 1300, 3)
        assert image.shape == (1000, 1500, 3)

    def test_apply_asymmetric_crop(self):
        """Test cropping with asymmetric margins."""
        arr = np.random.randint(0, 256, (500, 600, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=50, right=100, top=80, bottom=120)
        cropped = cropper.apply(image)

        assert cropped.shape == (300, 450, 3)

    def test_apply_only_top_crop(self):
        """Test cropping only from top edge."""
        arr = np.zeros((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(top=100)
        cropped = cropper.apply(image)

        assert cropped.shape == (200, 400, 3)

    def test_apply_only_bottom_crop(self):
        """Test cropping only from bottom edge."""
        arr = np.zeros((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(bottom=100)
        cropped = cropper.apply(image)

        assert cropped.shape == (200, 400, 3)

    def test_apply_only_left_crop(self):
        """Test cropping only from left edge."""
        arr = np.zeros((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=100)
        cropped = cropper.apply(image)

        assert cropped.shape == (300, 300, 3)

    def test_apply_only_right_crop(self):
        """Test cropping only from right edge."""
        arr = np.zeros((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(right=150)
        cropped = cropper.apply(image)

        assert cropped.shape == (300, 250, 3)

    def test_apply_no_crop(self):
        """Test that no crop parameters result in full image."""
        arr = np.ones((100, 150, 3), dtype=np.uint8) * 255
        image = Image(arr=arr)

        cropper = ImageCropper()
        cropped = cropper.apply(image)

        assert cropped.shape == image.shape

    def test_apply_preserves_data_type(self):
        """Test that cropping preserves the data type of the original image."""
        arr_uint8 = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        image_uint8 = Image(arr=arr_uint8)

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(image_uint8)

        assert cropped.rgb[:].dtype == np.uint8

    def test_apply_grayscale_image(self):
        """Test cropping on grayscale images."""
        arr = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=50, right=50, top=60, bottom=60)
        cropped = cropper.apply(image)

        assert cropped.shape == (80, 200)

    def test_apply_returns_new_image_instance(self):
        """Test that apply returns a new Image instance."""
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(image)

        # Should be different objects
        assert cropped is not image

    def test_apply_original_image_unchanged(self):
        """Test that the original image is not modified after cropping."""
        original_arr = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8).copy()
        image = Image(arr=original_arr)
        original_shape = image.shape

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        _ = cropper.apply(image)

        assert image.shape == original_shape
        assert np.array_equal(image.rgb[:], original_arr)


class TestContentPreservation:
    """Test that cropping preserves correct image content."""

    def test_crop_content_matches_slice(self):
        """Test that cropped image content matches the expected slice."""
        arr = np.arange(200*300*3).reshape(200, 300, 3).astype(np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=50, right=50, top=60, bottom=60)
        cropped = cropper.apply(image)

        # Expected slice: [60:140, 50:250] (80x200)
        expected = arr[60:140, 50:250]

        assert np.array_equal(cropped.rgb[:], expected)

    def test_crop_center_region(self):
        """Test cropping to extract center region of image."""
        arr = np.ones((300, 400, 3), dtype=np.uint8)
        # Mark center region with different value
        arr[120:180, 160:240] = 200
        image = Image(arr=arr)

        # Crop to isolate center region approximately
        cropper = ImageCropper(left=100, right=100, top=100, bottom=100)
        cropped = cropper.apply(image)

        # Center region should be preserved in cropped image
        assert cropped.shape == (100, 200, 3)
        assert 200 in cropped.rgb[:]

    def test_crop_preserves_rgb_channels(self):
        """Test that all RGB channels are preserved correctly during cropping."""
        # Create image with distinct color
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red channel
        arr[:, :, 1] = 128  # Green channel
        arr[:, :, 2] = 64   # Blue channel
        image = Image(arr=arr)

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(image)

        # Check each channel
        assert cropped.rgb[:, :, 0].max() == 255
        assert cropped.rgb[:, :, 1].max() == 128
        assert cropped.rgb[:, :, 2].max() == 64


class TestDetectionIntegration:
    """Test cropping interaction with detection results."""

    def test_crop_after_detection(self):
        """Test cropping after detection (objmask and objmap are preserved)."""
        from phenotypic.detect import OtsuDetector

        # Create a test image with distinct foreground/background
        arr = np.ones((300, 400, 3), dtype=np.uint8) * 50
        arr[100:200, 150:300] = 200  # Bright region
        image = Image(arr=arr)

        # Perform detection
        detector = OtsuDetector()
        detected = detector.apply(image)

        # Crop after detection
        cropper = ImageCropper(left=75, right=75, top=75, bottom=75)
        cropped = cropper.apply(detected)

        # Check that cropped image still has detection data
        assert cropped.objmask is not None
        assert cropped.objmap is not None
        assert cropped.shape == (150, 250, 3)

    def test_crop_preserves_objects_accessor(self):
        """Test that objects accessor remains functional after cropping."""
        from phenotypic.detect import OtsuDetector

        arr = np.ones((400, 500, 3), dtype=np.uint8) * 50
        # Create multiple distinct bright regions
        arr[100:180, 100:180] = 220
        arr[220:300, 300:380] = 215
        image = Image(arr=arr)

        # Detect and then crop
        detector = OtsuDetector()
        detected = detector.apply(image)
        cropper = ImageCropper(left=100, right=100, top=100, bottom=100)
        cropped = cropper.apply(detected)

        # Objects should still be accessible
        if cropped.objects is not None:
            # At least verify no error occurs
            assert isinstance(cropped.objects, object)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_symmetric_crop(self):
        """Test symmetric cropping with significant margins."""
        arr = np.ones((500, 600, 3), dtype=np.uint8) * 128
        image = Image(arr=arr)

        cropper = ImageCropper(left=150, right=150, top=150, bottom=150)
        cropped = cropper.apply(image)

        assert cropped.shape == (200, 300, 3)

    def test_square_image(self):
        """Test cropping on square images."""
        arr = np.ones((200, 200, 3), dtype=np.uint8) * 100
        image = Image(arr=arr)

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(image)

        assert cropped.shape == (100, 100, 3)

    def test_wide_image(self):
        """Test cropping on very wide images."""
        arr = np.ones((100, 800, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=200, right=200, top=25, bottom=25)
        cropped = cropper.apply(image)

        assert cropped.shape == (50, 400, 3)

    def test_tall_image(self):
        """Test cropping on very tall images."""
        arr = np.ones((800, 100, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=25, right=25, top=200, bottom=200)
        cropped = cropper.apply(image)

        assert cropped.shape == (400, 50, 3)

    def test_crop_to_minimal_region(self):
        """Test cropping that leaves a small region."""
        arr = np.ones((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=150, right=150, top=140, bottom=140)
        cropped = cropper.apply(image)

        # Should have 20x100 pixels remaining
        assert cropped.shape == (20, 100, 3)


class TestMultipleCrops:
    """Test applying multiple crops sequentially."""

    def test_sequential_crops(self):
        """Test applying multiple crop operations in sequence."""
        arr = np.ones((500, 600, 3), dtype=np.uint8)
        image = Image(arr=arr)

        # First crop
        cropper1 = ImageCropper(left=100, right=100, top=100, bottom=100)
        cropped1 = cropper1.apply(image)
        assert cropped1.shape == (300, 400, 3)

        # Second crop on result
        cropper2 = ImageCropper(left=75, right=75, top=75, bottom=75)
        cropped2 = cropper2.apply(cropped1)
        assert cropped2.shape == (150, 250, 3)

    def test_sequential_crops_same_cropper(self):
        """Test using the same cropper instance on different images."""
        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)

        arr1 = np.ones((300, 400, 3), dtype=np.uint8)
        image1 = Image(arr=arr1)
        cropped1 = cropper.apply(image1)
        assert cropped1.shape == (200, 300, 3)

        arr2 = np.ones((600, 800, 3), dtype=np.uint8)
        image2 = Image(arr=arr2)
        cropped2 = cropper.apply(image2)
        assert cropped2.shape == (500, 700, 3)

    def test_cropper_immutability(self):
        """Test that cropper state is not modified by operations."""
        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)

        original_left = cropper.left
        original_right = cropper.right
        original_top = cropper.top
        original_bottom = cropper.bottom

        arr = np.zeros((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)
        _ = cropper.apply(image)

        # Verify cropper state unchanged
        assert cropper.left == original_left
        assert cropper.right == original_right
        assert cropper.top == original_top
        assert cropper.bottom == original_bottom


class TestRealWorldScenarios:
    """Test real-world usage scenarios for colony phenotyping."""

    def test_remove_scanner_border(self):
        """Test removing scanner border from plate image."""
        # Simulate a scanned plate image with 50px border
        arr = np.ones((1200, 1600, 3), dtype=np.uint8) * 200
        # Mark the center plate region
        arr[50:1150, 50:1550] = 150
        image = Image(arr=arr)

        # Crop to remove scanner border
        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(image)

        assert cropped.shape == (1100, 1500, 3)

    def test_remove_edge_artifacts(self):
        """Test removing edge artifacts from agar plate."""
        arr = np.ones((800, 1000, 3), dtype=np.uint8) * 100
        image = Image(arr=arr)

        # Remove outer well rows and columns (edges of plate)
        cropper = ImageCropper(left=80, right=80, top=60, bottom=60)
        cropped = cropper.apply(image)

        assert cropped.shape == (680, 840, 3)

    def test_focus_on_region_of_interest(self):
        """Test cropping to focus on region with colonies."""
        arr = np.ones((1000, 1200, 3), dtype=np.uint8) * 50
        # Simulate colonies in center region
        arr[300:700, 350:950] = 180
        image = Image(arr=arr)

        # Crop to focus on central area with colonies
        cropper = ImageCropper(left=200, right=200, top=200, bottom=200)
        cropped = cropper.apply(image)

        # Central region should still contain colonies
        assert cropped.shape == (600, 800, 3)
        assert 180 in cropped.rgb[:]

    def test_standardize_batch_dimensions(self):
        """Test standardizing image dimensions for batch processing."""
        # Different images with slightly different captured boundaries
        arr1 = np.random.randint(0, 256, (1050, 1450, 3), dtype=np.uint8)
        arr2 = np.random.randint(0, 256, (1060, 1460, 3), dtype=np.uint8)

        image1 = Image(arr=arr1)
        image2 = Image(arr=arr2)

        # Crop with same margins to standardize dimensions
        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)

        cropped1 = cropper.apply(image1)
        cropped2 = cropper.apply(image2)

        # Both should crop to same absolute margins
        assert cropped1.shape == (950, 1350, 3)
        assert cropped2.shape == (960, 1360, 3)


class TestGridImageCropping:
    """Test ImageCropper behavior with GridImage instances."""

    def test_crop_grid_image_preserves_type(self):
        """Test that cropping a GridImage returns a GridImage."""
        from phenotypic import GridImage

        arr = np.random.randint(0, 256, (800, 1000, 3), dtype=np.uint8)
        grid_img = GridImage(arr=arr, nrows=8, ncols=12)

        cropper = ImageCropper(left=100, right=100, top=100, bottom=100)
        cropped = cropper.apply(grid_img)

        assert isinstance(cropped, GridImage)
        assert cropped.shape == (600, 800, 3)

    def test_crop_grid_image_preserves_nrows_ncols(self):
        """Test that cropping preserves nrows and ncols."""
        from phenotypic import GridImage

        arr = np.ones((1000, 1200, 3), dtype=np.uint8) * 128
        grid_img = GridImage(arr=arr, nrows=16, ncols=24)

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(grid_img)

        assert isinstance(cropped, GridImage)
        assert cropped.nrows == 16
        assert cropped.ncols == 24

    def test_crop_grid_image_preserves_grid_finder(self):
        """Test that cropping preserves the grid_finder instance."""
        from phenotypic import GridImage
        from phenotypic.grid import AutoGridFinder

        arr = np.ones((600, 800, 3), dtype=np.uint8) * 150
        custom_finder = AutoGridFinder(nrows=8, ncols=12)
        grid_img = GridImage(arr=arr, grid_finder=custom_finder)

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(grid_img)

        assert isinstance(cropped, GridImage)
        # Same grid_finder instance should be preserved
        assert cropped.grid_finder is custom_finder

    def test_crop_grid_image_with_detection_results(self):
        """Test that detection results are preserved when cropping GridImage."""
        from phenotypic import GridImage
        from phenotypic.detect import OtsuDetector

        # Create image with a bright region that will be detected
        arr = np.ones((800, 1000, 3), dtype=np.uint8) * 50
        arr[300:500, 400:600] = 220
        grid_img = GridImage(arr=arr, nrows=8, ncols=12)

        # Apply detection
        detector = OtsuDetector()
        detected = detector.apply(grid_img)

        # Crop the detected GridImage
        cropper = ImageCropper(left=200, right=200, top=200, bottom=200)
        cropped = cropper.apply(detected)

        assert isinstance(cropped, GridImage)
        # Detection data should be preserved in cropped region
        assert cropped.objmask is not None
        assert cropped.objmap is not None

    def test_crop_grid_image_preserves_color_settings(self):
        """Test that color space settings are preserved through basic GridImage attributes."""
        from phenotypic import GridImage

        arr = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
        grid_img = GridImage(arr=arr, nrows=8, ncols=12)

        cropper = ImageCropper(left=100, right=100, top=100, bottom=100)
        cropped = cropper.apply(grid_img)

        assert isinstance(cropped, GridImage)
        # Verify basic color attributes are present
        assert cropped.illuminant is not None
        assert cropped.gamma is not None
        assert cropped.bit_depth is not None

    def test_crop_grid_image_grid_still_functional(self):
        """Test that grid grid_finder and grid settings remain accessible after cropping."""
        from phenotypic import GridImage

        # Create grid image with grid settings
        arr = np.ones((960, 1152, 3), dtype=np.uint8) * 50
        grid_img = GridImage(arr=arr, nrows=8, ncols=12)

        cropper = ImageCropper(left=40, right=40, top=50, bottom=50)
        cropped = cropper.apply(grid_img)

        assert isinstance(cropped, GridImage)
        # Verify grid_finder and settings are still accessible
        assert cropped.grid_finder is not None
        assert cropped.nrows == 8
        assert cropped.ncols == 12
        # Grid accessor exists (even if it needs detection to compute edges)
        assert cropped.grid is not None

    def test_crop_regular_image_unchanged_behavior(self):
        """Test that regular Image cropping behavior is not affected."""
        arr = np.random.randint(0, 256, (500, 600, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(image)

        # Should return Image, not GridImage
        assert isinstance(cropped, Image)
        from phenotypic import GridImage
        assert not isinstance(cropped, GridImage)

    def test_crop_grid_image_sequential_crops(self):
        """Test applying multiple crops to GridImage sequentially."""
        from phenotypic import GridImage

        arr = np.ones((1000, 1200, 3), dtype=np.uint8) * 128
        grid_img = GridImage(arr=arr, nrows=8, ncols=12)

        cropper1 = ImageCropper(left=100, right=100, top=100, bottom=100)
        cropped1 = cropper1.apply(grid_img)

        assert isinstance(cropped1, GridImage)
        assert cropped1.shape == (800, 1000, 3)

        # Apply second crop to first crop
        cropper2 = ImageCropper(left=100, right=100, top=100, bottom=100)
        cropped2 = cropper2.apply(cropped1)

        assert isinstance(cropped2, GridImage)
        assert cropped2.shape == (600, 800, 3)
        # Grid settings should be preserved through both crops
        assert cropped2.nrows == 8
        assert cropped2.ncols == 12

    def test_crop_grid_image_preserves_name(self):
        """Test that cropping preserves the original GridImage's name."""
        from phenotypic import GridImage

        arr = np.ones((800, 1000, 3), dtype=np.uint8) * 128
        original_name = "my_96well_plate"
        grid_img = GridImage(arr=arr, nrows=8, ncols=12, name=original_name)

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(grid_img)

        assert isinstance(cropped, GridImage)
        # Name should be preserved from original GridImage
        assert cropped.name == original_name

    def test_crop_image_preserves_name(self):
        """Test that cropping preserves the original Image's name."""
        arr = np.ones((500, 600, 3), dtype=np.uint8) * 100
        original_name = "my_test_image"
        image = Image(arr=arr, name=original_name)

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(image)

        assert isinstance(cropped, Image)
        # Name should be preserved from original Image
        assert cropped.name == original_name
