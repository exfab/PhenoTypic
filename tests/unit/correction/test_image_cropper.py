"""
Focused tests for ImageCropper parameter validation and boundary logic.

Tests focus on parameter validation and edge cases. Basic initialization and apply()
are covered by smoke tests in test_operation.py.
"""

import pytest
import numpy as np

from phenotypic import Image, GridImage
from phenotypic.correction import ImageCropper


class TestImageCropperParameterValidation:
    """Test ImageCropper parameter validation and error handling."""

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

    def test_zero_values_allowed(self):
        """Test that zero values are allowed (means no cropping from that edge)."""
        cropper = ImageCropper(left=0, right=0, top=0, bottom=0)
        assert cropper.left == 0
        assert cropper.right == 0
        assert cropper.top == 0
        assert cropper.bottom == 0


class TestImageCropperBoundaryLogic:
    """Test boundary calculation and edge cases."""

    def test_crop_content_matches_slice(self):
        """Test that cropped image content matches the expected slice."""
        arr = np.arange(200*300*3).reshape(200, 300, 3).astype(np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=50, right=50, top=60, bottom=60)
        cropped = cropper.apply(image)

        # Expected slice: [60:140, 50:250] (80x200)
        expected = arr[60:140, 50:250]
        assert np.array_equal(cropped.rgb[:], expected)

    def test_crop_to_minimal_region(self):
        """Test cropping that leaves a small region."""
        arr = np.ones((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=150, right=150, top=140, bottom=140)
        cropped = cropper.apply(image)

        # Should have 20x100 pixels remaining
        assert cropped.shape == (20, 100, 3)

    def test_crop_larger_than_image_raises_error(self):
        """Test that cropping with margins larger than image dimensions raises ValueError."""
        arr = np.ones((100, 200, 3), dtype=np.uint8)
        image = Image(arr=arr)

        # Attempt to crop more than height allows
        cropper = ImageCropper(top=60, bottom=60)  # Total would be 120, but height is 100
        
        # Should raise ValueError when crop margins exceed image dimensions
        with pytest.raises((ValueError, RuntimeError)):
            cropper.apply(image)

    def test_asymmetric_cropping_accuracy(self):
        """Test that asymmetric cropping is calculated correctly."""
        arr = np.ones((500, 600, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=100, right=150, top=80, bottom=120)
        cropped = cropper.apply(image)

        # Height: 500 - 80 - 120 = 300
        # Width: 600 - 100 - 150 = 350
        assert cropped.shape == (300, 350, 3)

    def test_zero_crop_returns_same_shape(self):
        """Test that zero crop parameters return same shape."""
        arr = np.ones((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=0, right=0, top=0, bottom=0)
        cropped = cropper.apply(image)

        assert cropped.shape == image.shape

    def test_none_parameters_skip_cropping(self):
        """Test that None parameters skip cropping on that edge."""
        arr = np.ones((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=50, right=None, top=40, bottom=None)
        cropped = cropper.apply(image)

        # Only left and top should be cropped
        # Height: 300 - 40 = 260
        # Width: 400 - 50 = 350
        assert cropped.shape == (260, 350, 3)


class TestImageCropperEdgeCases:
    """Test edge cases and boundary conditions."""

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

        # Check each channel is preserved
        assert cropped.rgb[:, :, 0].max() == 255
        assert cropped.rgb[:, :, 1].max() == 128
        assert cropped.rgb[:, :, 2].max() == 64

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

    def test_grayscale_image(self):
        """Test cropping on grayscale images."""
        arr = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
        image = Image(arr=arr)

        cropper = ImageCropper(left=50, right=50, top=60, bottom=60)
        cropped = cropper.apply(image)

        assert cropped.shape == (80, 200)

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


class TestImageCropperGridImageHandling:
    """Test ImageCropper behavior with GridImage instances."""

    def test_crop_grid_image_preserves_type(self):
        """Test that cropping a GridImage returns a GridImage."""
        arr = np.random.randint(0, 256, (800, 1000, 3), dtype=np.uint8)
        grid_img = GridImage(arr=arr, nrows=8, ncols=12)

        cropper = ImageCropper(left=100, right=100, top=100, bottom=100)
        cropped = cropper.apply(grid_img)

        assert isinstance(cropped, GridImage)
        assert cropped.shape == (600, 800, 3)

    def test_crop_grid_image_preserves_nrows_ncols(self):
        """Test that cropping preserves nrows and ncols."""
        arr = np.ones((1000, 1200, 3), dtype=np.uint8) * 128
        grid_img = GridImage(arr=arr, nrows=16, ncols=24)

        cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
        cropped = cropper.apply(grid_img)

        assert isinstance(cropped, GridImage)
        assert cropped.nrows == 16
        assert cropped.ncols == 24


# Run all tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
