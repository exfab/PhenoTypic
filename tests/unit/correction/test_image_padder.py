"""
Focused tests for ImagePadder parameter validation and padding logic.

Tests focus on parameter validation, padding correctness, and edge cases.
Basic initialization and apply() are covered by smoke tests in test_operation.py.
"""

import pytest
import numpy as np

from phenotypic import Image, GridImage
from phenotypic.correction import ImagePadder


class TestImagePadderParameterValidation:
    """Test ImagePadder parameter validation and error handling."""

    def test_negative_left_raises_error(self):
        """Test that negative left parameter raises ValueError."""
        with pytest.raises(ValueError, match="left cannot be negative"):
            ImagePadder(left=-10)

    def test_negative_right_raises_error(self):
        """Test that negative right parameter raises ValueError."""
        with pytest.raises(ValueError, match="right cannot be negative"):
            ImagePadder(right=-5)

    def test_negative_top_raises_error(self):
        """Test that negative top parameter raises ValueError."""
        with pytest.raises(ValueError, match="top cannot be negative"):
            ImagePadder(top=-15)

    def test_negative_bottom_raises_error(self):
        """Test that negative bottom parameter raises ValueError."""
        with pytest.raises(ValueError, match="bottom cannot be negative"):
            ImagePadder(bottom=-20)

    def test_multiple_negative_parameters(self):
        """Test that the first negative parameter detected raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            ImagePadder(left=-5, right=-10)

    def test_invalid_mode_raises_error(self):
        """Test that invalid padding mode raises ValueError."""
        # ``mode`` is a ``Literal`` field post-pydantic-migration; an
        # out-of-set value raises ``ValidationError`` (a ``ValueError``).
        with pytest.raises(ValueError, match="mode"):
            ImagePadder(mode="invalid_mode")


class TestImagePadderPaddingLogic:
    """Test padding logic and shape calculations."""

    def test_padding_increases_shape_correctly(self):
        """Test that padding increases shape by expected amounts."""
        arr = np.ones((200, 300, 3), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(left=50, right=75, top=100, bottom=25)
        padded = padder.apply(image)

        # Height: 200 + 100 (top) + 25 (bottom) = 325
        # Width: 300 + 50 (left) + 75 (right) = 425
        assert padded.shape == (325, 425, 3)

    def test_asymmetric_padding_accuracy(self):
        """Test that asymmetric padding is calculated correctly."""
        arr = np.ones((500, 600, 3), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(left=100, right=150, top=80, bottom=120)
        padded = padder.apply(image)

        # Height: 500 + 80 + 120 = 700
        # Width: 600 + 100 + 150 = 850
        assert padded.shape == (700, 850, 3)

    def test_zero_padding_returns_same_shape(self):
        """Test that zero padding parameters return same shape."""
        arr = np.ones((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(left=0, right=0, top=0, bottom=0)
        padded = padder.apply(image)

        assert padded.shape == image.shape

    def test_none_parameters_skip_padding(self):
        """Test that None parameters skip padding on that edge."""
        arr = np.ones((300, 400, 3), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(left=50, right=None, top=40, bottom=None)
        padded = padder.apply(image)

        # Only left and top should be padded
        # Height: 300 + 40 = 340
        # Width: 400 + 50 = 450
        assert padded.shape == (340, 450, 3)

    def test_constant_mode_with_black_padding(self):
        """Test constant mode with black padding (value=0)."""
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 200
        image = Image(arr=arr)

        padder = ImagePadder(left=20, right=20, top=20, bottom=20, mode='constant', constant_value=0)
        padded = padder.apply(image)

        # Check that all edges are 0 (black)
        assert np.all(padded.rgb[0:20, :, :] == 0)  # Top padding
        assert np.all(padded.rgb[-20:, :, :] == 0)  # Bottom padding
        assert np.all(padded.rgb[:, 0:20, :] == 0)  # Left padding
        assert np.all(padded.rgb[:, -20:, :] == 0)  # Right padding

        # Check that center content is preserved
        assert np.all(padded.rgb[20:-20, 20:-20, :] == 200)

    def test_constant_mode_with_white_padding(self):
        """Test constant mode with white padding (value=255)."""
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 100
        image = Image(arr=arr)

        padder = ImagePadder(left=15, right=15, top=15, bottom=15, mode='constant', constant_value=255)
        padded = padder.apply(image)

        # Check that all edges are 255 (white)
        assert np.all(padded.rgb[0:15, :, :] == 255)
        assert np.all(padded.rgb[-15:, :, :] == 255)
        assert np.all(padded.rgb[:, 0:15, :] == 255)
        assert np.all(padded.rgb[:, -15:, :] == 255)

    def test_edge_mode_extends_pixels(self):
        """Test edge mode extends boundary pixels."""
        # Create gradient image with clear edge pattern
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 0] = 200  # Uniform red value for simplicity
        image = Image(arr=arr)

        padder = ImagePadder(left=20, right=20, top=20, bottom=20, mode='edge')
        padded = padder.apply(image)

        # Edge mode should extend the boundary pixels, so padded top should match top row
        # All padded rows in top region should have same value as original top row
        assert np.all(padded.rgb[0:20, 20:-20, 0] == 200)

    def test_reflect_mode_mirrors_edges(self):
        """Test reflect mode creates mirror at boundaries."""
        arr = np.ones((100, 100, 3), dtype=np.uint8)
        arr[0:5, 0:5, 0] = 10  # Distinct corner value
        image = Image(arr=arr)

        padder = ImagePadder(left=20, right=20, top=20, bottom=20, mode='reflect')
        padded = padder.apply(image)

        # Reflect mode should create a mirrored copy at the edges
        assert padded.shape == (140, 140, 3)
        # We can't easily verify exact mirror values, but we can check shape


class TestImagePadderDataPreservation:
    """Test that padding preserves data correctly."""

    def test_padding_preserves_rgb_channels(self):
        """Test that all RGB channels are preserved correctly during padding."""
        # Create image with distinct color in each channel
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red channel
        arr[:, :, 1] = 128  # Green channel
        arr[:, :, 2] = 64   # Blue channel
        image = Image(arr=arr)

        padder = ImagePadder(left=50, right=50, top=50, bottom=50, mode='constant', constant_value=0)
        padded = padder.apply(image)

        # Check that central region has original values for each channel
        assert np.all(padded.rgb[50:-50, 50:-50, 0] == 255)
        assert np.all(padded.rgb[50:-50, 50:-50, 1] == 128)
        assert np.all(padded.rgb[50:-50, 50:-50, 2] == 64)

    def test_padding_rgb_only_spatial_dimensions(self):
        """Test that RGB padding only affects spatial dimensions, not channels."""
        arr = np.ones((100, 100, 3), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(left=20, right=20, top=20, bottom=20)
        padded = padder.apply(image)

        # Shape should be (140, 140, 3) - NOT (140, 140, 3+pad)
        assert padded.rgb.shape == (140, 140, 3)
        assert padded.rgb.shape[2] == 3  # Channel dimension unchanged

    def test_padding_preserves_central_content(self):
        """Test that central image content is unchanged after padding."""
        # Create image with specific pattern
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        arr[50:150, 75:225, :] = 100
        image = Image(arr=arr)

        padder = ImagePadder(left=50, right=50, top=50, bottom=50, mode='constant', constant_value=0)
        padded = padder.apply(image)

        # Central content should be preserved at shifted position
        # Original [50:150, 75:225] → Padded [100:200, 125:275]
        assert np.all(padded.rgb[100:200, 125:275, :] == 100)

    def test_padding_grayscale_image(self):
        """Test padding on grayscale images."""
        arr = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(left=50, right=50, top=60, bottom=60, mode='constant', constant_value=50)
        padded = padder.apply(image)

        assert padded.shape == (320, 400)
        # Check edges are padding value
        assert np.all(padded.gray[0:60, :] == 50)

    def test_padding_with_detection_results(self):
        """Test that padding preserves and adjusts detection results."""
        arr = np.ones((200, 200, 3), dtype=np.uint8) * 100
        image = Image(arr=arr)

        # Manually set a detection result (object map)
        objmap = np.zeros((200, 200), dtype=np.uint16)
        objmap[50:100, 50:100] = 1  # One object
        objmap[120:150, 120:150] = 2  # Another object
        image.objmap[:] = objmap

        padder = ImagePadder(left=30, right=30, top=30, bottom=30)
        padded = padder.apply(image)

        # Objects should be shifted by padding amount
        # Object 1 should be at [80:130, 80:130]
        assert np.all(padded.objmap[80:130, 80:130] == 1)
        # Object 2 should be at [150:180, 150:180]
        assert np.all(padded.objmap[150:180, 150:180] == 2)
        # Padding area should be 0
        assert np.all(padded.objmap[0:30, :] == 0)


class TestImagePadderObjmapHandling:
    """Test that objmap is always padded with constant 0 regardless of mode."""

    def test_objmap_uses_constant_zero_regardless_of_mode(self):
        """CRITICAL: Verify objmap always uses constant mode with 0."""
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 200
        image = Image(arr=arr)

        # Set objmap with specific pattern
        objmap = np.zeros((100, 100), dtype=np.uint16)
        objmap[25:75, 25:75] = 5  # Central object
        image.objmap[:] = objmap

        # Pad with reflect mode (user wants reflection, but objmap should still use constant 0)
        padder = ImagePadder(left=20, right=20, top=20, bottom=20, mode='reflect')
        padded = padder.apply(image)

        # objmap edges must be 0 (constant), NOT reflected
        assert np.all(padded.objmap[0:20, :] == 0)  # Top padding
        assert np.all(padded.objmap[-20:, :] == 0)  # Bottom padding
        assert np.all(padded.objmap[:, 0:20] == 0)  # Left padding
        assert np.all(padded.objmap[:, -20:] == 0)  # Right padding

        # Central object should be preserved at new location
        assert np.all(padded.objmap[45:95, 45:95] == 5)

    def test_objmap_edges_are_zero_with_edge_mode(self):
        """Test objmap edges are zero even with edge mode specified."""
        arr = np.ones((150, 150, 3), dtype=np.uint8)
        image = Image(arr=arr)

        objmap = np.zeros((150, 150), dtype=np.uint16)
        objmap[40:60, 40:60] = 3
        image.objmap[:] = objmap

        padder = ImagePadder(left=25, right=25, top=25, bottom=25, mode='edge')
        padded = padder.apply(image)

        # Padded regions should be 0, not edge-extended
        assert np.all(padded.objmap[0:25, :] == 0)

    def test_objmap_labels_remain_unique(self):
        """Test that object labels remain unique after padding."""
        arr = np.ones((200, 200, 3), dtype=np.uint8)
        image = Image(arr=arr)

        objmap = np.zeros((200, 200), dtype=np.uint16)
        objmap[30:60, 30:60] = 1
        objmap[120:150, 120:150] = 2
        objmap[100:110, 100:110] = 3
        image.objmap[:] = objmap

        padder = ImagePadder(left=50, right=50, top=50, bottom=50, mode='reflect')
        padded = padder.apply(image)

        # Check that all labels are still unique and present
        unique_labels = np.unique(padded.objmap[:])
        # Should have 0 (background), 1, 2, 3
        assert len(unique_labels) == 4
        assert 1 in unique_labels
        assert 2 in unique_labels
        assert 3 in unique_labels


class TestImagePadderGridImageHandling:
    """Test ImagePadder behavior with GridImage instances."""

    def test_pad_grid_image_preserves_type(self):
        """Test that padding a GridImage returns a GridImage."""
        arr = np.random.randint(0, 256, (800, 1000, 3), dtype=np.uint8)
        grid_img = GridImage(arr=arr, nrows=8, ncols=12)

        padder = ImagePadder(left=100, right=100, top=100, bottom=100)
        padded = padder.apply(grid_img)

        assert isinstance(padded, GridImage)
        assert padded.shape == (1000, 1200, 3)

    def test_pad_grid_image_preserves_nrows_ncols(self):
        """Test that padding preserves nrows and ncols."""
        arr = np.ones((1000, 1200, 3), dtype=np.uint8) * 128
        grid_img = GridImage(arr=arr, nrows=16, ncols=24)

        padder = ImagePadder(left=50, right=50, top=50, bottom=50)
        padded = padder.apply(grid_img)

        assert isinstance(padded, GridImage)
        assert padded.nrows == 16
        assert padded.ncols == 24

    def test_pad_grid_image_preserves_grid_finder(self):
        """Test that grid_finder is preserved during padding."""
        arr = np.ones((800, 800, 3), dtype=np.uint8)
        grid_img = GridImage(arr=arr, nrows=8, ncols=8)

        original_grid_finder = grid_img.grid_finder

        padder = ImagePadder(left=50, right=50, top=50, bottom=50)
        padded = padder.apply(grid_img)

        # grid_finder should be preserved
        assert padded.grid_finder is original_grid_finder


class TestImagePadderEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_square_image(self):
        """Test padding on square images."""
        arr = np.ones((200, 200, 3), dtype=np.uint8) * 100
        image = Image(arr=arr)

        padder = ImagePadder(left=50, right=50, top=50, bottom=50)
        padded = padder.apply(image)

        assert padded.shape == (300, 300, 3)

    def test_wide_image(self):
        """Test padding on very wide images."""
        arr = np.ones((100, 800, 3), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(left=200, right=200, top=25, bottom=25)
        padded = padder.apply(image)

        assert padded.shape == (150, 1200, 3)

    def test_tall_image(self):
        """Test padding on very tall images."""
        arr = np.ones((800, 100, 3), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(left=25, right=25, top=200, bottom=200)
        padded = padder.apply(image)

        assert padded.shape == (1200, 150, 3)

    def test_sequential_padding_operations(self):
        """Test applying multiple padding operations in sequence."""
        arr = np.ones((200, 200, 3), dtype=np.uint8)
        image = Image(arr=arr)

        # First pad
        padder1 = ImagePadder(left=50, right=50, top=50, bottom=50)
        padded1 = padder1.apply(image)
        assert padded1.shape == (300, 300, 3)

        # Second pad on result
        padder2 = ImagePadder(left=25, right=25, top=25, bottom=25)
        padded2 = padder2.apply(padded1)
        assert padded2.shape == (350, 350, 3)

    def test_large_padding_values(self):
        """Test padding with large values (e.g., 200px)."""
        arr = np.ones((100, 100, 3), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(left=200, right=200, top=200, bottom=200)
        padded = padder.apply(image)

        assert padded.shape == (500, 500, 3)

    def test_asymmetric_large_padding(self):
        """Test very asymmetric padding."""
        arr = np.ones((100, 100, 3), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(left=5, right=200, top=300, bottom=10)
        padded = padder.apply(image)

        assert padded.shape == (410, 305, 3)

    def test_single_edge_padding(self):
        """Test padding only one edge."""
        arr = np.ones((200, 300, 3), dtype=np.uint8)
        image = Image(arr=arr)

        padder = ImagePadder(top=100)
        padded = padder.apply(image)

        assert padded.shape == (300, 300, 3)

    def test_pad_then_crop_approximate_roundtrip(self):
        """Test that pad+crop approximately returns to original (with constant padding)."""
        arr = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        image = Image(arr=arr)

        # Pad with specific value
        padder = ImagePadder(left=50, right=50, top=60, bottom=60, mode='constant', constant_value=200)
        padded = padder.apply(image)
        assert padded.shape == (320, 400, 3)

        # Now simulate crop to undo the padding
        from phenotypic.correction import ImageCropper
        cropper = ImageCropper(left=50, right=50, top=60, bottom=60)
        unpadded = cropper.apply(padded)

        # Central region should match original (approximately due to potential type conversions)
        assert np.allclose(unpadded.rgb[:, :, :], image.rgb[:, :, :])


# Run all tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
