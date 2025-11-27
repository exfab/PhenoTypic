import pytest
import numpy as np
import pandas as pd

import phenotypic
from phenotypic.measure import MeasureColorComposition
from phenotypic.data import load_plate_72hr
from phenotypic.detect import OtsuDetector
from phenotypic.refine import MaskOpener


class TestMeasureColorComposition:
    """Test suite for MeasureColorComposition class."""

    @pytest.fixture
    def detected_image(self):
        """Create a detected image for testing."""
        image = phenotypic.GridImage(load_plate_72hr())
        OtsuDetector(ignore_borders=True).apply(image, inplace=True)
        MaskOpener().apply(image, inplace=True)
        return image

    @pytest.fixture
    def synthetic_rgb_image(self):
        """Create a synthetic RGB image with known colors for testing."""
        # Create a 100x100 RGB image with different color regions
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)

        # Black region (0-20, 0-100)
        img_array[0:20, :] = [0, 0, 0]

        # White region (20-40, 0-100)
        img_array[20:40, :] = [255, 255, 255]

        # Gray region (40-60, 0-100)
        img_array[40:60, :] = [128, 128, 128]

        # Red region (60-70, 0-100)
        img_array[60:70, :] = [255, 0, 0]

        # Green region (70-80, 0-100)
        img_array[70:80, :] = [0, 255, 0]

        # Blue region (80-90, 0-100)
        img_array[80:90, :] = [0, 0, 255]

        # Pink region (90-100, 0-100) - light red
        img_array[90:100, :] = [255, 192, 203]

        return img_array

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        measurer = MeasureColorComposition()
        assert measurer.hue_normalization == 360.0
        assert measurer.sat_normalization == 100.0
        assert measurer.val_normalization == 100.0
        assert measurer.black_value_max == 20.0
        assert measurer.neutral_sat_max == 15.0
        assert measurer.white_value_min == 85.0
        assert measurer.gray_value_min == 20.0
        assert measurer.gray_value_max == 85.0

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        measurer = MeasureColorComposition(
            hue_normalization=1.0,
            sat_normalization=1.0,
            val_normalization=1.0,
            black_value_max=15.0,
            neutral_sat_max=10.0,
            white_value_min=90.0,
            gray_value_min=25.0,
            gray_value_max=80.0
        )
        assert measurer.hue_normalization == 1.0
        assert measurer.sat_normalization == 1.0
        assert measurer.val_normalization == 1.0
        assert measurer.black_value_max == 15.0
        assert measurer.neutral_sat_max == 10.0
        assert measurer.white_value_min == 90.0
        assert measurer.gray_value_min == 25.0
        assert measurer.gray_value_max == 80.0

    def test_measure_returns_dataframe(self, detected_image):
        """Test that measure returns a pandas DataFrame."""
        measurer = MeasureColorComposition()
        result = measurer.measure(detected_image)
        assert isinstance(result, pd.DataFrame)

    def test_measure_has_correct_columns(self, detected_image):
        """Test that the output DataFrame has the correct columns."""
        measurer = MeasureColorComposition()
        result = measurer.measure(detected_image)

        expected_columns = [
            'ObjectLabel',
            'ColorComposition_BlackPct', 'ColorComposition_WhitePct', 'ColorComposition_GrayPct',
            'ColorComposition_PinkPct', 'ColorComposition_BrownPct',
            'ColorComposition_RedPct', 'ColorComposition_OrangePct', 'ColorComposition_YellowPct',
            'ColorComposition_GreenPct', 'ColorComposition_CyanPct', 'ColorComposition_BluePct',
            'ColorComposition_PurplePct'
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_measure_percentages_sum_to_100(self, detected_image):
        """Test that color percentages sum to approximately 100% for each object."""
        measurer = MeasureColorComposition()
        result = measurer.measure(detected_image)

        color_columns = [
            'ColorComposition_BlackPct', 'ColorComposition_WhitePct', 'ColorComposition_GrayPct',
            'ColorComposition_PinkPct', 'ColorComposition_BrownPct',
            'ColorComposition_RedPct', 'ColorComposition_OrangePct', 'ColorComposition_YellowPct',
            'ColorComposition_GreenPct', 'ColorComposition_CyanPct', 'ColorComposition_BluePct',
            'ColorComposition_PurplePct'
        ]

        for idx, row in result.iterrows():
            total = sum(row[col] for col in color_columns)
            assert np.isclose(total, 100.0, atol=0.1), \
                f"Row {idx}: percentages sum to {total}, expected 100"

    def test_measure_with_custom_thresholds(self, detected_image):
        """Test measurement with custom neutral thresholds."""
        measurer_strict = MeasureColorComposition(
            black_value_max=15.0,
            neutral_sat_max=10.0,
            white_value_min=90.0
        )

        result = measurer_strict.measure(detected_image)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_classify_colors_empty_array(self):
        """Test classification with empty arrays."""
        empty_hue = np.array([])
        empty_sat = np.array([])
        empty_val = np.array([])

        result = MeasureColorComposition._classify_colors(empty_hue, empty_sat, empty_val)

        # Should return 12 zeros
        assert len(result) == 12
        assert all(x == 0.0 for x in result)

    def test_classify_colors_pure_black(self):
        """Test classification of pure black pixels."""
        # 100 pixels with H=0, S=0, V=0 (black)
        hue = np.zeros(100)
        sat = np.zeros(100)
        val = np.zeros(100)

        result = MeasureColorComposition._classify_colors(hue, sat, val)

        # Should be 100% black
        assert result[0] == 100.0  # BlackPct
        assert sum(result[1:]) == 0.0  # All others should be 0

    def test_classify_colors_pure_white(self):
        """Test classification of pure white pixels."""
        # 100 pixels with H=0, S=0, V=100 (white)
        hue = np.zeros(100)
        sat = np.zeros(100)
        val = np.full(100, 100.0)

        result = MeasureColorComposition._classify_colors(hue, sat, val)

        # Should be 100% white
        assert result[1] == 100.0  # WhitePct
        assert result[0] == 0.0  # BlackPct
        assert sum(result[2:]) == 0.0  # All others should be 0

    def test_classify_colors_pure_gray(self):
        """Test classification of pure gray pixels."""
        # 100 pixels with H=0, S=0, V=50 (gray)
        hue = np.zeros(100)
        sat = np.zeros(100)
        val = np.full(100, 50.0)

        result = MeasureColorComposition._classify_colors(hue, sat, val)

        # Should be 100% gray
        assert result[2] == 100.0  # GrayPct
        assert result[0] == 0.0  # BlackPct
        assert result[1] == 0.0  # WhitePct

    def test_classify_colors_pure_red(self):
        """Test classification of pure red pixels."""
        # 100 pixels with H=0 (red), S=100, V=100
        hue = np.zeros(100)
        sat = np.full(100, 100.0)
        val = np.full(100, 100.0)

        result = MeasureColorComposition._classify_colors(hue, sat, val)

        # Should be 100% red (index 5)
        assert result[5] == 100.0  # RedPct
        assert sum(result[:5]) == 0.0  # No neutrals
        assert sum(result[6:]) == 0.0  # No other colors

    def test_classify_colors_pure_green(self):
        """Test classification of pure green pixels."""
        # 100 pixels with H=120 (green), S=100, V=100
        hue = np.full(100, 120.0)
        sat = np.full(100, 100.0)
        val = np.full(100, 100.0)

        result = MeasureColorComposition._classify_colors(hue, sat, val)

        # Should be 100% green (index 8)
        assert result[8] == 100.0  # GreenPct

    def test_classify_colors_pure_blue(self):
        """Test classification of pure blue pixels."""
        # 100 pixels with H=240 (blue), S=100, V=100
        hue = np.full(100, 240.0)
        sat = np.full(100, 100.0)
        val = np.full(100, 100.0)

        result = MeasureColorComposition._classify_colors(hue, sat, val)

        # Should be 100% blue (index 10)
        assert result[10] == 100.0  # BluePct

    def test_classify_colors_pink(self):
        """Test classification of pink pixels."""
        # Pink: red/magenta hue, sat 20-60, val > 80
        hue = np.full(100, 10.0)  # Red hue
        sat = np.full(100, 40.0)  # Mid saturation
        val = np.full(100, 90.0)  # High value

        result = MeasureColorComposition._classify_colors(hue, sat, val)

        # Should be 100% pink (index 3)
        assert result[3] == 100.0  # PinkPct

    def test_classify_colors_brown(self):
        """Test classification of brown pixels."""
        # Brown: red/orange hue, val 20-60
        hue = np.full(100, 30.0)  # Orange hue
        sat = np.full(100, 60.0)  # Mid-high saturation
        val = np.full(100, 40.0)  # Mid value

        result = MeasureColorComposition._classify_colors(hue, sat, val)

        # Should be 100% brown (index 4)
        assert result[4] == 100.0  # BrownPct

    def test_classify_colors_mixed(self):
        """Test classification of mixed color pixels."""
        # 50 black + 50 white
        hue = np.zeros(100)
        sat = np.zeros(100)
        val = np.concatenate([np.zeros(50), np.full(50, 100.0)])

        result = MeasureColorComposition._classify_colors(hue, sat, val)

        # Should be 50% black, 50% white
        assert result[0] == 50.0  # BlackPct
        assert result[1] == 50.0  # WhitePct
        assert sum(result[2:]) == 0.0  # All others should be 0

    def test_classify_colors_priority_hierarchy(self):
        """Test that priority hierarchy is respected."""
        # Pixels with low saturation should be classified as neutrals
        # even if they have non-zero hue
        hue = np.full(100, 120.0)  # Green hue
        sat = np.full(100, 5.0)    # Very low saturation
        val = np.full(100, 50.0)   # Mid value

        result = MeasureColorComposition._classify_colors(hue, sat, val)

        # Should be classified as gray (neutral), not green
        assert result[2] == 100.0  # GrayPct
        assert result[8] == 0.0    # GreenPct should be 0

    def test_classify_colors_custom_thresholds(self):
        """Test classification with custom threshold parameters."""
        # Test with stricter black threshold
        hue = np.zeros(100)
        sat = np.zeros(100)
        val = np.full(100, 18.0)  # Just below default black_max=20

        # With default threshold (20), should be black
        result_default = MeasureColorComposition._classify_colors(
            hue, sat, val, black_max=20.0
        )
        assert result_default[0] == 100.0  # BlackPct

        # With stricter threshold (15), should be gray
        result_strict = MeasureColorComposition._classify_colors(
            hue, sat, val, black_max=15.0, gray_min=15.0
        )
        assert result_strict[2] == 100.0  # GrayPct

    def test_visualize_masks_returns_figure(self, detected_image):
        """Test that visualize_masks returns a matplotlib figure."""
        import matplotlib.pyplot as plt

        measurer = MeasureColorComposition()
        fig, axes = measurer.visualize_masks(detected_image, top_n=3)

        assert isinstance(fig, plt.Figure)
        assert len(axes) == 4  # 1 original + 3 color masks

        # Clean up
        plt.close(fig)

    def test_visualize_masks_with_different_top_n(self, detected_image):
        """Test visualize_masks with different top_n values."""
        import matplotlib.pyplot as plt

        measurer = MeasureColorComposition()

        for top_n in [1, 2, 3, 5]:
            fig, axes = measurer.visualize_masks(detected_image, top_n=top_n)
            assert len(axes) == top_n + 1
            plt.close(fig)

    def test_visualize_masks_custom_thresholds(self, detected_image):
        """Test that visualize_masks uses custom thresholds."""
        import matplotlib.pyplot as plt

        measurer = MeasureColorComposition(
            black_value_max=15.0,
            neutral_sat_max=10.0
        )

        fig, axes = measurer.visualize_masks(detected_image, top_n=2)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_number_of_rows_matches_objects(self, detected_image):
        """Test that the number of rows matches the number of detected objects."""
        measurer = MeasureColorComposition()
        result = measurer.measure(detected_image)

        # Get number of objects from the image
        n_objects = len(detected_image.objects.labels2series())

        assert len(result) == n_objects

    def test_object_labels_match(self, detected_image):
        """Test that object labels in result match those in the image."""
        measurer = MeasureColorComposition()
        result = measurer.measure(detected_image)

        expected_labels = detected_image.objects.labels2series().tolist()
        actual_labels = result['ObjectLabel'].tolist()

        assert expected_labels == actual_labels

    def test_all_values_are_numeric(self, detected_image):
        """Test that all measurement values are numeric."""
        measurer = MeasureColorComposition()
        result = measurer.measure(detected_image)

        color_columns = [
            'ColorComposition_BlackPct', 'ColorComposition_WhitePct', 'ColorComposition_GrayPct',
            'ColorComposition_PinkPct', 'ColorComposition_BrownPct',
            'ColorComposition_RedPct', 'ColorComposition_OrangePct', 'ColorComposition_YellowPct',
            'ColorComposition_GreenPct', 'ColorComposition_CyanPct', 'ColorComposition_BluePct',
            'ColorComposition_PurplePct'
        ]

        for col in color_columns:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_no_negative_percentages(self, detected_image):
        """Test that there are no negative percentage values."""
        measurer = MeasureColorComposition()
        result = measurer.measure(detected_image)

        color_columns = [
            'ColorComposition_BlackPct', 'ColorComposition_WhitePct', 'ColorComposition_GrayPct',
            'ColorComposition_PinkPct', 'ColorComposition_BrownPct',
            'ColorComposition_RedPct', 'ColorComposition_OrangePct', 'ColorComposition_YellowPct',
            'ColorComposition_GreenPct', 'ColorComposition_CyanPct', 'ColorComposition_BluePct',
            'ColorComposition_PurplePct'
        ]

        for col in color_columns:
            assert (result[col] >= 0).all(), f"Negative values found in {col}"

    def test_no_percentages_over_100(self, detected_image):
        """Test that no percentage value exceeds 100%."""
        measurer = MeasureColorComposition()
        result = measurer.measure(detected_image)

        color_columns = [
            'ColorComposition_BlackPct', 'ColorComposition_WhitePct', 'ColorComposition_GrayPct',
            'ColorComposition_PinkPct', 'ColorComposition_BrownPct',
            'ColorComposition_RedPct', 'ColorComposition_OrangePct', 'ColorComposition_YellowPct',
            'ColorComposition_GreenPct', 'ColorComposition_CyanPct', 'ColorComposition_BluePct',
            'ColorComposition_PurplePct'
        ]

        for col in color_columns:
            assert (result[col] <= 100).all(), f"Values over 100% found in {col}"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
