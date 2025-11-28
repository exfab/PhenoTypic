from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import pandas as pd
import logging

from phenotypic.abc_ import MeasureFeatures, MeasurementInfo
from phenotypic.tools.constants_ import OBJECT

logger = logging.getLogger(__name__)


class ColorComposition(MeasurementInfo):
    """Measurement info for perceptual color composition using 11-color model."""

    @classmethod
    def category(cls):
        return 'ColorComposition'

    # Define the 11 color categories with descriptions
    BLACK_PCT = ('BlackPct', 'Percentage of pixels classified as black (Value < 20)')
    WHITE_PCT = ('WhitePct', 'Percentage of pixels classified as white (Saturation < 15, Value > 85)')
    GRAY_PCT = ('GrayPct', 'Percentage of pixels classified as gray (Saturation < 15, Value 20-85)')
    PINK_PCT = ('PinkPct', 'Percentage of pixels classified as pink (Red/Magenta hue, Saturation 20-60, Value > 80)')
    BROWN_PCT = ('BrownPct', 'Percentage of pixels classified as brown (Red/Orange hue, Value 20-60)')
    RED_PCT = ('RedPct', 'Percentage of pixels classified as red (Hue 0-15° or 345-360°)')
    ORANGE_PCT = ('OrangePct', 'Percentage of pixels classified as orange (Hue 15-45°)')
    YELLOW_PCT = ('YellowPct', 'Percentage of pixels classified as yellow (Hue 45-75°)')
    GREEN_PCT = ('GreenPct', 'Percentage of pixels classified as green (Hue 75-150°)')
    CYAN_PCT = ('CyanPct', 'Percentage of pixels classified as cyan (Hue 150-180°)')
    BLUE_PCT = ('BluePct', 'Percentage of pixels classified as blue (Hue 180-250°)')
    PURPLE_PCT = ('PurplePct', 'Percentage of pixels classified as purple/magenta (Hue 250-345°)')

    @classmethod
    def all_headers(cls):
        """Return all color composition measurement headers."""
        return [
            str(cls.BLACK_PCT),
            str(cls.WHITE_PCT),
            str(cls.GRAY_PCT),
            str(cls.PINK_PCT),
            str(cls.BROWN_PCT),
            str(cls.RED_PCT),
            str(cls.ORANGE_PCT),
            str(cls.YELLOW_PCT),
            str(cls.GREEN_PCT),
            str(cls.CYAN_PCT),
            str(cls.BLUE_PCT),
            str(cls.PURPLE_PCT),
        ]


class MeasureColorComposition(MeasureFeatures):
    """
    Performs perceptual color composition analysis on segmented image objects.

    This class extends the MeasureFeatures class to provide color composition
    analysis using an 11-color perceptual model. The model applies a priority
    hierarchy to classify pixels into categories that better match human color
    perception than simple hue-based classification.

    The 11 color categories are:
    - Neutrals (Priority 1): Black, White, Gray
    - Special Colors (Priority 2): Pink, Brown
    - Standard Hues (Priority 3): Red, Orange, Yellow, Green, Cyan, Blue, Purple/Magenta

    Implementation uses NumPy vectorization for efficiency, with no pixel-level loops.
    The classification follows human perception principles where neutrals take priority,
    followed by special cases (pink/brown), and finally standard hue-based colors.

    Args:
        hue_normalization (float): Multiplier to normalize hue to 0-360 range. Default is 360.0
            (assuming input hue is in 0-1 range from skimage).
        sat_normalization (float): Multiplier to normalize saturation to 0-100 range. Default is 100.0.
        val_normalization (float): Multiplier to normalize value/brightness to 0-100 range. Default is 100.0.
        black_value_max (float): Maximum value threshold for black classification. Default is 20.
        neutral_sat_max (float): Maximum saturation threshold for white and gray classification. Default is 15.
        white_value_min (float): Minimum value threshold for white classification. Default is 85.
        gray_value_min (float): Minimum value threshold for gray classification. Default is 20.
        gray_value_max (float): Maximum value threshold for gray classification. Default is 85.

    Example:
        >>> from phenotypic import Image
        >>> from phenotypic.measure import MeasureColorComposition
        >>> img = Image.load('path/to/image.tif')
        >>> measurer = MeasureColorComposition()
        >>> composition = measurer.measure(img)
        >>> print(composition)
        >>>
        >>> # Custom thresholds for different lighting conditions
        >>> measurer_custom = MeasureColorComposition(
        ...     black_value_max=15,  # Stricter black threshold
        ...     white_value_min=90   # Stricter white threshold
        ... )
    """

    def __init__(self,
                 hue_normalization: float = 360.0,
                 sat_normalization: float = 100.0,
                 val_normalization: float = 100.0,
                 black_value_max: float = 20.0,
                 neutral_sat_max: float = 15.0,
                 white_value_min: float = 85.0,
                 gray_value_min: float = 20.0,
                 gray_value_max: float = 85.0):
        """
        Initialize the color composition measurer.

        Args:
            hue_normalization: Multiplier to normalize hue to 0-360 range
            sat_normalization: Multiplier to normalize saturation to 0-100 range
            val_normalization: Multiplier to normalize value to 0-100 range
            black_value_max: Maximum value threshold for black classification
            neutral_sat_max: Maximum saturation threshold for white and gray classification
            white_value_min: Minimum value threshold for white classification
            gray_value_min: Minimum value threshold for gray classification
            gray_value_max: Maximum value threshold for gray classification
        """
        self.hue_normalization = hue_normalization
        self.sat_normalization = sat_normalization
        self.val_normalization = val_normalization
        self.black_value_max = black_value_max
        self.neutral_sat_max = neutral_sat_max
        self.white_value_min = white_value_min
        self.gray_value_min = gray_value_min
        self.gray_value_max = gray_value_max

    def _operate(self, image: Image) -> pd.DataFrame:
        """
        Execute color composition analysis on the image.

        Args:
            image: The PhenoTypic Image object to analyze

        Returns:
            pd.DataFrame: DataFrame with object labels and color composition percentages
        """
        # Get HSV representation (shape: H x W x 3)
        # Note: skimage's rgb2hsv returns H in [0,1], S in [0,1], V in [0,1]
        hsv_foreground = image.color.hsv.foreground()

        # Normalize to human-readable ranges: H: 0-360, S: 0-100, V: 0-100
        hue = hsv_foreground[..., 0]*self.hue_normalization
        saturation = hsv_foreground[..., 1]*self.sat_normalization
        value = hsv_foreground[..., 2]*self.val_normalization

        # Get object map for per-object analysis
        objmap = image.objmap[:]

        # Get unique object labels (excluding background 0)
        object_labels = image.objects.labels2series()

        # Compute color composition for each object using vectorized masks
        logger.info("Computing color composition for each object")

        # Generate all color masks (2D boolean arrays)
        color_masks = self._get_all_color_masks(hue, saturation, value)

        # Calculate percentages for all objects using vectorized operations
        results = self._calculate_sum(objmap, object_labels, color_masks)

        # Create DataFrame
        data = {header: [result[i] for result in results]
                for i, header in enumerate(ColorComposition.all_headers())}

        meas = pd.DataFrame(data=data)
        meas.insert(loc=0, column=OBJECT.LABEL, value=object_labels)

        return meas

    def _get_black_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
        """Get mask for black pixels."""
        return val < self.black_value_max

    def _get_white_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                        exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for white pixels."""
        return (sat < self.neutral_sat_max) & (val > self.white_value_min) & ~exclude_mask

    def _get_gray_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                       exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for gray pixels."""
        return ((sat < self.neutral_sat_max) & (val >= self.gray_value_min) &
                (val <= self.gray_value_max) & ~exclude_mask)

    def _get_pink_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                       exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for pink pixels."""
        pink_hue_mask = ((hue <= 15) | (hue >= 250))
        return pink_hue_mask & (sat >= 20) & (sat <= 60) & (val > 80) & ~exclude_mask

    def _get_brown_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                        exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for brown pixels."""
        brown_hue_mask = (hue <= 45)
        return brown_hue_mask & (val >= 20) & (val <= 60) & ~exclude_mask

    def _get_red_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                      exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for red pixels."""
        return ((hue <= 15) | (hue >= 345)) & ~exclude_mask

    def _get_orange_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                         exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for orange pixels."""
        return (hue > 15) & (hue <= 45) & ~exclude_mask

    def _get_yellow_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                         exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for yellow pixels."""
        return (hue > 45) & (hue <= 75) & ~exclude_mask

    def _get_green_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                        exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for green pixels."""
        return (hue > 75) & (hue <= 150) & ~exclude_mask

    def _get_cyan_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                       exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for cyan pixels."""
        return (hue > 150) & (hue <= 180) & ~exclude_mask

    def _get_blue_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                       exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for blue pixels."""
        return (hue > 180) & (hue <= 250) & ~exclude_mask

    def _get_purple_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                         exclude_mask: np.ndarray) -> np.ndarray:
        """Get mask for purple pixels."""
        return (hue > 250) & (hue < 345) & ~exclude_mask

    def _get_all_color_masks(self, hue: np.ndarray, sat: np.ndarray,
                             val: np.ndarray) -> list[np.ndarray]:
        """
        Generate all 12 color masks in priority order.

        Args:
            hue: Hue values (H x W)
            sat: Saturation values (H x W)
            val: Value/brightness (H x W)

        Returns:
            List of 12 boolean masks (one per color category)
        """
        # Track which pixels have been classified (cumulative exclude mask)
        exclude_mask = np.zeros(hue.shape, dtype=bool)

        # Priority 1: Neutrals
        black_mask = self._get_black_mask(hue, sat, val)
        exclude_mask |= black_mask

        white_mask = self._get_white_mask(hue, sat, val, exclude_mask)
        exclude_mask |= white_mask

        gray_mask = self._get_gray_mask(hue, sat, val, exclude_mask)
        exclude_mask |= gray_mask

        # Priority 2: Special colors
        pink_mask = self._get_pink_mask(hue, sat, val, exclude_mask)
        exclude_mask |= pink_mask

        brown_mask = self._get_brown_mask(hue, sat, val, exclude_mask)
        exclude_mask |= brown_mask

        # Priority 3: Standard hues
        red_mask = self._get_red_mask(hue, sat, val, exclude_mask)
        exclude_mask |= red_mask

        orange_mask = self._get_orange_mask(hue, sat, val, exclude_mask)
        exclude_mask |= orange_mask

        yellow_mask = self._get_yellow_mask(hue, sat, val, exclude_mask)
        exclude_mask |= yellow_mask

        green_mask = self._get_green_mask(hue, sat, val, exclude_mask)
        exclude_mask |= green_mask

        cyan_mask = self._get_cyan_mask(hue, sat, val, exclude_mask)
        exclude_mask |= cyan_mask

        blue_mask = self._get_blue_mask(hue, sat, val, exclude_mask)
        exclude_mask |= blue_mask

        purple_mask = self._get_purple_mask(hue, sat, val, exclude_mask)

        return [black_mask, white_mask, gray_mask, pink_mask, brown_mask,
                red_mask, orange_mask, yellow_mask, green_mask, cyan_mask,
                blue_mask, purple_mask]

    def _calculate_sum(self, objmap: np.ndarray, object_labels: pd.Series,
                       color_masks: list[np.ndarray]) -> list[list[float]]:
        """
        Calculate color composition percentages for each object.

        Args:
            objmap: Object label map (H x W)
            object_labels: Series of object labels to process
            color_masks: List of 12 boolean masks (one per color)

        Returns:
            List of percentage lists (one per object, 12 values each)
        """
        results = []

        for label in object_labels:
            # Create mask for this object
            obj_mask = objmap == label
            total_pixels = obj_mask.sum()

            if total_pixels == 0:
                # No pixels in object, return zeros
                results.append([0.0] * 12)
                continue

            # Calculate percentage for each color
            percentages = []
            for color_mask in color_masks:
                count = (obj_mask & color_mask).sum()
                percentage = (count / total_pixels * 100)
                percentages.append(percentage)

            results.append(percentages)

        return results

    @staticmethod
    def _classify_colors(hue: np.ndarray, sat: np.ndarray, val: np.ndarray,
                         black_max: float = 20.0,
                         neutral_sat: float = 15.0,
                         white_min: float = 85.0,
                         gray_min: float = 20.0,
                         gray_max: float = 85.0) -> list:
        """
        Classify pixels into 11 perceptual color categories using priority hierarchy.

        This method implements a human perception-based color classification that differs
        from pure mathematical HSV classification. The priority order ensures that:
        1. Neutrals (black/white/gray) are identified first regardless of hue noise
        2. Special cases (pink/brown) are identified before standard hues
        3. Standard hue-based colors fill in the remaining pixels

        This approach better matches human color naming than simple hue binning.

        Args:
            hue: Hue values normalized to 0-360 range
            sat: Saturation values normalized to 0-100 range
            val: Value/brightness normalized to 0-100 range
            black_max: Maximum value threshold for black classification
            neutral_sat: Maximum saturation threshold for white and gray
            white_min: Minimum value threshold for white classification
            gray_min: Minimum value threshold for gray classification
            gray_max: Maximum value threshold for gray classification

        Returns:
            list: Percentages for each of the 11 color categories (in order matching all_headers)
        """
        total_pixels = len(hue)
        if total_pixels == 0:
            # Return zeros for all categories if no pixels
            return [0.0]*12

        # Initialize classification array (will store color index for each pixel)
        # -1 means unclassified
        classification = np.full(total_pixels, -1, dtype=np.int8)

        # Color indices (matching order in all_headers)
        BLACK, WHITE, GRAY, PINK, BROWN, RED, ORANGE, YELLOW, GREEN, CYAN, BLUE, PURPLE = range(12)

        # PRIORITY 1: NEUTRALS (take precedence over all hue-based classifications)
        # Human perception: Very low saturation means we perceive it as achromatic
        # regardless of nominal hue value (which is often noise in low-sat regions)

        # Black: Very dark pixels
        black_mask = val < black_max
        classification[black_mask] = BLACK

        # White: Very bright and desaturated pixels
        white_mask = (sat < neutral_sat) & (val > white_min) & (classification == -1)
        classification[white_mask] = WHITE

        # Gray: Mid-range value with low saturation
        gray_mask = (sat < neutral_sat) & (val >= gray_min) & (val <= gray_max) & (classification == -1)
        classification[gray_mask] = GRAY

        # PRIORITY 2: SPECIAL COLORS (complex saturation/value dependencies)
        # These are perceptual categories that don't fit pure hue binning

        # Pink: Desaturated red/magenta with high brightness
        # Human perception: We call high-value, low-mid saturation reds "pink" not "light red"
        pink_hue_mask = ((hue <= 15) | (hue >= 250))
        pink_mask = pink_hue_mask & (sat >= 20) & (sat <= 60) & (val > 80) & (classification == -1)
        classification[pink_mask] = PINK

        # Brown: Dark orange/red tones
        # Human perception: We perceive dark orange as "brown" not "dark orange"
        # This is a key difference between math (HSV) and human perception
        brown_hue_mask = (hue <= 45)
        brown_mask = brown_hue_mask & (val >= 20) & (val <= 60) & (classification == -1)
        classification[brown_mask] = BROWN

        # PRIORITY 3: STANDARD HUES (for remaining chromatic pixels)
        # Now apply standard hue-based classification to remaining pixels
        # These are the "pure" colors humans recognize from the color wheel

        # Red: 0-15° and 345-360° (wraps around)
        red_mask = ((hue <= 15) | (hue >= 345)) & (classification == -1)
        classification[red_mask] = RED

        # Orange: 15-45°
        orange_mask = (hue > 15) & (hue <= 45) & (classification == -1)
        classification[orange_mask] = ORANGE

        # Yellow: 45-75°
        yellow_mask = (hue > 45) & (hue <= 75) & (classification == -1)
        classification[yellow_mask] = YELLOW

        # Green: 75-150°
        green_mask = (hue > 75) & (hue <= 150) & (classification == -1)
        classification[green_mask] = GREEN

        # Cyan: 150-180°
        cyan_mask = (hue > 150) & (hue <= 180) & (classification == -1)
        classification[cyan_mask] = CYAN

        # Blue: 180-250°
        blue_mask = (hue > 180) & (hue <= 250) & (classification == -1)
        classification[blue_mask] = BLUE

        # Purple/Magenta: 250-345°
        purple_mask = (hue > 250) & (hue < 345) & (classification == -1)
        classification[purple_mask] = PURPLE

        # Calculate percentages for each color
        # Use np.bincount for efficient counting
        counts = np.bincount(classification[classification >= 0], minlength=12)
        percentages = (counts/total_pixels*100).tolist()

        return percentages

    def visualize_masks(self, image: Image, top_n: int = 3, figsize: tuple = (15, 10)):
        """
        Visualize the color classification masks for debugging purposes.

        Displays the original RGB image alongside masks for the top N most prevalent
        colors detected in the image. This method uses vectorized 2D masks to avoid
        memory issues with large images.

        Args:
            image: The PhenoTypic Image object to visualize
            top_n: Number of top colors to display (default: 3)
            figsize: Figure size as (width, height) tuple (default: (15, 10))

        Returns:
            tuple: (matplotlib.figure.Figure, numpy.ndarray of axes)

        Example:
            >>> from phenotypic import Image
            >>> from phenotypic.measure import MeasureColorComposition
            >>> img = Image.imread('path/to/image.tif')
            >>> measurer = MeasureColorComposition()
            >>> fig, axes = measurer.visualize_masks(img, top_n=3)
        """
        import matplotlib.pyplot as plt

        # Get HSV data - computed dynamically, not stored
        hsv_foreground = image.color.hsv.foreground()
        hue = hsv_foreground[..., 0]*self.hue_normalization
        saturation = hsv_foreground[..., 1]*self.sat_normalization
        value = hsv_foreground[..., 2]*self.val_normalization

        # Clean up large array immediately after extraction
        del hsv_foreground

        # Get object mask for foreground
        objmask = image.objmask[:] == 1

        # Generate all color masks using the helper methods (works on 2D arrays)
        color_masks = self._get_all_color_masks(hue, saturation, value)

        # Clean up HSV arrays
        del hue, saturation, value

        # Calculate percentages for foreground pixels only
        total_pixels = objmask.sum()
        if total_pixels == 0:
            total_pixels = 1  # Avoid division by zero

        percentages = []
        for color_mask in color_masks:
            # Count only foreground pixels for each color
            count = (objmask & color_mask).sum()
            percentage = (count / total_pixels * 100)
            percentages.append(percentage)

        percentages = np.array(percentages)

        # Get top N colors
        color_names = ['Black', 'White', 'Gray', 'Pink', 'Brown', 'Red',
                       'Orange', 'Yellow', 'Green', 'Cyan', 'Blue', 'Purple']
        top_indices = np.argsort(percentages)[::-1][:top_n]

        # Create visualization
        n_plots = top_n + 1  # +1 for original image
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        # Display original image
        axes[0].imshow(image.rgb[:])
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Display top color masks (only the ones we need)
        for i, color_idx in enumerate(top_indices):
            # Get the mask for this color (already 2D, no reconstruction needed)
            color_mask = color_masks[color_idx] & objmask

            # Display mask
            axes[i + 1].imshow(color_mask, cmap='gray')
            axes[i + 1].set_title(f'{color_names[color_idx]}\n{percentages[color_idx]:.1f}%')
            axes[i + 1].axis('off')

        plt.tight_layout()
        return fig, axes


# Append documentation from MeasurementInfo class
MeasureColorComposition.__doc__ = ColorComposition.append_rst_to_doc(MeasureColorComposition)
