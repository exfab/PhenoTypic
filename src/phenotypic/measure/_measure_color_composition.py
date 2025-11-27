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
        hue = hsv_foreground[..., 0] * self.hue_normalization
        saturation = hsv_foreground[..., 1] * self.sat_normalization
        value = hsv_foreground[..., 2] * self.val_normalization

        # Get object map for per-object analysis
        objmap = image.objmap[:]

        # Get unique object labels (excluding background 0)
        object_labels = image.objects.labels2series()

        # Compute color composition for each object
        logger.info("Computing color composition for each object")
        results = []

        for label in object_labels:
            # Create mask for this object
            obj_mask = objmap == label

            # Extract HSV values for this object only
            obj_hue = hue[obj_mask]
            obj_sat = saturation[obj_mask]
            obj_val = value[obj_mask]

            # Calculate color percentages using the 11-color model
            percentages = self._classify_colors(
                obj_hue, obj_sat, obj_val,
                black_max=self.black_value_max,
                neutral_sat=self.neutral_sat_max,
                white_min=self.white_value_min,
                gray_min=self.gray_value_min,
                gray_max=self.gray_value_max
            )
            results.append(percentages)

        # Create DataFrame
        data = {header: [result[i] for result in results]
                for i, header in enumerate(ColorComposition.all_headers())}

        meas = pd.DataFrame(data=data)
        meas.insert(loc=0, column=OBJECT.LABEL, value=object_labels)

        return meas

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
            return [0.0] * 12

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
        percentages = (counts / total_pixels * 100).tolist()

        return percentages

    def visualize_masks(self, image: Image, top_n: int = 3, figsize: tuple = (15, 10)):
        """
        Visualize the color classification masks for debugging purposes.

        Displays the original RGB image alongside masks for the top N most prevalent
        colors detected in the image. This method processes data dynamically to avoid
        memory issues, only creating the masks needed for visualization.

        Args:
            image: The PhenoTypic Image object to visualize
            top_n: Number of top colors to display (default: 3)
            figsize: Figure size as (width, height) tuple (default: (15, 10))

        Returns:
            tuple: (matplotlib.figure.Figure, numpy.ndarray of axes)

        Example:
            >>> from phenotypic import Image
            >>> from phenotypic.measure import MeasureColorComposition
            >>> img = Image.load('path/to/image.tif')
            >>> measurer = MeasureColorComposition()
            >>> fig, axes = measurer.visualize_masks(img, top_n=3)
        """
        import matplotlib.pyplot as plt

        # Get HSV data - computed dynamically, not stored
        hsv_foreground = image.color.hsv.foreground()
        hue = hsv_foreground[..., 0] * self.hue_normalization
        saturation = hsv_foreground[..., 1] * self.sat_normalization
        value = hsv_foreground[..., 2] * self.val_normalization

        # Clean up large array immediately after extraction
        del hsv_foreground

        # Get object mask for foreground
        objmask = image.objmask[:]

        # Flatten HSV arrays for classification (views, not copies)
        hue_flat = hue[objmask]
        sat_flat = saturation[objmask]
        val_flat = value[objmask]

        # Classify colors using instance parameters
        total_pixels = len(hue_flat)
        classification = np.full(total_pixels, -1, dtype=np.int8)

        # Color indices
        BLACK, WHITE, GRAY, PINK, BROWN, RED, ORANGE, YELLOW, GREEN, CYAN, BLUE, PURPLE = range(12)

        # Priority 1: Neutrals (use instance parameters)
        black_mask = val_flat < self.black_value_max
        classification[black_mask] = BLACK
        white_mask = (sat_flat < self.neutral_sat_max) & (val_flat > self.white_value_min) & (classification == -1)
        classification[white_mask] = WHITE
        gray_mask = (sat_flat < self.neutral_sat_max) & (val_flat >= self.gray_value_min) & (val_flat <= self.gray_value_max) & (classification == -1)
        classification[gray_mask] = GRAY

        # Priority 2: Special colors
        pink_hue_mask = ((hue_flat <= 15) | (hue_flat >= 250))
        pink_mask = pink_hue_mask & (sat_flat >= 20) & (sat_flat <= 60) & (val_flat > 80) & (classification == -1)
        classification[pink_mask] = PINK
        brown_hue_mask = (hue_flat <= 45)
        brown_mask = brown_hue_mask & (val_flat >= 20) & (val_flat <= 60) & (classification == -1)
        classification[brown_mask] = BROWN

        # Priority 3: Standard hues
        red_mask = ((hue_flat <= 15) | (hue_flat >= 345)) & (classification == -1)
        classification[red_mask] = RED
        orange_mask = (hue_flat > 15) & (hue_flat <= 45) & (classification == -1)
        classification[orange_mask] = ORANGE
        yellow_mask = (hue_flat > 45) & (hue_flat <= 75) & (classification == -1)
        classification[yellow_mask] = YELLOW
        green_mask = (hue_flat > 75) & (hue_flat <= 150) & (classification == -1)
        classification[green_mask] = GREEN
        cyan_mask = (hue_flat > 150) & (hue_flat <= 180) & (classification == -1)
        classification[cyan_mask] = CYAN
        blue_mask = (hue_flat > 180) & (hue_flat <= 250) & (classification == -1)
        classification[blue_mask] = BLUE
        purple_mask = (hue_flat > 250) & (hue_flat < 345) & (classification == -1)
        classification[purple_mask] = PURPLE

        # Clean up intermediate arrays
        del hue_flat, sat_flat, val_flat, hue, saturation, value

        # Calculate percentages to find top colors
        counts = np.bincount(classification[classification >= 0], minlength=12)
        percentages = (counts / total_pixels * 100) if total_pixels > 0 else np.zeros(12)

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

        # Display top color masks (create on-demand, don't store all)
        for i, color_idx in enumerate(top_indices):
            # Create mask for this specific color only (memory efficient)
            color_mask_flat = (classification == color_idx)

            # Reconstruct 2D mask only for this color
            mask_2d = np.zeros(image.shape[:2], dtype=bool)
            mask_2d[objmask] = color_mask_flat

            # Display mask
            axes[i + 1].imshow(mask_2d, cmap='gray')
            axes[i + 1].set_title(f'{color_names[color_idx]}\n{percentages[color_idx]:.1f}%')
            axes[i + 1].axis('off')

            # Clean up immediately
            del color_mask_flat, mask_2d

        # Clean up classification array
        del classification

        plt.tight_layout()
        return fig, axes


# Append documentation from MeasurementInfo class
MeasureColorComposition.__doc__ = ColorComposition.append_rst_to_doc(MeasureColorComposition)
