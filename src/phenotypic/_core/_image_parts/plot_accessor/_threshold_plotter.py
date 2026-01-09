from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold

from ._base_plotter import BasePlotter


class ThresholdPlotter(BasePlotter):
    """Provides thresholding visualization methods for image processing pipelines.

    This class offers methods to visualize and compare different thresholding
    techniques for segmenting colonies on agar plates.
    """

    def try_thresh(self, figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize and compare various thresholding techniques for colony segmentation.

        Applies multiple thresholding algorithms to the enhanced grayscale image and displays
        the results in a grid, enabling rapid evaluation of which method works best for
        segmenting colonies on agar plates. This is particularly useful during pipeline
        development when tuning detection parameters.

        Thresholding methods compared include:
            - Otsu's method (automatic threshold)
            - Li's method (minimum cross-entropy)
            - Isodata method (optimal threshold)
            - Mean method (simple average)
            - And several others from scikit-image

        Args:
            figsize: Figure size as (width, height) in inches. Larger figures make it
                easier to discern fine details in colony boundaries. Default: (10, 8).

        Returns:
            Tuple containing (fig, axes) where:
                - fig: Matplotlib figure object
                - axes: Array of matplotlib axes objects showing threshold results

        Note:
            This method does not cover all detection methods in phenotypic.
            For more sophisticated detection, use ObjectDetector classes directly.

        Raises:
            ValueError: If enhanced grayscale image is unavailable.
        """
        # Validate parameters
        self._validate_figsize(figsize)

        return try_all_threshold(image=self._root_image.enh_gray[:], figsize=figsize)



