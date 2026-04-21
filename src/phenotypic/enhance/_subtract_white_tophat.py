from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.morphology import white_tophat

from phenotypic.abc_ import ImageEnhancer


class SubtractWhiteTophat(ImageEnhancer):
    """Suppress small bright artifacts in ``detect_mat`` by subtracting the white top-hat.

    Computes the white top-hat (original minus morphological opening) and
    subtracts it from the image, removing small bright blobs such as dust
    specks, glare highlights, and condensation artifacts while preserving
    larger colony structures.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        shape: Footprint geometry. ``'diamond'`` (default) or ``'disk'``
            provide isotropic behavior; ``'square'`` can align with sensor
            grid artifacts.
        width: Maximum bright-object size (pixels) targeted for removal.
            Set slightly smaller than the smallest colonies to preserve
            them. ``None`` (default) derives a small value from image
            dimensions.

    Returns:
        Image: Input image with ``detect_mat`` smoothed by subtracting
        the white top-hat. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Removing small bright artifacts that could be mistaken for tiny
          colonies.
        - Reducing glare highlights on shiny plates before thresholding.
        - Cleaning up dust and condensation artifacts that confuse
          detection.

    Consider Also:
        - :class:`WhiteTophatEnhance` when you want to isolate (not
          suppress) small bright structures.
        - :class:`GrayOpening` for morphological smoothing that removes
          small bright features without explicit subtraction.
        - :class:`RankMedianEnhancer` for impulsive noise removal via
          median filtering.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of artifact removal on plate images.
    """

    def __init__(self, shape: str = "diamond", width: int = None):
        """
        Parameters:
            shape (str): Footprint geometry controlling which bright features are
                removed. 'diamond' or 'disk' provide isotropic behavior on plates;
                'square' can align with sensor grid artifacts. Advanced: 'sphere'
                or 'cube' for volumetric data.
            width (int | None): Maximum bright-object width (in pixels) targeted
                for removal. Set slightly smaller than the smallest colonies to
                avoid suppressing real colonies. None picks a small default based
                on image dimensions.
        """
        self.shape = shape
        self.width = width

    def _operate(self, image: Image) -> Image:
        white_tophat_results = white_tophat(
                image.detect_mat[:],
                footprint=self._get_footprint(
                        self._get_footprint_width(detection_matrix=image.detect_mat[:]),
                ),
        )
        image.detect_mat[:] = image.detect_mat[:] - white_tophat_results

        return image

    def _get_footprint_width(self, detection_matrix: np.ndarray) -> int:
        if self.width is None:
            return int(np.min(detection_matrix.shape) * 0.004)
        else:
            return self.width

    def _get_footprint(self, width: int) -> np.ndarray:
        match self.shape:
            # Use shared ImageEnhancer utility for common 2D shapes
            case "disk" | "square" | "diamond":
                return self._make_footprint(shape=self.shape, width=width)
            case _:
                raise ValueError(f"Unsupported footprint shape: {self.shape}")
