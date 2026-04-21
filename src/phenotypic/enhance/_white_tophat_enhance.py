from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.morphology import white_tophat

from phenotypic.abc_ import ImageEnhancer


class WhiteTophatEnhance(ImageEnhancer):
    """Isolate small bright structures in ``detect_mat`` with the white top-hat transform.

    Computes the white top-hat (original minus morphological opening) and
    retains the result, extracting bright features smaller than the
    structuring element while suppressing larger background structures.
    Highlights small bright colonies, inocula, or specks against uneven
    illumination.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        shape: Footprint geometry. ``'disk'`` (default) preserves rounded
            colony shapes; ``'diamond'`` is computationally efficient;
            ``'square'`` can align with sensor grid artifacts.
        width: Maximum bright-object size (pixels) targeted for extraction.
            Set slightly larger than the maximum size of colonies you want
            to isolate. ``None`` (default) derives a small value from
            image dimensions.

    Returns:
        Image: Input image with ``detect_mat`` containing only the
        extracted small bright structures. ``rgb`` and ``gray`` are
        unchanged.

    Raises:
        ValueError: If an unsupported footprint shape is provided.

    Best For:
        - Isolating small bright colonies from larger background structures.
        - Highlighting faint small colonies against uneven illumination.
        - Extracting tiny bright specks for detection or quantification.
        - Preprocessing before detecting small colony phenotypes.

    Consider Also:
        - :class:`SubtractWhiteTophat` when you want to suppress (not
          isolate) small bright artifacts.
        - :class:`OpeningSubtractBg` for OpenCV-accelerated white top-hat
          background subtraction.
        - :class:`MultiscaleLoGEnhancer` for scale-invariant blob detection
          that responds to both small and large colonies.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of morphological enhancement on plate images.
    """

    def __init__(self, shape: str = "disk", width: int = None):
        super().__init__()
        self.shape = shape
        self.width = width

    def _operate(self, image: Image) -> Image:
        white_tophat_results = white_tophat(
            image.detect_mat[:],
            footprint=self._get_footprint(
                self._get_footprint_width(detection_matrix=image.detect_mat[:]),
            ),
        )
        image.detect_mat[:] = white_tophat_results

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
