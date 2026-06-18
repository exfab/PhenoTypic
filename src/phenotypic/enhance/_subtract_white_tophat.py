from __future__ import annotations

from typing import Annotated, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.morphology import white_tophat

from phenotypic.abc_ import MorphologicalFiltering
from phenotypic.sdk_.typing_ import TuneSpec


class SubtractWhiteTophat(MorphologicalFiltering):
    """Suppress small bright artifacts in ``detect_mat`` by subtracting the white top-hat.

    Computes the white top-hat (original minus morphological opening) and
    subtracts it from the image, removing small bright blobs such as dust
    specks, glare highlights, and condensation artifacts while preserving
    larger colony structures intact.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Removing small bright artifacts that could be mistaken for tiny
          colonies during thresholding.
        - Reducing glare highlights on shiny agar plates before colony
          detection.
        - Cleaning up dust and condensation artifacts that confuse
          downstream segmentation.

    Consider Also:
        - :class:`WhiteTophatEnhance` when the goal is to isolate small
          bright structures rather than suppress them.
        - :class:`GrayOpening` for morphological smoothing that removes
          small bright features without explicit subtraction.
        - :class:`RankMedianEnhancer` for impulsive noise removal via
          median filtering when artifacts are single-pixel in scale.

    Args:
        shape: Footprint geometry for the structuring element. Accepted
            values: ``'diamond'`` (default) and ``'disk'`` provide
            isotropic behavior suited to round artifacts; ``'square'``
            can align with sensor grid patterns.
        width: Maximum bright-artifact size in pixels targeted for
            removal. Set smaller than the smallest genuine colony
            diameter to preserve colonies. ``None`` (default) auto-derives
            a small value as approximately 0.4 % of the shorter image
            dimension.

    Returns:
        Image: Input image with ``detect_mat`` artifact-suppressed by
        subtracting the white top-hat. ``rgb`` and ``gray`` are
        unchanged.

    Raises:
        ValueError: If an unsupported footprint shape is provided.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of artifact removal on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        top-hat transforms and artifact suppression strategies.
    """

    shape: str = "diamond"
    width: Annotated[int | None, TuneSpec(3, 15, step=2)] = None

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
