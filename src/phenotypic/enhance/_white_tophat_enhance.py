from __future__ import annotations

from typing import Annotated, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.morphology import white_tophat

from phenotypic.abc_ import MorphologicalFiltering
from phenotypic.sdk_.typing_ import TuneSpec


class WhiteTophatEnhance(MorphologicalFiltering):
    """Isolate small bright structures in ``detect_mat`` with the white top-hat transform.

    Computes the white top-hat (original minus morphological opening) and
    retains the result, extracting bright features smaller than the
    structuring element while suppressing larger background structures.
    Highlights small bright colonies, inocula, or specks against uneven
    illumination.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Isolating small bright colonies from larger background structures
          or broad illumination gradients.
        - Highlighting faint small colonies against uneven agar
          illumination.
        - Extracting early-stage inocula or pinpoints for quantification.
        - Preprocessing before applying a detector tuned for small colony
          phenotypes.

    Consider Also:
        - :class:`SubtractWhiteTophat` when the goal is to suppress small
          bright artifacts rather than isolate them.
        - :class:`SubtractOpening` for OpenCV-accelerated white top-hat
          background subtraction that also preserves the surrounding
          image context.
        - :class:`FocusBlobLoG` for scale-invariant blob detection that
          responds across a range of colony sizes.

    Args:
        shape: Footprint geometry for the structuring element. Accepted
            values: ``'disk'`` (default) for isotropic extraction that
            preserves rounded colony shapes; ``'diamond'`` for
            computational efficiency; ``'square'`` to align with sensor
            grid artifacts.
        width: Maximum bright-object size in pixels targeted for
            extraction. Set slightly larger than the maximum colony
            diameter you want to isolate. ``None`` (default) auto-derives
            a small value as approximately 0.4 % of the shorter image
            dimension.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the white
        top-hat result, containing only the extracted small bright
        structures. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If an unsupported footprint shape is provided.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of morphological enhancement on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        top-hat transforms and feature isolation strategies.
    """

    shape: str = "disk"
    width: Annotated[int | None, TuneSpec(3, 15, step=2)] = None

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
