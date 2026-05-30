from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np

from ..abc_ import ImageEnhancer


class ImageInverter(ImageEnhancer):
    """Invert ``detect_mat`` pixel intensities (negate brightness).

    Reverses the brightness scale so dark regions become bright and vice
    versa. For uint8 data the inversion is ``255 - pixel``; for
    floating-point data it is ``max_value - pixel``. Use this when
    detectors expect colonies as bright regions but the source image has
    colonies as dark regions.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Returns:
        Image: Input image with ``detect_mat`` intensity-inverted.
        ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Correcting inverted scan output from imaging systems that
          produce dark-on-bright colony images.
        - Preprocessing before detectors that expect bright colonies on
          dark backgrounds.
        - Plates where colony boundaries are defined by dark edges on a
          bright background.

    Consider Also:
        - :class:`SetDetectMode` when switching the detection channel
          (e.g., to red or green) would resolve the contrast issue.
        - :class:`SharpenEdgeGauss` when the issue is low contrast rather
          than inverted polarity.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of enhancement pipelines on plate images.
    """

    def _operate(self, image: Image) -> Image:
        enh = image.detect_mat[:]

        # Invert based on data type
        if enh.dtype == np.uint8:
            # For 8-bit integer: 255 - value
            inverted = 255 - enh
        else:
            # For floating-point: find max value and invert
            max_val = enh.max()
            inverted = max_val - enh

        image.detect_mat[:] = inverted
        return image
