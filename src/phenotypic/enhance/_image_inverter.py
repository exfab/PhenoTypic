from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np

from ..abc_ import ContrastAdjustment


class ImageInverter(ContrastAdjustment):
    """Invert ``detect_mat`` pixel intensities so dark colonies become bright.

    Reverses the brightness scale so dark regions become bright and vice versa.
    For uint8 data the inversion is ``255 - pixel``; for floating-point data
    it is ``max_value - pixel``. Corrects for imaging systems or scan settings
    that produce colonies as dark regions on a bright background, restoring the
    bright-colony-on-dark-background convention expected by downstream detectors.

    For how polarity correction fits into the pipeline, see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Correcting inverted scan output from imaging systems that produce
          dark-on-bright colony images.
        - Preprocessing before detectors that assume bright colonies on dark
          agar backgrounds.
        - Plates where colony boundaries are defined by dark absorption zones
          on a bright background.

    Consider Also:
        - :class:`SetDetectMode` when switching the detection channel (e.g.,
          to red or green) would resolve the contrast issue without polarity
          inversion.
        - :class:`SharpenEdgeGauss` when the issue is low contrast rather than
          inverted polarity.

    Returns:
        Image: Input image with ``detect_mat`` intensity-inverted.
        ``rgb`` and ``gray`` are unchanged.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a visual
        walkthrough of enhancement pipelines on plate images.
        :doc:`/explanation/what_enhancement_does` for how the detect_mat
        polarity convention is established.
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
