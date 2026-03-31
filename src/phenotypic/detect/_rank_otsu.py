from __future__ import annotations

import numpy as np
import skimage.filters.rank as rank
from skimage.util import img_as_ubyte
from phenotypic.abc_ import ObjectDetector
from phenotypic.tools_ import FootprintMixin
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class RankOtsuDetector(ObjectDetector, FootprintMixin):
    """Detect colonies by adaptive local Otsu thresholding within a sliding footprint.

    Compute an Otsu threshold independently for every pixel using a local
    spatial neighbourhood, producing a per-pixel adaptive threshold map.
    This compensates for vignetting, lighting gradients, and spatially
    varying agar colour that cause a single global threshold to over- or
    under-segment parts of the plate. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        * Plates with vignetting, hot-spots, or centre-to-edge illumination
          gradients.
        * Large-format plates (384-well or larger) where lighting uniformity
          is difficult to achieve.
        * Images with spatially varying agar colour or reflectance.

    Consider Also:
        * :class:`OtsuDetector` when illumination is uniform and a fast
          global threshold suffices.
        * :class:`HysteresisDetector` when colony brightness varies but
          spatial illumination is reasonably even.
        * :class:`ChanVeseDetector` when colonies have diffuse edges and
          region-based segmentation is more appropriate.

    Args:
        shape: Footprint shape for the local neighbourhood. Accepted
            values: ``'square'``, ``'diamond'``, ``'disk'``. Disk and
            diamond are rotationally symmetric; square is faster. Default:
            ``'square'``.

        width: Footprint width (or radius for disk/diamond) in pixels.
            If ``None``, auto-scales to ``min(height, width) // 8``.
            Larger values smooth the threshold spatially (less local
            adaptation); smaller values track finer illumination changes
            but may over-segment. Default: None.

        ignore_zeros: Exclude zero-intensity pixels from the local
            threshold computation. Enable for plates with black borders
            or masked regions. Default: False.

    Returns:
        Image: Input image with ``objmask`` set to binary mask and
        ``objmap`` set to labeled connected components.

    Raises:
        ValueError: If ``shape`` is not one of the accepted values or
            ``width`` is not positive.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    def __init__(
            self,
            shape: Literal["square", "diamond", "disk"] = "square",
            width: int | None = None,
            ignore_zeros: bool = False,
    ):
        self.shape = shape
        self.width = width
        self.ignore_zeros = ignore_zeros

    def _operate(self, image: Image) -> Image:
        detect_mat = img_as_ubyte(image.detect_mat[:])
        if self.ignore_zeros:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[detect_mat.nonzero()] = 1
            mask = mask > 0
        else:
            mask = None

        width = min(image.shape[:2]) // 8 if self.width is None else self.width

        image.objmask[:] = detect_mat >= rank.otsu(
                image=detect_mat,
                footprint=self._make_footprint(
                        shape=self.shape,
                        width=width,
                ),
                mask=mask,
        )
        return image
