from __future__ import annotations

import numpy as np
import skimage.filters.rank as rank
from skimage.util import img_as_ubyte, img_as_uint
from phenotypic.abc_ import ObjectDetector
from phenotypic.tools_ import FootprintMixin
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image


class RankOtsuDetector(ObjectDetector, FootprintMixin):
    """Local/rank-based Otsu detector for spatially varying illumination.

    RankOtsuDetector applies Otsu's threshold method locally within a spatial
    footprint, enabling per-pixel adaptive thresholding. Unlike global Otsu,
    this adapts to spatial intensity variations (vignetting, gradients, uneven
    lighting), detecting colonies across the entire plate even when illumination
    is non-uniform or background intensity varies spatially.

    Args:
        shape: Footprint shape for local neighborhood ('square', 'diamond', or
            'disk', default 'square'). Controls which pixels influence local
            threshold computation.

        width: Footprint width/radius in pixels (default None). If None, auto-scales
            to min(image_height, image_width) // 8. Larger footprints average
            threshold over larger regions (more smoothing, less local adaptation).
            Smaller footprints enable finer spatial adaptation but may oversegment.

        ignore_zeros: If True, exclude zero-intensity pixels from local threshold
            computation. Default False. Set True if images have black borders/masks.

    Attributes:
        shape, width, ignore_zeros

    Returns:
        Image: Input image with objmask set to binary mask from local Otsu thresholding.

    Raises:
        ValueError: If invalid footprint shape or width specified.

    **Use cases**

    - **Uneven plate illumination:** Vignetting, gradient lighting, or hotspots
      where single Otsu threshold fails to detect colonies uniformly.
    - **Varying background intensity:** Agar color variations, dust layers, or
      inconsistent substrate reflectance across the plate.
    - **Large plates with illumination falloff:** 384-well or larger plates with
      center-to-edge lighting gradients common in high-throughput setups.

    **Limitations**

    - Computationally expensive. Local Otsu for every pixel requires more operations
      than global Otsu. Slower on large images.
    - Requires footprint size tuning. Too large = insufficient adaptation to local
      variations; too small = oversegmentation, noisy thresholds.
    - Edge artifacts. Pixels near image borders have incomplete neighborhoods,
      causing unreliable local thresholds. Consider borders pre-masking.
    - Parameter dependent. Footprint shape and width significantly affect results.
      Different plate sizes may require different widths.

    **Parameter effects on colony detection**

    - **shape:** Determines neighborhood geometry. Disk/diamond more circular; square
      is faster. Choice affects which neighboring pixels influence local threshold.
    - **width:** Larger widths smooth spatial variations but reduce local adaptation.
      Auto-default (image_size/8) works well for most plates. Tune if detection
      quality varies spatially.
    - **ignore_zeros:** Enable if black borders create spurious local thresholds.

    Examples:
        Basic local Otsu detection for uneven illumination::

            from phenotypic import Image
            from phenotypic.detect import RankOtsuDetector

            plate = Image.imread("unevenly_lit_plate.jpg")
            detector = RankOtsuDetector(shape='disk', width=20)
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Detected {mask.sum()} colony pixels with local adaptation")

        Pipeline with local Otsu for high-throughput imaging::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur, CLAHE
            from phenotypic.detect import RankOtsuDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                CLAHE(clip_limit=2.0),
                RankOtsuDetector(shape='disk', width=30)
            ])

            image = Image.imread("plate_with_vignetting.jpg")
            result = pipeline.apply(image)
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
