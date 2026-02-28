from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.abc_ import ImageEnhancer
from phenotypic.tools_.mixin import FootprintMixin


class OpeningSubtractBg(ImageEnhancer, FootprintMixin):
    """OpenCV-accelerated background subtraction via morphological opening.

    Computes the white top-hat transform (original minus morphological opening)
    using OpenCV's ``cv2.morphologyEx(MORPH_TOPHAT)``, which leverages C++/SIMD
    for significantly better throughput than scikit-image equivalents. The result
    isolates bright foreground structures smaller than the structuring element
    while removing slow-varying background intensity.

    Args:
        shape (Literal["square", "diamond", "disk"]): Structuring element
            geometry. ``"disk"`` gives isotropic removal suited to round
            colonies; ``"square"`` is fastest but may introduce directional
            artifacts; ``"diamond"`` is a compromise.
        width (int): Diameter of the structuring element in pixels. Must be
            larger than colony diameter to avoid subtracting colony signal.
            Typical agar-plate values: 31-101 depending on resolution.

    Returns:
        Image: Modified image with ``detect_mat`` containing only foreground
        structures smaller than the structuring element.

    Raises:
        ValueError: If an unsupported footprint shape is provided.

    Use cases (agar plates):
        - Fast background subtraction for high-throughput plate screening.
        - Remove uneven illumination gradients and agar shading before
          colony detection.
        - Drop-in performance upgrade over ``SubtractRollingBall`` when a
          flat structuring element (rather than parabolic ball) is acceptable.
        - Pre-processing step in pipelines where speed matters (large batches,
          parameter sweeps).

    Limitations:
        - Flat SE approximation: unlike rolling-ball, the opening uses a flat
          structuring element, so very gradual intensity ramps may leave
          residual background.
        - ``width`` must exceed the largest colony diameter; too-small values
          erode colony signal.
        - Very large ``width`` values increase memory use and may slow down
          even the OpenCV backend.

    Parameter effects:
        - **shape:** ``"disk"`` preserves round colony morphology best and
          avoids directional artifacts. ``"square"`` is marginally faster.
          ``"diamond"`` is intermediate.
        - **width:** Controls the scale of background removal. Increase to
          preserve larger colonies; decrease to remove finer background
          texture. A good starting point is 2-3x the largest colony diameter.

    Examples:
        Basic background subtraction:

        >>> from phenotypic.enhance import OpeningSubtractBg
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> enhancer = OpeningSubtractBg(shape='disk', width=51)
        >>> result = enhancer.apply(image)
        >>> result.detect_mat[:].min() >= 0.0
        True

        In a detection pipeline:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import OpeningSubtractBg
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> pipeline = ImagePipeline([
        ...     OpeningSubtractBg(shape='disk', width=51),
        ...     OtsuDetector(),
        ... ])
        >>> result = pipeline.apply(image)
    """

    def __init__(
            self,
            shape: Literal["square", "diamond", "disk"] = "disk",
            width: int = 51,
    ):
        self.shape = shape
        self.width = width

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = cv2.morphologyEx(
                src=image.detect_mat[:],
                op=cv2.MORPH_TOPHAT,
                kernel=self._make_footprint(shape=self.shape, width=self.width),
        )
        return image
