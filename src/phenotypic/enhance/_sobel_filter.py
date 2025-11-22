from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from skimage.filters import sobel

from phenotypic.abc_ import ImageEnhancer


class SobelFilter(ImageEnhancer):
    """
    Sobel edge filter to highlight colony boundaries.

    Computes the gradient magnitude using the Sobel operator to emphasize edges
    (rapid intensity changes). On agar plate images, this highlights colony
    perimeters and helps downstream steps that rely on edge strength (e.g.,
    contour finding, watershed seeds, or boundary-based scoring).

    Use cases (agar plates):
    - Emphasize colony outlines before contour detection or watershed.
    - Separate touching colonies when combined with a distance or marker-based
      segmentation strategy.

    Guidance:
    - Consider light smoothing with `GaussianBlur` beforehand to suppress noise;
      Sobel is sensitive to high-frequency artifacts (dust, texture).

    Caveats:
    - Output is an edge map, not a background-corrected image. Use in tandem
      with background removal or thresholding for full segmentation.
    - Strong illumination gradients can still produce spurious edges; correct
      background first if needed.
    """

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = sobel(image=image.enh_gray[:])
        return image
