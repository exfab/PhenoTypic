from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image
from phenotypic.abc_ import ImageEnhancer

from skimage.filters import median


class MedianFilter(ImageEnhancer):
    """
    Median filtering to reduce impulsive noise while preserving edges.

    The median filter replaces each pixel with the median of its local neighborhood
    and is robust to outliers. For agar plate colony images, this is effective at
    removing speckle from condensation droplets, dust, or sensor noise without
    blurring colony edges as much as Gaussian smoothing would.

    Use cases (agar plates):
    - Reduce “salt-and-pepper” artifacts and tiny bright/dark specks prior to
      thresholding or edge detection.
    - Preserve colony boundaries better than linear blur when colonies are small
      or closely packed.

    Tuning and effects:
    - Footprint: This implementation uses the library default footprint when none
      is provided (a small neighborhood). For stronger denoising, prefer
      `RankMedianEnhancer` where you can set shape and radius explicitly.
    - mode/cval: Control how borders are handled. 'reflect' or 'nearest' avoids
      artificial artifacts at the plate boundary; 'constant' uses `cval` as fill.

    Caveats:
    - Using a very large neighborhood (when configured via alternative median
      functions) can remove small colonies or close thin gaps.
    - Median filtering can flatten fine texture within pigmented colonies; use a
      light application or a rank filter with an appropriate footprint.

    Attributes:
        mode (str): Boundary handling mode: 'nearest', 'reflect', 'constant',
            'mirror', or 'wrap'.
        cval (float): Constant fill when `mode='constant'`.
    """

    def __init__(self,
                 mode: Literal["nearest", "reflect", "constant", "mirror", "wrap"] = 'nearest',
                 shape: Literal["disk", "square", "diamond"] | None = None,
                 radius: int = 5,
                 cval: float = 0.0):
        """
        This class is designed to facilitate image processing tasks, particularly for analyzing microbe
        colonies on solid media agar. By adjusting the mode, footprint, radius, and cval attributes,
        users can modify the processing behavior and results to suit their specific requirements for
        studying spatial arrangements, colony boundaries, and other morphological features.

        Attributes:
            mode (Literal["nearest", "reflect", "constant", "mirror", "wrap"]):
                Determines how boundaries of the image are handled during processing.
                For instance, "reflect" can help minimize edge artifacts when analyzing
                colonies near the edge of the image by mirroring boundary pixels, while
                "constant" fills with a value (cval), which might highlight isolated colonies.
                Adjusting this can significantly affect how edge regions are interpreted.

            shape (Literal["disk", "square", "diamond"] | None):
                Specifies the shape of the structuring element used in morphological
                operations. For instance, "disk" simulates circular neighborhood which works
                well for circular colonies, whereas "square" gives a grid-like neighborhood.
                This can directly impact how structures are identified or segmented.

            radius (int):
                Size of the structuring element. Larger radii result in broader neighborhoods
                being considered, which may smooth or connect distant colonies, while smaller
                radii preserve finer details but may miss larger structural relationships. Only
                if shape is not None.

            cval (float):
                Value used to fill borders when mode is set to "constant". This directly affects
                colony recognition at the edges; for example, setting a high cval compared to
                colony intensity might obscure colonies near the borders.
        """
        if mode in ['nearest', 'reflect', 'constant', 'mirror', 'wrap']:
            self.mode = mode
            self.shape = shape
            self.radius = radius
            self.cval = cval
        else:
            raise ValueError('mode must be one of "nearest","reflect","constant","mirror","wrap"')

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = median(
                image=image.enh_gray[:],
                behavior='ndimage',
                footprint=(self.shape
                           if self.shape is None
                           else self._make_footprint(shape=self.shape, radius=self.radius)),
                mode=self.mode,
                cval=self.cval)
        return image
