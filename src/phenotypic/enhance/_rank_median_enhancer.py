from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image
import numpy as np
from skimage.filters.rank import median
from skimage.morphology import cube, ball
from skimage.util import img_as_ubyte, img_as_float

from phenotypic.abc_ import ImageEnhancer


class RankMedianEnhancer(ImageEnhancer):
    """
    Rank-based median filtering with configurable footprint.

    Applies a local median using rank filters with a user-defined footprint
    shape and width. For agar plate colony images, this enables targeted
    suppression of impulsive noise while tailoring the spatial scale to colony
    size and shape, offering more control than a default median.

    Use cases (agar plates):
    - Denoise while preserving colony boundaries by matching the footprint width
      to be smaller than colony diameters.
    - Use anisotropic or non-circular footprints (e.g., squares) for grid-like
      artifacts from imaging hardware.

    Tuning and effects:
    - shape: Choose 'disk' for circular isotropic smoothing on plates;
      'square' or 'cube' can align with grid artifacts; 'ball' for 3D stacks.
    - width: Controls neighborhood size. Larger radii remove more
      noise but can erode small colonies and close tight gaps.
    - shift_x/shift_y: Offset the footprint center to bias the neighborhood
      if imaging introduces directional streaks; typically left at 0.

    Caveats:
    - Very large footprints may over-smooth and merge nearby colonies.
    - Rank filters operate on uint8 here; intensity scaling occurs internally.
      Ensure consistency if comparing raw intensities elsewhere.

    Attributes:
        shape (str): 'disk', 'square', 'sphere', or 'cube' defining
            the footprint geometry.
        width (int | None): Width (pixels). If None, a small default
            derived from image size is used.
        shift_x (int): Horizontal footprint offset for advanced use.
        shift_y (int): Vertical footprint offset for advanced use.
    """

    def __init__(self, shape: str = "square", width: int = None, shift_x=0, shift_y=0):
        """
        Parameters:
            shape (str): Geometry of the neighborhood. Use 'disk' for
                isotropic smoothing on plates; 'square' to align with grid noise;
                'sphere'/'cube' for 3D contexts. Default 'square'.
            width (int | None): Neighborhood width in pixels. Set
                smaller than the minimum colony width to preserve colony edges;
                None chooses a small default based on image size.
            shift_x (int): Horizontal offset of the footprint center to bias the
                neighborhood if artifacts are directional. Typically 0.
            shift_y (int): Vertical offset of the footprint center. Typically 0.
        """
        if shape not in ["disk", "square", "sphere", "cube"]:
            raise ValueError(f"footprint shape {shape} is not supported")

        self.shape = shape
        self.width = width
        self.shift_x = shift_x
        self.shift_y = shift_y

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = img_as_float(
                median(
                        image=img_as_ubyte(image.enh_gray[:]),
                        footprint=self._get_footprint(
                                self._get_footprint_width(image.enh_gray[:])
                        ),
                )
        )
        return image

    def _get_footprint_width(self, enh_gray: np.ndarray) -> int:
        if self.width is None:
            return int(np.min(enh_gray.shape)*0.002)
        else:
            return self.width

    def _get_footprint(self, width: int) -> np.ndarray:
        match self.shape:
            # Use the central ImageEnhancer utility for common 2D shapes
            case "disk" | "square":
                return self._make_footprint(shape=self.shape, width=width)
            # Preserve alternative 3D options as originally implemented
            case "ball":
                return ball(radius)
            case "cube":
                return cube(int(radius*2))
            case _:
                raise TypeError("Unknown footprint shape")
