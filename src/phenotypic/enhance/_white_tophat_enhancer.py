from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from skimage.morphology import white_tophat, cube, ball

from phenotypic.abc_ import ImageEnhancer


class WhiteTophatEnhancer(ImageEnhancer):
    """
    White top-hat transform to suppress small bright structures.

    Computes the white top-hat (original minus opening) using a structuring
    element, then subtracts it from the image here (i.e., remove small bright
    blobs). In agar plate colony images, this helps reduce bright specks from
    dust, glare, or condensation, making colonies stand out against a smoother
    background.

    Use cases (agar plates):
    - Remove small bright artifacts that can be mistaken for tiny colonies.
    - Reduce glare highlights on shiny plates before thresholding.

    Tuning and effects:
    - shape: The morphology footprint geometry. 'diamond' or 'disk' are good
      isotropic choices on plates; 'square' can align with pixel grids.
    - radius: Sets the maximum size of bright features to remove. Choose slightly
      smaller than the minimum colony radius so real colonies are preserved.

    Caveats:
    - If radius is too large, real small colonies will be attenuated.
    - Operates on bright features; for dark colonies on bright agar, it primarily
      removes bright noise rather than enhancing the colonies themselves.

    Attributes:
        shape (str): Footprint shape: 'diamond', 'disk', 'square', 'sphere', 'cube'.
        radius (int | None): Footprint radius in pixels; if None, a small default
            is derived from the image size.
    """
    def __init__(self, shape: str = 'diamond', radius: int = None):
        """
        Parameters:
            shape (str): Footprint geometry controlling which bright features are
                removed. 'diamond' or 'disk' provide isotropic behavior on plates;
                'square' can align with sensor grid artifacts. Advanced: 'sphere'
                or 'cube' for volumetric data.
            radius (int | None): Maximum bright-object size (in pixels) targeted
                for removal. Set slightly smaller than the smallest colonies to
                avoid suppressing real colonies. None picks a small default based
                on image dimensions.
        """
        self.shape = shape
        self.radius = radius

    def _operate(self, image: Image) -> Image:
        white_tophat_results = white_tophat(
                image.enh_gray[:],
                footprint=self._get_footprint(
                        self._get_footprint_radius(detection_matrix=image.enh_gray[:]),
                ),
        )
        image.enh_gray[:] = image.enh_gray[:] - white_tophat_results

        return image

    def _get_footprint_radius(self, detection_matrix: np.ndarray) -> int:
        if self.radius is None:
            return int(np.min(detection_matrix.shape)*0.004)
        else:
            return self.radius

    def _get_footprint(self, radius: int) -> np.ndarray:
        match self.shape:
            # Use shared ImageEnhancer utility for common 2D shapes
            case 'disk' | 'square' | 'diamond':
                return self._make_footprint(shape=self.shape, radius=radius)
            # Preserve volumetric alternatives
            case 'sphere':
                return ball(radius)
            case 'cube':
                return cube(radius*2)
