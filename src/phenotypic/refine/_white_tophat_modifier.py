from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from skimage.morphology import disk, square, white_tophat

from phenotypic.abc_ import ObjectRefiner


class WhiteTophatModifier(ObjectRefiner):
    """Suppress small bright structures in the mask using white tophat.

    Intuition:
        White tophat highlights small, bright features relative to their local
        background. On agar plates, glare, dust, or bright halos can create
        thin connections or speckles that pollute colony masks. This modifier
        detects those bright micro-structures and subtracts them from the
        binary mask to improve separation and mask quality.

    Why this is useful for agar plates:
        Bright artifacts can bridge adjacent colonies or inflate perimeters.
        Removing those tiny bright elements yields cleaner, more compact masks
        that better match colony boundaries under uneven illumination.

    Use cases:
        - Reducing glare-induced bridges between neighboring colonies.
        - Removing bright speckles/dust that become embedded in masks after
          thresholding.

    Caveats:
        - Large footprints may remove real bright edges of colonies (e.g.,
          highly reflective rims), slightly eroding edge sharpness.
        - If the footprint is too small, bright artifacts may remain.

    Attributes:
        footprint_shape (str): Shape for the footprint used in the tophat
            transform. Supported: 'disk', 'square'. Disk tends to preserve
            round features, while square can be more aggressive along axes.
        footprint_radius (int | None): Radius of the footprint. Larger values
            remove broader bright features but risk shrinking thin colony
            appendages. ``None`` auto-scales with image size.

    Examples:
        >>> from phenotypic.refine import WhiteTophatModifier
        >>> op = WhiteTophatModifier(shape='disk', radius=5)
        >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
    """

    def __init__(self, footprint_shape='disk', footprint_radius: int = None):
        """Initialize the modifier.

        Args:
            footprint_shape (str): Footprint geometry for white tophat.
                - 'disk': Balanced in all directions; gentle on round colonies.
                - 'square': Slightly stronger along rows/columns; may remove
                  more rectilinear glare or sensor artifacts.
            footprint_radius (int | None): Radius in pixels. Increasing removes
                larger bright structures and can improve background suppression,
                but may thin colony edges. ``None`` auto-selects ~0.4% of the
                smaller image dimension.

        Raises:
            ValueError: If ``shape`` is not one of the supported
                values (raised during operation).
        """
        self.footprint_shape = footprint_shape
        self.footprint_radius = footprint_radius

    def _operate(self, image: Image) -> Image:
        white_tophat_results = white_tophat(
                image.objmask[:],
                footprint=self._get_footprint(
                        self._get_footprint_radius(array=image.objmask[:])
                )
        )
        image.objmask[:] = image.objmask[:] & ~white_tophat_results
        return image

    def _get_footprint_radius(self, array: np.ndarray) -> int:
        if self.footprint_radius is None:
            return int(np.min(array.shape)*0.004)
        else:
            return self.footprint_radius

    def _get_footprint(self, radius: int) -> np.ndarray:
        match self.footprint_shape:
            case 'disk':
                return disk(radius=radius)
            case 'square':
                return square(radius*2)
            case _:
                raise ValueError('invalid footprint shape. White tophat transform only supports two dimensional shapes')
