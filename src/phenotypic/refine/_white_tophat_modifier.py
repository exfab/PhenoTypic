from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from skimage.morphology import white_tophat

from phenotypic.abc_ import ObjectRefiner


class WhiteTophat(ObjectRefiner):
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
        footprint (str): Shape for the footprint used in the tophat
            transform. Supported: 'disk', 'square'. Disk tends to preserve
            round features, while square can be more aggressive along axes.
        width (int | None): Width of the footprint. Larger values
            remove broader bright features but risk shrinking thin colony
            appendages. ``None`` auto-scales with image size.

    Examples:
        .. dropdown:: Suppress small bright structures in the mask using white tophat

            >>> from phenotypic.refine import WhiteTophat
            >>> op = WhiteTophat(footprint='disk', width=5)
            >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
    """

    def __init__(self,
                 footprint: Literal["disk", "square", "diamond"] | np.ndarray = "disk",
                 width: int | None = None):
        """
        Represents a structural element used to analyze and process images, specifically useful for microbial
        colony analysis on solid media agar.

        The class encapsulates the shape and size of the structural element. Structural elements are commonly
        used in morphological image processing tasks such as dilations, erosions, opening, and closing. These
        operations can enhance or isolate features of microbe colonies on agar plates, such as determining
        colony size, spacing, or detecting connections between colonies.

        Attributes:
            shape (Literal["disk", "square", "diamond"] | np.ndarray):
                Defines the shape of the structural element. Choosing "disk" may help preserve the rounded
                geometry of typical microbial colonies. "Square" and "diamond" shapes may be more useful for
                colonies that form irregular or grid-based patterns. Supplying a custom numpy array (np.ndarray)
                allows for complete customization of the structural element, which could be beneficial for non-
                standard colony morphologies.

            width (int | None):
                Specifies the size of the structural element by defining the width. Larger widths will create
                structural elements that can encompass larger colonies or areas of colonies, potentially aiding
                in operations designed to merge close colonies. Smaller widths will result in more localized
                structural elements, which can preserve fine details and delineate smaller colonies. A None
                value assumes a default or minimal size.
        """
        self.footprint = footprint
        self.width = width

    def _operate(self, image: Image) -> Image:
        white_tophat_results = white_tophat(
                image.objmask[:],
                footprint=self._make_footprint(
                        shape=self.footprint,
                        width=self._get_footprint_width(array=image.objmask[:]),
                ),
        )
        image.objmask[:] = image.objmask[:] & ~white_tophat_results
        return image

    def _get_footprint_width(self, array: np.ndarray) -> int:
        if self.width is None:
            return int(np.min(array.shape)*0.004)
        else:
            return self.width
