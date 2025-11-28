from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from phenotypic.abc_ import ObjectRefiner

import numpy as np
from skimage.morphology import binary_opening


class MaskOpener(ObjectRefiner):
    """Morphologically open binary masks to remove thin connections and specks.

    Intuition:
        Binary opening (erosion followed by dilation) removes small isolated
        pixels and breaks narrow bridges between objects. On agar plates, this
        helps separate touching colonies and suppresses tiny artifacts from
        dust or condensation without overly shrinking well-formed colonies.

    Why this is useful for agar plates:
        Colonies may develop halos or be linked by faint film on the agar. A
        gentle opening step can restore separated masks, improving count and
        phenotype accuracy.

    Use cases:
        - After thresholding, to split colonies connected by 1–2-pixel bridges.
        - To remove tiny noise specks before measuring morphology.

    Caveats:
        - Too large a footprint erodes small colonies or weakly-stained edges,
          lowering recall and edge sharpness.
        - Opening can remove thin filaments that are biologically meaningful in
          spreading/filamentous phenotypes.

    Attributes:
        footprint (Literal["auto"] | np.ndarray | int | None): Structuring
            element used for opening. A larger or denser footprint removes more
            thin connections and specks but risks eroding colony boundaries.

    Examples:
        >>> from phenotypic.refine import MaskOpener
        >>> op = MaskOpener(footprint='auto')
        >>> image = op.apply(image, inplace=True)  # doctest: +SKIP

    Raises:
        AttributeError: If an invalid ``footprint`` type is provided (checked
            during operation).
    """

    def __init__(self, footprint: Literal["auto"] | np.ndarray | int | None = None):
        """Initialize the opener.

        Args:
            footprint (Literal["auto"] | np.ndarray | int | None): Structuring
                element for opening. Use:
                - "auto" to select a diamond footprint scaled to image size
                  (larger plates → slightly larger radius),
                - a NumPy array to pass a custom footprint,
                - an ``int`` radius to build a diamond footprint of that size,
                - or ``None`` to use the library default.

                Larger radii disconnect wider bridges and suppress more
                speckles, but erode edges and can remove small colonies.
        """
        super().__init__()
        self.footprint: Literal["auto"] | np.ndarray | int | None = footprint

    def _operate(self, image: Image) -> Image:
        if self.footprint == 'auto':
            footprint = self._make_footprint("diamond", radius=max(3, round(np.min(image.shape)*0.005)))
        elif isinstance(self.footprint, np.ndarray):
            footprint = self.footprint
        elif isinstance(self.footprint, (int, float)):
            footprint = self._make_footprint("diamond", radius=int(self.footprint))
        elif not self.footprint:
            footprint = self.footprint
        else:
            raise AttributeError('Invalid footprint type')

        image.objmask[:] = binary_opening(image.objmask[:], footprint=footprint)
        return image
