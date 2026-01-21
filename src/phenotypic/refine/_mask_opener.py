from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_ import FootprintMixin

import numpy as np
from skimage.morphology import opening
from phenotypic.tools_.typing_ import FootprintShape


class MaskOpener(ObjectRefiner, FootprintMixin):
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
        - GrayOpening can remove thin filaments that are biologically meaningful in
          spreading/filamentous phenotypes.

    Attributes:
        shape (Literal["auto"] | np.ndarray | int | None): Structuring
            element used for opening. A larger or denser footprint removes more
            thin connections and specks but risks eroding colony boundaries.

    Examples:
        .. dropdown:: Morphologically open masks to separate touching colonies

            >>> from phenotypic.refine import MaskOpener
            >>> op = MaskOpener(shape='auto')
            >>> image = op.apply(image, inplace=True)  # doctest: +SKIP

    Raises:
        AttributeError: If an invalid ``footprint`` type is provided (checked
            during operation).
    """

    def __init__(
            self,
            shape: Literal["auto"] | FootprintShape | np.ndarray | None = None,
            width: int = 5
    ):
        """Initialize the opener.

        Args:
            shape (Literal["auto"] | np.ndarray | int | None): Structuring
                element for opening. Use:
                - "auto" to select a diamond shape scaled to image size
                  (larger plates → slightly larger width),
                - a NumPy array to pass a custom shape,
                - an ``int`` width to build a diamond shape of that size,
                - or ``None`` to use the library default.

                Larger widths disconnect wider bridges and suppress more
                speckles, but erode edges and can remove small colonies.
        """
        super().__init__()
        self.shape: Literal["auto"] | FootprintShape | np.ndarray | None = shape
        self.width = width

    def _operate(self, image: Image) -> Image:
        if self.shape == "auto":
            footprint = FootprintMixin._make_footprint(
                    "diamond", width=max(3, round(np.min(image.shape) * 0.005))
            )
        elif isinstance(self.shape, np.ndarray):
            footprint = self.shape
        elif self.shape in self._footprint_shapes:
            footprint = FootprintMixin._make_footprint(self.shape,
                                                       width=int(self.width))
        elif not self.shape:
            footprint = self.shape
        else:
            raise AttributeError("Invalid shape type")

        image.objmask[:] = opening(image.objmask[:], footprint=footprint)
        return image
