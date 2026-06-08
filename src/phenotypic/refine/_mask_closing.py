from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin
from phenotypic.tools_.typing_ import NdArrayField, TuneSpec

import numpy as np
from skimage.morphology import closing


class MaskClosing(ObjectRefiner, FootprintMixin):
    """Bridge narrow background gaps in colony masks using morphological closing.

    Applies binary closing (dilation followed by erosion) to the object mask,
    reconnecting nearby fragments of the same colony that are separated by thin
    background channels and filling small internal holes. Colony size is largely
    preserved because the dilation and erosion nearly cancel outside solid regions.

    For an overview of morphological refinement methods, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Colonies fragmented by uneven pigmentation or shadow-induced background
          channels across the colony body.
        - Masks with small internal holes caused by condensation droplets or
          specular glare on the agar surface.
        - Pre-measurement cleanup to reconnect fragments before area or shape
          features are computed.

    Consider Also:
        - :class:`MaskFill` for filling enclosed holes within objects without
          bridging separately labeled colonies.
        - :class:`MaskOpening` for the opposite effect — removing thin connections
          between objects that should remain distinct.
        - :class:`NearestNeighborMerger` for merging distant fragments that are
          too far apart for morphological closing to bridge.

    Args:
        shape: Structuring element shape for the closing footprint. ``"auto"``
            scales a disk to the image size; ``"disk"``, ``"square"``, and
            ``"diamond"`` use named shapes at the given ``width``; a NumPy
            array provides a custom element; ``None`` uses the skimage library
            default. Default: ``None``.
        width: Footprint width in pixels when using a named shape. Larger values
            bridge wider gaps but risk merging distinct adjacent colonies. Typical
            range: 3--9. Default: 5.
        n_iter: Number of closing iterations applied sequentially. Each additional
            iteration extends the effective reach by one footprint radius. Default: 1.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` morphologically closed.

    See Also:
        :doc:`/how_to/notebooks/merge_fragmented_detections` for fragment
        merging strategies including morphological closing.
    """

    shape: Literal["auto", "square", "diamond", "disk"] | NdArrayField | None = None
    width: Annotated[int, TuneSpec(3, 9)] = 5
    n_iter: Annotated[int, TuneSpec(1, 3)] = 1

    def _operate(self, image: Image) -> Image:
        if self.shape == "auto":
            footprint = FootprintMixin._make_footprint(
                    "disk", width=max(3, round(np.min(image.shape) * 0.005))
            )
        elif isinstance(self.shape, np.ndarray):
            footprint = self.shape
        elif self.shape in self._footprint_shapes:
            footprint = FootprintMixin._make_footprint(self.shape, width=self.width)
        elif not self.shape:
            footprint = None
        else:
            raise AttributeError("Invalid shape type")

        for _ in range(self.n_iter):
            image.objmask[:] = closing(image.objmask[:], footprint=footprint)
        return image
