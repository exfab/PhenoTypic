from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin

import numpy as np
from skimage.morphology import opening
from phenotypic.tools_.typing_ import NdArrayField


class MaskOpening(ObjectRefiner, FootprintMixin):
    """Smooth mask boundaries and separate touching colonies using morphological opening.

    Applies binary erosion followed by dilation with the chosen structuring
    element, removing isolated noise pixels and narrow bridges between adjacent
    colony masks. Colony bodies that survive the erosion step are restored by
    dilation, so well-formed masks lose little area while thin connections are
    severed.

    For an overview of morphological refinement strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Colonies joined by a 1--3 pixel agar bridge after global thresholding.
        - Detection masks with isolated noise specks smaller than the
          structuring element.
        - Plates where jagged or pixelated colony boundaries would distort
          perimeter or shape measurements.
        - Post-detection cleanup before :class:`RemoveNonCircular` or area
          measurement to reduce false negatives from ragged edges.

    Consider Also:
        - :class:`MaskClosing` when the goal is filling small holes or
          bridging gaps within a colony mask rather than severing connections.
        - :class:`SmallObjectRemover` when small artefacts are fully
          disconnected objects best removed by area threshold.
        - :class:`SeparateObjects` when touching colonies share a large
          overlapping region and watershed-based separation is needed.

    Args:
        shape: Structuring element geometry. ``"auto"`` chooses a diamond
            sized to 0.5% of the smaller image dimension. ``"diamond"``,
            ``"square"``, or ``"disk"`` use the ``width`` parameter to set
            the element size. A NumPy array supplies a custom footprint.
            ``None`` passes no footprint to skimage, which uses a 3×3
            cross. Default: ``None``.
        width: Radius of the structuring element in pixels when ``shape``
            is ``"diamond"``, ``"square"``, or ``"disk"``. Larger values
            sever broader bridges and smooth rougher edges but risk eroding
            thin genuine colony appendages. Typical range: 3--9. Default: 5.
        n_iter: Number of opening iterations. Each iteration compounds
            the erosion-dilation cycle, progressively widening the gap at
            each connection point. Typical range: 1--3. Default: 1.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` updated after
        morphological opening. ``rgb``, ``gray``, and ``detect_mat`` are
        unchanged.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of morphological refinement on real plate images.
        :doc:`/explanation/refinement_strategies` for the recommended
        refinement sequence and structuring element guidance.
    """

    shape: Literal["auto", "square", "diamond", "disk"] | NdArrayField | None = None
    width: int = 5
    n_iter: int = 1

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

        for _ in range(self.n_iter):
            image.objmask[:] = opening(image.objmask[:], footprint=footprint)
        return image
