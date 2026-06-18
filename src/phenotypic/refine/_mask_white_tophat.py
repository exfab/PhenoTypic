from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.morphology import white_tophat

from phenotypic.abc_ import ObjectRefiner
from phenotypic.sdk_.mixin import FootprintMixin
from phenotypic.sdk_.typing_ import NdArrayField, TuneSpec


class MaskWhiteTophat(ObjectRefiner, FootprintMixin):
    """Remove bright protrusions and thin bridges from the detection mask using white tophat subtraction.

    Computes the white tophat transform of the binary mask (mask minus its
    morphological opening) and subtracts the result, isolating and eliminating
    small bright structures that are narrower than the structuring element.
    Glare-induced bridges, dust speckles, and thin connections are suppressed
    while the main colony body is preserved.

    For an overview of morphological refinement strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Plates with specular glare that creates bright bridges between
          adjacent colony masks after thresholding.
        - Detection masks with small bright dust or condensation speckles
          that survived the detector.
        - Colonies whose perimeter measurements are inflated by thin
          protruding artefacts.
        - Images where illumination hot-spots produce narrow mask
          connections narrower than the structuring element width.

    Consider Also:
        - :class:`MaskOpening` for general morphological opening that
          removes thin protrusions and breaks bridges without the tophat
          detection step.
        - :class:`SmallObjectRemover` when small artefacts are fully
          disconnected objects better targeted by an area threshold.
        - :class:`ExtractColonyCore` for intensity-based core extraction
          when diffuse halos rather than thin bridges are the primary
          artefact.

    Args:
        shape: Structuring element geometry. ``"disk"`` (default) best
            preserves round colony boundaries; ``"square"`` is more
            aggressive along cardinal axes; ``"diamond"`` is a compromise.
            A NumPy array supplies a custom footprint. Default: ``"disk"``.
        width: Footprint radius in pixels. Structures narrower than this
            value are candidates for removal. Larger values eliminate
            broader protrusions but risk eroding genuine thin appendages.
            ``None`` auto-scales to approximately 0.4% of the smaller image
            dimension. Typical range: 3--10. Default: ``None``.

    Returns:
        Image: Input image with ``objmask`` updated by subtracting the
        white tophat result. Assigning ``objmask`` rebuilds ``objmap``
        from the binary mask; ``rgb``, ``gray``, and ``detect_mat`` are
        unchanged.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for tophat-based
        cleanup workflows on real plate images.
        :doc:`/explanation/refinement_strategies` for a comparison of
        morphological refinement methods.
    """

    shape: Literal["disk", "square", "diamond"] | NdArrayField = "disk"
    width: Annotated[int | None, TuneSpec(3, 10)] = None

    def _operate(self, image: Image) -> Image:
        white_tophat_results = white_tophat(
                image.objmask[:],
                footprint=FootprintMixin._make_footprint(
                        shape=self.shape,
                        width=self._get_footprint_width(array=image.objmask[:]),
                ),
        )
        image.objmask[:] = image.objmask[:] & ~white_tophat_results
        return image

    def _get_footprint_width(self, array: np.ndarray) -> int:
        if self.width is None:
            return int(np.min(array.shape) * 0.004)
        else:
            return self.width
