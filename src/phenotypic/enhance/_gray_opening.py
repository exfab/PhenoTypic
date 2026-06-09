from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING

from skimage import morphology

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import MorphologicalFiltering
from phenotypic.tools_.mixin import FootprintMixin
from phenotypic.tools_.typing_ import TuneSpec


class GrayOpening(MorphologicalFiltering, FootprintMixin):
    """Remove small bright artefacts from ``detect_mat`` via morphological opening.

    Applies erosion followed by dilation with a structuring element, removing
    bright features whose spatial extent is smaller than the element while
    preserving larger colony structures. Effectively suppresses dust particles,
    small noise speckles, and tiny satellite colonies that would otherwise
    generate false detections.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Removing dust particles and small bright noise from plate scans
          before thresholding.
        - Suppressing tiny satellite colonies or debris that would be counted
          as false positives by downstream detectors.
        - Smoothing the detection surface before background subtraction to
          reduce artefact interference.

    Consider Also:
        - :class:`WhiteTophatEnhance` when the goal is to isolate and enhance
          small bright structures rather than suppress them.
        - :class:`SubtractWhiteTophat` for subtracting small bright artefacts
          from the image while retaining the broader background.
        - :class:`LocalEdgeDenoise` for noise reduction that preserves colony
          edges without assuming a specific feature size.

    Args:
        shape: Structuring element geometry. Accepted values: ``'square'``
            (default, preserves axis-aligned edges), ``'diamond'`` (rounded
            at diagonals), ``'disk'`` (uniform circular neighbourhood).
            Default: ``'square'``.
        width: Diameter of the structuring element in pixels. Features smaller
            than ``width`` are removed. Typical range: 3--15. Default: 5.
        n_iter: Number of times to apply the opening in sequence. Repeated
            opening with a small element produces smoother suppression than a
            single pass with a larger element. Default: 1.

    Returns:
        Image: Input image with ``detect_mat`` morphologically opened.
        ``rgb`` and ``gray`` are unchanged.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a visual
        walkthrough of enhancement pipelines on plate images.
        :doc:`/explanation/what_enhancement_does` for how morphological
        filtering fits into the pipeline model.
    """

    shape: Literal["square", "diamond", "disk"] = "square"
    # TODO: review bound (unverified vs literature)
    width: Annotated[int, TuneSpec(3, 15, step=2)] = 5
    n_iter: Annotated[int, TuneSpec(1, 3)] = 1

    def _operate(self, image: Image) -> Image:
        footprint = self._make_footprint(shape=self.shape, width=self.width)
        for _ in range(self.n_iter):
            image.detect_mat[:] = morphology.opening(
                    image=image.detect_mat[:],
                    footprint=footprint,
            )
        return image
