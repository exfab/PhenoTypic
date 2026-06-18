from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.sdk_.mixin import FootprintMixin
from phenotypic.sdk_.typing_ import NdArrayField, TuneSpec

import numpy as np
from skimage.morphology import dilation


class MaskDilation(ObjectRefiner, FootprintMixin):
    """Expand colony masks outward by morphological dilation.

    Adds pixels around all object boundaries, extending each colony mask by
    one or more structuring element widths. This bridges thin gaps between
    nearby fragments and recovers faint colony halos that strict thresholding
    excluded. Dilation permanently inflates mask area; pair with
    :class:`MaskErosion` or use :class:`MaskClosing` if area accuracy matters.

    For an overview of morphological refinement methods, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Bridging thin background gaps between fragments of the same colony
          before merge-based refinement.
        - Recovering faint outgrowth halos that fall just below the detection
          threshold on plates with heterogeneous colony intensity.
        - Expanding colony masks to include diffuse colony edges before
          intensity or color measurements.

    Consider Also:
        - :class:`MaskClosing` for dilation followed by erosion that bridges
          gaps without permanently inflating colony area.
        - :class:`MaskErosion` for the opposite effect — shrinking masks
          inward to remove thin protrusions or boundary noise.

    Args:
        shape: Structuring element shape for the dilation footprint. ``"auto"``
            scales a diamond to the image size; ``"disk"``, ``"square"``, and
            ``"diamond"`` use named shapes at the given ``width``; a NumPy
            array provides a custom element; ``None`` uses the skimage library
            default. Default: ``None``.
        width: Footprint width in pixels when using a named shape. Larger values
            expand masks further. Typical range: 1--7. Default: 3.
        n_iter: Number of dilation iterations applied sequentially. Each
            additional iteration extends the mask by one further footprint
            radius. Default: 1.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` dilated outward.

    See Also:
        :doc:`/explanation/refinement_strategies` for the recommended
        morphological refinement sequence.
    """

    shape: Literal["auto", "square", "diamond", "disk"] | NdArrayField | None = None
    width: Annotated[int, TuneSpec(1, 7)] = 3
    n_iter: Annotated[int, TuneSpec(1, 3)] = 1

    def _operate(self, image: Image) -> Image:
        if self.shape == "auto":
            footprint = FootprintMixin._make_footprint(
                    shape="diamond", width=max(2, round(np.min(image.shape) * 0.003))
            )
        elif isinstance(self.shape, np.ndarray):
            footprint = self.shape
        elif self.shape in self._footprint_shapes:
            footprint = FootprintMixin._make_footprint(shape=self.shape,
                                                       width=self.width)
        elif not self.shape:
            footprint = None
        else:
            raise AttributeError("Invalid shape type")

        for _ in range(self.n_iter):
            image.objmask[:] = dilation(image.objmask[:], footprint=footprint)
        return image
