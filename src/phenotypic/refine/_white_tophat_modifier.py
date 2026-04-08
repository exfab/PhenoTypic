from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.morphology import white_tophat

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin


class WhiteTophat(ObjectRefiner, FootprintMixin):
    """Remove small bright mask structures using white tophat subtraction.

    Applies a white tophat transform to the binary mask to detect small
    bright features (glare bridges, dust speckles, thin connections) and
    subtracts them. Produces cleaner, more compact masks that better match
    colony boundaries under uneven illumination.

    Best For:
        - Reducing glare-induced bridges between neighboring colonies.
        - Removing bright speckles or dust embedded in masks after
          thresholding.
        - Cleaning up thin bright connections that inflate colony
          perimeters.

    Consider Also:
        - :class:`MaskOpener` for general morphological opening that
          removes thin protrusions without tophat detection.
        - :class:`SmallObjectRemover` when small artifacts are entire
          disconnected objects rather than thin bridges.
        - :class:`GMMCoreExtractor` for intensity-based core extraction
          when halos are the primary artifact.

    Args:
        shape: Structuring element shape. ``"disk"`` preserves round
            features, ``"square"`` is more aggressive along axes,
            ``"diamond"`` provides a compromise. A NumPy array provides
            a custom element. Default: ``"disk"``.
        width: Footprint width in pixels. Larger values remove broader
            bright features but risk shrinking thin colony appendages.
            ``None`` auto-scales to ~0.4% of the smallest image
            dimension. Default: None.

    Returns:
        Image: Input image with ``objmask`` updated by subtracting the
        white tophat result.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for tophat-based
        cleanup workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        morphological refinement methods.
    """

    def __init__(self,
                 shape: Literal["disk", "square", "diamond"] | np.ndarray = "disk",
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
        self.shape = shape
        self.width = width

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
