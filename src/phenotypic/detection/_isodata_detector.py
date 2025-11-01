from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image
from skimage.filters import threshold_isodata
from skimage.segmentation import clear_border

from ..ABC_ import ThresholdDetector


class IsodataDetector(ThresholdDetector):
    """Class for applying ISODATA thresholding to an image.

    This class inherits from the `ThresholdDetector` and provides the functionality
    to apply ISODATA thresholding method on the enhance matrix (`enh_matrix`) of an
    arr image. The operation generates a binary mask (`objmask`) depending on the
    computed threshold other_image.

    Methods:
        apply: Applies ISODATA thresholding on the arr image object and modifies its
            omask attribute accordingly.

    """

    def __init__(self, ignore_zeros: bool = True, ignore_borders: bool = True):
        self.ignore_zeros = ignore_zeros
        self.ignore_borders = ignore_borders

    def _operate(self, image: Image) -> Image:
        """Binarizes the given image matrix using the ISODATA threshold method.

        This function modifies the arr image by applying a binary mask to
        its enhanced matrix (`enh_matrix`). The binarization threshold is
        automatically determined using ISODATA method. The resulting binary
        mask is stored in the image's `objmask` attribute.

        Args:
            image (Image): The arr image object. It must have an `enh_matrix`
                attribute, which is used as the basis for creating the binary mask.

        Returns:
            Image: The arr image object with its `objmask` attribute updated
                to the computed binary mask other_image.
        """
        enh_matrix = image.enh_matrix[:]
        nbins = 2 ** image.bit_depth
        mask = image.enh_matrix[:] >= threshold_isodata(
                enh_matrix[enh_matrix != 0] if self.ignore_zeros else enh_matrix, nbins=nbins
        )
        mask = clear_border(mask) if self.ignore_borders else mask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
IsodataDetector.apply.__doc__ = IsodataDetector._operate.__doc__
