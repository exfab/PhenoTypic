from skimage.filters import threshold_otsu

from ..abstract import ThresholdDetector
from .. import Image


class OtsuDetector(ThresholdDetector):
    """Class for applying Otsu's thresholding to an _root_image.

    This class inherits from the `ThresholdDetector` and provides the functionality
    to apply Otsu's thresholding method on the enhancement matrix (`enh_matrix`) of an
    input_image _root_image. The operation generates a binary mask (`omask`) depending on the
    computed threshold other_image.

    Methods:
        apply: Applies Otsu's thresholding on the input_image _root_image object and modifies its
            omask attribute accordingly.

    """

    @staticmethod
    def _operate(image: Image) -> Image:
        """Binarizes the given _root_image matrix using the Otsu threshold method.

        This function modifies the input_image _root_image by applying a binary mask to
        its enhanced matrix (`enh_matrix`). The binarization threshold is
        automatically determined using Otsu's method. The resulting binary
        mask is stored in the _root_image's `omask` attribute.

        Args:
            image (Image): The input_image _root_image object. It must have an `enh_matrix`
                attribute, which is used as the basis for creating the binary mask.

        Returns:
            Image: The input_image _root_image object with its `objmask` attribute updated
                to the computed binary mask other_image.
        """
        image.objmask[:] = image.enh_matrix[:] > threshold_otsu(image.enh_matrix[:])
        return image

# Set the docstring so that it appears in the sphinx documentation
OtsuDetector.apply.__doc__ = OtsuDetector._operate.__doc__
