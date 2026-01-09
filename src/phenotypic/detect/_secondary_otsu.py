from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.filters import threshold_otsu
from phenotypic.abc_ import ThresholdDetector


class SecondaryOtsuDetector(ThresholdDetector):
    """Applies otsu thresholding again to an image that already has an object map. If no,
    object map is found, it will apply the otsu threshold twice"""

    def _operate(self, image: Image) -> Image:
        """Applies otsu thresholding again to an image that already has an object map. If no,
        object map is found, it will apply the otsu threshold twice"""

        # If there are no objects in the image already perform an initial otsu
        enh_gray = image.enh_gray[:]
        if image.num_objects == 0:
            objmask = enh_gray >= threshold_otsu(enh_gray)
        else:
            objmask = image.objmask[:]

        objvals = enh_gray * objmask
        image.objmask = objvals >= threshold_otsu(objvals[objvals.nonzero()])
        return image
