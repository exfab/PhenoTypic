from skimage.filters import threshold_otsu

from ..interface import ThresholdDetector
from .. import Image


class OtsuDetector(ThresholdDetector):
    def _operate(self, image: Image) -> Image:
        image.object_mask = image.enhanced_array > threshold_otsu(image.enhanced_array)
        return image
