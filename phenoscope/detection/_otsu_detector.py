from skimage.filters import threshold_otsu

from ..abstract import ThresholdDetector
from .. import Image


class OtsuDetector(ThresholdDetector):
    def _operate(self, image: Image) -> Image:
        image.omask = image.enh_matrix[:] > threshold_otsu(image.enh_matrix[:])
        return image
