from skimage.filters import threshold_otsu

from ..interface import ThresholdDetector
from .. import Image


class OtsuDetector(ThresholdDetector):
    def _operate(self, image: Image) -> Image:
        image.obj_mask = image.det_matrix[:] > threshold_otsu(image.det_matrix[:])
        return image
