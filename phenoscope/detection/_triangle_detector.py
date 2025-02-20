from skimage.filters import threshold_triangle

from ..interface import ThresholdDetector
from .. import Image


class TriangleDetector(ThresholdDetector):
    def _operate(self, image: Image) -> Image:
        image.obj_mask = image.det_matrix[:] >= threshold_triangle(image.det_matrix[:])
        return image
