from skimage.filters import threshold_triangle

from ..abstract import ThresholdDetector
from .. import Image


class TriangleDetector(ThresholdDetector):
    def _operate(self, image: Image) -> Image:
        image.omask = image.enh_matrix[:] >= threshold_triangle(image.enh_matrix[:])
        return image
