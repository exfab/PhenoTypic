import numpy as np
from skimage.exposure import rescale_intensity

from ..abstract import ImagePreprocessor
from .. import Image


class ContrastStretching(ImagePreprocessor):
    def __init__(self, lower_percentile: int = 2, upper_percentile: int = 98):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def _operate(self, image: Image) -> Image:
        p_lower, p_upper = np.percentile(image.enh_matrix[:], (self.lower_percentile, self.upper_percentile))
        image.enh_matrix[:] = rescale_intensity(image=image.enh_matrix[:], in_range=(p_lower, p_upper))
        return image
