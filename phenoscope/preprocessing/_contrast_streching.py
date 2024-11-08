import numpy as np
import numpy as pd
from skimage.exposure import rescale_intensity

from ..interface import ImagePreprocessor
from .. import Image

class ContrastStretching(ImagePreprocessor):
    def __init__(self,lower_percentile:int=2, upper_percentile:int=98):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def _operate(self, image: Image) -> Image:
        p_lower, p_upper = np.percentile(image.enhanced_array, (self.lower_percentile, self.upper_percentile))
        image.enhanced_array = rescale_intensity(image.enhanced_array, in_range=(p_lower, p_upper))
        return image