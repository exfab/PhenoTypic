import numpy as np
from skimage.exposure import rescale_intensity

from ..abstract import ImageEnhancer
from .. import Image


class ContrastStretching(ImageEnhancer):
    """
    Performs contrast stretching on an image to enhance its visual quality.

    Contrast stretching is a technique used to improve the contrast in the image by
    redistributing the range of pixel intensity values. This class allows adjustment
    of the intensity range using lower and upper percentiles of pixel values, enabling
    fine-tuning of contrast enhancement for different types of images.

    Parameters:
        lower_percentile (int): The lower percentile value used for intensity rescaling.
            Pixel values below this percentile will be adjusted to the lower bound of the
            intensity range.
        upper_percentile (int): The upper percentile value used for intensity rescaling.
            Pixel values above this percentile will be adjusted to the upper bound of the
            intensity range.
    """
    def __init__(self, lower_percentile: int = 2, upper_percentile: int = 98):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def _operate(self, image: Image) -> Image:
        p_lower, p_upper = np.percentile(image.enh_matrix[:], (self.lower_percentile, self.upper_percentile))
        image.enh_matrix[:] = rescale_intensity(image=image.enh_matrix[:], in_range=(p_lower, p_upper))
        return image