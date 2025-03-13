from skimage.filters import gaussian

from ..abstract import ImagePreprocessor
from .. import Image


class GaussianPreprocessor(ImagePreprocessor):
    def __init__(self, sigma=2, mode='reflect', truncate=4.0, channel_axis=None):
        self._sigma = sigma
        self._mode = mode
        self._truncate = truncate
        self._channel_axis = channel_axis

    def _operate(self, image: Image) -> Image:
        image.enh_matrix[:] = gaussian(
                image=image.enh_matrix[:],
                sigma=self._sigma,
                mode=self._mode,
                truncate=self._truncate,
                channel_axis=self._channel_axis
        )
        return image
