from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:from phenotypic import Image

import colour

from phenotypic.abstract import ImageCorrector


class GammaDecoder(ImageCorrector):
    """Linearizes the rgb values that removes device-dependent biases.

    In converting from raw spectral information to non-raw formats such as jpeg or .tiff, many cameras will apply gamma encoding to make
    the image appear visually closer to what humans would see. This introduces a nonlinear bias to our quantification for analysis. For
    the most accurate measurements of spectral information from images, it's important to remove this bias.

    Args:
        **kwargs: other keyword arguments to be passed into colour.cctf_decoding. The function will use the Image.color_profile attribute

    References:
        - https://colour.readthedocs.io/en/latest/generated/colour.cctf_decoding.html

    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _operate(self, image: Image) -> Image:
        # Skip grayscale images
        if image.imformat.is_matrix():
            image._known_gamma_decoding = True
            return image

        tmp_kwargs = self.kwargs.copy()
        decoding_func = tmp_kwargs.pop('function', image.color_profile)

        linear_rgb = colour.cctf_decoding(value=image.array[:], function=decoding_func, **tmp_kwargs)
        image.array[:] = linear_rgb[:]

        image._known_gamma_decoding = True
        return image
