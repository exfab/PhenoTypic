from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ImageOperation
from phenotypic.tools_.typing_ import DetectMode


class SetDetectMode(ImageOperation):
    """Switch the detection matrix source channel mid-pipeline.

    Resets the detection matrix to a fresh copy of the chosen channel,
    discarding any enhancements applied so far.

    Args:
        mode: Channel to use for the detection matrix.
            ``'gray'`` (default), ``'red'``, ``'green'``, ``'blue'``,
            or ``'min_rgb'``.

    Returns:
        Image: The image with ``detect_mode`` and ``detect_mat`` updated.

    Examples:
        Switch to the red channel before detection:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import SetDetectMode, GaussianBlur
        >>> from phenotypic import ImagePipeline
        >>> image = load_synth_yeast_plate()
        >>> pipeline = ImagePipeline([SetDetectMode(mode='red'), GaussianBlur(sigma=1)])
        >>> result = pipeline.apply(image)
        >>> result.detect_mode
        'red'
    """

    def __init__(self, mode: DetectMode = "gray"):
        super().__init__()
        self.mode = mode

    def _operate(self, image: Image) -> Image:
        image.set_detect_mode(self.mode)
        return image
