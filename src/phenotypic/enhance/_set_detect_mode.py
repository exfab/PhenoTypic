from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ImageOperation
from phenotypic.tools_.typing_ import DetectMode


class SetDetectMode(ImageOperation):
    """Switch the detection matrix source channel mid-pipeline.

    Resets ``detect_mat`` to a fresh copy of the chosen channel, discarding
    any enhancements applied so far. Useful when different pipeline stages
    need to operate on different color channels.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        mode: Channel to use for the detection matrix. Accepted values:
            ``'gray'`` (default), ``'red'``, ``'green'``, ``'blue'``,
            ``'min_rgb'``.

    Returns:
        Image: Input image with ``detect_mode`` and ``detect_mat``
        updated to the chosen channel.

    Best For:
        - Switching to a specific color channel (e.g., red) that provides
          better colony-background contrast.
        - Resetting enhancements mid-pipeline to start fresh on a
          different channel.
        - Plates where colonies are more visible in a single color channel
          than in grayscale.

    Consider Also:
        - :class:`ImageInverter` when the issue is inverted polarity
          rather than channel selection.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of channel selection strategies.
    """

    mode: DetectMode = "gray"

    def _operate(self, image: Image) -> Image:
        image.set_detect_mode(self.mode)
        return image
