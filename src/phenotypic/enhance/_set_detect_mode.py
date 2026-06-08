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
    need to operate on different color channels or when a specific channel
    provides superior colony-background contrast for downstream detection.

    For background on channel selection, see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Plates where colonies are more visible in a single color channel
          than in grayscale (e.g., pigmented yeast on agar).
        - Switching to a specific channel (such as red or green) that
          provides stronger colony-background contrast.
        - Resetting enhancements mid-pipeline to start fresh on a
          different channel for a later detection stage.

    Consider Also:
        - :class:`ImageInverter` when the issue is inverted intensity
          polarity rather than channel selection.

    Args:
        mode: Channel to use as the detection matrix. Accepted values:
            ``'gray'`` (default), ``'red'``, ``'green'``, ``'blue'``,
            ``'MinRGB'`` (per-pixel minimum across R/G/B),
            ``'LabL'``, ``'LabA'``, ``'LabB'``, ``'HsvS'``, ``'HsvV'``,
            ``'InvS'``.

    Returns:
        Image: Input image with ``detect_mode`` and ``detect_mat``
        updated to the chosen channel. Prior enhancements to
        ``detect_mat`` are discarded.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of channel selection strategies on plate
        images.
    """

    mode: DetectMode = "gray"

    def _operate(self, image: Image) -> Image:
        image.set_detect_mode(self.mode)
        return image
