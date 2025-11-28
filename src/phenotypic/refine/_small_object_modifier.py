from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from skimage.morphology import remove_small_objects

from ..abc_ import ObjectRefiner


class SmallObjectRemover(ObjectRefiner):
    """Remove small, likely spurious objects from a labeled object map.

    Intuition:
        After thresholding/segmentation of agar-plate images, tiny specks from
        dust, condensation, camera noise, or over-segmentation can appear as
        separate labeled objects. Removing these below a minimum size reduces
        false positives and stabilizes downstream phenotyping.

    Use cases (agar plates):
        - Clean up salt-and-pepper detections before measuring colony size or
          shape.
        - Suppress fragmented debris around large colonies that may bias
          counts or area statistics.
        - Post-processing step after aggressive enhancement/thresholding.

    Tuning and effects:
        - min_size: Sets the minimum object area (in pixels). Increasing this
          value removes more small fragments, typically improving mask quality
          and background suppression, but may also delete legitimate micro-
          colonies when colonies are extremely small or underexposed.

    Caveats:
        - Setting ``min_size`` too high can remove small but real colonies or
          early-time-point growth, reducing recall.
        - The optimal threshold depends on resolution; what is “small” at
          high-resolution imaging may be substantial at low resolution.

    Attributes:
        (No public attributes)

    Examples:
        >>> from phenotypic.refine import SmallObjectRemover
        >>> op = SmallObjectRemover(min_size=100)
        >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
    """

    def __init__(self, min_size=64):
        """Initialize the remover.

        Args:
            min_size (int): Minimum object area (in pixels) to keep. Higher
                values remove more small artifacts and fragmented edges,
                generally improving mask cleanliness but risking loss of tiny
                colonies.
        """
        self.__min_size = min_size

    def _operate(self, image: Image) -> Image:
        image.objmap[:] = remove_small_objects(image.objmap[:], min_size=self.__min_size)
        return image
