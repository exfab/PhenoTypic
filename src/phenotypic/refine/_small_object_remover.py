from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.morphology import remove_small_objects

from ..abc_ import ObjectRefiner


class SmallObjectRemover(ObjectRefiner):
    """Remove objects smaller than a minimum area from the detection mask.

    Eliminates dust, condensation specks, and noise fragments that appear
    as tiny labeled objects after thresholding. Reduces false positives
    and stabilizes colony counts.

    Best For:
        - Cleaning up salt-and-pepper artifacts after detection.
        - Removing fragmented debris around large colonies.
        - Post-processing after aggressive enhancement or thresholding.

    Consider Also:
        - :class:`BorderObjectRemover` for removing partial colonies at
          image edges (size-independent).
        - :class:`LowCircularityRemover` for removing non-circular artifacts
          regardless of size.

    Args:
        min_size: Minimum object area in pixels to keep. Objects below this
            threshold are removed. Typical range: 20--200 depending on
            image resolution. Default: 64.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` updated to
        exclude small objects.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of refinement operations.
        :doc:`/explanation/refinement_strategies` for choosing the right
        refinement sequence.
    """

    def __init__(self, min_size=64):
        """Initialize the remover.

        Args:
            min_size (int): Minimum object area (in pixels) to keep. Higher
                values remove more small artifacts and fragmented edges,
                generally improving mask cleanliness but risking loss of tiny
                colonies.
        """
        self.min_size = min_size

    def _operate(self, image: Image) -> Image:
        image.objmap[:] = remove_small_objects(
                image.objmap[:], min_size=self.min_size
        )
        return image
