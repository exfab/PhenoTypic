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

    Args:
        min_size: Minimum object area in pixels to keep. Objects below this
            threshold are removed. Typical range: 20--200 depending on
            image resolution. Default: 64.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` updated to
        exclude small objects.

    Best For:
        - Cleaning up salt-and-pepper artifacts after detection.
        - Removing fragmented debris around large colonies.
        - Post-processing after aggressive enhancement or thresholding.

    Consider Also:
        - :class:`RemoveBorderObjects` for removing partial colonies at
          image edges (size-independent).
        - :class:`LowCircularityRemover` for removing non-circular artifacts
          regardless of size.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of refinement operations.
        :doc:`/explanation/refinement_strategies` for choosing the right
        refinement sequence.
    """

    min_size: int = 64

    def _operate(self, image: Image) -> Image:
        objmap = image.objmap[:]
        # remove_small_objects warns when given a label image with a single
        # non-zero label (treats it as ambiguous binary vs labeled). Pass a
        # boolean array in that case; re-label afterwards if needed.
        if objmap.max() <= 1:
            cleaned = remove_small_objects(
                    objmap.astype(bool), min_size=self.min_size
            )
            image.objmap[:] = cleaned.astype(objmap.dtype)
        else:
            image.objmap[:] = remove_small_objects(objmap, min_size=self.min_size)
        return image
