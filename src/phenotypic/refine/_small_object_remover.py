from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.morphology import remove_small_objects

from ..abc_ import ObjectRefiner


class SmallObjectRemover(ObjectRefiner):
    """Remove detected objects below a minimum pixel area from the detection mask.

    Eliminates dust particles, condensation specks, agar debris, and
    noise fragments that appear as small labeled regions after thresholding
    or edge detection. Reduces false-positive colony counts and stabilizes
    downstream measurements.

    For guidance on choosing the right refinement sequence, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Cleaning up salt-and-pepper artefacts left after aggressive
          thresholding or Frangi filtering.
        - Removing fragmented debris and micro-condensation specks that
          cluster around genuine colonies.
        - Post-processing dense detection masks where noise fragments
          inflate colony counts.
        - Stabilizing counts before grid assignment or size measurement.

    Consider Also:
        - :class:`RemoveBorderObjects` for removing partial colonies at
          image edges regardless of their size.
        - :class:`RemoveNonCircular` for discarding non-circular artefacts
          regardless of area.
        - :class:`SmallToLargeMerger` when small fragments should be
          absorbed into a nearby large colony rather than discarded.

    Args:
        min_size: Minimum object area in pixels to retain. Objects whose
            pixel area falls strictly below this value are removed. Scale
            with image resolution, since colony area grows roughly with the
            square of dpi; as a rough starting point, try 20--100 px near
            300 dpi and 100--500 px near 1200 dpi. Default: 64.

    Returns:
        Image: Input image with ``objmap`` updated to exclude objects
        smaller than ``min_size``. ``objmask`` is updated to match.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of refinement operations on real plate images.
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
