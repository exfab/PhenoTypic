from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from scipy.spatial.distance import euclidean

from phenotypic.abc_ import ObjectRefiner
from phenotypic.schema import BBOX


class KeepNearestCenter(ObjectRefiner):
    """Retain only the object whose bounding-box center lies closest to the image center.

    Computes the Euclidean distance from each detected object's bounding-box
    center to the image center and discards all objects except the single
    nearest. The operation
    produces an ``objmap`` with at most one labeled region. It is most effective on
    per-cell crops where the intended colony occupies the center of the field and
    peripheral detections are debris or bleed-through from adjacent wells.

    For an overview of refinement approaches, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Single-colony crops produced by grid-based workflows where debris
          and condensation appear near the crop boundary.
        - Automated pipelines that assume exactly one colony per field of view
          and need a robust single-object guarantee.
        - Post-crop cleanup on images extracted from 96-well or 384-well grids
          where bleed-through from neighbouring cells is visible at the edges.

    Consider Also:
        - :class:`KeepSectionLargest` when the largest object per grid cell is
          a more reliable proxy for the true colony than spatial centrality.
        - :class:`SmallObjectRemover` when peripheral artefacts are consistently
          smaller than the genuine colony and size is the better discriminant.
        - :class:`ReduceSectionsByLine` for grid-aware multi-detection reduction
          using expected positional regression rather than centroid distance.

    Returns:
        Image: Input image with ``objmap`` reduced to the single object whose
        bounding-box center is closest to the image center; all other objects
        are set to background.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for single-object
        cleanup workflows on cropped plate images.
    """

    def _operate(self, image: Image):
        img_center_cc = image.shape[1] // 2
        img_center_rr = image.shape[0] // 2

        bound_info = image.objects.info()

        # Add a column to the bound info for center deviation
        bound_info.loc[:, "Measurement_CenterDeviation"] = bound_info.apply(
                lambda row: euclidean(
                        u=[row[str(BBOX.CENTER_CC)], row[str(BBOX.CENTER_RR)]],
                        v=[img_center_cc, img_center_rr],
                ),
                axis=1,
        )

        # Get the label of the obj w/ the least deviation
        obj_to_keep = bound_info.loc[:, "Measurement_CenterDeviation"].idxmin()

        # Get a working copy of the object map
        objmap = image.objmap[:]

        # Set Image object map to new other_image
        image.objmap[objmap != obj_to_keep] = 0

        return image
