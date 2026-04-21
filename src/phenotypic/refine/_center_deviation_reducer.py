from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from scipy.spatial.distance import euclidean

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.measurement_info_ import BBOX


class CenterDeviationReducer(ObjectRefiner):
    """Retain only the object whose centroid is closest to the image center.

    Computes the Euclidean distance from each object's centroid to the image
    center and keeps the single nearest object, removing all others. Useful
    for per-cell crops where the intended colony sits near the center and
    peripheral detections are artifacts.

    Returns:
        Image: Input image with ``objmap`` reduced to the single most centered
        object.

    Best For:
        - Single-colony crops from grid plates where debris appears near edges.
        - Automated pipelines that assume one colony per field-of-view.
        - Post-crop cleanup in conjunction with grid-based workflows.

    Consider Also:
        - :class:`GridSectionLargest` when the largest object per grid cell is
          more reliable than the most centered.
        - :class:`SmallObjectRemover` when peripheral artifacts are simply
          smaller than the true colony.
        - :class:`ReduceMultipleGridObjects` for grid-aware multi-detection
          reduction using positional regression.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for boundary cleanup
        workflows.
        :doc:`/explanation/refinement_strategies` for an overview of
        refinement approaches.
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
