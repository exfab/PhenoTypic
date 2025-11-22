from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from scipy.spatial.distance import euclidean

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools.constants_ import OBJECT, BBOX


class CenterDeviationReducer(ObjectRefiner):
    """Keep the object closest to center and remove off-center detections.

    Intuition:
        For isolated-colony images (e.g., single-spot captures or per-grid-cell
        crops), the true colony is typically near the center. Spurious blobs from
        glare, dust, or agar texture may appear off-center. This operation keeps
        only the object whose centroid is closest to the image center, removing
        all others.

    Why this is useful for agar plates:
        When imaging a grid of pinned colonies, per-cell crops may contain extra
        detections (ringing, condensation, halo). Selecting the most centered
        object stabilizes downstream phenotyping by focusing on the intended
        colony in each crop.

    Use cases:
        - Single-colony crops from a grid plate where occasional debris is picked
          up near edges.
        - Automated pipelines that assume one colony per field-of-view.

    Caveats:
        - If the true colony is notably off-center (misalignment, drift), this
          method can remove it and keep a distractor.
        - Not suitable for multi-colony fields; it will drop all but one object.

    Attributes:
        (No public attributes)

    Examples:
        >>> from phenotypic.refine import CenterDeviationReducer
        >>> op = CenterDeviationReducer()
        >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
    """

    def _operate(self, image: Image):
        img_center_cc = image.shape[1]//2
        img_center_rr = image.shape[0]//2

        bound_info = image.objects.info()

        # Add a column to the bound info for center deviation
        bound_info.loc[:, 'Measurement_CenterDeviation'] = bound_info.apply(
                lambda row: euclidean(u=[row[str(BBOX.CENTER_CC)], row[str(BBOX.CENTER_RR)]],
                                      v=[img_center_cc, img_center_rr]
                                      ),
                axis=1
        )

        # Get the label of the obj w/ the least deviation
        obj_to_keep = bound_info.loc[:, 'Measurement_CenterDeviation'].idxmin()

        # Get a working copy of the object map
        objmap = image.objmap[:]

        # Set Image object map to new other_image
        image.objmap[objmap != obj_to_keep] = 0

        return image
