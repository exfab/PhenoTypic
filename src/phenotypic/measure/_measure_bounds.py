from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import pandas as pd
from skimage.measure import regionprops_table

from phenotypic.abc_ import MeasureFeatures

from ..tools_.constants_ import OBJECT
from ..tools_.measurement_info_ import BBOX


class MeasureBounds(MeasureFeatures):
    """Extract bounding box coordinates and centroids of detected colonies.

    Compute the axis-aligned bounding box and centroid (geometric and
    intensity-weighted) for each detected colony. These spatial
    measurements form the foundation for region-of-interest extraction,
    grid alignment assessment, and neighbor-distance calculations.

    Best For:
        - Computing centroids for aligning colonies to expected grid
          positions in arrayed assays.
        - Extracting region-of-interest crops for downstream intensity,
          color, or texture analysis.
        - Assessing colony positioning relative to plate edges or well
          boundaries.

    Consider Also:
        - :class:`MeasureShape` for full morphological metrics built on
          top of bounding box data.
        - :class:`MeasureGridLinRegStats` for regression-based grid
          alignment quality using centroid positions.
        - :class:`MeasureGridSpatial` for neighbor distance calculations
          using bounding boxes.

    Returns:
        pd.DataFrame: Object-level spatial data with columns:

            - Label: unique object identifier.
            - CenterRR, CenterCC: geometric centroid coordinates.
            - WeightedCenterRR, WeightedCenterCC: intensity-weighted
              centroid coordinates.
            - MinRR, MinCC: top-left corner of bounding box.
            - MaxRR, MaxCC: bottom-right corner of bounding box.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
    """

    _measurement_info_class = BBOX

    def _operate(self, image: Image) -> pd.DataFrame:
        results = pd.DataFrame(
                data=regionprops_table(
                        label_image=image.objmap[:],
                        intensity_image=image.gray[:],
                        properties=["label", "centroid", "bbox", "centroid_weighted"]
                )
        ).rename(
                columns={
                    "label"              : OBJECT.LABEL,
                    "centroid-0"         : str(BBOX.CENTER_RR),
                    "centroid-1"         : str(BBOX.CENTER_CC),
                    "centroid_weighted-0": BBOX.WEIGHTED_CENTER_RR,
                    "centroid_weighted-1": BBOX.WEIGHTED_CENTER_CC,
                    "bbox-0"             : str(BBOX.MIN_RR),
                    "bbox-1"             : str(BBOX.MIN_CC),
                    "bbox-2"             : str(BBOX.MAX_RR),
                    "bbox-3"             : str(BBOX.MAX_CC),
                }
        )

        return results


MeasureBounds.__doc__ = BBOX.append_rst_to_doc(MeasureBounds)
