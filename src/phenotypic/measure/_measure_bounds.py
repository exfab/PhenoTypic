from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import pandas as pd
from skimage.measure import regionprops_table

from phenotypic.abc_ import MeasureFeatures

from ..tools.constants_ import OBJECT, BBOX


class MeasureBounds(MeasureFeatures):
    """Extract spatial boundaries and centroids of detected microbial colonies.

    This class computes the bounding box coordinates and centroid position of each detected colony
    in the image. These measurements form the foundation for shape analysis, grid alignment assessment,
    and spatial statistics in colony phenotyping workflows.

    **Intuition:** Every detected colony occupies a region in the image; understanding its bounds and
    center is essential for downstream analyses. The bounding box defines the minimal rectangular region
    containing the colony, while centroids enable distance-based metrics (e.g., grid alignment, spread
    measurements) and are used to relate colonies to expected well positions in arrayed assays.

    **Use cases (agar plates):**
    - Establish the spatial footprint of each detected colony for morphological analysis.
    - Compute centroids for aligning colonies to grid positions in high-throughput assays.
    - Enable region-of-interest (ROI) extraction for downstream intensity, color, or texture measurements.
    - Assess colony positioning relative to plate edges to detect spreading beyond well boundaries.
    - Support grid refinement operations by providing predicted centroid positions for outlier filtering.

    **Caveats:**
    - Bounding box depends entirely on accurate object segmentation/masking; poor or over-segmented masks
      yield misleading bounds.
    - Bounding box is axis-aligned; it may include empty space for rotated or crescent-shaped colonies.
    - Centroid assumes the colony is a connected component; fragmented or multi-component objects will
      have unreliable centroid positions.
    - Boundaries are computed in image pixel coordinates; they must be scaled or transformed if results
      are used with data in different coordinate systems (e.g., physical microns).

    Returns:
        pd.DataFrame: Object-level spatial data with columns:
            - Label: Unique object identifier.
            - CenterRR, CenterCC: Centroid row and column coordinates.
            - MinRR, MinCC: Minimum (top-left) row and column of bounding box.
            - MaxRR, MaxCC: Maximum (bottom-right) row and column of bounding box.

    Examples:
        .. dropdown:: Extract colony boundaries for a plate image

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.detect import OtsuDetector
                from phenotypic.measure import MeasureBounds

                # Load image and detect colonies
                image = Image.from_image_path("colony_plate.jpg")
                detector = OtsuDetector()
                image = detector.operate(image)

                # Extract boundaries
                boundsizer = MeasureBounds()
                bounds = boundsizer.operate(image)
                print(bounds.head())
                # Output: Label, CenterRR, CenterCC, MinRR, MinCC, MaxRR, MaxCC

        .. dropdown:: Use boundaries to extract colony ROIs

            .. code-block:: python

                # Extract a region for each colony for detailed analysis
                bounds = boundsizer.operate(image)
                for idx, row in bounds.iterrows():
                    min_rr, max_rr = int(row['BBOX_MinRR']), int(row['BBOX_MaxRR'])
                    min_cc, max_cc = int(row['BBOX_MinCC']), int(row['BBOX_MaxCC'])
                    colony_roi = image.rgb[min_rr:max_rr, min_cc:max_cc]
                    # Process ROI independently (e.g., color analysis, morphology)
    """

    def _operate(self, image: Image) -> pd.DataFrame:
        results = pd.DataFrame(
            data=regionprops_table(
                label_image=image.objmap[:], properties=["label", "centroid", "bbox"]
            )
        ).rename(
            columns={
                "label": OBJECT.LABEL,
                "centroid-0": str(BBOX.CENTER_RR),
                "centroid-1": str(BBOX.CENTER_CC),
                "bbox-0": str(BBOX.MIN_RR),
                "bbox-1": str(BBOX.MIN_CC),
                "bbox-2": str(BBOX.MAX_RR),
                "bbox-3": str(BBOX.MAX_CC),
            }
        )

        return results


MeasureBounds.__doc__ = BBOX.append_rst_to_doc(MeasureBounds)
