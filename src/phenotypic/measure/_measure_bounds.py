from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
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

    Returns:
        pd.DataFrame: Object-level spatial data with columns:

            - Label: unique object identifier.
            - CenterRR, CenterCC: geometric centroid coordinates.
            - IntensityWeightedCenterRR, IntensityWeightedCenterCC:
              intensity-weighted centroid coordinates (skimage
              ``centroid_weighted``).
            - DistWeightedCenterRR, DistWeightedCenterCC: row/column of
              the per-object distance-transform maximum (deepest interior
              point — robust to thin filamentous extensions).
            - MinRR, MinCC: top-left corner of bounding box.
            - MaxRR, MaxCC: bottom-right corner of bounding box.

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

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
    """

    _measurement_info_class = BBOX

    def _operate(self, image: Image) -> pd.DataFrame:
        objmap = image.objmap[:]
        gray = image.gray[:]

        results = pd.DataFrame(
                data=regionprops_table(
                        label_image=objmap,
                        intensity_image=gray,
                        properties=["label", "centroid", "bbox", "centroid_weighted"]
                )
        ).rename(
                columns={
                    "label"              : OBJECT.LABEL,
                    "centroid-0"         : str(BBOX.CENTER_RR),
                    "centroid-1"         : str(BBOX.CENTER_CC),
                    "centroid_weighted-0": str(BBOX.INTENSITY_WEIGHTED_CENTER_RR),
                    "centroid_weighted-1": str(BBOX.INTENSITY_WEIGHTED_CENTER_CC),
                    "bbox-0"             : str(BBOX.MIN_RR),
                    "bbox-1"             : str(BBOX.MIN_CC),
                    "bbox-2"             : str(BBOX.MAX_RR),
                    "bbox-3"             : str(BBOX.MAX_CC),
                }
        )

        # Per-object distance-transform maximum (deepest interior point).
        # Robust to thin filamentous extensions that pull intensity-weighted
        # centroids off-body.
        #
        # Vectorized via global EDT + ndi.maximum_position. Inter-object
        # boundaries are zeroed before the EDT so touching colonies don't
        # bleed into each other.
        #
        # Memory note: the global EDT allocates a full-image float64 array
        # (~128 MB at 4000x4000, ~288 MB at 6000x6000). If this becomes a
        # bottleneck on large plates, switch to a per-object loop using
        # scipy.ndimage.find_objects + per-bbox distance_transform_edt; the
        # pattern at _measure_symmetric_zones.py:237-239 is the precedent.
        nonzero = objmap > 0
        if nonzero.any():
            max_label_value = np.iinfo(objmap.dtype).max
            masked_for_min = np.where(nonzero, objmap, max_label_value)
            nbr_max = ndi.maximum_filter(objmap, size=3)
            nbr_min_nz = ndi.minimum_filter(masked_for_min, size=3)
            inter_boundary = nonzero & (
                (nbr_max > objmap) | (nbr_min_nz < objmap)
            )

            binary = nonzero & ~inter_boundary
            dt = ndi.distance_transform_edt(binary)

            labels = np.unique(objmap[nonzero])
            positions = ndi.maximum_position(dt, labels=objmap, index=labels)
            positions = np.asarray(positions, dtype=float)

            dist_df = pd.DataFrame({
                OBJECT.LABEL                      : labels,
                str(BBOX.DIST_WEIGHTED_CENTER_RR): positions[:, 0],
                str(BBOX.DIST_WEIGHTED_CENTER_CC): positions[:, 1],
            })
            results = results.merge(dist_df, on=OBJECT.LABEL, how="left")
        else:
            results[str(BBOX.DIST_WEIGHTED_CENTER_RR)] = np.array([], dtype=float)
            results[str(BBOX.DIST_WEIGHTED_CENTER_CC)] = np.array([], dtype=float)

        return results


MeasureBounds.__doc__ = BBOX.append_rst_to_doc(MeasureBounds)
