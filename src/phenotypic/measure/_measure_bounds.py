from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage.measure import regionprops_table

from phenotypic.abc_ import MeasureFeatures

from phenotypic.schema import OBJECT
from phenotypic.schema import BBOX


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
            - DistWeightedCenterRR, DistWeightedCenterCC: per-object
              DT-weighted centroid (Sum(dt*pos)/Sum(dt)) — robust to thin
              filamentous extensions (low DT weight) and to budding doublets
              (the two lobes balance to the neck).
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

    _measurement_infoclass: ClassVar[type] = BBOX

    def _operate(self, image: Image) -> pd.DataFrame:
        objmap = image.objmap[:]
        # gray is stored float32; upcast for the intensity-weighted centroid
        # accumulation so values match the historical float64 layer.
        gray = image.gray[:].astype(np.float64, copy=False)

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

        # Per-object DT-weighted centroid (Sum(dt*pos)/Sum(dt)). Robust to thin
        # filamentous extensions (low DT weight keeps the center on-body) and to
        # budding doublets (the two lobes' DT mass balances to the neck), unlike a
        # DT-argmax which would snap onto a single lobe.
        #
        # Vectorized via global EDT + ndi.center_of_mass. Inter-object
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
            # DT-weighted centroid (Sum(dt*pos)/Sum(dt)) — robust to filament hyphae
            # (low DT weight) AND budding doublets (two lobes balance to the neck),
            # replacing the former DT-argmax which snapped onto a single lobe.
            positions = np.asarray(
                ndi.center_of_mass(dt, labels=objmap, index=labels), dtype=float
            )
            # Degenerate guard: objects whose pixels were all zeroed as inter-object
            # boundary have zero DT mass -> center_of_mass returns NaN. Fall back to
            # the unweighted (geometric) mask centroid for those labels.
            nan_rows = np.isnan(positions).any(axis=1)
            if nan_rows.any():
                geom = np.asarray(
                    ndi.center_of_mass(nonzero.astype(float), labels=objmap, index=labels),
                    dtype=float,
                )
                positions[nan_rows] = geom[nan_rows]

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
