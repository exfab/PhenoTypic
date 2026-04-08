from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
from phenotypic.abc_ import GridMeasureFeatures
from ..tools_.measurement_info_ import GRID_SPREAD

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from phenotypic.tools_.measurement_info_ import BBOX, GRID


class MeasureGridSpread(GridMeasureFeatures):
    """Quantify within-well colony dispersion using pairwise centroid distances.

    Compute the sum of squared pairwise Euclidean distances between all
    colony centroids in each grid section. High values indicate multiple
    dispersed objects within a single well -- a sign of over-segmentation,
    fragmented growth, or invasive spreading.

    Best For:
        - Detecting over-segmented wells where multiple objects were
          found instead of a single cohesive colony.
        - Identifying invasive or spreading growth that extends beyond
          the designated grid position.
        - Flagging wells with questionable data quality for manual
          review or exclusion from downstream analysis.

    Consider Also:
        - :class:`MeasureGridSpatial` for between-well neighbor
          distances rather than within-well dispersion.
        - :class:`MeasureGridLinRegStats` for positional accuracy
          metrics based on linear regression.
        - :class:`MeasureBounds` for raw centroid positions per colony.

    Returns:
        pd.DataFrame: Section-level statistics sorted by spread
        (descending) with columns:

            - count: number of colonies detected in the section.
            - ObjectSpread: sum of squared pairwise Euclidean distances
              between colony centroids in the section.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of grid-level measurements.
    """

    _measurement_info_class = GRID_SPREAD

    def _operate(self, image: GridImage) -> pd.DataFrame:
        gs_table = image.grid.info()
        gs_counts = pd.DataFrame(
            gs_table.loc[:, str(GRID.ROW_MAJOR_IDX)].value_counts())

        obj_spread = []
        for gs_bindex in gs_counts.index:
            curr_gs_subtable = gs_table.loc[
                gs_table.loc[:, str(GRID.ROW_MAJOR_IDX)] == gs_bindex, :
            ]

            x_vector = curr_gs_subtable.loc[:, str(BBOX.CENTER_CC)]
            y_vector = curr_gs_subtable.loc[:, str(BBOX.CENTER_RR)]
            obj_vector = np.array(list(zip(x_vector, y_vector)))
            gs_distance_matrix = distance_matrix(x=obj_vector, y=obj_vector, p=2)

            obj_spread.append(np.sum(np.unique(gs_distance_matrix) ** 2))
        gs_counts.insert(loc=1, column=str(GRID_SPREAD.OBJECT_SPREAD),
                         value=pd.Series(obj_spread))
        gs_counts.sort_values(by=str(GRID_SPREAD.OBJECT_SPREAD), ascending=False,
                              inplace=True)
        return gs_counts


MeasureGridSpread.__doc__ = GRID_SPREAD.append_rst_to_doc(MeasureGridSpread)
