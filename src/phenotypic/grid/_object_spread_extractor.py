from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage
from phenotypic.abstract import GridMeasureFeatures

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from phenotypic.util.constants_ import GRID, OBJECT, BBOX

class ObjectSpreadExtractor(GridMeasureFeatures):
    """
    This module measure's an objects spread from the grid section's center points
    """

    @staticmethod
    def _operate(image: GridImage) -> pd.DataFrame:
        gs_table = image.grid.info()
        gs_counts = pd.DataFrame(gs_table.loc[:, GRID.GRID_SECTION_NUM].value_counts())

        obj_spread = []
        for gs_bindex in gs_counts.index:
            curr_gs_subtable = gs_table.loc[gs_table.loc[:, GRID.GRID_SECTION_NUM] == gs_bindex, :]

            x_vector = curr_gs_subtable.loc[:, str(BBOX.CENTER_CC)]
            y_vector = curr_gs_subtable.loc[:, str(BBOX.CENTER_RR)]
            obj_vector = np.array(list(zip(x_vector, y_vector)))
            gs_distance_matrix = distance_matrix(x=obj_vector, y=obj_vector, p=2)

            obj_spread.append(np.sum(np.unique(gs_distance_matrix) ** 2))
        gs_counts.insert(loc=1, column='ObjectSpread', value=pd.Series(obj_spread))
        gs_counts.sort_values(by='ObjectSpread', ascending=False, inplace=True)
        return gs_counts
