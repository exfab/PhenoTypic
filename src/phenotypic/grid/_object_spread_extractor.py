from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING: from phenotypic import GridImage
from phenotypic.abstract import GridFeatureMeasure

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix


class ObjectSpreadExtractor(GridFeatureMeasure):
    """
    This module measure's an objects spread from the grid section's center points
    """

    def _operate(self, image: GridImage) -> pd.DataFrame:
        gs_table = image.grid.info()
        gs_counts = pd.DataFrame(gs_table.loc[:, 'grid_section_bin'].value_counts())

        obj_spread = []
        for gs_bindex in gs_counts.index:
            curr_gs_subtable = gs_table.loc[gs_table.loc[:, 'grid_section_bin'] == gs_bindex, :]

            x_vector = curr_gs_subtable.loc[:, 'center_cc']
            y_vector = curr_gs_subtable.loc[:, 'center_rr']
            obj_vector = np.array(list(zip(x_vector, y_vector)))
            gs_distance_matrix = distance_matrix(x=obj_vector, y=obj_vector, p=2)

            obj_spread.append(np.sum(np.unique(gs_distance_matrix) ** 2))
        gs_counts.insert(loc=1, column='object_spread', value=obj_spread)
        gs_counts.sort_values(by='object_spread', ascending=False, inplace=True)
        return gs_counts
