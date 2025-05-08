from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING: from phenotypic import GridImage

import numpy as np

from phenotypic.abstract import GridMapModifier
from phenotypic.grid import MeasureGridLinRegStats
from phenotypic.util.constants_ import GRID_LINREG_STATS_EXTRACTOR


class MinResidualErrorReducer(GridMapModifier):
    """
    This map modifier reduces the amount of objects in sections where there are multiple based on their distance from the linreg predicted location.
    This modifier is relatively slow, but shows good results in removing the correct obj when paired with small object removers and other filters.
    """

    # TODO: Add a setting to retain a certain number of objects in the event of removal

    def _operate(self, image: GridImage) -> GridImage:
        # Get the section objects in order of most amount. More objects in a section means
        # more potential spread that can affect linreg results.
        max_iter = (image.grid.nrows * image.grid.ncols) * 4

        # Initialize extractor here to save obj construction time
        linreg_stat_extractor = MeasureGridLinRegStats()

        # Get initial section obj count
        section_obj_counts = image.grid.get_section_counts(ascending=False)

        n_iters = 0
        # Check that there exist sections with more than one object
        while n_iters < max_iter and (section_obj_counts > 1).any():
            # Get the current object map. This is inside the loop to ensure latest version each iteration
            obj_map = image.objmap[:]

            # Get the section idx with the most objects
            section_with_most_obj = section_obj_counts.idxmax()

            # Set the target_section for linreg_stat_extractor
            linreg_stat_extractor.section_num = section_with_most_obj

            # Get the section info
            section_info = linreg_stat_extractor.measure(image)

            # Isolate the object id with the smallest residual error
            min_err_obj_id = section_info.loc[:, GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR].idxmin()

            # Isolate which objects within the section should be dropped
            objects_to_drop = section_info.index.drop(min_err_obj_id).to_numpy()

            # Set the objects with the labels to the background value
            image.objmap[np.isin(obj_map, objects_to_drop)] = 0


            # Reset section obj count and add counter
            section_obj_counts = image.grid.get_section_counts(ascending=False)
            n_iters += 1

        return image
