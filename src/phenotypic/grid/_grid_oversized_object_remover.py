from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING: from phenotypic import GridImage

import numpy as np

from phenotypic.abstract import GridMapModifier
from phenotypic.util.constants_ import OBJECT_INFO


class GridOversizedObjectRemover(GridMapModifier):
    """
    Removes oversized objects from grid-based image data.

    This class inherits from `GridMapModifier` and is designed to remove objects from the
    grid-based image representation that exceed the maximum allowable width or height of the
    grid cells. The removal process sets the oversized object regions to the background value
    of 0. This class is useful for preprocessing grid images for further analysis or visualization.
    """
    def _operate(self, image: GridImage) -> GridImage:
        """
        Applies operations on the given GridImage to remove objects based on maximum width and height constraints.

        This method processes the grid metadata of a `GridImage` object to identify objects
        that exceed the maximum calculated width and height. It sets such objects to a
        background value of 0 in the object's mapping array. This helps filter out undesired
        large objects in the image.

        Args:
            image (GridImage): The input grid image containing grid metadata and object map.

        Returns:
            GridImage: The processed grid image with specified objects removed.
        """
        row_edges = image.grid.get_row_edges()
        col_edges = image.grid.get_col_edges()
        grid_info = image.grid.info()

        # To simplify calculation use the max width & distance
        max_width = max(col_edges[1:] - col_edges[:-1])
        max_height = max(row_edges[1:] - row_edges[:-1])

        # Calculate the width and height of each object
        grid_info.loc[:, 'width'] = grid_info.loc[:, OBJECT_INFO.MAX_CC] \
                                    - grid_info.loc[:, OBJECT_INFO.MIN_CC]

        grid_info.loc[:, 'height'] = grid_info.loc[:, OBJECT_INFO.MAX_RR] \
                                     - grid_info.loc[:, OBJECT_INFO.MIN_RR]

        # Find objects that are past the max height & width
        over_width_obj = grid_info.loc[grid_info.loc[:, 'width'] >= max_width, :].index.tolist()

        over_height_obj = grid_info.loc[grid_info.loc[:, 'height'] >= max_height, :].index.tolist()

        # Create a numpy array with the objects to be removed
        obj_to_remove = np.array(over_width_obj + over_height_obj)

        # Set the target objects to the background val of 0
        image.objmap[np.isin(image.objmap[:], obj_to_remove)] = 0

        return image
