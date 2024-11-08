import numpy as np

from phenoscope.grid.interface import GridMapModifier
from phenoscope.grid import GriddedImage


class GridOversizedObjectRemover(GridMapModifier):
    def _operate(self, image: GriddedImage) -> GriddedImage:
        row_edges = image.grid_row_edges
        col_edges = image.grid_col_edges
        grid_info = image.grid_info

        # To simplify calculation use the max width & distance
        max_width = max(col_edges[1:] - col_edges[:-1])
        max_height = max(row_edges[1:] - row_edges[:-1])

        # Calculate the width and height of each object
        grid_info.loc[:, 'width'] = grid_info.loc[:, image.bound_extractor.LABEL_MAX_CC] \
                                    - grid_info.loc[:, image.bound_extractor.LABEL_MIN_CC]

        grid_info.loc[:, 'height'] = grid_info.loc[:, image.bound_extractor.LABEL_MAX_RR] \
                                     - grid_info.loc[:, image.bound_extractor.LABEL_MIN_RR]

        # Find objects that are past the max height & width
        over_width_obj = grid_info.loc[grid_info.loc[:,'width'] >= max_width, :].index.tolist()

        over_height_obj = grid_info.loc[grid_info.loc[:,'height'] >= max_height, :].index.tolist()

        # Create a numpy array with the objects to be removed
        obj_to_remove = np.array(over_width_obj + over_height_obj)

        # Create a working copy of the object map
        obj_map = image.object_map

        # Set the target objects to the background val of 0
        obj_map[np.isin(obj_map, obj_to_remove)] = 0

        # Set the Image objects object_map to the new map
        image.object_map = obj_map
        return image
