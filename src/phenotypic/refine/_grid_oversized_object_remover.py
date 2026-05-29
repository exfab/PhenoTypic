from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import numpy as np

from phenotypic.abc_ import GridObjectRefiner
from phenotypic.tools_.constants_ import OBJECT
from phenotypic.tools_.measurement_info import BBOX


class GridOversizedObjectRemover(GridObjectRefiner):
    """Remove objects whose bounding box exceeds the maximum grid cell dimension.

    Compares each object's width and height against the largest cell span in
    the grid and discards objects that match or exceed it. Eliminates merged
    colonies, agar rim intrusions, and segmentation spillover that span
    entire grid cells.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` updated to exclude
        objects exceeding the grid cell size.

    Best For:
        - Dropping merged blobs that span adjacent grid positions.
        - Removing strong edge artifacts near the plate rim that intrude
          into the grid.
        - Post-detection cleanup on pinned colony grids where each cell
          should contain one confined colony.

    Consider Also:
        - :class:`KeepSectionLargest` when you want to keep the single
          largest object per cell rather than removing oversized ones.
        - :class:`GridAlignmentRefiner` for full grid-aware dominant-object
          selection.
        - :class:`SmallObjectRemover` when the problem is undersized debris
          rather than oversized detections.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for grid-based
        refinement workflows.
        :doc:`/explanation/refinement_strategies` for an overview of
        grid refinement strategies.
    """

    def _operate(self, image: GridImage) -> GridImage:
        """
        Applies operations on the given GridImage to remove objects based on maximum width and height constraints.

        This method processes the grid metadata of a `GridImage` object to identify objects
        that exceed the maximum calculated width and height. It sets such objects to a
        background value of 0 in the object's mapping array. This helps filter out undesired
        large objects in the image.

        Args:
            image (GridImage): The arr grid image containing grid metadata and object map.

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
        grid_info.loc[:, "width"] = (
                grid_info.loc[:, str(BBOX.MAX_CC)] - grid_info.loc[:, str(BBOX.MIN_CC)]
        )

        grid_info.loc[:, "height"] = (
                grid_info.loc[:, str(BBOX.MAX_RR)] - grid_info.loc[:, str(BBOX.MIN_RR)]
        )

        # Find objects that are past the max height & width
        over_width_obj = grid_info.loc[:, "width"] >= max_width

        over_height_obj = grid_info.loc[:, "height"] >= max_height
        oversized_obj_labels = grid_info.loc[
            over_width_obj | over_height_obj, OBJECT.LABEL
        ].unique()

        # Set the target objects to the background val of 0
        image.objmap[np.isin(image.objmap[:], oversized_obj_labels)] = 0

        return image
