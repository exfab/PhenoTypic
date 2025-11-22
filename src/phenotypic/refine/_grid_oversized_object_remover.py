from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

import numpy as np

from phenotypic.abc_ import GridRefiner
from phenotypic.tools.constants_ import BBOX, OBJECT


class GridOversizedObjectRemover(GridRefiner):
    """Remove objects that are larger than their grid cell allows.

    Intuition:
        In pinned colony grids, each cell should contain at most one colony of
        limited extent. Objects spanning nearly an entire cell width/height are
        often merged colonies, agar edges, or segmentation spillover. Removing
        these oversized objects improves the reliability of per-cell analysis.

    Why this is useful for agar plates:
        Grid-based assays assume spatial confinement. Oversized detections
        disrupt row/column statistics and bias growth comparisons. Filtering
        them stabilizes downstream measurements.

    Use cases:
        - Dropping merged blobs that span adjacent positions.
        - Removing strong edge artifacts near the plate rim that intrude into
          the grid.

    Caveats:
        - If genuine colonies expand to fill the cell (late time points), this
          remover may exclude real growth.
        - Highly irregular grids or mis-registered edges can cause over-
          removal; ensure grid metadata is accurate.

    Attributes:
        (No public attributes)

    Examples:
        >>> from phenotypic.refine import GridOversizedObjectRemover
        >>> op = GridOversizedObjectRemover()
        >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
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
        grid_info.loc[:, 'width'] = grid_info.loc[:, str(BBOX.MAX_CC)] \
                                    - grid_info.loc[:, str(BBOX.MIN_CC)]

        grid_info.loc[:, 'height'] = grid_info.loc[:, str(BBOX.MAX_RR)] \
                                     - grid_info.loc[:, str(BBOX.MIN_RR)]

        # Find objects that are past the max height & width
        over_width_obj = grid_info.loc[:, 'width'] >= max_width

        over_height_obj = grid_info.loc[:, 'height'] >= max_height
        oversized_obj_labels = grid_info.loc[
            over_width_obj | over_height_obj,
            OBJECT.LABEL
        ].unique()

        # Set the target objects to the background val of 0
        image.objmap[np.isin(image.objmap[:], oversized_obj_labels)] = 0

        return image
