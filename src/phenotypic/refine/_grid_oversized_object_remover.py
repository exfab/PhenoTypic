from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import numpy as np

from phenotypic.abc_ import GridObjectRefiner
from phenotypic.schema import OBJECT
from phenotypic.schema import BBOX


class GridOversizedObjectRemover(GridObjectRefiner):
    """Discard objects whose bounding box equals or exceeds the maximum grid cell span.

    Measures each detected object's bounding-box width and height and removes
    any object that meets or exceeds the largest cell dimension in the grid.
    This eliminates merged colonies spanning adjacent grid positions, agar-rim
    intrusions, and segmentation spillover that the grid layout guarantees are
    not single confined colonies.

    For an overview of grid refinement strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Pinned colony grids (96-well, 384-well) where valid colonies should
          fit within a single cell and oversized objects are always artefacts.
        - Post-detection cleanup when detector outputs contain merged blobs
          spanning two or more adjacent grid positions.
        - Removing strong edge or rim artefacts that intrude into the grid area
          and produce abnormally large detected regions.

    Consider Also:
        - :class:`KeepSectionLargest` when keeping the single largest valid
          object per cell is preferable to removing oversized candidates.
        - :class:`GridAlignmentRefiner` for full grid-aware dominant-object
          selection with configurable strategies.
        - :class:`SmallObjectRemover` when the problem is undersized debris
          rather than oversized merged detections.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` updated to exclude
        objects whose bounding box equals or exceeds the maximum grid cell size.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for grid-based
        refinement workflows on real plate images.
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
