from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

import numpy as np

from phenotypic.abc_ import GridRefiner
from phenotypic.measure import MeasureGridLinRegStats
from phenotypic.tools.constants_ import GRID_LINREG_STATS_EXTRACTOR


class MinResidualErrorReducer(GridRefiner):
    """Reduce multi-detections per grid cell by keeping objects closest to a
    linear-regression prediction of expected positions.

    Intuition:
        In grid assays, some cells contain multiple detections due to halos,
        debris, or over-segmentation. By modeling the expected colony position
        in each row/column with linear regression, we can retain the object with
        the smallest residual error (closest to the predicted location) and
        remove the rest. This iterates across the grid until each cell is
        simplified.

    Why this is useful for agar plates:
        Pinned arrays assume consistent spatial layout. Selecting the object
        nearest the learned grid trend eliminates off-grid artifacts while
        preserving the most plausible colony per cell.

    Use cases:
        - Over-segmentation yields several blobs per grid position.
        - Condensation or glare introduces extra detections near a colony.

    Caveats:
        - If the grid fit is inaccurate (bad registration, warped plates), the
          closest-to-trend object may not be the true colony.
        - Relatively slow due to repeated measurement and iteration over cells.

    Attributes:
        (No public attributes)

    Examples:
        >>> from phenotypic.refine import MinResidualErrorReducer
        >>> op = MinResidualErrorReducer()
        >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
    """

    # TODO: Add a setting to retain a certain number of objects in the event of removal

    @staticmethod
    def _operate(image: GridImage) -> GridImage:
        # Get the section objects in order of most amount. More objects in a section means
        # more potential spread that can affect linreg results.
        max_iter = (image.grid.nrows*image.grid.ncols)*4

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

            # Set the objects with the labels to the background other_image
            image.objmap[np.isin(obj_map, objects_to_drop)] = 0

            # Reset section obj count and add counter
            section_obj_counts = image.grid.get_section_counts(ascending=False)
            n_iters += 1

        return image
