from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import numpy as np

from phenotypic.abc_ import GridObjectRefiner
from phenotypic.measure import MeasureGridLinRegStats
from phenotypic.tools_.measurement_info_ import GRID_LINREG_STATS


class ReduceMultipleGridObjects(GridObjectRefiner):
    """Reduce multi-detections per grid cell by keeping the object closest to a linear-regression prediction.

    Models expected colony positions along each row and column using linear
    regression, then iteratively removes objects with the largest positional
    residuals until each grid cell contains at most one detection. Cells with
    the most objects are processed first to stabilize the regression fit.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` reduced to at most
        one object per grid cell based on minimum residual error.

    Best For:
        - Grid cells with multiple detections from halos, debris, or
          over-segmentation.
        - Condensation or glare artifacts that create extra detections near
          true colonies.
        - Pinned arrays where consistent spatial layout makes positional
          prediction reliable.

    Consider Also:
        - :class:`GridAlignmentRefiner` for faster dominant-object-per-cell
          selection without regression modeling.
        - :class:`GridSectionLargest` for a simpler largest-per-cell
          strategy.
        - :class:`ResidualOutlierRemover` for removing outliers within noisy
          rows or columns rather than reducing to one per cell.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for grid-based
        refinement workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        grid refinement approaches.
    """

    # TODO: Add a setting to retain a certain number of objects in the event of removal

    def _operate(self, image: GridImage) -> GridImage:
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
            min_err_obj_id = section_info.loc[
                :, str(GRID_LINREG_STATS.RESIDUAL_ERR)
            ].idxmin()

            # Isolate which objects within the section should be dropped
            objects_to_drop = section_info.index.drop(min_err_obj_id).to_numpy()

            # Set the objects with the labels to the background other_image
            image.objmap[np.isin(obj_map, objects_to_drop)] = 0

            # Reset section obj count and add counter
            section_obj_counts = image.grid.get_section_counts(ascending=False)
            n_iters += 1

        return image
