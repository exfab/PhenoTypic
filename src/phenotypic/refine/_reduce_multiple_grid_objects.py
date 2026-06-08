from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import numpy as np

from phenotypic.abc_ import GridObjectRefiner
from phenotypic.measure import MeasureGridLinRegStats
from phenotypic.schema import GRID_LINREG_STATS


class ReduceSectionsByLine(GridObjectRefiner):
    """Reduce multi-detections per grid cell to one by keeping the object best predicted by linear regression.

    Fits linear trends to colony centroids along each row and column, then
    iteratively removes objects with the largest positional residuals until
    every grid cell contains at most one detection. Grid cells with the most
    objects are processed first to stabilize the regression fit before
    resolving cells with fewer extra detections.

    Requires a ``GridImage`` with grid metadata already populated.

    For an overview of grid refinement strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Grid cells with multiple detections from halos, condensation, or
          agar debris that shadow nearby true colonies.
        - Pinned arrays where a consistent spatial layout makes positional
          prediction reliable.
        - Over-segmented detections where one cell contains a genuine colony
          plus one or more satellite artefacts.

    Consider Also:
        - :class:`GridAlignmentRefiner` for a faster dominant-object-per-cell
          approach that does not require iterative regression.
        - :class:`KeepSectionLargest` for a simpler strategy that keeps only
          the largest object per grid cell without position modeling.
        - :class:`RemoveGridOutliers` when the goal is pruning positional
          outliers within noisy rows or columns rather than enforcing one
          detection per cell.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` reduced so that
        each grid cell contains at most one object, selected by minimum
        linear-regression residual error.

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
