from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

import numpy as np
from typing import Optional

from phenotypic.abc_ import GridRefiner
from phenotypic.measure import MeasureGridLinRegStats
from phenotypic.tools.constants_ import GRID_LINREG_STATS_EXTRACTOR, GRID


class ResidualOutlierRemover(GridRefiner):
    """Remove objects with large regression residuals in noisy grid rows/columns.

    Intuition:
        In grid assays, colony centroids should align along near-linear trends
        within each row/column. Rows or columns with high variability suggest
        mis-detections or artifacts. Within such noisy lines, this operation
        removes objects whose positional residuals exceed a robust cutoff.

    Why this is useful for agar plates:
        Condensation, glare, and debris can produce off-grid detections that
        inflate row/column variance and break gridding assumptions. Pruning
        residual outliers restores alignment and improves subsequent measures.

    Use cases:
        - Cleaning rows with multiple off-line blobs before measuring growth.
        - Stabilizing grid registration when a subset of positions is noisy.

    Caveats:
        - If true colonies deviate due to warping or growth spreading, strict
          cutoffs may remove real data.
        - Depends on reasonable initial grid fit; with severe misregistration
          it may prune valid colonies.

    Attributes:
        axis (Optional[int]): Axis to analyze for outliers. ``None`` analyzes
            both rows and columns; ``0`` analyzes rows; ``1`` analyzes columns.
            Restricting the axis can speed up processing or focus on suspected
            directions of error.
        cutoff_multiplier (float): Multiplier applied to a robust dispersion
            estimate (IQR-based in implementation) to set the outlier cutoff.
            Higher values are more permissive (fewer removals) and preserve
            edge cases; lower values prune more aggressively.
        max_coeff_variance (int): Maximum coefficient of variance (std/mean)
            allowed for a row/column before it is considered for outlier
            pruning. Smaller values trigger cleaning sooner; larger values only
            clean severely noisy lines.

    Examples:
        >>> from phenotypic.refine import ResidualOutlierRemover
        >>> op = ResidualOutlierRemover(axis=None, stddev_multiplier=1.5, max_coeff_variance=1)
        >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
    """

    def __init__(self, axis: Optional[int] = None, stddev_multiplier=1.5, max_coeff_variance: int = 1):
        """Initialize the remover.

        Args:
            axis (Optional[int]): Axis selection for analysis. ``None`` runs
                both directions; ``0`` rows; ``1`` columns. Limiting the axis
                reduces runtime and targets known problem directions.
            stddev_multiplier (float): Robust residual cutoff multiplier. Lower
                values remove more outliers (stronger cleanup) but risk dropping
                valid off-center colonies; higher values are conservative.
            max_coeff_variance (int): Threshold for row/column variability
                (std/mean) to trigger outlier analysis. Lower values clean more
                lines; higher values only address extremely noisy lines.

        Raises:
            ValueError: If parameters are not consistent with the operation
                (e.g., invalid types). Errors may arise during execution when
                measuring grid statistics.
        """
        self.axis = axis  # Either none for both axis, 0 for row, or 1 for column
        self.cutoff_multiplier = stddev_multiplier
        self.max_coeff_variance = max_coeff_variance

    def _operate(self, image: GridImage) -> GridImage:
        """Identify and remove residual outliers per noisy row/column.

        Args:
            image (GridImage): Grid image with object map and grid metadata.

        Returns:
            GridImage: Modified grid image with outlier objects removed.

        Raises:
            ValueError: If parameters are misconfigured in a way that prevents
                computation (propagated from measurement utilities).
        """
        # Generate cached version of grid_info
        linreg_stat_extractor = MeasureGridLinRegStats()
        grid_info = linreg_stat_extractor.measure(image)

        # Create container to hold the id of objects to be removed
        outlier_obj_ids = []

        # Row-wise residual outlier discovery
        if self.axis is None or self.axis == 0:
            # Calculate the coefficient of variance (std/mean)
            #   Collect the standard deviation
            row_variance = grid_info.groupby(str(GRID.ROW_NUM))[GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR].std()

            #   Divide standard deviation by mean
            row_variance = row_variance \
                           /grid_info.groupby(str(GRID.ROW_NUM))[GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR].mean()

            over_limit_row_variance = row_variance.loc[row_variance > self.max_coeff_variance]

            # Collect outlier objects in the nrows with a variance over the maximum
            for row_idx in over_limit_row_variance.index:
                row_err = grid_info.loc[
                    grid_info.loc[:, str(GRID.ROW_NUM)] == row_idx,
                    GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR
                ]
                row_err_mean = row_err.mean()
                row_q3, row_q1 = row_err.quantile([0.75, 0.25])
                row_iqr = row_q3 - row_q1

                # row_stddev = row_err.std()
                # upper_row_cutoff = row_err_mean + row_stddev * self.cutoff_multiplier

                upper_row_cutoff = row_err_mean + row_iqr*self.cutoff_multiplier
                outlier_obj_ids += row_err.loc[row_err >= upper_row_cutoff].index.tolist()

        # Column-wise residual outlier discovery
        if self.axis is None or self.axis == 1:
            # Calculate the coefficient of variance (std/mean)
            #   Collect the standard deviation
            col_variance = grid_info.groupby(str(GRID.COL_NUM))[GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR].std()

            #   Divide standard deviation by mean
            col_variance = col_variance/grid_info.groupby(str(GRID.COL_NUM))[
                GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR].mean()

            over_limit_col_variance = col_variance.loc[col_variance > self.max_coeff_variance]

            # Collect outlier objects in the columns with a variance over the maximum
            for col_idx in over_limit_col_variance.index:
                col_err = grid_info.loc[
                    grid_info.loc[:, str(GRID.COL_NUM)] == col_idx,
                    GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR
                ]
                col_err_mean = col_err.mean()
                col_q3, col_q1 = col_err.quantile([0.75, 0.25])
                col_iqr = col_q3 - col_q1
                # col_stddev = col_err.std()

                upper_col_cutoff = col_err_mean + col_iqr*self.cutoff_multiplier
                outlier_obj_ids += col_err.loc[col_err >= upper_col_cutoff].index.tolist()

        # Remove objects from obj map
        image.objmap[np.isin(image.objmap[:], outlier_obj_ids)] = 0

        return image
