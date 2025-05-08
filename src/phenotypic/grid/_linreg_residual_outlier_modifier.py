from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING: from phenotypic import GridImage

import numpy as np
from typing import Optional

from phenotypic.abstract import GridMapModifier
from phenotypic.grid import MeasureGridLinRegStats
from phenotypic.util.constants_ import GRID, GRID_LINREG_STATS_EXTRACTOR


class LinRegResidualOutlierRemover(GridMapModifier):
    """
    Identifies and removes linear regression residual outliers in a grid-based image.

    This class is designed to analyze the variance of each row and column in a grid-based image.
    Rows or columns with variances exceeding a specified maximum threshold are analyzed further for
    residual outliers using linear regression. Objects with residuals above the defined cutoff
    (mean + standard deviation * multiplier) are considered outliers and removed. This is useful for
    removing noise that interferes with gridding

    Attributes:
        axis (Optional[int]): Axis to analyze for outliers. If None, both rows and columns are
            analyzed; 0 analyzes rows; 1 analyzes columns.
        stddev_multiplier (float): Multiplier to define the cutoff for residual error relative
            to the standard deviation. Higher values make the cutoff less strict.
        max_coeff_variance (int): Maximum coefficient of variance (standard deviation divided
            by mean) allowed for rows or columns before they are analyzed for outliers.
    """

    def __init__(self, axis: Optional[int] = None, stddev_multiplier=1.5, max_coeff_variance: int = 1):
        self.axis = axis  # Either none for both axis, 0 for row, or 1 for column
        self.stddev_multiplier = stddev_multiplier
        self.max_coeff_variance = max_coeff_variance

    def _operate(self, image: GridImage) -> GridImage:
        """
        Processes a GridImage to identify and remove row-wise and column-wise outlier
        objects based on residual errors and coefficient of variance.

        Outlier identification is performed by first calculating the coefficient of
        variance for each row or column. Rows or columns with variance above a specified
        threshold are considered for outlier detection. Within these rows or columns,
        objects with residual errors exceeding a computed cutoff based on standard
        deviation multiplier are identified and subsequently removed.

        Args:
            image (GridImage): The GridImage object that represents the input grid
                containing object information for analysis and modification.

        Returns:
            GridImage: The modified GridImage object with outlier objects removed.

        Raises:
            ValueError: If max_coeff_variance or stddev_multiplier attributes are not
                properly specified for the operation.
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
            row_variance = grid_info.groupby(GRID.GRID_ROW_NUM)[GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR].std()

            #   Divide standard deviation by mean
            row_variance = row_variance \
                           / grid_info.groupby(GRID.GRID_ROW_NUM)[GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR].mean()

            over_limit_row_variance = row_variance.loc[row_variance > self.max_coeff_variance]

            # Collect outlier objects in the rows with a variance over the maximum
            for row_idx in over_limit_row_variance.index:
                row_err = grid_info.loc[
                    grid_info.loc[:, GRID.GRID_ROW_NUM] == row_idx,
                    GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR
                ]
                row_err_mean = row_err.mean()
                row_stddev = row_err.std()

                upper_row_cutoff = row_err_mean + row_stddev * self.stddev_multiplier
                outlier_obj_ids += row_err.loc[row_err >= upper_row_cutoff].index.tolist()

        # Column-wise residual outlier discovery
        if self.axis is None or self.axis == 1:
            # Calculate the coefficient of variance (std/mean)
            #   Collect the standard deviation
            col_variance = grid_info.groupby(GRID.GRID_COL_NUM)[GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR].std()

            #   Divide standard deviation by mean
            col_variance = col_variance / grid_info.groupby(GRID.GRID_COL_NUM)[GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR].mean()

            over_limit_col_variance = col_variance.loc[col_variance > self.max_coeff_variance]

            # Collect outlier objects in the columns with a variance over the maximum
            for col_idx in over_limit_col_variance.index:
                col_err = grid_info.loc[
                    grid_info.loc[:, GRID.GRID_COL_NUM] == col_idx,
                    GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR
                ]
                col_err_mean = col_err.mean()
                col_stddev = col_err.std()

                upper_col_cutoff = col_err_mean + col_stddev * self.stddev_multiplier
                outlier_obj_ids += col_err.loc[col_err >= upper_col_cutoff].index.tolist()

        # Remove objects from obj map
        image.objmap[np.isin(image.objmap[:], outlier_obj_ids)] = 0

        return image
