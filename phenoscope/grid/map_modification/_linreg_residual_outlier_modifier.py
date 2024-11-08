import numpy as np
from typing import Optional

from phenoscope.grid import GriddedImage
from phenoscope.grid.interface import GridMapModifier
from phenoscope.grid.feature_extraction import GridLinRegStatsExtractor


class LinRegResidualOutlierModifier(GridMapModifier):
    def __init__(self, axis: Optional[int] = None, stddev_multiplier=1.5, max_coeff_variance: int = 1):
        """
        This operation measures the variance of each column and row of the grid in the image. If the amount of variance surpasses the maximum,
        the operation then measures the row or column for linear regression residual outliers and removes any objects with a residual above the
        mean + standard deviation * multiplier.

        :param axis: (Optional[int])
        :param stddev_multiplier:
        :param ddof: (int) Delta degrees of freedom for the variance calculation
        :param max_coeff_variance: (int) The maximum value the coefficient of variance a row or column can have before it will be analyzed for residual outlier removal.
        """
        self.axis = axis  # Either none for both axis, 0 for row, or 1 for column
        self.stddev_multiplier = stddev_multiplier
        self.max_coeff_variance = max_coeff_variance

    def _operate(self, image: GriddedImage) -> GriddedImage:
        """
        Removes the objects with the above cutoff from the mean of error. The lower cutoff is kept because those objects are not likely noise.
        :param image:
        :return:
        """
        # TODO: Finish Implementation

        # Generate cached version of grid_info
        linreg_stat_extractor = GridLinRegStatsExtractor()
        grid_info = linreg_stat_extractor.extract(image)

        # Create container to hold the id of objects to be removed
        outlier_obj_ids = []

        # Row-wise residual outlier discovery
        if self.axis is None or self.axis == 0:

            # Calculate the coefficient of variance (std/mean)
            #   Collect the standard deviation
            row_variance = grid_info.groupby(image.grid_extractor.LABEL_GRID_ROW_NUM)[linreg_stat_extractor.LABEL_RESIDUAL_ERR].std()

            #   Divide standard deviation by mean
            row_variance = row_variance \
                           / grid_info.groupby(image.grid_extractor.LABEL_GRID_ROW_NUM)[linreg_stat_extractor.LABEL_RESIDUAL_ERR].mean()

            over_limit_row_variance = row_variance.loc[row_variance > self.max_coeff_variance]

            # Collect outlier objects in the rows with a variance over the maximum
            for row_idx in over_limit_row_variance.index:
                row_err = grid_info.loc[
                    grid_info.loc[:, image.grid_extractor.LABEL_GRID_ROW_NUM] == row_idx,
                    linreg_stat_extractor.LABEL_RESIDUAL_ERR
                ]
                row_err_mean = row_err.mean()
                row_stddev = row_err.std()

                upper_row_cutoff = row_err_mean + row_stddev * self.stddev_multiplier
                outlier_obj_ids += row_err.loc[row_err >= upper_row_cutoff].index.tolist()

        # Column-wise residual outlier discovery
        if self.axis is None or self.axis == 1:

            # Calculate the coefficient of variance (std/mean)
            #   Collect the standard deviation
            col_variance = grid_info.groupby(image.grid_extractor.LABEL_GRID_COL_NUM)[linreg_stat_extractor.LABEL_RESIDUAL_ERR].std()

            #   Divide standard deviation by mean
            col_variance = col_variance \
                           / grid_info.groupby(image.grid_extractor.LABEL_GRID_COL_NUM)[linreg_stat_extractor.LABEL_RESIDUAL_ERR].mean()

            over_limit_col_variance = col_variance.loc[col_variance > self.max_coeff_variance]

            # Collect outlier objects in the columns with a variance over the maximum
            for col_idx in over_limit_col_variance.index:
                col_err = grid_info.loc[
                    grid_info.loc[:, image.grid_extractor.LABEL_GRID_COL_NUM] == col_idx,
                    linreg_stat_extractor.LABEL_RESIDUAL_ERR
                ]
                col_err_mean = col_err.mean()
                col_stddev = col_err.std()

                upper_col_cutoff = col_err_mean + col_stddev * self.stddev_multiplier
                outlier_obj_ids += col_err.loc[col_err >= upper_col_cutoff].index.tolist()

        # Remove objects from obj map
        obj_map = image.object_map
        obj_map[np.isin(obj_map, outlier_obj_ids)] = 0

        image.object_map = obj_map
        return image
