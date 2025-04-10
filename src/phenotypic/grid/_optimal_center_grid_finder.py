from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from functools import partial

from phenotypic.abstract import GridFinder
from phenotypic.util.constants_ import OBJECT_INFO, GRID


class OptimalCenterGridFinder(GridFinder):
    """
    Defines a class for finding the grid parameters based on optimal center of objects in a provided image.

    The OptimalCenterGridSetter class provides methods for setting up a grid on
    an image using row and column parameters, optimizing grid boundaries based on
    object centroids, and categorizing objects based on their positions in grid
    sections. This class facilitates gridding for structured analysis, such as object
    segmentation or classification within images.

    Attributes:
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.

    """

    def __init__(self, nrows: int = 8, ncols: int = 12):
        """Initializes the OptimalCenterGridSetter object.

        Args:
            nrows (int): number of rows in the grid
            ncols (int): number of columns in the grid
        """
        self.nrows: int = nrows
        self.ncols: int = ncols

    def _operate(self, image: Image) -> pd.DataFrame:
        """
        Processes an input image to calculate and organize grid-based boundaries and centroids using coordinates. This
        function implements a two-pass approach to refine row and column boundaries with exact precision, ensuring accurate
        grid labeling and indexing. The function dynamically computes boundary intervals and optimally segments the input
        space into grids based on specified rows and columns.

        Args:
            image (Image): The input image to be analyzed and processed.

        Returns:
            pd.DataFrame: A DataFrame containing the grid results including boundary intervals, grid indices, and section
            numbers corresponding to the segmented input image.
        """
        # Find the centroid and boundaries
        initial_grid_results = self._get_grid_info(image=image, row_padding=0, column_padding=0)

        # Calculate the smallest distance between obj bound min and the edge of the image
        max_row_pad_size = min(
            abs(initial_grid_results.loc[:, OBJECT_INFO.MIN_RR].min()),
            abs(image.shape[0] - initial_grid_results.loc[:, OBJECT_INFO.MAX_RR].max())
        )

        first_row_group = initial_grid_results.loc[initial_grid_results.loc[:, GRID.GRID_ROW_NUM] == 0, OBJECT_INFO.CENTER_RR]
        last_row_group = initial_grid_results.loc[initial_grid_results.loc[:, GRID.GRID_ROW_NUM] == self.nrows - 1, OBJECT_INFO.CENTER_RR]
        partial_row_pad_finder = partial(self._optimal_pad_finder,
                                         centerpoint_array=initial_grid_results.loc[:, OBJECT_INFO.CENTER_RR].values,
                                         num_bins=self.nrows,
                                         overall_bound_min=initial_grid_results.loc[:, OBJECT_INFO.MIN_RR].min(),
                                         overall_bound_max=initial_grid_results.loc[:, OBJECT_INFO.MAX_RR].max(),
                                         first_grid_group_center_mean=first_row_group.mean(),
                                         last_grid_group_center_mean=last_row_group.mean()
                                         )
        optimal_row_padding = round(minimize_scalar(
            partial_row_pad_finder,
            bounds=(0, max_row_pad_size)
        ).x
                                    )

        # Get column padding
        max_col_pad_size = min(
            abs(initial_grid_results.loc[:, OBJECT_INFO.MIN_CC].min()),
            abs(image.shape[1] - initial_grid_results.loc[:, OBJECT_INFO.MAX_CC].max())
        )

        first_col_group = initial_grid_results.loc[initial_grid_results.loc[:, GRID.GRID_COL_NUM] == 0, OBJECT_INFO.CENTER_CC]
        last_col_group = initial_grid_results.loc[initial_grid_results.loc[:, GRID.GRID_COL_NUM] == self.ncols - 1, OBJECT_INFO.CENTER_CC]
        partial_col_pad_finder = partial(self._optimal_pad_finder,
                                         centerpoint_array=initial_grid_results.loc[:, OBJECT_INFO.CENTER_CC].values,
                                         num_bins=self.ncols,
                                         overall_bound_min=initial_grid_results.loc[:, OBJECT_INFO.MIN_CC].min(),
                                         overall_bound_max=initial_grid_results.loc[:, OBJECT_INFO.MAX_CC].max(),
                                         first_grid_group_center_mean=first_col_group.mean(),
                                         last_grid_group_center_mean=last_col_group.mean())
        optimal_col_padding = round(minimize_scalar(
            partial_col_pad_finder,
            bounds=(0, max_col_pad_size)
        ).x
                                    )
        return self._get_grid_info(image=image, row_padding=optimal_row_padding, column_padding=optimal_col_padding)

    def _get_grid_info(self, image: Image, row_padding: int = 0, column_padding: int = 0) -> pd.DataFrame:
        info_table = image.objects.info()

        # Grid Rows
        lower_row_bound = round(info_table.loc[:, OBJECT_INFO.MIN_RR].min() - row_padding)
        upper_row_bound = round(info_table.loc[:, OBJECT_INFO.MAX_RR].max() + row_padding)
        obj_row_range = np.clip(
            a=[lower_row_bound, upper_row_bound],
            a_min=0, a_max=image.shape[0] - 1,
        )

        row_edges = np.histogram_bin_edges(
            a=info_table.loc[:, OBJECT_INFO.CENTER_RR],
            bins=self.nrows,
            range=tuple(obj_row_range)
        )
        np.round(a=row_edges, out=row_edges).astype(int)
        row_edges.sort()

        # Add row number info
        info_table.loc[:, GRID.GRID_ROW_NUM] = pd.cut(
            info_table.loc[:, OBJECT_INFO.CENTER_RR],
            bins=row_edges,
            labels=range(self.nrows),
            include_lowest=True,
            right=True
        )

        # Add row interval info
        info_table.loc[:, GRID.GRID_ROW_INTERVAL] = pd.cut(
            info_table.loc[:, OBJECT_INFO.CENTER_RR],
            bins=row_edges,
            labels=[(row_edges[i], row_edges[i + 1]) for i in range(len(row_edges) - 1)],
            include_lowest=True,
            right=True
        )

        # Grid Columns
        lower_col_bound = round(info_table.loc[:, OBJECT_INFO.MIN_CC].min() - column_padding)
        upper_col_bound = round(info_table.loc[:, OBJECT_INFO.MAX_CC].max() + column_padding)
        obj_col_range = np.clip(
            a=[lower_col_bound, upper_col_bound],
            a_min=0, a_max=image.shape[1] - 1,
        )
        col_edges = np.histogram_bin_edges(
            a=info_table.loc[:, OBJECT_INFO.CENTER_CC],
            bins=self.ncols,
            range=obj_col_range
        )

        # Add column number info
        info_table.loc[:, GRID.GRID_COL_NUM] = pd.cut(
            info_table.loc[:, OBJECT_INFO.CENTER_CC],
            bins=col_edges,
            labels=range(self.ncols),
            include_lowest=True,
            right=True
        )

        # Add column interval info
        info_table.loc[:, GRID.GRID_COL_INTERVAL] = pd.cut(
            info_table.loc[:, OBJECT_INFO.CENTER_CC],
            bins=col_edges,
            labels=[(col_edges[i], col_edges[i + 1]) for i in range(len(col_edges) - 1)],
            include_lowest=True,
            right=True
        )

        # Grid Section Info
        info_table.loc[:, GRID.GRID_SECTION_IDX] = list(zip(
            info_table.loc[:, GRID.GRID_ROW_NUM],
            info_table.loc[:, GRID.GRID_COL_NUM]
        )
        )

        idx_map = np.reshape(np.arange(self.nrows * self.ncols), newshape=(self.nrows, self.ncols))
        for idx in np.sort(np.unique(info_table.loc[:, GRID.GRID_SECTION_IDX].values)):
            info_table.loc[info_table.loc[:, GRID.GRID_SECTION_IDX] == idx, GRID.GRID_SECTION_NUM] = idx_map[idx[0], idx[1]]

        # Reduce memory consumption with categorical labels
        info_table.loc[:, GRID.GRID_SECTION_IDX] = info_table.loc[:, GRID.GRID_SECTION_IDX].astype('category')
        info_table[GRID.GRID_SECTION_NUM] = info_table[GRID.GRID_SECTION_NUM].astype(int).astype('category')

        return info_table

    @staticmethod
    def _optimal_pad_finder(pad_sz,
                            centerpoint_array,
                            num_bins: int,
                            overall_bound_min: float, overall_bound_max: float,
                            first_grid_group_center_mean: float, last_grid_group_center_mean: float) -> float:
        """
        Finds the optimal padding value that minimizes the squared differences between
        the calculated midpoints of histogram bins and the provided grid group center means.

        Args:
            pad_sz (float): Padding size to be evaluated.
            centerpoint_array (np.ndarray): Array containing the center points of the grid groups.
            num_bins (int): Number of bins to use in the histogram calculation.
            overall_bound_min (float): Minimum bound of the overall grid range.
            overall_bound_max (float): Maximum bound of the overall grid range.
            first_grid_group_center_mean (float): Mean center of the first grid group.
            last_grid_group_center_mean (float): Mean center of the last grid group.

        Returns:
            float: The squared sum of differences between expected and calculated midpoints.

        """
        bins = np.histogram_bin_edges(
            a=centerpoint_array,
            bins=num_bins,
            range=(
                overall_bound_min - pad_sz,
                overall_bound_max + pad_sz
            )
        )
        bins.sort()

        # (larger_point-smaller_point)/2 + smaller_point
        lower_midpoint = (bins[1] - bins[0]) / 2 + bins[0]
        upper_midpoint = (bins[-1] - bins[-2]) / 2 + bins[-2]
        return (first_grid_group_center_mean - lower_midpoint) ** 2 + (last_grid_group_center_mean - upper_midpoint) ** 2


OptimalCenterGridFinder.measure.__doc__ = OptimalCenterGridFinder._operate.__doc__
