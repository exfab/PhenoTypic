from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING: from phenoscope import Image

from phenoscope.grid.abstract import GridFinder
from phenoscope.measure import BoundaryExtractor

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

from phenoscope.util.constants_ import OBJECT_INFO, GRID


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

        self._minus_rr_bound = self._plus_rr_bound = None
        self._minus_rr_mean = self._plus_rr_mean = None

        self._minus_cc_bound = self._plus_cc_bound = None
        self._minus_cc_mean = self._plus_cc_mean = None

    def _operate(self, image: Image) -> pd.DataFrame:
        """

        Args:
            image:

        Returns:
            pd.DataFrame: A dataframe with the objects gridding information with the following column labels:
                (ObjectLabel, Bbox_CenterRR, BboxCenterCC, Bbox_MinRR, Bbox_MinCC, Bbox_MaxRR, Bbox_MaxCC,
                Grid_RowNum, Grid_RowInterval, Grid_ColNum, Grid_ColInterval, Grid_SectionNum, Grid_SectionInterval)

        """
        # Find the centroid and boundaries
        bound_extractor = BoundaryExtractor()
        boundary_table = bound_extractor.measure(image)

        grid_results_one = boundary_table.copy()

        # Generate row bins
        gs_row_bins_one = np.histogram_bin_edges(
            a=grid_results_one.loc[:, OBJECT_INFO.CENTER_RR],
            bins=self.nrows,
            range=(
                grid_results_one.loc[:, OBJECT_INFO.MIN_RR].min() - 1,
                grid_results_one.loc[:, OBJECT_INFO.MAX_RR].max() + 1
            )
        )
        grid_results_one.loc[:, GRID.GRID_ROW_NUM] = pd.cut(
            grid_results_one.loc[:, OBJECT_INFO.CENTER_RR],
            bins=gs_row_bins_one,
            labels=range(self.nrows)
        )

        # Generate column bins
        gs_col_bins_one = np.histogram_bin_edges(
            a=grid_results_one.loc[:, OBJECT_INFO.CENTER_CC],
            bins=self.ncols,
            range=(
                grid_results_one.loc[:, OBJECT_INFO.MIN_CC].min() - 1,
                grid_results_one.loc[:, OBJECT_INFO.MAX_CC].max() + 1
            )
        )
        grid_results_one.loc[:, GRID.GRID_COL_NUM] = pd.cut(
            grid_results_one.loc[:, OBJECT_INFO.CENTER_CC],
            bins=gs_col_bins_one,
            labels=range(self.ncols)
        )

        # Find optimal row padding
        self._minus_rr_mean = grid_results_one.loc[
            grid_results_one.loc[:, GRID.GRID_ROW_NUM] == 0,
            OBJECT_INFO.CENTER_RR
        ].mean()

        self._plus_rr_mean = grid_results_one.loc[
            grid_results_one.loc[:, GRID.GRID_ROW_NUM] == self.nrows - 1,
            OBJECT_INFO.CENTER_RR
        ].mean()

        # Set up equation for solver
        def optimal_row_bound_finder(padding_sz):
            _pred_bin = np.histogram_bin_edges(
                a=boundary_table.loc[:, OBJECT_INFO.CENTER_RR],
                bins=self.nrows,
                range=(
                    boundary_table.loc[:, OBJECT_INFO.MIN_RR].min() - padding_sz,
                    boundary_table.loc[:, OBJECT_INFO.MAX_RR].max() + padding_sz
                )
            )
            _pred_bin.sort()
            _lower_midpoint = (_pred_bin[1] - _pred_bin[0]) / 2 + _pred_bin[0]
            _upper_midpoint = (_pred_bin[-1] - _pred_bin[-2]) / 2 + _pred_bin[-2]
            return (self._minus_rr_mean - _lower_midpoint) ** 2 + (self._plus_rr_mean - _upper_midpoint) ** 2

        max_row_pad_size = min(abs(boundary_table.loc[:, OBJECT_INFO.MIN_RR].min() - 1),
                               abs(image.shape[0] - boundary_table.loc[:, OBJECT_INFO.MAX_RR].max())
                               )
        optimal_row_padding = minimize_scalar(optimal_row_bound_finder, bounds=(0, max_row_pad_size)).x

        # Find optimal col boundaries
        self._minus_cc_mean = grid_results_one.loc[
            grid_results_one.loc[:, GRID.GRID_COL_NUM] == 0,
            OBJECT_INFO.CENTER_CC
        ].mean()

        self._plus_cc_mean = grid_results_one.loc[
            grid_results_one.loc[:, GRID.GRID_COL_NUM] == self.ncols - 1,
            OBJECT_INFO.CENTER_CC
        ].mean()

        # Set up equation for solver
        def optimal_col_bound_finder(padding_sz):
            _pred_bin = np.histogram_bin_edges(
                a=boundary_table.loc[:, OBJECT_INFO.CENTER_CC],
                bins=self.ncols,
                range=(
                    boundary_table.loc[:, OBJECT_INFO.MIN_CC].min() - padding_sz,
                    boundary_table.loc[:, OBJECT_INFO.MAX_CC].max() + padding_sz
                )
            )
            _pred_bin.sort()
            _lower_midpoint = (_pred_bin[1] - _pred_bin[0]) / 2 + _pred_bin[0]
            _upper_midpoint = (_pred_bin[-1] - _pred_bin[-2]) / 2 + _pred_bin[-2]
            return (self._minus_cc_mean - _lower_midpoint) ** 2 + (self._plus_cc_mean - _upper_midpoint) ** 2

        max_col_pad_size = min(abs(boundary_table.loc[:, OBJECT_INFO.MIN_CC].min() - 1),
                               abs(image.shape[1] - boundary_table.loc[:, OBJECT_INFO.MAX_CC].max())
                               )
        optimal_col_padding = minimize_scalar(optimal_col_bound_finder, bounds=(0, max_col_pad_size)).x

        # begin second pass
        grid_results_two = boundary_table.copy()

        # Generate new row bins
        gs_row_bins_two = np.histogram_bin_edges(
            a=grid_results_two.loc[:, OBJECT_INFO.CENTER_RR],
            bins=self.nrows,
            range=(
                int(grid_results_two.loc[:, OBJECT_INFO.MIN_RR].min() - optimal_row_padding),
                int(grid_results_two.loc[:, OBJECT_INFO.MAX_RR].max() + optimal_row_padding)
            )
        )
        np.round(a=gs_row_bins_two, out=gs_row_bins_two)
        gs_row_bins_two.sort()

        row_intervals = []
        for i in range(len(gs_row_bins_two) - 1):
            row_intervals.append(
                (gs_row_bins_two[i], gs_row_bins_two[i + 1])
            )

        # Add row grid results
        grid_results_two.loc[:, GRID.GRID_ROW_NUM] = pd.cut(
            grid_results_two.loc[:, OBJECT_INFO.CENTER_RR],
            bins=gs_row_bins_two,
            labels=range(self.nrows)

        )
        grid_results_two.loc[:, GRID.GRID_ROW_INTERVAL] = pd.cut(
            grid_results_two.loc[:, OBJECT_INFO.CENTER_RR],
            bins=gs_row_bins_two,
            labels=row_intervals
        )

        # generate new col bins
        gs_col_bins_two = np.histogram_bin_edges(
            a=grid_results_two.loc[:, OBJECT_INFO.CENTER_CC],
            bins=self.ncols,
            range=(
                grid_results_two.loc[:, OBJECT_INFO.MIN_CC].min() - optimal_col_padding,
                grid_results_two.loc[:, OBJECT_INFO.MAX_CC].max() + optimal_col_padding
            ),
        )
        np.round(gs_col_bins_two, out=gs_col_bins_two)
        gs_col_bins_two.sort()

        col_intervals = []
        for i in range(len(gs_col_bins_two) - 1):
            col_intervals.append(
                (gs_col_bins_two[i], gs_col_bins_two[i + 1])
            )

        # Add col results
        grid_results_two.loc[:, GRID.GRID_COL_NUM] = pd.cut(
            grid_results_two.loc[:, OBJECT_INFO.CENTER_CC],
            bins=gs_col_bins_two,
            labels=range(self.ncols)
        )
        grid_results_two.loc[:, GRID.GRID_COL_INTERVAL] = pd.cut(
            grid_results_two.loc[:, OBJECT_INFO.CENTER_CC],
            bins=gs_col_bins_two,
            labels=col_intervals
        )

        # Add section indexes
        grid_results_two.loc[:, GRID.GRID_SECTION_IDX] = list(zip(
            grid_results_two.loc[:, GRID.GRID_ROW_NUM],
            grid_results_two.loc[:, GRID.GRID_COL_NUM]
        )
        )

        idx_map = np.arange(self.nrows * self.ncols)
        idx_map = np.reshape(idx_map, (self.nrows, self.ncols))

        # Add section numbers
        for num, idx in enumerate(np.sort(grid_results_two.loc[:, GRID.GRID_SECTION_IDX].unique())):
            grid_results_two.loc[grid_results_two.loc[:, GRID.GRID_SECTION_IDX] == idx, GRID.GRID_SECTION_NUM] = idx_map[idx]

        # Reduce memory consumption with categorical labels
        grid_results_two.loc[:, GRID.GRID_SECTION_IDX] = grid_results_two.loc[:, GRID.GRID_SECTION_IDX].astype('category')
        grid_results_two[GRID.GRID_SECTION_NUM] = grid_results_two[GRID.GRID_SECTION_NUM].astype(int).astype('category')

        return grid_results_two
