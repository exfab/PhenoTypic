from .. import Image
from ..interface import GridFeatureExtractor
from . import BoundaryExtractor
from ..filter import BorderObjectFilter

from typing import Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar


class GridSectionNodeExtractor(GridFeatureExtractor):
    def __init__(self, n_rows: int = 8, n_cols: int = 12, tol=1.0e-4):
        self._n_rows: int = n_rows
        self._n_cols: int = n_cols
        self.__tol = tol

        self._minus_rr_bound = self._plus_rr_bound = None
        self._minus_rr_mean = self._plus_rr_mean = None

        self._minus_cc_bound = self._plus_cc_bound = None
        self._minus_cc_mean = self._plus_cc_mean = None

    def _operate(self, image: Image) -> pd.DataFrame:
        # Find the centroid and boundaries

        image = BorderObjectFilter(border_size=1).filter(image)
        boundary_table = BoundaryExtractor().extract(image)

        grid_results_one = boundary_table.copy()

        gs_row_bins_one = np.histogram_bin_edges(
                a=grid_results_one.loc[:, 'center_rr'],
                bins=self._n_rows,
                range=(
                    grid_results_one.loc[:, 'min_rr'].min() - 1,
                    grid_results_one.loc[:, 'max_rr'].max() + 1
                )
        )
        grid_results_one.loc[:, 'grid_row_bin'] = pd.cut(
                grid_results_one.loc[:, 'center_rr'],
                bins=gs_row_bins_one,
                labels=range(self._n_rows)
        )

        gs_col_bins_one = np.histogram_bin_edges(
                a=grid_results_one.loc[:, 'center_cc'],
                bins=self._n_cols,
                range=(
                    grid_results_one.loc[:, 'min_cc'].min() - 1,
                    grid_results_one.loc[:, 'max_cc'].max() + 1
                )
        )
        grid_results_one.loc[:, 'grid_col_bin'] = pd.cut(
                grid_results_one.loc[:, 'center_cc'],
                bins=gs_col_bins_one,
                labels=range(self._n_cols)
        )

        # Second Pass
        self._minus_rr_mean = grid_results_one.loc[
            grid_results_one.loc[:, 'grid_row_bin'] == 0,
            'center_rr'
        ].mean()

        self._plus_rr_mean = grid_results_one.loc[
            grid_results_one.loc[:, 'grid_row_bin'] == self._n_rows - 1,
            'center_rr'
        ].mean()

        def optimal_row_bound_finder(padding_sz):
            _pred_bin = np.histogram_bin_edges(
                    a=boundary_table.loc[:, 'center_rr'],
                    bins=self._n_rows,
                    range=(
                        boundary_table.loc[:, 'min_rr'].min() - padding_sz,
                        boundary_table.loc[:, 'max_rr'].max() + padding_sz
                    )
            )
            _pred_bin = np.sort(_pred_bin)
            _lower_midpoint = (_pred_bin[1] - _pred_bin[0]) / 2 + _pred_bin[0]
            _upper_midpoint = (_pred_bin[-1] - _pred_bin.shape[-2]) / 2 + _pred_bin[-1]
            return (self._minus_rr_mean - _lower_midpoint)**2 + (self._plus_rr_mean - _upper_midpoint)**2

        max_pad_size = np.min(boundary_table.loc[:, 'min_rr'].min() - 1, boundary_table.loc[:, 'max_rr'].max() - 1)
        _optimal_row_padding = minimize_scalar(optimal_row_bound_finder, bounds=(0, max_pad_size))

        self._minus_cc_mean = grid_results_one.loc[
            grid_results_one.loc[:, 'grid_col_bin'] == 0,
            'center_cc'
        ].mean()

        self._plus_cc_mean = grid_results_one.loc[
            grid_results_one.loc[:, 'grid_col_bin'] == self._n_cols - 1,
            'center_cc'
        ].mean()


