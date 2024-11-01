from phenoscope import Image
from phenoscope.grid.interface import GridExtractor
from phenoscope.feature_extraction import BoundaryExtractor

from typing import Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar


class OptimalCenterGridExtractor(GridExtractor):

    def __init__(self, n_rows: int = 8, n_cols: int = 12):
        self.n_rows: int = n_rows
        self.n_cols: int = n_cols

        self._minus_rr_bound = self._plus_rr_bound = None
        self._minus_rr_mean = self._plus_rr_mean = None

        self._minus_cc_bound = self._plus_cc_bound = None
        self._minus_cc_mean = self._plus_cc_mean = None

    def _operate(self, image: Image) -> pd.DataFrame:
        # Find the centroid and boundaries
        bound_extractor = BoundaryExtractor()
        boundary_table = bound_extractor.extract(image)

        grid_results_one = boundary_table.copy()

        # Generate row bins
        gs_row_bins_one = np.histogram_bin_edges(
                a=grid_results_one.loc[:, bound_extractor.LABEL_CENTER_RR],
                bins=self.n_rows,
                range=(
                    grid_results_one.loc[:, bound_extractor.LABEL_MIN_RR].min() - 1,
                    grid_results_one.loc[:, bound_extractor.LABEL_MAX_RR].max() + 1
                )
        )
        grid_results_one.loc[:, self.LABEL_GRID_ROW_NUM] = pd.cut(
                grid_results_one.loc[:, bound_extractor.LABEL_CENTER_RR],
                bins=gs_row_bins_one,
                labels=range(self.n_rows)
        )

        # Generate column bins
        gs_col_bins_one = np.histogram_bin_edges(
                a=grid_results_one.loc[:, bound_extractor.LABEL_CENTER_CC],
                bins=self.n_cols,
                range=(
                    grid_results_one.loc[:, bound_extractor.LABEL_MIN_CC].min() - 1,
                    grid_results_one.loc[:, bound_extractor.LABEL_MAX_CC].max() + 1
                )
        )
        grid_results_one.loc[:, self.LABEL_GRID_COL_NUM] = pd.cut(
                grid_results_one.loc[:, bound_extractor.LABEL_CENTER_CC],
                bins=gs_col_bins_one,
                labels=range(self.n_cols)
        )

        # Find optimal row padding
        self._minus_rr_mean = grid_results_one.loc[
            grid_results_one.loc[:, self.LABEL_GRID_ROW_NUM] == 0,
            bound_extractor.LABEL_CENTER_RR
        ].mean()

        self._plus_rr_mean = grid_results_one.loc[
            grid_results_one.loc[:, self.LABEL_GRID_ROW_NUM] == self.n_rows - 1,
            bound_extractor.LABEL_CENTER_RR
        ].mean()

        def optimal_row_bound_finder(padding_sz):
            _pred_bin = np.histogram_bin_edges(
                    a=boundary_table.loc[:, bound_extractor.LABEL_CENTER_RR],
                    bins=self.n_rows,
                    range=(
                        boundary_table.loc[:, bound_extractor.LABEL_MIN_RR].min() - padding_sz,
                        boundary_table.loc[:, bound_extractor.LABEL_MAX_RR].max() + padding_sz
                    )
            )
            _pred_bin.sort()
            _lower_midpoint = (_pred_bin[1] - _pred_bin[0]) / 2 + _pred_bin[0]
            _upper_midpoint = (_pred_bin[-1] - _pred_bin[-2]) / 2 + _pred_bin[-2]
            return (self._minus_rr_mean - _lower_midpoint) + (self._plus_rr_mean - _upper_midpoint)

        max_row_pad_size = min(abs(boundary_table.loc[:, bound_extractor.LABEL_MIN_RR].min() - 1),
                               abs(image.shape[0] - boundary_table.loc[:, bound_extractor.LABEL_MAX_RR].max()))
        optimal_row_padding = minimize_scalar(optimal_row_bound_finder, bounds=(0, max_row_pad_size)).x

        # Find optimal col boundaries
        self._minus_cc_mean = grid_results_one.loc[
            grid_results_one.loc[:, self.LABEL_GRID_COL_NUM] == 0,
            bound_extractor.LABEL_CENTER_CC
        ].mean()

        self._plus_cc_mean = grid_results_one.loc[
            grid_results_one.loc[:, self.LABEL_GRID_COL_NUM] == self.n_cols - 1,
            bound_extractor.LABEL_CENTER_CC
        ].mean()

        def optimal_col_bound_finder(padding_sz):
            _pred_bin = np.histogram_bin_edges(
                    a=boundary_table.loc[:, bound_extractor.LABEL_CENTER_CC],
                    bins=self.n_cols,
                    range=(
                        boundary_table.loc[:, bound_extractor.LABEL_MIN_CC].min() - padding_sz,
                        boundary_table.loc[:, bound_extractor.LABEL_MAX_CC].max() + padding_sz
                    )
            )
            _pred_bin.sort()
            _lower_midpoint = (_pred_bin[1] - _pred_bin[0]) / 2 + _pred_bin[0]
            _upper_midpoint = (_pred_bin[-1] - _pred_bin[-2]) / 2 + _pred_bin[-2]
            return (self._minus_cc_mean - _lower_midpoint) + (self._plus_cc_mean - _upper_midpoint)

        max_col_pad_size = min(abs(boundary_table.loc[:, bound_extractor.LABEL_MIN_CC].min() - 1),
                               abs(image.shape[1] - boundary_table.loc[:, bound_extractor.LABEL_MAX_CC].max()))
        optimal_col_padding = minimize_scalar(optimal_col_bound_finder, bounds=(0, max_col_pad_size)).x

        # begin second pass
        grid_results_two = boundary_table.copy()

        # Generate new row bins
        gs_row_bins_two = np.histogram_bin_edges(
                a=grid_results_two.loc[:, bound_extractor.LABEL_CENTER_RR],
                bins=self.n_rows,
                range=(
                    int(grid_results_two.loc[:, bound_extractor.LABEL_MIN_RR].min() - optimal_row_padding),
                    int(grid_results_two.loc[:, bound_extractor.LABEL_MAX_RR].max() + optimal_row_padding)
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
        grid_results_two.loc[:, self.LABEL_GRID_ROW_NUM] = pd.cut(
                grid_results_two.loc[:, bound_extractor.LABEL_CENTER_RR],
                bins=gs_row_bins_two,
                labels=range(self.n_rows)

        )
        grid_results_two.loc[:, self.LABEL_GRID_ROW_INTERVAL] = pd.cut(
                grid_results_two.loc[:, bound_extractor.LABEL_CENTER_RR],
                bins=gs_row_bins_two,
                labels=row_intervals
        )

        # generate new col bins
        gs_col_bins_two = np.histogram_bin_edges(
                a=grid_results_two.loc[:, bound_extractor.LABEL_CENTER_CC],
                bins=self.n_cols,
                range=(
                    grid_results_two.loc[:, bound_extractor.LABEL_MIN_CC].min() - optimal_col_padding,
                    grid_results_two.loc[:, bound_extractor.LABEL_MAX_CC].max() + optimal_col_padding
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
        grid_results_two.loc[:, self.LABEL_GRID_COL_NUM] = pd.cut(
                grid_results_two.loc[:, bound_extractor.LABEL_CENTER_CC],
                bins=gs_col_bins_two,
                labels=range(self.n_cols)
        )
        grid_results_two.loc[:, self.LABEL_GRID_COL_INTERVAL] = pd.cut(
                grid_results_two.loc[:, bound_extractor.LABEL_CENTER_CC],
                bins=gs_col_bins_two,
                labels=col_intervals
        )

        # Add section indexes
        grid_results_two.loc[:, self.LABEL_GRID_SECTION_IDX] = list(zip(
                grid_results_two.loc[:, self.LABEL_GRID_ROW_NUM],
                grid_results_two.loc[:, self.LABEL_GRID_COL_NUM]
        ))

        # Add section numbers
        for num, idx in enumerate(grid_results_two.loc[:, self.LABEL_GRID_SECTION_IDX].unique()):
            grid_results_two.loc[grid_results_two.loc[:, self.LABEL_GRID_SECTION_IDX] == idx, self.LABEL_GRID_SECTION_NUM] = int(num)

        # Reduce memory consumption with categorical labels
        grid_results_two.loc[:, self.LABEL_GRID_SECTION_IDX] = grid_results_two.loc[:, self.LABEL_GRID_SECTION_IDX].astype('category')
        grid_results_two[self.LABEL_GRID_SECTION_NUM] = grid_results_two[self.LABEL_GRID_SECTION_NUM].astype(int).astype('category')

        return grid_results_two
