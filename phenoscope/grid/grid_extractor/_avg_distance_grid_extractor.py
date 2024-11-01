from phenoscope import Image
from phenoscope.grid.interface import GridExtractor
from phenoscope.feature_extraction import BoundaryExtractor
from phenoscope.map_modification import BorderObjectModifier

from typing import Optional
import pandas as pd
import numpy as np


# TODO: Update accesor names in class
class AvgDistanceGridSectionExtractor(GridExtractor):
    def __init__(self, n_rows: int = 8, n_cols: int = 12):
        self.n_rows: int = n_rows
        self.n_cols: int = n_cols

    def _operate(self, image: Image) -> pd.DataFrame:
        # Find the centroid and boundaries

        image = BorderObjectModifier(border_size=1).modify(image)
        boundary_table = BoundaryExtractor().extract(image)

        grid_results_one = boundary_table.copy()

        gs_row_bins_one = np.histogram_bin_edges(
                a=grid_results_one.loc[:, 'center_rr'],
                bins=self.n_rows,
                range=(
                    grid_results_one.loc[:, 'min_rr'].min() - 1,
                    grid_results_one.loc[:, 'max_rr'].max() + 1
                )
        )
        grid_results_one.loc[:, 'grid_row_bin'] = pd.cut(
                grid_results_one.loc[:, 'center_rr'],
                bins=gs_row_bins_one,
                labels=range(self.n_rows)
        )

        gs_col_bins_one = np.histogram_bin_edges(
                a=grid_results_one.loc[:, 'center_cc'],
                bins=self.n_cols,
                range=(
                    grid_results_one.loc[:, 'min_cc'].min() - 1,
                    grid_results_one.loc[:, 'max_cc'].max() + 1
                )
        )
        grid_results_one.loc[:, 'grid_col_bin'] = pd.cut(
                grid_results_one.loc[:, 'center_cc'],
                bins=gs_col_bins_one,
                labels=range(self.n_cols)
        )

        # Find Average Row Distances
        row_bindexes = grid_results_one.loc[:, 'grid_row_bin'].unique().sort_values(ascending=True)
        row_obj_distances = np.zeros(len(row_bindexes) - 1)
        for i in range(len(row_bindexes) - 1):
            curr_row = grid_results_one.loc[grid_results_one.loc[:, 'grid_row_bin'] == row_bindexes[i], 'max_rr']
            curr_avg_rr = curr_row.mean()

            next_row = grid_results_one.loc[grid_results_one.loc[:, 'grid_row_bin'] == row_bindexes[i + 1], 'min_rr']
            next_avg_rr = next_row.mean()

            row_obj_distances[i] = next_avg_rr - curr_avg_rr

        gridrow_obj_midpoint = int(np.mean(row_obj_distances) / 2)

        overall_min_rr = grid_results_one.loc[:, 'min_rr'].min()  # The uppermost pixel of all objects
        min_rr_gap = overall_min_rr - 0  # The amount of space between the top of the image to the nearest object

        overall_max_rr = grid_results_one.loc[:, 'max_rr'].max()  # The lowermost pixel of all objects
        max_rr_gap = image.shape[0] - overall_max_rr  # The amount of space between the bottom of the image to the nearest object

        # Find Average Column Distances
        col_bindexes = grid_results_one.loc[:, 'grid_col_bin'].unique().sort_values(ascending=True)
        col_obj_distances = np.zeros(len(col_bindexes) - 1)
        for i in range(len(col_bindexes) - 1):
            curr_col = grid_results_one.loc[grid_results_one.loc[:, 'grid_col_bin'] == col_bindexes[i], 'max_cc']
            curr_avg_cc = curr_col.mean()

            next_col = grid_results_one.loc[grid_results_one.loc[:, 'grid_col_bin'] == col_bindexes[i + 1], 'min_cc']
            next_avg_cc = next_col.mean()

            col_obj_distances[i] = next_avg_cc - curr_avg_cc

        gridcol_obj_midpoint = abs(int(np.mean(col_obj_distances) / 2))

        overall_min_cc = grid_results_one.loc[:, 'min_cc'].min()  # The leftmost pixel of all objects
        min_cc_gap = overall_min_cc - 0  # The amount of space between the left of the image to the nearest object

        overall_max_cc = grid_results_one.loc[:, 'max_cc'].max()  # The rightmost pixel of all objects
        max_cc_gap = image.shape[1] - overall_max_cc  # The amount of space between the right of the image to the nearest object

        grid_results_two = boundary_table.copy()
        # grid_padding = abs(min(gridcol_obj_midpoint, gridrow_obj_midpoint))

        # Choose safe row padding size
        gridrow_padding = abs(min(min_rr_gap - 1, max_rr_gap - 1, gridrow_obj_midpoint))
        row_range = (
            int(grid_results_two.loc[:, 'min_rr'].min() - gridrow_padding),
            int(grid_results_two.loc[:, 'max_rr'].max() + gridrow_padding)
        )

        # Add grid row indices
        gs_row_bins_two = np.histogram_bin_edges(
                a=grid_results_two.loc[:, 'center_rr'],
                bins=self.n_rows,
                range=row_range
        )
        np.round(a=gs_row_bins_two, out=gs_row_bins_two)
        grid_results_two.loc[:, 'grid_row_bin'] = pd.cut(
                grid_results_two.loc[:, 'center_rr'],
                bins=gs_row_bins_two,
                labels=range(self.n_rows)
        )

        # Add row intervals
        row_intervals = []
        for i in range(len(gs_row_bins_two) - 1):
            row_intervals.append(
                    pd.Interval(left=gs_row_bins_two[i], right=gs_row_bins_two[i + 1], closed='right')
            )

        grid_results_two.loc[:, 'grid_row_intervals'] = pd.cut(
                grid_results_two.loc[:, 'center_rr'],
                bins=gs_row_bins_two,
                labels=row_intervals
        )

        # Choose safe column padding size
        gridcol_padding = abs(min(min_cc_gap - 1, max_cc_gap - 1, gridcol_obj_midpoint))
        col_range = (
            int(grid_results_two.loc[:, 'min_cc'].min() - gridcol_padding),
            int(grid_results_two.loc[:, 'max_cc'].max() + gridcol_padding)
        )

        # Add grid column indices
        gs_col_bins_two = np.histogram_bin_edges(
                a=grid_results_two.loc[:, 'center_cc'],
                bins=self.n_cols,
                range=col_range
        )
        np.round(a=gs_col_bins_two, out=gs_col_bins_two)
        grid_results_two.loc[:, 'grid_col_bin'] = pd.cut(
                grid_results_two.loc[:, 'center_cc'],
                bins=gs_col_bins_two,
                labels=range(self.n_cols)
        )

        # Add column intervals
        col_intervals = []
        for i in range(len(gs_col_bins_two) - 1):
            col_intervals.append(
                    pd.Interval(left=gs_col_bins_two[i], right=gs_col_bins_two[i + 1], closed='right')
            )

        grid_results_two.loc[:, 'grid_col_intervals'] = pd.cut(
                grid_results_two.loc[:, 'center_cc'],
                bins=gs_col_bins_two,
                labels=col_intervals
        )

        grid_results_two.loc[:, 'grid_section_bin'] = list(zip(
                grid_results_two.loc[:, 'grid_row_bin'],
                grid_results_two.loc[:, 'grid_col_bin']
        ))
        grid_results_two.loc[:, 'grid_section_bin'] = grid_results_two.loc[:, 'grid_section_bin'].astype('category')

        return grid_results_two
