from .. import Image
from ..interface import GridFeatureExtractor
from . import BoundaryExtractor

from typing import Optional
import pandas as pd
import numpy as np


class GridSectionExtractor(GridFeatureExtractor):
    def __init__(self, n_rows: int = 8, n_cols: int = 12):
        self._n_rows: int = n_rows
        self._n_cols: int = n_cols

    def _operate(self, image: Image) -> pd.DataFrame:
        # Find the centroid and boundaries
        results = BoundaryExtractor().extract(image)

        gs_row_bins = np.histogram_bin_edges(
                a=results.loc[:,'center_rr'],
                bins=self._n_rows,
                range=(
                    results.loc[:,'min_rr'].min(),
                    results.loc[:,'max_rr'].max()
                )
        )
        results.loc[:,'grid_row_intervals'] = pd.cut(
                results.loc[:, 'center_rr'],
                bins=gs_row_bins,
        )
        results.loc[:, 'grid_row_bin'] = pd.cut(
                results.loc[:, 'center_rr'],
                bins=gs_row_bins,
                labels=range(self._n_rows)
        )

        gs_col_bins = np.histogram_bin_edges(
                a=results.loc[:,'center_cc'],
                bins=self._n_cols,
                range=(
                    results.loc[:,'min_cc'].min(),
                    results.loc[:,'max_cc'].max()
                )
        )
        results.loc[:, 'grid_col_intervals'] = pd.cut(
                results.loc[:, 'center_cc'],
                bins=gs_col_bins,
        )
        results.loc[:, 'grid_col_bin'] = pd.cut(
                results.loc[:, 'center_cc'],
                bins=gs_col_bins,
                labels=range(self._n_cols)
        )



        results.loc[:, 'grid_section_bin'] = list(zip(
                results.loc[:, 'grid_row_bin'],
                results.loc[:, 'grid_col_bin']
        ))

        results.loc[:, 'grid_section_bin'] = results.loc[:, 'grid_section_bin'].astype('category')

        return results