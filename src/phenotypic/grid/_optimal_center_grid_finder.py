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
        method (str): The optimization method to use ('bisection' or 'solver'. Defaults to 'bisection'.
        tol (float): Tolerance for the solver method. Defaults to 10e-3.

    """

    def __init__(self, nrows: int = 8, ncols: int = 12, method: str = 'bisection', tol: float = 10e-3, max_iter: int | None = 1000):
        """Initializes the OptimalCenterGridSetter object.

        Args:
            nrows (int): number of rows in the grid
            ncols (int): number of columns in the grid
            method (str): The optimization method to use ('bisection' or 'solver'. Defaults to 'bisection'.)
            tol (float): Tolerance for the solver method. Defaults to 10e-3.
            max_iter (int|None): Maximum number of iterations for the solver method. Defaults to 1000. If None, the solver will run until convergence or 50000 iterations.
        """
        self.nrows: int = nrows
        self.ncols: int = ncols

        self.method: str = method
        if self.method not in ['bisection', 'solver']:
            raise ValueError(f"Invalid method value: {self.method}")

        self.tol: float = tol

        self.__convergence_limit = 50000
        self.max_iter: int = max_iter if max_iter else self.__convergence_limit

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
        obj_info = image.objects.info()

        # Find row padding search boundaries
        min_rr, max_rr = obj_info.loc[:, OBJECT_INFO.MIN_RR].min(), obj_info.loc[:, OBJECT_INFO.MAX_RR].max()
        max_row_pad_size = min(min_rr - 1, abs(image.shape[0] - max_rr - 1))
        max_row_pad_size = 0 if max_row_pad_size < 0 else max_row_pad_size  # Clip in case pad size is negative

        partial_row_pad_finder = partial(self._find_padding_midpoint_error, image=image, axis=0, row_pad=0, y_pad=0)
        optimal_row_padding = round(
            minimize_scalar(
                partial_row_pad_finder,
                bounds=(0, max_row_pad_size)
            ).x
        )

        # Column Padding

        ## Find column paddingsearch boundaries
        min_cc, max_cc = obj_info.loc[:, OBJECT_INFO.MIN_CC].min(), obj_info.loc[:, OBJECT_INFO.MAX_CC].max()
        max_col_pad_size = min(min_cc - 1, abs(image.shape[1] - max_cc - 1))
        max_col_pad_size = 0 if max_col_pad_size < 0 else max_col_pad_size  # Clip in case pad size is negative

        partial_col_pad_finder = partial(self._find_padding_midpoint_error, image=image, axis=1, row_pad=optimal_row_padding, y_pad=0)

        optimal_col_padding = round(
            minimize_scalar(
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
        np.round(a=col_edges, out=col_edges).astype(int)

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

    def _find_padding_midpoint_error(self, pad_sz, image, axis, row_pad=0, y_pad=0) -> float:
        """
        Finds the optimal padding value that minimizes the squared differences between
        the calculated midpoints of histogram bins and the provided grid group center means, while recalculating gridding each iteration.

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
        if axis == 0:
            current_grid_info = self._get_grid_info(image=image, row_padding=pad_sz, column_padding=y_pad)
            current_obj_midpoints = (current_grid_info.loc[:, [OBJECT_INFO.CENTER_RR, GRID.GRID_ROW_NUM]]
                                     .groupby(GRID.GRID_ROW_NUM, observed=False)
                                     .mean().values)

            bin_edges = np.histogram_bin_edges(
                a=current_grid_info.loc[:, OBJECT_INFO.CENTER_RR].values,
                bins=self.nrows,
                range=(
                    current_grid_info.loc[:, OBJECT_INFO.MIN_RR].min() - pad_sz,
                    current_grid_info.loc[:, OBJECT_INFO.MAX_RR].max() + pad_sz
                )
            )

        elif axis == 1:
            current_grid_info = self._get_grid_info(image=image, row_padding=row_pad, column_padding=pad_sz)
            current_obj_midpoints = (current_grid_info.loc[:, [OBJECT_INFO.CENTER_CC, GRID.GRID_COL_NUM]]
                                     .groupby(GRID.GRID_COL_NUM, observed=False)
                                     .mean().values)

            bin_edges = np.histogram_bin_edges(
                a=current_grid_info.loc[:, OBJECT_INFO.CENTER_CC].values,
                bins=self.ncols,
                range=(
                    current_grid_info.loc[:, OBJECT_INFO.MIN_CC].min() - pad_sz,
                    current_grid_info.loc[:, OBJECT_INFO.MAX_CC].max() + pad_sz
                )
            )
        else:
            raise ValueError(f"Invalid axis value: {axis}")

        bin_edges.sort()

        # (larger_point-smaller_point)/2 + smaller_point; Across all axis vectors
        larger_edges = bin_edges[1:]
        smaller_edges = bin_edges[:-1]
        bin_midpoint = (larger_edges - smaller_edges) // 2 + smaller_edges

        return ((current_obj_midpoints - bin_midpoint) ** 2).sum()

    def _apply_solver(self, partial_cost_func, max_value, min_value=0) -> int:
        """Returns the optimal padding value that minimizes the squared differences between the object midpoints and grid midpoints."""
        if max_value == 0:
            return 0

        elif self.method == 'solver':
            return round(minimize_scalar(partial_cost_func, bounds=(min_value, max_value),
                                         tol=self.tol, options={'maxiter': self.max_iter if self.max_iter else 1000}
                                         ).x
                         )

        elif self.method == 'bisection':
            return self._bisection_solver(partial_cost_func, max_value, min_value)

        else:
            raise AttributeError(f"Unknown error occured. Most likely from invalid method value: {self.method}")

    def _bisection_solver(self, partial_cost_func, max_value, min_value=0) -> int:
        """
        Finds the closest point to the zero error using a bisection algorithm that iterates with only integeres.

        This method attempts to minimize the error obtained from the 
        `partial_cost_func` by iteratively narrowing the range defined 
        by `min_value` and `max_value`. It computes midpoints within 
        the range to evaluate where the error is minimized closer 
        to zero.

        Args:
            partial_cost_func (Callable[[int], int]): A function that takes an integer input and 
                returns an integer representing the error.
            max_value (int): The upper bound of the range for the bisection algorithm.
            min_value (int, optional): The lower bound of the range for the bisection algorithm. 
                Defaults to 0.

        Returns:
            int: The closest point within the range where the error is minimized to zero.
        """
        left_bound, right_bound = min_value, max_value
        center = max_value // 2

        solver_iter = 0
        while solver_iter < self.__convergence_limit:
            if solver_iter > self.max_iter: raise RuntimeError("Bisection solver failed to converge.")
            center_err = partial_cost_func(center)

            left_midpoint, right_midpoint = self.__update_bisection_midpoints(center, left_bound, right_bound, min_value, max_value)
            left_midpoint_err, right_midpoint_err = partial_cost_func(left_midpoint), partial_cost_func(right_midpoint)
            if left_midpoint_err < right_midpoint_err:
                right_bound = center
                center = left_midpoint

                change_err = left_midpoint_err - center_err
                if change_err <= self.tol: return center
                
            elif left_midpoint_err > right_midpoint_err:
                left_bound = center
                center = right_midpoint

                change_err = right_midpoint_err - center_err
                if change_err <= self.tol: return center

            solver_iter += 1

        raise RuntimeError("Bisection solver failed to converge.")

    @staticmethod
    def __update_bisection_midpoints(center, left_bound, right_bound, min_value, max_value) -> (int, int):
        """Updates the lower and upper midpoints of the bisection search.
        
        Returns:
            (int, int): The updated lower and upper midpoints respectively.
        
        """
        left_midpoint = (center - left_bound) // 2 + left_bound
        left_midpoint = left_midpoint if left_midpoint >= min_value else min_value

        right_midpoint = (right_bound - center) // 2 + center
        right_midpoint = right_midpoint if right_midpoint <= max_value else max_value

        return left_midpoint, right_midpoint


OptimalCenterGridFinder.measure.__doc__ = OptimalCenterGridFinder._operate.__doc__
