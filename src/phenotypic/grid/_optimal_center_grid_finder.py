from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from functools import partial

from phenotypic.ABC_ import GridFinder
from phenotypic.tools.constants_ import BBOX, GRID


class OptimalBinsGridFinder(GridFinder):
    """
    Defines a class for finding the grid parameters based on the optimal center of object in a provided image.

    The OptimalCenterGridSetter class provides methods for setting up a grid on
    an image using row and column parameters, optimizing grid boundaries based on
    object centroids, and categorizing object based on their positions in grid
    sections. This class facilitates gridding for structured analysis, such as object
    segmentation or classification within images.

    Attributes:
        nrows (int): Number of nrows in the grid.
        ncols (int): Number of columns in the grid.
        tol (float): Tolerance for the solver method. Defaults to 10e-3.

    """
    __iter_limit = 1e5

    def __init__(self, nrows: int = 8, ncols: int = 12,
                 tol: float = 0.01, max_iter: int | None = None):
        """
        Represents a configuration object for iterative computations with constraints on
        the number of nrows, columns, tolerance, and a maximum number of iterations. This
        provides a flexible structure enabling adjustments to the computation parameters
        such as matrix dimensions and convergence criteria.

        Attributes:
            nrows (int): Number of nrows for the computation grid or array.
            ncols (int): Number of columns for the computation grid or array.
            tol (float): Tolerance level for the convergence criteria.
            max_iter (int | None): Maximum number of allowable iterations. Defaults to
                the predefined internal convergence limit if not provided.

        """
        super().__init__(nrows=nrows, ncols=ncols)

        self.tol: float = tol

        self.max_iter: int = max_iter if max_iter else self.__iter_limit

    def _operate(self, image: Image) -> pd.DataFrame:
        """
        Processes an arr image to calculate and organize grid-based boundaries and centroids using coordinates. This
        function implements a two-pass approach to refine row and column boundaries with exact precision, ensuring accurate
        grid labeling and indexing. The function dynamically computes boundary intervals and optimally segments the arr
        space into grids based on specified nrows and columns.

        Args:
            image (Image): The arr image to be analyzed and processed.

        Returns:
            pd.DataFrame: A DataFrame containing the grid results including boundary intervals, grid indices, and section
            numbers corresponding to the segmented arr image.
        """
        # Calculate optimal edges using optimization
        row_edges = self.get_row_edges(image)
        col_edges = self.get_col_edges(image)

        # Use base class helper to assemble complete grid info
        return super()._get_grid_info(image=image, row_edges=row_edges, col_edges=col_edges)

    def _find_padding_midpoint_error(self, pad_sz, image, axis, row_pad=0, col_pad=0) -> float:
        """
        Calculate the mean squared error between object midpoints and grid bin midpoints.
        
        Args:
            pad_sz: Padding size to test for the specified axis.
            image: Image object containing objects to be gridded.
            axis: 0 for rows, 1 for columns.
            row_pad: Current row padding (used when optimizing columns).
            col_pad: Current column padding (used when optimizing rows).
            
        Returns:
            float: Mean squared error between object and bin midpoints.
        """
        obj_info = image.objects.info(include_metadata=False)

        if axis == 0:
            # Calculate row edges with current padding
            row_edges = self._get_row_edges(image=image, row_padding=pad_sz, info_table=obj_info)
            col_edges = self._get_col_edges(image=image, column_padding=col_pad, info_table=obj_info)

            # Get grid info with these edges
            current_grid_info = super()._get_grid_info(image=image, row_edges=row_edges, col_edges=col_edges)
            current_obj_midpoints = (current_grid_info.loc[:, [str(BBOX.CENTER_RR), str(GRID.ROW_NUM)]]
                                     .groupby(str(GRID.ROW_NUM), observed=False)[str(BBOX.CENTER_RR)]
                                     .mean().values)

            bin_edges = np.histogram_bin_edges(
                    a=current_grid_info.loc[:, str(BBOX.CENTER_RR)].values,
                    bins=self.nrows,
                    range=(
                        current_grid_info.loc[:, str(BBOX.MIN_RR)].min() - pad_sz,
                        current_grid_info.loc[:, str(BBOX.MAX_RR)].max() + pad_sz
                    ),
            )

        elif axis == 1:
            # Calculate column edges with current padding
            row_edges = self._get_row_edges(image=image, row_padding=row_pad, info_table=obj_info)
            col_edges = self._get_col_edges(image=image, column_padding=pad_sz, info_table=obj_info)

            # Get grid info with these edges
            current_grid_info = super()._get_grid_info(image=image, row_edges=row_edges, col_edges=col_edges)
            current_obj_midpoints = (current_grid_info.loc[:, [str(BBOX.CENTER_CC), str(GRID.COL_NUM)]]
                                     .groupby(str(GRID.COL_NUM), observed=False)[str(BBOX.CENTER_CC)]
                                     .mean().values)

            bin_edges = np.histogram_bin_edges(
                    a=current_grid_info.loc[:, str(BBOX.CENTER_CC)].values,
                    bins=self.ncols,
                    range=(
                        current_grid_info.loc[:, str(BBOX.MIN_CC)].min() - pad_sz,
                        current_grid_info.loc[:, str(BBOX.MAX_CC)].max() + pad_sz
                    ),
            )
        else:
            raise ValueError(f"Invalid axis other_image: {axis}")

        bin_edges.sort()

        # (larger_point-smaller_point)/2 + smaller_point; Across all axis vectors
        larger_edges = bin_edges[1:]
        smaller_edges = bin_edges[:-1]
        bin_midpoint = (larger_edges - smaller_edges)//2 + smaller_edges

        return ((current_obj_midpoints - bin_midpoint) ** 2).sum()/len(current_obj_midpoints)

    def _get_optimal_row_pad(self, image: Image) -> int:
        """
        Determines the optimal row padding for the given image by analyzing the metadata of the
        detected objects and finding the maximum allowable padding that adheres to the constraints
        of the image shape.

        Uses the object information from the image to compute the padding range, which is derived
        from the minimum and maximum bounding box nrows of the detected objects. Clips the calculated
        padding size in case it results in a negative value.

        Args:
            image (Image): The image object containing detected objects and their associated metadata.

        Returns:
            int: The optimal row padding value based on the image's object information and calculated
            constraints.
        """
        obj_info = image.objects.info(include_metadata=False)
        min_rr, max_rr = obj_info.loc[:, str(BBOX.MIN_RR)].min(), obj_info.loc[:, str(BBOX.MAX_RR)].max()
        max_row_pad_size = min(min_rr - 1, abs(image.shape[0] - max_rr - 1))
        max_row_pad_size = 0 if max_row_pad_size < 0 else max_row_pad_size  # Clip in case pad size is negative

        partial_row_pad_finder = partial(self._find_padding_midpoint_error, image=image, axis=0, row_pad=0, col_pad=0)
        return int(self._apply_solver(partial_row_pad_finder, max_value=max_row_pad_size, min_value=0))

    def _get_row_edges(self, image: Image, row_padding: int, info_table: pd.DataFrame):
        """
        Determine the row edges of an image based on object positions and padding.

        This method calculates the edges defining nrows for objects within an image
        based on their positions provided in a DataFrame, applying padding and
        binning logic. The row edges are adjusted to fit within the boundaries
        of the image.

        Args:
            image (Image): The image where the row edges will be determined. The
                shape of the image is used to establish boundaries.
            row_padding (int): An additional padding applied to object bounds when
                calculating row edges.
            info_table (pd.DataFrame): A DataFrame containing object data, including
                their minimal and maximal row positions and central row coordinates.

        Returns:
            np.ndarray: An array of row edges sorted in ascending order.
        """
        lower_row_bound = round(info_table.loc[:, str(BBOX.MIN_RR)].min() - row_padding)
        upper_row_bound = round(info_table.loc[:, str(BBOX.MAX_RR)].max() + row_padding)
        obj_row_range = np.clip(
                a=[lower_row_bound, upper_row_bound],
                a_min=0, a_max=image.shape[0] - 1,
        )

        row_edges = np.histogram_bin_edges(
                a=info_table.loc[:, str(BBOX.CENTER_RR)],
                bins=self.nrows,
                range=tuple(obj_row_range),
        )
        np.round(a=row_edges, out=row_edges)
        row_edges.sort()

        return row_edges.astype(int)

    def get_row_edges(self, image: Image):
        """
        Extracts and returns the edges of nrows from the given image.

        This method first calculates the optimal row padding for the provided image
        using an internal utility method and subsequently determines the row edges
        based on the calculated padding and metadata of the image.

        Args:
            image (Image): The input image from which the row edges need to
                be identified.

        Returns:
            list: A list representing the edges of the nrows in the image.
        """
        optimal_row_padding = self._get_optimal_row_pad(image=image)
        return self._get_row_edges(
                image=image,
                row_padding=optimal_row_padding,
                info_table=image.objects.info(include_metadata=False),
        )

    def _get_optimal_col_pad(self, image: Image) -> int:
        obj_info = image.objects.info(include_metadata=False)
        min_cc, max_cc = obj_info.loc[:, str(BBOX.MIN_CC)].min(), obj_info.loc[:, str(BBOX.MAX_CC)].max()
        max_col_pad_size = min(min_cc - 1, abs(image.shape[1] - max_cc - 1))
        max_col_pad_size = 0 if max_col_pad_size < 0 else max_col_pad_size  # Clip in case pad size is negative

        partial_col_pad_finder = partial(self._find_padding_midpoint_error, image=image, axis=1, row_pad=0, col_pad=0)
        return self._apply_solver(partial_col_pad_finder, max_value=max_col_pad_size, min_value=0)

    def _get_col_edges(self, image: Image, column_padding: int, info_table: pd.DataFrame):
        lower_col_bound = round(info_table.loc[:, str(BBOX.MIN_CC)].min() - column_padding)
        upper_col_bound = round(info_table.loc[:, str(BBOX.MAX_CC)].max() + column_padding)
        obj_col_range = np.clip(
                a=[lower_col_bound, upper_col_bound],
                a_min=0, a_max=image.shape[1] - 1,
        )
        col_edges = np.histogram_bin_edges(
                a=info_table.loc[:, str(BBOX.CENTER_CC)],
                bins=self.ncols,
                range=tuple(obj_col_range),
        )
        np.round(a=col_edges, out=col_edges)
        col_edges.sort()

        return col_edges.astype(int)

    def get_col_edges(self, image: Image):
        optimal_col_padding = self._get_optimal_col_pad(image=image)
        return self._get_col_edges(
                image=image,
                column_padding=optimal_col_padding,
                info_table=image.objects.info(include_metadata=False),
        )

    def _apply_solver(self, partial_cost_func, max_value, min_value=0) -> int:
        """Returns the optimal padding other_image that minimizes the mean squared differences between the object midpoints and grid midpoints."""
        if max_value == 0:
            return 0

        else:
            return round(
                    minimize_scalar(partial_cost_func, bounds=(min_value, max_value),
                                    options={'maxiter': self.max_iter if self.max_iter else 1000,
                                             'xatol'  : self.tol},
                                    ).x,
            )


OptimalBinsGridFinder.measure.__doc__ = OptimalBinsGridFinder._operate.__doc__
