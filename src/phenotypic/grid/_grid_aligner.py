from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize_scalar

from phenotypic.abstract import GridCorrector
from phenotypic.util.constants_ import OBJECT_INFO, GRID


class GridAligner(GridCorrector):
    """Calculates the optimal gridding orientation based on the alignment of the objects in the image and rotates the image accordingly.

    This class inherits from `GridCorrector` and is designed to calculate the optimal gridding orientation. This is used to align the image,
    and helps to improve the quality of automated gridding results. It's highly recommended to redetect objects in the image after alignment.

    """

    def __init__(self, axis: int = 0, mode: str = 'edge'):
        self.axis = axis
        self.mode = mode

    def _operate(self, image: GridImage):
        """Calculates the optimal rotation angle and applies it to a grid image for alignment along the specified axis.

        The method performs alignment of a `GridImage` object along either rows or columns based on the specified
        axis. It calculates the linear regression slope and intercept for the axis, determines geometric properties of the grid
        vertices, and computes rotation angles needed to align the image. The optimal angle is found by minimizing the error
        across all computed angles, and the image is rotated accordingly.

        Raises:
            ValueError: If the axis is not 0 (row-wise) or 1 (column-wise).

        Args:
            image (ImageGridHandler): The input grid image object to be aligned.

        Returns:
            ImageGridHandler: The rotated grid image object after alignment.
        """
        if self.axis == 0:
            # If performing row-wise alignment, the x value is the cc value
            x_group = GRID.GRID_ROW_NUM
            x_val = OBJECT_INFO.CENTER_CC
        elif self.axis == 1:
            # If performing column-wise alignment, the x value is the rr value
            x_group = GRID.GRID_COL_NUM
            x_val = OBJECT_INFO.CENTER_RR
        else:
            raise ValueError('Axis must be either 0 or 1')

        # Find the slope info along the axis
        m, b = image.grid.get_linreg_info(axis=self.axis)
        grid_info = image.grid.info()

        # Collect the X position of the vertices
        x_min = grid_info.groupby(x_group, observed=True)[x_val].min().to_numpy()

        y_0 = (x_min * m) + b  # Find the corresponding y-value at the above x values

        # Find the x value of the upper ray
        x_max = grid_info.groupby(x_group, observed=True)[x_val].max().to_numpy()

        y_1 = (x_max * m) + b  # Find the corresponding y-value at the above x values

        # Collect opening angle ray coordinate info
        xy_vertices = np.vstack([x_min, y_0]).T  # An array containing the x & y coordinates of the vertices

        xy_upper_ray = np.vstack([x_max, y_1]).T  # An array containing the x & y coordinates of the upper ray endpoint

        # Functinon to find the euclidead distance between two points within two xy arrays stacked column-wise
        def find_hyp_dist(row):
            return euclidean(u=[row[0], row[1]], v=[row[2], row[3]])

        # Get the size of each hypotenuse
        hyp_dist = np.apply_along_axis(func1d=find_hyp_dist, axis=1, arr=np.column_stack([xy_vertices, xy_upper_ray]))

        adj_dist = x_max - x_min

        adj_over_hyp = np.divide(adj_dist, hyp_dist, where=hyp_dist != 0)

        # Find the angle of rotation from horizon in degrees
        theta = np.arccos(adj_over_hyp) * (180.0 / np.pi)

        # Adds the correct orientation to the angle
        theta_sign = y_0 - y_1
        theta = theta * (np.divide(theta_sign, abs(theta_sign), where=theta_sign != 0))

        def find_angle_of_rot(x):
            new_theta = theta + x
            err = np.mean(new_theta ** 2)
            return err

        largest_angle = np.abs(theta).max()
        optimal_angle = minimize_scalar(
            fun=find_angle_of_rot,
            bounds=(-largest_angle, largest_angle)
        )

        image.rotate(angle_of_rotation=optimal_angle.x, mode=self.mode)
        return image

# Set the documentation to match for sphinx. This is unavoidable due to sphinx statically resolving.
GridAligner.apply.__doc__ = GridAligner._operate.__doc__
