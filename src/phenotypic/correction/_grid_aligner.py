from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize_scalar

from phenotypic.abc_ import GridCorrector
from phenotypic.tools_.measurement_info import BBOX, GRID


class GridAligner(GridCorrector):
    """Correct grid rotation by aligning colony centroids to row or column axes.

    Compute the optimal rotation angle from linear regression of colony
    centroid positions along the chosen axis, then rotate the entire image
    to minimize angular misalignment. Re-detection of objects after
    alignment is strongly recommended because pixel coordinates shift.

    For algorithm details, see :doc:`/explanation/grid_vs_non_grid_detection`.

    Args:
        axis: Alignment axis. ``0`` aligns rows (row-wise regression on
            column centroid positions); ``1`` aligns columns. Default: ``0``.
        mode: Edge-fill mode passed to the rotation function. ``'edge'``
            replicates border pixels; ``'constant'`` fills with zeros.
            Default: ``'edge'``.

    Returns:
        GridImage: Input image rotated so that colony centroids align with
        the specified axis. All image components are transformed.

    Raises:
        ValueError: If ``axis`` is not ``0`` or ``1``.

    Best For:
        - Arrayed plates scanned at a slight angle where grid rows or
          columns are not axis-aligned.
        - High-throughput imaging setups with inconsistent plate
          orientation between scans.
        - Pre-processing before grid-based measurement to ensure accurate
          row and column assignment.

    Consider Also:
        - :class:`ImagePadder` to add safety margins before rotation so
          corner colonies are not clipped.
        - :class:`ImageCropper` to remove excess background after
          alignment.

    See Also:
        :doc:`/how_to/notebooks/correct_grid_rotation` for a visual
        walkthrough of grid alignment on real plate images.
    """

    axis: int = 0
    mode: str = "edge"

    def _operate(self, image: GridImage):
        """Calculates the optimal rotation angle and applies it to a grid image for alignment along the specified axis.

        The method performs alignment of a `GridImage` object along either nrows or columns based on the specified
        axis. It calculates the linear regression slope and intercept for the axis, determines geometric properties of the grid
        vertices, and computes rotation angles needed to align the image. The optimal angle is found by minimizing the error
        across all computed angles, and the image is rotated accordingly.

        Raises:
            ValueError: If the axis is not 0 (row-wise) or 1 (column-wise).

        Args:
            image (ImageGridHandler): The arr grid image object to be aligned.

        Returns:
            ImageGridHandler: The rotated grid image object after alignment.
        """
        if self.axis == 0:
            # If performing row-wise alignment, the x value is the cc value
            x_group = str(GRID.ROW_NUM)
            x_val = str(BBOX.CENTER_CC)
        elif self.axis == 1:
            # If performing column-wise alignment, the x value is the rr value
            x_group = str(GRID.COL_NUM)
            x_val = str(BBOX.CENTER_RR)
        else:
            raise ValueError("Axis must be either 0 or 1")

        # Find the slope info along the axis
        m, b = image.grid.get_centroid_alignment_info(axis=self.axis)
        grid_info = image.grid.info()

        # Collect aligned X positions of the vertices
        grouped = (grid_info.groupby(x_group, observed=True)[x_val]
                   .agg(["min", "max"])
                   .to_numpy())

        # Collect the X position of the vertices
        x_min = grouped[:, 0]

        # Find the x value of the upper ray
        x_max = grouped[:, 1]

        # Find the corresponding y-value at the above x values
        y_0 = (x_min * m) + b

        # Find the corresponding y-value at the above x values
        y_1 = (x_max * m) + b

        # Collect opening angle ray coordinate info

        # An array containing the x & y coordinates of the vertices
        xy_vertices = np.vstack([x_min, y_0]).T

        # An array containing the x & y coordinates of the upper ray endpoint
        xy_upper_ray = np.vstack([x_max, y_1]).T

        # Function to find the euclidead distance between two points within 
        # two xy arrays stacked column-wise

        # Get the size of each hypotenuse
        hyp_dist = np.apply_along_axis(
                func1d=self._find_hyp_dist,
                axis=1,
                arr=np.column_stack([xy_vertices, xy_upper_ray]),
        )

        adj_dist = x_max - x_min

        adj_over_hyp = np.divide(
                adj_dist, hyp_dist, where=(hyp_dist != 0) | (adj_dist != 0)
        )

        # Find the angle of rotation from horizon in degrees
        theta = np.arccos(adj_over_hyp) * (180.0 / np.pi)

        # Adds the correct orientation to the angle
        theta_sign = y_0 - y_1
        theta = theta * (np.divide(theta_sign, abs(theta_sign), where=theta_sign != 0))

        largest_angle = np.abs(theta).max()
        optimal_angle = minimize_scalar(
                fun=self._find_angle_of_rot,
                bounds=(-largest_angle, largest_angle),
                args=theta
        )

        image.rotate(angle_of_rotation=optimal_angle.x, mode=self.mode)
        return image

    def _find_angle_of_rot(self, X, theta):
        new_theta = theta + X
        err = np.mean(new_theta ** 2)
        return err

    @staticmethod
    def _find_hyp_dist(row):
        return euclidean(u=[row[0], row[1]], v=[row[2], row[3]])


# Set the documentation to match for sphinx.
# This is unavoidable due to sphinx statically resolving.
GridAligner.apply.__doc__ = GridAligner._operate.__doc__
