import numpy as np
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize_scalar
from skimage.transform import rotate

from ..interface import GridTransformer
from .. import GriddedImage


class AutoGridAlignmentTransformer(GridTransformer):
    def __init__(self, axis: int = 0, mode: str = 'edge'):
        self.axis = axis
        self.mode = mode

    def _operate(self, image: GriddedImage):

        if self.axis == 0:

            # If performing row-wise alignment, the x value is the cc value
            x_group = image.grid_extractor.LABEL_GRID_ROW_NUM
            x_val = image.bound_extractor.LABEL_CENTER_CC
        elif self.axis == 1:

            # If performing column-wise alignment, the x value is the rr value
            x_group = image.grid_extractor.LABEL_GRID_COL_NUM
            x_val = image.bound_extractor.LABEL_CENTER_RR
        else:
            raise ValueError('Axis must be either 0 or 1')

        # Find the slope info along the axis
        m, b = image.get_linreg_info(axis=self.axis)
        grid_info = image.grid_info

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

        array = rotate(image=image.matrix, angle=optimal_angle.x, mode=self.mode)
        enhanced_array = rotate(image=image.enhanced_matrix, angle=optimal_angle.x, mode=self.mode)

        if image.array is not None:
            carray = rotate(image=image.array, angle=optimal_angle.x, mode=self.mode)
            image.array = carray

        # TODO: Maybe find a way to streamline this?
        image.matrix = array
        image.enhanced_matrix = enhanced_array
        return image
