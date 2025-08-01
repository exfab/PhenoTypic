from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from functools import partial

from phenotypic.abstract import ObjectDetector
import skimage.filters as filters
import skimage.morphology as morphology


# TODO: Complete this integration
# Reference: https://omarwagih.github.io/gitter/
class SineDetector(ObjectDetector):
    """A python implementation of the Sine wave signal detection algorithm used by `gitter` in R."""

    def __init__(self, thresh_method='otsu', subtract_background: bool = True, remove_noise:bool=True,footprint_radius: int = 3):
        self.thresh_method = thresh_method
        self.subtract_background = subtract_background
        self.footprint_radius = footprint_radius
        self.remove_noise = remove_noise

    def _operate(self, image: Image) -> Image:
        enh_matrix = image.enh_matrix[:]
        objmask = self._thresholding(enh_matrix)
        if self.remove_noise:
            objmask = morphology.binary_opening(objmask, morphology.diamond(radius=self.footprint_radius))

        sum_rows = self._clean_and_sum_binary(objmask, axis=0)
        sum_cols = self._clean_and_sum_binary(objmask, axis=1)

        pass

    def _thresholding(self, image: Image) -> np.array:
        """
        Thresholds the image and returns a binary mask.
        Args:
            image:

        Returns:
            np.array: the binary object mask
        """
        kernel = morphology.square(width=self.footprint_radius * 2)

        enh_matrix = image.enh_matrix[:]

        # Subtract background
        if self.subtract_background:
            tophat_res = morphology.white_tophat(enh_matrix, kernel)
            enh_matrix = enh_matrix - tophat_res

        match self.thresh_method:
            case 'otsu':
                thresh = filters.threshold_otsu(enh_matrix)
            case 'mean':
                thresh = filters.threshold_mean(enh_matrix)
            case 'local':
                thresh = filters.threshold_local(enh_matrix, block_size=self.footprint_radius * 2)
            case 'triangle':
                thresh = filters.threshold_triangle(enh_matrix)
            case 'minimum':
                thresh = filters.threshold_minimum(enh_matrix)
            case 'isodata':
                thresh = filters.threshold_isodata(enh_matrix)
            case _:
                thresh = filters.threshold_otsu(enh_matrix)

        return enh_matrix >= thresh

    def _clean_and_sum_binary(self, binary_image: np.ndarray, p: float = 0.2, axis: int = 1) -> np.ndarray:
        """Remove long stretches of 1s (possibly lines) and sum's the mask across axes"""
        # Calculate threshold based on image dimensions
        if axis == 1:
            c = p * binary_image.shape[0]  # For rows: threshold based on number of rows
        else:
            c = p * binary_image.shape[1]  # For columns: threshold based on number of columns

        # Identify problematic rows/columns with long stretches of 1s
        problematic = np.zeros(binary_image.shape[axis - 1], dtype=bool)

        for i in range(binary_image.shape[axis - 1]):
            if axis == 1:
                row_or_col = binary_image[i, :]
            else:
                row_or_col = binary_image[:, i]

            # Run-length encoding to find stretches of 1s
            diff = np.diff(np.concatenate(([0], row_or_col, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            lengths = ends - starts

            # Check if any stretch of 1s is longer than threshold
            if len(lengths) > 0 and np.any(lengths > c):
                problematic[i] = True

        # Compute sums after identifying problematic regions
        if axis == 1:
            sums = np.sum(binary_image, axis=1)
        else:
            sums = np.sum(binary_image, axis=0)

        # Split problematic array in half and zero out problematic regions
        mid = len(problematic) // 2
        left_prob = problematic[:mid]
        right_prob = problematic[mid:]

        # Zero out sums for problematic regions at edges
        if np.any(left_prob):
            last_prob = np.where(left_prob)[0][-1]
            sums[:last_prob + 1] = 0

        if np.any(right_prob):
            first_prob = np.where(right_prob)[0][0] + mid
            sums[first_prob:] = 0

        return sums

    @staticmethod
    def _has_long_run(vec: np.ndarray, thresh: float) -> bool:
        """
        Args:
            vec: The subject vector
            thresh: The run length threshold

        Returns:

        """
        # run-length encode: find transitions and lengths
        diffs = np.diff(np.concatenate(([0], vec, [0])))
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]
        lengths = run_ends - run_starts
        return np.any(lengths > thresh)

    # ------------------------------------------------------------------
    # Tiny helpers mirroring .xl / .xr from the R code
    def upper_crop(self, z: np.ndarray, w: int) -> int:
        """Find the distance from the higher value edge to the last global minimum.
            - For row-wise, this is from the right
            - For column-wise, this is from the bottom

        Args:
            z: a numeric vector sum of pixel values
            w: the guaranteed floor width
        """
        m = np.where(z == z.min())[0]  # all minima
        t = len(z) - m[-1]  # distance from right edge
        return max(t, w)

    def lower_crop(self, z: np.ndarray, w: int) -> int:
        """Finds the distance from the lower value edge to the first global minimum.
            - For row-wise, this is from the left
            - For column-wise, this is from the top
        Args:
            z: a numeric vector sum of pixel values
            w: the guaranteed floor width
        """
        t = np.argmin(z) + 1  # first min + 1
        return max(t, w)
