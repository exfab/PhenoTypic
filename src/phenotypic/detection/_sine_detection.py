from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from functools import partial

from phenotypic.abstract import ObjectDetector


# TODO: Complete this integration
# Reference: https://omarwagih.github.io/gitter/
class SineDetector(ObjectDetector):
    """A python implementation of the Sine wave signal detection algorithm used by `gitter` in R."""

    def _operate(self, image: Image) -> Image:
        enh_matrix = image.enh_matrix[:]

        pass

    def _remove_lines_and_sum(self, binary_mask: np.ndarray, p: float = 0.2, axis: int = 1) -> np.ndarray:
        """
        Removes lines from a binary mask based on the presence of long consecutive runs of 1's,
        and computes the sum of remaining rows or columns. The process applies a threshold
        proportional to the specified axis length.

        Args:
            binary_mask (np.ndarray): A binary (2D) matrix representing the mask to analyze and modify.
            p (float): Proportional threshold for detecting long runs of 1's. Specifies the fraction
                of the axis length needed to detect a streak. Defaults to 0.2.
            axis (int): The axis along which the operation is performed. 1 corresponds to rows
                and 2 corresponds to columns.

        Returns:
            np.ndarray: A 1D array containing the sums of the binary mask rows or columns
                after removing offending sections based on the threshold.
        """
        if axis not in (1, 2):
            raise ValueError("margin must be 1 (rows) or 2 (columns)")

        # run-length threshold
        c = p * (binary_mask.shape[0] if axis == 1 else binary_mask.shape[1])

        # Detect rows/cols with a long streak of 1’s
        line_checker = partial(self._has_long_run, thresh=c)
        axis = 0 if axis == 1 else 1
        z = np.apply_along_axis(line_checker, axis=axis, array=binary_mask)  # boolean vector

        # Per-row / per-column sums
        x = binary_mask.sum(axis=1) if axis == 1 else binary_mask.sum(axis=0)

        # Split z in half and zero-out from edges to last/first offending line
        mid = (len(z) + 1) // 2
        left, right = z[:mid - 1], z[mid:]

        if left.any():
            x[: np.max(np.where(left)) + 1] = 0
        if right.any():
            first_bad = np.min(np.where(right)) + len(left) + 1
            x[first_bad:] = 0

        return x

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
