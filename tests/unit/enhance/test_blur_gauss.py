from __future__ import annotations

import numpy as np

from phenotypic import Image
from phenotypic.enhance import BlurGauss


def test_blur_gauss_smooths_detect_mat_across_rows_and_columns():
    """A 2D detect_mat must be blurred over both spatial axes."""
    detect_mat = np.zeros((21, 21), dtype=float)
    detect_mat[10, 10] = 1.0
    image = Image(arr=detect_mat)

    result = BlurGauss(sigma=1.0, mode="constant").apply(image)
    blurred = result.detect_mat[:]

    assert blurred[9, 10] > 0
    assert blurred[10, 9] > 0
