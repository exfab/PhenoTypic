from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.correction import BayesShrinkCorrector, VisuShrinkCorrector


@pytest.mark.parametrize(
        "corrector_cls",
        [BayesShrinkCorrector, VisuShrinkCorrector],
)
@pytest.mark.parametrize(
        ("dtype", "value", "minimum_expected"),
        [
            (np.uint8, 128, 100),
            (np.uint16, 32768, 30000),
        ],
)
def test_wavelet_correctors_restore_rgb_integer_dtype_and_range(
        corrector_cls,
        dtype,
        value,
        minimum_expected,
):
    """RGB wavelet output is normalized by skimage and must be scaled back."""
    rgb = np.full((32, 32, 3), value, dtype=dtype)
    image = Image(arr=rgb)

    result = corrector_cls(sigma=0.01, wavelet_levels=1).apply(image)
    result_rgb = result.rgb[:]

    assert result_rgb.dtype == dtype
    assert result_rgb.max() >= minimum_expected
