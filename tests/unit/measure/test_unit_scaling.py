"""Measured columns scale with image resolution exactly as their units say.

One bright disk is measured at radius 60 on a 600x600 image and at radius 120
on a 1200x1200 image (k = 2). A column whose unit has length power p must grow
by k**p: lengths double, areas and integrated intensities quadruple, and
scale-free values hold. A failure names a member whose unit tag is wrong; fix
the tag, never the tolerance (plan 02, Task 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phenotypic as pht
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureIntensity, MeasureShape, MeasureSize
from phenotypic.schema import INTENSITY, SHAPE, SIZE, UNITS


def _disk_image(radius: int) -> pht.Image:
    size = 10 * radius
    rows, cols = np.ogrid[:size, :size]
    centre = (size - 1) / 2
    disk = (rows - centre) ** 2 + (cols - centre) ** 2 <= radius**2
    return pht.Image(np.where(disk, 0.9, 0.1).astype(np.float64))


def _measure_disk(measurer: type, radius: int) -> pd.DataFrame:
    image = OtsuDetector().apply(_disk_image(radius))
    assert image.num_objects == 1, f"expected one disk, found {image.num_objects}"
    return measurer().measure(image)


@pytest.mark.parametrize(
    "measurer, info",
    [(MeasureSize, SIZE), (MeasureShape, SHAPE), (MeasureIntensity, INTENSITY)],
)
def test_columns_scale_as_their_unit_says(measurer, info):
    small, large = _measure_disk(measurer, radius=60), _measure_disk(measurer, radius=120)
    checked = []
    for member in info:
        p = member.unit.length_power
        if p is None or member.unit is UNITS.INTENSITY or str(member) not in small:
            continue
        a, b = float(small[str(member)].iloc[0]), float(large[str(member)].iloc[0])
        if p == 0:
            assert abs(b - a) <= 0.05 * max(1.0, abs(a)), (str(member), a, b)
        else:
            assert b / a == pytest.approx(2**p, rel=0.05), (str(member), a, b)
        checked.append(str(member))
    # A loop that skipped every member would pass while checking nothing.
    assert checked, f"no {info.__name__} column was checked"
