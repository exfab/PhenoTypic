"""Every prefab that measures shape also measures size (area and radii moved there)."""

from __future__ import annotations

import pytest

import phenotypic.prefab as prefab
from phenotypic.measure import MeasureShape, MeasureSize


def _prefab_classes():
    for name in prefab.__all__:
        cls = getattr(prefab, name)
        if isinstance(cls, type):
            yield pytest.param(cls, id=name)


@pytest.mark.parametrize("cls", list(_prefab_classes()))
def test_prefab_with_measure_shape_also_has_measure_size(cls):
    kinds = {type(m) for m in cls().meas.values()}  # `meas` is normalised to a dict
    if MeasureShape in kinds:
        assert MeasureSize in kinds, f"{cls.__name__} measures shape but not size"
