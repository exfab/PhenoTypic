"""KeepSectionLargest selects the largest object per grid section."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.refine import KeepSectionLargest
from phenotypic.schema import GRID, OBJECT, SIZE


@pytest.fixture(scope="module")
def detected():
    """Otsu on the synth plate over-segments (552 objects for 96 wells), so most
    sections hold several candidates and the selection is non-trivial."""
    image = OtsuDetector().apply(load_synth_yeast_plate())
    assert image.num_objects > 96
    return image


def _labels_by_the_old_algorithm(image) -> np.ndarray:
    """main's implementation, kept as the oracle: MeasureSize area, idxmax per section."""
    table = MeasureSize().measure(image, include_meta=True)
    max_idx = table.groupby(by=GRID.ROW_MAJOR_IDX, observed=True)[SIZE.AREA].idxmax()
    return np.sort(table.loc[max_idx, OBJECT.LABEL].to_numpy())


def test_selects_the_same_labels_as_before(detected):
    expected = _labels_by_the_old_algorithm(detected.copy())
    result = KeepSectionLargest().apply(detected.copy())
    kept = np.unique(result.objmap[:])
    assert np.array_equal(kept[kept > 0], expected)


def test_does_not_run_a_measurer(detected, monkeypatch):
    """Mutation: restore `MeasureSize().measure(...)` in _operate -> fails."""
    def _refuse(self, image):
        raise AssertionError("KeepSectionLargest must not run MeasureSize")

    monkeypatch.setattr(MeasureSize, "_operate", _refuse)
    KeepSectionLargest().apply(detected.copy())


def test_an_empty_plate_fails_with_runtime_error(detected):
    """Review LOW-7. With no objects, `labels2series()` raises `NoObjectsError`
    (main raised `OperationFailedError` from `MeasureSize().measure`), and
    `ImageOperation.apply` wraps either in `RuntimeError`. That outer type is
    the contract; no consumer inspects the cause.

    Mutation: return the image unchanged when there are no labels -> fails.
    """
    empty = detected.copy()
    empty.objmap[:] = 0
    with pytest.raises(RuntimeError, match="KeepSectionLargest failed"):
        KeepSectionLargest().apply(empty)
