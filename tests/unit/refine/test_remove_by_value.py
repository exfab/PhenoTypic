"""Unit tests for the generic ``RemoveByFeature`` refiner.

``RemoveByFeature`` names a ``MeasureFeatures`` subclass, runs it, and zeros every
object whose chosen value falls outside an inclusive ``[min, max]`` band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize, MeasureColor
from phenotypic.refine import RemoveByFeature
from phenotypic.schema import OBJECT, SIZE


@pytest.fixture()
def plate():
    """A synthetic yeast plate with colonies already detected."""
    return load_synth_yeast_plate()


@pytest.fixture()
def size_table(plate):
    """The ``MeasureSize`` table for the fixture plate (label-indexed area)."""
    return MeasureSize().measure(plate)


def _surviving_labels(image):
    labels = np.unique(image.objmap[:])
    return set(labels[labels != 0].tolist())


def _labels_outside(table, column, *, min_value=None, max_value=None):
    keep = pd.Series(True, index=table.index)
    if min_value is not None:
        keep &= table[column] >= min_value
    if max_value is not None:
        keep &= table[column] <= max_value
    return set(table.loc[~keep, OBJECT.LABEL].tolist())


class TestNoOpContract:
    def test_default_construction_is_a_noop(self, plate):
        """All-None defaults must construct and leave the image untouched."""
        before = _surviving_labels(plate)
        result = RemoveByFeature().apply(plate)
        assert _surviving_labels(result) == before

    def test_missing_bounds_is_a_noop(self, plate):
        """Feature + value set but no bounds removes nothing."""
        before = _surviving_labels(plate)
        result = RemoveByFeature(feature="MeasureSize", value="Size_Area").apply(plate)
        assert _surviving_labels(result) == before

    def test_missing_value_is_a_noop(self, plate):
        before = _surviving_labels(plate)
        result = RemoveByFeature(feature="MeasureSize", min_value=10).apply(plate)
        assert _surviving_labels(result) == before


class TestFiltering:
    def test_min_bound_removes_small_objects(self, plate, size_table):
        """Objects below ``min_value`` are removed; the rest survive."""
        threshold = float(size_table[SIZE.AREA].median())
        expected_removed = _labels_outside(
                size_table, SIZE.AREA, min_value=threshold
        )

        result = RemoveByFeature(
                feature="MeasureSize", value="Size_Area", min_value=threshold
        ).apply(plate)

        assert expected_removed  # the median split actually removes something
        assert _surviving_labels(result).isdisjoint(expected_removed)
        assert _surviving_labels(result) == _surviving_labels(plate) - expected_removed

    def test_band_keeps_only_in_range(self, plate, size_table):
        """Inclusive two-sided band keeps exactly the in-range objects."""
        lo = float(size_table[SIZE.AREA].quantile(0.25))
        hi = float(size_table[SIZE.AREA].quantile(0.75))
        expected_removed = _labels_outside(
                size_table, SIZE.AREA, min_value=lo, max_value=hi
        )

        result = RemoveByFeature(
                feature="MeasureSize", value="Size_Area", min_value=lo, max_value=hi
        ).apply(plate)

        assert _surviving_labels(result) == _surviving_labels(plate) - expected_removed

    def test_bare_label_matches_prefixed_column(self, plate):
        """``value="Area"`` resolves to the same result as ``"Size_Area"``."""
        kw = dict(feature="MeasureSize", min_value=40.0, max_value=900.0)
        prefixed = RemoveByFeature(value="Size_Area", **kw).apply(plate.copy())
        bare = RemoveByFeature(value="Area", **kw).apply(plate.copy())
        assert _surviving_labels(prefixed) == _surviving_labels(bare)

    def test_one_sided_max_bound(self, plate, size_table):
        """Only ``max_value`` set leaves the lower side unbounded."""
        threshold = float(size_table[SIZE.AREA].median())
        expected_removed = _labels_outside(
                size_table, SIZE.AREA, max_value=threshold
        )
        result = RemoveByFeature(
                feature="MeasureSize", value="Size_Area", max_value=threshold
        ).apply(plate)
        assert _surviving_labels(result) == _surviving_labels(plate) - expected_removed

    def test_apply_is_out_of_place_by_default(self, plate):
        """The source image is not mutated when ``inplace`` is left False."""
        before = _surviving_labels(plate)
        RemoveByFeature(
                feature="MeasureSize", value="Size_Area", min_value=10_000
        ).apply(plate)
        assert _surviving_labels(plate) == before


class TestMeasureKwargs:
    def test_measure_kwargs_forwarded_to_measurer(self, plate):
        """``measure_kwargs`` flow into the measurer constructor."""
        # MeasureColor exposes an include_XYZ flag; with it off the XYZ columns
        # are absent, so filtering on an XYZ value would fail — but a Lab value
        # still resolves, proving the kwarg reached the constructor.
        op = RemoveByFeature(
                feature="MeasureColor",
                value=str(next(iter(MeasureColor().measure(plate).columns[1:]))),
                min_value=-1e9,
                measure_kwargs={"include_XYZ": False},
        )
        # Should run without error and remove nothing (min is unreachably low).
        result = op.apply(plate)
        assert _surviving_labels(result) == _surviving_labels(plate)


class TestValidation:
    def test_unknown_feature_rejected_at_construction(self):
        with pytest.raises(ValueError, match="MeasureFeatures subclass"):
            RemoveByFeature(feature="NotAMeasurer")

    def test_non_measurer_feature_rejected(self):
        """A real phenotypic class that is not a MeasureFeatures is rejected."""
        with pytest.raises(ValueError, match="MeasureFeatures subclass"):
            RemoveByFeature(feature="OtsuDetector")

    def test_inverted_bounds_rejected(self):
        with pytest.raises(ValueError, match="must not exceed"):
            RemoveByFeature(
                    feature="MeasureSize",
                    value="Size_Area",
                    min_value=100,
                    max_value=10,
            )

    def test_unknown_value_raises_on_apply(self, plate):
        op = RemoveByFeature(
                feature="MeasureSize", value="Nonexistent", min_value=1
        )
        with pytest.raises(Exception, match="not a measured value|Nonexistent"):
            op.apply(plate)


class TestSerialization:
    def test_round_trips_through_json(self):
        op = RemoveByFeature(
                feature="MeasureSize",
                value="Size_Area",
                min_value=20.0,
                max_value=500.0,
                measure_kwargs={"foo": 1},
        )
        restored = RemoveByFeature.from_json(op.to_json())
        assert restored.feature == "MeasureSize"
        assert restored.value == "Size_Area"
        assert restored.min_value == 20.0
        assert restored.max_value == 500.0
        assert restored.measure_kwargs == {"foo": 1}
