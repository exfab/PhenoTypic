"""Unit tests for ``PointPickerMixin`` — coordinate coercion + napari hook."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from phenotypic.tools_.mixin import PointPickerMixin


class _Dummy(PointPickerMixin):
    """Minimal subclass for exercising the mixin in isolation."""

    def __init__(self, centers=None):
        self.centers = centers


class _CustomParam(PointPickerMixin):
    """Subclass that renames the picked-coordinates parameter."""

    _point_picker_param_name = "points"

    def __init__(self, points=None):
        self.points = points


class TestParamNameAttribute:
    """The class attribute must default to ``"centers"`` and override cleanly."""

    def test_default_param_name(self):
        assert PointPickerMixin._point_picker_param_name == "centers"
        assert _Dummy._point_picker_param_name == "centers"

    def test_subclass_can_override(self):
        assert _CustomParam._point_picker_param_name == "points"


class TestSetattrCoercion:
    """``__setattr__`` coerces the picked-coords parameter into a NumPy array."""

    def test_list_is_coerced(self):
        d = _Dummy(centers=[[1, 2], [3, 4]])
        assert isinstance(d.centers, np.ndarray)
        np.testing.assert_array_equal(d.centers, [[1, 2], [3, 4]])

    def test_tuple_iterable_is_coerced(self):
        d = _Dummy(centers=[(10, 20), (30, 40)])
        assert isinstance(d.centers, np.ndarray)
        np.testing.assert_array_equal(d.centers, [[10, 20], [30, 40]])

    def test_ndarray_passes_through(self):
        arr = np.array([[5, 6]])
        d = _Dummy(centers=arr)
        # asarray on an array is allowed to return the same object
        assert isinstance(d.centers, np.ndarray)
        np.testing.assert_array_equal(d.centers, arr)

    def test_none_passes_through(self):
        d = _Dummy()
        assert d.centers is None
        d.centers = None
        assert d.centers is None

    def test_other_attributes_untouched(self):
        d = _Dummy()
        d.some_label = [1, 2, 3]  # type: ignore[attr-defined]
        assert d.some_label == [1, 2, 3]
        assert not isinstance(d.some_label, np.ndarray)

    def test_custom_param_name_is_coerced(self):
        c = _CustomParam(points=[[0, 0]])
        assert isinstance(c.points, np.ndarray)
        # Default ``centers`` attribute on the same instance is NOT coerced.
        c.centers = [1, 2, 3]  # type: ignore[attr-defined]
        assert c.centers == [1, 2, 3]


class TestNapariMethod:
    """``napari()`` writes confirmed picks back via the configured param name."""

    def test_napari_writes_picks(self):
        d = _Dummy()
        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            MockWidget.return_value.run.return_value = np.array(
                [[10.0, 20.0], [30.0, 40.0]]
            )
            result = d.napari(MagicMock())
        assert result is d
        np.testing.assert_array_equal(d.centers, [[10.0, 20.0], [30.0, 40.0]])

    def test_napari_empty_preserves_existing(self):
        d = _Dummy(centers=[[1, 2]])
        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            MockWidget.return_value.run.return_value = np.empty((0, 2))
            d.napari(MagicMock())
        np.testing.assert_array_equal(d.centers, [[1, 2]])

    def test_napari_uses_param_name_attribute(self):
        c = _CustomParam()
        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            MockWidget.return_value.run.return_value = np.array([[7.0, 8.0]])
            c.napari(MagicMock())
        np.testing.assert_array_equal(c.points, [[7.0, 8.0]])
        # The default ``centers`` attribute is untouched.
        assert getattr(c, "centers", None) is None


class TestMixinIsMarkerForRealOps:
    """``ManualPointDetector`` and ``ManualSelector`` mix the marker in."""

    def test_manual_point_detector_inherits(self):
        from phenotypic.detect import ManualPointDetector

        det = ManualPointDetector()
        assert isinstance(det, PointPickerMixin)
        assert det._point_picker_param_name == "centers"

    def test_manual_selector_inherits(self):
        from phenotypic.refine import ManualSelector

        sel = ManualSelector()
        assert isinstance(sel, PointPickerMixin)
        assert sel._point_picker_param_name == "centers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
