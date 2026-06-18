"""Unit tests for ``PointPickerMixin`` — picked-coords param marker + napari hook.

Coordinate coercion no longer lives on the mixin: ``PointPickerMixin``'s
``__setattr__`` override was removed in the pydantic migration (it
conflicts with pydantic's own ``__setattr__``). The picked-coordinates
field is now coerced by a ``field_validator`` on each consuming
operation (``ManualPointDetector`` / ``ManualRefine``) — covered by
``tests/unit/detect`` and ``tests/unit/refine``.
"""

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
    """``ManualPointDetector`` and ``ManualRefine`` mix the marker in."""

    def test_manual_point_detector_inherits(self):
        from phenotypic.detect import ManualPointDetector

        det = ManualPointDetector()
        assert isinstance(det, PointPickerMixin)
        assert det._point_picker_param_name == "centers"

    def test_manual_selector_inherits(self):
        from phenotypic.refine import ManualRefine

        sel = ManualRefine()
        assert isinstance(sel, PointPickerMixin)
        assert sel._point_picker_param_name == "centers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
