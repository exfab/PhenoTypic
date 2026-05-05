"""Tests for ManualPointDetector: init, __setattr__ coercion, _operate, and napari."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import ManualPointDetector


# ---- Fixtures ----


@pytest.fixture(scope="module")
def synth_image():
    """Load the synthetic yeast plate once for all tests in this module."""
    return load_synth_yeast_plate()


# ---- Init Tests ----


class TestManualPointDetectorInit:
    """Test __init__ defaults and parameter coercion."""

    def test_default_init(self):
        """Default ManualPointDetector has centers=None, shape='disk', width=15."""
        det = ManualPointDetector()
        assert det.centers is None
        assert det.shape == "disk"
        assert det.width == 15

    def test_init_with_centers_list(self):
        """Centers given as a list of lists are coerced to np.ndarray."""
        det = ManualPointDetector(centers=[[50, 60], [100, 120]])
        assert isinstance(det.centers, np.ndarray)
        assert det.centers.shape == (2, 2)
        np.testing.assert_array_equal(det.centers, [[50, 60], [100, 120]])

    def test_setattr_coercion(self):
        """Setting .centers to a list coerces it to np.ndarray."""
        det = ManualPointDetector()
        det.centers = [[10, 20]]
        assert isinstance(det.centers, np.ndarray)
        np.testing.assert_array_equal(det.centers, [[10, 20]])

    def test_setattr_none_passthrough(self):
        """Setting .centers to None keeps it as None (no coercion)."""
        det = ManualPointDetector(centers=[[10, 20]])
        det.centers = None
        assert det.centers is None

    def test_inherits_point_picker_mixin(self):
        """ManualPointDetector mixes in PointPickerMixin and exposes its marker."""
        from phenotypic.tools_.mixin import PointPickerMixin

        det = ManualPointDetector()
        assert isinstance(det, PointPickerMixin)
        assert det._point_picker_param_name == "centers"


# ---- _operate Tests ----


class TestManualPointDetectorOperate:
    """Test _operate behaviour via .apply()."""

    def test_operate_no_centers(self, synth_image):
        """With no centers, objmask is all False and objmap is all zeros."""
        det = ManualPointDetector(centers=None)
        result = det.apply(synth_image, inplace=False)
        assert not result.objmask[:].any()
        assert (result.objmap[:] == 0).all()

    def test_operate_empty_centers(self, synth_image):
        """Empty centers array also produces all-zero masks."""
        det = ManualPointDetector(centers=np.empty((0, 2), dtype=int))
        result = det.apply(synth_image, inplace=False)
        assert not result.objmask[:].any()
        assert (result.objmap[:] == 0).all()

    def test_operate_single_center(self, synth_image):
        """Single center produces True pixels near the center and label=1."""
        cy, cx = 50, 60
        det = ManualPointDetector(centers=[[cy, cx]], shape="disk", width=15)
        result = det.apply(synth_image, inplace=False)

        objmask = result.objmask[:]
        objmap = result.objmap[:]

        # There should be True pixels in the vicinity of (50, 60)
        assert objmask[cy, cx]
        # Label at center should be 1
        assert objmap[cy, cx] == 1
        # Only one unique label apart from background
        unique_labels = np.unique(objmap)
        assert set(unique_labels) == {0, 1}

    def test_operate_multiple_centers(self, synth_image):
        """Two well-separated centers produce two distinct labels (1 and 2)."""
        det = ManualPointDetector(
            centers=[[50, 60], [200, 200]], shape="disk", width=15
        )
        result = det.apply(synth_image, inplace=False)

        objmap = result.objmap[:]
        assert objmap[50, 60] == 1
        assert objmap[200, 200] == 2

        unique_labels = np.unique(objmap)
        assert set(unique_labels) == {0, 1, 2}

    def test_edge_clipping(self, synth_image):
        """Center near image edge (0, 0) does not raise and clips correctly."""
        det = ManualPointDetector(centers=[[0, 0]], shape="disk", width=15)
        result = det.apply(synth_image, inplace=False)

        objmask = result.objmask[:]
        objmap = result.objmap[:]

        # Some pixels should be stamped near the top-left corner
        assert objmask[:8, :8].any()
        # Label near origin should be 1
        assert objmap[0, 0] == 1


# ---- napari Tests ----


class TestManualPointDetectorNapari:
    """Test .napari() integration via mocked PointPickerWidget."""

    def test_napari_sets_centers(self):
        """napari() with picked points sets self.centers and returns self."""
        det = ManualPointDetector()

        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            mock_instance = MockWidget.return_value
            mock_instance.run.return_value = np.array([[50, 60], [100, 120]])
            result = det.napari(MagicMock())

        assert isinstance(det.centers, np.ndarray)
        np.testing.assert_array_equal(det.centers, [[50, 60], [100, 120]])
        assert result is det

    def test_napari_empty_result_no_change(self):
        """napari() with empty result leaves centers unchanged."""
        det = ManualPointDetector()
        assert det.centers is None

        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            mock_instance = MockWidget.return_value
            mock_instance.run.return_value = np.empty((0, 2))
            result = det.napari(MagicMock())

        # centers should remain None since no points were picked
        assert det.centers is None
        assert result is det

    def test_napari_empty_result_preserves_existing_centers(self):
        """napari() with empty result preserves previously set centers."""
        original_centers = [[10, 20], [30, 40]]
        det = ManualPointDetector(centers=original_centers)

        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            mock_instance = MockWidget.return_value
            mock_instance.run.return_value = np.empty((0, 2))
            det.napari(MagicMock())

        np.testing.assert_array_equal(det.centers, original_centers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
