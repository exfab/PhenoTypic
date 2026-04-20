"""Tests for ManualGridDetector.napari() interactive coordinate picking."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from phenotypic.detect import ManualGridDetector


class TestManualGridDetectorNapari:
    """Test .napari() with mocked PointPickerWidget."""

    def test_napari_with_two_points(self):
        """Two picked points set coord1 and coord2."""
        det = ManualGridDetector(coord1=(10, 10))

        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            mock_instance = MockWidget.return_value
            mock_instance.run.return_value = np.array([[50, 60], [100, 120]])
            result = det.napari(MagicMock())

        assert det.coord1 == (50, 60)
        assert det.coord2 == (100, 120)
        assert result is det

    def test_napari_with_one_point(self):
        """One picked point sets coord1; coord2 becomes None."""
        det = ManualGridDetector(coord1=(10, 10), coord2=(30, 30))

        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            mock_instance = MockWidget.return_value
            mock_instance.run.return_value = np.array([[50, 60]])
            result = det.napari(MagicMock())

        assert det.coord1 == (50, 60)
        assert det.coord2 is None
        assert result is det

    def test_napari_with_zero_points(self):
        """Zero picked points (user closed without confirm) leaves state unchanged."""
        det = ManualGridDetector(coord1=(10, 10), coord2=(30, 30))

        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            mock_instance = MockWidget.return_value
            mock_instance.run.return_value = np.empty((0, 2))
            result = det.napari(MagicMock())

        # State unchanged since no points were returned
        assert det.coord1 == (10, 10)
        assert det.coord2 == (30, 30)
        assert result is det

    def test_napari_returns_self(self):
        """napari() returns self for method chaining."""
        det = ManualGridDetector()

        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            mock_instance = MockWidget.return_value
            mock_instance.run.return_value = np.array([[50, 60], [100, 120]])
            result = det.napari(MagicMock())

        assert result is det

    def test_napari_coordinates_are_rounded_tuples(self):
        """Coordinates from napari are rounded to int and stored as tuples."""
        det = ManualGridDetector()

        with patch("phenotypic.tools_.napari_.PointPickerWidget") as MockWidget:
            mock_instance = MockWidget.return_value
            mock_instance.run.return_value = np.array([[50.7, 60.3], [100.1, 120.9]])
            det.napari(MagicMock())

        assert det.coord1 == (51, 60)
        assert det.coord2 == (100, 121)
        assert isinstance(det.coord1, tuple)
        assert isinstance(det.coord2, tuple)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
