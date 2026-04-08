"""Tests for show/dash API contract.

show() is always matplotlib — no patching needed.
dash() requires plotly — when unavailable, raises ImportError.
"""

from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from phenotypic.data import load_synth_yeast_plate


@pytest.fixture(scope="module")
def grid_image():
    """Load a GridImage with detected objects for reuse across tests."""
    return load_synth_yeast_plate()


# ---------------------------------------------------------------------------
# show() always returns matplotlib — no patch needed
# ---------------------------------------------------------------------------


class TestShowAlwaysMpl:
    """Verify show() always returns (fig, ax) regardless of plotly."""

    def test_gray_show_returns_mpl(self, grid_image):
        result = grid_image.gray.show()
        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_detect_mat_show_returns_mpl(self, grid_image):
        fig, ax = grid_image.detect_mat.show()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_rgb_show_returns_mpl(self, grid_image):
        fig, ax = grid_image.rgb.show()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_rgb_show_channel_returns_mpl(self, grid_image):
        fig, ax = grid_image.rgb.show(channel=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_image_show_returns_mpl(self, grid_image):
        fig, ax = grid_image.show()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# show(overlay=True) always returns matplotlib
# ---------------------------------------------------------------------------


class TestShowOverlayAlwaysMpl:
    """Verify show(overlay=True) returns (fig, ax)."""

    def test_show_overlay_returns_mpl(self, grid_image):
        fig, ax = grid_image.show(overlay=True)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_show_overlay_with_labels_returns_mpl(self, grid_image):
        fig, ax = grid_image.show(overlay=True, show_labels=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# dash() requires plotly — ImportError when unavailable
# ---------------------------------------------------------------------------


@patch("phenotypic._core._image_parts.accessor_abstracts._image_accessor_base_parents._accessor_dash_handler.PLOTLY_AVAILABLE", False)
class TestDashRequiresPlotly:
    """Verify dash() raises ImportError when plotly unavailable."""

    def test_gray_dash_raises(self, grid_image):
        with pytest.raises(ImportError, match="plotly is required"):
            grid_image.gray.dash()

    def test_rgb_dash_raises(self, grid_image):
        with pytest.raises(ImportError, match="plotly is required"):
            grid_image.rgb.dash()

    def test_detect_mat_dash_raises(self, grid_image):
        with pytest.raises(ImportError, match="plotly is required"):
            grid_image.detect_mat.dash()

    def test_image_dash_raises(self, grid_image):
        with pytest.raises(ImportError, match="plotly is required"):
            grid_image.dash()
