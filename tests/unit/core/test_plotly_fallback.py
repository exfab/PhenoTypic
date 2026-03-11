"""Tests for matplotlib fallback when plotly is not installed."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from phenotypic.data import load_synth_yeast_plate


@pytest.fixture(scope="module")
def grid_image():
    """Load a GridImage with detected objects for reuse across tests."""
    return load_synth_yeast_plate()


@patch("phenotypic.tools_._plotly_helpers.PLOTLY_AVAILABLE", False)
class TestMplFallbackSingleChannel:
    """Test matplotlib fallback for single-channel show()."""

    def test_gray_show_returns_mpl(self, grid_image):
        result = grid_image.gray.show()
        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_detect_mat_show_returns_mpl(self, grid_image):
        result = grid_image.detect_mat.show()
        assert isinstance(result, tuple)
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@patch("phenotypic.tools_._plotly_helpers.PLOTLY_AVAILABLE", False)
class TestMplFallbackMultiChannel:
    """Test matplotlib fallback for multichannel show()."""

    def test_rgb_show_returns_mpl(self, grid_image):
        result = grid_image.rgb.show()
        assert isinstance(result, tuple)
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_rgb_show_channel_returns_mpl(self, grid_image):
        result = grid_image.rgb.show(channel=0)
        assert isinstance(result, tuple)
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@patch("phenotypic.tools_._plotly_helpers.PLOTLY_AVAILABLE", False)
class TestMplFallbackImageShow:
    """Test matplotlib fallback for Image.show()."""

    def test_image_show_returns_mpl(self, grid_image):
        result = grid_image.show()
        assert isinstance(result, tuple)
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@patch("phenotypic.tools_._plotly_helpers.PLOTLY_AVAILABLE", False)
class TestMplFallbackShowOverlay:
    """Test matplotlib fallback for show_overlay()."""

    def test_show_overlay_returns_mpl(self, grid_image):
        result = grid_image.rgb.show_overlay()
        assert isinstance(result, tuple)
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_show_overlay_with_labels_returns_mpl(self, grid_image):
        result = grid_image.rgb.show_overlay(show_labels=True)
        assert isinstance(result, tuple)
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@patch("phenotypic.tools_._plotly_helpers.PLOTLY_AVAILABLE", False)
class TestMplFallbackOverlayPlotter:
    """Test matplotlib fallback for image.plot.overlay()."""

    def test_overlay_plotter_returns_mpl(self, grid_image):
        result = grid_image.plot.overlay()
        assert isinstance(result, tuple)
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_overlay_plotter_with_labels_returns_mpl(self, grid_image):
        result = grid_image.plot.overlay(show_labels=True)
        assert isinstance(result, tuple)
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
