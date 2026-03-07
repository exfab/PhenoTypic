"""Tests for plotly-based show() methods across accessors and Image.

Verifies that accessor show() methods, show_overlay(), and OverlayPlotter
all return plotly.graph_objects.Figure instances with correct structure.
"""

import pytest
import plotly.graph_objects as go

from phenotypic import GridImage
from phenotypic.data import load_synth_yeast_plate


@pytest.fixture(scope="module")
def grid_image():
    """Load a GridImage with detected objects for reuse across tests."""
    return load_synth_yeast_plate()


# ---------------------------------------------------------------------------
# Accessor show() tests
# ---------------------------------------------------------------------------


class TestRGBShow:
    """Tests for image.rgb.show() returning plotly Figure."""

    def test_rgb_show_returns_figure(self, grid_image):
        """Test image.rgb.show() returns a plotly Figure."""
        fig = grid_image.rgb.show()
        assert isinstance(fig, go.Figure)

    def test_rgb_show_with_channel(self, grid_image):
        """Test image.rgb.show(channel=0) returns a plotly Figure."""
        fig = grid_image.rgb.show(channel=0)
        assert isinstance(fig, go.Figure)

    def test_rgb_show_with_title(self, grid_image):
        """Test image.rgb.show() with title parameter."""
        fig = grid_image.rgb.show(title="Test Title")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Test Title"

    def test_rgb_show_with_figsize(self, grid_image):
        """Test image.rgb.show() with custom figsize sets pixel dimensions."""
        fig = grid_image.rgb.show(figsize=(10, 8))
        assert isinstance(fig, go.Figure)
        # figsize is in inches, converted at 100 dpi
        assert fig.layout.width == 1000
        assert fig.layout.height == 800

    def test_rgb_show_foreground_only(self, grid_image):
        """Test image.rgb.show(foreground_only=True) returns a Figure."""
        fig = grid_image.rgb.show(foreground_only=True)
        assert isinstance(fig, go.Figure)


class TestGrayShow:
    """Tests for image.gray.show() returning plotly Figure."""

    def test_gray_show_returns_figure(self, grid_image):
        """Test image.gray.show() returns a plotly Figure."""
        fig = grid_image.gray.show()
        assert isinstance(fig, go.Figure)

    def test_gray_show_with_cmap(self, grid_image):
        """Test image.gray.show() with custom colormap."""
        fig = grid_image.gray.show(cmap="viridis")
        assert isinstance(fig, go.Figure)

    def test_gray_show_foreground_only(self, grid_image):
        """Test image.gray.show(foreground_only=True) returns a Figure."""
        fig = grid_image.gray.show(foreground_only=True)
        assert isinstance(fig, go.Figure)

    def test_gray_show_with_figsize(self, grid_image):
        """Test image.gray.show() with figsize sets pixel dimensions."""
        fig = grid_image.gray.show(figsize=(12, 6))
        assert isinstance(fig, go.Figure)
        assert fig.layout.width == 1200
        assert fig.layout.height == 600


class TestDetectMatShow:
    """Tests for image.detect_mat.show() returning plotly Figure."""

    def test_detect_mat_show_returns_figure(self, grid_image):
        """Test image.detect_mat.show() returns a plotly Figure."""
        fig = grid_image.detect_mat.show()
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# show_overlay() tests
# ---------------------------------------------------------------------------


class TestShowOverlay:
    """Tests for accessor show_overlay() method."""

    def test_show_overlay_returns_figure(self, grid_image):
        """Test image.rgb.show_overlay() returns a plotly Figure."""
        fig = grid_image.rgb.show_overlay()
        assert isinstance(fig, go.Figure)

    def test_show_overlay_with_labels(self, grid_image):
        """Test show_overlay(show_labels=True) adds annotations."""
        fig = grid_image.rgb.show_overlay(show_labels=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0

    def test_show_overlay_with_object_label(self, grid_image):
        """Test show_overlay(object_label=1) returns a Figure."""
        fig = grid_image.rgb.show_overlay(object_label=1)
        assert isinstance(fig, go.Figure)

    def test_show_overlay_gridimage_has_shapes(self, grid_image):
        """Test show_overlay on GridImage adds gridlines as shapes."""
        assert isinstance(grid_image, GridImage)
        fig = grid_image.rgb.show_overlay(
            show_gridlines=True, show_section_boxes=True
        )
        assert isinstance(fig, go.Figure)
        # Gridlines and section boxes are drawn as shapes
        assert len(fig.layout.shapes) > 0


# ---------------------------------------------------------------------------
# OverlayPlotter tests
# ---------------------------------------------------------------------------


class TestOverlayPlotter:
    """Tests for image.plot.overlay() via OverlayPlotter."""

    def test_overlay_returns_figure(self, grid_image):
        """Test image.plot.overlay() returns a plotly Figure."""
        fig = grid_image.plot.overlay()
        assert isinstance(fig, go.Figure)

    def test_overlay_with_labels(self, grid_image):
        """Test image.plot.overlay(show_labels=True) adds annotations."""
        fig = grid_image.plot.overlay(show_labels=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0


# ---------------------------------------------------------------------------
# Image.show() tests
# ---------------------------------------------------------------------------


class TestImageShow:
    """Tests for Image.show() delegation."""

    def test_image_show_returns_figure(self, grid_image):
        """Test image.show() returns a plotly Figure (delegates to rgb)."""
        fig = grid_image.show()
        assert isinstance(fig, go.Figure)

    def test_image_show_with_title(self, grid_image):
        """Test image.show() passes kwargs through to accessor."""
        fig = grid_image.show(title="Plate Overview")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Plate Overview"
