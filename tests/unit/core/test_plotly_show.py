"""Tests for show()/dash() methods across accessors and Image.

Verifies that accessor show() methods always return (plt.Figure, plt.Axes)
and dash() methods always return plotly.graph_objects.Figure.
"""

import matplotlib.pyplot as plt
import pytest

go = pytest.importorskip("plotly.graph_objects")

from phenotypic import GridImage  # noqa: E402
from phenotypic.data import load_synth_yeast_plate  # noqa: E402


@pytest.fixture(scope="module")
def grid_image():
    """Load a GridImage with detected objects for reuse across tests."""
    return load_synth_yeast_plate()


# ---------------------------------------------------------------------------
# Accessor show() tests — always matplotlib
# ---------------------------------------------------------------------------


class TestRGBShow:
    """Tests for image.rgb.show() returning matplotlib tuple."""

    def test_rgb_show_returns_mpl(self, grid_image):
        fig, ax = grid_image.rgb.show()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_rgb_show_with_title(self, grid_image):
        fig, ax = grid_image.rgb.show(title="Test Title")
        assert isinstance(fig, plt.Figure)
        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_rgb_show_with_figsize(self, grid_image):
        fig, ax = grid_image.rgb.show(figsize=(10, 8))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_rgb_show_foreground_only(self, grid_image):
        fig, ax = grid_image.rgb.show(foreground_only=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestGrayShow:
    """Tests for image.gray.show() returning matplotlib tuple."""

    def test_gray_show_returns_mpl(self, grid_image):
        fig, ax = grid_image.gray.show()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_gray_show_with_cmap(self, grid_image):
        fig, ax = grid_image.gray.show(cmap="viridis")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_gray_show_foreground_only(self, grid_image):
        fig, ax = grid_image.gray.show(foreground_only=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestDetectMatShow:
    """Tests for image.detect_mat.show() returning matplotlib tuple."""

    def test_detect_mat_show_returns_mpl(self, grid_image):
        fig, ax = grid_image.detect_mat.show()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Accessor dash() tests — always plotly
# ---------------------------------------------------------------------------


class TestRGBDash:
    """Tests for image.rgb.dash() returning plotly Figure."""

    def test_rgb_dash_returns_figure(self, grid_image):
        fig = grid_image.rgb.dash()
        assert isinstance(fig, go.Figure)

    def test_rgb_dash_with_channel(self, grid_image):
        fig = grid_image.rgb.dash(channel=0)
        assert isinstance(fig, go.Figure)

    def test_rgb_dash_with_title(self, grid_image):
        fig = grid_image.rgb.dash(title="Test Title")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Test Title"

    def test_rgb_dash_with_figsize(self, grid_image):
        fig = grid_image.rgb.dash(figsize=(10, 8))
        assert isinstance(fig, go.Figure)
        assert fig.layout.width == 1000
        assert fig.layout.height == 800

    def test_rgb_dash_foreground_only(self, grid_image):
        fig = grid_image.rgb.dash(foreground_only=True)
        assert isinstance(fig, go.Figure)


class TestGrayDash:
    """Tests for image.gray.dash() returning plotly Figure."""

    def test_gray_dash_returns_figure(self, grid_image):
        fig = grid_image.gray.dash()
        assert isinstance(fig, go.Figure)

    def test_gray_dash_with_cmap(self, grid_image):
        fig = grid_image.gray.dash(cmap="viridis")
        assert isinstance(fig, go.Figure)

    def test_gray_dash_with_figsize(self, grid_image):
        fig = grid_image.gray.dash(figsize=(12, 6))
        assert isinstance(fig, go.Figure)
        assert fig.layout.width == 1200
        assert fig.layout.height == 600


class TestDetectMatDash:
    """Tests for image.detect_mat.dash() returning plotly Figure."""

    def test_detect_mat_dash_returns_figure(self, grid_image):
        fig = grid_image.detect_mat.dash()
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# show(overlay=True) tests
# ---------------------------------------------------------------------------


class TestShowWithOverlay:
    """Tests for accessor show(overlay=True) returning matplotlib tuple."""

    def test_show_overlay_returns_mpl(self, grid_image):
        fig, ax = grid_image.rgb.show(overlay=True)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_show_overlay_with_labels(self, grid_image):
        fig, ax = grid_image.rgb.show(overlay=True, show_labels=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_overlay_with_object_label(self, grid_image):
        fig, ax = grid_image.rgb.show(overlay=True, object_label=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_overlay_gridimage(self, grid_image):
        assert isinstance(grid_image, GridImage)
        fig, ax = grid_image.rgb.show(
            overlay=True, show_grid=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# dash(overlay=True) tests
# ---------------------------------------------------------------------------


class TestDashWithOverlay:
    """Tests for accessor dash(overlay=True) returning plotly Figure."""

    def test_dash_overlay_returns_figure(self, grid_image):
        fig = grid_image.rgb.dash(overlay=True)
        assert isinstance(fig, go.Figure)

    def test_dash_overlay_with_labels(self, grid_image):
        fig = grid_image.rgb.dash(overlay=True, show_labels=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0

    def test_dash_overlay_with_object_label(self, grid_image):
        fig = grid_image.rgb.dash(overlay=True, object_label=1)
        assert isinstance(fig, go.Figure)

    def test_dash_overlay_gridimage_has_shapes(self, grid_image):
        assert isinstance(grid_image, GridImage)
        fig = grid_image.rgb.dash(
            overlay=True, show_grid=True,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.shapes) > 0

    def test_dash_overlay_has_axis_ticks(self, grid_image):
        assert isinstance(grid_image, GridImage)
        fig = grid_image.rgb.dash(
            overlay=True, show_grid=True,
        )
        assert fig.layout.xaxis.tickmode == "array"
        assert fig.layout.yaxis.tickmode == "array"
        assert len(fig.layout.xaxis.tickvals) > 0
        assert len(fig.layout.yaxis.tickvals) > 0
        assert fig.layout.xaxis.side == "top"
        assert fig.layout.yaxis.side == "right"


# ---------------------------------------------------------------------------
# Image.show(overlay=True) / Image.dash(overlay=True) — auto-pick accessor
# ---------------------------------------------------------------------------


class TestImageShow:
    """Tests for Image.show() delegation returning matplotlib tuple."""

    def test_image_show_returns_mpl(self, grid_image):
        fig, ax = grid_image.show()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_image_show_with_title(self, grid_image):
        fig, ax = grid_image.show(title="Plate Overview")
        assert isinstance(fig, plt.Figure)
        assert ax.get_title() == "Plate Overview"
        plt.close(fig)

    def test_image_show_overlay(self, grid_image):
        fig, ax = grid_image.show(overlay=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_image_show_overlay_with_grid(self, grid_image):
        """Verify grid features flow through Image.show(overlay=True)."""
        assert isinstance(grid_image, GridImage)
        fig, ax = grid_image.show(
            overlay=True, show_grid=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestImageDash:
    """Tests for Image.dash() delegation returning plotly Figure."""

    def test_image_dash_returns_figure(self, grid_image):
        fig = grid_image.dash()
        assert isinstance(fig, go.Figure)

    def test_image_dash_with_title(self, grid_image):
        fig = grid_image.dash(title="Plate Overview")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Plate Overview"

    def test_image_dash_overlay(self, grid_image):
        fig = grid_image.dash(overlay=True)
        assert isinstance(fig, go.Figure)

    def test_image_dash_overlay_with_grid(self, grid_image):
        """Verify grid features flow through Image.dash(overlay=True)."""
        assert isinstance(grid_image, GridImage)
        fig = grid_image.dash(
            overlay=True, show_grid=True,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.shapes) > 0


# ---------------------------------------------------------------------------
# Graceful overlay fallback — no objects detected
# ---------------------------------------------------------------------------


class TestOverlayNoObjects:
    """Test overlay gracefully falls back when no objects detected."""

    @pytest.fixture()
    def empty_image(self):
        from phenotypic import Image
        import numpy as np
        return Image(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_show_overlay_no_objects(self, empty_image):
        fig, ax = empty_image.show(overlay=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dash_overlay_no_objects(self, empty_image):
        fig = empty_image.dash(overlay=True)
        assert isinstance(fig, go.Figure)

    def test_accessor_show_overlay_no_objects(self, empty_image):
        fig, ax = empty_image.rgb.show(overlay=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_accessor_dash_overlay_no_objects(self, empty_image):
        fig = empty_image.rgb.dash(overlay=True)
        assert isinstance(fig, go.Figure)
