"""Tests for plotly helper functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

go = pytest.importorskip("plotly.graph_objects")

from phenotypic.tools_._plotly_helpers import (  # noqa: E402
    PLOTLY_CONFIG,
    add_plotly_gridlines,
    add_plotly_obj_labels,
    add_plotly_section_boxes,
    mpl_cmap_to_plotly,
    plotly_imshow,
)


# ---------------------------------------------------------------------------
# mpl_cmap_to_plotly
# ---------------------------------------------------------------------------


class TestMplCmapToPlotly:
    def test_known_colormap_gray(self):
        assert mpl_cmap_to_plotly("gray") == "gray"

    def test_known_colormap_viridis(self):
        assert mpl_cmap_to_plotly("viridis") == "Viridis"

    def test_known_colormap_coolwarm(self):
        assert mpl_cmap_to_plotly("coolwarm") == "RdBu"

    def test_unknown_colormap_returns_list(self):
        result = mpl_cmap_to_plotly("spring")
        assert isinstance(result, list)
        assert len(result) == 256
        # Each entry is [position, "rgb(r,g,b)"]
        assert result[0][0] == 0.0
        assert result[-1][0] == 1.0
        assert result[0][1].startswith("rgb(")

    def test_unknown_colormap_invalid_raises(self):
        with pytest.raises(KeyError):
            mpl_cmap_to_plotly("not_a_real_colormap_xyz")


# ---------------------------------------------------------------------------
# plotly_imshow
# ---------------------------------------------------------------------------


class TestPlotlyImshow:
    def test_uint8_rgb(self):
        arr = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
        fig = plotly_imshow(arr, title="RGB uint8")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "RGB uint8"

    def test_uint16_rgb(self):
        arr = np.random.randint(0, 65535, (50, 60, 3), dtype=np.uint16)
        fig = plotly_imshow(arr)
        assert isinstance(fig, go.Figure)

    def test_float32_rgb(self):
        arr = np.random.rand(50, 60, 3).astype(np.float32)
        fig = plotly_imshow(arr)
        assert isinstance(fig, go.Figure)

    def test_uint8_grayscale(self):
        arr = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        fig = plotly_imshow(arr, cmap="viridis")
        assert isinstance(fig, go.Figure)

    def test_uint16_grayscale(self):
        arr = np.random.randint(0, 65535, (50, 60), dtype=np.uint16)
        fig = plotly_imshow(arr)
        assert isinstance(fig, go.Figure)

    def test_float32_grayscale(self):
        arr = np.random.rand(50, 60).astype(np.float32)
        fig = plotly_imshow(arr, cmap="inferno")
        assert isinstance(fig, go.Figure)

    def test_custom_figsize(self):
        arr = np.zeros((50, 60), dtype=np.uint8)
        fig = plotly_imshow(arr, figsize=(10, 8))
        assert fig.layout.width == 1000
        assert fig.layout.height == 800

    def test_auto_figsize(self):
        arr = np.zeros((50, 60), dtype=np.uint8)
        fig = plotly_imshow(arr)
        assert fig.layout.width is None
        assert fig.layout.autosize is True

    def test_layout_defaults(self):
        arr = np.zeros((50, 60, 3), dtype=np.uint8)
        fig = plotly_imshow(arr)
        assert fig.layout.dragmode == "zoom"
        assert fig.layout.xaxis.showticklabels is False
        assert fig.layout.yaxis.showticklabels is False
        assert fig.layout.yaxis.scaleanchor == "x"

    def test_no_title(self):
        arr = np.zeros((50, 60, 3), dtype=np.uint8)
        fig = plotly_imshow(arr)
        assert fig.layout.title.text is None

    def test_config_constant(self):
        assert PLOTLY_CONFIG == {"scrollZoom": True}


# ---------------------------------------------------------------------------
# add_plotly_gridlines
# ---------------------------------------------------------------------------


class TestAddPlotlyGridlines:
    def test_adds_shapes_and_axis_ticks(self):
        fig = go.Figure()
        col_edges = np.array([0, 100, 200, 300])
        row_edges = np.array([0, 150, 300])
        add_plotly_gridlines(fig, col_edges, row_edges, ncols=3, nrows=2)

        # 4 vertical + 3 horizontal = 7 line shapes
        line_shapes = [s for s in fig.layout.shapes if s.type == "line"]
        assert len(line_shapes) == 7

        # Labels are now axis tick labels, not annotations
        assert len(fig.layout.annotations) == 0

        # Column labels on the top axis
        assert fig.layout.xaxis.tickmode == "array"
        assert list(fig.layout.xaxis.tickvals) == [50.0, 150.0, 250.0]
        assert list(fig.layout.xaxis.ticktext) == ["0", "1", "2"]
        assert fig.layout.xaxis.showticklabels is True
        assert fig.layout.xaxis.side == "top"
        assert fig.layout.xaxis.automargin is True

        # Row labels on the right axis
        assert fig.layout.yaxis.tickmode == "array"
        assert list(fig.layout.yaxis.tickvals) == [75.0, 225.0]
        assert list(fig.layout.yaxis.ticktext) == ["0", "1"]
        assert fig.layout.yaxis.showticklabels is True
        assert fig.layout.yaxis.side == "right"
        assert fig.layout.yaxis.automargin is True

        # Regression guard: image aspect lock preserved
        assert fig.layout.yaxis.scaleanchor == "x"

    def test_empty_edges_no_shapes(self):
        fig = go.Figure()
        add_plotly_gridlines(fig, np.array([]), np.array([]), ncols=0, nrows=0)
        assert len(fig.layout.shapes) == 0
        assert len(fig.layout.annotations) == 0

    def test_single_edge_no_labels(self):
        fig = go.Figure()
        col_edges = np.array([50])
        row_edges = np.array([50])
        add_plotly_gridlines(fig, col_edges, row_edges, ncols=0, nrows=0)
        # 1 vertical + 1 horizontal = 2 shapes
        assert len(fig.layout.shapes) == 2
        # No labels (need at least 2 edges for centers)
        assert len(fig.layout.annotations) == 0
        # Axis tick config is not touched on the single-edge fallback
        assert fig.layout.xaxis.tickmode is None
        assert fig.layout.yaxis.tickmode is None


# ---------------------------------------------------------------------------
# add_plotly_section_boxes
# ---------------------------------------------------------------------------


class TestAddPlotlySectionBoxes:
    def test_draws_per_section_boxes(self):
        fig = go.Figure()
        # 2x2 grid, all sections populated
        min_rr = np.array([10.0, 10.0, 110.0, 110.0])
        max_rr = np.array([50.0, 50.0, 150.0, 150.0])
        min_cc = np.array([10.0, 110.0, 10.0, 110.0])
        max_cc = np.array([50.0, 150.0, 50.0, 150.0])
        add_plotly_section_boxes(fig, min_rr, max_rr, min_cc, max_cc)

        rect_shapes = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rect_shapes) == 4

    def test_skips_nan_sections(self):
        fig = go.Figure()
        min_rr = np.array([10.0, np.nan, 110.0, 110.0])
        max_rr = np.array([50.0, np.nan, 150.0, 150.0])
        min_cc = np.array([10.0, np.nan, 10.0, 110.0])
        max_cc = np.array([50.0, np.nan, 50.0, 150.0])
        add_plotly_section_boxes(fig, min_rr, max_rr, min_cc, max_cc)

        rect_shapes = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rect_shapes) == 3

    def test_all_nan_no_shapes(self):
        fig = go.Figure()
        nan4 = np.full(4, np.nan)
        add_plotly_section_boxes(fig, nan4, nan4.copy(), nan4.copy(), nan4.copy())
        assert len(fig.layout.shapes) == 0

    def test_empty_arrays(self):
        fig = go.Figure()
        empty = np.array([])
        add_plotly_section_boxes(fig, empty, empty.copy(), empty.copy(), empty.copy())
        assert len(fig.layout.shapes) == 0

    def test_rect_coords_match(self):
        fig = go.Figure()
        min_rr = np.array([11.0, 22.0])
        max_rr = np.array([33.0, 44.0])
        min_cc = np.array([55.0, 66.0])
        max_cc = np.array([77.0, 88.0])
        add_plotly_section_boxes(fig, min_rr, max_rr, min_cc, max_cc)

        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == 2
        assert rects[0].x0 == 55.0 and rects[0].x1 == 77.0
        assert rects[0].y0 == 11.0 and rects[0].y1 == 33.0
        assert rects[1].x0 == 66.0 and rects[1].x1 == 88.0
        assert rects[1].y0 == 22.0 and rects[1].y1 == 44.0


# ---------------------------------------------------------------------------
# add_plotly_obj_labels
# ---------------------------------------------------------------------------


class TestAddPlotlyObjLabels:
    def _make_mock_image(self, n_objects: int = 3):
        mock = MagicMock()
        props = []
        labels = []
        for i in range(n_objects):
            prop = MagicMock()
            prop.label = i + 1
            prop.centroid = (50.0 + i * 10, 100.0 + i * 20)
            props.append(prop)
            labels.append(i + 1)
        mock.objects.props = props
        mock.objects.labels = labels
        return mock

    def test_labels_all_objects(self):
        fig = go.Figure()
        mock_image = self._make_mock_image(5)
        add_plotly_obj_labels(fig, mock_image)
        assert len(fig.layout.annotations) == 5

    def test_label_single_object(self):
        fig = go.Figure()
        mock_image = self._make_mock_image(5)
        add_plotly_obj_labels(fig, mock_image, object_label=3)
        assert len(fig.layout.annotations) == 1
        assert fig.layout.annotations[0].text == "3"

    def test_label_not_found(self):
        fig = go.Figure()
        mock_image = self._make_mock_image(3)
        add_plotly_obj_labels(fig, mock_image, object_label=99)
        assert len(fig.layout.annotations) == 0

    def test_annotation_properties(self):
        fig = go.Figure()
        mock_image = self._make_mock_image(1)
        add_plotly_obj_labels(
            fig, mock_image, color="red", size=14, bgcolor="blue"
        )
        ann = fig.layout.annotations[0]
        assert ann.font.color == "red"
        assert ann.font.size == 14
        assert ann.bgcolor == "blue"
        assert ann.opacity == 0.6
        assert ann.showarrow is False

    def test_centroid_positions(self):
        fig = go.Figure()
        mock_image = self._make_mock_image(1)
        add_plotly_obj_labels(fig, mock_image)
        ann = fig.layout.annotations[0]
        assert ann.x == 100.0
        assert ann.y == 50.0
