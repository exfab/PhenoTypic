"""Unit tests for the pure ``build_heatmap_figure`` builder.

The figure builder is a side-effect-free pure function (no Dash imports)
so each test constructs a synthetic polars/pandas frame inline, calls
``build_heatmap_figure``, and asserts on the returned
``plotly.graph_objects.Figure``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytest

from phenotypic.gui._design import OI_VERMILION
from phenotypic.gui.results_viewer._heatmap_tab._figure import build_heatmap_figure
from phenotypic.schema import IMAGE


def _make_minimal_frame(
    *,
    rows: int = 2,
    cols: int = 2,
    image_files: tuple[str, ...] = ("image_a.tif",),
    add_time: bool = False,
    add_label: bool = True,
) -> pl.DataFrame:
    """Build a synthetic per-well frame with one row per (image, r, c[, t])."""
    records: list[dict[str, object]] = []
    label_counter = 0
    for img in image_files:
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                label_counter += 1
                rec: dict[str, object] = {
                    str(IMAGE.IMAGE_NAME): img,
                    "Grid_RowNum": r,
                    "Grid_ColNum": c,
                    "Size_Area": float(100 + r * 10 + c),
                }
                if add_label:
                    rec["Object_Label"] = label_counter
                if add_time:
                    rec["Metadata_Time"] = 4
                records.append(rec)
    return pl.from_dicts(records)


class TestBasicShape:
    """Round-trip a one-image, multi-well frame through the builder."""

    def test_returns_plotly_figure(self) -> None:
        frame = _make_minimal_frame(rows=2, cols=3)
        fig = build_heatmap_figure(
            frame,
            color_col="Size_Area",
            image_file="image_a.tif",
            time_value=None,
            aggregator="mean",
            removed_keys=set(),
        )
        assert isinstance(fig, go.Figure)
        # One data trace (no removals): the heatmap.
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"


class TestEmptyStateNoGridColumns:
    """When grid columns are missing, return a placeholder figure."""

    def test_empty_state_when_grid_columns_absent(self) -> None:
        frame = pl.DataFrame(
            {
                str(IMAGE.IMAGE_NAME): ["image_a.tif"] * 4,
                "Object_Label": list(range(1, 5)),
                "Size_Area": [10.0, 20.0, 30.0, 40.0],
            }
        )
        fig = build_heatmap_figure(
            frame,
            color_col="Size_Area",
            image_file="image_a.tif",
            time_value=None,
            aggregator="mean",
            removed_keys=set(),
        )
        # No data traces; an annotation explains the missing columns.
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert len(fig.layout.annotations) >= 1
        assert "Grid" in fig.layout.annotations[0].text


class TestImageFilterAppliedBeforeAggregator:
    """Image filter must run before the aggregator, not after."""

    def test_image_filter_applied_before_aggregator(self) -> None:
        # Two images, each carrying two rows at (Grid_RowNum=1, Grid_ColNum=1).
        # image_a values: [1, 5]; image_b values: [100, 200].
        # If the aggregator runs before the filter, max across both
        # images at (1,1) is 200. After filtering to image_a only, the
        # max should be 5.
        frame = pl.DataFrame(
            {
                str(IMAGE.IMAGE_NAME): [
                    "image_a.tif",
                    "image_a.tif",
                    "image_b.tif",
                    "image_b.tif",
                ],
                "Object_Label": [1, 2, 3, 4],
                "Grid_RowNum": [1, 1, 1, 1],
                "Grid_ColNum": [1, 1, 1, 1],
                "Size_Area": [1.0, 5.0, 100.0, 200.0],
            }
        )
        fig = build_heatmap_figure(
            frame,
            color_col="Size_Area",
            image_file="image_a.tif",
            time_value=None,
            aggregator="max",
            removed_keys=set(),
        )
        assert len(fig.data) == 1
        z = np.asarray(fig.data[0].z, dtype=float)
        # Single (1,1) cell: image_a-only max is 5, not 200.
        assert z.shape == (1, 1)
        assert z[0, 0] == 5.0


class TestAggregatorSemantics:
    """``mean`` and ``max`` produce different values for the same bin."""

    def test_aggregator_mean_vs_max_changes_value(self) -> None:
        frame = pl.DataFrame(
            {
                str(IMAGE.IMAGE_NAME): ["image_a.tif"] * 2,
                "Object_Label": [1, 2],
                "Grid_RowNum": [1, 1],
                "Grid_ColNum": [1, 1],
                "Size_Area": [1.0, 3.0],
            }
        )
        common: dict[str, object] = {
            "color_col": "Size_Area",
            "image_file": "image_a.tif",
            "time_value": None,
            "removed_keys": set(),
        }
        fig_mean = build_heatmap_figure(frame, aggregator="mean", **common)  # type: ignore[arg-type]
        fig_max = build_heatmap_figure(frame, aggregator="max", **common)  # type: ignore[arg-type]
        z_mean = np.asarray(fig_mean.data[0].z, dtype=float)
        z_max = np.asarray(fig_max.data[0].z, dtype=float)
        assert z_mean[0, 0] == pytest.approx(2.0)
        assert z_max[0, 0] == pytest.approx(3.0)


class TestRemovedOverlay:
    """Removed cells render as a second overlay trace."""

    def test_removed_cells_render_overlay_trace(self) -> None:
        frame = _make_minimal_frame(rows=2, cols=2)
        # Object_Label 1 is the (image_a.tif, 1, 1) cell per the helper.
        # Make sure that's the one we're targeting:
        # the helper labels in order rows->cols, so (img_a, r=1, c=1)
        # gets label 1.
        removed = {("image_a.tif", 1)}
        fig = build_heatmap_figure(
            frame,
            color_col="Size_Area",
            image_file="image_a.tif",
            time_value=None,
            aggregator="mean",
            removed_keys=removed,
        )
        # >= 2 traces: data heatmap + overlay (heatmap and/or scatter).
        assert len(fig.data) >= 2

    def test_removed_overlay_uses_vermilion(self) -> None:
        frame = _make_minimal_frame(rows=2, cols=2)
        removed = {("image_a.tif", 1)}
        fig = build_heatmap_figure(
            frame,
            color_col="Size_Area",
            image_file="image_a.tif",
            time_value=None,
            aggregator="mean",
            removed_keys=removed,
        )
        # Locate the scatter overlay (only the overlay uses Scatter; the
        # data heatmap is a Heatmap trace). Removed/excluded cells render in
        # the spec's failed/null color (vermilion), not grey (DESIGN.md "06"/"10").
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        assert scatter_traces, "Expected at least one overlay Scatter trace"
        scatter = scatter_traces[0]
        # ``marker.color`` may be either a plain hex string or a list-of-hex.
        marker_color = scatter.marker.color
        if isinstance(marker_color, (list, tuple)):
            assert all(c == OI_VERMILION for c in marker_color)
        else:
            assert marker_color == OI_VERMILION


class TestTimeFilter:
    """When ``time_value`` is supplied, only rows at that time contribute."""

    def test_time_filter_applied(self) -> None:
        frame = pl.DataFrame(
            {
                str(IMAGE.IMAGE_NAME): ["image_a.tif"] * 4,
                "Object_Label": [1, 2, 3, 4],
                "Grid_RowNum": [1, 1, 1, 1],
                "Grid_ColNum": [1, 1, 1, 1],
                "Metadata_Time": [4, 4, 8, 8],
                "Size_Area": [1.0, 3.0, 100.0, 200.0],
            }
        )
        fig = build_heatmap_figure(
            frame,
            color_col="Size_Area",
            image_file="image_a.tif",
            time_value=4,
            aggregator="max",
            removed_keys=set(),
        )
        z = np.asarray(fig.data[0].z, dtype=float)
        assert z.shape == (1, 1)
        # Max of [1.0, 3.0] at time=4 is 3.0, not 200.0.
        assert z[0, 0] == 3.0


class TestNanOnlyPivot:
    """All-NaN values should not crash the builder."""

    def test_nan_only_pivot_does_not_crash(self) -> None:
        frame = pl.DataFrame(
            {
                str(IMAGE.IMAGE_NAME): ["image_a.tif"] * 4,
                "Object_Label": [1, 2, 3, 4],
                "Grid_RowNum": [1, 1, 2, 2],
                "Grid_ColNum": [1, 2, 1, 2],
                "Size_Area": [None, None, None, None],
            },
            schema={
                str(IMAGE.IMAGE_NAME): pl.String,
                "Object_Label": pl.Int64,
                "Grid_RowNum": pl.Int64,
                "Grid_ColNum": pl.Int64,
                "Size_Area": pl.Float64,
            },
        )
        fig = build_heatmap_figure(
            frame,
            color_col="Size_Area",
            image_file="image_a.tif",
            time_value=None,
            aggregator="mean",
            removed_keys=set(),
        )
        # Did not raise; figure is well-formed.
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1


class TestPandasInputAccepted:
    """Pandas input is converted to polars at the top of the builder."""

    def test_accepts_pandas_dataframe_input(self) -> None:
        frame = pd.DataFrame(
            {
                str(IMAGE.IMAGE_NAME): ["image_a.tif"] * 4,
                "Object_Label": [1, 2, 3, 4],
                "Grid_RowNum": [1, 1, 2, 2],
                "Grid_ColNum": [1, 2, 1, 2],
                "Size_Area": [1.0, 2.0, 3.0, 4.0],
            }
        )
        fig = build_heatmap_figure(
            frame,
            color_col="Size_Area",
            image_file="image_a.tif",
            time_value=None,
            aggregator="mean",
            removed_keys=set(),
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        z = np.asarray(fig.data[0].z, dtype=float)
        # 2x2 grid with values 1..4 - no aggregation collapse needed.
        assert z.shape == (2, 2)
        assert set(z.ravel().tolist()) == {1.0, 2.0, 3.0, 4.0}
