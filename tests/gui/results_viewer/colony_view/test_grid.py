"""Unit tests for :mod:`phenotypic.gui.results_viewer.colony_view._grid`.

Exercises the pure helpers:

- ``selectable_axis_columns`` filters by prefix and cardinality.
- ``compute_max_bbox_size`` honours min crop size + padding.
- ``expand_range`` is direction-agnostic and slices grid_order correctly.

``build_grid`` is exercised lightly here because its output is a Dash
component tree without a public-facing schema. We assert on its
``grid_order`` companion (the row-major key list) as the testable invariant.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view._grid import (
    build_grid,
    compute_max_bbox_size,
    expand_range,
    selectable_axis_columns,
)

from tests._output_layout import write_master


# -------------------------------------------------------------------------
# selectable_axis_columns
# -------------------------------------------------------------------------


def test_selectable_axis_columns_excludes_measurement_prefixes() -> None:
    """Columns with measurement prefixes are not offered as axis options."""
    df = pl.DataFrame(
        {
            "Metadata_Strain": ["A", "B", "A"],
            "Bbox_MinRR": [1, 2, 3],
            "Shape_Area": [10, 20, 30],
            "Intensity_Mean": [0.1, 0.2, 0.3],
            "TextureGray_AvgContrast": [0.0, 0.1, 0.2],
            "Grid_RowNum": [1, 2, 1],
        }
    )
    column_value_sets = {col: sorted(df[col].cast(pl.String).unique().to_list()) for col in df.columns}

    out = selectable_axis_columns(df, column_value_sets)

    assert "Bbox_MinRR" not in out
    assert "Shape_Area" not in out
    assert "Intensity_Mean" not in out
    assert "TextureGray_AvgContrast" not in out
    # Metadata_* and Grid_* survive.
    assert "Metadata_Strain" in out
    assert "Grid_RowNum" in out


def test_selectable_axis_columns_filters_by_cardinality() -> None:
    """Columns with cardinality 1 or > max_cardinality are dropped."""
    df = pl.DataFrame(
        {
            "Metadata_Constant": ["x"] * 60,                 # card 1 → drop
            "Metadata_HighCard": [str(i) for i in range(60)],  # card 60 → drop (> 50)
            "Metadata_OK": ["a", "b"] * 30,                   # card 2 → keep
        }
    )
    column_value_sets = {col: sorted(df[col].unique().to_list()) for col in df.columns}

    out = selectable_axis_columns(df, column_value_sets, max_cardinality=50)

    assert "Metadata_Constant" not in out  # cardinality 1
    assert "Metadata_HighCard" not in out  # cardinality 60 > 50
    assert "Metadata_OK" in out


def test_selectable_axis_columns_orders_metadata_first() -> None:
    """Metadata_* columns sort before Grid_* and other columns."""
    df = pl.DataFrame(
        {
            "Other_Col": ["a", "b"],
            "Metadata_Foo": ["1", "2"],
            "Grid_Bar": [1, 2],
            "Metadata_Aaa": ["x", "y"],
        }
    )
    column_value_sets = {col: sorted(df[col].cast(pl.String).unique().to_list()) for col in df.columns}

    out = selectable_axis_columns(df, column_value_sets)

    metadata_idx = [i for i, c in enumerate(out) if c.startswith("Metadata_")]
    grid_idx = [i for i, c in enumerate(out) if c.startswith("Grid_")]
    other_idx = [i for i, c in enumerate(out) if not (c.startswith("Metadata_") or c.startswith("Grid_"))]
    assert max(metadata_idx) < min(grid_idx)
    assert max(grid_idx) < min(other_idx)


# -------------------------------------------------------------------------
# compute_max_bbox_size
# -------------------------------------------------------------------------


def test_compute_max_bbox_size_returns_floor_for_empty_frame() -> None:
    """Empty frame returns the minimum (64) rather than failing."""
    empty = pl.DataFrame(
        schema={
            "Bbox_MinRR": pl.Int64,
            "Bbox_MaxRR": pl.Int64,
            "Bbox_MinCC": pl.Int64,
            "Bbox_MaxCC": pl.Int64,
        }
    )
    assert compute_max_bbox_size(empty) == 64


def test_compute_max_bbox_size_uses_max_extent_plus_padding() -> None:
    """Computed size = max(width, height) over rows, plus 2*padding."""
    df = pl.DataFrame(
        {
            "Bbox_MinRR": [0, 10, 100],
            "Bbox_MaxRR": [40, 50, 200],   # extents 40, 40, 100
            "Bbox_MinCC": [0, 5, 100],
            "Bbox_MaxCC": [30, 40, 150],   # extents 30, 35, 50
        }
    )
    # Max(rr, cc) extent = 100; default padding 8 → 100 + 16 = 116.
    assert compute_max_bbox_size(df) == 116


def test_compute_max_bbox_size_honours_minimum() -> None:
    """A tiny computed size is bumped up to the 64-pixel floor."""
    df = pl.DataFrame(
        {
            "Bbox_MinRR": [0],
            "Bbox_MaxRR": [10],
            "Bbox_MinCC": [0],
            "Bbox_MaxCC": [10],
        }
    )
    # Computed = 10 + 16 = 26 → bumped to 64.
    assert compute_max_bbox_size(df) == 64


# -------------------------------------------------------------------------
# expand_range
# -------------------------------------------------------------------------


def test_expand_range_inclusive_slice() -> None:
    """Returns the slice between anchor and target, inclusive."""
    order = [("a", 1), ("a", 2), ("b", 1), ("b", 2), ("c", 1)]
    assert expand_range(order, ("a", 2), ("b", 2)) == [
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]


def test_expand_range_is_direction_agnostic() -> None:
    """Reversing anchor/target yields the same slice."""
    order = [("a", 1), ("a", 2), ("b", 1), ("b", 2), ("c", 1)]
    assert expand_range(order, ("b", 2), ("a", 2)) == expand_range(
        order, ("a", 2), ("b", 2)
    )


def test_expand_range_single_cell() -> None:
    """anchor == target returns just that one element."""
    order = [("a", 1), ("a", 2), ("b", 1)]
    assert expand_range(order, ("a", 1), ("a", 1)) == [("a", 1)]


def test_expand_range_raises_for_unknown_key() -> None:
    """A key outside grid_order raises ``ValueError``."""
    order = [("a", 1), ("a", 2)]
    with pytest.raises(ValueError):
        expand_range(order, ("a", 1), ("z", 99))


# -------------------------------------------------------------------------
# build_grid (light integration)
# -------------------------------------------------------------------------


def _make_output_root(tmp_path: Path) -> OutputRoot:
    """Spin up a minimal OutputRoot pointing at a tmp folder.

    We only need ``master_df``, ``has_overlay``, and ``column_value_sets``
    plumbed for ``build_grid``; the rest of OutputRoot's API is exercised
    elsewhere.
    """
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["plate1"] * 4,
            "Metadata_ImageFile": ["img-001", "img-001", "img-002", "img-002"],
            "Object_Label": [1, 2, 1, 2],
            "Bbox_MinRR": [0, 5, 10, 15],
            "Bbox_MaxRR": [40, 45, 50, 55],
            "Bbox_MinCC": [0, 5, 10, 15],
            "Bbox_MaxCC": [40, 45, 50, 55],
            "Bbox_CenterRR": [20, 25, 30, 35],
            "Bbox_CenterCC": [20, 25, 30, 35],
            "Grid_RowNum": [1, 2, 1, 2],
            "Grid_ColNum": [1, 1, 2, 2],
        }
    )
    # Build the directory shell discover() expects, just enough to load
    # without crashing. Use the minimum: master parquet + an overlays dir
    # for the "plate1" dataset (we don't need actual PNGs here).
    (tmp_path / "results" / "plate1" / "overlays").mkdir(parents=True)
    write_master(tmp_path, master)
    return OutputRoot.discover(tmp_path)


def test_build_grid_returns_component_and_row_major_order(tmp_path: Path) -> None:
    """grid_order is row-major (Y-axis outer, X-axis inner)."""
    root = _make_output_root(tmp_path)
    df = root.master_df

    component, grid_order = build_grid(
        df=df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
    )

    assert component is not None
    # 2 X-values * 2 Y-values = 4 cells; each cell maps to one
    # representative (image_file, label).
    assert len(grid_order) == 4
    # Every key is a real (str, int) tuple.
    for img, label in grid_order:
        assert isinstance(img, str)
        assert isinstance(label, int)


# -------------------------------------------------------------------------
# Tile-spotlight dim threading (Phase 3)
# -------------------------------------------------------------------------


def _collect_img_srcs(component: object) -> list[str]:
    """Recursively collect every ``html.Img.src`` in a Dash component tree."""
    srcs: list[str] = []

    def _walk(node: object) -> None:
        src = getattr(node, "src", None)
        if isinstance(src, str):
            srcs.append(src)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                _walk(child)
        elif children is not None:
            _walk(children)

    _walk(component)
    return srcs


def _make_output_root_with_overlays(tmp_path: Path) -> OutputRoot:
    """Like ``_make_output_root`` but writes real overlay PNGs.

    ``build_grid`` only emits an ``<img>`` when ``OutputRoot.has_overlay``
    is True, which is backed by an on-disk scan — so the dim-threading test
    needs actual PNG files for every represented colony's image.
    """
    from PIL import Image as PILImage

    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["plate1"] * 4,
            "Metadata_ImageFile": ["img-001", "img-001", "img-002", "img-002"],
            "Object_Label": [1, 2, 1, 2],
            "Bbox_MinRR": [0, 5, 10, 15],
            "Bbox_MaxRR": [40, 45, 50, 55],
            "Bbox_MinCC": [0, 5, 10, 15],
            "Bbox_MaxCC": [40, 45, 50, 55],
            "Bbox_CenterRR": [20, 25, 30, 35],
            "Bbox_CenterCC": [20, 25, 30, 35],
            "Grid_RowNum": [1, 2, 1, 2],
            "Grid_ColNum": [1, 1, 2, 2],
        }
    )
    overlays = tmp_path / "results" / "plate1" / "overlays"
    overlays.mkdir(parents=True)
    for stem in ("img-001", "img-002"):
        PILImage.new("RGB", (64, 64), (200, 0, 0)).save(overlays / f"{stem}.png")
    write_master(tmp_path, master)
    return OutputRoot.discover(tmp_path)


def test_build_grid_threads_dim_alpha_into_tile_urls(tmp_path: Path) -> None:
    """Each tile ``<img src>`` carries ``&dim=<alpha>`` from ``build_grid``."""
    root = _make_output_root_with_overlays(tmp_path)
    alpha = 0.45

    component, grid_order = build_grid(
        df=root.master_df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
        dim_alpha=alpha,
    )

    srcs = _collect_img_srcs(component)
    assert len(srcs) == len(grid_order) == 4
    # Every tile URL threads the exact store alpha as the ``&dim=`` param.
    for src in srcs:
        assert "?size=" in src
        assert f"&dim={alpha}" in src


def test_build_grid_default_dim_alpha_is_zero(tmp_path: Path) -> None:
    """With no ``dim_alpha`` the URLs degrade to today's ``&dim=0.0``."""
    root = _make_output_root_with_overlays(tmp_path)

    component, _order = build_grid(
        df=root.master_df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
    )

    srcs = _collect_img_srcs(component)
    assert srcs
    for src in srcs:
        assert "&dim=0.0" in src


# -------------------------------------------------------------------------
# Radial category trigger + per-cell category badge (Phase 2, Task 4a)
# -------------------------------------------------------------------------


def _collect_buttons(component: object) -> list[object]:
    """Recursively collect every ``dbc.Button`` in a component tree."""
    import dash_bootstrap_components as dbc

    buttons: list[object] = []

    def _walk(node: object) -> None:
        if isinstance(node, dbc.Button):
            buttons.append(node)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                _walk(child)
        elif children is not None:
            _walk(children)

    _walk(component)
    return buttons


def test_build_grid_tiles_carry_radial_trigger_not_old_remove_button(
    tmp_path: Path,
) -> None:
    """Tiles render the radial ▾ trigger and not the legacy ✕ remove button."""
    root = _make_output_root(tmp_path)

    component, grid_order = build_grid(
        df=root.master_df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
    )

    buttons = _collect_buttons(component)
    trigger_types = {
        btn.id.get("type")  # type: ignore[union-attr]
        for btn in buttons
        if isinstance(btn.id, dict)
    }
    # The new radial trigger type is present...
    assert "colony-radial-trigger" in trigger_types
    # ...and the legacy single-cell remove button type is gone.
    assert "colony-cell-remove-btn" not in trigger_types
    # One trigger per representative cell.
    triggers = [
        btn
        for btn in buttons
        if isinstance(btn.id, dict) and btn.id.get("type") == "colony-radial-trigger"
    ]
    assert len(triggers) == len(grid_order) == 4


def test_build_grid_renders_category_badge_for_labeled_cell(tmp_path: Path) -> None:
    """A cell whose key is in ``category_of`` renders a colored category badge."""
    from phenotypic.gui._design import category_color

    root = _make_output_root(tmp_path)
    # Mark img-001/label-1 as ``debris``.
    category_of = {("img-001", 1): "debris"}

    component, _order = build_grid(
        df=root.master_df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
        category_of=category_of,
    )

    buttons = _collect_buttons(component)
    # The trigger for the labeled cell renders as a colored badge with the
    # category's display text and color.
    labeled_trigger = next(
        (
            btn
            for btn in buttons
            if isinstance(btn.id, dict)
            and btn.id.get("type") == "colony-radial-trigger"
            and btn.id.get("image_file") == "img-001"
            and btn.id.get("label") == 1
        ),
        None,
    )
    assert labeled_trigger is not None
    assert "radial-badge" in (labeled_trigger.className or "")  # type: ignore[union-attr]
    assert labeled_trigger.style.get("backgroundColor") == category_color("debris")  # type: ignore[union-attr]


def test_build_grid_default_category_of_renders_neutral_triggers(
    tmp_path: Path,
) -> None:
    """With no ``category_of`` every trigger is the neutral ▾ (no badge color)."""
    root = _make_output_root(tmp_path)

    component, _order = build_grid(
        df=root.master_df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
    )

    buttons = _collect_buttons(component)
    triggers = [
        btn
        for btn in buttons
        if isinstance(btn.id, dict) and btn.id.get("type") == "colony-radial-trigger"
    ]
    assert triggers
    for btn in triggers:
        assert "radial-badge--neutral" in (btn.className or "")  # type: ignore[union-attr]
        assert not btn.style.get("backgroundColor")  # type: ignore[union-attr]


def test_build_grid_custom_category_badge_marks_is_custom(tmp_path: Path) -> None:
    """A custom (non-core) category token renders the ``radial-badge--custom`` modifier."""
    root = _make_output_root(tmp_path)
    category_of = {("img-001", 1): "halo"}  # not an ErrorCategory token

    component, _order = build_grid(
        df=root.master_df,
        x_axis_col="Grid_ColNum",
        y_axis_col="Grid_RowNum",
        max_size=64,
        removed_keys=set(),
        selected_keys=set(),
        output_root=root,
        category_of=category_of,
    )

    buttons = _collect_buttons(component)
    labeled_trigger = next(
        (
            btn
            for btn in buttons
            if isinstance(btn.id, dict)
            and btn.id.get("type") == "colony-radial-trigger"
            and btn.id.get("image_file") == "img-001"
            and btn.id.get("label") == 1
        ),
        None,
    )
    assert labeled_trigger is not None
    assert "radial-badge--custom" in (labeled_trigger.className or "")  # type: ignore[union-attr]
