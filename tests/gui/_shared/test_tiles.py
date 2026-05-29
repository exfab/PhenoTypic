"""Unit tests for :mod:`phenotypic.gui._shared.tiles`.

The shared tile primitives back both the colony-view grid and the QC
review gallery, so these tests pin the contract the colony view relied
on before the extraction (verified separately by
``tests/gui/results_viewer/colony_view``) and the flat-gallery + key-order
behaviour the QC tab will consume:

- :func:`crop_overlay` — fixed-size, edge-padded crops.
- :func:`is_safe_path_component` — the path-traversal guard.
- :func:`register_crop_route` — the crop-route factory mounts under an
  arbitrary segment and serves a centered PNG.
- :func:`build_tile_cell` — per-tile chrome (img/placeholder, checkbox
  ``data-key``, caller-supplied remove button, optional siblings).
- :func:`build_tile_grid` — flat gallery + row-major key list.
- :func:`expand_range` — direction-agnostic inclusive slice.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from dash import html
from dash.development.base_component import Component
from PIL import Image as PILImage

from phenotypic.gui._shared.tiles import (
    build_tile_cell,
    build_tile_grid,
    crop_overlay,
    expand_range,
    is_safe_path_component,
    register_crop_route,
)
from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot


# ---------------------------------------------------------------------------
# Component-tree helpers
# ---------------------------------------------------------------------------


def _walk(component: object) -> Iterator[object]:
    """Yield ``component`` and every descendant component, depth-first."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _find_by_class(root: object, class_substr: str) -> list[object]:
    """Return every descendant whose ``className`` contains ``class_substr``."""
    found = []
    for node in _walk(root):
        cls = getattr(node, "className", None)
        if isinstance(cls, str) and class_substr in cls:
            found.append(node)
    return found


# ---------------------------------------------------------------------------
# crop_overlay
# ---------------------------------------------------------------------------


def _write_solid_png(path: Path, w: int, h: int, color: tuple[int, int, int]) -> None:
    PILImage.new("RGB", (w, h), color).save(path, format="PNG")


def test_crop_overlay_centered_returns_requested_size(tmp_path: Path) -> None:
    """A crop in the interior of a solid image is solid and exactly size×size."""
    src = tmp_path / "src.png"
    _write_solid_png(src, 100, 100, (255, 255, 255))

    png_bytes = crop_overlay(src, center_rr=50, center_cc=50, size=20)
    img = PILImage.open(io.BytesIO(png_bytes))

    assert img.size == (20, 20)
    assert img.mode == "RGB"
    assert img.getextrema() == ((255, 255), (255, 255), (255, 255))


def test_crop_overlay_corner_is_padded(tmp_path: Path) -> None:
    """A crop near the (0, 0) corner is padded with ``pad_value``."""
    src = tmp_path / "src.png"
    _write_solid_png(src, 100, 100, (255, 255, 255))

    png_bytes = crop_overlay(
        src, center_rr=5, center_cc=5, size=20, pad_value=(0, 0, 0)
    )
    img = PILImage.open(io.BytesIO(png_bytes))

    assert img.size == (20, 20)
    assert img.getpixel((0, 0)) == (0, 0, 0)  # padding
    assert img.getpixel((19, 19)) == (255, 255, 255)  # source


# ---------------------------------------------------------------------------
# is_safe_path_component
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["plate1", "img-001", "Run_12-01_29_26", "a.b_c-1"],
)
def test_is_safe_path_component_accepts_clean_identifiers(name: str) -> None:
    """Filesystem-safe identifiers are accepted."""
    assert is_safe_path_component(name) is True


@pytest.mark.parametrize(
    "name",
    ["", ".hidden", "..", "a/b", "a\\b", "../etc", "with space", "weird*char"],
)
def test_is_safe_path_component_rejects_traversal_and_bad_charset(name: str) -> None:
    """Empty, dot-leading, separator-bearing, or odd-charset names are rejected."""
    assert is_safe_path_component(name) is False


# ---------------------------------------------------------------------------
# register_crop_route (factory, arbitrary segment)
# ---------------------------------------------------------------------------


@pytest.fixture()
def output_root(tmp_path: Path) -> OutputRoot:
    """Minimal output dir: one colony in dataset 'd1' + an overlay PNG."""
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"],
            "Metadata_ImageFile": ["img-1"],
            "Object_Label": [7],
            "Bbox_CenterRR": [50],
            "Bbox_CenterCC": [50],
            "Bbox_MinRR": [40],
            "Bbox_MaxRR": [60],
            "Bbox_MinCC": [40],
            "Bbox_MaxCC": [60],
        }
    )
    master.write_parquet(tmp_path / "master_measurements.parquet")
    overlay_dir = tmp_path / "results" / "d1" / "overlays"
    overlay_dir.mkdir(parents=True)
    PILImage.new("RGB", (100, 100), (255, 0, 0)).save(
        overlay_dir / "img-1.png", format="PNG"
    )
    return OutputRoot.discover(tmp_path)


def test_register_crop_route_serves_centered_png_under_custom_segment(
    output_root: OutputRoot,
) -> None:
    """The factory mounts under an arbitrary segment and serves a sized PNG."""
    app = create_app(output_root)
    register_crop_route(app, output_root, "qc-crops")
    client = app.server.test_client()

    resp = client.get("/qc-crops/d1/img-1/7.png?size=24")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    img = PILImage.open(io.BytesIO(resp.data))
    assert img.size == (24, 24)


def test_register_crop_route_validates_size_and_label(
    output_root: OutputRoot,
) -> None:
    """Missing size → 400; unknown label → 404."""
    app = create_app(output_root)
    register_crop_route(app, output_root, "qc-crops")
    client = app.server.test_client()

    assert client.get("/qc-crops/d1/img-1/7.png").status_code == 400
    assert client.get("/qc-crops/d1/img-1/99.png?size=24").status_code == 404


def test_register_crop_route_distinct_segments_coexist(
    output_root: OutputRoot,
) -> None:
    """Two segments register without a blueprint-name collision."""
    app = create_app(output_root)
    # The colony "/crops" route is already mounted by create_app; mount a
    # second segment and confirm both answer.
    register_crop_route(app, output_root, "qc-crops")
    client = app.server.test_client()

    assert client.get("/crops/d1/img-1/7.png?size=24").status_code == 200
    assert client.get("/qc-crops/d1/img-1/7.png?size=24").status_code == 200


# ---------------------------------------------------------------------------
# build_tile_cell
# ---------------------------------------------------------------------------


def _url_builder(dataset: str, image_file: str, label: int, crop_size: int) -> str:
    return f"/seg/{dataset}/{image_file}/{label}.png?size={crop_size}"


def _remove_button(image_file: str, label: int, is_removed: bool) -> Component:
    return html.Button(
        "↺" if is_removed else "✕",
        id=f"rm-{image_file}-{label}",
        className="colony-cell-remove-btn",
    )


def test_build_tile_cell_renders_img_with_url_builder_src() -> None:
    """When the overlay exists, the tile carries an <img> from the url_builder."""
    cell = build_tile_cell(
        image_file="img-1",
        label=7,
        dataset="d1",
        crop_size=128,
        display_size=96,
        has_overlay=True,
        is_removed=False,
        is_selected=False,
        url_builder=_url_builder,
        remove_button=_remove_button("img-1", 7, False),
    )
    imgs = [n for n in _walk(cell) if isinstance(n, html.Img)]
    assert len(imgs) == 1
    assert imgs[0].src == "/seg/d1/img-1/7.png?size=128"


def test_build_tile_cell_placeholder_when_no_overlay() -> None:
    """A missing overlay renders the striped placeholder and no <img>."""
    cell = build_tile_cell(
        image_file="img-1",
        label=7,
        dataset="d1",
        crop_size=128,
        display_size=96,
        has_overlay=False,
        is_removed=False,
        is_selected=False,
        url_builder=_url_builder,
        remove_button=_remove_button("img-1", 7, False),
    )
    assert not [n for n in _walk(cell) if isinstance(n, html.Img)]
    assert _find_by_class(cell, "colony-cell-placeholder")


def test_build_tile_cell_checkbox_carries_data_key() -> None:
    """The checkbox span carries ``data-key='<image_file>::<label>'``."""
    cell = build_tile_cell(
        image_file="img-1",
        label=7,
        dataset="d1",
        crop_size=128,
        display_size=96,
        has_overlay=True,
        is_removed=False,
        is_selected=False,
        url_builder=_url_builder,
        remove_button=_remove_button("img-1", 7, False),
    )
    checkboxes = _find_by_class(cell, "colony-cell-checkbox")
    # The inner span (not the wrap) carries the data-key attribute.
    keyed = [
        n
        for n in checkboxes
        if getattr(n, "data-key", None) == "img-1::7"
    ]
    assert keyed, "checkbox inner span missing data-key='img-1::7'"


def test_build_tile_cell_selected_and_removed_modifiers() -> None:
    """Selected/removed states toggle the outer modifiers + checkbox state."""
    cell = build_tile_cell(
        image_file="img-1",
        label=7,
        dataset="d1",
        crop_size=128,
        display_size=96,
        has_overlay=True,
        is_removed=True,
        is_selected=True,
        url_builder=_url_builder,
        remove_button=_remove_button("img-1", 7, True),
    )
    outer_cls = cell.className
    assert "is-selected" in outer_cls
    assert "is-removed" in outer_cls
    # Checkbox inner gains is-checked when selected.
    assert _find_by_class(cell, "is-checked")
    # Dimmed crop.
    imgs = [n for n in _walk(cell) if isinstance(n, html.Img)]
    assert imgs[0].style["opacity"] == "0.3"


def test_build_tile_cell_appends_extra_children_and_outer_height() -> None:
    """``extra_children`` are appended after the frame; ``outer_height`` is honoured."""
    badge = html.Div("N=3", className="colony-cell-stack-tab")
    cell = build_tile_cell(
        image_file="img-1",
        label=7,
        dataset="d1",
        crop_size=128,
        display_size=96,
        has_overlay=True,
        is_removed=False,
        is_selected=False,
        url_builder=_url_builder,
        remove_button=_remove_button("img-1", 7, False),
        extra_children=[badge],
        outer_height=110,
    )
    assert _find_by_class(cell, "colony-cell-stack-tab")
    # frame is first child, badge appended after it.
    assert isinstance(cell.children, list) and len(cell.children) == 2
    assert cell.style["height"] == "110px"


# ---------------------------------------------------------------------------
# build_tile_grid
# ---------------------------------------------------------------------------


def test_build_tile_grid_row_major_order_and_tile_count() -> None:
    """grid_order mirrors the input key order; one tile per key."""
    keys = [
        ("d1", "img-1", 1),
        ("d1", "img-1", 2),
        ("d1", "img-2", 5),
    ]
    component, grid_order = build_tile_grid(
        keys,
        _url_builder,
        selected=set(),
        removed=set(),
        crop_size=128,
        display_size=80,
        has_overlay=lambda dataset, image_file: True,
        remove_button_builder=_remove_button,
    )
    assert grid_order == [("img-1", 1), ("img-1", 2), ("img-2", 5)]
    cells = _find_by_class(component, "colony-cell")
    # Each tile's outer div + (no extra children) — count the keyed checkboxes.
    keyed = [
        n
        for n in _walk(component)
        if getattr(n, "data-key", None) is not None
    ]
    assert {getattr(n, "data-key") for n in keyed} == {
        "img-1::1",
        "img-1::2",
        "img-2::5",
    }
    assert cells  # gallery rendered tile chrome


def test_build_tile_grid_marks_removed_tiles() -> None:
    """Keys in ``removed`` render with the removed modifier + dimmed crop."""
    keys = [("d1", "img-1", 1), ("d1", "img-1", 2)]
    component, _ = build_tile_grid(
        keys,
        _url_builder,
        selected=set(),
        removed={("img-1", 2)},
        crop_size=64,
        display_size=64,
        has_overlay=lambda dataset, image_file: True,
        remove_button_builder=_remove_button,
    )
    removed_cells = _find_by_class(component, "is-removed")
    assert len(removed_cells) == 1


def test_build_tile_grid_honours_missing_overlay_per_tile() -> None:
    """has_overlay=False tiles render a placeholder instead of an <img>."""
    keys = [("d1", "img-1", 1), ("d1", "img-2", 1)]
    component, _ = build_tile_grid(
        keys,
        _url_builder,
        selected=set(),
        removed=set(),
        crop_size=64,
        display_size=64,
        has_overlay=lambda dataset, image_file: image_file == "img-1",
        remove_button_builder=_remove_button,
    )
    imgs = [n for n in _walk(component) if isinstance(n, html.Img)]
    placeholders = _find_by_class(component, "colony-cell-placeholder")
    assert len(imgs) == 1
    assert len(placeholders) == 1


# ---------------------------------------------------------------------------
# expand_range
# ---------------------------------------------------------------------------


def test_expand_range_inclusive_and_direction_agnostic() -> None:
    """Returns the inclusive slice regardless of click order."""
    order = [("a", 1), ("a", 2), ("b", 1), ("b", 2), ("c", 1)]
    assert expand_range(order, ("a", 2), ("b", 2)) == [
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]
    assert expand_range(order, ("b", 2), ("a", 2)) == expand_range(
        order, ("a", 2), ("b", 2)
    )


def test_expand_range_raises_for_unknown_key() -> None:
    """A key outside grid_order raises ``ValueError``."""
    order = [("a", 1), ("a", 2)]
    with pytest.raises(ValueError):
        expand_range(order, ("a", 1), ("z", 99))
