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

import numpy as np
import polars as pl
import pytest
from dash import html
from dash.development.base_component import Component
from PIL import Image as PILImage

from phenotypic.gui._config import (
    TILE_DIM_MAX,
    TILE_DIM_MIN,
    step_dim_alpha,
)
from phenotypic.gui._shared.tiles import (
    _dim_outside_bbox,
    build_tile_cell,
    build_tile_grid,
    crop_hdf_rgb,
    crop_overlay,
    expand_range,
    is_safe_path_component,
    register_crop_route,
)
from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot

from tests._output_layout import write_master


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
# crop_hdf_rgb — full-res HDF-layer cropper (Batch B1)
# ---------------------------------------------------------------------------


def test_crop_hdf_rgb_returns_full_res_png(tmp_path: Path) -> None:
    """``crop_hdf_rgb`` slices the raw ``/layers/rgb`` HDF dataset at full res."""
    from phenotypic import Image

    # Build a tiny image with a distinctive RGB layer and save to HDF.
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[10:30, 10:30] = (255, 0, 0)
    img = Image(arr=rgb)
    h5 = tmp_path / "img001.h5"
    img.save2hdf5(str(h5))

    out = crop_hdf_rgb(
        h5,
        "rgb",
        center_rr=20,
        center_cc=20,
        size=16,
        mtime_ns=h5.stat().st_mtime_ns,
    )
    crop = PILImage.open(io.BytesIO(out)).convert("RGB")
    assert crop.size == (16, 16)
    # Centre pixel falls inside the red square.
    assert crop.getpixel((8, 8)) == (255, 0, 0)


def test_crop_hdf_rgb_matches_crop_overlay_geometry(tmp_path: Path) -> None:
    """The HDF cropper and the overlay cropper share byte-identical geometry.

    Same pixel source (a solid RGB plane stored once as a PNG, once as an HDF
    ``/layers/rgb`` dataset) must yield the same edge-padded crop, proving the
    shared ``_crop_pil_source`` body is reused unchanged.
    """
    from phenotypic import Image

    rgb = np.full((100, 100, 3), (200, 120, 40), dtype=np.uint8)

    png = tmp_path / "src.png"
    PILImage.fromarray(rgb, mode="RGB").save(png, format="PNG")
    h5 = tmp_path / "src.h5"
    Image(arr=rgb).save2hdf5(str(h5))

    from_overlay = crop_overlay(png, center_rr=5, center_cc=5, size=24)
    from_hdf = crop_hdf_rgb(
        h5, "rgb", center_rr=5, center_cc=5, size=24, mtime_ns=h5.stat().st_mtime_ns
    )
    assert _decode_rgb(from_hdf).tolist() == _decode_rgb(from_overlay).tolist()


# ---------------------------------------------------------------------------
# crop_colony — per-image source dispatcher (Batch B1, Task 7)
# ---------------------------------------------------------------------------


def test_crop_colony_prefers_hdf_falls_back_to_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatch: HDF when present, overlay otherwise, ``None`` when neither."""
    from phenotypic.gui._shared import tiles

    # Sentinel bytes mark which source the dispatcher chose.
    monkeypatch.setattr(tiles, "crop_hdf_rgb", lambda *a, **k: b"H")
    monkeypatch.setattr(tiles, "crop_overlay", lambda *a, **k: b"O")

    class FakeRoot:
        def __init__(self, hdf: bool, overlay_ok: bool) -> None:
            self._hdf = hdf
            self._overlay_ok = overlay_ok

        def hdf_path(self, ds: str, stem: str) -> Path | None:
            return tmp_path / "x.h5" if self._hdf else None

        def has_overlay(self, ds: str, stem: str) -> bool:
            return self._overlay_ok

        def overlay_path(self, ds: str, stem: str) -> Path:
            return tmp_path / "x.png"

    # HDF present -> HDF path (os.stat needs the file to exist).
    (tmp_path / "x.h5").write_bytes(b"")
    assert tiles.crop_colony(FakeRoot(True, True), "p", "s", "rgb", 1, 1, 8) == b"H"
    # No HDF but overlay -> overlay path.
    assert tiles.crop_colony(FakeRoot(False, True), "p", "s", "rgb", 1, 1, 8) == b"O"
    # Neither -> None (caller serves 404).
    assert tiles.crop_colony(FakeRoot(False, False), "p", "s", "rgb", 1, 1, 8) is None


# ---------------------------------------------------------------------------
# crop_overlay — tile-spotlight dim pass (Phase 1)
# ---------------------------------------------------------------------------


def _decode_rgb(png_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes to an ``(H, W, 3)`` uint8 RGB array."""
    return np.asarray(PILImage.open(io.BytesIO(png_bytes)).convert("RGB"))


def test_crop_overlay_dim_alpha_zero_is_byte_identical(tmp_path: Path) -> None:
    """``dim_alpha=0.0`` returns bytes identical to the no-kwargs output."""
    src = tmp_path / "src.png"
    _write_solid_png(src, 100, 100, (200, 120, 40))

    baseline = crop_overlay(src, center_rr=50, center_cc=50, size=24)
    dimmed_off = crop_overlay(
        src,
        center_rr=50,
        center_cc=50,
        size=24,
        dim_alpha=0.0,
        bbox=(45.0, 55.0, 45.0, 55.0),
    )
    assert dimmed_off == baseline


def test_crop_overlay_bbox_none_is_byte_identical(tmp_path: Path) -> None:
    """``bbox=None`` (even with a positive alpha) returns the baseline bytes."""
    src = tmp_path / "src.png"
    _write_solid_png(src, 100, 100, (200, 120, 40))

    baseline = crop_overlay(src, center_rr=50, center_cc=50, size=24)
    dimmed_none = crop_overlay(
        src,
        center_rr=50,
        center_cc=50,
        size=24,
        dim_alpha=0.6,
        bbox=None,
    )
    assert dimmed_none == baseline


def test_crop_overlay_dims_outside_bbox_keeps_inside(tmp_path: Path) -> None:
    """Inside-bbox pixels stay 255; outside pixels darken to round(255*(1-a))."""
    src = tmp_path / "src.png"
    _write_solid_png(src, 100, 100, (255, 255, 255))

    # alpha=0.6 → 255*(1-0.6) = 102.0 exactly, so round() and the uint8
    # truncating cast agree (no half-integer boundary ambiguity).
    alpha = 0.6
    # Centroid (50, 50), size 20 → canvas origin at (40, 40) image px.
    # Bbox spans rows/cols [45, 55] → canvas px [5, 15).
    png_bytes = crop_overlay(
        src,
        center_rr=50,
        center_cc=50,
        size=20,
        dim_alpha=alpha,
        bbox=(45.0, 55.0, 45.0, 55.0),
    )
    arr = _decode_rgb(png_bytes)
    assert arr.shape == (20, 20, 3)

    expected_dim = round(255 * (1.0 - alpha))  # 102

    # An interior pixel of the keep-rect stays fully lit.
    assert tuple(arr[10, 10]) == (255, 255, 255)
    # A corner pixel (well outside the bbox) is dimmed toward black.
    assert tuple(arr[0, 0]) == (expected_dim, expected_dim, expected_dim)
    assert tuple(arr[19, 19]) == (expected_dim, expected_dim, expected_dim)


def test_crop_overlay_dim_edge_clamped_origin_frames_bbox(tmp_path: Path) -> None:
    """Near a border (negative unclamped origin) the keep-rect still frames the bbox."""
    src = tmp_path / "src.png"
    _write_solid_png(src, 100, 100, (255, 255, 255))

    # alpha=0.6 → 255*0.4 = 102.0 exactly (no half-integer boundary).
    alpha = 0.6
    # Centroid near the top-left corner: center (5, 5), size 20 →
    # unclamped origin at (-5, -5). Source pastes at (5, 5) on the canvas;
    # rows/cols [0, 5) of the canvas are black padding (off-image).
    # Bbox spans image rows/cols [2, 8] → canvas px [7, 13) after the
    # origin shift (2 - (-5) = 7, 8 - (-5) = 13). No origin drift: the
    # keep-rect must track the bbox even though the origin is negative.
    png_bytes = crop_overlay(
        src,
        center_rr=5,
        center_cc=5,
        size=20,
        dim_alpha=alpha,
        bbox=(2.0, 8.0, 2.0, 8.0),
    )
    arr = _decode_rgb(png_bytes)
    expected_dim = round(255 * (1.0 - alpha))  # 102

    # Inside the keep-rect AND inside the pasted (on-image) source region
    # → fully lit white.
    assert tuple(arr[10, 10]) == (255, 255, 255)
    # Outside the keep-rect but still on-image source (e.g. canvas row 16,
    # which is image row 11) → dimmed white.
    assert tuple(arr[16, 16]) == (expected_dim, expected_dim, expected_dim)
    # Off-image padding outside the keep-rect (canvas (0, 0) is image
    # (-5, -5)) is black already; dimming a black pixel keeps it black.
    assert tuple(arr[0, 0]) == (0, 0, 0)


# ---------------------------------------------------------------------------
# _dim_outside_bbox (helper)
# ---------------------------------------------------------------------------


def test_dim_outside_bbox_alpha_zero_returns_canvas_unchanged() -> None:
    """``alpha <= 0`` short-circuits and returns the canvas as given."""
    canvas = np.full((8, 8, 3), 200, dtype=np.uint8)
    out = _dim_outside_bbox(canvas, (2, 2, 6, 6), alpha=0.0)
    assert np.array_equal(out, canvas)


def test_dim_outside_bbox_blends_complement_toward_black() -> None:
    """Pixels outside the keep-rect blend toward black; inside stay put."""
    canvas = np.full((8, 8, 3), 100, dtype=np.uint8)
    # keep = (top, left, bottom, right)
    out = _dim_outside_bbox(canvas, (2, 2, 6, 6), alpha=0.5)
    # Inside the keep-rect: untouched.
    assert tuple(out[3, 3]) == (100, 100, 100)
    # Outside: 100 * (1 - 0.5) = 50.
    assert tuple(out[0, 0]) == (50, 50, 50)


def test_dim_outside_bbox_empty_keep_dims_everything() -> None:
    """A degenerate keep-rect (zero area) dims the whole canvas."""
    canvas = np.full((4, 4, 3), 80, dtype=np.uint8)
    # bottom == top → no kept region.
    out = _dim_outside_bbox(canvas, (2, 2, 2, 2), alpha=0.5)
    assert np.all(out == 40)


# ---------------------------------------------------------------------------
# step_dim_alpha (pure stepper arithmetic)
# ---------------------------------------------------------------------------


def test_step_dim_alpha_increments_and_decrements() -> None:
    """A single ``+``/``−`` click moves by exactly TILE_DIM_STEP."""
    assert step_dim_alpha(0.60, +1) == 0.65
    assert step_dim_alpha(0.60, -1) == 0.55


def test_step_dim_alpha_clamps_at_bounds() -> None:
    """Stepping past either bound saturates at TILE_DIM_MIN / TILE_DIM_MAX."""
    assert step_dim_alpha(TILE_DIM_MIN, -1) == TILE_DIM_MIN
    assert step_dim_alpha(TILE_DIM_MAX, +1) == TILE_DIM_MAX
    # Already at the floor: another ``−`` keeps it at the floor.
    assert step_dim_alpha(0.0, -1) == 0.0
    # Already at the ceiling: another ``+`` keeps it at the ceiling.
    assert step_dim_alpha(0.9, +1) == 0.9


def test_step_dim_alpha_is_two_decimal_float_safe() -> None:
    """Repeated stepping never accumulates binary-float drift past 2 dp."""
    value = 0.0
    for _ in range(6):
        value = step_dim_alpha(value, +1)
    # 0.0 → 0.05 → 0.10 → 0.15 → 0.20 → 0.25 → 0.30, exactly (no 0.30000004).
    assert value == 0.30
    # Every reachable value rounds clean to two decimals.
    probe = 0.0
    seen = []
    for _ in range(20):
        probe = step_dim_alpha(probe, +1)
        seen.append(probe)
    for v in seen:
        assert round(v, 2) == v


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
    write_master(tmp_path, master)
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlay_dir = tmp_path / "deliverables" / "overlays" / "d1"
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
    # Use a test-only segment; ``create_app`` already mounts ``/crops``
    # (colony view) and ``/qc-crops`` (QC review) at boot, so a fresh
    # segment exercises the factory without a blueprint-name collision.
    register_crop_route(app, output_root, "extra-crops")
    client = app.server.test_client()

    resp = client.get("/extra-crops/d1/img-1/7.png?size=24")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    img = PILImage.open(io.BytesIO(resp.data))
    assert img.size == (24, 24)


def test_register_crop_route_validates_size_and_label(
    output_root: OutputRoot,
) -> None:
    """Missing size → 400; unknown label → 404."""
    app = create_app(output_root)
    register_crop_route(app, output_root, "extra-crops")
    client = app.server.test_client()

    assert client.get("/extra-crops/d1/img-1/7.png").status_code == 400
    assert client.get("/extra-crops/d1/img-1/99.png?size=24").status_code == 404


def test_register_crop_route_distinct_segments_coexist(
    output_root: OutputRoot,
) -> None:
    """Multiple crop segments register without a blueprint-name collision."""
    app = create_app(output_root)
    # ``create_app`` already mounts ``/crops`` (colony) and ``/qc-crops``
    # (QC review); mount a third segment and confirm all three answer.
    register_crop_route(app, output_root, "extra-crops")
    client = app.server.test_client()

    assert client.get("/crops/d1/img-1/7.png?size=24").status_code == 200
    assert client.get("/qc-crops/d1/img-1/7.png?size=24").status_code == 200
    assert client.get("/extra-crops/d1/img-1/7.png?size=24").status_code == 200


# ---------------------------------------------------------------------------
# register_crop_route — ?dim spotlight wiring (Phase 2)
# ---------------------------------------------------------------------------


def test_register_crop_route_dim_param_serves_png(output_root: OutputRoot) -> None:
    """A valid ``?dim=`` returns a 200 PNG (the master has the bbox columns)."""
    app = create_app(output_root)
    register_crop_route(app, output_root, "extra-crops")
    client = app.server.test_client()

    resp = client.get("/extra-crops/d1/img-1/7.png?size=24&dim=0.5")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"


def test_register_crop_route_dim_omitted_defaults_undimmed(
    output_root: OutputRoot,
) -> None:
    """Omitting ``?dim`` defaults to 0.0 — same bytes as an explicit dim=0."""
    app = create_app(output_root)
    register_crop_route(app, output_root, "extra-crops")
    client = app.server.test_client()

    no_dim = client.get("/extra-crops/d1/img-1/7.png?size=24")
    zero_dim = client.get("/extra-crops/d1/img-1/7.png?size=24&dim=0")
    assert no_dim.status_code == 200
    assert zero_dim.status_code == 200
    assert no_dim.data == zero_dim.data


def test_register_crop_route_dim_out_of_range_is_clamped(
    output_root: OutputRoot,
) -> None:
    """An over-range ``?dim`` clamps to TILE_DIM_MAX (200, not 400)."""
    app = create_app(output_root)
    register_crop_route(app, output_root, "extra-crops")
    client = app.server.test_client()

    over = client.get("/extra-crops/d1/img-1/7.png?size=24&dim=5")
    at_max = client.get(
        f"/extra-crops/d1/img-1/7.png?size=24&dim={TILE_DIM_MAX}"
    )
    assert over.status_code == 200
    assert at_max.status_code == 200
    # Clamped to the ceiling → identical raster to an explicit max request.
    assert over.data == at_max.data


def test_register_crop_route_dim_negative_is_clamped_to_zero(
    output_root: OutputRoot,
) -> None:
    """A negative ``?dim`` clamps to 0.0 (undimmed), never a 400."""
    app = create_app(output_root)
    register_crop_route(app, output_root, "extra-crops")
    client = app.server.test_client()

    neg = client.get("/extra-crops/d1/img-1/7.png?size=24&dim=-1")
    undimmed = client.get("/extra-crops/d1/img-1/7.png?size=24&dim=0")
    assert neg.status_code == 200
    assert neg.data == undimmed.data


@pytest.fixture()
def output_root_no_bbox(tmp_path: Path) -> OutputRoot:
    """An output dir whose master lacks the Bbox_Min/Max columns (older run)."""
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"],
            "Metadata_ImageFile": ["img-1"],
            "Object_Label": [7],
            "Bbox_CenterRR": [50],
            "Bbox_CenterCC": [50],
            # No Bbox_MinRR/MaxRR/MinCC/MaxCC — graceful-degrade path.
        }
    )
    write_master(tmp_path, master)
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlay_dir = tmp_path / "deliverables" / "overlays" / "d1"
    overlay_dir.mkdir(parents=True)
    PILImage.new("RGB", (100, 100), (255, 0, 0)).save(
        overlay_dir / "img-1.png", format="PNG"
    )
    return OutputRoot.discover(tmp_path)


def test_register_crop_route_degrades_when_bbox_columns_absent(
    output_root_no_bbox: OutputRoot,
) -> None:
    """Missing bbox columns → undimmed 200 (no 500), even with ``?dim`` set."""
    app = create_app(output_root_no_bbox)
    register_crop_route(app, output_root_no_bbox, "extra-crops")
    client = app.server.test_client()

    resp = client.get("/extra-crops/d1/img-1/7.png?size=24&dim=0.6")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"


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
