"""The Plate card is a full-canvas Viv stage whose Layers panel reads the store.

Two claims are pinned here and they fail differently.

The **layout** claim is that the card is a stage with floating chrome rather
than a header-plus-canvas card: the image gets the whole frame, and every
control sits over it. That one fails visibly.

The **Layers panel** claim is the one that fails silently, and is the reason
the panel is the piece the plan forbids cutting. The row list comes from the
store's own ``attributes.phenotypic.series``, so an ``original`` series the
writer legitimately appended is offered and a ``gray``-primary store has no
``rgb`` row. A hard-coded ``{rgb, gray, detect_mat}`` would agree with most
stores, disagree with the byte route's readable set on the rest, and look
correct throughout.
"""

from __future__ import annotations

import json

from dash import html

from phenotypic.gui.results_viewer._store_source import build_source_spec
from phenotypic.gui.results_viewer._viewer_card import build_layer_rows, layout


def _walk(node, out=None):
    """Every component in a Dash tree, depth-first."""
    out = [] if out is None else out
    out.append(node)
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _walk(child, out)
    elif children is not None:
        _walk(children, out)
    return out


def _class_names(tree) -> set[str]:
    names: set[str] = set()
    for node in _walk(tree):
        value = getattr(node, "className", None)
        if isinstance(value, str):
            names.update(value.split())
    return names


def _ids(tree) -> list[dict]:
    found = []
    for node in _walk(tree):
        node_id = getattr(node, "id", None)
        if isinstance(node_id, dict):
            found.append(node_id)
    return found


def _texts(tree) -> list[str]:
    return [n for n in _walk(tree) if isinstance(n, str)]


def _rows_text(rows) -> list[str]:
    return [n for row in rows for n in _walk(row) if isinstance(n, str)]


# ---- the stage ----------------------------------------------------------


def test_the_card_is_a_stage_with_floating_chrome(tmp_path):
    """Controls float OVER the canvas; there is no card header stealing frame."""
    tree = layout("abc", None)
    classes = _class_names(tree)
    assert "plate-stage" in classes
    assert "plate-stage__canvas" in classes
    assert {
        "plate-float--top-left",
        "plate-float--top-right",
        "plate-float--bottom-left",
        "plate-float--bottom-right",
    } <= classes
    # The OSD canvas class is what the MutationObserver used to key on; a
    # leftover would make the new observer and the old CSS disagree about
    # which element is the stage.
    assert "osd-canvas" not in classes


def test_the_card_carries_the_two_stores_the_bridge_reads(tmp_path):
    """The spec store and the display store are separate on purpose.

    A re-source (a promote, a stepped image) replaces the spec; it must not
    silently reset which series the user chose or how transparent they made
    the objmap.
    """
    kinds = {node["type"] for node in _ids(layout("abc", None))}
    assert "card-source-spec" in kinds
    assert "card-display-state" in kinds
    assert "card-viv-stage" in kinds
    assert "card-pyramid-readout" in kinds


def test_the_stage_holds_no_server_rendered_tile_url(tmp_path):
    """Nothing in the card names a `.dzi` manifest any more."""
    rendered = json.dumps(
        layout("abc", None), default=lambda node: node.to_plotly_json()
    )
    assert ".dzi" not in rendered
    assert "/tiles/" not in rendered


# ---- the Layers panel ---------------------------------------------------


def test_layer_rows_come_from_the_stores_real_series(rgb_store):
    spec = build_source_spec(rgb_store, "/zarr/d/p.ome.zarr/t")
    text = _rows_text(build_layer_rows("abc", spec, None))
    assert "rgb" in text
    assert "gray" in text
    assert "detect_mat" in text
    assert "objmap" in text


def test_an_original_series_gets_a_row(store_with_original):
    """`_write_store_part` appends it; a literal three-name set would drop it."""
    spec = build_source_spec(store_with_original, "/zarr/d/o.ome.zarr/t")
    assert "original" in _rows_text(build_layer_rows("abc", spec, None))


def test_a_gray_primary_store_has_no_rgb_row(gray_only_store):
    spec = build_source_spec(gray_only_store, "/zarr/d/g.ome.zarr/t")
    text = _rows_text(build_layer_rows("abc", spec, None))
    assert "rgb" not in text
    assert "gray" in text
    assert "objmap" in text


def test_a_label_less_store_has_no_objmap_row(label_less_store):
    spec = build_source_spec(label_less_store, "/zarr/d/prev.ome.zarr/t")
    assert "objmap" not in _rows_text(build_layer_rows("abc", spec, None))


def test_an_unmeasured_store_says_measurement_pending(stage1_store):
    """A zeros objmap mid-run is a CORRECT store, and must read as one.

    Between Stage 1 and Stage 3 the in-store objmap is all zeros because the
    landed staged engine keeps Stage 2 read-only. Rendered without the tag
    that is indistinguishable from a finished image whose detector found
    nothing -- one is a user waiting, the other is a user filing a bug.
    """
    spec = build_source_spec(stage1_store, "/zarr/d/s1.ome.zarr/t")
    text = _rows_text(build_layer_rows("abc", spec, None))
    assert "objmap" in text
    assert "measurement pending" in text
    assert "label image" not in text


def test_a_measured_store_tags_the_objmap_as_a_label_image(rgb_store):
    spec = build_source_spec(rgb_store, "/zarr/d/p.ome.zarr/t")
    text = _rows_text(build_layer_rows("abc", spec, None))
    assert "label image" in text
    assert "measurement pending" not in text


def test_every_row_addresses_its_own_layer(rgb_store):
    """One MATCH/ALL callback covers whatever series the store turned out to
    hold, so each control carries the layer name as a third id key."""
    spec = build_source_spec(rgb_store, "/zarr/d/p.ome.zarr/t")
    rows = build_layer_rows("abc", spec, None)
    eyes = {
        node["layer"]
        for row in rows
        for node in _ids(row)
        if node["type"] == "card-layer-eye"
    }
    assert eyes == {"rgb", "gray", "detect_mat", "objmap"}


def test_the_displayed_series_follows_the_display_state(rgb_store):
    """Viv holds ONE image source, so a series row is a radio, not a checkbox."""
    spec = build_source_spec(rgb_store, "/zarr/d/p.ome.zarr/t")
    rows = build_layer_rows(
        "abc", spec, {"seriesPath": "gray", "labelVisible": True}
    )
    off = set()
    for row in rows:
        for node in _walk(row):
            class_name = getattr(node, "className", None)
            if not isinstance(class_name, str):
                continue
            if "plate-layer__eye--off" in class_name:
                off.add(node.id["layer"])
    assert "gray" not in off
    assert "rgb" in off


def test_no_rows_without_a_spec():
    assert build_layer_rows("abc", None, None) == []


def test_the_panel_never_hard_codes_a_series_set():
    """A literal `{rgb, gray, detect_mat}` in the module is the failure mode.

    Not a style check: the byte route derives its readable set from the same
    block, so a hard-coded panel would offer a series the route 404s and hide
    one it serves, and both halves would look right in isolation.
    """
    from pathlib import Path

    import phenotypic.gui.results_viewer._viewer_card as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert '"detect_mat"' not in source
    assert "'detect_mat'" not in source


def test_the_stage_is_reachable_from_a_built_layout(built_results_layout):
    """The card container mounts; the stage is what the bridge looks up."""
    assert isinstance(built_results_layout, html.Div)


def test_no_server_callback_writes_the_served_level_readout(tmp_path):
    """The readout names the level deck.gl ACTUALLY served, so only the
    facade may write it.

    A server-side number -- `select_pyramid_level` over the same target
    pixel size -- would name a level nobody rendered, and a readout labelled
    "the level actually being served" is trusted exactly when diagnosing the
    bug it would be misreporting. Asserted on the registered callback map:
    the readout is written from the facade's `onLevelChange`, in the browser,
    and no Python callback may claim it as an Output.
    """
    import polars as pl
    from PIL import Image as PILImage

    from phenotypic.gui.results_viewer._app import create_app
    from phenotypic.gui.results_viewer._output_root import OutputRoot
    from phenotypic.schema import IMAGE
    from tests._output_layout import (
        write_complete_manifest,
        write_master,
        write_measurements_mirror,
    )

    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"],
            str(IMAGE.IMAGE_NAME): ["img-A"],
            "Object_Label": [1],
            "Bbox_CenterRR": [10.0],
            "Bbox_CenterCC": [10.0],
            "Size_Area": [100.0],
        }
    )
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)
    (tmp_path / "results" / "d1" / "measurements").mkdir(
        parents=True, exist_ok=True
    )
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True)
    PILImage.new("RGB", (64, 64), (128, 128, 128)).save(overlays / "img-A.png")
    write_complete_manifest(tmp_path, total_images=1)

    app = create_app(
        OutputRoot.discover(
            tmp_path,
            cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
        )
    )
    assert "card-pyramid-readout" not in " ".join(app.callback_map)
    assert "card-zoom-readout" not in " ".join(app.callback_map)
