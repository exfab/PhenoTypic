"""The Plate card resolves a real store into a facade source spec.

Unit tests pin ``build_source_spec`` and ``build_layer_rows`` in isolation.
This one drives the callback that joins them, through the real Dash update
route, against a real ``save2zarr`` store discovered by ``OutputRoot`` -- so
the URL the browser is handed is the URL the byte route serves, and a
mismatch between the two fails here rather than as an empty dark canvas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic import Image
from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import zarr_store_path

from tests._output_layout import (
    write_complete_manifest,
    write_master,
    write_measurements_mirror,
)

DATASET = "d1"
STEM = "img-A"
CARD = "abc"


@pytest.fixture()
def output_root(tmp_path: Path) -> OutputRoot:
    """A discoverable run whose one image has a real OME-Zarr store."""
    master = pl.DataFrame(
        {
            "Metadata_Dataset": [DATASET],
            str(IMAGE.IMAGE_NAME): [STEM],
            "Object_Label": [1],
            "Bbox_CenterRR": [16.0],
            "Bbox_CenterCC": [16.0],
            "Size_Area": [100.0],
        }
    )
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)
    (tmp_path / "results" / DATASET / "measurements").mkdir(
        parents=True, exist_ok=True
    )
    overlays = tmp_path / "deliverables" / "overlays" / DATASET
    overlays.mkdir(parents=True)
    PILImage.new("RGB", (64, 64), (128, 128, 128)).save(overlays / f"{STEM}.png")
    write_complete_manifest(tmp_path, total_images=1)

    store = zarr_store_path(tmp_path, DATASET, STEM)
    store.parent.mkdir(parents=True, exist_ok=True)
    Image(arr=np.zeros((64, 64, 3), dtype=np.uint8)).save2zarr(store)

    return OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
    )


def _output_key(app) -> str:
    for key, callback in app.callback_map.items():
        searchable = key + json.dumps(callback.get("inputs", []), default=str)
        if "card-source-spec" in key and "card-state" in searchable:
            return key
    raise KeyError("no card-source-spec callback registered")


def _resolve(app, dataset: str | None, stem: str | None):
    """Fire the source-resolution callback for one card and return its outputs."""
    key = _output_key(app)
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": [
                {"id": {"type": "card-source-spec", "index": CARD},
                 "property": "data"},
                {"id": {"type": "card-layers-panel", "index": CARD},
                 "property": "children"},
                {"id": {"type": "card-source-note", "index": CARD},
                 "property": "children"},
                {"id": {"type": "card-display-state", "index": CARD},
                 "property": "data"},
            ],
            "inputs": [
                {
                    "id": {"type": "card-state", "index": CARD},
                    "property": "data",
                    "value": {"dataset": dataset, "stem": stem},
                }
            ],
            "changedPropIds": [
                '{"index":"%s","type":"card-state"}.data' % CARD
            ],
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    # A pattern-matching multi-output response is keyed by the SERIALIZED
    # concrete id (keys sorted alphabetically), not by component type.
    return {
        json.loads(serialized)["type"]: props
        for serialized, props in response.get_json()["response"].items()
    }


def test_a_selected_image_resolves_to_a_byte_route_url(output_root):
    """The spec's `storeUrl` is the URL the byte route actually answers."""
    app = create_app(output_root)
    spec = _resolve(app, DATASET, STEM)["card-source-spec"]["data"]

    assert spec["seriesPath"] == "rgb"
    assert spec["labelPath"] == "rgb/labels/objmap"
    assert spec["storeUrl"].startswith(f"/zarr/{DATASET}/{STEM}.ome.zarr/")
    assert spec["storeUrl"].endswith(spec["token"])

    # The URL is not decoration: the route must serve the root the client
    # bootstraps from. A prefix or token mismatch fails here.
    client = app.server.test_client()
    assert client.get(f"{spec['storeUrl']}/zarr.json").status_code == 200
    assert client.get(f"{spec['storeUrl']}/{spec['seriesPath']}/0/zarr.json").status_code == 200


def test_the_layers_panel_is_rendered_from_the_resolved_spec(output_root):
    payload = _resolve(create_app(output_root), DATASET, STEM)
    rendered = json.dumps(payload["card-layers-panel"]["children"])
    for name in ("rgb", "gray", "detect_mat", "objmap"):
        assert name in rendered


def test_the_display_state_starts_on_the_primary_series(output_root):
    display = _resolve(create_app(output_root), DATASET, STEM)[
        "card-display-state"
    ]["data"]
    assert display["seriesPath"] == "rgb"
    assert display["labelVisible"] is True
    assert display["opacity"] == {"image": 1.0, "labels": 0.5}


def test_an_unselected_card_resolves_to_no_source(output_root):
    payload = _resolve(create_app(output_root), None, None)
    assert payload["card-source-spec"]["data"] is None
    assert payload["card-layers-panel"]["children"] == []


def _layer_controls_key(app) -> str:
    for key in app.callback_map:
        if "card-display-state" in key and "card-layers-panel" in key:
            if "card-source-note" in key:
                continue  # that is the source-resolution callback
            return key
    raise KeyError("no layer-controls callback registered")


def _click_eye(app, layer: str, display: dict, spec: dict):
    """Fire one Layers-panel visibility click and return the new outputs."""
    key = _layer_controls_key(app)
    eye_ids = [
        {"type": "card-layer-eye", "index": CARD, "layer": name}
        for name in ("rgb", "gray", "detect_mat", "objmap")
    ]
    slider_ids = [
        {"type": "card-layer-opacity", "index": CARD, "layer": name}
        for name in ("rgb", "gray", "detect_mat", "objmap")
    ]
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": [
                {"id": {"type": "card-display-state", "index": CARD},
                 "property": "data"},
                {"id": {"type": "card-layers-panel", "index": CARD},
                 "property": "children"},
            ],
            "inputs": [
                [
                    {"id": id_, "property": "n_clicks",
                     "value": 1 if id_["layer"] == layer else 0}
                    for id_ in eye_ids
                ],
                [
                    {"id": id_, "property": "value", "value": 1.0}
                    for id_ in slider_ids
                ],
            ],
            "state": [
                {"id": {"type": "card-display-state", "index": CARD},
                 "property": "data", "value": display},
                {"id": {"type": "card-source-spec", "index": CARD},
                 "property": "data", "value": spec},
            ],
            "changedPropIds": [
                '{"index":"%s","layer":"%s","type":"card-layer-eye"}.n_clicks'
                % (CARD, layer)
            ],
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    return {
        json.loads(serialized)["type"]: props
        for serialized, props in response.get_json()["response"].items()
    }


def test_clicking_a_series_row_switches_the_displayed_series(output_root):
    """Viv holds ONE image source, so a series row is a radio, not a checkbox."""
    app = create_app(output_root)
    resolved = _resolve(app, DATASET, STEM)
    spec = resolved["card-source-spec"]["data"]
    display = resolved["card-display-state"]["data"]

    updated = _click_eye(app, "gray", display, spec)["card-display-state"]["data"]
    assert updated["seriesPath"] == "gray"
    assert updated["labelVisible"] is True


def test_clicking_the_objmap_row_toggles_the_label_layer(output_root):
    app = create_app(output_root)
    resolved = _resolve(app, DATASET, STEM)
    spec = resolved["card-source-spec"]["data"]
    display = resolved["card-display-state"]["data"]

    updated = _click_eye(app, "objmap", display, spec)["card-display-state"][
        "data"
    ]
    assert updated["labelVisible"] is False
    # The displayed series is untouched: the two gestures are different in
    # kind, and conflating them would hide the image when hiding the label.
    assert updated["seriesPath"] == "rgb"
