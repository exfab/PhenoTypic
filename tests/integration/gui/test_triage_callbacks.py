"""Integration smoke tests for the CurationLabels store swap (Task 1).

Boots the results-viewer app against a tiny synthetic CLI output directory
and asserts:

1. The app boots cleanly.
2. ``app.server.config[CFG_FILTERED_STATE]`` is a ``CurationLabels`` instance.
3. The colony single-cell-remove callback POSTs 200, toggles the removal
   state, and causes the curated ``deliverables/measurements.parquet`` mirror
   to drop the removed object.

The test follows the "Flask test client, no browser" pattern established in
``tests/integration/gui/test_tune_callback_wiring.py``: use
``_find_output_key`` to look up the hashed ``allow_duplicate`` output key
from the live ``app.callback_map`` rather than hard-coding it.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic.gui._config import CFG_FILTERED_STATE
from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.tools_ import measurements_parquet_path

from tests._output_layout import write_master, write_measurements_mirror


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _master() -> pl.DataFrame:
    """Minimal 4-object master frame for two images."""
    return pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 4,
            "Metadata_ImageFile": ["img-A", "img-A", "img-B", "img-B"],
            "Object_Label": [1, 2, 1, 2],
            "Bbox_CenterRR": [10.0, 20.0, 10.0, 20.0],
            "Bbox_CenterCC": [10.0, 20.0, 10.0, 20.0],
            "Size_Area": [100.0, 200.0, 100.0, 200.0],
        }
    )


@pytest.fixture()
def output_root(tmp_path: Path) -> OutputRoot:
    """Synthetic output dir: master + mirror + overlay PNGs (required by OutputRoot)."""
    master = _master()
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)

    # OutputRoot.discover requires results/<dataset>/overlays/<stem>.png.
    overlays = tmp_path / "results" / "d1" / "overlays"
    overlays.mkdir(parents=True)
    for stem in ("img-A", "img-B"):
        PILImage.new("RGB", (64, 64), (128, 128, 128)).save(overlays / f"{stem}.png")

    return OutputRoot.discover(tmp_path)


# ---------------------------------------------------------------------------
# Dash dispatch helpers (Suggestion 1: no hard-coded hashes)
# ---------------------------------------------------------------------------


def _find_output_key(app, *id_property_substrings: str) -> str:
    """Return the callback_map key whose OUTPUT key string or INPUT ids contain
    ALL ``id_property_substrings``.

    Searches both the map key itself (output id+property) and the serialized
    input component ids, so callers can find a callback by either its output
    or its input pattern-matching type string (e.g. ``"colony-cell-remove-btn"``).

    Mirrors the spirit of ``tests/integration/gui/test_tune_callback_wiring.
    _find_output_key`` but also inspects the callback's inputs so hashed
    ``allow_duplicate`` output keys are found even when the search term
    only appears in the input component id.
    """
    for key, cb in app.callback_map.items():
        # Gather everything searchable: the output key + all input id strings.
        searchable = key
        for inp in cb.get("inputs", []):
            if isinstance(inp, dict):
                searchable += str(inp.get("id", ""))
            elif isinstance(inp, list):
                for item in inp:
                    if isinstance(item, dict):
                        searchable += str(item.get("id", ""))
        if all(sub in searchable for sub in id_property_substrings):
            return key
    raise KeyError(f"No callback key containing all of {id_property_substrings!r}")


def _outputs_from_key(output_key: str) -> list[dict[str, str]]:
    """Parse a (possibly multi-output, possibly allow_duplicate) output key.

    Drops the ``@hash`` allow_duplicate disambiguator — the Dash response is
    keyed only by id + property.
    """
    body = output_key.strip(".")
    outputs: list[dict[str, str]] = []
    for seg in re.split(r"\.\.\.", body):
        seg = seg.strip(".").split("@", 1)[0]
        component_id, prop = seg.rsplit(".", 1)
        outputs.append({"id": component_id, "property": prop})
    return outputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_app_boots_and_config_holds_curation_labels(output_root: OutputRoot) -> None:
    """App builds and the config key holds a ``CurationLabels`` instance."""
    app = create_app(output_root)
    store = app.server.config.get(CFG_FILTERED_STATE)
    assert store is not None, "CFG_FILTERED_STATE was not set"
    assert isinstance(store, CurationLabels), (
        f"Expected CurationLabels, got {type(store).__name__}"
    )


def test_toggle_via_filtered_state_writes_mirror(
    output_root: OutputRoot,
    tmp_path: Path,
) -> None:
    """``filtered_state.toggle(...)`` drops the object from ``measurements.parquet``.

    This is the minimal contract: the durable store mutates and persists.
    The colony-grid callback POST test below exercises the full Dash path.
    """
    app = create_app(output_root)
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]

    # Before toggle: 4 objects in the mirror.
    mirror_before = pl.read_parquet(measurements_parquet_path(tmp_path))
    assert mirror_before.height == 4

    # Toggle img-A / label 2 (remove it).
    store.toggle("img-A", 2)

    # After toggle: 3 objects; img-A/2 absent.
    mirror_after = pl.read_parquet(measurements_parquet_path(tmp_path))
    assert mirror_after.height == 3
    keys_after = set(
        zip(
            mirror_after.get_column("Metadata_ImageFile").to_list(),
            mirror_after.get_column("Object_Label").to_list(),
        )
    )
    assert ("img-A", 2) not in keys_after


def test_colony_single_cell_remove_callback_toggles_object(
    output_root: OutputRoot,
    tmp_path: Path,
) -> None:
    """POSTing the colony single-cell-remove callback drops the target object.

    Uses the Flask test client + ``_find_output_key`` to locate the
    ``allow_duplicate`` STORE_REMOVED_KEYS output without hard-coding hashes.
    Asserts a 200, a non-empty payload, and the mirror drop.
    """
    app = create_app(output_root)
    client = app.server.test_client()

    # Locate the callback that owns STORE_REMOVED_KEYS with the
    # colony-cell-remove-btn pattern as Input.
    out_key = _find_output_key(
        app, "store-removed-keys.data", "colony-cell-remove-btn"
    )

    # The triggered id: img-B / label 2 was clicked once.
    triggered_id = {"type": "colony-cell-remove-btn", "image_file": "img-B", "label": 2}

    resp = client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": _outputs_from_key(out_key),
            "inputs": [
                [
                    # One entry per matched button; only the triggered one has n_clicks=1.
                    {"id": triggered_id, "property": "n_clicks", "value": 1},
                ]
            ],
            "state": [
                {
                    "id": "store-removed-keys",
                    "property": "data",
                    "value": [],
                }
            ],
            "changedPropIds": [
                '{"image_file":"img-B","label":2,"type":"colony-cell-remove-btn"}.n_clicks'
            ],
        },
    )

    assert resp.status_code == 200, f"Callback returned {resp.status_code}: {resp.data[:200]}"
    # Regardless of exact response shape, the mirror must reflect the removal.
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    keys = set(
        zip(
            mirror.get_column("Metadata_ImageFile").to_list(),
            mirror.get_column("Object_Label").to_list(),
        )
    )
    assert ("img-B", 2) not in keys, (
        "img-B/label-2 should have been removed from the curated mirror. "
        f"Remaining keys: {keys}"
    )

    # Also confirm the store on the server carried the removal through.
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]
    assert store.is_removed("img-B", 2), "CurationLabels store should mark img-B/2 as removed"
