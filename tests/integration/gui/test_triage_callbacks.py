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
from phenotypic.gui._shared._radial import RADIAL_RESTORE_SENTINEL
from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.sdk_ import (
    custom_categories_json_path,
    error_category_parquet_path,
    measurements_parquet_path,
)

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

    # OutputRoot.discover requires results/<dataset>/ dir + overlays under deliverables/.
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
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


# ---------------------------------------------------------------------------
# Colony radial triage (Task 4) — wedge mark + restore round-trip
# ---------------------------------------------------------------------------


def _post_colony_wedge(
    app,
    *,
    image_file: str,
    label: int,
    category: str,
):
    """POST a colony radial wedge click via the Dash update route.

    Resolves the ``STORE_REMOVED_KEYS`` allow_duplicate output owned by the
    ``colony-cat-wedge`` mark callback (no hard-coded hash) and fires it for
    the given ``(image_file, label, category)`` wedge.
    """
    client = app.server.test_client()
    out_key = _find_output_key(app, "store-removed-keys.data", "colony-cat-wedge")
    triggered_id = {
        "type": "colony-cat-wedge",
        "image_file": image_file,
        "label": label,
        "category": category,
    }
    # Dash serializes pattern-matched ids with keys sorted alphabetically.
    changed_prop = (
        '{"category":"%s","image_file":"%s","label":%d,"type":"colony-cat-wedge"}.n_clicks'
        % (category, image_file, label)
    )
    return client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": _outputs_from_key(out_key),
            "inputs": [
                [
                    # One entry per matched wedge; only the triggered one fires.
                    {"id": triggered_id, "property": "n_clicks", "value": 1},
                ]
            ],
            "changedPropIds": [changed_prop],
        },
    )


def test_colony_wedge_mark_writes_category_parquet_and_drops_mirror(
    output_root: OutputRoot,
    tmp_path: Path,
) -> None:
    """A ``debris`` wedge click categorizes the object and drops it from the mirror.

    Asserts ``deliverables/errors/debris.parquet`` gains the object and the
    curated ``deliverables/measurements.parquet`` no longer carries it.
    """
    app = create_app(output_root)

    resp = _post_colony_wedge(app, image_file="img-B", label=2, category="debris")
    assert resp.status_code == 200, (
        f"Wedge callback returned {resp.status_code}: {resp.data[:200]}"
    )

    # The per-category parquet gains img-B/label-2.
    debris_path = error_category_parquet_path(tmp_path, "debris")
    assert debris_path.exists(), "debris.parquet should be written after the mark"
    debris = pl.read_parquet(debris_path)
    debris_keys = set(
        zip(
            debris.get_column("Metadata_ImageFile").to_list(),
            debris.get_column("Object_Label").to_list(),
        )
    )
    assert ("img-B", 2) in debris_keys

    # The curated mirror drops it.
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    mirror_keys = set(
        zip(
            mirror.get_column("Metadata_ImageFile").to_list(),
            mirror.get_column("Object_Label").to_list(),
        )
    )
    assert ("img-B", 2) not in mirror_keys

    # And the in-memory store agrees, with the right category.
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]
    assert store.is_removed("img-B", 2)
    assert store.labels[("img-B", 2)] == "debris"


def test_colony_wedge_restore_round_trip(
    output_root: OutputRoot,
    tmp_path: Path,
) -> None:
    """A restore-sentinel wedge click clears a prior category and restores the object."""
    app = create_app(output_root)
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]

    # First mark img-A/label-1 as debris.
    resp = _post_colony_wedge(app, image_file="img-A", label=1, category="debris")
    assert resp.status_code == 200
    assert store.is_removed("img-A", 1)

    # Then fire the center restore node for the same object.
    resp = _post_colony_wedge(
        app, image_file="img-A", label=1, category=RADIAL_RESTORE_SENTINEL
    )
    assert resp.status_code == 200, (
        f"Restore callback returned {resp.status_code}: {resp.data[:200]}"
    )

    # The object is back in the curated mirror and no longer labeled.
    assert not store.is_removed("img-A", 1)
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    mirror_keys = set(
        zip(
            mirror.get_column("Metadata_ImageFile").to_list(),
            mirror.get_column("Object_Label").to_list(),
        )
    )
    assert ("img-A", 1) in mirror_keys, "restored object should be back in the mirror"


def test_colony_radial_body_lazy_populates_on_trigger_click(
    output_root: OutputRoot,
) -> None:
    """Clicking a tile's ▾ trigger fills the empty radial popover body.

    The body ships empty (build_radial_trigger); the populate-on-click MATCH
    callback fills it with the wedge ring. POST the trigger ``n_clicks`` and
    assert the response carries the core category wedges (e.g. ``debris``).
    """
    app = create_app(output_root)
    client = app.server.test_client()

    image_file, label = "img-A", 1
    trigger_id = {
        "type": "colony-radial-trigger",
        "image_file": image_file,
        "label": label,
    }
    store_id = {
        "type": "colony-radial-store",
        "image_file": image_file,
        "label": label,
    }
    body_id = {
        "type": "colony-radial-popover-body",
        "image_file": image_file,
        "label": label,
    }
    # Dispatch a MATCH callback: the ``output`` field is the registered
    # pattern key (with ``MATCH`` placeholders) from callback_map, while the
    # ``outputs``/``inputs``/``state`` carry the CONCRETE resolved ids.
    out_key = _find_output_key(
        app, "colony-radial-popover-body", "colony-radial-trigger"
    )
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": [{"id": body_id, "property": "children"}],
            "inputs": [{"id": trigger_id, "property": "n_clicks", "value": 1}],
            "state": [
                {
                    "id": store_id,
                    "property": "data",
                    "value": {
                        "image_file": image_file,
                        "label": label,
                        "surface": "colony",
                    },
                }
            ],
            "changedPropIds": [
                '{"image_file":"img-A","label":1,"type":"colony-radial-trigger"}.n_clicks'
            ],
        },
    )

    assert resp.status_code == 200, (
        f"Populate callback returned {resp.status_code}: {resp.data[:200]}"
    )
    # The response carries the wedge ring; assert a couple of core wedge ids
    # made it into the rendered body.
    body_text = resp.get_data(as_text=True)
    assert "colony-cat-wedge" in body_text
    assert "debris" in body_text
    assert RADIAL_RESTORE_SENTINEL in body_text


# ---------------------------------------------------------------------------
# Plate-view (viewer-card) status-cell restore-to-empty 500 regression
# ---------------------------------------------------------------------------


def _post_status_cell_toggle(
    app,
    *,
    card_index: str,
    image_file: str,
    label: int,
):
    """POST a plate-view Status-cell click via the ``card-details-table`` callback.

    The ``_toggle_status_cell`` callback is an ALL-pattern over every card's
    details DataTable; it resolves the card by id, reads the clicked row's
    ``(Metadata_ImageFile, Object_Label)`` from the table ``data``, and toggles
    the curated removal. Synthesizing one card's id + data + active_cell is
    enough to drive it without a fully-rendered card.
    """
    client = app.server.test_client()
    out_key = _find_output_key(app, "store-removed-keys.data", "card-details-table")
    table_id = {"type": "card-details-table", "index": card_index}
    row = {
        "Status": "Removed",
        "Metadata_ImageFile": image_file,
        "Object_Label": label,
    }
    return client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": _outputs_from_key(out_key),
            "inputs": [
                [
                    {
                        "id": table_id,
                        "property": "active_cell",
                        "value": {"row": 0, "column_id": "Status", "column": 0},
                    }
                ]
            ],
            "state": [
                [{"id": table_id, "property": "data", "value": [row]}],
                [{"id": table_id, "property": "id", "value": table_id}],
            ],
            "changedPropIds": [
                '{"index":"%s","type":"card-details-table"}.active_cell' % card_index
            ],
        },
    )


def test_plate_view_status_cell_restore_to_empty_no_500(
    output_root: OutputRoot,
) -> None:
    """Restoring the LAST removed object from a plate-view Status cell is a 200.

    The bug: ``_toggle_status_cell`` returned the bare ``mutate_and_payload``
    list; when restoring the final removal that list is ``[]``, which Dash's
    multi-mode (``allow_duplicate``) response validator treats as zero output
    values and 500s. The fix wraps the return in a 1-tuple so Dash always sees
    exactly one output value.
    """
    app = create_app(output_root)
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]

    # Seed a single removal so the toggle below restores-to-empty.
    store.toggle("img-A", 1)
    assert store.is_removed("img-A", 1)

    resp = _post_status_cell_toggle(
        app, card_index="card0", image_file="img-A", label=1
    )
    assert resp.status_code == 200, (
        f"Status-cell restore-to-empty returned {resp.status_code}: "
        f"{resp.data[:200]}"
    )
    # The object is restored and the label set is now empty.
    assert not store.is_removed("img-A", 1)
    assert store.removed_keys == set()


# ---------------------------------------------------------------------------
# QC review radial triage (Task 5) — wedge mark + lazy populate
# ---------------------------------------------------------------------------


def _post_qc_wedge(app, *, image_file: str, label: int, category: str):
    """POST a QC radial wedge click via the Dash update route (surface=qc)."""
    client = app.server.test_client()
    out_key = _find_output_key(app, "store-removed-keys.data", "qc-cat-wedge")
    triggered_id = {
        "type": "qc-cat-wedge",
        "image_file": image_file,
        "label": label,
        "category": category,
    }
    changed_prop = (
        '{"category":"%s","image_file":"%s","label":%d,"type":"qc-cat-wedge"}.n_clicks'
        % (category, image_file, label)
    )
    return client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": _outputs_from_key(out_key),
            "inputs": [
                [{"id": triggered_id, "property": "n_clicks", "value": 1}]
            ],
            "changedPropIds": [changed_prop],
        },
    )


def test_qc_wedge_mark_writes_category_parquet_and_drops_mirror(
    output_root: OutputRoot,
    tmp_path: Path,
) -> None:
    """A QC ``merged`` wedge click categorizes the object and drops the mirror."""
    app = create_app(output_root)

    resp = _post_qc_wedge(app, image_file="img-B", label=1, category="merged")
    assert resp.status_code == 200, (
        f"QC wedge callback returned {resp.status_code}: {resp.data[:200]}"
    )

    merged_path = error_category_parquet_path(tmp_path, "merged")
    assert merged_path.exists()
    merged = pl.read_parquet(merged_path)
    merged_keys = set(
        zip(
            merged.get_column("Metadata_ImageFile").to_list(),
            merged.get_column("Object_Label").to_list(),
        )
    )
    assert ("img-B", 1) in merged_keys

    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]
    assert store.is_removed("img-B", 1)
    assert store.labels[("img-B", 1)] == "merged"


def test_qc_radial_body_lazy_populates_on_trigger_click(
    output_root: OutputRoot,
) -> None:
    """Clicking a QC tile's ▾ trigger fills the empty radial popover body."""
    app = create_app(output_root)
    client = app.server.test_client()

    image_file, label = "img-A", 2
    trigger_id = {
        "type": "qc-radial-trigger",
        "image_file": image_file,
        "label": label,
    }
    store_id = {"type": "qc-radial-store", "image_file": image_file, "label": label}
    body_id = {
        "type": "qc-radial-popover-body",
        "image_file": image_file,
        "label": label,
    }
    out_key = _find_output_key(app, "qc-radial-popover-body", "qc-radial-trigger")
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": [{"id": body_id, "property": "children"}],
            "inputs": [{"id": trigger_id, "property": "n_clicks", "value": 1}],
            "state": [
                {
                    "id": store_id,
                    "property": "data",
                    "value": {
                        "image_file": image_file,
                        "label": label,
                        "surface": "qc",
                    },
                }
            ],
            "changedPropIds": [
                '{"image_file":"img-A","label":2,"type":"qc-radial-trigger"}.n_clicks'
            ],
        },
    )

    assert resp.status_code == 200, (
        f"QC populate callback returned {resp.status_code}: {resp.data[:200]}"
    )
    body_text = resp.get_data(as_text=True)
    assert "qc-cat-wedge" in body_text
    assert "merged" in body_text
    assert RADIAL_RESTORE_SENTINEL in body_text


# ---------------------------------------------------------------------------
# Bulk "Mark N selected as ▾" (Task 6) — colony + QC
# ---------------------------------------------------------------------------


def _post_bulk_mark(app, *, dropdown_id: str, category: str, selected):
    """POST a bulk-mark dropdown selection via the Dash update route.

    The bulk-mark callbacks are 3-output (multi-output), so the response is a
    fixed-arity list — no empty-payload 500 risk. The selection is passed as
    the ``STORE_COLONY_SELECTION`` State.
    """
    client = app.server.test_client()
    out_key = _find_output_key(app, "store-removed-keys.data", dropdown_id)
    selection = {
        "anchor": None,
        "selected": [[im, lbl] for im, lbl in selected],
    }
    return client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": _outputs_from_key(out_key),
            "inputs": [{"id": dropdown_id, "property": "value", "value": category}],
            "state": [
                {
                    "id": "store-colony-selection",
                    "property": "data",
                    "value": selection,
                }
            ],
            "changedPropIds": [f"{dropdown_id}.value"],
        },
    )


def test_colony_bulk_mark_marks_whole_selection(
    output_root: OutputRoot,
    tmp_path: Path,
) -> None:
    """The colony bulk-mark dropdown marks every selected colony with a category."""
    app = create_app(output_root)
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]

    resp = _post_bulk_mark(
        app,
        dropdown_id="colony-bulk-mark-dropdown",
        category="debris",
        selected=[("img-A", 1), ("img-B", 2)],
    )
    assert resp.status_code == 200, (
        f"Colony bulk-mark returned {resp.status_code}: {resp.data[:200]}"
    )
    assert store.labels[("img-A", 1)] == "debris"
    assert store.labels[("img-B", 2)] == "debris"
    # Per-category parquet carries both keys.
    debris = pl.read_parquet(error_category_parquet_path(tmp_path, "debris"))
    debris_keys = set(
        zip(
            debris.get_column("Metadata_ImageFile").to_list(),
            debris.get_column("Object_Label").to_list(),
        )
    )
    assert {("img-A", 1), ("img-B", 2)} <= debris_keys


def test_qc_bulk_mark_marks_whole_selection(
    output_root: OutputRoot,
) -> None:
    """The QC bulk-mark dropdown marks every selected colony with a category."""
    app = create_app(output_root)
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]

    resp = _post_bulk_mark(
        app,
        dropdown_id="qc-review-bulk-mark-dropdown",
        category="merged",
        selected=[("img-A", 2), ("img-B", 1)],
    )
    assert resp.status_code == 200, (
        f"QC bulk-mark returned {resp.status_code}: {resp.data[:200]}"
    )
    assert store.labels[("img-A", 2)] == "merged"
    assert store.labels[("img-B", 1)] == "merged"


# ---------------------------------------------------------------------------
# QC-review selection parity (M1) — the selection-delta consumer drives the
# SHARED selection store, resolving shift-ranges against the QC gallery's
# own order store.
#
# The full in-browser chain (JS bridge → QC delta store → consumer →
# STORE_COLONY_SELECTION → styler → .is-selected) is proven by the REAL
# e2e ``tests/e2e/gui/test_qc_review_splitter.py::
# test_qc_tile_click_selects_via_shared_store`` (no hand-injected State).
# These integration tests pin the two Dash-free pieces deterministically:
# (1) the consumer callback is actually registered with the QC delta store
# as its Input and the SHARED selection store as its Output; and (2) the
# pure fold helper resolves single-toggle + shift-range against the QC
# gallery's own order — the exact logic the consumer is a thin adapter
# over. (Driving the single-output ``allow_duplicate`` callback through the
# POST harness is intractable: ``validate_multi_return`` forces multi-mode
# on the synthesized ``outputs`` list; the e2e exercises the real path.)
# ---------------------------------------------------------------------------


def test_qc_selection_delta_consumer_is_registered(
    output_root: OutputRoot,
) -> None:
    """The QC selection-delta consumer is wired: QC delta in → shared store out.

    Proves the callback exists with the right Input (the QC gallery delta
    store the JS bridge writes) and Output (the SHARED selection store the
    bulk bar reads) — the wiring the broken-in-browser bug was about.
    """
    app = create_app(output_root)
    key = _find_output_key(
        app, "store-colony-selection.data", "store-qc-gallery-selection-delta"
    )
    cb = app.callback_map[key]
    # Single Output to the shared selection store.
    assert "store-colony-selection.data" in str(cb["output"])
    # Its sole Input is the QC gallery's delta store.
    input_ids = [
        inp.get("id") for inp in cb.get("inputs", []) if isinstance(inp, dict)
    ]
    assert "store-qc-gallery-selection-delta" in input_ids
    # And it reads the QC gallery order as State (for shift-range resolution).
    state_ids = [
        st.get("id") for st in cb.get("state", []) if isinstance(st, dict)
    ]
    assert "store-qc-gallery-order" in state_ids


def test_qc_selection_fold_single_toggle_sets_anchor() -> None:
    """A single (non-shift) QC tile click selects the key and sets the anchor.

    The consumer is a thin adapter over ``fold_selection_delta``; this pins
    the single-toggle semantics it produces for ``STORE_COLONY_SELECTION``.
    """
    from phenotypic.gui._shared._triage_callbacks import fold_selection_delta

    gallery_order = [["img-A", 1], ["img-A", 2], ["img-B", 1], ["img-B", 2]]
    payload = fold_selection_delta(
        {"key": ["img-A", 2], "shift": False, "ts": 1},
        {"anchor": None, "selected": []},
        gallery_order,
    )
    assert payload is not None
    assert payload["selected"] == [["img-A", 2]]
    assert payload["anchor"] == ["img-A", 2]


def test_qc_selection_fold_shift_range_resolves_against_gallery_order() -> None:
    """A shift-click resolves the inclusive range against the QC gallery order.

    The range A2..B1 selects exactly the slice of the QC gallery's own order
    between the prior anchor and the clicked key — proving the consumer
    resolves against ``STORE_QC_GALLERY_ORDER`` (passed through as the order
    payload), not the colony grid's order.
    """
    from phenotypic.gui._shared._triage_callbacks import fold_selection_delta

    gallery_order = [["img-A", 1], ["img-A", 2], ["img-B", 1], ["img-B", 2]]
    payload = fold_selection_delta(
        {"key": ["img-B", 1], "shift": True, "ts": 2},
        # Anchor already at A2 (set by a prior single click).
        {"anchor": ["img-A", 2], "selected": [["img-A", 2]]},
        gallery_order,
    )
    assert payload is not None
    selected = {tuple(entry) for entry in payload["selected"]}
    assert selected == {("img-A", 2), ("img-B", 1)}
    # The anchor is preserved across a shift-range extension.
    assert payload["anchor"] == ["img-A", 2]


# ---------------------------------------------------------------------------
# Add-custom-category from the radial folder (Task 7) — colony + QC
# ---------------------------------------------------------------------------


def _post_radial_custom_add(
    app,
    *,
    surface: str,
    image_file: str,
    label: int,
    name: str,
):
    """POST a radial ＋ Add-custom submit via the Dash update route.

    The submit callback is a 3-output MATCH callback (body re-render + inline
    message + vocab-revision bump). ``output`` is the registered MATCH pattern
    key; the concrete resolved ids go in ``outputs``/``inputs``/``state``.
    """
    client = app.server.test_client()
    out_key = _find_output_key(
        app, f"{surface}-radial-popover-body", f"{surface}-radial-custom-submit"
    )
    submit_id = {
        "type": f"{surface}-radial-custom-submit",
        "image_file": image_file,
        "label": label,
    }
    input_id = {
        "type": f"{surface}-radial-custom-input",
        "image_file": image_file,
        "label": label,
    }
    body_id = {
        "type": f"{surface}-radial-popover-body",
        "image_file": image_file,
        "label": label,
    }
    msg_id = {
        "type": f"{surface}-radial-custom-msg",
        "image_file": image_file,
        "label": label,
    }
    return client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": [
                {"id": body_id, "property": "children"},
                {"id": msg_id, "property": "children"},
                {"id": "store-category-vocab-revision", "property": "data"},
            ],
            "inputs": [
                # The ＋ Add button n_clicks (the firing Input here) plus the
                # input's n_submit (Enter-to-submit, S1) — registration order.
                {"id": submit_id, "property": "n_clicks", "value": 1},
                {"id": input_id, "property": "n_submit", "value": None},
            ],
            "state": [
                {"id": input_id, "property": "value", "value": name},
                {
                    "id": "store-category-vocab-revision",
                    "property": "data",
                    "value": 0,
                },
            ],
            "changedPropIds": [
                '{"image_file":"%s","label":%d,"type":"%s-radial-custom-submit"}.n_clicks'
                % (image_file, label, surface)
            ],
        },
    )


def test_colony_custom_add_registers_and_persists(
    output_root: OutputRoot,
    tmp_path: Path,
) -> None:
    """Submitting ＋ Add 'Halo' registers a custom category and persists it.

    Asserts: 200, ``categories()`` includes ``halo``, the registry json
    persists, and the re-rendered body carries a ``halo`` custom wedge.
    """
    app = create_app(output_root)
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]

    resp = _post_radial_custom_add(
        app, surface="colony", image_file="img-A", label=1, name="Halo"
    )
    assert resp.status_code == 200, (
        f"Custom-add returned {resp.status_code}: {resp.data[:200]}"
    )

    # The sanitized token is registered + persisted.
    assert "halo" in store.categories()
    registry = custom_categories_json_path(tmp_path)
    assert registry.exists()
    assert "halo" in registry.read_text(encoding="utf-8")

    # The re-rendered body carries the new custom wedge (clickable as a
    # colony-cat-wedge with category 'halo').
    body_text = resp.get_data(as_text=True)
    assert "halo" in body_text
    assert "colony-cat-wedge" in body_text


def test_colony_custom_add_via_enter_submits(
    output_root: OutputRoot,
    tmp_path: Path,
) -> None:
    """Pressing Enter in the ＋ Add input (``n_submit``) registers the category (S1).

    The submit callback now also has the input's ``n_submit`` as an Input;
    this drives that path (the input id is the trigger, not the button) and
    asserts the category is registered exactly as a button click would.
    """
    app = create_app(output_root)
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]
    client = app.server.test_client()

    out_key = _find_output_key(
        app, "colony-radial-popover-body", "colony-radial-custom-submit"
    )
    submit_id = {"type": "colony-radial-custom-submit", "image_file": "img-A", "label": 1}
    input_id = {"type": "colony-radial-custom-input", "image_file": "img-A", "label": 1}
    body_id = {"type": "colony-radial-popover-body", "image_file": "img-A", "label": 1}
    msg_id = {"type": "colony-radial-custom-msg", "image_file": "img-A", "label": 1}
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": [
                {"id": body_id, "property": "children"},
                {"id": msg_id, "property": "children"},
                {"id": "store-category-vocab-revision", "property": "data"},
            ],
            "inputs": [
                {"id": submit_id, "property": "n_clicks", "value": None},
                {"id": input_id, "property": "n_submit", "value": 1},
            ],
            "state": [
                {"id": input_id, "property": "value", "value": "Ghost"},
                {"id": "store-category-vocab-revision", "property": "data", "value": 0},
            ],
            # Enter fires the INPUT's n_submit, not the button.
            "changedPropIds": [
                '{"image_file":"img-A","label":1,"type":"colony-radial-custom-input"}.n_submit'
            ],
        },
    )
    assert resp.status_code == 200, (
        f"Enter-submit returned {resp.status_code}: {resp.data[:200]}"
    )
    assert "ghost" in store.categories()
    body_text = resp.get_data(as_text=True)
    assert "ghost" in body_text


def test_colony_custom_add_empty_name_is_rejected_inline(
    output_root: OutputRoot,
) -> None:
    """A blank custom name is rejected with an inline message, not a crash."""
    app = create_app(output_root)
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]

    resp = _post_radial_custom_add(
        app, surface="colony", image_file="img-A", label=1, name="   "
    )
    assert resp.status_code == 200
    # No category registered; an inline message was surfaced.
    assert store.custom_categories == []
    body_text = resp.get_data(as_text=True)
    assert "category name" in body_text.lower()


def test_qc_custom_add_registers_and_persists(
    output_root: OutputRoot,
    tmp_path: Path,
) -> None:
    """The QC surface's ＋ Add-custom registers + persists a custom category."""
    app = create_app(output_root)
    store: CurationLabels = app.server.config[CFG_FILTERED_STATE]

    resp = _post_radial_custom_add(
        app, surface="qc", image_file="img-B", label=1, name="Ghost"
    )
    assert resp.status_code == 200, (
        f"QC custom-add returned {resp.status_code}: {resp.data[:200]}"
    )
    assert "ghost" in store.categories()
    assert custom_categories_json_path(tmp_path).exists()
    body_text = resp.get_data(as_text=True)
    assert "ghost" in body_text
    assert "qc-cat-wedge" in body_text
