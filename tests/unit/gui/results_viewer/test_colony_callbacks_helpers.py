"""Unit tests for the colony-view tab's pure callback helpers.

The bulk-mark + dropdown-option helpers live as module-level functions in
``colony_view._callbacks`` so the ``mutate_and_payload`` contract and the
dropdown option shape are testable without booting a Dash app.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer.colony_view._callbacks import (
    bulk_mark,
    category_dropdown_options,
    register_custom_category_safe,
)
from phenotypic.schema import ErrorCategory
from phenotypic.schema import METADATA


def _master() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 3,
            str(METADATA.IMAGE_NAME): ["img-A", "img-A", "img-B"],
            "Object_Label": [1, 2, 1],
            "Bbox_CenterRR": [10.0, 20.0, 30.0],
            "Bbox_CenterCC": [10.0, 20.0, 30.0],
            "Size_Area": [100.0, 200.0, 300.0],
        }
    )


def _store(tmp_path: Path) -> CurationLabels:
    from phenotypic.sdk_ import BundleLayout
    from tests._output_layout import write_master, write_measurements_mirror

    master = _master()
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)
    return CurationLabels.load(BundleLayout.detect(tmp_path), master)


# ---------------------------------------------------------------------------
# category_dropdown_options
# ---------------------------------------------------------------------------


def test_dropdown_options_cover_core_categories() -> None:
    """Every core ErrorCategory token appears as a dropdown value."""
    options = category_dropdown_options(ErrorCategory.labels())
    values = {opt["value"] for opt in options}
    for token in ErrorCategory.labels():
        assert token in values


def test_dropdown_options_humanize_label() -> None:
    """Underscored tokens render as spaced, title-cased labels."""
    options = category_dropdown_options(["background_noise"])
    assert options == [{"label": "Background Noise", "value": "background_noise"}]


def test_dropdown_options_include_custom_tokens() -> None:
    """Custom tokens are passed through with the same humanized shape."""
    options = category_dropdown_options([*ErrorCategory.labels(), "halo"])
    values = {opt["value"] for opt in options}
    assert "halo" in values


# ---------------------------------------------------------------------------
# bulk_mark
# ---------------------------------------------------------------------------


def test_bulk_mark_assigns_category_to_every_selected(tmp_path: Path) -> None:
    """``bulk_mark`` labels every selected key with the chosen category."""
    store = _store(tmp_path)
    selected = [("img-A", 1), ("img-A", 2), ("img-B", 1)]
    payload = bulk_mark(store, selected, "debris")
    for image_file, label in selected:
        assert store.labels[(image_file, label)] == "debris"
        assert [image_file, label] in payload


def test_bulk_mark_custom_category(tmp_path: Path) -> None:
    """A registered custom token marks the whole selection."""
    store = _store(tmp_path)
    token = store.register_custom_category("Halo")
    bulk_mark(store, [("img-A", 1)], token)
    assert store.labels[("img-A", 1)] == "halo"


# ---------------------------------------------------------------------------
# register_custom_category_safe
# ---------------------------------------------------------------------------


def test_register_custom_safe_returns_token_and_message(tmp_path: Path) -> None:
    """A valid name registers, returning the sanitized token + a confirmation."""
    store = _store(tmp_path)
    token, message = register_custom_category_safe(store, "Halo Ring")
    assert token == "halo_ring"
    assert "halo_ring" in message
    assert "halo_ring" in store.categories()


def test_register_custom_safe_empty_name_rejected(tmp_path: Path) -> None:
    """A blank name returns ``(None, message)`` and registers nothing."""
    store = _store(tmp_path)
    token, message = register_custom_category_safe(store, "   ")
    assert token is None
    assert message
    assert store.custom_categories == []


def test_register_custom_safe_none_name_rejected(tmp_path: Path) -> None:
    """``None`` is rejected the same as a blank name (no crash)."""
    store = _store(tmp_path)
    token, message = register_custom_category_safe(store, None)
    assert token is None
    assert message


def test_register_custom_safe_core_collision_rejected(tmp_path: Path) -> None:
    """A name colliding with a core token returns the registry's ValueError text."""
    store = _store(tmp_path)
    token, message = register_custom_category_safe(store, "debris")
    assert token is None
    assert "debris" in message
    assert "debris" not in store.custom_categories


# ---------------------------------------------------------------------------
# decode_wedge_trigger / apply_wedge_mark (shared radial-mark dispatch, P1)
# ---------------------------------------------------------------------------


def test_decode_wedge_trigger_returns_keys_on_real_click() -> None:
    """A concrete wedge id with a non-empty fire decodes to (image, label, cat)."""
    from phenotypic.gui._shared._triage_callbacks import decode_wedge_trigger

    triggered_id = {
        "type": "colony-cat-wedge",
        "image_file": "img-A",
        "label": "3",
        "category": "debris",
    }
    triggered_list = [{"prop_id": "x.n_clicks", "value": 1}]
    assert decode_wedge_trigger(triggered_id, triggered_list) == ("img-A", 3, "debris")


def test_decode_wedge_trigger_initial_empty_fire_is_none() -> None:
    """The ALL pattern's initial all-empty-n_clicks fire decodes to None."""
    from phenotypic.gui._shared._triage_callbacks import decode_wedge_trigger

    triggered_id = {
        "type": "colony-cat-wedge",
        "image_file": "img-A",
        "label": 3,
        "category": "debris",
    }
    triggered_list = [{"prop_id": "x.n_clicks", "value": None}]
    assert decode_wedge_trigger(triggered_id, triggered_list) is None


def test_decode_wedge_trigger_none_id_is_none() -> None:
    """A missing / non-dict triggered_id decodes to None."""
    from phenotypic.gui._shared._triage_callbacks import decode_wedge_trigger

    assert decode_wedge_trigger(None, []) is None


def test_decode_wedge_trigger_custom_folder_is_none() -> None:
    """The custom-folder placeholder is inert (opens the folder, never marks)."""
    from phenotypic.gui._shared._radial import RADIAL_CUSTOM_FOLDER_SENTINEL
    from phenotypic.gui._shared._triage_callbacks import decode_wedge_trigger

    triggered_id = {
        "type": "colony-cat-wedge",
        "image_file": "img-A",
        "label": 3,
        "category": RADIAL_CUSTOM_FOLDER_SENTINEL,
    }
    triggered_list = [{"prop_id": "x.n_clicks", "value": 1}]
    assert decode_wedge_trigger(triggered_id, triggered_list) is None


def test_apply_wedge_mark_assigns_category(tmp_path: Path) -> None:
    """A non-sentinel category marks the colony (durable removal)."""
    from phenotypic.gui._shared._triage_callbacks import apply_wedge_mark

    store = _store(tmp_path)
    apply_wedge_mark(store, "img-A", 1, "debris")
    assert store.labels[("img-A", 1)] == "debris"
    assert store.is_removed("img-A", 1)


def test_apply_wedge_mark_restore_sentinel_unmarks(tmp_path: Path) -> None:
    """The RADIAL_RESTORE_SENTINEL category clears a prior label (restore)."""
    from phenotypic.gui._shared._radial import RADIAL_RESTORE_SENTINEL
    from phenotypic.gui._shared._triage_callbacks import apply_wedge_mark

    store = _store(tmp_path)
    apply_wedge_mark(store, "img-A", 1, "debris")
    assert store.is_removed("img-A", 1)
    apply_wedge_mark(store, "img-A", 1, RADIAL_RESTORE_SENTINEL)
    assert not store.is_removed("img-A", 1)
    assert ("img-A", 1) not in store.labels
