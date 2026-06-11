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
)
from phenotypic.schema import ErrorCategory


def _master() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 3,
            "Metadata_ImageFile": ["img-A", "img-A", "img-B"],
            "Object_Label": [1, 2, 1],
            "Bbox_CenterRR": [10.0, 20.0, 30.0],
            "Bbox_CenterCC": [10.0, 20.0, 30.0],
            "Size_Area": [100.0, 200.0, 300.0],
        }
    )


def _store(tmp_path: Path) -> CurationLabels:
    from tests._output_layout import write_master, write_measurements_mirror

    master = _master()
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)
    return CurationLabels.load(tmp_path, master)


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
