"""Unit tests for the QC tab's pure callback helpers.

The helpers live as module-level functions inside
``_qc_tab._callbacks`` so they can be tested without booting a Dash
app. Each test constructs a synthetic input inline.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import polars as pl

from phenotypic.gui._shared._radial import RADIAL_RESTORE_SENTINEL
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._qc_tab._callbacks import (
    _badge_color_for_status,
    _gather_modal_raw_values,
    _left_join_qc_columns,
    _merge_removed_keys,
    _render_summary_strip,
    _worst_status,
)
from phenotypic.gui.results_viewer._qc_tab.review._callbacks import (
    mark_review_tile,
)


def test_left_join_qc_columns_preserves_left_rows() -> None:
    """A left-join over partial QC results keeps every left row."""
    left = pl.DataFrame(
        {
            "Metadata_ImageFile": [f"img_{i}.tif" for i in range(10)],
            "Object_Label": list(range(10)),
            "Size_Area": [float(100 + i) for i in range(10)],
        }
    )
    right = pd.DataFrame(
        {
            "Metadata_ImageFile": ["img_0.tif", "img_1.tif", "img_2.tif"],
            "Object_Label": [0, 1, 2],
            "QC_SE_Metric": [0.01, 0.05, 0.12],
        }
    )
    result = _left_join_qc_columns(left, right)
    assert result.height == 10
    assert "QC_SE_Metric" in result.columns
    # Non-matching rows should be NaN/null.
    metric = result["QC_SE_Metric"].to_list()
    assert metric[0] is not None
    assert metric[9] is None


def test_render_summary_strip_format() -> None:
    """The summary strip text matches the documented shape.

    Built with the ``qc_``-prefixed column names ``QualityCheck.summary``
    now emits (``qc_n_members`` / ``qc_n_flagged`` / ``qc_worst_metric`` /
    ``qc_status``) so this test guards against the old stale-column reads.
    """
    summary = pd.DataFrame(
        {
            "Plate": ["P1", "P2", "P3"],
            "qc_n_members": [4, 4, 4],
            "qc_n_flagged": [1, 0, 0],
            "qc_worst_metric": [0.12, 0.04, 0.02],
            "qc_status": ["fail", "pass", "pass"],
        }
    )
    text = _render_summary_strip(summary)
    assert re.match(
        r"groups:\s+\d+\s+\|\s+flagged:\s+\d+\s+\|\s+worst metric:\s+\d+\.\d+",
        text,
    ), f"unexpected summary text: {text!r}"


def test_worst_status_pass_warn_fail() -> None:
    """``fail`` wins over ``warn`` which wins over ``pass``."""
    pass_only = pd.DataFrame({"qc_status": ["pass", "pass"]})
    pass_warn = pd.DataFrame({"qc_status": ["pass", "warn"]})
    pass_warn_fail = pd.DataFrame({"qc_status": ["pass", "warn", "fail"]})
    assert _worst_status(pass_only) == "pass"
    assert _worst_status(pass_warn) == "warn"
    assert _worst_status(pass_warn_fail) == "fail"


def test_badge_color_for_status() -> None:
    """Each status maps to the documented Bootstrap colour name."""
    assert _badge_color_for_status("pass") == "success"
    assert _badge_color_for_status("warn") == "warning"
    assert _badge_color_for_status("fail") == "danger"


def test_merge_removed_keys_dedupes() -> None:
    """Union of two payloads produces no duplicates."""
    current: list[list] = [["a", 1], ["b", 2]]
    new = [("b", 2), ("c", 3)]
    merged = _merge_removed_keys(current, new)
    assert merged == [["a", 1], ["b", 2], ["c", 3]]


# ---------------------------------------------------------------------------
# _gather_modal_raw_values — the Add/Edit modal's widget-state reducer
# ---------------------------------------------------------------------------


def _gather(
    simple=(),
    multi_tags=([], []),
    multi_values=([], []),
    column_scalars=([], []),
    column_modes=([], []),
):
    """Call the gatherer with the QC modal's prefix marker."""
    return _gather_modal_raw_values(
        prefix_marker="qc-modal-",
        simple=simple,
        multi_tags=multi_tags,
        multi_values=multi_values,
        column_scalars=column_scalars,
        column_modes=column_modes,
    )


def test_gather_maps_single_widgets_by_name() -> None:
    """Single-widget kinds map their value directly under the param name."""
    raw = _gather(
        simple=(
            ([0.9], [{"prefix": "qc-modal-X", "name": "warn_threshold"}]),
            (["/layout.csv"], [{"prefix": "qc-modal-X", "name": "metadata"}]),
        )
    )
    assert raw == {"warn_threshold": 0.9, "metadata": "/layout.csv"}


def test_gather_drops_out_of_scope_prefix() -> None:
    """Widgets whose prefix isn't this modal's are ignored (shared id types)."""
    raw = _gather(
        simple=(
            (["mine"], [{"prefix": "qc-modal-X", "name": "on"}]),
            (["leak"], [{"prefix": "analysis-Y", "name": "on"}]),
        )
    )
    assert raw == {"on": "mine"}


def test_gather_packs_multi_union_tag_value_pair() -> None:
    """Multi-union widgets repack to the (tag, value) tuple parse expects.

    Regression guard: the QC modal submit must collect ``param-multi-tag``
    + ``param-multi-value`` so a genuine multi-primitive union param is not
    silently dropped on Save.
    """
    raw = _gather(
        multi_tags=(["number"], [{"prefix": "qc-modal-X", "name": "thing"}]),
        multi_values=(["1.5"], [{"prefix": "qc-modal-X", "name": "thing"}]),
    )
    assert raw == {"thing": ("number", "1.5")}


def test_gather_pairs_multi_union_independent_of_order() -> None:
    """Tag/value are paired by (prefix, name), not by list position."""
    raw = _gather(
        multi_tags=(
            ["string", "none"],
            [
                {"prefix": "qc-modal-X", "name": "a"},
                {"prefix": "qc-modal-X", "name": "b"},
            ],
        ),
        multi_values=(
            ["", "hi"],
            [
                {"prefix": "qc-modal-X", "name": "b"},
                {"prefix": "qc-modal-X", "name": "a"},
            ],
        ),
    )
    assert raw == {"a": ("string", "hi"), "b": ("none", "")}


def test_gather_packs_column_with_alt_mode_scalar_pair() -> None:
    """A ``ColumnRef | None`` widget repacks to the (mode, scalar) tuple.

    ``param-column-mode`` + ``param-column-scalar`` sharing one
    (prefix, name) must pair so a "None" selection round-trips as ``None``
    via ``parse_widget_value``.
    """
    raw = _gather(
        column_modes=(["none"], [{"prefix": "qc-modal-X", "name": "cell_label"}]),
        column_scalars=(
            ["Grid_RowMajorIdx"],
            [{"prefix": "qc-modal-X", "name": "cell_label"}],
        ),
    )
    assert raw == {"cell_label": ("none", "Grid_RowMajorIdx")}


def test_gather_plain_column_scalar_stays_bare_value() -> None:
    """A plain ``ColumnRef`` scalar (no mode toggle) stays a bare value."""
    raw = _gather(
        column_scalars=(["Size_Area"], [{"prefix": "qc-modal-X", "name": "on"}]),
    )
    assert raw == {"on": "Size_Area"}


# ---------------------------------------------------------------------------
# mark_review_tile — the QC radial mark/restore pure helper (Task 5)
# ---------------------------------------------------------------------------


def _qc_master() -> pl.DataFrame:
    """Minimal 3-object master frame for the curation-labels helper tests."""
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


def _curation_store(tmp_path: Path) -> CurationLabels:
    """Build a CurationLabels store over a synthetic master + mirror."""
    from tests._output_layout import write_master, write_measurements_mirror

    master = _qc_master()
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)
    return CurationLabels.load(tmp_path, master)


def test_mark_review_tile_assigns_category(tmp_path: Path) -> None:
    """A core category token marks the colony and returns the payload."""
    store = _curation_store(tmp_path)
    payload = mark_review_tile(store, "img-A", 2, "debris")
    assert store.labels[("img-A", 2)] == "debris"
    assert store.is_removed("img-A", 2)
    # The payload carries the removed key as a [image_file, label] pair.
    assert ["img-A", 2] in payload


def test_mark_review_tile_restore_sentinel_clears(tmp_path: Path) -> None:
    """The restore sentinel clears a prior label and restores the object."""
    store = _curation_store(tmp_path)
    mark_review_tile(store, "img-A", 1, "merged")
    assert store.is_removed("img-A", 1)
    payload = mark_review_tile(store, "img-A", 1, RADIAL_RESTORE_SENTINEL)
    assert not store.is_removed("img-A", 1)
    assert ["img-A", 1] not in payload


def test_mark_review_tile_custom_category(tmp_path: Path) -> None:
    """A registered custom token marks the colony with that category."""
    store = _curation_store(tmp_path)
    token = store.register_custom_category("Halo")
    assert token == "halo"
    mark_review_tile(store, "img-B", 1, token)
    assert store.labels[("img-B", 1)] == "halo"
