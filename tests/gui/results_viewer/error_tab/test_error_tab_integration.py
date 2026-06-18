"""Integration tests for the Error-analysis tab wiring.

Drives the extracted module-level ``_recompute`` helper directly (per the
memory note: Dash callback bugs only fire on ``/_dash-update-component``,
so the load-bearing body lives in a unit-testable helper), plus a smoke
test that ``register_error_callbacks`` registers on a real ``dash.Dash``
and ``build_error_tab_body`` contains the table + figure ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import dash
import numpy as np
import polars as pl
import pytest

from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._error_tab import (
    build_error_tab_body,
    register_error_callbacks,
)
from phenotypic.gui.results_viewer._error_tab import _callbacks, _ids as ids
from phenotypic.gui.results_viewer._qc_tab.review._review_state import ReviewState
from phenotypic.sdk_ import (
    error_analysis_parquet_path,
    qc_members_parquet_path,
    qc_summary_parquet_path,
    verified_parquet_path,
)

from tests._output_layout import write_master, write_measurements_mirror

KEY_IMAGE_FILE = "Metadata_ImageFile"
KEY_OBJECT_LABEL = "Object_Label"


@dataclass
class FakeOutputRoot:
    """Minimal OutputRoot stand-in exposing ``.root`` + ``.master_df``."""

    root: Path
    master_df: pl.DataFrame

    @property
    def clean_master_df(self) -> pl.DataFrame:
        """The Error tab reads the clean master; here it is the same frame."""
        return self.master_df


def _separating_master() -> pl.DataFrame:
    """20 objects: 10 small-area errors vs 10 large-area good, clean split."""
    n = 10
    rng = np.random.default_rng(0)
    small = rng.normal(10.0, 0.5, n)
    large = rng.normal(500.0, 0.5, n)
    return pl.DataFrame(
        {
            KEY_IMAGE_FILE: ["img1"] * (2 * n),
            KEY_OBJECT_LABEL: list(range(1, 2 * n + 1)),
            "Size_Area": [*small, *large],
            "Shape_Circularity": rng.normal(0.8, 0.01, 2 * n),
        }
    )


@pytest.fixture()
def seeded_root(tmp_path: Path) -> FakeOutputRoot:
    """Output root with a separating master + measurements mirror."""
    master = _separating_master()
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)
    return FakeOutputRoot(root=tmp_path, master_df=master)


def _label_errors(root: Path, master: pl.DataFrame, n: int) -> CurationLabels:
    """Label the first ``n`` (small-area) objects as ``debris``."""
    state = CurationLabels.load(root, master)
    keys = [("img1", lbl) for lbl in range(1, n + 1)]
    state.mark_many(keys, "debris")
    return state


# ---------------------------------------------------------------------------
# Layout + registration smoke tests
# ---------------------------------------------------------------------------


def test_build_error_tab_body_contains_table_and_figure():
    body = build_error_tab_body(MagicMock(), MagicMock())
    rendered = str(body)
    assert ids.ERROR_TABLE_ID in rendered
    assert ids.ERROR_FIGURE_ID in rendered


def test_register_error_callbacks_registers_without_raising(seeded_root):
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = build_error_tab_body(seeded_root, MagicMock())
    filtered = CurationLabels.load(seeded_root.root, seeded_root.master_df)
    register_error_callbacks(app, seeded_root, filtered)
    # At least one callback now writes the chips/table/figure.
    outputs = "".join(app.callback_map.keys())
    assert ids.ERROR_TABLE_ID in outputs
    assert ids.ERROR_FIGURE_ID in outputs


# ---------------------------------------------------------------------------
# _recompute helper — all-unlabeled mode
# ---------------------------------------------------------------------------


def test_recompute_ranks_size_area_top_and_persists(seeded_root):
    filtered = _label_errors(seeded_root.root, seeded_root.master_df, 10)
    result = _callbacks._recompute(
        seeded_root, filtered, "debris", "all_unlabeled"
    )
    assert not result.empty_state
    # Size_Area separates cleanly; it must be the top-ranked measurement.
    assert result.table_data[0]["measurement"] == "Size_Area"
    # error_analysis.parquet was written with a leading category column (R3).
    path = error_analysis_parquet_path(seeded_root.root)
    assert path.is_file()
    written = pl.read_parquet(path)
    assert written.columns[0] == "category"
    assert set(written.get_column("category").to_list()) == {"debris"}


def test_recompute_empty_state_when_insufficient(seeded_root):
    # Only 3 labeled errors — below the default min_error_n (8).
    filtered = _label_errors(seeded_root.root, seeded_root.master_df, 3)
    result = _callbacks._recompute(
        seeded_root, filtered, "debris", "all_unlabeled"
    )
    assert result.empty_state
    assert not result.table_data
    # No parquet written in the insufficient state.
    assert not error_analysis_parquet_path(seeded_root.root).is_file()


# ---------------------------------------------------------------------------
# _recompute helper — verified mode
# ---------------------------------------------------------------------------


def _write_qc_one_group(root: Path, good_labels: list[int]) -> None:
    """Write a qc artifact whose single reviewed group holds ``good_labels``."""
    instance_id = "qc-SE-aaaa"
    qc_summary_parquet_path(root).parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "instance_id": [instance_id],
            "class": ["SE"],
            "plate": ["A"],
            "metric": [0.9],
            "status": ["fail"],
            "flag": [True],
            "n_members": [len(good_labels)],
            "n_flagged": [0],
            "rank": [0],
        }
    ).write_parquet(qc_summary_parquet_path(root))
    pl.DataFrame(
        {
            "instance_id": [instance_id] * len(good_labels),
            "plate": ["A"] * len(good_labels),
            KEY_IMAGE_FILE: ["img1"] * len(good_labels),
            KEY_OBJECT_LABEL: good_labels,
            "member_value": [0.0] * len(good_labels),
        }
    ).write_parquet(qc_members_parquet_path(root))
    state = ReviewState.load(root)
    state.mark_reviewed(instance_id, ("A",))


def test_recompute_verified_mode_writes_verified_parquet(seeded_root):
    # Label the 10 small-area objects as errors; the 10 large-area objects
    # (labels 11..20) are the good pool and all reviewed in one QC group.
    filtered = _label_errors(seeded_root.root, seeded_root.master_df, 10)
    _write_qc_one_group(seeded_root.root, list(range(11, 21)))

    all_unlabeled = _callbacks._recompute(
        seeded_root, filtered, "debris", "all_unlabeled"
    )
    verified = _callbacks._recompute(
        seeded_root, filtered, "debris", "verified"
    )
    assert not verified.empty_state
    # Verified good is the reviewed-and-unlabeled set (10), a subset of the
    # all-unlabeled good pool (10 here, since exactly those are unlabeled).
    assert verified.good_n <= all_unlabeled.good_n
    # verified.parquet written only in the non-degenerate verified branch (R4).
    assert verified_parquet_path(seeded_root.root).is_file()


def test_recompute_all_unlabeled_does_not_write_verified_parquet(seeded_root):
    filtered = _label_errors(seeded_root.root, seeded_root.master_df, 10)
    _callbacks._recompute(seeded_root, filtered, "debris", "all_unlabeled")
    assert not verified_parquet_path(seeded_root.root).is_file()
