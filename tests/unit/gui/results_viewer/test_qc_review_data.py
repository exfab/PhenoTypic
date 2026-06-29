"""Unit tests for the QC Review data layer + per-module review state.

Post-DuckDB cutover: the flat qc-parquet helpers are gone from
``review/_data.py``. Module slicing, the worklist, summary stats, and
member resolution now live behind the catalog-driven read API
(:mod:`phenotypic.gui.results_viewer._qc_tab.review._db`, covered by
``test_qc_db_api``). This module covers what remains:

- the picker-option builder the Review callback derives from
  ``_db.list_modules`` (catalog-driven),
- the recompute frame = post-applied ``measurements.parquet`` anti-joined
  with the removal set (NOT ``master − removed``),
- ``review_state.json`` round-trip + reset semantics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
from PIL import Image as PILImage

from phenotypic import ImagePipeline
from phenotypic.analysis.qc import MaxModifiedZScore
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._qc_tab.review import _callbacks, _data
from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
    ReviewState,
    decode_group_key,
    encode_group_key,
)
from phenotypic.sdk_ import BundleLayout, measurements_parquet_path
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import run_qc

from tests._output_layout import write_master, write_measurements_mirror


def _layout(tmp_path: Path) -> BundleLayout:
    """Full-run-style layout rooted at ``tmp_path`` (deliverables under it)."""
    return BundleLayout(
        deliverables_base=tmp_path / "deliverables", output_root=tmp_path
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_output_root(tmp_path: Path) -> OutputRoot:
    """Lay out a minimal CLI output dir (master + mirror + overlays).

    Two images each with two colonies, grouped under one dataset. No QC
    artifact is seeded here — the recompute-frame tests read only the
    measurements mirror / master.
    """
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 4,
            "Metadata_ImageFile": ["img-1", "img-1", "img-2", "img-2"],
            "Object_Label": [1, 2, 1, 2],
            "Bbox_CenterRR": [50, 60, 50, 60],
            "Bbox_CenterCC": [50, 60, 50, 60],
            "Bbox_MinRR": [40, 50, 40, 50],
            "Bbox_MaxRR": [60, 70, 60, 70],
            "Bbox_MinCC": [40, 50, 40, 50],
            "Bbox_MaxCC": [60, 70, 60, 70],
            "Size_Area": [100.0, 102.0, 300.0, 80.0],
        }
    )
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)

    (tmp_path / "results" / "d1" / "measurements").mkdir(
        parents=True, exist_ok=True
    )
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True)
    for stem in ("img-1", "img-2"):
        PILImage.new("RGB", (120, 120), (255, 0, 0)).save(
            overlays / f"{stem}.png"
        )

    return OutputRoot.discover(tmp_path)


# ---------------------------------------------------------------------------
# Module-picker options (catalog-driven)
# ---------------------------------------------------------------------------


def test_module_picker_options_from_catalog(tmp_path: Path) -> None:
    """The picker options come from the DuckDB catalog (recipe order)."""
    root = _write_output_root(tmp_path)
    pipe = ImagePipeline()
    pipe.set_qc(
        [
            QcRecipeEntry(
                cls=MaxModifiedZScore,
                params={"on": "Size_Area", "groupby": ["Metadata_ImageFile"]},
                instance_id="qc-ZMax-00000001",
                enabled=True,
            )
        ]
    )
    frame = pd.DataFrame(
        {
            "Metadata_ImageFile": ["img-1", "img-1", "img-2", "img-2"],
            "Object_Label": [1, 2, 1, 2],
            "Size_Area": [100.0, 102.0, 300.0, 80.0],
        }
    )
    run_qc(frame, pipe, root.root, qc_output_dir=root.layout.qc_dir)

    options = _callbacks._module_picker_options(root)
    assert options == [
        {"label": "MaxModifiedZScore (00000001)", "value": "qc-ZMax-00000001"}
    ]


def test_module_picker_options_empty_without_db(tmp_path: Path) -> None:
    """No qc.duckdb → an empty picker (graceful degradation)."""
    root = _write_output_root(tmp_path)
    assert _callbacks._module_picker_options(root) == []


def _component_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_component_text(item) for item in value)
    children = getattr(value, "children", None)
    return _component_text(children)


def test_review_empty_state_calls_out_legacy_qc_parquets(
    tmp_path: Path,
) -> None:
    root = _write_output_root(tmp_path)
    qc_dir = root.layout.qc_dir
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "qc_summary.parquet").write_bytes(b"legacy summary")

    text = _component_text(_callbacks._review_empty_state_children(root))

    assert "Legacy QC parquet artifacts" in text
    assert "qc.duckdb" in text
    assert "recompile" in text.lower()


# ---------------------------------------------------------------------------
# Recompute frame: post-applied mirror anti-joined with removals
# ---------------------------------------------------------------------------


def test_build_recompute_frame_anti_joins_removals(tmp_path: Path) -> None:
    root = _write_output_root(tmp_path)
    frame_all = _data.build_recompute_frame(root, set())
    assert len(frame_all) == 4

    frame = _data.build_recompute_frame(root, {("img-2", 1)})
    assert len(frame) == 3
    remaining = set(zip(frame["Metadata_ImageFile"], frame["Object_Label"]))
    assert ("img-2", 1) not in remaining
    assert ("img-2", 2) in remaining


def test_build_recompute_frame_reads_mirror_not_master(tmp_path: Path) -> None:
    """The recompute frame must come from measurements.parquet (post mirror).

    We give the mirror an extra metadata column the clean master lacks; the
    recompute frame must carry it (proving it read the mirror), since QC
    groupby often names metadata-only columns.
    """
    root = _write_output_root(tmp_path)
    # Rewrite the mirror with an extra metadata-only column.
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    mirror = mirror.with_columns(pl.lit("strainX").alias("Metadata_Strain"))
    mirror.write_parquet(measurements_parquet_path(tmp_path))

    frame = _data.build_recompute_frame(root, set())
    assert "Metadata_Strain" in frame.columns


# ---------------------------------------------------------------------------
# Review state: per-module persistence + reset
# ---------------------------------------------------------------------------


def test_review_state_round_trip(tmp_path: Path) -> None:
    state = ReviewState.load(_layout(tmp_path))
    key = ("img-2",)
    state.mark_reviewed("qc-SE-aaaa1111", key)
    state.set_last("qc-SE-aaaa1111", key)

    reloaded = ReviewState.load(_layout(tmp_path))
    assert reloaded.is_reviewed("qc-SE-aaaa1111", key)
    assert reloaded.reviewed_count("qc-SE-aaaa1111") == 1
    progress = reloaded.progress_for("qc-SE-aaaa1111")
    assert decode_group_key(progress.last) == key


def test_review_state_is_per_module(tmp_path: Path) -> None:
    state = ReviewState.load(_layout(tmp_path))
    state.mark_reviewed("module-A", ("g1",))
    assert state.is_reviewed("module-A", ("g1",))
    assert not state.is_reviewed("module-B", ("g1",))
    assert state.reviewed_count("module-B") == 0


def test_review_state_unmark(tmp_path: Path) -> None:
    state = ReviewState.load(_layout(tmp_path))
    state.mark_reviewed("m", ("g1",))
    state.unmark_reviewed("m", ("g1",))
    assert not state.is_reviewed("m", ("g1",))


def test_review_state_reset_when_file_cleared(tmp_path: Path) -> None:
    """Deleting review_state.json (CLI finalize reset) starts progress over."""
    from phenotypic.sdk_ import qc_review_state_path

    state = ReviewState.load(_layout(tmp_path))
    state.mark_reviewed("m", ("g1",))
    qc_review_state_path(tmp_path).unlink()  # simulate CLI finalize clear
    fresh = ReviewState.load(_layout(tmp_path))
    assert fresh.reviewed_count("m") == 0


def test_encode_decode_group_key_multicolumn() -> None:
    key = ("plate1", "A", 3)
    assert decode_group_key(encode_group_key(key)) == key
