"""Unit tests for the QC Review data layer + per-module review state.

Exercises the pure, Dash-free core the Review callbacks delegate to
(:mod:`phenotypic.gui.results_viewer._qc_tab.review._data` and
``._review_state``) against a synthetic ``qc/`` artifact + output root,
covering the spec §D risk refinements:

- module slicing + structural ``groupby`` recovery,
- worst-first frozen worklist order (``rank``),
- summary stats that count NaN/insufficient separately from ``pass`` and
  use a robust (inf/NaN-safe) median,
- timepoint faceting (and fallback when no timepoints),
- recompute frame = post-applied ``measurements.parquet`` anti-joined with
  the removal set (NOT ``master − removed``),
- ``review_state.json`` round-trip + reset semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._qc_tab.review import _data
from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
    ReviewState,
    decode_group_key,
    encode_group_key,
)
from phenotypic.tools_ import measurements_parquet_path

from tests._output_layout import write_master, write_measurements_mirror


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_output_root(tmp_path: Path, *, with_time: bool = False) -> OutputRoot:
    """Lay out a minimal CLI output dir with a two-group QC artifact.

    Two images (img-1: agreeing, img-2: disagreeing) each with two
    colonies; one module ``qc-SE`` grouped by ``Metadata_ImageFile``.
    """
    cols: dict[str, list] = {
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
    if with_time:
        cols["Metadata_Time"] = [0, 1, 0, 1]
    master = pl.DataFrame(cols)
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)

    overlays = tmp_path / "results" / "d1" / "overlays"
    overlays.mkdir(parents=True)
    for stem in ("img-1", "img-2"):
        PILImage.new("RGB", (120, 120), (255, 0, 0)).save(overlays / f"{stem}.png")

    qc = tmp_path / "qc"
    qc.mkdir()
    summary = pd.DataFrame(
        {
            "instance_id": ["qc-SE-aaaa1111", "qc-SE-aaaa1111"],
            "class": ["ReplicateAgreement", "ReplicateAgreement"],
            "Metadata_ImageFile": ["img-1", "img-2"],
            "metric": [0.05, 0.42],
            "status": ["pass", "fail"],
            "flag": [False, True],
            "n_members": [2, 2],
            "n_flagged": [0, 2],
            "rank": [1, 0],  # img-2 (fail) is worst → rank 0
        }
    )
    summary.to_parquet(qc / "qc_summary.parquet")
    members = pd.DataFrame(
        {
            "instance_id": ["qc-SE-aaaa1111"] * 4,
            "Metadata_ImageFile": ["img-1", "img-1", "img-2", "img-2"],
            "Object_Label": [1, 2, 1, 2],
            "member_value": [100.0, 102.0, 300.0, 80.0],
        }
    )
    members.to_parquet(qc / "qc_members.parquet")
    (qc / "qc_config.json").write_text(
        json.dumps({"qc": [{"instance_id": "qc-SE-aaaa1111"}]}), encoding="utf-8"
    )
    return OutputRoot.discover(tmp_path)


# ---------------------------------------------------------------------------
# Artifact slicing + worklist
# ---------------------------------------------------------------------------


def test_module_options_and_groupby_recovery(tmp_path: Path) -> None:
    root = _write_output_root(tmp_path)
    summary = _data.load_qc_summary(root)
    assert summary is not None

    assert _data.module_options(summary) == [
        {"label": "ReplicateAgreement (aaaa1111)", "value": "qc-SE-aaaa1111"}
    ]
    assert _data.groupby_cols_for(summary, "qc-SE-aaaa1111") == [
        "Metadata_ImageFile"
    ]


def test_worklist_is_worst_first(tmp_path: Path) -> None:
    """The worklist sorts ascending by ``rank`` (0 = worst), frozen."""
    root = _write_output_root(tmp_path)
    summary = _data.load_qc_summary(root)
    worklist = _data.module_worklist(summary, "qc-SE-aaaa1111")
    assert worklist.get_column("rank").to_list() == [0, 1]
    # Worst group (img-2, fail) leads.
    assert worklist.get_column("Metadata_ImageFile").to_list() == ["img-2", "img-1"]


def test_summary_stats_counts_and_robust_median(tmp_path: Path) -> None:
    root = _write_output_root(tmp_path)
    summary = _data.load_qc_summary(root)
    stats = _data.summary_stats(summary)
    assert stats["total"] == 2
    assert stats["fail"] == 1
    assert stats["pass"] == 1
    assert stats["insufficient"] == 0
    # median over {0.05, 0.42} = 0.235
    assert abs(stats["median_metric"] - 0.235) < 1e-9


def test_summary_stats_nan_is_insufficient_not_pass() -> None:
    """A NaN-metric group counts as insufficient, never as pass/green."""
    summary = pl.DataFrame(
        {
            "instance_id": ["x"] * 3,
            "class": ["ICC"] * 3,
            "Metadata_Dataset": ["a", "b", "c"],
            "metric": [float("nan"), 0.8, float("inf")],
            "status": ["pass", "pass", "warn"],
            "flag": [False, False, True],
            "n_members": [1, 3, 3],
            "n_flagged": [0, 0, 1],
            "rank": [2, 1, 0],
        }
    )
    stats = _data.summary_stats(summary)
    assert stats["insufficient"] == 1
    assert stats["pass"] == 1  # only the finite 0.8 / non-nan pass row
    assert stats["warn"] == 1
    # robust median ignores nan AND inf → median of {0.8} = 0.8
    assert stats["median_metric"] == 0.8


# ---------------------------------------------------------------------------
# Detail / gallery key resolution + faceting
# ---------------------------------------------------------------------------


def test_group_member_keys_resolves_dataset(tmp_path: Path) -> None:
    root = _write_output_root(tmp_path)
    members = _data.load_qc_members(root)
    dbi = _data.dataset_by_image_map(root)
    keys = _data.group_member_keys(
        members, "qc-SE-aaaa1111", ["Metadata_ImageFile"], ("img-2",), dbi
    )
    assert keys == [("d1", "img-2", 1), ("d1", "img-2", 2)]


def test_facet_fallback_when_no_timepoints(tmp_path: Path) -> None:
    root = _write_output_root(tmp_path, with_time=False)
    keys = [("d1", "img-2", 1), ("d1", "img-2", 2)]
    assert _data.time_by_key_map(root) == {}
    assert _data.facet_keys_by_timepoint(keys, {}) == [(None, keys)]


def test_facet_by_timepoint(tmp_path: Path) -> None:
    root = _write_output_root(tmp_path, with_time=True)
    tmap = _data.time_by_key_map(root)
    keys = [("d1", "img-2", 1), ("d1", "img-2", 2)]
    facets = _data.facet_keys_by_timepoint(keys, tmap)
    assert [t for t, _ in facets] == [0, 1]


# ---------------------------------------------------------------------------
# Recompute frame: post-applied mirror anti-joined with removals
# ---------------------------------------------------------------------------


def test_build_recompute_frame_anti_joins_removals(tmp_path: Path) -> None:
    root = _write_output_root(tmp_path)
    frame_all = _data.build_recompute_frame(root, set())
    assert len(frame_all) == 4

    frame = _data.build_recompute_frame(root, {("img-2", 1)})
    assert len(frame) == 3
    remaining = set(
        zip(frame["Metadata_ImageFile"], frame["Object_Label"])
    )
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
    state = ReviewState.load(tmp_path)
    key = ("img-2",)
    state.mark_reviewed("qc-SE-aaaa1111", key)
    state.set_last("qc-SE-aaaa1111", key)

    reloaded = ReviewState.load(tmp_path)
    assert reloaded.is_reviewed("qc-SE-aaaa1111", key)
    assert reloaded.reviewed_count("qc-SE-aaaa1111") == 1
    progress = reloaded.progress_for("qc-SE-aaaa1111")
    assert decode_group_key(progress.last) == key


def test_review_state_is_per_module(tmp_path: Path) -> None:
    state = ReviewState.load(tmp_path)
    state.mark_reviewed("module-A", ("g1",))
    assert state.is_reviewed("module-A", ("g1",))
    assert not state.is_reviewed("module-B", ("g1",))
    assert state.reviewed_count("module-B") == 0


def test_review_state_unmark(tmp_path: Path) -> None:
    state = ReviewState.load(tmp_path)
    state.mark_reviewed("m", ("g1",))
    state.unmark_reviewed("m", ("g1",))
    assert not state.is_reviewed("m", ("g1",))


def test_review_state_reset_when_file_cleared(tmp_path: Path) -> None:
    """Deleting review_state.json (CLI finalize reset) starts progress over."""
    from phenotypic.tools_ import qc_review_state_path

    state = ReviewState.load(tmp_path)
    state.mark_reviewed("m", ("g1",))
    qc_review_state_path(tmp_path).unlink()  # simulate CLI finalize clear
    fresh = ReviewState.load(tmp_path)
    assert fresh.reviewed_count("m") == 0


def test_encode_decode_group_key_multicolumn() -> None:
    key = ("plate1", "A", 3)
    assert decode_group_key(encode_group_key(key)) == key


@pytest.mark.parametrize("value", [None, float("nan")])
def test_eq_or_null_matches_null_group_key(tmp_path: Path, value) -> None:
    """A null/NaN group key is still selectable in group_record."""
    summary = pl.DataFrame(
        {
            "instance_id": ["x"],
            "class": ["C"],
            "Metadata_Plate": [None],
            "metric": [0.5],
            "status": ["warn"],
            "flag": [True],
            "n_members": [2],
            "n_flagged": [1],
            "rank": [0],
        }
    )
    record = _data.group_record(summary, "x", ["Metadata_Plate"], (value,))
    assert record is not None
    assert record["metric"] == 0.5
