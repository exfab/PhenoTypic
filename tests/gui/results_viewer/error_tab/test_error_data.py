"""Tests for the Error-analysis tab's pure data layer.

Covers the verified-good derivation (from QC artifacts + review state),
good/error frame construction in both baseline modes, per-category
counts, the default-category pick, and the at-cutoff classification
metrics that drive the draggable readout. All Dash-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from phenotypic.gui.results_viewer._error_tab._data import (
    build_good_error_frames,
    category_counts,
    classify_at_cutoff,
    default_category,
    verified_good_keys,
)
from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
    ReviewState,
    encode_group_key,
)
from phenotypic.tools_ import (
    qc_members_parquet_path,
    qc_summary_parquet_path,
)

KEY_IMAGE_FILE = "Metadata_ImageFile"
KEY_OBJECT_LABEL = "Object_Label"


@dataclass
class FakeOutputRoot:
    """Minimal OutputRoot stand-in exposing ``.root`` + ``.master_df``."""

    root: Path
    master_df: pl.DataFrame


def _master_six() -> pl.DataFrame:
    """Six objects across two images, one Size_Area measurement column."""
    return pl.DataFrame(
        {
            KEY_IMAGE_FILE: ["img1", "img1", "img1", "img2", "img2", "img2"],
            KEY_OBJECT_LABEL: [1, 2, 3, 1, 2, 3],
            "Size_Area": [10.0, 11.0, 12.0, 100.0, 101.0, 102.0],
        }
    )


def _write_qc_artifacts(
    root: Path,
    groupby_col: str,
    *,
    group_a_value: str,
    group_b_value: str,
    members: list[tuple[str, str, int]],
    instance_id: str = "qc-SE-aaaa",
) -> None:
    """Write tiny qc_summary/qc_members parquets matching the runner schema.

    Args:
        root: Output root directory.
        groupby_col: Name of the single group-key column.
        group_a_value: Group-A key value.
        group_b_value: Group-B key value.
        members: ``(group_value, image_file, label)`` member rows.
        instance_id: The module instance id.
    """
    qc_summary_parquet_path(root).parent.mkdir(parents=True, exist_ok=True)
    summary = pl.DataFrame(
        {
            "instance_id": [instance_id, instance_id],
            "class": ["SE", "SE"],
            groupby_col: [group_a_value, group_b_value],
            "metric": [0.9, 0.1],
            "status": ["fail", "pass"],
            "flag": [True, False],
            "n_members": [2, 2],
            "n_flagged": [2, 0],
            "rank": [0, 1],
        }
    )
    summary.write_parquet(qc_summary_parquet_path(root))

    members_df = pl.DataFrame(
        {
            "instance_id": [instance_id] * len(members),
            groupby_col: [m[0] for m in members],
            KEY_IMAGE_FILE: [m[1] for m in members],
            KEY_OBJECT_LABEL: [m[2] for m in members],
            "member_value": [0.0] * len(members),
        }
    )
    members_df.write_parquet(qc_members_parquet_path(root))


# ---------------------------------------------------------------------------
# category_counts / default_category
# ---------------------------------------------------------------------------


def test_category_counts_tallies_per_token():
    labels = {
        ("img1", 1): "debris",
        ("img1", 2): "debris",
        ("img2", 1): "other",
    }
    assert category_counts(labels) == {"debris": 2, "other": 1}


def test_default_category_prefers_highest_non_other():
    counts = {"other": 5, "debris": 3, "merged": 1}
    assert default_category(counts, "other") == "debris"


def test_default_category_falls_back_to_other_when_only_other():
    counts = {"other": 4}
    assert default_category(counts, "other") == "other"


def test_default_category_none_when_empty():
    assert default_category({}, "other") is None


# ---------------------------------------------------------------------------
# verified_good_keys
# ---------------------------------------------------------------------------


def test_verified_good_keys_returns_unlabeled_members_of_reviewed_group(tmp_path: Path):
    root = tmp_path / "run"
    master = _master_six()
    out = FakeOutputRoot(root=root, master_df=master)
    # Group A holds img1's 3 objects; group B holds img2's 3 objects.
    _write_qc_artifacts(
        root,
        "plate",
        group_a_value="A",
        group_b_value="B",
        members=[
            ("A", "img1", 1),
            ("A", "img1", 2),
            ("A", "img1", 3),
            ("B", "img2", 1),
            ("B", "img2", 2),
            ("B", "img2", 3),
        ],
    )
    # Mark group A reviewed; (img1, 1) is labeled, so it is excluded.
    state = ReviewState.load(root)
    state.mark_reviewed("qc-SE-aaaa", ("A",))
    labeled = {("img1", 1)}

    keys = verified_good_keys(out, labeled)
    # img1/2 and img1/3 are reviewed-and-unlabeled; img1/1 excluded (labeled);
    # img2/* excluded (group B not reviewed).
    assert keys == {("img1", 2), ("img1", 3)}


def test_verified_good_keys_multicolumn_key_roundtrips(tmp_path: Path):
    root = tmp_path / "run"
    master = _master_six()
    out = FakeOutputRoot(root=root, master_df=master)
    instance_id = "qc-SE-bbbb"
    qc_summary_parquet_path(root).parent.mkdir(parents=True, exist_ok=True)
    summary = pl.DataFrame(
        {
            "instance_id": [instance_id],
            "class": ["SE"],
            "Metadata_Plate": ["plate1"],
            "Metadata_Group": ["A"],
            "metric": [0.9],
            "status": ["fail"],
            "flag": [True],
            "n_members": [2],
            "n_flagged": [2],
            "rank": [0],
        }
    )
    summary.write_parquet(qc_summary_parquet_path(root))
    members = pl.DataFrame(
        {
            "instance_id": [instance_id, instance_id],
            "Metadata_Plate": ["plate1", "plate1"],
            "Metadata_Group": ["A", "A"],
            KEY_IMAGE_FILE: ["img1", "img1"],
            KEY_OBJECT_LABEL: [2, 3],
            "member_value": [0.0, 0.0],
        }
    )
    members.write_parquet(qc_members_parquet_path(root))

    state = ReviewState.load(root)
    state.mark_reviewed(instance_id, ("plate1", "A"))
    # Sanity: the encoded key matches the review state's encoding.
    assert encode_group_key(("plate1", "A")) in state.modules[instance_id].reviewed

    keys = verified_good_keys(out, set())
    assert keys == {("img1", 2), ("img1", 3)}


def test_verified_good_keys_empty_when_artifacts_absent(tmp_path: Path):
    root = tmp_path / "run"
    out = FakeOutputRoot(root=root, master_df=_master_six())
    assert verified_good_keys(out, set()) == set()


# ---------------------------------------------------------------------------
# build_good_error_frames
# ---------------------------------------------------------------------------


def test_build_good_error_frames_all_unlabeled(tmp_path: Path):
    root = tmp_path / "run"
    out = FakeOutputRoot(root=root, master_df=_master_six())
    labels = {("img1", 1): "debris", ("img1", 2): "debris"}

    good_pdf, error_pdf = build_good_error_frames(
        out, labels, "debris", "all_unlabeled"
    )
    good_keys = set(
        zip(good_pdf[KEY_IMAGE_FILE].tolist(), good_pdf[KEY_OBJECT_LABEL].tolist())
    )
    error_keys = set(
        zip(error_pdf[KEY_IMAGE_FILE].tolist(), error_pdf[KEY_OBJECT_LABEL].tolist())
    )
    # error = labeled debris; good = all master rows NOT in labels.
    assert error_keys == {("img1", 1), ("img1", 2)}
    assert ("img1", 1) not in good_keys and ("img1", 2) not in good_keys
    assert len(good_keys) == 4
    assert "Size_Area" in good_pdf.columns


def test_build_good_error_frames_verified(tmp_path: Path):
    root = tmp_path / "run"
    out = FakeOutputRoot(root=root, master_df=_master_six())
    _write_qc_artifacts(
        root,
        "plate",
        group_a_value="A",
        group_b_value="B",
        members=[
            ("A", "img1", 1),
            ("A", "img1", 2),
            ("A", "img1", 3),
            ("B", "img2", 1),
            ("B", "img2", 2),
            ("B", "img2", 3),
        ],
    )
    state = ReviewState.load(root)
    state.mark_reviewed("qc-SE-aaaa", ("A",))
    labels = {("img1", 1): "debris"}

    good_pdf, error_pdf = build_good_error_frames(out, labels, "debris", "verified")
    good_keys = set(
        zip(good_pdf[KEY_IMAGE_FILE].tolist(), good_pdf[KEY_OBJECT_LABEL].tolist())
    )
    error_keys = set(
        zip(error_pdf[KEY_IMAGE_FILE].tolist(), error_pdf[KEY_OBJECT_LABEL].tolist())
    )
    # verified good = reviewed-and-unlabeled = img1/2, img1/3.
    assert good_keys == {("img1", 2), ("img1", 3)}
    # error is independent of group review state.
    assert error_keys == {("img1", 1)}


# ---------------------------------------------------------------------------
# classify_at_cutoff
# ---------------------------------------------------------------------------


def test_classify_at_cutoff_perfect_separation_high_side():
    good = np.array([1.0, 2.0, 3.0])
    error = np.array([8.0, 9.0, 10.0])
    out = classify_at_cutoff(good, error, 5.0, ">")
    assert out["recall"] == pytest.approx(1.0)
    assert out["specificity"] == pytest.approx(1.0)
    assert out["good_flagged"] == pytest.approx(0.0)


def test_classify_at_cutoff_low_specificity():
    good = np.array([1.0, 2.0, 3.0])
    error = np.array([8.0, 9.0, 10.0])
    # cutoff 2.5, ">" flags good values 3.0 (1 of 3) -> specificity 2/3.
    out = classify_at_cutoff(good, error, 2.5, ">")
    assert out["recall"] == pytest.approx(1.0)
    assert out["specificity"] == pytest.approx(2.0 / 3.0)
    assert out["good_flagged"] == pytest.approx(1.0)


def test_classify_at_cutoff_low_direction():
    good = np.array([8.0, 9.0, 10.0])
    error = np.array([1.0, 2.0, 3.0])
    out = classify_at_cutoff(good, error, 5.0, "<")
    assert out["recall"] == pytest.approx(1.0)
    assert out["specificity"] == pytest.approx(1.0)


def test_classify_at_cutoff_is_nan_safe():
    good = np.array([1.0, np.nan, 3.0])
    error = np.array([8.0, np.nan, 10.0])
    out = classify_at_cutoff(good, error, 5.0, ">")
    # NaN dropped: good = [1, 3], error = [8, 10].
    assert out["recall"] == pytest.approx(1.0)
    assert out["specificity"] == pytest.approx(1.0)
