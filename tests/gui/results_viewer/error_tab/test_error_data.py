"""Tests for the Error-analysis tab's pure data layer.

Covers the verified-good derivation (from QC artifacts + review state),
good/error frame construction in both baseline modes, per-category
counts, the default-category pick, and the at-cutoff classification
metrics that drive the draggable readout. All Dash-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis.qc import MaxModifiedZScore
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
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import run_qc
from phenotypic.schema import METADATA

KEY_IMAGE_FILE = str(METADATA.IMAGE_NAME)
KEY_OBJECT_LABEL = "Object_Label"


@dataclass
class FakeOutputRoot:
    """Minimal OutputRoot stand-in exposing ``.root`` + ``.master_df`` + ``.layout``."""

    root: Path
    master_df: pl.DataFrame

    @property
    def layout(self):
        """Full-run-style :class:`BundleLayout` rooted at ``self.root``."""
        from phenotypic.sdk_ import BundleLayout

        return BundleLayout(
            deliverables_base=self.root / "deliverables", output_root=self.root
        )

    @property
    def clean_master_df(self) -> pl.DataFrame:
        """The Error tab reads the clean master; here it is the same frame."""
        return self.master_df


def _master_six() -> pl.DataFrame:
    """Six objects across two images, one Size_Area measurement column."""
    return pl.DataFrame(
        {
            KEY_IMAGE_FILE: ["img1", "img1", "img1", "img2", "img2", "img2"],
            KEY_OBJECT_LABEL: [1, 2, 3, 1, 2, 3],
            "Size_Area": [10.0, 11.0, 12.0, 100.0, 101.0, 102.0],
        }
    )


def _seed_qc_db(
    root: Path,
    frame: pd.DataFrame,
    *,
    groupby: list[str],
    instance_id: str = "qc-ZMax-aaaa0001",
) -> None:
    """Seed ``<root>/deliverables/qc/qc.duckdb`` via the real ``run_qc`` writer.

    Runs a single :class:`MaxModifiedZScore` (a per-object,
    curation-supporting check) grouped by ``groupby`` over ``frame`` so the
    member table's ``(Metadata_ImageName, Object_Label)`` rows mirror a real
    QC artifact — exactly what the Error tab's verified-good derivation
    reads through the DuckDB catalog.

    Args:
        root: Output root directory.
        frame: The measurement frame to analyze (must carry ``Size_Area`` +
            the ``groupby`` columns + the curation key columns).
        groupby: The check's group-key columns.
        instance_id: The module instance id.
    """
    pipe = ImagePipeline()
    pipe.set_qc(
        [
            QcRecipeEntry(
                cls=MaxModifiedZScore,
                params={"on": "Size_Area", "groupby": groupby},
                instance_id=instance_id,
                enabled=True,
            )
        ]
    )
    run_qc(frame, pipe, root)


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
    # Group by image: group "img1" holds img1's 3 objects; "img2" holds img2's.
    _seed_qc_db(
        root, master.to_pandas(), groupby=[KEY_IMAGE_FILE], instance_id="qc-ZMax-a1"
    )
    # Mark group img1 reviewed; (img1, 1) is labeled, so it is excluded.
    state = ReviewState.load(out.layout)
    state.mark_reviewed("qc-ZMax-a1", ("img1",))
    labeled = {("img1", 1)}

    keys = verified_good_keys(out, labeled)
    # img1/2 and img1/3 are reviewed-and-unlabeled; img1/1 excluded (labeled);
    # img2/* excluded (group img2 not reviewed).
    assert keys == {("img1", 2), ("img1", 3)}


def test_verified_good_keys_multicolumn_key_roundtrips(tmp_path: Path):
    root = tmp_path / "run"
    master = _master_six()
    out = FakeOutputRoot(root=root, master_df=master)
    instance_id = "qc-ZMax-bbbb"
    # A 2-column groupby: only img1/2 and img1/3 fall in (plate1, A).
    frame = pd.DataFrame(
        {
            KEY_IMAGE_FILE: ["img1", "img1"],
            KEY_OBJECT_LABEL: [2, 3],
            "Metadata_Plate": ["plate1", "plate1"],
            "Metadata_Group": ["A", "A"],
            "Size_Area": [11.0, 12.0],
        }
    )
    _seed_qc_db(
        root,
        frame,
        groupby=["Metadata_Plate", "Metadata_Group"],
        instance_id=instance_id,
    )

    state = ReviewState.load(out.layout)
    state.mark_reviewed(instance_id, ("plate1", "A"))
    # Sanity: the encoded key matches the review state's encoding.
    assert encode_group_key(("plate1", "A")) in state.modules[instance_id].reviewed

    keys = verified_good_keys(out, set())
    assert keys == {("img1", 2), ("img1", 3)}


def test_verified_good_keys_empty_when_artifacts_absent(tmp_path: Path):
    root = tmp_path / "run"
    out = FakeOutputRoot(root=root, master_df=_master_six())
    assert verified_good_keys(out, set()) == set()


def test_verified_good_keys_skips_diagnostic_only_modules(tmp_path: Path):
    """A reviewed group of a diagnostic-only module contributes nothing.

    Seeds a real per-object curation module (its reviewed group DOES
    contribute), then flips its catalog ``supports_object_curation`` flag to
    ``False`` (mirroring a GridOccupancy-style diagnostic module) and asserts
    the same reviewed group now contributes no verified-good members.
    """
    root = tmp_path / "run"
    master = _master_six()
    out = FakeOutputRoot(root=root, master_df=master)
    _seed_qc_db(
        root, master.to_pandas(), groupby=[KEY_IMAGE_FILE], instance_id="qc-ZMax-a1"
    )
    state = ReviewState.load(out.layout)
    state.mark_reviewed("qc-ZMax-a1", ("img1",))

    # As a curation module the reviewed group contributes its members.
    assert verified_good_keys(out, set()) == {("img1", 1), ("img1", 2), ("img1", 3)}

    # Flip it to diagnostic-only; now it must be skipped entirely.
    con = duckdb.connect(str(out.layout.qc_duckdb))
    con.execute("UPDATE qc_modules SET supports_object_curation = false")
    con.close()
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
    master = _master_six()
    out = FakeOutputRoot(root=root, master_df=master)
    _seed_qc_db(
        root, master.to_pandas(), groupby=[KEY_IMAGE_FILE], instance_id="qc-ZMax-a1"
    )
    state = ReviewState.load(out.layout)
    state.mark_reviewed("qc-ZMax-a1", ("img1",))
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
