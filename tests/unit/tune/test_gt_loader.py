"""4.3 — the path-configured ground-truth mask loader (DEFERRED-WORK §1).

``GroundTruthMasks`` mirrors ``ExpectedVsDetectedCount.metadata_source``: a serializable
``gt_masks_source`` path, an excluded resolved cache, and a ``model_validator`` that
captures the path. v1 covers **construction / round-trip / modality / abstain only** —
NOT numeric correctness against real annotated plates (that is deferred). Filename match
is directory + image stem with suffix priority ``[.npy, .tif, .png]``.
"""
from __future__ import annotations

import numpy as np
import pytest

from phenotypic.tune._scoring._gt_loader import GroundTruthMasks


@pytest.fixture
def mask_dir(tmp_path):
    """A directory of per-image GT masks keyed by image stem."""
    d = tmp_path / "gt"
    d.mkdir()
    np.save(d / "plate1.npy", np.array([[True, False], [False, True]]))
    # plate2 as a .tif (lower suffix priority than .npy, higher than .png).
    from skimage import io as skio

    skio.imsave(d / "plate2.tif", np.array([[1, 0], [0, 1]], dtype=np.uint8))
    return d


# --------------------------------------------------------------------------- #
# construction + round-trip (mirrors ExpectedVsDetectedCount.metadata_source)
# --------------------------------------------------------------------------- #
def test_constructs_from_path(mask_dir):
    gt = GroundTruthMasks(gt_masks_source=mask_dir)
    assert gt.gt_masks_source is not None


def test_path_survives_model_dump_json_round_trip(mask_dir):
    gt = GroundTruthMasks(gt_masks_source=mask_dir)
    payload = gt.model_dump_json()
    reloaded = GroundTruthMasks.model_validate_json(payload)
    assert str(reloaded.gt_masks_source) == str(mask_dir)
    # The reloaded loader still resolves masks from the persisted path.
    assert reloaded.masks_for("plate1") is not None


def test_in_memory_only_loader_fails_on_reload(mask_dir):
    # A loader with no source path cannot be rebuilt from JSON (mirrors the
    # ExpectedVsDetectedCount in-memory-only failure).
    none_loader = GroundTruthMasks(gt_masks_source=None)
    payload = none_loader.model_dump_json()
    # Round-trips to another sourceless loader that abstains (modality "none").
    reloaded = GroundTruthMasks.model_validate_json(payload)
    assert reloaded.gt_masks_source is None
    assert reloaded.modality() == "none"


# --------------------------------------------------------------------------- #
# masks_for + available_names
# --------------------------------------------------------------------------- #
def test_masks_for_resolves_by_stem(mask_dir):
    gt = GroundTruthMasks(gt_masks_source=mask_dir)
    mask = gt.masks_for("plate1")
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, np.array([[True, False], [False, True]]))


def test_masks_for_accepts_filename_with_suffix(mask_dir):
    # The image *name* may carry an extension; only the stem is matched.
    gt = GroundTruthMasks(gt_masks_source=mask_dir)
    assert gt.masks_for("plate1.png") is not None


def test_masks_for_reads_tif(mask_dir):
    gt = GroundTruthMasks(gt_masks_source=mask_dir)
    mask = gt.masks_for("plate2")
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, np.array([[True, False], [False, True]]))


def test_unknown_name_returns_none(mask_dir):
    gt = GroundTruthMasks(gt_masks_source=mask_dir)
    assert gt.masks_for("does_not_exist") is None


def test_available_names_lists_stems(mask_dir):
    gt = GroundTruthMasks(gt_masks_source=mask_dir)
    assert gt.available_names() == frozenset({"plate1", "plate2"})


def test_suffix_priority_prefers_npy(tmp_path):
    # When both .npy and .png exist for one stem, .npy wins (priority order).
    d = tmp_path / "gt"
    d.mkdir()
    np.save(d / "p.npy", np.array([[True, True]]))
    from skimage import io as skio

    skio.imsave(d / "p.png", np.array([[0, 0]], dtype=np.uint8))  # all-False
    gt = GroundTruthMasks(gt_masks_source=d)
    np.testing.assert_array_equal(gt.masks_for("p"), np.array([[True, True]]))


# --------------------------------------------------------------------------- #
# modality
# --------------------------------------------------------------------------- #
def test_modality_mask_when_masks_present(mask_dir):
    assert GroundTruthMasks(gt_masks_source=mask_dir).modality() == "mask"


def test_modality_none_without_source():
    assert GroundTruthMasks(gt_masks_source=None).modality() == "none"


def test_modality_none_for_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert GroundTruthMasks(gt_masks_source=empty).modality() == "none"


def test_modality_count_for_csv_source(tmp_path):
    # A CSV/Parquet source (counts, not per-image masks) reports "count".
    csv = tmp_path / "counts.csv"
    csv.write_text("image,count\nplate1,96\n")
    gt = GroundTruthMasks(gt_masks_source=csv)
    assert gt.modality() == "count"
    # A count source yields no per-image masks.
    assert gt.masks_for("plate1") is None
    assert gt.available_names() == frozenset()
