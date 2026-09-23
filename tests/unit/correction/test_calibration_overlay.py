"""The calibration tile overlay: its record (spec §1) and its figure (§2)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.correction import CalibrateColorRpcc

from ._checker_frames import (
    SRGB,
    TILE,
    TOP,
    _band_patch,
    band_rois,
    frozen_op,
    quietly,
    render_frame,
)

GREEN = np.array([0.1, 0.95, 0.1])


def planted_faults() -> np.ndarray:
    """Three neutrals painted green (outliers) and a grey occluder on orange."""
    arr = render_frame(overrides={(1, r, 1): GREEN for r in (1, 2, 3)})
    arr[TOP:TOP + TILE, 85:85 + int(0.35 * TILE)] = 128
    return arr


def calibrated(arr: np.ndarray, **kwargs) -> CalibrateColorRpcc:
    operation = frozen_op(**kwargs)
    quietly(operation, Image(arr=arr))
    return operation


def tile(record, roi: int, row: int, col: int):
    (found,) = [t for t in record.rois[roi].tiles if (t.row, t.col) == (row, col)]
    return found


# -- spec test 1: record fidelity -----------------------------------------
def test_record_statuses_follow_the_rule_table() -> None:
    record = calibrated(planted_faults(), on_qc_fail="warn").calibration_record

    assert record.verdict == "corrected_with_warnings"
    assert tile(record, 0, 0, 1).status == "partly_covered"       # occluded orange
    for row in (1, 2, 3):
        assert tile(record, 1, row, 1).status == "rejected"         # green neutrals
    assert tile(record, 0, 0, 0).status == "used"
    assert record.n_fitted == 21 and record.n_expected == 24


def test_record_values_match_the_run() -> None:
    operation = calibrated(planted_faults(), on_qc_fail="warn")
    record = operation.calibration_record
    patches = operation.fitted_profile.diagnostics["patches"]
    tiles = {(t["roi_index"], t["row"], t["col"]): t for t in operation.diagnostics["tiles"]}

    for roi in record.rois:
        for t in roi.tiles:
            source = tiles[(roi.roi_index, t.row, t.col)]
            assert t.patch == source["patch"]
            assert t.measured_srgb == pytest.approx(source["srgb"])
            assert t.delta_e_before == pytest.approx(patches[t.patch]["deltaE00_before"])
            assert t.delta_e_after == pytest.approx(patches[t.patch]["deltaE00_after"])


# -- spec test 2: the crop is as shot, and owns its memory ------------------
def test_crop_is_the_as_shot_pixels_and_owns_its_buffer() -> None:
    arr = planted_faults()
    operation = frozen_op(on_qc_fail="warn")
    out = quietly(operation, Image(arr=arr))
    roi = operation.rois[0]
    crop = operation.calibration_record.rois[0].crop

    np.testing.assert_array_equal(crop, arr[roi.row_slice, roi.col_slice])
    assert not np.array_equal(crop, out.rgb[roi.row_slice, roi.col_slice])
    assert crop.base is None


# -- spec test 3: refused and skipped frames keep a record ------------------
def test_a_gate_refusal_keeps_a_record() -> None:
    operation = frozen_op()                                   # on_qc_fail="raise"
    with pytest.raises(RuntimeError, match="quality gate failed"):
        quietly(operation, Image(arr=render_frame(gain=1.6)))  # saturated card

    record = operation.calibration_record
    assert record.verdict == "refused" and "quality gate failed" in record.refusal
    assert record.n_fitted is None
    assert {t.status for roi in record.rois for t in roi.tiles} <= {"excluded", "empty"}
    assert all(t.delta_e_after is None for roi in record.rois for t in roi.tiles)


def test_a_skipped_frame_keeps_a_record() -> None:
    operation = calibrated(render_frame(gain=1.6), on_qc_fail="skip")
    assert operation.calibration_record.verdict == "skipped"


def test_no_usable_tiles_keeps_a_record_with_no_lattice() -> None:
    rng = np.random.default_rng(0)
    blank = rng.normal(120, 2, render_frame().shape).clip(0, 255).astype(np.uint8)
    operation = CalibrateColorRpcc(rois=band_rois(), grid=(6, 2), on_qc_fail="warn")
    with pytest.raises(RuntimeError, match="No ROI produced usable tiles"):
        quietly(operation, Image(arr=blank))

    record = operation.calibration_record
    assert record.verdict == "refused"
    assert [roi.lattice_found for roi in record.rois] == [False, False]
    assert all(roi.flags for roi in record.rois)


def test_a_post_rejection_rank_failure_keeps_a_record() -> None:
    operation = frozen_op(degree=4, on_qc_fail="warn")
    with pytest.raises(RuntimeError, match="remain after outlier rejection"):
        quietly(operation, Image(arr=render_frame(
                overrides={(1, r, 1): GREEN for r in (1, 2, 3)})))
    assert operation.calibration_record.verdict == "refused"


def collided_frame() -> np.ndarray:
    """ROI 1 shows ROI 0's top row, so its placement collides and it claims nothing."""
    grey = np.array([0.3, 0.3, 0.3])
    overrides = {(1, 0, c): SRGB[_band_patch(0, 0, c)] for c in (0, 1)}
    overrides |= {(1, r, c): grey for r in range(1, 6) for c in (0, 1)}
    return render_frame(overrides=overrides)


def test_a_pre_fit_rank_failure_keeps_a_record() -> None:
    # Only ROI 0's 12 patches are measured: below degree 3's 13 terms.
    operation = frozen_op(on_qc_fail="warn")
    with pytest.raises(RuntimeError, match="were measured"):
        quietly(operation, Image(arr=collided_frame()))

    record = operation.calibration_record
    assert record.verdict == "refused" and "were measured" in record.refusal
    assert record.n_fitted is None


def test_a_collided_roi_marks_its_tiles_excluded() -> None:
    # degree 2: ROI 1 loses its patches, leaving ROI 0's 12, below degree 3's 13 terms.
    record = calibrated(collided_frame(), on_qc_fail="warn",
                        degree=2).calibration_record

    assert any("already claimed" in flag for flag in record.rois[1].flags)
    assert {t.status for t in record.rois[1].tiles} <= {"excluded", "empty"}


# -- spec test 4: per-run reset ---------------------------------------------
def test_each_apply_replaces_the_record() -> None:
    operation = frozen_op(on_qc_fail="skip")
    quietly(operation, Image(arr=render_frame()))
    assert operation.calibration_record.verdict == "corrected"

    quietly(operation, Image(arr=render_frame(gain=1.6)))
    assert operation.calibration_record.verdict == "skipped"


def test_there_is_no_record_before_apply() -> None:
    assert frozen_op().calibration_record is None
