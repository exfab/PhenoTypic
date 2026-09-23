"""The calibration tile overlay: its record (spec §1) and its figure (§2)."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from phenotypic import Image
from phenotypic.correction import CalibrateColorRpcc
from phenotypic.correction._color_correction import _calibration_overlay as overlay
from phenotypic.correction._color_correction._calibration_overlay import (
    STATUS_COLOURS,
    CalibrationOverlayRecord,
    RoiOverlay,
    TileOverlay,
    render_calibration_overlay,
)

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


def text_boxes(fig):
    """Window extents of every visible, non-empty text, after an Agg draw."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [
        (t.get_text(), t.get_window_extent(renderer))
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip()
    ]


def assert_no_overlap_or_clipping(fig) -> None:
    """No two texts intersect by more than 0.5 px, and none leaves the figure.

    0.5 px absorbs Agg's sub-pixel rounding of glyph extents; any real
    collision between two label lines is several pixels.
    """
    boxes = text_boxes(fig)
    frame = fig.bbox
    for text, box in boxes:
        assert box.x0 >= frame.x0 - 0.5 and box.x1 <= frame.x1 + 0.5, f"clipped: {text!r}"
        assert box.y0 >= frame.y0 - 0.5 and box.y1 <= frame.y1 + 0.5, f"clipped: {text!r}"
    for i, (a_text, a) in enumerate(boxes):
        for b_text, b in boxes[i + 1:]:
            dx = min(a.x1, b.x1) - max(a.x0, b.x0)
            dy = min(a.y1, b.y1) - max(a.y0, b.y0)
            assert dx <= 0.5 or dy <= 0.5, f"{a_text!r} overlaps {b_text!r}"


def card_record(rows: int, cols: int, *, name: str = "neutral 6.5 (.44 D)",
                status: str = "rejected", tile_px: int = 40) -> CalibrationOverlayRecord:
    """A synthetic record: one ROI holding a rows x cols card of long names."""
    pitch = tile_px + 12
    crop = np.full((rows * pitch + 24, cols * pitch + 24, 3), 30, dtype=np.uint8)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y0, x0 = 12 + r * pitch, 12 + c * pitch
            box = (y0, y0 + tile_px, x0, x0 + tile_px)
            core = (y0 + 8, y0 + tile_px - 8, x0 + 8, x0 + tile_px - 8)
            tiles.append(TileOverlay(
                    row=r, col=c, patch=name, status=status, full_box=box, core_box=core,
                    measured_srgb=(0.4, 0.4, 0.4), reference_srgb=(0.5, 0.5, 0.5),
                    impurity=0.0, delta_e_before=31.4, delta_e_after=34.4,
            ))
    roi = RoiOverlay(roi_index=0, label=None, crop=crop, lattice_found=True,
                     n_tile_columns=cols, flags=[], warnings=[], tiles=tiles)
    return CalibrationOverlayRecord(image_name="card", verdict="corrected", degree=3,
                                    n_fitted=24, n_expected=24, refusal=None, rois=[roi])


# -- spec test 5: no overlapping text, no clipping ---------------------------
def test_two_band_figure_has_no_overlap() -> None:
    record = calibrated(planted_faults(), on_qc_fail="warn").calibration_record
    assert_no_overlap_or_clipping(render_calibration_overlay(record))


def test_longest_names_have_no_overlap() -> None:
    assert_no_overlap_or_clipping(render_calibration_overlay(card_record(6, 2)))


def test_a_full_card_uses_a_numbered_key_without_overlap() -> None:
    fig = render_calibration_overlay(card_record(4, 6))
    assert_no_overlap_or_clipping(fig)
    numbers = {t for t, _ in text_boxes(fig) if t.isdigit()}
    assert {str(n) for n in range(1, 25)} <= numbers


def test_a_refused_frame_renders_its_flags() -> None:
    operation = frozen_op()
    with pytest.raises(RuntimeError):
        quietly(operation, Image(arr=render_frame(gain=1.6)))
    fig = render_calibration_overlay(operation.calibration_record)
    assert_no_overlap_or_clipping(fig)
    texts = " ".join(t for t, _ in text_boxes(fig))
    assert "sensor limit" in texts and "not fitted" in texts


# -- spec test 6: the overlap test can fail --------------------------------
def test_halved_label_widths_are_caught(monkeypatch) -> None:
    real = overlay._TextMeter.size

    def half_width(self, text, **kwargs):
        width, height = real(self, text, **kwargs)
        return width / 2, height

    monkeypatch.setattr(overlay._TextMeter, "size", half_width)
    fig = render_calibration_overlay(card_record(6, 2))
    with pytest.raises(AssertionError):
        assert_no_overlap_or_clipping(fig)


def test_a_figsize_too_small_is_refused() -> None:
    with pytest.raises(ValueError, match="too small"):
        render_calibration_overlay(card_record(6, 2), figsize=(2.0, 2.0))


# -- spec test 7: structure ------------------------------------------------
def test_one_image_axes_per_roi_and_one_core_box_per_tile() -> None:
    record = calibrated(planted_faults(), on_qc_fail="warn").calibration_record
    fig = render_calibration_overlay(record)
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == len(record.rois)
    for ax, roi in zip(image_axes, record.rois):
        cores = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_linewidth() == 1.8]
        assert len(cores) == len(roi.tiles)
        assert sorted(tuple(p.get_edgecolor()) for p in cores) == sorted(
                to_rgba(STATUS_COLOURS[t.status]) for t in roi.tiles)


def test_a_roi_without_a_lattice_has_an_image_and_no_boxes() -> None:
    rng = np.random.default_rng(0)
    blank = rng.normal(120, 2, render_frame().shape).clip(0, 255).astype(np.uint8)
    operation = CalibrateColorRpcc(rois=band_rois(), grid=(6, 2), on_qc_fail="warn")
    with pytest.raises(RuntimeError):
        quietly(operation, Image(arr=blank))
    fig = render_calibration_overlay(operation.calibration_record)
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 2 and all(not ax.patches for ax in image_axes)
    assert_no_overlap_or_clipping(fig)


# -- show_tiles() -------------------------------------------------------------
def test_show_tiles_renders_the_last_apply() -> None:
    operation = calibrated(planted_faults(), on_qc_fail="warn")
    fig = operation.show_tiles()
    assert len([ax for ax in fig.axes if ax.images]) == 2
    assert_no_overlap_or_clipping(fig)


def test_show_tiles_before_apply_raises() -> None:
    with pytest.raises(RuntimeError, match="call apply\\(\\) first"):
        frozen_op().show_tiles()


def test_show_tiles_works_after_a_refusal() -> None:
    operation = frozen_op()
    with pytest.raises(RuntimeError):
        quietly(operation, Image(arr=render_frame(gain=1.6)))
    assert operation.show_tiles() is not None
