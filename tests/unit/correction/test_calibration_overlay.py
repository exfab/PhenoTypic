"""The calibration tile overlay: its record (spec §1) and its figure (§2)."""

from __future__ import annotations

import warnings

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
    RoiDraft,
    RoiOverlay,
    TileOverlay,
    build_overlay_record,
    render_calibration_overlay,
    render_delta_e_bars,
)
from phenotypic.correction._color_correction._checker_measure import TileMeasurement
from phenotypic.correction._color_correction._checker_roi import CheckerLattice

from ._checker_frames import (
    NAMES,
    SRGB,
    TILE,
    TOP,
    _band_patch,
    band_prior,
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


@pytest.mark.parametrize("run", ["planted", "expect_tiles"])
def test_record_boxes_are_the_lattice_boxes(run) -> None:
    if run == "planted":
        operation = calibrated(planted_faults(), on_qc_fail="warn")
    else:
        operation = frozen_op(degree=1, on_qc_fail="warn")
        operation.rois[1].expect_tiles = 7
        quietly(operation, Image(arr=render_frame()))
    record = operation.calibration_record

    for roi, dumped in zip(record.rois, operation.diagnostics["lattices"], strict=True):
        lattice = CheckerLattice.model_validate(dumped)
        full = [b[2:] for b in lattice.boxes(rot=lattice.rot)]
        core = [b[2:] for b in lattice.boxes(core=operation.core_trim, rot=lattice.rot)]
        if roi.tiles:
            by_rc = {(b[0], b[1]): b[2:] for b in lattice.boxes(rot=lattice.rot)}
            by_rc_core = {(b[0], b[1]): b[2:]
                          for b in lattice.boxes(core=operation.core_trim, rot=lattice.rot)}
            assert len(roi.tiles) == len(full)
            for t in roi.tiles:
                assert t.full_box == pytest.approx(by_rc[(t.row, t.col)])
                assert t.core_box == pytest.approx(by_rc_core[(t.row, t.col)])
        else:
            assert [pair[0] for pair in roi.unidentified_boxes] == pytest.approx(full)
            assert [pair[1] for pair in roi.unidentified_boxes] == pytest.approx(core)
    assert any(not roi.tiles for roi in record.rois) == (run == "expect_tiles")


# -- build_overlay_record on plain inputs -----------------------------------
def measurement(row: int, col: int, *, n_pixels: int = 100,
                impurity: float = 0.0) -> TileMeasurement:
    return TileMeasurement(
            row=row, col=col, roi_index=0, n_pixels=n_pixels, lab=(50.0, 0.0, 0.0),
            srgb=(0.2 + 0.1 * row, 0.3, 0.2 + 0.1 * col), spread_delta_e=0.5,
            medoid_rank=0, medoid_widened=False, impurity=impurity,
            robust_shift=0.0, clipped=0.0,
    )


#: Patches the winning ROI fitted, with made-up ΔE00 that a test can trace.
SCORES = {name: {"deltaE00_before": 10.0 + i, "deltaE00_after": 1.0 + i}
          for i, name in enumerate(NAMES[:4])}
IMPURITY_LIMIT = 0.2


def synthetic_record(*, fitted: bool, rejected: set[str], lattice=None,
                     core_trim: float = 0.4) -> CalibrationOverlayRecord:
    """ROI 0 claims patches 0-3; ROI 1 lost a collision over patches 0 and 1.

    ROI 0: patch 0 clean, patch 1 whatever *rejected* says, patch 2 occluded
    past the impurity limit, patch 3 an empty box. ROI 1: the same names as
    ROI 0's patches 0 and 1, which it did not claim.
    """
    lattice = lattice or band_prior(nrows=2)
    winner = RoiDraft(
            roi_index=0, label=None, crop=np.zeros((10, 10, 3), np.uint8),
            lattice=lattice, claimed=True,
            tiles=[(measurement(0, 0), NAMES[0]), (measurement(1, 0), NAMES[1]),
                   (measurement(0, 1, impurity=0.5), NAMES[2]),
                   (measurement(1, 1, n_pixels=0), NAMES[3])],
    )
    loser = RoiDraft(
            roi_index=1, label=None, crop=np.zeros((10, 10, 3), np.uint8),
            lattice=lattice, claimed=False,
            tiles=[(measurement(0, 0), NAMES[0]), (measurement(1, 0), NAMES[1])],
    )
    return build_overlay_record(
            image_name="synthetic", verdict="corrected" if fitted else "refused",
            refusal=None if fitted else "refused", degree=1, n_expected=24,
            n_fitted=3 if fitted else None, drafts=[winner, loser], qc=[],
            reference_srgb={name: (0.5, 0.5, 0.5) for name in NAMES},
            fitted_patches=SCORES if fitted else None, rejected=rejected,
            impurity_limit=IMPURITY_LIMIT, core_trim=core_trim,
    )


def test_an_unclaimed_tile_whose_patch_was_rejected_is_excluded() -> None:
    # The winner's rejection of patch 1 is not the losing ROI's own.
    record = synthetic_record(fitted=True, rejected={NAMES[1]})
    assert tile(record, 0, 1, 0).status == "rejected"
    assert tile(record, 1, 1, 0).status == "excluded"
    assert tile(record, 1, 0, 0).status == "excluded"


@pytest.mark.parametrize("fitted", [True, False])
def test_a_claimed_rejected_tile_is_rejected(fitted) -> None:
    # Also without an accepted fit: a post-rejection rank refusal.
    record = synthetic_record(fitted=fitted, rejected={NAMES[1]})
    assert tile(record, 0, 1, 0).status == "rejected"
    assert tile(record, 0, 0, 0).status == ("used" if fitted else "excluded")
    assert tile(record, 0, 1, 1).status == "empty"


def test_delta_e_is_attached_only_to_used_partly_covered_and_rejected() -> None:
    record = synthetic_record(fitted=True, rejected={NAMES[1]})
    statuses = {}
    for roi in record.rois:
        for t in roi.tiles:
            statuses[(roi.roi_index, t.row, t.col)] = t.status
            if t.status in ("used", "partly_covered", "rejected"):
                assert t.delta_e_before == SCORES[t.patch]["deltaE00_before"]
                assert t.delta_e_after == SCORES[t.patch]["deltaE00_after"]
            else:
                # ROI 1's patches 0 and 1 were scored -- for ROI 0's tiles.
                assert t.delta_e_before is None and t.delta_e_after is None, t
    assert statuses == {
        (0, 0, 0): "used", (0, 1, 0): "rejected", (0, 0, 1): "partly_covered",
        (0, 1, 1): "empty", (1, 0, 0): "excluded", (1, 1, 0): "excluded",
    }
    empty = tile(record, 0, 1, 1)
    assert empty.measured_srgb is None and empty.impurity is None


def test_no_fit_attaches_no_delta_e() -> None:
    record = synthetic_record(fitted=False, rejected={NAMES[1]})
    assert all(t.delta_e_before is None and t.delta_e_after is None
               for roi in record.rois for t in roi.tiles)


def test_synthetic_boxes_follow_rotation_and_core_trim() -> None:
    lattice = band_prior(nrows=2).model_copy(update={"rot": 0.05})
    record = synthetic_record(fitted=True, rejected=set(), lattice=lattice,
                              core_trim=0.3)
    full = {(b[0], b[1]): b[2:] for b in lattice.boxes(rot=0.05)}
    core = {(b[0], b[1]): b[2:] for b in lattice.boxes(core=0.3, rot=0.05)}
    assert full[(0, 0)] != pytest.approx(lattice.boxes()[0][2:])  # rotation moves it
    for roi in record.rois:
        for t in roi.tiles:
            assert t.full_box == pytest.approx(full[(t.row, t.col)])
            assert t.core_box == pytest.approx(core[(t.row, t.col)])


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


def test_the_record_crop_is_read_only() -> None:
    # The record is frozen; its pixels must be too, or a caller can rewrite
    # what the figure says was shot.
    crop = calibrated(planted_faults(), on_qc_fail="warn").calibration_record.rois[0].crop
    with pytest.raises(ValueError, match="read-only"):
        crop[0, 0] = 0


# -- spec test 3: refused and skipped frames keep a record ------------------
def test_a_gate_refusal_keeps_a_record() -> None:
    operation = frozen_op()                                   # on_qc_fail="raise"
    with pytest.raises(RuntimeError, match="quality gate failed"):
        quietly(operation, Image(arr=render_frame(gain=1.6)))  # saturated card

    record = operation.calibration_record
    assert record.verdict == "refused" and "quality gate failed" in record.refusal
    assert record.n_fitted is None
    assert all(roi.tiles for roi in record.rois)
    assert {t.status for roi in record.rois for t in roi.tiles} <= {"excluded", "empty"}
    assert all(t.delta_e_after is None for roi in record.rois for t in roi.tiles)


def test_a_skipped_frame_keeps_a_record() -> None:
    operation = calibrated(render_frame(gain=1.6), on_qc_fail="skip")
    assert operation.calibration_record.verdict == "skipped"


def test_a_failed_correction_leaves_no_corrected_record(monkeypatch) -> None:
    from phenotypic.correction import ColorCorrector

    def broken(self, image, inplace=False):
        raise RuntimeError("planted correction failure")

    monkeypatch.setattr(ColorCorrector, "apply", broken)
    operation = frozen_op(on_qc_fail="warn")
    with pytest.raises(RuntimeError, match="planted correction failure"):
        quietly(operation, Image(arr=planted_faults()))

    record = operation.calibration_record
    assert record.verdict == "refused"
    assert record.refusal.startswith("correction failed: ")
    assert "planted correction failure" in record.refusal


@pytest.mark.parametrize(("policy", "verdict"), [("skip", "skipped"), ("warn", "refused")])
def test_warnings_as_errors_still_leave_a_record(policy, verdict) -> None:
    # The gate's warning is the first thing that raises under -W error; the
    # frame the user most wants to see must not lose its record to it.
    operation = frozen_op(on_qc_fail=policy)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(RuntimeError, match="quality gate failed"):
            operation.apply(Image(arr=render_frame(gain=1.6)))

    record = operation.calibration_record
    assert record is not None and record.verdict == verdict
    if verdict == "refused":
        assert "quality gate failed" in record.refusal
    assert all(roi.tiles for roi in record.rois)


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
    record = operation.calibration_record
    assert record.verdict == "refused"
    assert "remain after outlier rejection" in record.refusal
    # Which tiles were rejected is what explains this refusal, so it is kept;
    # no fit was accepted, so there is no ΔE anywhere.
    green = {(1, row, 1) for row in (1, 2, 3)}
    for roi in record.rois:
        assert roi.tiles
        for t in roi.tiles:
            expected = "rejected" if (roi.roi_index, t.row, t.col) in green else "excluded"
            assert t.status == expected, (roi.roi_index, t.row, t.col)
            assert t.delta_e_before is None and t.delta_e_after is None


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
    assert record.rois[1].tiles
    assert {t.status for t in record.rois[1].tiles} <= {"excluded", "empty"}
    # Its patch names were scored -- for ROI 0's tiles, not these.
    assert all(t.delta_e_before is None and t.delta_e_after is None
               for t in record.rois[1].tiles)


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


def image_axes_of(fig) -> list:
    return [ax for ax in fig.axes if ax.images]


def side_axes_of(fig, image_ax) -> list:
    """The label columns, which share y with their ROI's image axes."""
    return [ax for ax in fig.axes if ax is not image_ax and not ax.images
            and image_ax.get_shared_y_axes().joined(image_ax, ax)]


def assert_core_boxes_sit_on_their_tiles(fig, record) -> None:
    """Each tile has exactly one core box, at its core_box, in its status style."""
    for ax, roi in zip(image_axes_of(fig), record.rois, strict=True):
        cores = [p for p in ax.patches
                 if isinstance(p, Rectangle) and p.get_linewidth() == 1.8]
        for t in roi.tiles:
            cy0, cy1, cx0, cx1 = t.core_box
            at_tile = [p for p in cores
                       if np.allclose(p.get_xy(), (cx0 - 0.5, cy0 - 0.5))
                       and np.allclose((p.get_width(), p.get_height()),
                                       (cx1 - cx0, cy1 - cy0))]
            assert len(at_tile) == 1, f"ROI {roi.roi_index} {t.patch}: {len(at_tile)} boxes"
            (box,) = at_tile
            assert tuple(box.get_edgecolor()) == to_rgba(STATUS_COLOURS[t.status]), t.patch
            dashed = t.status in ("rejected", "excluded", "empty")
            assert box.get_linestyle() == ("--" if dashed else "-"), t.patch


def assert_labels_sit_at_their_tiles(fig, record) -> None:
    """Each side label, ΔE line and swatch pair lies within its tile's rows.

    Pairs a name to its tile by text, so it needs unique names per ROI.
    """
    for image_ax, roi in zip(image_axes_of(fig), record.rois, strict=True):
        if not roi.tiles or roi.n_tile_columns > 2:
            continue
        sides = side_axes_of(fig, image_ax)
        texts = [t for ax in sides for t in ax.texts]
        swatches = [p for ax in sides for p in ax.patches if isinstance(p, Rectangle)]
        assert len({overlay._name_line(t) for t in roi.tiles}) == len(roi.tiles)
        for t in roi.tiles:
            # The shared y axis is the image's: pixel rows, box edges at -0.5.
            lo, hi = t.full_box[0] - 0.5, t.full_box[1] - 0.5
            (name,) = [x for x in texts if x.get_text() == overlay._name_line(t)]
            name_x, name_y = name.get_position()
            assert lo <= name_y <= hi, f"{t.patch}: name at {name_y}, tile {lo}-{hi}"
            delta_e = overlay._delta_e_line(t)[0]
            assert any(x.get_text() == delta_e and x.axes is name.axes
                       and x.get_position()[0] == name_x
                       and name_y < x.get_position()[1] <= hi for x in texts), t.patch
            for rgb in (t.measured_srgb, t.reference_srgb):
                if rgb is None:
                    continue
                centres = [p.get_y() + p.get_height() / 2 for p in swatches
                           if p.axes is name.axes
                           and np.allclose(p.get_facecolor()[:3], rgb)]
                assert any(lo <= c <= hi for c in centres), f"{t.patch}: swatch {rgb}"


def assert_labels_clear_the_images(fig) -> None:
    """No text outside an image axes lands on any ROI's image.

    Text drawn *on* an image axes -- the key mode's tile numbers -- belongs there.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    images = image_axes_of(fig)
    image_boxes = [ax.get_window_extent(renderer) for ax in images]
    for text in fig.findobj(Text):
        if not (text.get_visible() and text.get_text().strip()) or text.axes in images:
            continue
        box = text.get_window_extent(renderer)
        for image_box in image_boxes:
            dx = min(box.x1, image_box.x1) - max(box.x0, image_box.x0)
            dy = min(box.y1, image_box.y1) - max(box.y0, image_box.y0)
            assert dx <= 0.5 or dy <= 0.5, f"{text.get_text()!r} is drawn over an image"


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


def edge_record() -> CalibrationOverlayRecord:
    """Tiles flush with the crop's top and bottom edges, short against their labels.

    A wide crop with 6 px tiles: each label block is taller than the tile it
    names, so the first and last blocks overhang the image. A long ROI label
    makes the ROI title span the label columns, so an overhang it did not
    reserve room for collides with it.
    """
    rows, pitch, tile_h, width = 6, 30, 6, 160
    crop = np.full(((rows - 1) * pitch + tile_h, width, 3), 30, dtype=np.uint8)
    tiles = []
    for c, x0 in enumerate((20, width - 60)):
        for r in range(rows):
            name = NAMES[c * rows + r]
            y0 = r * pitch
            reference = tuple(float(v) for v in SRGB[name])
            tiles.append(TileOverlay(
                    row=r, col=c, patch=name, status="used",
                    full_box=(y0, y0 + tile_h, x0, x0 + 40),
                    core_box=(y0 + 1, y0 + tile_h - 1, x0 + 8, x0 + 32),
                    measured_srgb=tuple(0.9 * v for v in reference),
                    reference_srgb=reference, impurity=0.0,
                    delta_e_before=4.0, delta_e_after=1.0,
            ))
    roi = RoiOverlay(roi_index=0, label="left card band, flush with the top of its rectangle",
                     crop=crop, lattice_found=True, n_tile_columns=2, flags=[],
                     warnings=[], tiles=tiles)
    return CalibrationOverlayRecord(image_name="edge", verdict="corrected", degree=3,
                                    n_fitted=12, n_expected=24, refusal=None, rois=[roi])


def status_record() -> CalibrationOverlayRecord:
    """One ROI with a used, an excluded and an empty tile (no measured colour)."""
    crop = np.full((3 * 60 + 24, 2 * 60 + 24, 3), 30, dtype=np.uint8)
    spec = [(0, 0, "used", (0.4, 0.3, 0.2)), (1, 0, "excluded", (0.3, 0.4, 0.2)),
            (2, 0, "empty", None), (0, 1, "used", (0.2, 0.3, 0.4))]
    tiles = []
    for i, (r, c, status, measured) in enumerate(spec):
        y0, x0 = 12 + r * 60, 12 + c * 60
        name = NAMES[i]
        fitted = status == "used"
        tiles.append(TileOverlay(
                row=r, col=c, patch=name, status=status,
                full_box=(y0, y0 + 48, x0, x0 + 48),
                core_box=(y0 + 10, y0 + 38, x0 + 10, x0 + 38),
                measured_srgb=measured,
                reference_srgb=tuple(float(v) for v in SRGB[name]),
                impurity=None if measured is None else 0.0,
                delta_e_before=3.0 if fitted else None,
                delta_e_after=1.0 if fitted else None,
        ))
    roi = RoiOverlay(roi_index=0, label=None, crop=crop, lattice_found=True,
                     n_tile_columns=2, flags=[], warnings=[], tiles=tiles)
    return CalibrationOverlayRecord(image_name="statuses", verdict="corrected", degree=1,
                                    n_fitted=2, n_expected=24, refusal=None, rois=[roi])


# -- spec test 5: no overlapping text, no clipping ---------------------------
def test_two_band_figure_has_no_overlap() -> None:
    record = calibrated(planted_faults(), on_qc_fail="warn").calibration_record
    fig = render_calibration_overlay(record)
    assert_no_overlap_or_clipping(fig)
    assert_labels_clear_the_images(fig)


def test_two_band_labels_sit_at_their_tiles() -> None:
    record = calibrated(planted_faults(), on_qc_fail="warn").calibration_record
    assert_labels_sit_at_their_tiles(render_calibration_overlay(record), record)


def test_tiles_flush_with_the_crop_edge_have_no_overlap() -> None:
    record = edge_record()
    fig = render_calibration_overlay(record)
    assert_no_overlap_or_clipping(fig)
    assert_labels_clear_the_images(fig)
    assert_labels_sit_at_their_tiles(fig, record)
    assert_core_boxes_sit_on_their_tiles(fig, record)


def test_empty_and_excluded_tiles_render_their_suffixes_and_a_hatched_swatch() -> None:
    record = status_record()
    fig = render_calibration_overlay(record)
    assert_no_overlap_or_clipping(fig)
    assert_labels_sit_at_their_tiles(fig, record)
    assert_core_boxes_sit_on_their_tiles(fig, record)
    texts = {t for t, _ in text_boxes(fig)}
    assert f"{NAMES[1]} · excluded" in texts and f"{NAMES[2]} · empty" in texts
    assert texts >= {"ΔE00 not fitted"}
    hatched = [p for ax in fig.axes for p in ax.patches
               if isinstance(p, Rectangle) and p.get_hatch()]
    assert len(hatched) == 1 and hatched[0].get_hatch() == "////"
    (empty,) = [t for t in record.rois[0].tiles if t.status == "empty"]
    y_mid = hatched[0].get_y() + hatched[0].get_height() / 2
    assert empty.full_box[0] - 0.5 <= y_mid <= empty.full_box[1] - 0.5


def test_longest_names_have_no_overlap() -> None:
    fig = render_calibration_overlay(card_record(6, 2))
    assert_no_overlap_or_clipping(fig)
    assert_labels_clear_the_images(fig)


def test_a_full_card_uses_a_numbered_key_without_overlap() -> None:
    fig = render_calibration_overlay(card_record(4, 6))
    assert_no_overlap_or_clipping(fig)
    assert_labels_clear_the_images(fig)
    numbers = {t for t, _ in text_boxes(fig) if t.isdigit()}
    assert {str(n) for n in range(1, 25)} <= numbers


@pytest.mark.parametrize("dpi", [50, 300])
@pytest.mark.parametrize("shape", [(6, 2), (4, 6)])
def test_labels_measured_at_one_dpi_still_fit_at_another(shape, dpi) -> None:
    # savefig(dpi=...) redraws at a dpi the labels were not measured at.
    fig = render_calibration_overlay(card_record(*shape))
    fig.set_dpi(dpi)
    assert_no_overlap_or_clipping(fig)


def test_a_degenerate_core_box_in_key_mode_does_not_divide_by_zero() -> None:
    # tile_px=16 trims each core box to zero height and width.
    record = card_record(4, 6, tile_px=16)
    assert min(t.core_box[1] - t.core_box[0] for t in record.rois[0].tiles) == 0
    assert render_calibration_overlay(record) is not None


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


@pytest.mark.parametrize("record", [edge_record, lambda: card_record(4, 6)],
                         ids=["edge", "key"])
def test_halved_label_heights_are_caught(monkeypatch, record) -> None:
    real = overlay._TextMeter.size

    def half_height(self, text, **kwargs):
        width, height = real(self, text, **kwargs)
        return width, height / 2

    monkeypatch.setattr(overlay._TextMeter, "size", half_height)
    fig = render_calibration_overlay(record())
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
    assert_core_boxes_sit_on_their_tiles(fig, record)


def expect_tiles_refused() -> CalibrationOverlayRecord:
    """ROI 1 declares 7 tiles; its 12 are found, then refused before identity."""
    operation = frozen_op(degree=1, on_qc_fail="warn")
    operation.rois[1].expect_tiles = 7
    quietly(operation, Image(arr=render_frame()))
    return operation.calibration_record


def is_unidentified_core(p) -> bool:
    return (isinstance(p, Rectangle) and p.get_linewidth() == 1.8
            and p.get_linestyle() == "--"
            and tuple(p.get_edgecolor()) == to_rgba(STATUS_COLOURS["excluded"]))


def test_a_roi_refused_after_its_lattice_keeps_its_boxes() -> None:
    record = expect_tiles_refused()
    roi = record.rois[1]
    assert any("declared to hold 7 tiles" in flag for flag in roi.flags)
    assert roi.lattice_found and not roi.tiles
    assert len(roi.unidentified_boxes) == 12
    assert record.rois[0].unidentified_boxes == []

    fig = render_calibration_overlay(record)
    image_axes = [ax for ax in fig.axes if ax.images]
    cores = [p for p in image_axes[1].patches if is_unidentified_core(p)]
    assert len(cores) == len(roi.unidentified_boxes)
    for (full, core), p in zip(roi.unidentified_boxes, cores):
        cy0, cy1, cx0, cx1 = core
        assert p.get_xy() == pytest.approx((cx0 - 0.5, cy0 - 0.5))
        assert (p.get_width(), p.get_height()) == pytest.approx((cx1 - cx0, cy1 - cy0))
    # No identity, so no labels: ROI 1's side columns carry no text at all.
    side = [ax for ax in fig.axes if not ax.images
            and ax.get_shared_y_axes().joined(ax, image_axes[1])]
    assert side and not any(ax.texts for ax in side)
    assert_no_overlap_or_clipping(fig)


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


def test_a_16_bit_frame_draws_its_pixels_not_white() -> None:
    """imshow reads integers as 0-255, so the crop is scaled by its bit depth.

    Without it a 16-bit plate photograph draws as a white field with speckle.
    The record keeps the as-shot uint16 pixels; only the drawing is scaled.
    """
    frame8 = render_frame()
    assert frame8.dtype == np.uint8
    frame16 = frame8.astype(np.uint16) * 257  # same picture, 16-bit
    shown = {}
    for bits, frame in ((8, frame8), (16, frame16)):
        operation = frozen_op(degree=1, on_qc_fail="warn")
        quietly(operation, Image(arr=frame))
        record = operation.calibration_record
        assert record.rois[0].crop.dtype == frame.dtype
        fig = render_calibration_overlay(record)
        shown[bits] = [np.asarray(ax.images[0].get_array()) for ax in fig.axes if ax.images]
    for a8, a16 in zip(shown[8], shown[16]):
        assert a16.max() <= 1.0
        np.testing.assert_allclose(a16, a8, atol=1e-6)


# -- the ΔE00 bar chart -------------------------------------------------------
def bar_series(fig) -> dict[str, list]:
    """The chart's bar containers by label, each a list of Rectangles."""
    (ax,) = fig.axes
    return {c.get_label(): list(c) for c in ax.containers}


def test_bars_are_the_records_delta_e_one_pair_per_scored_tile() -> None:
    record = calibrated(planted_faults(), on_qc_fail="warn").calibration_record
    scored = [
        t for roi in record.rois for t in sorted(roi.tiles, key=lambda t: (t.col, t.row))
        if t.delta_e_after is not None
    ]
    assert len(scored) == 24  # 21 fitted + 3 rejected neutrals
    series = bar_series(render_delta_e_bars(record))
    assert list(series) == ["before", "after"]
    assert [b.get_height() for b in series["before"]] == [t.delta_e_before for t in scored]
    assert [b.get_height() for b in series["after"]] == [t.delta_e_after for t in scored]
    (ax,) = render_delta_e_bars(record).axes
    ticks = [label.get_text() for label in ax.get_xticklabels()]
    assert ticks == [t.patch + (" (rejected)" if t.status == "rejected" else "")
                     for t in scored]


def test_rejected_tiles_are_hatched_and_left_out_of_the_mean() -> None:
    record = calibrated(planted_faults(), on_qc_fail="warn").calibration_record
    fig = render_delta_e_bars(record)
    series = bar_series(fig)
    scored = [
        t for roi in record.rois for t in sorted(roi.tiles, key=lambda t: (t.col, t.row))
        if t.delta_e_after is not None
    ]
    for label in ("before", "after"):
        hatched = [bool(b.get_hatch()) for b in series[label]]
        assert hatched == [t.status == "rejected" for t in scored]
    fitted = [t for t in scored if t.status != "rejected"]
    mean_after = np.mean([t.delta_e_after for t in fitted])
    assert f"-> {mean_after:.2f} over {len(fitted)} fitted patches" in fig._suptitle.get_text()


@pytest.mark.parametrize("policy", ["raise", "skip"])
def test_a_frame_with_no_fit_draws_no_bars_and_says_why(policy) -> None:
    operation = frozen_op(on_qc_fail=policy)
    frame = Image(arr=render_frame(gain=1.6))  # saturated card
    if policy == "raise":
        with pytest.raises(RuntimeError, match="quality gate failed"):
            quietly(operation, frame)
    else:
        quietly(operation, frame)
    assert operation.calibration_record.verdict == {"raise": "refused", "skip": "skipped"}[policy]
    fig = operation.show_delta_bar_plot()
    (ax,) = fig.axes
    assert ax.containers == []
    texts = [t.get_text() for t in ax.texts]
    assert any("not fitted" in t for t in texts)
    if policy == "raise":
        assert any("quality gate failed" in t for t in texts)


def test_the_bar_chart_labels_do_not_overlap_or_clip() -> None:
    """No text leaves the figure; no two upright texts collide.

    The rotated tick labels are excluded from the overlap half only: their
    axis-aligned extents overlap by construction while the glyphs do not.
    The planted frame's rejected neutrals stretch the y-range to ~40, which
    squeezes the good and fair lines to a few pixels apart.
    """
    fig = render_delta_e_bars(calibrated(planted_faults(), on_qc_fail="warn").calibration_record)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = fig.bbox
    upright = []
    for t in fig.findobj(Text):
        if not (t.get_visible() and t.get_text().strip()):
            continue
        box = t.get_window_extent(renderer)
        assert box.x0 >= frame.x0 - 0.5 and box.x1 <= frame.x1 + 0.5, f"clipped: {t.get_text()!r}"
        assert box.y0 >= frame.y0 - 0.5 and box.y1 <= frame.y1 + 0.5, f"clipped: {t.get_text()!r}"
        if t.get_rotation() == 0:
            upright.append((t.get_text(), box))
    assert {"good <= 2", "fair <= 5"} <= {text for text, _ in upright}
    for i, (a_text, a) in enumerate(upright):
        for b_text, b in upright[i + 1:]:
            dx = min(a.x1, b.x1) - max(a.x0, b.x0)
            dy = min(a.y1, b.y1) - max(a.y0, b.y0)
            assert dx <= 0.5 or dy <= 0.5, f"{a_text!r} overlaps {b_text!r}"


def test_a_16_bit_frame_charts_the_same_delta_e_as_its_8_bit_twin() -> None:
    frame8 = render_frame()
    heights = {}
    for bits, frame in ((8, frame8), (16, frame8.astype(np.uint16) * 257)):
        fig = render_delta_e_bars(calibrated(frame).calibration_record)
        series = bar_series(fig)
        heights[bits] = {k: [b.get_height() for b in v] for k, v in series.items()}
        # Nothing rejected on a clean frame: no hatch, and no rejected key entry.
        assert not any(b.get_hatch() for bars in series.values() for b in bars)
        (ax,) = fig.axes
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["before", "after"]
    # uint8 * 257 is exact in [0, 1]: the same normed pixels, the same fit.
    for label in ("before", "after"):
        np.testing.assert_allclose(heights[16][label], heights[8][label], atol=1e-9)


def test_show_delta_bar_plot_needs_an_apply() -> None:
    with pytest.raises(RuntimeError, match="call apply"):
        frozen_op().show_delta_bar_plot()
