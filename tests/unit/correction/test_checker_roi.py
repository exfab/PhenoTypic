from __future__ import annotations

import pytest

from phenotypic.correction import CheckerRoi
from phenotypic.correction._color_correction._checker_roi import (
    CheckerLattice,
    ColumnLattice,
)


# ---------------------------------------------------------------------------
# Bounding-box shorthand
# ---------------------------------------------------------------------------
def test_bbox_shorthand_matches_the_explicit_form() -> None:
    """``[row_min, col_min, row_max, col_max]`` is regionprops' bbox order."""
    assert CheckerRoi.from_bbox([1170, 5856, 2840, 6016]) == CheckerRoi(
            row=(1170, 2840), col=(5856, 6016)
    )


def test_bbox_shorthand_accepts_a_regionprops_bbox_unchanged() -> None:
    """A tuple straight off ``regionprops`` needs no re-ordering."""
    bbox = (1170, 0, 2840, 340)  # (min_row, min_col, max_row, max_col)
    assert CheckerRoi.from_bbox(bbox) == CheckerRoi(row=(1170, 2840), col=(0, 340))


def test_bbox_shorthand_carries_optional_fields_through() -> None:
    roi = CheckerRoi.from_bbox([1170, 0, 2840, 340], label="left band", expect_tiles=12)
    assert (roi.label, roi.expect_tiles) == ("left band", 12)


@pytest.mark.parametrize(
        "bad, expected_message",
        [
            ([1170, 0, 2840], "four"),
            ([1170, 0, 2840, 340, 7], "four"),
            ([1170.5, 0, 2840, 340], "integer"),
            # (x, y, width, height) — width 2840 lands in the row_max slot, so
            # row_max(340) < row_min(1170).
            ([1170, 0, 340, 2840], "row_min, col_min, row_max, col_max"),
            # (row, col, height, width) with a zero-height box.
            ([1170, 5856, 1170, 6016], "row_min, col_min, row_max, col_max"),
        ],
)
def test_bbox_shorthand_rejects_the_wrong_convention(bad, expected_message) -> None:
    """The message must name the order; a wrong-convention box otherwise crops
    somewhere plausible and only fails much later as a detection problem."""
    with pytest.raises(ValueError, match=expected_message):
        CheckerRoi.from_bbox(bad)


# ---------------------------------------------------------------------------
# CheckerRoi itself
# ---------------------------------------------------------------------------
def test_roi_rejects_an_inverted_rectangle() -> None:
    with pytest.raises(ValueError):
        CheckerRoi(row=(2840, 1170), col=(0, 340))


def test_roi_rejects_negative_bounds() -> None:
    with pytest.raises(ValueError):
        CheckerRoi(row=(-1, 2840), col=(0, 340))


def test_roi_forbids_unknown_fields() -> None:
    """Strict construction: a typo'd kwarg is an error, not a silent no-op."""
    with pytest.raises(ValueError):
        CheckerRoi(row=(0, 10), col=(0, 10), layout=(6, 2))


def test_roi_declares_nothing_about_its_contents() -> None:
    """Identity and tile count are derived from the image, never supplied.

    Guards the design decision directly: if a future edit adds a field naming
    chart patches or a layout, this fails and the spec must be revisited.
    """
    assert set(CheckerRoi.model_fields) == {
        "row", "col", "label", "expect_tiles", "anchor_col",
    }


def test_roi_slices_the_array_it_describes() -> None:
    import numpy as np

    arr = np.arange(100 * 50).reshape(100, 50)
    roi = CheckerRoi(row=(10, 40), col=(5, 25))
    assert arr[roi.row_slice, roi.col_slice].shape == (30, 20) == roi.shape


# ---------------------------------------------------------------------------
# Lattice geometry
# ---------------------------------------------------------------------------
def _lattice() -> CheckerLattice:
    return CheckerLattice(
            columns=[ColumnLattice(x0=0, x1=76, start=150.0, pitch=255.5, duty=0.76),
                     ColumnLattice(x0=143, x1=203, start=150.0, pitch=255.5, duty=0.76)],
            nrows=6,
    )


def test_lattice_yields_one_box_per_tile_in_row_major_order() -> None:
    boxes = _lattice().boxes()
    assert len(boxes) == 12
    assert [(row, col) for row, col, *_ in boxes[:3]] == [(0, 0), (1, 0), (2, 0)]


def test_lattice_translation_shifts_every_box_equally() -> None:
    plain = _lattice().boxes()
    moved = _lattice().boxes(dy=7.5, dx=-3.0)
    for (_, _, y0, y1, x0, x1), (_, _, my0, my1, mx0, mx1) in zip(plain, moved):
        assert (my0 - y0, my1 - y1, mx0 - x0, mx1 - x1) == (7.5, 7.5, -3.0, -3.0)


def test_core_trim_keeps_the_central_fraction_and_the_same_centre() -> None:
    """``core=0.4`` keeps the middle 60 % of each axis — 36 % of the area."""
    full = _lattice().boxes()[0]
    core = _lattice().boxes(core=0.4)[0]
    _, _, y0, y1, x0, x1 = full
    _, _, cy0, cy1, cx0, cx1 = core
    assert (cy1 - cy0) == pytest.approx(0.6 * (y1 - y0))
    assert (cx1 - cx0) == pytest.approx(0.6 * (x1 - x0))
    assert (cy0 + cy1) / 2 == pytest.approx((y0 + y1) / 2)
    assert (cx0 + cx1) / 2 == pytest.approx((x0 + x1) / 2)


def test_lattice_centres_match_its_boxes() -> None:
    lat = _lattice()
    for (_, _, y0, y1, x0, x1), (cy, cx) in zip(lat.boxes(), lat.centers()):
        assert (cy, cx) == pytest.approx(((y0 + y1) / 2, (x0 + x1) / 2))


def test_lattice_round_trips_through_serialisation() -> None:
    lat = _lattice()
    assert CheckerLattice.model_validate(lat.model_dump()) == lat


def test_rotation_turns_plus_x_toward_plus_y_like_opencv() -> None:
    """A tile right of the centroid moves down under a positive rotation.

    That is OpenCV's sense (image y points down); ECC's recovered angle is
    applied through this, so the two must agree.
    """
    lattice = CheckerLattice(
            columns=[
                ColumnLattice(x0=0, x1=10, start=0.0, pitch=10.0, duty=1.0),
                ColumnLattice(x0=90, x1=100, start=0.0, pitch=10.0, duty=1.0),
            ],
            nrows=1,
    )
    right = [b for b in lattice.boxes(rot=0.1) if b[1] == 1][0]

    assert (right[2] + right[3]) / 2 > 5.0 + 1.0


# ---------------------------------------------------------------------------
# Coercion on the operation
# ---------------------------------------------------------------------------
def test_operation_field_coerces_and_stores_canonical() -> None:
    from phenotypic.correction import CalibrateColorRpcc

    op = CalibrateColorRpcc(
            rois=[[1170, 0, 2840, 340],
                  CheckerRoi(row=(1170, 2840), col=(5856, 6016))],
    )
    assert all(isinstance(roi, CheckerRoi) for roi in op.rois)
    assert op.model_dump()["rois"][0] == {
        "row"  : (1170, 2840), "col": (0, 340), "label": None,
        "expect_tiles": None, "anchor_col": None,
    }
