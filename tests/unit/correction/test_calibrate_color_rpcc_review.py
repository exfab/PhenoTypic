"""Regression guards for the 2026-09-21 code review of ``CalibrateColorRpcc``.

One test per finding in
``docs/superpowers/reports/2026-09-21-in-frame-checker-color-correction/code-review.md``,
numbered to match.  Each was written to fail against the reviewed code, before
any fix, so that it is known to be able to fail.

The frames are synthetic and rig-shaped: two vertical bands, each holding a
6x2 block of real ColorChecker24 colours (a transposed half-card), with the
lattice supplied as a prior -- the production path.  ``fit_lattice`` cannot
bootstrap on this fixture (a low-contrast column falls below its column
threshold), which is unrelated to any finding and is why no test here fits a
lattice from scratch unless that is the thing under test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.correction import CalibrateColorRpcc, CheckerRoi
from phenotypic.correction._color_correction._checker_qc import QcLimits
from phenotypic.correction._color_correction._checker_roi import (
    CheckerLattice,
    ColumnLattice,
)

CHECKER = "ColorChecker24 - After November 2014"
PITCH, TILE, TOP = 60, 48, 80
BAND_H = 2 * TOP + 6 * PITCH
GAP = 200
#: Keeps every patch off the uint8 floor, so the clipped-pixel gate stays quiet
#: on a clean frame (black and cyan otherwise have a channel at 0).
FLOOR, GAIN = 0.04, 0.85


def _patch_srgb() -> tuple[list[str], dict[str, np.ndarray]]:
    import colour

    from phenotypic.correction._color_correction._color_checker_profile import (
        _load_reference_data,
    )

    ref_lab, ref_linear, _ = _load_reference_data(CHECKER, "D65")
    names = list(ref_lab)
    return names, {
        name: colour.cctf_encoding(np.clip(ref_linear[name], 0, 1), function="sRGB")
        for name in names
    }


NAMES, SRGB = _patch_srgb()


def _band_patch(band: int, row: int, col: int) -> str:
    """Band *band* holds chart rows ``2*band`` and ``2*band+1``, transposed."""
    return NAMES[(2 * band + col) * 6 + row]


def render_frame(
        *,
        band_w: int = 140,
        col_x: tuple[int, int] = (25, 85),
        dx: int = 0,
        dy: int = 0,
        gain: float = GAIN,
        overrides: dict[tuple[int, int, int], np.ndarray] | None = None,
        seed: int = 0,
) -> np.ndarray:
    """A uint8 sRGB frame with two card bands at the left and right edges.

    ``overrides`` maps ``(band, row, col)`` to a replacement sRGB colour.
    """
    width = 2 * band_w + GAP
    rng = np.random.default_rng(seed)
    frame = np.full((BAND_H, width, 3), 0.55)
    for band, left in ((0, 0), (1, width - band_w)):
        region = np.full((BAND_H, band_w, 3), 0.12)
        for row in range(6):
            for col in range(2):
                colour = (overrides or {}).get(
                        (band, row, col), SRGB[_band_patch(band, row, col)]
                )
                y0 = TOP + dy + row * PITCH
                x0 = col_x[col] + dx
                ys, xs = max(0, y0), max(0, x0)
                region[ys:y0 + TILE, xs:x0 + TILE] = FLOOR + colour * gain
        frame[:, left:left + band_w] = region
    frame = np.clip(frame + rng.normal(0, 0.004, frame.shape), 0, 1)
    return (frame * 255).round().astype(np.uint8)


def band_rois(band_w: int = 140) -> list[list[int]]:
    width = 2 * band_w + GAP
    return [[0, 0, BAND_H, band_w], [0, width - band_w, BAND_H, width]]


def band_prior(
        col_x: tuple[int, int] = (25, 85), nrows: int = 6
) -> CheckerLattice:
    return CheckerLattice(
            columns=[
                ColumnLattice(
                        x0=x, x1=x + TILE, start=float(TOP), pitch=float(PITCH),
                        duty=TILE / PITCH,
                )
                for x in col_x
            ],
            nrows=nrows,
    )


def frozen_op(**kwargs) -> CalibrateColorRpcc:
    prior = band_prior()
    kwargs.setdefault("rois", band_rois())
    kwargs.setdefault("lattice_prior", [prior, prior])
    kwargs.setdefault("refine_method", "frozen")
    return CalibrateColorRpcc(**kwargs)


def quietly(operation: CalibrateColorRpcc, image: Image) -> Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return operation.apply(image)


def test_the_fixture_calibrates_cleanly() -> None:
    """Control: every other test here perturbs this frame, so it must pass."""
    operation = frozen_op()

    quietly(operation, Image(arr=render_frame()))

    assert all(record.ok for record in operation.qc)
    assert len(operation.diagnostics["patch_census"]["accepted"]) == 24
    assert np.isfinite(operation.fitted_profile.correction_matrix).all()


# ---------------------------------------------------------------------------
# 1. A tile that measured nothing must not reach identity scoring or the fit
# ---------------------------------------------------------------------------
def test_an_empty_tile_never_turns_the_fit_into_nan() -> None:
    """A prior whose last row falls outside a shorter ROI yields empty tiles.

    Their NaN colour used to pass every gate (``NaN < x`` is False) and poison
    the placement margin, the outlier threshold and the matrix.
    """
    rois = band_rois()
    rois[0][2] = TOP + 5 * PITCH + 5  # ROI 0 now ends inside row 5's box
    operation = frozen_op(rois=rois, on_qc_fail="warn")

    quietly(operation, Image(arr=render_frame()))

    assert np.isfinite(operation.qc[0].signals["placement_margin"])
    assert np.isfinite(operation.fitted_profile.correction_matrix).all()
    for name in (_band_patch(0, 5, 0), _band_patch(0, 5, 1)):
        assert name not in operation.diagnostics["patch_census"]["accepted"]
    assert operation.qc[0].signals["empty_tiles"] == 2
    assert any("outside the ROI" in flag for flag in operation.qc[0].flags)


# ---------------------------------------------------------------------------
# 3. Rank sufficiency must be checked on the patches the fit actually uses
# ---------------------------------------------------------------------------
def test_rank_is_checked_after_outlier_rejection() -> None:
    """Degree 4 needs 22 patches.  24 measured minus 3 outliers is 21.

    Three neutrals painted saturated green: the only override set found that
    the ``mean + 2 sd`` rule rejects in full on this fixture, because the
    neutrals' baseline delta-E is small and uniform.
    """
    green = np.array([0.1, 0.95, 0.1])
    wrong = {(1, row, 1): green for row in (1, 2, 3)}
    operation = frozen_op(degree=4, on_qc_fail="warn")

    with pytest.raises(RuntimeError, match="only 21 remain after outlier rejection"):
        quietly(operation, Image(arr=render_frame(overrides=wrong)))

    assert operation.fitted_profile is None


def test_census_accepted_excludes_outlier_rejected_patches() -> None:
    """``accepted`` is what reached the fit, not what was measured.

    The same three green neutrals as above, at degree 3, where 21 patches
    still clear the 13-term rank line and the fit runs.
    """
    green = np.array([0.1, 0.95, 0.1])
    wrong = {(1, row, 1): green for row in (1, 2, 3)}
    operation = frozen_op(degree=3, on_qc_fail="warn")

    quietly(operation, Image(arr=render_frame(overrides=wrong)))

    fitted = operation.fitted_profile.diagnostics
    assert fitted["n_patches_rejected"] >= 3, (
        "fixture precondition: the swapped tiles must be rejected as outliers"
    )
    accepted = operation.diagnostics["patch_census"]["accepted"]
    for row in (1, 2, 3):
        assert _band_patch(1, row, 1) not in accepted
    assert len(accepted) == (
        fitted["n_patches_detected"] - fitted["n_patches_rejected"]
    )


def test_a_frame_with_every_roi_refused_names_the_refusals_not_the_rank() -> None:
    """With nothing measured, the error is the refusals, not a degree advice."""
    rng = np.random.default_rng(0)
    flat = rng.normal(120, 2, (BAND_H, 480, 3)).clip(0, 255).astype(np.uint8)
    operation = CalibrateColorRpcc(
            rois=band_rois(), grid=(6, 2), on_qc_fail="warn",
    )

    with pytest.raises(RuntimeError, match="No ROI produced usable tiles"):
        quietly(operation, Image(arr=flat))


# ---------------------------------------------------------------------------
# 4. patch_census must survive a frame that cannot be calibrated
# ---------------------------------------------------------------------------
def test_patch_census_records_a_failing_frame_instead_of_crashing() -> None:
    rng = np.random.default_rng(0)
    flat = rng.normal(120, 2, (BAND_H, 480, 3)).clip(0, 255).astype(np.uint8)
    blank = Image(arr=flat, name="blank")

    census = CalibrateColorRpcc.patch_census(
            [blank], rois=band_rois(), grid=(6, 2)
    )

    assert census["per_image"] == {"blank": 0}


# ---------------------------------------------------------------------------
# 5. (and 2) ECC is not offered, so there is no reference band to lose
# ---------------------------------------------------------------------------
def test_ecc_is_not_offered_by_the_operation() -> None:
    """Dropped 2026-09-21: a reference band would ride in every pipeline JSON.

    Finding 5 (bands lost on rebuild) and the operation-level half of
    finding 2 (rotation ignored) both disappear with it; the rotation bug in
    ``refine_ecc``/``boxes`` is guarded in ``test_checker_detect.py``.
    """
    assert "reference_bands" not in CalibrateColorRpcc.model_fields
    prior = band_prior()
    with pytest.raises(ValueError):
        CalibrateColorRpcc(
                rois=band_rois(), lattice_prior=[prior, prior], refine_method="ecc",
        )


# ---------------------------------------------------------------------------
# 6. A border-clipped column must not vote in the anchor-disagreement check
# ---------------------------------------------------------------------------
def test_a_clipped_column_does_not_refuse_an_in_range_shift() -> None:
    """Column 0 sits flush with the frame edge; the card moves 25 px left.

    25 px is inside ``max_shift_px`` (30).  The clipped column sees only
    about half the shift, and letting it vote made the columns look like
    they disagreed by more than 12 px.
    """
    col_x = (0, 100)
    prior = band_prior(col_x=col_x)
    operation = CalibrateColorRpcc(
            rois=[[0, 0, BAND_H, 180]], lattice_prior=[prior],
            refine_method="rigid", degree=2, on_qc_fail="warn",
    )

    quietly(operation, Image(arr=render_frame(band_w=180, col_x=col_x, dx=-25)))

    signals = operation.qc[0].signals
    assert signals["shift_px"] == pytest.approx(25.0, abs=3.0)
    assert signals["anchor_columns_voting"] == 1
    assert any("could not run" in w for w in operation.qc[0].warnings)
    assert not any("rigid card cannot" in flag for flag in operation.qc[0].flags), (
        f"refused at disagreement {signals['anchor_disagreement_px']:.1f} px"
    )


# ---------------------------------------------------------------------------
# 7. Detection failures must go through the on_qc_fail policy
# ---------------------------------------------------------------------------
def test_an_expect_tiles_mismatch_is_skipped_under_skip() -> None:
    rois = [CheckerRoi.from_bbox(r, expect_tiles=10) for r in band_rois()]
    operation = frozen_op(rois=rois, on_qc_fail="skip")
    source = Image(arr=render_frame())

    out = quietly(operation, source)

    np.testing.assert_array_equal(out.rgb[:], source.rgb[:])
    assert not operation.qc[0].ok
    assert operation.fitted_profile is None


def test_a_patch_collision_is_skipped_under_skip() -> None:
    """A partly occluded card whose surviving row resembles another ROI's.

    ROI 1's row 0 is painted in ROI 0's row-0 colours and the rest is flat
    grey, so its best placement is a guess that lands on ROI 0's patches.
    The collision used to raise past the policy; it is a flag on ROI 1.
    """
    grey = np.array([0.3, 0.3, 0.3])
    overrides = {(1, 0, c): SRGB[_band_patch(0, 0, c)] for c in (0, 1)}
    overrides.update({(1, r, c): grey for r in range(1, 6) for c in (0, 1)})
    operation = frozen_op(on_qc_fail="skip")
    source = Image(arr=render_frame(overrides=overrides))

    out = quietly(operation, source)

    np.testing.assert_array_equal(out.rgb[:], source.rgb[:])
    assert not operation.qc[1].ok
    assert any("ROI 0 already claimed" in flag for flag in operation.qc[1].flags)


def test_an_roi_whose_every_box_misses_is_skipped_under_skip() -> None:
    """A prior entirely outside its ROI measures nothing at all.

    Without its own refusal, identity scoring raises ``No tiles are allowed
    to vote``, a ``ValueError`` that would bypass the policy.
    """
    operation = frozen_op(
            lattice_prior=[band_prior(col_x=(500, 560)), band_prior()],
            on_qc_fail="skip",
    )
    source = Image(arr=render_frame())

    out = quietly(operation, source)

    np.testing.assert_array_equal(out.rgb[:], source.rgb[:])
    assert operation.qc[0].flags == ["every tile box falls outside the ROI"]


def test_an_roi_with_no_card_is_skipped_under_skip() -> None:
    rng = np.random.default_rng(0)
    flat = rng.normal(120, 2, (BAND_H, 480, 3)).clip(0, 255).astype(np.uint8)
    source = Image(arr=flat)
    operation = CalibrateColorRpcc(
            rois=band_rois(), grid=(6, 2), on_qc_fail="skip",
    )

    out = quietly(operation, source)

    np.testing.assert_array_equal(out.rgb[:], source.rgb[:])
    assert operation.qc and not all(record.ok for record in operation.qc)


# ---------------------------------------------------------------------------
# 8. A skipped frame must not report the previous frame's profile
# ---------------------------------------------------------------------------
def test_a_skipped_frame_clears_the_previous_profile() -> None:
    operation = frozen_op(on_qc_fail="skip")
    quietly(operation, Image(arr=render_frame()))
    assert operation.fitted_profile is not None  # precondition

    quietly(operation, Image(arr=render_frame(gain=1.6)))  # saturated card

    assert not all(record.ok for record in operation.qc)  # precondition
    assert operation.fitted_profile is None
    assert operation.diagnostics["patch_census"]["accepted"] == []


# ---------------------------------------------------------------------------
# 9. A user's qc_limits.min_patches must not be silently overwritten
# ---------------------------------------------------------------------------
def test_min_patches_has_one_home() -> None:
    """The spec puts ``min_patches`` on the operation; a second copy on
    ``QcLimits`` was silently overwritten by it.  One knob, so nothing can be
    ignored."""
    with pytest.raises(ValueError):
        QcLimits(min_patches=10)


def test_the_operation_min_patches_is_honoured() -> None:
    """18 patches, ``min_patches=10``: no below-minimum warning."""
    rois = band_rois()
    rois[1][2] = TOP + 3 * PITCH + 5  # ROI 1 keeps three rows
    operation = CalibrateColorRpcc(
            rois=rois,
            lattice_prior=[band_prior(), band_prior(nrows=3)],
            refine_method="frozen",
            min_patches=10,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        operation.apply(Image(arr=render_frame()))

    assert len(operation.diagnostics["patch_census"]["accepted"]) == 18
    messages = [str(w.message) for w in caught]
    assert not any("noticeably worse fit" in m for m in messages), messages


# ---------------------------------------------------------------------------
# 10. Per-ROI lists must match the ROI count at construction
# ---------------------------------------------------------------------------
def test_a_short_lattice_prior_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="lattice_prior"):
        CalibrateColorRpcc(rois=band_rois(), lattice_prior=[band_prior()])


# ---------------------------------------------------------------------------
# 11. ROI Lab must be the image's own Lab, under the image's illuminant
# ---------------------------------------------------------------------------
def test_roi_lab_uses_the_image_illuminant() -> None:
    image = Image(arr=render_frame(), illuminant="D50")
    roi = CheckerRoi.from_bbox(band_rois()[0])

    lab, _srgb = frozen_op()._roi_views(image, roi)

    np.testing.assert_allclose(
            lab, image.color.Lab[roi.row_slice, roi.col_slice], atol=1e-9
    )
