from __future__ import annotations

import numpy as np
import pytest

from phenotypic.correction._color_correction._checker_detect import (
    column_signal,
    fit_lattice,
    fit_phase,
    plateaus,
    refine,
    refine_rigid,
    row_signal,
)
from phenotypic.correction._color_correction._checker_roi import (
    CheckerLattice,
    ColumnLattice,
)

# A synthetic band shaped like the real rig: two columns of six patches each,
# gutters between them, uniform card background around them.
PITCH = 60.0
DUTY = 0.8
NROWS = 6
COLUMNS = ((10, 40), (60, 90))
HEIGHT, WIDTH = 560, 110


def synthetic_band(dy: float = 0.0, dx: float = 0.0, noise: float = 0.4) -> np.ndarray:
    """A Lab band with six patches down each of two columns."""
    rng = np.random.default_rng(0)
    band = np.full((HEIGHT, WIDTH, 3), [60.0, 0.0, 0.0])
    colours = [
        [35.0, 20.0, 18.0], [70.0, -25.0, 40.0], [45.0, 50.0, -30.0],
        [80.0, -5.0, 60.0], [30.0, 10.0, -45.0], [55.0, -40.0, 5.0],
    ]
    # Margin above and below the tile block is ~20 % of its span, matching the
    # real rig's band (6 tiles at 255.5 px pitch inside a 2150 px band). Too
    # tight a margin makes a large shift push the first tile out of the
    # analysed window, which limits capture range for the fixture rather than
    # for the method.
    start = 80.0
    for col_index, (x0, x1) in enumerate(COLUMNS):
        for row in range(NROWS):
            y0 = start + dy + row * PITCH
            y1 = y0 + DUTY * PITCH
            ys, ye = int(round(y0)), int(round(y1))
            xs, xe = int(round(x0 + dx)), int(round(x1 + dx))
            ys, ye = max(0, ys), min(HEIGHT, ye)
            xs, xe = max(0, xs), min(WIDTH, xe)
            if ye <= ys or xe <= xs:
                continue
            colour = colours[(row + 3 * col_index) % len(colours)]
            band[ys:ye, xs:xe] = colour
    return band + rng.normal(0, noise, band.shape)


def hand_written_lattice() -> CheckerLattice:
    """The lattice as constructed, before any estimator has seen it."""
    return CheckerLattice(
            columns=[
                ColumnLattice(x0=x0, x1=x1, start=80.0, pitch=PITCH, duty=DUTY)
                for x0, x1 in COLUMNS
            ],
            nrows=NROWS,
    )


def reference_lattice() -> CheckerLattice:
    """The prior as production builds it: fitted from an undisplaced frame.

    Fitting both the prior and the frame with the same estimator is what makes
    a displacement measurement unbiased. A hand-written prior carries a few
    pixels of constant offset against the fitted edge, because smoothing moves
    the apparent rising edge of every patch the same way.
    """
    return fit_lattice(synthetic_band(), grid=(NROWS, 2))


# ---------------------------------------------------------------------------
# Signals and primitives
# ---------------------------------------------------------------------------
def test_column_signal_peaks_on_the_patch_columns() -> None:
    profile = column_signal(synthetic_band())

    for x0, x1 in COLUMNS:
        assert profile[(x0 + x1) // 2] > profile[(COLUMNS[0][1] + COLUMNS[1][0]) // 2]


def test_row_signal_starts_at_the_reported_offset() -> None:
    band = synthetic_band()
    profile, offset = row_signal(band, *COLUMNS[0])

    assert offset == int(round(HEIGHT * 0.05))
    assert profile.size == HEIGHT - 2 * offset


def test_row_signal_is_empty_for_a_degenerate_column() -> None:
    profile, _ = row_signal(synthetic_band(), 50, 50)

    assert profile.size == 0


def test_plateaus_ignores_runs_below_the_minimum_length() -> None:
    signal = np.zeros(100)
    signal[10:40] = 1.0   # long enough
    signal[60:65] = 1.0   # a blur, not a patch

    assert plateaus(signal, 0.5, min_length=20) == [(10, 40)]


def test_fit_phase_returns_none_when_the_wave_cannot_fit() -> None:
    assert fit_phase(np.zeros(30), n_periods=6, pitch=60.0) is None


# ---------------------------------------------------------------------------
# Fitting a lattice with nothing declared
# ---------------------------------------------------------------------------
def test_fit_lattice_finds_the_columns_without_being_told() -> None:
    """Column detection needs nothing declared, on synthetic and real bands.

    Verified against six real half-card bands (three frames x two sides):
    two columns found on every one, at the known patch extents.
    """
    lattice = fit_lattice(synthetic_band())

    assert len(lattice.columns) == 2
    for column, (x0, x1) in zip(lattice.columns, COLUMNS):
        assert abs(column.x0 - x0) <= 3 and abs(column.x1 - x1) <= 3


def test_fit_lattice_recovers_the_pitch_when_given_the_grid() -> None:
    """With the row count known, the pitch fit is a one-parameter search.

    On the real bands this recovers 255.0-256.0 px against a rig pitch of
    255.5 on all six.
    """
    lattice = fit_lattice(synthetic_band(), grid=(NROWS, 2))

    assert lattice.nrows == NROWS
    assert lattice.n_tiles == 12
    assert lattice.columns[0].pitch == pytest.approx(PITCH, abs=1.5)


def test_fit_lattice_row_count_is_unreliable_without_a_grid() -> None:
    """A documented limitation, asserted so it cannot be assumed away.

    The span-over-pitch estimate counts the ROI's margin as tiles, so it
    over-counts. The pitch is right; the count is not. Supply ``grid`` when
    bootstrapping a prior from a real card, and check the result.
    """
    inferred = fit_lattice(synthetic_band())

    assert inferred.columns[0].pitch == pytest.approx(PITCH, abs=3.0)
    assert inferred.nrows > NROWS


def test_fit_lattice_rejects_a_grid_whose_column_count_is_wrong() -> None:
    with pytest.raises(ValueError, match="Expected 3 patch column"):
        fit_lattice(synthetic_band(), grid=(NROWS, 3))


def test_fit_lattice_places_tiles_on_the_real_patches() -> None:
    band = synthetic_band()
    lattice = fit_lattice(band)

    for _, _, y0, y1, x0, x1 in lattice.boxes(core=0.4):
        patch = band[int(y0):int(y1), int(x0):int(x1)]
        # a real patch is uniform; a gutter or a straddled edge is not
        assert patch.reshape(-1, 3).std(axis=0).max() < 2.0


def test_fit_lattice_ignores_a_blurred_ghost_column() -> None:
    """Refracted ghosts through a plate wall must not be counted as patches.

    They are too diffuse to hold a plateau above threshold, which is exactly
    what the minimum-length rule is for.
    """
    band = synthetic_band()
    from scipy.ndimage import gaussian_filter

    ghost = gaussian_filter(band[:, 60:90], sigma=(8, 8, 0)) * 0.5 + 30.0
    band = np.concatenate([ghost, band], axis=1)

    lattice = fit_lattice(band)

    assert len(lattice.columns) == 2
    for column in lattice.columns:
        assert column.x0 >= 30  # nothing found in the ghost strip


def test_fit_lattice_refuses_an_roi_with_no_card() -> None:
    rng = np.random.default_rng(1)
    empty = np.full((200, 100, 3), [60.0, 0.0, 0.0]) + rng.normal(0, 0.3, (200, 100, 3))

    with pytest.raises(ValueError, match="No patch columns"):
        fit_lattice(empty)


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dy", [-40.0, -15.0, 0.0, 15.0, 40.0])
def test_rigid_refinement_recovers_vertical_displacement(dy: float) -> None:
    result = refine_rigid(synthetic_band(dy=dy), reference_lattice())

    assert result.dy == pytest.approx(dy, abs=2.0)


@pytest.mark.parametrize("dx", [-8.0, 0.0, 8.0, 16.0])
def test_rigid_refinement_recovers_horizontal_displacement(dx: float) -> None:
    result = refine_rigid(synthetic_band(dx=dx), reference_lattice())

    assert result.dx == pytest.approx(dx, abs=3.0)


def test_rigid_refinement_moves_the_whole_lattice_as_one_body() -> None:
    """Every column shifts by the same amount -- that is what 'rigid' means,
    and why a border-clipped column cannot drag the fit off."""
    prior = reference_lattice()
    result = refine_rigid(synthetic_band(dy=12.0, dx=6.0), prior)

    offsets = {
        (new.x0 - old.x0, new.start - old.start)
        for old, new in zip(prior.columns, result.lattice.columns)
    }
    assert len(offsets) == 1


def test_frozen_refinement_returns_the_prior_untouched() -> None:
    prior = reference_lattice()
    result = refine(synthetic_band(dy=20.0), prior, method="frozen")

    assert result.lattice == prior
    assert (result.dy, result.dx, result.confidence) == (0.0, 0.0, 1.0)


def test_ecc_refinement_needs_a_reference_band() -> None:
    with pytest.raises(ValueError, match="needs reference_lab"):
        refine(synthetic_band(), reference_lattice(), method="ecc")


def test_unknown_refine_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown refine method"):
        refine(synthetic_band(), reference_lattice(), method="blob")


def test_ecc_refinement_recovers_displacement_when_opencv_is_present() -> None:
    cv2 = pytest.importorskip("cv2")
    assert cv2  # silence the unused warning

    result = refine(
            synthetic_band(dy=18.0), reference_lattice(),
            method="ecc", reference_lab=synthetic_band(),
    )

    assert result.dy == pytest.approx(18.0, abs=2.0)
    assert result.confidence > 0.9


def test_ecc_refinement_rejects_a_mismatched_reference() -> None:
    pytest.importorskip("cv2")

    with pytest.raises(ValueError, match="same shape"):
        refine(
                synthetic_band(), reference_lattice(),
                method="ecc", reference_lab=synthetic_band()[:100],
        )

def test_anchor_search_cannot_lock_onto_a_neighbouring_column() -> None:
    """Columns 50 px apart with a 45 px search would otherwise overlap.

    Unclamped, the neighbour's plateau wins and the measured shift is wrong by
    a whole column pitch -- silently, because that plateau looks healthy. The
    window is clamped to the midpoint between columns instead.
    """
    prior = reference_lattice()
    spacing = (
        (prior.columns[1].x0 + prior.columns[1].x1)
        - (prior.columns[0].x0 + prior.columns[0].x1)
    ) / 2.0

    result = refine_rigid(synthetic_band(dx=6.0), prior, search_x=45)

    assert spacing < 2 * 45, "the fixture must exercise the clamp"
    assert abs(result.dx) < spacing / 2
    assert result.dx == pytest.approx(6.0, abs=3.0)


def test_hand_written_prior_shows_the_estimator_offset() -> None:
    """Documents why the prior must come from the same estimator.

    Not a defect: it is why production fits the prior with ``fit_lattice``
    rather than writing pitch and start by hand.
    """
    fitted = refine_rigid(synthetic_band(), reference_lattice()).dy
    by_hand = refine_rigid(synthetic_band(), hand_written_lattice()).dy

    assert abs(fitted) < 1.0
    assert abs(by_hand) > 1.0
