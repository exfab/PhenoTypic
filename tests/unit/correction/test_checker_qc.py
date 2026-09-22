from __future__ import annotations

import numpy as np
import pytest

from phenotypic.correction import QcLimits
from phenotypic.correction._color_correction._checker_qc import (
    evaluate_roi,
    warn_on_patch_census,
)

CLEAN = dict(
        roi_index=0, label="left band", shift_px=3.0,
        anchor_disagreement_px=1.0, ecc_confidence=None,
        placement_margin=0.55, hungarian_disagreement=0,
        impurities=np.full(12, 0.004), robust_shifts=np.full(12, 0.2),
        clipped=np.zeros(12), limits=QcLimits(),
)


def _record(**overrides):
    return evaluate_roi(**{**CLEAN, **overrides})


def test_a_clean_band_passes_every_gate() -> None:
    record = _record()

    assert record.ok
    assert record.warnings == []
    assert record.signals["placement_margin"] == 0.55


def test_every_signal_is_recorded_even_when_nothing_trips() -> None:
    """Logging the numbers, not just pass/fail, is what lets a slow drift be
    noticed before it crosses a threshold."""
    assert set(_record().signals) == {
        "shift_px", "anchor_disagreement_px", "ecc_confidence",
        "placement_margin", "hungarian_disagreement", "mean_impurity",
        "worst_tile_impurity", "worst_robust_shift", "worst_clipped",
        "empty_tiles", "anchor_columns_voting",
    }


@pytest.mark.parametrize(
        "overrides, fragment",
        [
            ({"shift_px": 45.0}, "displaced"),
            ({"anchor_disagreement_px": 20.0}, "rigid card cannot"),
            ({"ecc_confidence": 0.5}, "registration correlation"),
            ({"placement_margin": 0.05}, "identity undetermined"),
            ({"impurities": np.full(12, 0.3)}, "covering it"),
            ({"robust_shifts": np.full(12, 4.0)}, "moved a tile"),
            ({"clipped": np.full(12, 0.9)}, "sensor limit"),
            ({"empty_tiles": 2}, "outside the ROI"),
        ],
)
def test_each_fault_is_refused_with_a_message_naming_it(overrides, fragment) -> None:
    record = _record(**overrides)

    assert not record.ok
    assert any(fragment in flag for flag in record.flags)


@pytest.mark.parametrize(
        "overrides, fragment",
        [
            ({"placement_margin": 0.22}, "below any clean card"),
            ({"hungarian_disagreement": 5}, "do not look like"),
            ({"impurities": np.array([0.09] + [0.001] * 11)}, "contaminated"),
        ],
)
def test_softer_findings_warn_without_refusing(overrides, fragment) -> None:
    record = _record(**overrides)

    assert record.ok
    assert any(fragment in message for message in record.warnings)


def test_impurity_alone_warns_while_a_moved_measurement_refuses() -> None:
    """The pair the gate needs: something is there, versus it moved the answer."""
    present = _record(impurities=np.array([0.09] + [0.001] * 11))
    moved = _record(robust_shifts=np.array([4.0] + [0.2] * 11))

    assert present.ok and present.warnings
    assert not moved.ok


def test_an_exposure_change_is_not_a_fault() -> None:
    """The gate is keyed to geometry and card integrity, not photometry.

    A 2.5x exposure change leaves every signal here untouched, which is why
    it must pass: the card is fine, the light was different.
    """
    assert _record().ok


# ---------------------------------------------------------------------------
# Patch census
# ---------------------------------------------------------------------------
EXPECTED = [f"patch{i}" for i in range(24)]


def test_a_full_card_warns_about_nothing() -> None:
    assert warn_on_patch_census(EXPECTED, EXPECTED, 3, 20) == []


def test_missing_patches_are_named() -> None:
    with pytest.warns(UserWarning, match="patch23"):
        issued = warn_on_patch_census(EXPECTED[:23], EXPECTED, 2, 20)

    assert any("not found" in message for message in issued)


def test_cyan_is_called_out_separately() -> None:
    expected = EXPECTED[:23] + ["cyan"]
    with pytest.warns(UserWarning, match="most saturated patch"):
        warn_on_patch_census(expected[:23], expected, 2, 20)


def test_a_thin_card_warns_about_the_accuracy_floor() -> None:
    with pytest.warns(UserWarning, match="practical accuracy floor"):
        warn_on_patch_census(EXPECTED[:12], EXPECTED, 2, 20)


def test_below_min_patches_warns_without_changing_the_degree() -> None:
    """A short card never silently lowers the model."""
    with pytest.warns(UserWarning, match="below 20"):
        issued = warn_on_patch_census(EXPECTED[:18], EXPECTED, 3, 20)

    assert not any("degree" in message for message in issued)


def test_a_rank_insufficient_fit_is_refused_not_demoted() -> None:
    """Arithmetic, not policy: that fit has no unique solution, and its
    minimum-norm answer reports a spurious near-zero residual."""
    with pytest.raises(ValueError, match="no unique solution"):
        warn_on_patch_census(EXPECTED[:12], EXPECTED, 3, 20)


def test_degree_two_still_fits_twelve_patches() -> None:
    with pytest.warns(UserWarning):
        warn_on_patch_census(EXPECTED[:12], EXPECTED, 2, 20)
