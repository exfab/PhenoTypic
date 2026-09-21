"""Quality gates for in-frame colour-checker calibration.

An unattended batch must never emit a correction fitted to a card that was
occluded, mis-identified, or clipped, so every signal here either refuses a
frame or warns about it -- and the two are kept distinct.  The impurity
statistics answer *is something there*; the robust shift answers *and did it
move the answer*, and only the second is grounds to reject.

Nothing here can change the model.  A short card warns; it never silently
lowers the polynomial degree, because correcting two frames of one experiment
with different models puts a reproducible bias into the contrast between them.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from ._checker_identity import (
    MAX_HUNGARIAN_DISAGREEMENT,
    MIN_PLACEMENT_MARGIN,
    WARN_PLACEMENT_MARGIN,
)

#: The patch whose loss costs more than any other. Removing it raises
#: card-wide error from 1.96 to 3.08 delta-E 2000 at degree 3, against under
#: 0.44 for every other patch, because it is the most saturated patch on the
#: card and sits outside the sRGB gamut -- dropping it turns interpolation
#: into extrapolation at the edge of the fit's range.
LOAD_BEARING_PATCHES: tuple[str, ...] = ("cyan",)

#: Accepted patches below which accuracy is past its practical floor for
#: either degree, whatever the rank says.
ACCURACY_FLOOR_PATCHES = 13


class QcLimits(BaseModel):
    """Thresholds for the per-ROI quality gate.

    Defaults are the values calibrated on the reference band set; see the
    design spec for where each came from.

    Args:
        max_shift_px: Displacement from the prior beyond which the prior is
            stale or the rig has moved.
        max_anchor_disagreement_px: Spread between the horizontal shifts two
            different anchor columns imply.  A reference-free internal
            consistency check: the columns of one rigid card must agree.
        min_ecc: ECC correlation floor, checked only for ``refine="ecc"``.
        min_placement_margin: Identity margin below which the card is refused.
        warn_placement_margin: Margin below which it is used but warned about.
        max_hungarian_disagreement: Tiles a free assignment may label
            differently from the winning placement before warning.
        max_mean_impurity: Mean per-tile contaminated fraction.
        max_tile_impurity: Worst single tile's contaminated fraction (warns).
        max_robust_shift: delta-E by which contamination may move a tile's
            measured colour.
        max_clipped: Fraction of a tile's pixels that may sit at the sensor
            floor or ceiling.
        min_patches: Accepted patches below which the frame is flagged.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_shift_px: float = 30.0
    max_anchor_disagreement_px: float = 12.0
    min_ecc: float = 0.90
    min_placement_margin: float = MIN_PLACEMENT_MARGIN
    warn_placement_margin: float = WARN_PLACEMENT_MARGIN
    max_hungarian_disagreement: int = MAX_HUNGARIAN_DISAGREEMENT
    max_mean_impurity: float = 0.05
    max_tile_impurity: float = 0.05
    max_robust_shift: float = 1.5
    max_clipped: float = 0.20
    min_patches: int = 20


class QcRecord(BaseModel):
    """What the gate measured on one ROI, and what it concluded.

    Attributes:
        roi_index: Which ROI this describes.
        label: The ROI's label, when it has one.
        flags: Reasons the ROI is not trustworthy.  Non-empty means refuse.
        warnings: Observations worth surfacing that do not refuse.
        signals: Every measured value, whether or not it tripped anything --
            logging these rather than a pass/fail is what lets a slow drift
            be noticed before it crosses a threshold.
    """

    model_config = ConfigDict(extra="forbid")

    roi_index: int
    label: str | None = None
    flags: list[str] = []
    warnings: list[str] = []
    signals: dict[str, Any] = {}

    @property
    def ok(self) -> bool:
        """Whether the ROI cleared every gate."""
        return not self.flags


def evaluate_roi(
        roi_index: int,
        label: str | None,
        shift_px: float,
        anchor_disagreement_px: float | None,
        ecc_confidence: float | None,
        placement_margin: float,
        hungarian_disagreement: int,
        impurities: np.ndarray,
        robust_shifts: np.ndarray,
        clipped: np.ndarray,
        limits: QcLimits,
) -> QcRecord:
    """Score one ROI against *limits*.

    Args:
        roi_index: Index of the ROI.
        label: Its label, for messages.
        shift_px: Displacement magnitude from the prior.
        anchor_disagreement_px: Spread between anchor-column estimates, or
            ``None`` when only one column was available.
        ecc_confidence: ECC correlation, or ``None`` for other methods.
        placement_margin: From identity scoring.
        hungarian_disagreement: Tiles the free assignment labels differently.
        impurities: Per-tile contaminated fractions.
        robust_shifts: Per-tile colour displacement under trimming.
        clipped: Per-tile clipped fractions.
        limits: Thresholds to apply.

    Returns:
        A :class:`QcRecord`.
    """
    flags: list[str] = []
    warns: list[str] = []

    mean_impurity = float(np.nanmean(impurities)) if impurities.size else float("nan")
    worst_impurity = float(np.nanmax(impurities)) if impurities.size else float("nan")
    worst_shift = float(np.nanmax(robust_shifts)) if robust_shifts.size else float("nan")
    worst_clipped = float(np.nanmax(clipped)) if clipped.size else float("nan")

    if shift_px > limits.max_shift_px:
        flags.append(
                f"card displaced {shift_px:.1f} px from the prior "
                f"(limit {limits.max_shift_px:.0f}) - the prior is stale or the "
                "rig has moved"
        )
    if (
            anchor_disagreement_px is not None
            and anchor_disagreement_px > limits.max_anchor_disagreement_px
    ):
        flags.append(
                f"columns imply displacements {anchor_disagreement_px:.1f} px "
                "apart; a rigid card cannot do that"
        )
    if ecc_confidence is not None and ecc_confidence < limits.min_ecc:
        flags.append(
                f"registration correlation {ecc_confidence:.2f} below "
                f"{limits.min_ecc:.2f}"
        )
    if placement_margin < limits.min_placement_margin:
        flags.append(
                f"patch identity undetermined (margin {placement_margin:.3f} < "
                f"{limits.min_placement_margin:.2f}); refusing rather than "
                "guessing which patch is which"
        )
    elif placement_margin < limits.warn_placement_margin:
        warns.append(
                f"patch identity margin {placement_margin:.3f} is below any "
                f"clean card observed (>= {limits.warn_placement_margin:.2f})"
        )
    if hungarian_disagreement > limits.max_hungarian_disagreement:
        warns.append(
                f"{hungarian_disagreement} tile(s) do not look like the patch "
                "the card geometry implies"
        )
    if mean_impurity == mean_impurity and mean_impurity > limits.max_mean_impurity:
        flags.append(
                f"{mean_impurity * 100:.1f} % of the card's pixels are unlike "
                "their own tile - something is covering it"
        )
    if worst_impurity == worst_impurity and worst_impurity > limits.max_tile_impurity:
        warns.append(
                f"worst tile is {worst_impurity * 100:.1f} % contaminated; see "
                "the robust shift for whether it moved the measurement"
        )
    if worst_shift == worst_shift and worst_shift > limits.max_robust_shift:
        flags.append(
                f"contamination moved a tile's measured colour by "
                f"{worst_shift:.1f} delta-E (limit {limits.max_robust_shift:.1f})"
        )
    if worst_clipped == worst_clipped and worst_clipped > limits.max_clipped:
        flags.append(
                f"{worst_clipped * 100:.0f} % of a tile's pixels are pinned at "
                "the sensor limit; that colour was never recorded"
        )

    return QcRecord(
            roi_index=roi_index,
            label=label,
            flags=flags,
            warnings=warns,
            signals={
                "shift_px"               : float(shift_px),
                "anchor_disagreement_px" : anchor_disagreement_px,
                "ecc_confidence"         : ecc_confidence,
                "placement_margin"       : float(placement_margin),
                "hungarian_disagreement" : int(hungarian_disagreement),
                "mean_impurity"          : mean_impurity,
                "worst_tile_impurity"    : worst_impurity,
                "worst_robust_shift"     : worst_shift,
                "worst_clipped"          : worst_clipped,
            },
    )


def warn_on_patch_census(
        accepted: list[str],
        expected: list[str],
        degree: int,
        limits: QcLimits,
) -> list[str]:
    """Warn about missing patches without ever changing the model.

    Args:
        accepted: Patch names that survived detection and outlier rejection.
        expected: Every patch the chart has.
        degree: The configured polynomial degree.
        limits: Supplies ``min_patches``.

    Returns:
        The warning messages issued, so they can be recorded in diagnostics.

    Raises:
        ValueError: If *degree* needs more patches than were accepted. That
            fit has no unique solution, and its minimum-norm answer reports a
            near-zero in-sample residual that is pure interpolation artifact.
    """
    from ._checker_identity import CHART_SHAPES  # noqa: F401  (documented link)

    terms = {1: 3, 2: 6, 3: 13, 4: 22}.get(degree)
    issued: list[str] = []
    missing = [name for name in expected if name not in accepted]

    if terms is not None and len(accepted) < terms:
        raise ValueError(
                f"A degree-{degree} root-polynomial fit needs at least {terms} "
                f"patches but only {len(accepted)} were accepted. The fit would "
                "have no unique solution, and its minimum-norm answer reports a "
                "near-zero in-sample residual that is an artifact of exact "
                "interpolation rather than accuracy. Re-shoot the card, or "
                "configure a lower degree for the whole batch."
        )

    if missing:
        issued.append(
                f"{len(missing)} of {len(expected)} chart patches were not found: "
                f"{', '.join(missing)}."
        )
    for name in LOAD_BEARING_PATCHES:
        if name in expected and name not in accepted:
            issued.append(
                    f"The {name!r} patch is missing. It is the only single patch "
                    "whose loss costs more than 0.5 delta-E 2000 (about +1.1 at "
                    "degree 3), because it is the most saturated patch on the "
                    "card and sits outside the sRGB gamut."
            )
    if len(accepted) < ACCURACY_FLOOR_PATCHES:
        issued.append(
                f"Only {len(accepted)} patches were accepted, below the "
                f"{ACCURACY_FLOOR_PATCHES}-patch practical accuracy floor for "
                "either degree - rank sufficiency is not accuracy."
        )
    elif len(accepted) < limits.min_patches:
        issued.append(
                f"Only {len(accepted)} patches were accepted (below "
                f"{limits.min_patches}); expect a noticeably worse fit."
        )

    for message in issued:
        warnings.warn(message, UserWarning, stacklevel=3)
    return issued
