"""What one ``CalibrateColorRpcc`` run looked like, and a figure of it.

:class:`CalibrationOverlayRecord` is plain data -- the as-shot ROI crops, every
tile's boxes, the chart patch it was matched to, its status and ΔE00 -- built
by the operation during ``apply()`` and kept even when the frame is refused.
:func:`render_calibration_overlay` draws it and reads nothing else, so a record
persisted elsewhere can be drawn anywhere.

See the spec: ``docs/superpowers/specs/2026-09-22-checker-calibration-overlay/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from ._checker_measure import TileMeasurement
    from ._checker_qc import QcRecord
    from ._checker_roi import CheckerLattice

Verdict = Literal["corrected", "corrected_with_warnings", "skipped", "refused"]
TileStatus = Literal["used", "partly_covered", "rejected", "excluded", "empty"]
Box = tuple[float, float, float, float]
Rgb = tuple[float, float, float]


class TileOverlay(BaseModel):
    """One tile: where it was, what it was matched to, how it came out.

    Attributes:
        row: Tile row within its ROI lattice.
        col: Tile column within its ROI lattice.
        patch: Chart patch the tile was identified as.
        status: See the spec's tile-status table.
        full_box: ``(y0, y1, x0, x1)`` of the whole tile, ROI-local pixels.
        core_box: ``(y0, y1, x0, x1)`` of the pixels the medoid came from.
        measured_srgb: The medoid pixel's own sRGB; ``None`` for an empty tile.
        reference_srgb: The chart's reference colour, sRGB-encoded.
        impurity: Contaminated fraction; ``None`` for an empty tile.
        delta_e_before: ΔE00 before correction; ``None`` unless the fit ran.
        delta_e_after: ΔE00 after correction; ``None`` unless the fit ran.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    row: int
    col: int
    patch: str
    status: TileStatus
    full_box: Box
    core_box: Box
    measured_srgb: Rgb | None
    reference_srgb: Rgb
    impurity: float | None
    delta_e_before: float | None
    delta_e_after: float | None


class RoiOverlay(BaseModel):
    """One ROI: its as-shot pixels, its tiles and what the gate said."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    roi_index: int
    label: str | None
    crop: np.ndarray
    lattice_found: bool
    n_tile_columns: int
    flags: list[str]
    warnings: list[str]
    tiles: list[TileOverlay]


class CalibrationOverlayRecord(BaseModel):
    """Everything :func:`render_calibration_overlay` draws, and nothing else."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_name: str | None
    verdict: Verdict
    degree: int
    n_fitted: int | None
    n_expected: int
    refusal: str | None
    rois: list[RoiOverlay]


@dataclass
class RoiDraft:
    """What the ROI loop learned about one ROI, before the frame's outcome."""

    roi_index: int
    label: str | None
    crop: np.ndarray
    lattice: CheckerLattice | None = None
    tiles: list[tuple[TileMeasurement, str]] = field(default_factory=list)
    claimed: bool = False


def _tile_status(
        tile: TileMeasurement,
        patch: str,
        *,
        claimed: bool,
        fitted: bool,
        rejected: set[str],
        impurity_limit: float,
) -> TileStatus:
    """The spec's tile-status table, in its order of precedence."""
    if not tile.n_pixels:
        return "empty"
    if not fitted or not claimed:
        return "excluded"
    if patch in rejected:
        return "rejected"
    if tile.impurity == tile.impurity and tile.impurity > impurity_limit:
        return "partly_covered"
    return "used"


def build_overlay_record(
        *,
        image_name: str | None,
        verdict: Verdict,
        refusal: str | None,
        degree: int,
        n_expected: int,
        n_fitted: int | None,
        drafts: Sequence[RoiDraft],
        qc: Sequence[QcRecord],
        reference_srgb: Mapping[str, Rgb],
        fitted_patches: Mapping[str, Mapping[str, float]] | None,
        rejected: set[str],
        impurity_limit: float,
        core_trim: float,
) -> CalibrationOverlayRecord:
    """Assemble the record from the ROI loop's drafts and the frame's outcome.

    Args:
        image_name: The image's name, for the figure title.
        verdict: See the spec's verdict table.
        refusal: The refusal message when ``verdict == "refused"``.
        degree: The configured polynomial degree.
        n_expected: Patches the chart has.
        n_fitted: Patches that reached the fit, or ``None`` with no fit.
        drafts: One per ROI, in ROI order.
        qc: The gate's records; matched to drafts by ``roi_index``.
        reference_srgb: Patch name -> reference colour, sRGB-encoded.
        fitted_patches: ``ColorCheckerProfile.diagnostics["patches"]`` when
            the fit ran and was accepted, else ``None``.
        rejected: Patches outlier rejection removed.
        impurity_limit: ``QcLimits.max_tile_impurity``.
        core_trim: The operation's ``core_trim``, for the core boxes.

    Returns:
        The frozen record.
    """
    qc_by_roi = {record.roi_index: record for record in qc}
    rois = []
    for draft in drafts:
        gate = qc_by_roi.get(draft.roi_index)
        tiles: list[TileOverlay] = []
        if draft.lattice is not None:
            lattice = draft.lattice
            full = {(r, c): (y0, y1, x0, x1)
                    for r, c, y0, y1, x0, x1 in lattice.boxes(rot=lattice.rot)}
            core = {(r, c): (y0, y1, x0, x1)
                    for r, c, y0, y1, x0, x1 in lattice.boxes(core=core_trim, rot=lattice.rot)}
            for measured, patch in draft.tiles:
                status = _tile_status(
                        measured, patch, claimed=draft.claimed,
                        fitted=fitted_patches is not None, rejected=rejected,
                        impurity_limit=impurity_limit,
                )
                scored = (
                    fitted_patches.get(patch)
                    if fitted_patches is not None
                    and status in ("used", "partly_covered", "rejected")
                    else None
                )
                key = (measured.row, measured.col)
                tiles.append(TileOverlay(
                        row=measured.row, col=measured.col, patch=patch, status=status,
                        full_box=full[key], core_box=core[key],
                        measured_srgb=measured.srgb if measured.n_pixels else None,
                        reference_srgb=reference_srgb[patch],
                        impurity=float(measured.impurity) if measured.n_pixels else None,
                        delta_e_before=None if scored is None else float(scored["deltaE00_before"]),
                        delta_e_after=None if scored is None else float(scored["deltaE00_after"]),
                ))
        rois.append(RoiOverlay(
                roi_index=draft.roi_index,
                label=draft.label,
                crop=draft.crop,
                lattice_found=draft.lattice is not None,
                n_tile_columns=0 if draft.lattice is None else len(draft.lattice.columns),
                flags=list(gate.flags) if gate is not None else [],
                warnings=list(gate.warnings) if gate is not None else [],
                tiles=tiles,
        ))
    return CalibrationOverlayRecord(
            image_name=image_name, verdict=verdict, degree=degree,
            n_fitted=n_fitted, n_expected=n_expected, refusal=refusal, rois=rois,
    )
