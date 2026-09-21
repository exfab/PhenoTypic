"""Fit a root-polynomial colour correction from the chart in this frame.

:class:`CalibrateColorRpcc` is the end-to-end operation: given the rectangles
where colour-checker cards appear in a plate image, it locates the tiles,
works out which chart patch each one is, measures them, fits a profile and
applies it.  :class:`ColorCorrector` by contrast only applies a profile that
already exists.

Fitting from the frame's own card is most of the accuracy, not a convenience.
The root-polynomial expansion is positively homogeneous of degree 1, so a fit
on a frame that happens to be dark absorbs the gain as a matrix scaled by 1/k
-- a profile fitted elsewhere is exposure-invariant by construction and
preserves the deficit.  A transferred chip profile measured 15.93 mean
delta-E 2000 on the reference set's in-frame tiles against 1.95 for a
per-image fit, both via the earlier notebook extraction.

**What this implementation currently achieves is worse than that notebook
figure.**  On the same three reference frames it measures 3.40, 2.88 and 4.74
mean delta-E 2000 in-sample, and 4.37 to 11.35 held out on another frame.  The
median per-patch error is 2.11 -- at the notebook's level -- so the mean is
carried by a handful of tiles: purple, foliage and the four neutrals, which
are also the tiles with the highest within-tile spread.  That points at core
boxes straddling a patch edge rather than at the fit, with the deferred
white-balance confound (whose signature is error concentrated in the neutrals)
second.  One frame is corrected worse than it started.  Treat the numbers here
as the current state, not a specification, and see the plan's Task 7 for the
open diagnosis.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Annotated, Any, Literal, Sequence, overload

import numpy as np
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ...abc_ import ImageCorrector
from ...sdk_.typing_ import TuneSpec
from ._checker_detect import fit_lattice, refine
from ._checker_identity import (
    assign_placement,
    chart_grid,
    placements,
)
from ._checker_measure import (
    DEFAULT_MEDOID_CANDIDATES,
    extract_patch,
    measure_tile,
)
from ._checker_qc import (
    QcLimits,
    QcRecord,
    evaluate_roi,
    require_rank,
    warn_on_patch_census,
)
from ._checker_roi import CheckerLattice, CheckerRoi, coerce_rois
from ._color_checker_profile import ColorCheckerProfile
from ._color_corrector import ColorCorrector

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image

logger = logging.getLogger(__name__)

OnQcFail = Literal["raise", "warn", "skip"]

#: Refinement methods the operation offers.  ``"ecc"`` is deliberately absent:
#: it registers against a stored reference band, which would have to ride in
#: every serialised pipeline (see the spec's Detection section).
OperationRefineMethod = Literal["rigid", "frozen"]


def _root_cause(exc: BaseException) -> BaseException:
    """The innermost exception in a ``raise ... from`` chain."""
    while exc.__cause__ is not None:
        exc = exc.__cause__
    return exc


class CalibrateColorRpcc(ImageCorrector):
    """Fit and apply a root-polynomial correction from this frame's own chart.

    Best For:
        - Plate photographs with a colour-checker card mounted in the frame,
          where every image can calibrate itself.
        - Multi-session studies where a stored profile would carry a stale
          exposure: a same-image fit absorbs the gain, a transferred one
          cannot.
        - Unattended batches, where the quality gate must refuse a frame
          rather than emit a correction fitted to an obscured card.

    Consider Also:
        - :class:`ColorCorrector` when a fitted
          :class:`ColorCheckerProfile` already exists and only needs applying.
        - :class:`ColorCheckerProfile` directly when the chart is
          free-standing and fully visible rather than part of the scene.
        - :class:`DenoiseBlockMatch` afterwards: correction improves accuracy
          while mildly amplifying per-pixel noise.

    Args:
        rois: Rectangles to look inside, one per card region.  Required.
            Accepts :class:`CheckerRoi` objects or plain
            ``[row_min, col_min, row_max, col_max]`` sequences
            (``regionprops`` bbox order), mixed freely.  The rectangle is all
            that is declared: how many tiles it holds and which chart patch
            each one is are recovered from the pixels.
        checker_type: Key in ``colour.CCS_COLOURCHECKERS``.
        target_illuminant: Working illuminant.  Reference values are
            Bradford-adapted from the chart's own illuminant to this one
            before any comparison.
        degree: Root-polynomial degree.  **Fixed for every image in a run**;
            never adapted to what a frame happened to detect.  Correcting two
            frames of one experiment with different degrees puts a
            reproducible bias into the contrast between them.
        grid: Optional ``(nrows, ncols)`` of each card's tile block, used only
            when fitting a lattice from scratch.  Says nothing about which
            patches are present.
        lattice_prior: Per-ROI lattices from an earlier frame.  When given,
            they are refined onto this frame rather than re-fitted.
        refine_method: ``"rigid"`` (reference-free, default) or
            ``"frozen"``.  ECC registration is not offered: it needs a
            reference band shipped with every serialised pipeline.
        core_trim: Fraction trimmed from each tile box before measuring.
            ``0.4`` keeps the central 60 % of each axis, which is where the
            detector's residual error is absorbed.
        medoid_candidates: Candidate-set size for the deterministic medoid.
        outlier_sigma: Patches beyond ``mean + sigma * sd`` delta-E are
            rejected before fitting.
        min_patches: Accepted patches below which the frame is flagged.
        qc_limits: Gate thresholds.
        on_qc_fail: ``"raise"``, ``"warn"`` (correct anyway) or ``"skip"``
            (return the image uncorrected with the record attached).

    Returns:
        Image: ``rgb`` corrected, with ``gray`` and ``detect_mat`` recomputed.
        ``fitted_profile`` and ``qc`` are populated on the operation.

    Raises:
        ValueError: If any ROI fails the gate under ``on_qc_fail="raise"``,
            if the patches that reach the fit cannot support ``degree`` (under
            any policy), or if two ROIs claim the same chart patch.
            ``apply()`` re-raises every failure as ``RuntimeError`` with the
            ``ValueError`` as its root cause.
    """

    rois: list[CheckerRoi] = Field(min_length=1)
    checker_type: str = "ColorChecker24 - After November 2014"
    target_illuminant: str = "D65"
    degree: int = 3
    grid: tuple[int, int] | None = None
    lattice_prior: list[CheckerLattice] | None = None
    refine_method: OperationRefineMethod = "rigid"
    core_trim: Annotated[float, TuneSpec(0.2, 0.6)] = 0.4
    medoid_candidates: int = DEFAULT_MEDOID_CANDIDATES
    outlier_sigma: Annotated[float, TuneSpec(1.5, 4.0)] = 2.0
    min_patches: int = 20
    qc_limits: QcLimits = QcLimits()
    on_qc_fail: OnQcFail = "raise"

    fitted_profile: ColorCheckerProfile | None = None
    qc: list[QcRecord] = []

    _diagnostics: dict[str, Any] = PrivateAttr(default_factory=dict)

    @field_validator("rois", mode="before")
    @classmethod
    def _coerce_rois(cls, value: Any) -> Any:
        """Accept bounding-box sequences alongside :class:`CheckerRoi`."""
        if value is None or isinstance(value, CheckerRoi):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return coerce_rois(value)
        return value

    @field_validator("degree")
    @classmethod
    def _validate_degree(cls, degree: int) -> int:
        """Reject a degree the expansion does not define."""
        if degree not in (1, 2, 3, 4):
            raise ValueError(f"degree must be 1-4; got {degree}.")
        return degree

    @model_validator(mode="after")
    def _validate_lattice_prior_length(self) -> CalibrateColorRpcc:
        """One prior per ROI, in ROI order.

        Checked here so a short list fails when the operation is built,
        naming the field, rather than deep inside ``_operate`` as an
        ``IndexError`` on whichever image happened to be first.
        """
        if self.lattice_prior is not None and len(self.lattice_prior) != len(self.rois):
            raise ValueError(
                    f"lattice_prior has {len(self.lattice_prior)} entries but there "
                    f"are {len(self.rois)} rois; supply exactly one per ROI, in the "
                    "same order."
            )
        return self

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Per-run detail: illuminant handling, patch census, per-tile values."""
        return self._diagnostics

    @overload
    def apply(self, image: GridImage, inplace: bool = False) -> GridImage: ...

    @overload
    def apply(self, image: Image, inplace: bool = False) -> Image: ...

    def apply(self, image: Image, inplace: bool = False) -> Image:
        """Calibrate from this image's own chart, then correct it."""
        return super().apply(image, inplace=inplace)

    # -- stages ------------------------------------------------------------
    def _roi_views(self, image: Image, roi: CheckerRoi):
        """Lab and normalised sRGB for one ROI.

        Lab comes from ``Image.color.Lab`` so the encoding is handled once and
        correctly; feeding linear RGB to a converter that expects sRGB is the
        defect this operation exists partly to avoid.
        """
        from phenotypic._core._image import Image as _Image

        sub = image.rgb[roi.row_slice, roi.col_slice]
        wrapped = _Image(arr=np.ascontiguousarray(sub), gamma=image.gamma)
        return wrapped.color.Lab[:], wrapped.rgb.normed()

    def _measure_roi(self, lab, srgb, lattice, roi_index):
        """Measure every tile of one ROI's lattice."""
        return [
            measure_tile(
                    extract_patch(lab, (y0, y1, x0, x1)),
                    extract_patch(srgb, (y0, y1, x0, x1)),
                    row=row, col=col, roi_index=roi_index,
                    candidates=self.medoid_candidates,
            )
            for row, col, y0, y1, x0, x1 in lattice.boxes(core=self.core_trim)
        ]

    def _operate(self, image: Image) -> Image:
        """Detect, identify, measure, fit, apply."""
        import colour

        from ._color_checker_profile import _load_reference_data

        # One instance may process many images; nothing from the last one may
        # survive into this one's result, least of all on the skip path.
        self.fitted_profile = None
        self.qc = []
        self._diagnostics = {}

        ref_lab, ref_linear, _wp = _load_reference_data(
                self.checker_type, self.target_illuminant
        )
        patch_names = list(ref_lab.keys())
        grid = chart_grid(patch_names)
        chart_illuminant = colour.CCS_COLOURCHECKERS[self.checker_type].illuminant

        measured: dict[str, tuple[float, float, float]] = {}
        records: list[QcRecord] = []
        tiles_out: list[dict[str, Any]] = []
        lattices: list[CheckerLattice] = []

        for index, roi in enumerate(self.rois):
            lab, srgb = self._roi_views(image, roi)

            if self.lattice_prior is not None:
                refined = refine(
                        lab, self.lattice_prior[index], method=self.refine_method,
                        anchor_col=roi.anchor_col,
                )
                lattice, shift = refined.lattice, float(np.hypot(refined.dy, refined.dx))
            else:
                try:
                    lattice = fit_lattice(lab, grid=self.grid)
                except ValueError as exc:
                    # No card in the rectangle is a property of this frame,
                    # not of the configuration: the policy decides.
                    records.append(self._refusal(index, roi, f"lattice not found: {exc}"))
                    continue
                shift = 0.0
            lattices.append(lattice)

            refusals: list[str] = []
            if roi.expect_tiles is not None and lattice.n_tiles != roi.expect_tiles:
                refusals.append(
                        f"declared to hold {roi.expect_tiles} tiles but "
                        f"{lattice.n_tiles} were detected"
                )
            candidates = placements(grid, (lattice.nrows, len(lattice.columns)))
            if len(candidates) < 2:
                refusals.append(
                        f"a {lattice.nrows}x{len(lattice.columns)} tile block cannot "
                        f"sit on a {len(grid)}x{len(grid[0])} chart in more than one "
                        "way; patch identity would be assumed, not measured"
                )
            if refusals:
                records.append(self._refusal(index, roi, *refusals))
                continue

            tiles = self._measure_roi(lab, srgb, lattice, index)
            # Index explicitly by (row, col): CheckerLattice.boxes() yields
            # column-major, so reshaping the flat list would transpose the
            # card and mislabel every patch.
            observed = np.empty((lattice.nrows, len(lattice.columns), 3))
            for tile in tiles:
                observed[tile.row, tile.col] = colour.cctf_decoding(
                        np.clip(tile.srgb, 0, 1), function="sRGB"
                )
            identity = assign_placement(observed, candidates, ref_linear)

            record = evaluate_roi(
                    roi_index=index,
                    label=roi.label,
                    shift_px=shift,
                    anchor_disagreement_px=self._anchor_disagreement(
                            lab, index, lattice
                    ),
                    ecc_confidence=None,
                    placement_margin=identity.margin,
                    hungarian_disagreement=(
                        identity.n_tiles - identity.hungarian_agreement
                    ),
                    impurities=np.array([t.impurity for t in tiles]),
                    robust_shifts=np.array([t.robust_shift for t in tiles]),
                    clipped=np.array([t.clipped for t in tiles]),
                    limits=self.qc_limits,
            )
            records.append(record)

            # Measurements are collected whatever the gate said; the
            # ``on_qc_fail`` policy below decides whether they are used.
            # Dropping them here would make "warn" behave like "skip".
            for tile in tiles:
                name = identity.placement.names[tile.row][tile.col]
                if name in measured:
                    raise ValueError(
                            f"Two ROIs both identified a patch as {name!r}. Each "
                            "card region must cover a different part of the chart; "
                            "check the rectangles, or the placement margins in "
                            "`qc` if they overlap."
                    )
                measured[name] = tile.srgb
            for message in record.warnings:
                warnings.warn(
                        f"ROI {index} ({roi.label or 'unlabelled'}): {message}",
                        UserWarning, stacklevel=4,
                )
            tiles_out.extend(
                    {
                        **tile.model_dump(),
                        "patch": identity.placement.names[tile.row][tile.col],
                    }
                    for tile in tiles
            )

        self.qc = records
        failed = [record for record in records if not record.ok]
        if failed:
            summary = "; ".join(
                    f"ROI {r.roi_index}: {', '.join(r.flags)}" for r in failed
            )
            if self.on_qc_fail == "raise":
                raise ValueError(f"Colour-checker quality gate failed. {summary}")
            warnings.warn(
                    f"Colour-checker quality gate failed. {summary}",
                    UserWarning, stacklevel=3,
            )
            if self.on_qc_fail == "skip":
                self._diagnostics = self._build_diagnostics(
                        chart_illuminant, [], patch_names, tiles_out, lattices,
                        accepted=[], skipped=True,
                )
                return image

        accepted = list(measured.keys())
        census = warn_on_patch_census(
                accepted, patch_names, self.degree, self.min_patches,
        )

        profile = ColorCheckerProfile(
                checker_type=self.checker_type,
                target_illuminant=self.target_illuminant,
                degree=self.degree,
                outlier_sigma=self.outlier_sigma,
        )
        profile.fit_from_patch_colors(
                {name: np.asarray(value) for name, value in measured.items()}
        )
        fitted = profile.diagnostics
        require_rank(
                fitted["n_patches_detected"] - fitted["n_patches_rejected"],
                self.degree,
                stage="remain after outlier rejection",
        )
        self.fitted_profile = profile
        self._diagnostics = self._build_diagnostics(
                chart_illuminant, census, patch_names, tiles_out, lattices,
                accepted=accepted,
        )
        return ColorCorrector(
                profile=profile, output_illuminant=self.target_illuminant
        ).apply(image, inplace=True)

    @staticmethod
    def _refusal(index: int, roi: CheckerRoi, *flags: str) -> QcRecord:
        """A record for an ROI that failed before it could be measured."""
        return QcRecord(roi_index=index, label=roi.label, flags=list(flags))

    def _anchor_disagreement(
            self, lab, roi_index: int, lattice: CheckerLattice
    ) -> float | None:
        """Spread between the shifts different anchor columns imply.

        A reference-free internal consistency check: the columns of one rigid
        card must agree about where it moved. Only meaningful when a prior is
        being refined and the card has more than one column.
        """
        from ._checker_detect import refine_rigid

        if self.lattice_prior is None or len(lattice.columns) < 2:
            return None
        prior = self.lattice_prior[roi_index]
        estimates = [
            refine_rigid(lab, prior, anchor_col=index).dx
            for index in range(len(prior.columns))
        ]
        return float(max(estimates) - min(estimates))

    def _build_diagnostics(
            self, chart_illuminant, census, patch_names, tiles, lattices,
            accepted: list[str], skipped: bool = False,
    ) -> dict[str, Any]:
        """Assemble the per-run diagnostics record."""
        import numpy as _np

        return {
            "degree"    : self.degree,
            "skipped"   : skipped,
            "illuminant": {
                "checker"         : _np.asarray(chart_illuminant).tolist(),
                "target"          : self.target_illuminant,
                "bradford_adapted": True,
            },
            "patch_census": {
                "expected": patch_names,
                "accepted": accepted,
                "warnings": census,
            },
            "tiles"    : tiles,
            "lattices" : [lattice.model_dump() for lattice in lattices],
            "qc"       : [record.model_dump() for record in self.qc],
        }

    @classmethod
    def patch_census(
            cls, images: Sequence[Image], *, rois: Sequence[Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Count what each image's card offers, before choosing a degree.

        Counts only -- it recommends nothing.  Run it once over a batch to see
        whether every frame can support the degree you intend to fix, since
        that degree must then be the same for all of them.

        Args:
            images: Images to survey.
            rois: As for the constructor.
            **kwargs: Passed to the constructor.

        Returns:
            ``{"per_image": {name: accepted_count}, "missing": [...],
            "cyan_everywhere": bool, "worst": int}``.
        """
        per_image: dict[str, int] = {}
        missing: set[str] = set()
        for image in images:
            operation = cls(rois=list(rois), on_qc_fail="warn", **kwargs)
            try:
                operation.apply(image)
            except RuntimeError as exc:
                # apply() wraps every failure as RuntimeError; a ValueError at
                # the root is this frame failing to calibrate, which is what a
                # census counts. Anything else is a bug and propagates.
                cause = _root_cause(exc)
                if not isinstance(cause, ValueError):
                    raise
                logger.info("patch_census: %s failed (%s)", image.name, cause)
                per_image[image.name] = 0
                continue
            accepted = operation.diagnostics["patch_census"]["accepted"]
            expected = operation.diagnostics["patch_census"]["expected"]
            per_image[image.name] = len(set(accepted))
            missing |= set(expected) - set(accepted)
        return {
            "per_image"      : per_image,
            "missing"        : sorted(missing),
            "cyan_everywhere": "cyan" not in missing,
            "worst"          : min(per_image.values()) if per_image else 0,
        }
