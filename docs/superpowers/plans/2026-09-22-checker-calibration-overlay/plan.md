# Calibration tile overlay figure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `CalibrateColorRpcc.show_tiles()` returns a matplotlib figure. For every checker ROI it shows where each tile was measured, which chart patch the tile was matched to, and ΔE00 before → after correction. It also works for a frame the quality gate refused.

**Architecture:** One new private module, `_calibration_overlay.py`, holds three things:
- a frozen plain-data record (`CalibrationOverlayRecord`);
- `build_overlay_record(...)`, which assembles it from what the ROI loop saw;
- `render_calibration_overlay(record)`, a pure matplotlib renderer.

The operation builds the record during `apply()`, capturing an owned copy of each as-shot ROI crop, and keeps it privately. It assigns the record before every exit, including the refusal raises. `show_tiles()` renders it. Every label position is computed in inches from measured text extents, so overlap is prevented by construction, and a test proves it.

**Tech Stack:** Python 3.11+, numpy, pydantic v2, matplotlib (Agg canvas, explicit `Figure`), pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-22-checker-calibration-overlay/README.md`. Read it first. The approved look is `docs/superpowers/artifacts/2026-09-22-checker-calibration-overlay/panel-b-matplotlib.png`.

**Branch / worktree:** `feat/checker-calibration-overlay` in `.worktrees/calibrate-color-rpcc`, based on `origin/main` at `dcfb3df01`.

## Global Constraints

- `uv run` only. Tests: `QT_QPA_PLATFORM=offscreen uv run pytest <paths> -p no:cacheprovider -q -o addopts="" -n 4`. Never use `-x`, and never run the full `tests/unit` suite.
- **Explicit matplotlib only:** build `matplotlib.figure.Figure` and attach `FigureCanvasAgg`. Never import `matplotlib.pyplot`.
- **Lazy imports:** `matplotlib` and `phenotypic.sdk_.viz.figures._mpl_theme` are imported **inside** `render_calibration_overlay`, never at module level. The new module is registered in `tests/unit/ci/test_deferred_imports.py`, and `tests/unit/ci/test_startup_imports.py` must stay green.
- **Crops own their buffers:** `np.array(sub, copy=True)`, never a view into `image.rgb` (`src/phenotypic/abc_/CLAUDE.md`).
- **Out of scope** (spec §Non-goals): saving the record into the store, `@figure` / `PlotImage`, CLI publication, panels A, C and D, plotly, and configurable ΔE bands. Do not add any of them.
- **Keep each file's existing line endings.** Working copies of existing files are CRLF (`core.autocrlf=true`). Check `grep -c $'\r' f` against `wc -l < f` before committing.
- `uv run ruff check <explicit paths>` only. Never run it bare.
- `MeasurementInfo` and `bio_desc` are not touched.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `tests/unit/correction/_checker_frames.py` (new) | Shared synthetic rig frame, ROIs, prior and op factory, moved out of the review test | 1 |
| `tests/unit/correction/test_calibrate_color_rpcc_review.py` | Imports the helpers from `_checker_frames` instead of defining them | 1 |
| `src/phenotypic/correction/_color_correction/_calibration_overlay.py` (new) | Record models, `RoiDraft`, `build_overlay_record`, `render_calibration_overlay` | 2, 3 |
| `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` | Builds and keeps the record in `_operate`; `calibration_record` and `show_tiles()` | 2, 4 |
| `tests/unit/correction/test_calibration_overlay.py` (new) | Record-fidelity, lifecycle and renderer tests | 2, 3, 4 |
| `tests/unit/ci/test_deferred_imports.py` | Registers the new module's function-local imports | 3 |
| `src/phenotypic/correction/CLAUDE.md` | One paragraph on the overlay and its record | 4 |

**DAG:** 1 → 2 → 3 → 4 → 5, strictly sequential. Tasks 2 and 4 both edit `_calibrate_color_rpcc.py`, and every later task imports what Task 2 defines.

---

### Task 1: Share the synthetic frame helpers

**Files:**
- Create: `tests/unit/correction/_checker_frames.py`
- Modify: `tests/unit/correction/test_calibrate_color_rpcc_review.py`, deleting the helper block (from `CHECKER = ...` through `def quietly(...)`) and importing it instead

**Interfaces:**
- Produces: `from ._checker_frames import (BAND_H, CHECKER, GAP, NAMES, PITCH, SRGB, TILE, TOP, _band_patch, band_prior, band_rois, frozen_op, quietly, render_frame)`. The signatures are exactly those in the review file today.

- [ ] **Step 1: Move the helpers.** Cut everything from `CHECKER = "ColorChecker24 - After November 2014"` down to the end of `def quietly(...)` in `test_calibrate_color_rpcc_review.py`, and paste it into `_checker_frames.py` under this header:

```python
"""Rig-shaped synthetic colour-checker frames shared by the correction tests.

Two vertical bands, each a transposed 6x2 half-card of real ColorChecker24
colours, with the lattice supplied as a prior -- the production path.
"""

from __future__ import annotations

import warnings

import numpy as np

from phenotypic import Image
from phenotypic.correction import CalibrateColorRpcc
from phenotypic.correction._color_correction._checker_roi import (
    CheckerLattice,
    ColumnLattice,
)
```

In the review file, replace the removed block with:

```python
from ._checker_frames import (
    BAND_H,
    PITCH,
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
```

Then drop any imports the review file no longer uses. `ruff` will name them.

- [ ] **Step 2: Run the review tests unchanged.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction/test_calibrate_color_rpcc_review.py -p no:cacheprovider -q -o addopts="" -n 4`
Expected: every test passes, with the same passed count as before the move. Run the file once before Step 1 and note the count. `uv run ruff check tests/unit/correction/_checker_frames.py tests/unit/correction/test_calibrate_color_rpcc_review.py` is clean.

- [ ] **Step 3: Commit.**

```bash
git add tests/unit/correction/_checker_frames.py tests/unit/correction/test_calibrate_color_rpcc_review.py
git commit -m "test(color): share the synthetic checker frame helpers"
```

---

### Task 2: The record, built on every exit of `apply()`

**Files:**
- Create: `src/phenotypic/correction/_color_correction/_calibration_overlay.py` (the models, `RoiDraft`, `build_overlay_record`)
- Modify: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` (`_operate`, the new private attribute and the `calibration_record` property)
- Test: `tests/unit/correction/test_calibration_overlay.py`

**Interfaces:**
- Produces:
  - `CalibrationOverlayRecord`, `RoiOverlay`, `TileOverlay` (frozen pydantic)
  - `RoiDraft` (dataclass)
  - `build_overlay_record(*, image_name, verdict, refusal, degree, n_expected, n_fitted, drafts, qc, reference_srgb, fitted_patches, rejected, impurity_limit, core_trim) -> CalibrationOverlayRecord`
  - `CalibrateColorRpcc.calibration_record -> CalibrationOverlayRecord | None`

- [ ] **Step 1: Write the failing tests** in `tests/unit/correction/test_calibration_overlay.py`:

```python
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


def test_a_collided_roi_marks_its_tiles_excluded() -> None:
    grey = np.array([0.3, 0.3, 0.3])
    overrides = {(1, 0, c): SRGB[_band_patch(0, 0, c)] for c in (0, 1)}
    overrides |= {(1, r, c): grey for r in range(1, 6) for c in (0, 1)}
    # degree 2: ROI 1 loses its patches, leaving ROI 0's 12, below degree 3's 13 terms.
    record = calibrated(render_frame(overrides=overrides), on_qc_fail="warn",
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
```

- [ ] **Step 2: Run them to confirm they fail.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction/test_calibration_overlay.py -p no:cacheprovider -q -o addopts="" -n 4`
Expected: every test fails with `AttributeError: 'CalibrateColorRpcc' object has no attribute 'calibration_record'`.

- [ ] **Step 3: Create `_calibration_overlay.py`** with the record half. The renderer is added in Task 3.

```python
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
        fitted_patches: Mapping[str, Mapping[str, object]] | None,
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
                        measured_srgb=tuple(measured.srgb) if measured.n_pixels else None,
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
```

- [ ] **Step 4: Build and keep the record in `_calibrate_color_rpcc.py`.** Make these edits, keeping CRLF:

  1. **Imports:** `from ._calibration_overlay import CalibrationOverlayRecord, RoiDraft, build_overlay_record`.
  2. **Private attribute,** next to `_diagnostics`: `_calibration_record: CalibrationOverlayRecord | None = PrivateAttr(default=None)`.
  3. **Property,** after `diagnostics`:

     ```python
         @property
         def calibration_record(self) -> CalibrationOverlayRecord | None:
             """The last ``apply()``'s tile overlay record, or ``None`` before one.

             Kept even when the frame was refused or skipped, and replaced by
             every ``apply()``. Never serialised.
             """
             return self._calibration_record
     ```
  4. **Per-run reset:** in `_operate`, directly after `self._diagnostics = {}`, add `self._calibration_record = None`.
  5. **Local state:** after `lattices: list[...] = []`, add the drafts list, the reference colours, and a helper that assigns the record:

     ```python
             drafts: list[RoiDraft] = []
             reference_srgb = {
                 name: tuple(float(v) for v in np.clip(colour.cctf_encoding(
                         np.clip(ref_linear[name], 0, 1), function="sRGB"), 0, 1))
                 for name in patch_names
             }

             def keep_record(verdict, *, refusal=None, fitted=None, rejected=(), n_fitted=None):
                 self._calibration_record = build_overlay_record(
                         image_name=image.name, verdict=verdict, refusal=refusal,
                         degree=self.degree, n_expected=len(patch_names),
                         n_fitted=n_fitted, drafts=drafts, qc=records,
                         reference_srgb=reference_srgb, fitted_patches=fitted,
                         rejected=set(rejected),
                         impurity_limit=self.qc_limits.max_tile_impurity,
                         core_trim=self.core_trim,
                 )
     ```
  6. **Crop capture:** first line inside `for index, roi in enumerate(self.rois):`, before `_roi_views`:

     ```python
                 draft = RoiDraft(
                         roi_index=index, label=roi.label,
                         crop=np.array(image.rgb[roi.row_slice, roi.col_slice], copy=True),
                 )
                 drafts.append(draft)
     ```
  7. **Lattice:** after `lattices.append(lattice)` (the success branch, not the `None` one), add `draft.lattice = lattice`.
  8. **Tiles:** after `identity = assign_placement(...)`, add `draft.tiles = [(t, identity.placement.names[t.row][t.col]) for t in tiles]`.
  9. **Claim:** in the collision `else:` branch, after `claimed_by.update(...)`, add `draft.claimed = True`.
  10. **Assign before every exit:**

      ```python
              if failed:
                  summary = self._flag_summary(failed)
                  if self.on_qc_fail == "raise":
                      message = f"Colour-checker quality gate failed. {summary}"
                      keep_record("refused", refusal=message)
                      raise ValueError(message)
                  ...                                  # warnings.warn unchanged
                  if self.on_qc_fail == "skip":
                      keep_record("skipped")
                      self._diagnostics = ...          # unchanged
                      return image

              if not measured:
                  summary = ...                        # unchanged
                  message = f"No ROI produced usable tiles, so there is nothing to fit. {summary}"
                  keep_record("refused", refusal=message)
                  raise ValueError(message)
              try:
                  require_rank(len(measured), self.degree, stage="were measured")
              except ValueError as exc:
                  keep_record("refused", refusal=str(exc))
                  raise
              ...                                      # profile fit unchanged
              try:
                  require_rank(len(accepted), self.degree, stage="remain after outlier rejection")
              except ValueError as exc:
                  keep_record("refused", refusal=str(exc))
                  raise
              census = ...                             # unchanged
              self.fitted_profile = profile
              self._diagnostics = ...                  # unchanged
              keep_record(
                      "corrected_with_warnings"
                      if any(r.flags or r.warnings for r in records) else "corrected",
                      fitted=profile.diagnostics["patches"], rejected=rejected,
                      n_fitted=len(accepted),
              )
              return ColorCorrector(...).apply(image, inplace=True)   # unchanged
      ```

- [ ] **Step 5: Run the tests.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction/test_calibration_overlay.py tests/unit/correction/test_calibrate_color_rpcc.py tests/unit/correction/test_calibrate_color_rpcc_review.py -p no:cacheprovider -q -o addopts="" -n 4`
Expected: all pass. Nothing that passed before fails.

- [ ] **Step 6: Commit.**

```bash
git add src/phenotypic/correction/_color_correction/_calibration_overlay.py src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py tests/unit/correction/test_calibration_overlay.py
git commit -m "feat(color): keep a tile overlay record from every CalibrateColorRpcc run"
```

---

### Task 3: The renderer

**Files:**
- Modify: `src/phenotypic/correction/_color_correction/_calibration_overlay.py` (append the renderer)
- Modify: `tests/unit/ci/test_deferred_imports.py` (register the module)
- Test: `tests/unit/correction/test_calibration_overlay.py` (append)

**Interfaces:**
- Consumes: `CalibrationOverlayRecord`, `RoiOverlay` and `TileOverlay` from Task 2.
- Produces: `render_calibration_overlay(record, *, figsize=None, dpi=160) -> matplotlib.figure.Figure`, plus the constants `DELTA_E_GOOD = 2.0`, `DELTA_E_FAIR = 5.0`, `STATUS_COLOURS` and `delta_e_band(value) -> str`.

- [ ] **Step 1: Write the failing renderer tests.** Append to `test_calibration_overlay.py`:

```python
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from phenotypic.correction._color_correction import _calibration_overlay as overlay
from phenotypic.correction._color_correction._calibration_overlay import (
    STATUS_COLOURS,
    CalibrationOverlayRecord,
    RoiOverlay,
    TileOverlay,
    render_calibration_overlay,
)


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
```

- [ ] **Step 2: Run them to confirm they fail.** Expected: `ImportError: cannot import name 'STATUS_COLOURS'`.

- [ ] **Step 3: Append the renderer** to `_calibration_overlay.py`. Add `import math` and `import textwrap` to the imports, and `from matplotlib.figure import Figure` under `TYPE_CHECKING`.

```python
#: ΔE00 after-correction bands (spec §Colour): <= GOOD is good, <= FAIR fair.
DELTA_E_GOOD = 2.0
DELTA_E_FAIR = 5.0

#: Okabe-Ito semantic colours per tile status (DESIGN.md §01).
STATUS_COLOURS: dict[str, str] = {
    "used": "#009E73", "partly_covered": "#E69F00", "rejected": "#D55E00",
    "excluded": "#BBBBBB", "empty": "#BBBBBB",
}
_DASHED = frozenset({"rejected", "excluded", "empty"})
#: Darkened text variants of the data colours, legible on a light ground.
_DELTA_E_TEXT = {"good": "#007a5a", "fair": "#a86f00", "poor": "#b04a00"}
_FLAG_TEXT, _WARN_TEXT, _BODY_TEXT = "#b04a00", "#a86f00", "#2e3a4e"
_SUFFIX = {"used": "", "rejected": " · rejected", "excluded": " · excluded",
           "empty": " · empty"}

_NAME_PT, _DE_PT, _TITLE_PT, _ROI_PT, _NOTE_PT = 8.0, 7.5, 10.0, 9.0, 7.5
_IMAGE_W_IN = 1.6        # nominal image width; grows when labels need room
_SWATCH_IN = 0.14        # each of the measured | reference swatches
_GAP_IN = 0.06           # image <-> swatch, swatch <-> text
_EDGE_IN = 0.08          # outer padding
_GROUP_GAP_IN = 0.3      # between ROI groups
_LINE_GAP = 1.15         # line pitch as a multiple of text height
_CORE_LW = 1.8


def delta_e_band(value: float) -> str:
    """``"good"``, ``"fair"`` or ``"poor"`` for a ΔE00 after-correction value."""
    if value <= DELTA_E_GOOD:
        return "good"
    return "fair" if value <= DELTA_E_FAIR else "poor"


class _TextMeter:
    """Measures rendered text in inches, on the figure's own renderer."""

    def __init__(self, probe, dpi: float) -> None:
        self._probe = probe
        self._renderer = probe.canvas.get_renderer()
        self._dpi = dpi

    def size(self, text: str, *, fontsize: float, family: str = "sans-serif"):
        artist = self._probe.text(0, 0, text, fontsize=fontsize, family=family)
        box = artist.get_window_extent(self._renderer)
        artist.remove()
        return box.width / self._dpi, box.height / self._dpi


@dataclass
class _Block:
    name: str
    delta_e: str
    colour: str
    width: float
    height: float


@dataclass
class _RoiPlan:
    title: str
    title_h: float
    side: bool
    left_w: float
    image_w: float
    image_h: float
    right_w: float
    width: float
    blocks: dict[tuple[int, int], _Block]
    key_h: float = 0.0
    key_cols: int = 1
    key_entry_w: float = 0.0
    key_row_h: float = 0.0
    number_w: float = 0.0
    notes: list[tuple[str, str]] = field(default_factory=list)
    note_line_h: float = 0.0
    note_h: float = 0.0


def _name_line(tile: TileOverlay) -> str:
    if tile.status == "partly_covered":
        return f"{tile.patch} · {(tile.impurity or 0.0) * 100:.0f}% covered"
    return tile.patch + _SUFFIX[tile.status]


def _delta_e_line(tile: TileOverlay) -> tuple[str, str]:
    if tile.delta_e_after is None:
        return "ΔE00 not fitted", _BODY_TEXT
    return (f"ΔE00 {tile.delta_e_before:.1f} → {tile.delta_e_after:.1f}",
            _DELTA_E_TEXT[delta_e_band(tile.delta_e_after)])


def _block(tile: TileOverlay, meter: _TextMeter) -> _Block:
    name = _name_line(tile)
    delta_e, colour = _delta_e_line(tile)
    name_w, name_h = meter.size(name, fontsize=_NAME_PT)
    de_w, de_h = meter.size(delta_e, fontsize=_DE_PT, family="monospace")
    return _Block(name, delta_e, colour, max(name_w, de_w), name_h + de_h)


def _min_centre_gap_px(tiles: Sequence[TileOverlay]) -> float | None:
    """Smallest vertical gap between neighbouring tile centres in one column."""
    gaps = []
    for col in {t.col for t in tiles}:
        centres = sorted((t.full_box[0] + t.full_box[1]) / 2 for t in tiles if t.col == col)
        gaps += [b - a for a, b in zip(centres, centres[1:])]
    return min(gaps) if gaps else None


def _plan_roi(roi: RoiOverlay, meter: _TextMeter) -> _RoiPlan:
    """Every size for one ROI group, in inches, from measured text."""
    h_px, w_px = roi.crop.shape[:2]
    title = f"ROI {roi.roi_index}" + (f" · {roi.label}" if roi.label else "")
    title_w, title_h = meter.size(title, fontsize=_ROI_PT, family="monospace")
    blocks = {(t.row, t.col): _block(t, meter) for t in roi.tiles}
    block_h = max((b.height for b in blocks.values()), default=0.0)
    side = roi.n_tile_columns <= 2
    image_w, image_h = _IMAGE_W_IN, _IMAGE_W_IN * h_px / w_px

    if side:
        gap_px = _min_centre_gap_px(roi.tiles)
        if gap_px:
            have = gap_px * image_h / h_px
            scale = max(1.0, block_h * _LINE_GAP / have)
            image_w, image_h = image_w * scale, image_h * scale
        swatches = _GAP_IN + 2 * _SWATCH_IN + _GAP_IN
        widths = {c: max((b.width for (_, cc), b in blocks.items() if cc == c), default=None)
                  for c in (0, 1)}
        left_w = _EDGE_IN + (widths[0] + swatches if widths[0] is not None else 0.0)
        right_w = _EDGE_IN + (widths[1] + swatches if widths[1] is not None else 0.0)
        plan = _RoiPlan(title, title_h, True, left_w, image_w, image_h, right_w,
                        left_w + image_w + right_w, blocks)
    else:
        number_w, number_h = meter.size(str(len(roi.tiles)), fontsize=_NAME_PT, family="monospace")
        smallest_px = min(min(t.full_box[1] - t.full_box[0], t.full_box[3] - t.full_box[2])
                          for t in roi.tiles)
        need = max(number_w, number_h) * 1.6
        scale = max(1.0, need / (smallest_px * image_w / w_px))
        image_w, image_h = image_w * scale, image_h * scale
        entry_w = (number_w + _GAP_IN + 2 * _SWATCH_IN + _GAP_IN
                   + max(b.width for b in blocks.values()) + _EDGE_IN)
        width = max(_EDGE_IN + image_w + _EDGE_IN, entry_w)
        pad = (width - image_w) / 2
        cols = max(1, int(width // entry_w))
        row_h = block_h * _LINE_GAP
        plan = _RoiPlan(title, title_h, False, pad, image_w, image_h, pad, width, blocks,
                        key_h=_GAP_IN + math.ceil(len(roi.tiles) / cols) * row_h,
                        key_cols=cols, key_entry_w=entry_w, key_row_h=row_h,
                        number_w=number_w)

    if title_w > plan.width:                     # widen both sides equally
        extra = (title_w - plan.width) / 2
        plan.left_w, plan.right_w, plan.width = plan.left_w + extra, plan.right_w + extra, title_w

    # Flags and warnings, wrapped to the group width in monospace so a
    # character count is a width.
    char_w = meter.size("x" * 40, fontsize=_NOTE_PT, family="monospace")[0] / 40
    chars = max(12, int((plan.width - 2 * _EDGE_IN) / char_w))
    for message, colour in [(f, _FLAG_TEXT) for f in roi.flags] + [(w, _WARN_TEXT) for w in roi.warnings]:
        plan.notes += [(line, colour) for line in textwrap.wrap(message, chars)]
    if plan.notes:
        plan.note_line_h = meter.size("Ag", fontsize=_NOTE_PT, family="monospace")[1] * _LINE_GAP
        plan.note_h = _GAP_IN + len(plan.notes) * plan.note_line_h
    return plan


def _figure_title(record: CalibrationOverlayRecord) -> str:
    fitted = ("not fitted" if record.n_fitted is None
              else f"{record.n_fitted}/{record.n_expected} patches fitted")
    return (f"{record.image_name or 'unnamed image'} · {record.verdict.replace('_', ' ')}"
            f" · degree {record.degree} · {fitted}")


def _swatches(ax, x: float, cy: float, height: float, tile: TileOverlay, rectangle) -> None:
    for k, rgb in enumerate((tile.measured_srgb, tile.reference_srgb)):
        style = (dict(facecolor=rgb) if rgb is not None
                 else dict(facecolor="none", hatch="////"))
        ax.add_patch(rectangle((x + k * _SWATCH_IN, cy - height / 2), _SWATCH_IN, height,
                               edgecolor=_BODY_TEXT, linewidth=0.4, **style))


def render_calibration_overlay(
        record: CalibrationOverlayRecord,
        *,
        figsize: tuple[float, float] | None = None,
        dpi: float = 160,
) -> Figure:
    """Draw each ROI's tiles, the patch each was matched to, and its ΔE00.

    Every size is computed in inches from measured text, so no two labels
    overlap and none leaves the figure (spec §2).

    Args:
        record: From ``CalibrateColorRpcc.calibration_record``.
        figsize: Optional ``(width, height)`` in inches. Must be at least the
            computed size; the content is centred in any extra room.
        dpi: Resolution the text is measured and drawn at.

    Returns:
        A ``matplotlib.figure.Figure`` with an Agg canvas attached.

    Raises:
        ValueError: If *figsize* is smaller than the labels need.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    from phenotypic.sdk_.viz.figures._mpl_theme import phenotypic_mpl_context

    with phenotypic_mpl_context():
        probe = Figure(dpi=dpi)
        FigureCanvasAgg(probe)
        meter = _TextMeter(probe, dpi)
        title = _figure_title(record)
        title_w, title_h = meter.size(title, fontsize=_TITLE_PT, family="monospace")
        plans = [_plan_roi(roi, meter) for roi in record.rois]

        content_w = sum(p.width for p in plans) + _GROUP_GAP_IN * max(0, len(plans) - 1)
        stack = [p.title_h + _GAP_IN + p.image_h + p.key_h + p.note_h for p in plans]
        top_band = title_h + 2 * _EDGE_IN
        need_w = max(content_w, title_w) + 2 * _EDGE_IN
        need_h = top_band + max(stack, default=0.0) + _EDGE_IN
        if figsize is not None and (figsize[0] < need_w - 1e-9 or figsize[1] < need_h - 1e-9):
            raise ValueError(
                    f"figsize={figsize} is too small for these labels; this record "
                    f"needs at least ({need_w:.2f}, {need_h:.2f}) inches."
            )
        fig_w, fig_h = figsize if figsize is not None else (need_w, need_h)
        fig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
        FigureCanvasAgg(fig)

        def axes(x, bottom, w, h, **kwargs):
            return fig.add_axes((x / fig_w, bottom / fig_h, w / fig_w, h / fig_h), **kwargs)

        fig.text(0.5, 1 - _EDGE_IN / fig_h, title, ha="center", va="top",
                 fontsize=_TITLE_PT, family="monospace", color=_BODY_TEXT)
        x = (fig_w - content_w) / 2
        y_top = fig_h - top_band - (fig_h - need_h) / 2
        for roi, plan in zip(record.rois, plans):
            _draw_roi(roi, plan, axes, x, y_top, Rectangle)
            x += plan.width + _GROUP_GAP_IN
        return fig


def _draw_roi(roi: RoiOverlay, plan: _RoiPlan, axes, x: float, y_top: float, rectangle) -> None:
    """Title, image with boxes, labels or key, and notes for one ROI group."""
    h_px, w_px = roi.crop.shape[:2]
    title_ax = axes(x, y_top - plan.title_h, plan.width, plan.title_h)
    title_ax.set_axis_off()
    title_ax.text(0.5, 0.0, plan.title, ha="center", va="bottom", fontsize=_ROI_PT,
                  family="monospace", color=_BODY_TEXT, transform=title_ax.transAxes)

    img_bottom = y_top - plan.title_h - _GAP_IN - plan.image_h
    ax = axes(x + plan.left_w, img_bottom, plan.image_w, plan.image_h)
    ax.imshow(roi.crop, interpolation="nearest", aspect="auto")
    ax.set_xlim(-0.5, w_px - 0.5)
    ax.set_ylim(h_px - 0.5, -0.5)
    ax.set_axis_off()

    for tile in roi.tiles:
        y0, y1, x0, x1 = tile.full_box
        ax.add_patch(rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                               edgecolor="white", linewidth=0.6, linestyle=(0, (1.5, 1.5)),
                               alpha=0.8))
        cy0, cy1, cx0, cx1 = tile.core_box
        ax.add_patch(rectangle((cx0 - 0.5, cy0 - 0.5), cx1 - cx0, cy1 - cy0, fill=False,
                               edgecolor=STATUS_COLOURS[tile.status], linewidth=_CORE_LW,
                               linestyle="--" if tile.status in _DASHED else "-"))

    bottom = img_bottom
    if plan.side:
        sides = {0: axes(x, img_bottom, plan.left_w, plan.image_h, sharey=ax),
                 1: axes(x + plan.left_w + plan.image_w, img_bottom, plan.right_w,
                         plan.image_h, sharey=ax)}
        for col, side_ax in sides.items():
            side_ax.set_xlim(0, plan.left_w if col == 0 else plan.right_w)
            side_ax.set_axis_off()
        swatch_h = _SWATCH_IN * h_px / plan.image_h
        for tile in roi.tiles:
            block = plan.blocks[(tile.row, tile.col)]
            cy = (tile.full_box[0] + tile.full_box[1]) / 2 - 0.5
            if tile.col == 0:
                side_ax, sw_x = sides[0], plan.left_w - _GAP_IN - 2 * _SWATCH_IN
                tx, ha = sw_x - _GAP_IN, "right"
            else:
                side_ax, sw_x = sides[1], _GAP_IN
                tx, ha = _GAP_IN + 2 * _SWATCH_IN + _GAP_IN, "left"
            _swatches(side_ax, sw_x, cy, swatch_h, tile, rectangle)
            side_ax.text(tx, cy, block.name, ha=ha, va="bottom", fontsize=_NAME_PT,
                         color=_BODY_TEXT)
            side_ax.text(tx, cy, block.delta_e, ha=ha, va="top", fontsize=_DE_PT,
                         family="monospace", color=block.colour)
    else:
        ordered = sorted(roi.tiles, key=lambda t: (t.col, t.row))
        key_ax = axes(x, img_bottom - plan.key_h, plan.width, plan.key_h)
        key_ax.set_xlim(0, plan.width)
        key_ax.set_ylim(plan.key_h, 0)
        key_ax.set_axis_off()
        for number, tile in enumerate(ordered, start=1):
            y0, y1, x0, x1 = tile.core_box
            ax.text((x0 + x1) / 2 - 0.5, (y0 + y1) / 2 - 0.5, str(number), ha="center",
                    va="center", fontsize=_NAME_PT, family="monospace", color=_BODY_TEXT,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.85))
            row, col = divmod(number - 1, plan.key_cols)
            ex = col * plan.key_entry_w
            ey = _GAP_IN + (row + 0.5) * plan.key_row_h
            block = plan.blocks[(tile.row, tile.col)]
            key_ax.text(ex + plan.number_w, ey, str(number), ha="right", va="center",
                        fontsize=_NAME_PT, family="monospace", color=_BODY_TEXT)
            sw_x = ex + plan.number_w + _GAP_IN
            _swatches(key_ax, sw_x, ey, _SWATCH_IN, tile, rectangle)
            tx = sw_x + 2 * _SWATCH_IN + _GAP_IN
            key_ax.text(tx, ey, block.name, ha="left", va="bottom", fontsize=_NAME_PT,
                        color=_BODY_TEXT)
            key_ax.text(tx, ey, block.delta_e, ha="left", va="top", fontsize=_DE_PT,
                        family="monospace", color=block.colour)
        bottom = img_bottom - plan.key_h

    if plan.notes:
        note_ax = axes(x, bottom - plan.note_h, plan.width, plan.note_h)
        note_ax.set_xlim(0, plan.width)
        note_ax.set_ylim(plan.note_h, 0)
        note_ax.set_axis_off()
        for i, (line, colour) in enumerate(plan.notes):
            note_ax.text(_EDGE_IN, _GAP_IN + i * plan.note_line_h, line, ha="left", va="top",
                         fontsize=_NOTE_PT, family="monospace", color=colour)
```

- [ ] **Step 4: Register the deferred imports.** In `tests/unit/ci/test_deferred_imports.py`, add this entry to `DEFERRED_SITES`, keeping alphabetical order by module path:

```python
    "correction/_color_correction/_calibration_overlay.py": {
        "FigureCanvasAgg": ("render_calibration_overlay",),
        "Figure": ("render_calibration_overlay",),
        "Rectangle": ("render_calibration_overlay",),
        "phenotypic_mpl_context": ("render_calibration_overlay",),
    },
```

  If the test's semantics refuse the `TYPE_CHECKING` import of `Figure` at module level, read the docstring at the top of `DEFERRED_SITES` and adjust the entry. Never move the import to module level.

- [ ] **Step 5: Run the tests.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction/test_calibration_overlay.py tests/unit/ci/test_deferred_imports.py tests/unit/ci/test_startup_imports.py -p no:cacheprovider -q -o addopts="" -n 4`
Expected: all pass.

  **If an overlap test fails,** fix the layout arithmetic. Do not loosen `assert_no_overlap_or_clipping`: its 0.5 px tolerance is the only slack, and it exists only to absorb Agg's sub-pixel rounding.

  **Render a PNG and look at it** before committing: `render_calibration_overlay(record).savefig("/tmp/overlay.png", dpi=160)`, using the two-band record from `planted_faults()`. It should match `docs/superpowers/artifacts/2026-09-22-checker-calibration-overlay/panel-b-matplotlib.png` in structure.

- [ ] **Step 6: Commit.**

```bash
git add src/phenotypic/correction/_color_correction/_calibration_overlay.py tests/unit/correction/test_calibration_overlay.py tests/unit/ci/test_deferred_imports.py
git commit -m "feat(color): render the calibration tile overlay with measured, overlap-free labels"
```

---

### Task 4: `show_tiles()` and documentation

**Files:**
- Modify: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` (`show_tiles`, class docstring)
- Modify: `src/phenotypic/correction/CLAUDE.md`
- Test: `tests/unit/correction/test_calibration_overlay.py` (append)

**Interfaces:**
- Consumes: `render_calibration_overlay` (Task 3) and `calibration_record` (Task 2).
- Produces: `CalibrateColorRpcc.show_tiles(*, figsize=None) -> Figure`.

- [ ] **Step 1: Write the failing tests** (append):

```python
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
```

- [ ] **Step 2: Run them to confirm they fail.** Expected: `AttributeError: ... has no attribute 'show_tiles'`.

- [ ] **Step 3: Add the method** after `calibration_record`. Import `render_calibration_overlay` alongside the Task 2 names, and add `from matplotlib.figure import Figure` under the file's `TYPE_CHECKING` block.

```python
    def show_tiles(self, *, figsize: tuple[float, float] | None = None) -> Figure:
        """Draw where the last ``apply()`` measured each tile and what it matched.

        One panel per ROI: the as-shot pixels, each detected tile outlined, the
        core box the medoid came from coloured by status (used, partly
        covered, rejected, excluded, empty), and beside each tile the chart
        patch it was matched to, a measured | reference swatch pair and ΔE00
        before -> after correction. A refused frame still draws, with its
        reasons under each ROI and no after-values.

        .. code-block:: python

            try:
                corrected = op.apply(plate)
            finally:
                op.show_tiles().savefig("calibration.png", dpi=160)

        Args:
            figsize: Optional ``(width, height)`` in inches; must be at least
                the size the labels need.

        Returns:
            A ``matplotlib.figure.Figure``.

        Raises:
            RuntimeError: If ``apply()`` has not run on this instance.
            ValueError: If *figsize* is too small for the labels.
        """
        record = self._calibration_record
        if record is None:
            raise RuntimeError(
                    "show_tiles() draws the last apply(); call apply() first."
            )
        return render_calibration_overlay(record, figsize=figsize)
```

  In the class docstring, add after the `Returns:` paragraph: `` ``calibration_record`` holds the tile overlay of the last run, even a refused one, and ``show_tiles()`` draws it. ``

- [ ] **Step 4: Document it** in `src/phenotypic/correction/CLAUDE.md`, as a new paragraph at the end of "Colour correction: which entry point", keeping CRLF:

```markdown
**Checking a calibration by eye:** after `apply()`, `op.show_tiles()` draws
each ROI's tiles, the chart patch each was matched to, and ΔE00 before and
after. It draws from `op.calibration_record`, plain data that is kept even
when the gate refuses the frame. `render_calibration_overlay(record)` in
`_calibration_overlay.py` reads nothing else, so a record persisted elsewhere
draws the same figure. Saving the record and CLI publication are not
implemented yet.
```

- [ ] **Step 5: Run the tests.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction/test_calibration_overlay.py tests/unit/correction/test_calibrate_color_rpcc.py -p no:cacheprovider -q -o addopts="" -n 4`
Expected: all pass.

- [ ] **Step 6: Commit.**

```bash
git add src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py src/phenotypic/correction/CLAUDE.md tests/unit/correction/test_calibration_overlay.py
git commit -m "feat(color): CalibrateColorRpcc.show_tiles() draws the last calibration"
```

---

### Task 5: Gate

- [ ] **Affected surface, once:**
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction tests/unit/ci tests/unit/tune/test_annotation_coverage.py tests/smoke/test_operation.py tests/smoke/test_serialization.py -p no:cacheprovider -q -o addopts="" -n 4`
  All must pass. The only exceptions are pre-existing failures, and each of those must be confirmed by running it alone on `origin/main`.
- [ ] **Serialisation is unchanged:** `CalibrateColorRpcc.model_validate(op.model_dump())` after an `apply()` must not carry the record or the crops, because the record is a `PrivateAttr`. `tests/smoke/test_serialization.py` covers the class; confirm it passes.
- [ ] **Lint and types:** ruff on every changed path, and `uv run mypy src/phenotypic/correction/_color_correction/`. There must be no new errors; record the count before Task 2.
- [ ] **Line endings:** for each changed file, `grep -c $'\r'` equals `wc -l`, or the file was LF before and still is.
- [ ] **Look at the figure:** render the two-band planted-faults record and the 4 × 6 card record to PNG, and check both by eye against the approved mockup.
- [ ] **Review:** dispatch `xander-local:implementation-test-reviewer` on the diff since `639c5cfec`, with its report written to `docs/superpowers/reports/2026-09-22-checker-calibration-overlay/implementation-review.md`. Fix confirmed findings, then re-run the affected surface.
