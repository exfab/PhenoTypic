# CalibrateColorRpcc review fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the eleven verified findings from the 2026-09-21 code review of `CalibrateColorRpcc`, turning every test in `tests/unit/correction/test_calibrate_color_rpcc_review.py` green without regressing the existing correction suite.

**Architecture:** No new modules. Most of the change is in the orchestrator, `_calibrate_color_rpcc.py`. `_operate` is restructured so that every per-ROI failure becomes a `QcRecord` flag and goes through `on_qc_fail`, and every piece of per-run state is reset on entry. Three stage modules also get targeted fixes: `_checker_identity` learns to ignore non-finite tiles, `_checker_roi` and `_checker_detect` agree on one rotation convention and pivot, and `_checker_qc` gains a rank check that can run after outlier rejection.

**Tech Stack:** Python 3.11+, pydantic v2, numpy, colour-science, scipy, OpenCV (ECC), pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-21-in-frame-checker-color-correction/README.md` (the operation interface, around line 498; the missing-tile warnings table, around line 390).
**Findings:** `docs/superpowers/reports/2026-09-21-in-frame-checker-color-correction/code-review.md`. It records the verified mechanism for each finding, which differs from the reviewer's account for #1 and #2.
**Failing tests (already written):** `tests/unit/correction/test_calibrate_color_rpcc_review.py`. At the start there are 14 failing and 2 passing: the fixture control and the `min_patches` pin.

## Global Constraints

- Run everything with `uv run`. Test command for this plan:
  `QT_QPA_PLATFORM=offscreen uv run pytest <paths> -p no:cacheprovider -q -o addopts="" -n 4`.
  Never use `-x`. The full `tests/unit` suite is not run in this plan (it is about 65 minutes and runs through Slurm); see Task 7.
- **The existing source files use CRLF line endings** (`_calibrate_color_rpcc.py`, `_checker_*.py`, `test_calibrate_color_rpcc.py`). Preserve them. After editing, check with `grep -c $'\r' <file>` against `wc -l <file>`; the two counts must match.
- Operations are keyword-only pydantic v2 models, `extra="forbid"`, `validate_assignment=True`. Model validators therefore re-run on every `self.x = ...` in `_operate`, so they must stay cheap.
- **Degree is fixed.** Rank insufficiency raises whatever `on_qc_fail` says. Everything else about a short card warns.
- **Identity is decided by whole-placement scoring.** The Hungarian count corroborates it and never decides it.
- **All Lab comes from `Image.color.Lab`** semantics. An ROI view must convert exactly as the full image would.
- `uv run ruff check <explicit paths>` only. Never run it bare.
- Do not edit the review test file's assertions to make a test pass. If one looks wrong, stop and report it. The only sanctioned test edits are the `test_checker_qc.py` call-site updates in Task 2.

## File structure

| File | Change | Tasks |
|---|---|---|
| `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` | fields, validator, `_operate` restructure, census, anchor, ROI view | 1, 2, 3, 4, 5, 6 |
| `src/phenotypic/correction/_color_correction/_checker_qc.py` | `ROOT_POLYNOMIAL_TERMS`, `require_rank`, `min_patches` moved off `QcLimits`, `empty_tiles` signal | 2, 4 |
| `src/phenotypic/correction/_color_correction/_checker_identity.py` | non-finite tiles excluded from scoring and Hungarian | 4 |
| `src/phenotypic/correction/_color_correction/_checker_roi.py` | rotation sense matches OpenCV | 5 |
| `src/phenotypic/correction/_color_correction/_checker_detect.py` | ECC translation re-pivoted to the lattice centroid | 5 |
| `tests/unit/correction/test_checker_qc.py` | `warn_on_patch_census` call sites | 2 |
| `tests/unit/correction/test_checker_identity.py`, `test_checker_roi.py`, `test_checker_detect.py` | one new unit test each | 4, 5 |

**DAG:** 1 → 2 → 3 → 4 → 5 → 6 → 7. The tasks are strictly sequential: Tasks 1–6 all edit `_calibrate_color_rpcc.py`, and Task 4 relies on Task 3's restructured loop. Do not parallelise them. Each task is a Keystone-shaped edit to shared files.

---

### Task 1: Per-ROI inputs survive serialisation and are validated at construction (findings 5, 10)

**Files:**
- Modify: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` (imports, the `reference_bands` field at line 137, and a new validator after `_validate_degree`)
- Test: `tests/unit/correction/test_calibrate_color_rpcc_review.py` (already written)

**Interfaces:**
- Produces: `CalibrateColorRpcc.reference_bands: list[np.ndarray] | None`, serialised as nested lists; construction raises `ValueError` naming the field when a per-ROI list length differs from `len(rois)`, or when `refine_method="ecc"` has a prior but no bands.

**Decision to confirm with the user before starting:** the spec declares `reference_bands: list[NdArrayField]`. That puts the band pixels into every `to_json()`, and so into every per-image provenance journal. A real band (2150 × 340 × 3 float64) comes to about 2.2 M numbers, roughly 40 MB of JSON per band. This task implements the spec as written. If that size is unacceptable, stop and choose between storing a path to a `.npy` file and storing only the single-channel float32 registration image (`_registration_image` output), which is about 3× smaller.

- [ ] **Step 1: Confirm the tests fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction/test_calibrate_color_rpcc_review.py -p no:cacheprovider -q -o addopts="" -k "reference_bands or construction"`
Expected: 4 FAILED (`assert None is not None` ×2, `DID NOT RAISE` ×2).

- [ ] **Step 2: Make `reference_bands` a serialised array list**

In the imports:

```python
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ...abc_ import ImageCorrector
from ...sdk_.typing_ import NdArrayField, TuneSpec
```

Replace the field:

```python
    reference_bands: list[NdArrayField] | None = None
```

- [ ] **Step 3: Add the per-ROI validator** directly after `_validate_degree`

```python
    @model_validator(mode="after")
    def _validate_per_roi_inputs(self) -> CalibrateColorRpcc:
        """One prior and one reference band per ROI, in ROI order.

        Checked here so that a short list fails when the operation is built,
        naming the field, rather than deep inside ``_operate`` as an
        ``IndexError`` on whichever image happened to be first.
        """
        n_rois = len(self.rois)
        for name in ("lattice_prior", "reference_bands"):
            value = getattr(self, name)
            if value is not None and len(value) != n_rois:
                raise ValueError(
                        f"{name} has {len(value)} entries but there are {n_rois} "
                        "rois; supply exactly one per ROI, in the same order."
                )
        if (
                self.refine_method == "ecc"
                and self.lattice_prior is not None
                and self.reference_bands is None
        ):
            raise ValueError(
                    "refine_method='ecc' needs reference_bands, the ROI bands the "
                    "lattice_prior was fitted on. Use refine_method='rigid' if no "
                    "reference band is stored."
            )
        return self
```

- [ ] **Step 4: Run the tests and the existing operation tests**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction/test_calibrate_color_rpcc_review.py tests/unit/correction/test_calibrate_color_rpcc.py tests/smoke/test_serialization.py -p no:cacheprovider -q -o addopts="" -n 4 -k "reference_bands or construction or CalibrateColorRpcc or rois or degree or serialis"`
Expected: the 4 target tests PASS, and nothing that passed before now fails.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py
git commit -m "fix(color): serialise reference_bands and validate per-ROI list lengths"
```

---

### Task 2: One `min_patches`, and a rank check on the patches actually fitted (findings 3, 9)

**Files:**
- Modify: `src/phenotypic/correction/_color_correction/_checker_qc.py` (remove `QcLimits.min_patches`; add `ROOT_POLYNOMIAL_TERMS` and `require_rank`; change the `warn_on_patch_census` signature)
- Modify: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py:339-354`
- Modify: `tests/unit/correction/test_checker_qc.py` (lines 102, 107, 115, 120, 126, 135 and 140: `QcLimits()` → `20`)

**Interfaces:**
- Produces: `ROOT_POLYNOMIAL_TERMS: dict[int, int]`; `require_rank(n_patches: int, degree: int, stage: str = "were accepted") -> None`, which raises `ValueError` containing `"A degree-{d} root-polynomial fit needs at least {terms}"`; `warn_on_patch_census(accepted, expected, degree, min_patches: int) -> list[str]`. `QcLimits` no longer has `min_patches` (`extra="forbid"` makes passing it an error).

- [ ] **Step 1: Confirm the tests fail**

Run: `... -k "rank_is_checked or min_patches_has_one_home"`
Expected: 2 FAILED (`a degree-4 fit ran on 21 patches for 22 terms`; `DID NOT RAISE`).

- [ ] **Step 2: In `_checker_qc.py`, delete `min_patches: int = 20` from `QcLimits`** along with its `Args:` line.

- [ ] **Step 3: Add the rank helper** above `warn_on_patch_census`

```python
#: Columns in the root-polynomial expansion of each degree (Finlayson et al.
#: 2015). A fit with fewer patches than terms has no unique solution.
ROOT_POLYNOMIAL_TERMS: dict[int, int] = {1: 3, 2: 6, 3: 13, 4: 22}


def require_rank(n_patches: int, degree: int, stage: str = "were accepted") -> None:
    """Refuse a fit with fewer patches than the expansion has terms.

    Called twice: on the accepted count before fitting, and on the count
    that survives outlier rejection, because rejection can take a
    rank-sufficient set below the line.

    Args:
        n_patches: Patches that would enter the solve.
        degree: The configured polynomial degree.
        stage: How the count was arrived at, for the message.

    Raises:
        ValueError: If *n_patches* is below the term count for *degree*.
    """
    terms = ROOT_POLYNOMIAL_TERMS.get(degree)
    if terms is not None and n_patches < terms:
        raise ValueError(
                f"A degree-{degree} root-polynomial fit needs at least {terms} "
                f"patches but only {n_patches} {stage}. The fit would have no "
                "unique solution, and its minimum-norm answer reports a "
                "near-zero in-sample residual that is an artifact of exact "
                "interpolation rather than accuracy. Re-shoot the card, or "
                "configure a lower degree for the whole batch."
        )
```

- [ ] **Step 4: Rewire `warn_on_patch_census`**. Change the signature to `(accepted, expected, degree, min_patches: int)`. Replace its `Args:` entry `limits: Supplies min_patches.` with `min_patches: Accepted patches below which a worse fit is warned about.`. Replace the inline `terms` block with `require_rank(len(accepted), degree)`, and replace `limits.min_patches` with `min_patches` in the two places it appears.

- [ ] **Step 5: Update the seven call sites in `tests/unit/correction/test_checker_qc.py`**. Replace the fourth positional argument `QcLimits()` with `20` at lines 102, 107, 115, 120, 126, 135 and 140. Leave line 17 (`limits=QcLimits()` passed to `evaluate_roi`) alone.

- [ ] **Step 6: In `_calibrate_color_rpcc.py`, use the single knob and re-check after rejection**. Update the import:

```python
from ._checker_qc import (
    QcLimits, QcRecord, evaluate_roi, require_rank, warn_on_patch_census,
)
```

Replace the census call and the profile block (currently lines 340–354) with:

```python
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
```

- [ ] **Step 7: Run the QC and review tests**

Run: `... tests/unit/correction/test_checker_qc.py tests/unit/correction/test_calibrate_color_rpcc_review.py -k "qc or rank or min_patches"`
Expected: `test_rank_is_checked_after_outlier_rejection`, `test_min_patches_has_one_home` and `test_the_operation_min_patches_is_honoured` PASS, and every `test_checker_qc.py` test PASSES.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/correction/_color_correction/_checker_qc.py src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py tests/unit/correction/test_checker_qc.py
git commit -m "fix(color): check rank after outlier rejection; give min_patches one home"
```

---

### Task 3: Every per-ROI failure goes through `on_qc_fail`; per-run state resets; census survives (findings 7, 8, 4)

**Files:**
- Modify: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py`: `_operate`, `_build_diagnostics`, `patch_census`, and the class docstring's `Raises:` section

**Interfaces:**
- Consumes: `require_rank` and the new `warn_on_patch_census` signature (Task 2).
- Produces: a `_operate` loop in which each ROI either `continue`s with a flag-only `QcRecord`, or appends a full record with measured tiles. `_build_diagnostics(chart_illuminant, census, patch_names, tiles, lattices, accepted, skipped=False)`, where *accepted* is now explicit. Task 4 adds to this loop.

- [ ] **Step 1: Confirm the tests fail**

Run: `... -k "skipped_under_skip or clears_the_previous or patch_census"`
Expected: 4 FAILED.

- [ ] **Step 2: Reset per-run state at the top of `_operate`**, before `_load_reference_data`:

```python
        # One instance may process many images; nothing from the last one may
        # survive into this one's result, least of all on the skip path.
        self.fitted_profile = None
        self.qc = []
        self._diagnostics = {}
```

- [ ] **Step 3: Route the three detection failures into flag-only records**. Replace the loop body from `lab, srgb = ...` down to `identity = assign_placement(...)` with:

```python
            lab, srgb = self._roi_views(image, roi)

            if self.lattice_prior is not None:
                reference = (
                    self.reference_bands[index]
                    if self.reference_bands is not None
                    else None
                )
                refined = refine(
                        lab, self.lattice_prior[index], method=self.refine_method,
                        reference_lab=reference, anchor_col=roi.anchor_col,
                )
                lattice, shift = refined.lattice, float(np.hypot(refined.dy, refined.dx))
                ecc = refined.confidence if self.refine_method == "ecc" else None
            else:
                try:
                    lattice = fit_lattice(lab, grid=self.grid)
                except ValueError as exc:
                    # No card in the rectangle is a property of this frame,
                    # not of the configuration: the policy decides.
                    records.append(self._refusal(index, roi, f"lattice not found: {exc}"))
                    continue
                shift, ecc = 0.0, None
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
```

Add the helper next to `_anchor_disagreement`:

```python
    @staticmethod
    def _refusal(index: int, roi: CheckerRoi, *flags: str) -> QcRecord:
        """A record for an ROI that failed before it could be measured."""
        return QcRecord(roi_index=index, label=roi.label, flags=list(flags))
```

The rest of the loop body (`record = evaluate_roi(...)` onward) is unchanged. The existing test `test_it_refuses_an_roi_that_holds_no_card` matches `"No patch columns|too short|cannot sit"`. It still passes, because under `on_qc_fail="raise"` the gate's summary includes the flag text, which quotes the `fit_lattice` message.

- [ ] **Step 4: Make *accepted* explicit in diagnostics**. Change `_build_diagnostics` to take `accepted: list[str]` after `lattices`, and set `"accepted": accepted` in `patch_census` (instead of `[t["patch"] for t in tiles]`). In the skip branch, pass `accepted=[]`. In the fitted path, pass `accepted=accepted` (the `list(measured.keys())` already computed).

- [ ] **Step 5: Make `patch_census` catch what `apply()` actually raises**. Replace the `try/except` with:

```python
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
```

Add at module level, below `OnQcFail`:

```python
def _root_cause(exc: BaseException) -> BaseException:
    """The innermost exception in a ``raise ... from`` chain."""
    while exc.__cause__ is not None:
        exc = exc.__cause__
    return exc
```

- [ ] **Step 6: Correct the class docstring's `Raises:`**

```python
    Raises:
        ValueError: If any ROI fails the gate under ``on_qc_fail="raise"``,
            if the patches that reach the fit cannot support ``degree`` (under
            any policy), or if two ROIs claim the same chart patch.
            ``apply()`` re-raises every failure as ``RuntimeError`` with the
            ``ValueError`` as its root cause.
```

- [ ] **Step 7: Run the tests**

Run: `... tests/unit/correction/test_calibrate_color_rpcc_review.py tests/unit/correction/test_calibrate_color_rpcc.py`
Expected: the 4 target tests plus `test_it_refuses_an_roi_that_holds_no_card` PASS, and every test that passed at the end of Task 2 still passes.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py
git commit -m "fix(color): route detection failures through on_qc_fail; reset per-run state"
```

---

### Task 4: Tiles that measured nothing never reach identity scoring or the fit (finding 1)

**Files:**
- Modify: `src/phenotypic/correction/_color_correction/_checker_identity.py` (`assign_placement`, `_hungarian_agreement`)
- Modify: `src/phenotypic/correction/_color_correction/_checker_qc.py` (`evaluate_roi` gains `empty_tiles`)
- Modify: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` (the Task 3 loop)
- Test: `tests/unit/correction/test_checker_identity.py` (one new test)

**Interfaces:**
- Consumes: the Task 3 loop.
- Produces: `assign_placement` accepts NaN rows in `observed_linear`. They are excluded from voting, from luminance normalisation on both sides, and from the Hungarian count, and `n_tiles` counts only finite voting tiles. `evaluate_roi(..., empty_tiles: int = 0)` flags a non-zero count and records `signals["empty_tiles"]`.

- [ ] **Step 1: Write the identity unit test** in `test_checker_identity.py`, after `test_a_blank_card_is_refused_rather_than_guessed_at`:

```python
def test_a_missing_tile_is_left_out_rather_than_poisoning_the_score(cards) -> None:
    """A NaN tile (a box outside the ROI) must not vote or normalise.

    Luminance is normalised by the block's brightest tile, so one NaN used to
    turn every feature NaN and crash the Hungarian corroboration.
    """
    block = cards["blocks"][0].astype(float).copy()
    candidates = placements(chart_grid(cards["names"]), block.shape[:2])
    clean = assign_placement(block, candidates, cards["ref_linear"])

    block[-1, :] = np.nan
    result = assign_placement(block, candidates, cards["ref_linear"])

    assert np.isfinite(result.margin)
    assert result.placement == clean.placement
    assert result.n_tiles == block.shape[0] * block.shape[1] - block.shape[1]
```

- [ ] **Step 2: Run it and the review test to confirm they fail**

Run: `... tests/unit/correction/test_checker_identity.py tests/unit/correction/test_calibrate_color_rpcc_review.py -k "missing_tile or empty_tile"`
Expected: 2 FAILED with `matrix contains invalid numeric entries`.

- [ ] **Step 3: Exclude non-finite rows in `assign_placement`**. Replace the block from `flat_observed = observed.reshape(-1, 3)` through the construction of `mask`, and the per-placement feature and distance computation, with:

```python
    flat_observed = observed.reshape(-1, 3)
    # A tile whose box missed the ROI measured nothing. Excluding it only
    # from the vote is not enough: luminance is normalised by the block's
    # brightest tile, so a NaN anywhere would make every feature NaN.
    finite = np.isfinite(flat_observed).all(axis=1)
    obs_features = np.full_like(flat_observed, np.nan)
    obs_features[finite] = identity_features(flat_observed[finite])

    mask = (
        np.ones(n_rows * n_cols, dtype=bool)
        if voting is None
        else np.asarray(voting, dtype=bool).reshape(-1)
    ) & finite
    if not mask.any():
        raise ValueError("No tiles are allowed to vote.")
```

Inside the placement loop, after `reference = np.vstack(...)`:

```python
        ref_features = np.full_like(reference, np.nan)
        ref_features[finite] = identity_features(reference[finite])
        distance = np.linalg.norm(
                obs_features[mask] - ref_features[mask], axis=1
        ).mean()
```

When every tile is finite this is identical to the current code, so none of the existing identity tests can move. Pass `finite` to the Hungarian check: `_hungarian_agreement(obs_features, best, reference_linear, finite)`.

- [ ] **Step 4: Restrict `_hungarian_agreement` to the finite tiles**

```python
def _hungarian_agreement(
        obs_features: np.ndarray,
        placement: Placement,
        reference_linear: Mapping[str, np.ndarray],
        rows: np.ndarray,
) -> int:
    """Tiles a free assignment labels the same way as *placement*.

    Corroboration only.  This must never decide identity: a free permutation
    relabels an occluded tile onto whatever reference happens to fit it.
    Only tiles flagged in *rows* take part.
    """
    from scipy.optimize import linear_sum_assignment

    names = [name for row in placement.names for name in row]
    names = [name for name, keep in zip(names, rows) if keep]
    reference = np.vstack([reference_linear[name] for name in names])
    ref_features = identity_features(reference)
    cost = np.linalg.norm(
            obs_features[rows][:, None, :] - ref_features[None, :, :], axis=2
    )
    assigned_rows, assigned_cols = linear_sum_assignment(cost)
    return int(sum(1 for r, c in zip(assigned_rows, assigned_cols) if r == c))
```

- [ ] **Step 5: Add `empty_tiles` to `evaluate_roi`**. Add the parameter `empty_tiles: int = 0` after `clipped`, and the `Args:` entry `empty_tiles: Tiles whose box fell outside the ROI and measured nothing.`. Add this check as the first one, before `shift_px`:

```python
    if empty_tiles:
        flags.append(
                f"{empty_tiles} tile box(es) fall outside the ROI and measured "
                "nothing; the lattice does not fit this rectangle"
        )
```

Also add `"empty_tiles": int(empty_tiles),` to `signals`.

- [ ] **Step 6: In the operation loop, keep empty tiles out of the fit**. Replace the `observed` construction from Task 3 with:

```python
            observed = np.full((lattice.nrows, len(lattice.columns), 3), np.nan)
            for tile in tiles:
                if tile.n_pixels:
                    observed[tile.row, tile.col] = colour.cctf_decoding(
                            np.clip(tile.srgb, 0, 1), function="sRGB"
                    )
            empty = sum(1 for tile in tiles if not tile.n_pixels)
            if empty == len(tiles):
                records.append(self._refusal(
                        index, roi, "every tile box falls outside the ROI"
                ))
                continue
```

Pass `empty_tiles=empty,` to `evaluate_roi(...)`. In the loop that fills `measured`, skip empty tiles: put `if not tile.n_pixels: continue` as its first line. The diagnostics `tiles_out` still lists them, with NaN values, so the census shows what was lost.

- [ ] **Step 7: Run the tests**

Run: `... tests/unit/correction/test_checker_identity.py tests/unit/correction/test_checker_qc.py tests/unit/correction/test_calibrate_color_rpcc_review.py`
Expected: the 2 target tests PASS, every identity and QC test still PASSES, and nothing that passed after Task 3 now fails.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/correction/_color_correction/_checker_identity.py src/phenotypic/correction/_color_correction/_checker_qc.py src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py tests/unit/correction/test_checker_identity.py
git commit -m "fix(color): keep tiles that measured nothing out of identity and the fit"
```

---

### Task 5: The rotation ECC recovers is applied, in ECC's own sense and pivot (finding 2)

**Files:**
- Modify: `src/phenotypic/correction/_color_correction/_checker_roi.py` (the rotation block in `CheckerLattice.boxes`, plus the `rot` docstrings)
- Modify: `src/phenotypic/correction/_color_correction/_checker_detect.py` (`refine_ecc`, after `findTransformECC`)
- Modify: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py:206` (`_measure_roi`)
- Test: `tests/unit/correction/test_checker_roi.py`, `tests/unit/correction/test_checker_detect.py` (one new test each)

**Interfaces:**
- Produces: `CheckerLattice.rot` has one meaning everywhere: radians about the lattice centroid, positive turning +x toward +y (OpenCV's sense). `refine_ecc` returns `dy`/`dx` as the displacement **of the lattice centroid**, and a lattice whose `boxes(rot=lattice.rot)` land on the rotated tiles.

**Why both halves:** ECC maps a reference point *p* to *R p + t*, pivoting on the ROI origin. `boxes()` pivots on the lattice centroid *c*, and its current formula turns the other way. The correct translation is where the warp sends the centroid, *R c + t − c*. This also stops a pure rotation from being reported to the gate as a ~24 px displacement.

- [ ] **Step 1: Write the direction test** in `test_checker_roi.py`

```python
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
```

(Both tiles start at y centre 5. After rotation the right tile's centre is 45·sin 0.1 ≈ 4.5 px lower.)

- [ ] **Step 2: Write the ECC placement test** in `test_checker_detect.py`, after `test_ecc_refinement_recovers_displacement_when_opencv_is_present`

```python
def test_ecc_refinement_places_boxes_on_a_rotated_band() -> None:
    """Every refined box on a 4-degree-rotated band covers its own patch.

    Self-checking: each box's median colour in the rotated band must match
    the same tile's colour in the unrotated band.
    """
    from scipy.ndimage import rotate

    band = synthetic_band(noise=0.0)
    turned = rotate(band, 4.0, reshape=False, order=1, mode="nearest")
    prior = reference_lattice()

    result = refine(turned, prior, method="ecc", reference_lab=band)

    before = prior.boxes(core=0.4)
    after = result.lattice.boxes(core=0.4, rot=result.lattice.rot)
    for (_, _, *b0), (_, _, *b1) in zip(before, after):
        want = np.median(band[int(b0[0]):int(b0[1]), int(b0[2]):int(b0[3])].reshape(-1, 3), axis=0)
        got = np.median(turned[int(b1[0]):int(b1[1]), int(b1[2]):int(b1[3])].reshape(-1, 3), axis=0)
        assert np.linalg.norm(got - want) < 2.0
    assert np.hypot(result.dy, result.dx) < 3.0  # rotated about its own centre
```

- [ ] **Step 3: Run the three tests to confirm they fail**

Run: `... tests/unit/correction/test_checker_roi.py tests/unit/correction/test_checker_detect.py tests/unit/correction/test_calibrate_color_rpcc_review.py -k "opencv or rotated"`
Expected: 3 FAILED. The review test shows `tile (1, 3, 0) read 63.1 dE00 off`.

- [ ] **Step 4: Flip the sense in `CheckerLattice.boxes`**. Replace the two lines computing `ny`/`nx`:

```python
                # OpenCV's sense (image y points down): positive rot turns
                # +x toward +y, matching the angle refine_ecc recovers.
                ny = mx * sin_r + my * cos_r + cy
                nx = mx * cos_r - my * sin_r + cx
```

In both the `CheckerLattice` class docstring (`rot:`) and `boxes` (`rot:`), change the description to: `Rotation in radians about the lattice centroid; positive turns +x toward +y, as OpenCV's warps do.`

- [ ] **Step 5: Re-pivot ECC's translation in `refine_ecc`**. Replace everything from `dy, dx = float(warp[1, 2]), float(warp[0, 2])` to the `return`:

```python
    rot = float(np.arctan2(warp[1, 0], warp[0, 0]))
    # ECC maps a reference point p to R p + t, pivoting on the ROI origin;
    # the lattice rotates about its own centroid. Its translation is
    # therefore where the warp sends that centroid, not t -- otherwise a
    # pure rotation is reported as a displacement and the boxes land off
    # their tiles.
    cy, cx = np.mean(prior.centers(), axis=0)
    moved_x = warp[0, 0] * cx + warp[0, 1] * cy + warp[0, 2]
    moved_y = warp[1, 0] * cx + warp[1, 1] * cy + warp[1, 2]
    dy, dx = float(moved_y - cy), float(moved_x - cx)
    lattice = prior.translated(dy=dy, dx=dx)
    lattice = lattice.model_copy(update={"rot": prior.rot + rot})
    return RefineResult(lattice, dy, dx, rot, float(correlation), "ecc")
```

- [ ] **Step 6: Apply the rotation when measuring**. In `_measure_roi`, replace `lattice.boxes(core=self.core_trim)` with `lattice.boxes(core=self.core_trim, rot=lattice.rot)`.

- [ ] **Step 7: Run the tests**

Run: `... tests/unit/correction/test_checker_roi.py tests/unit/correction/test_checker_detect.py tests/unit/correction/test_calibrate_color_rpcc_review.py`
Expected: the 3 target tests PASS, and `test_ecc_refinement_recovers_displacement_when_opencv_is_present` (a pure translation, rot ≈ 0) still PASSES.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/correction/_color_correction/_checker_roi.py src/phenotypic/correction/_color_correction/_checker_detect.py src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py tests/unit/correction/test_checker_roi.py tests/unit/correction/test_checker_detect.py
git commit -m "fix(color): apply ECC rotation to tile boxes about the lattice centroid"
```

---

### Task 6: Only unclipped columns vote on disagreement; ROI Lab keeps the image's illuminant (findings 6, 11)

**Files:**
- Modify: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py` (`_anchor_disagreement`, `_roi_views`)

- [ ] **Step 1: Confirm the tests fail**

Run: `... -k "clipped_column or illuminant"`
Expected: 2 FAILED (`refused at disagreement 12.5 px`; `Mismatched elements`).

- [ ] **Step 2: Exclude clipped columns in `_anchor_disagreement`**

```python
    def _anchor_disagreement(
            self, lab, roi_index: int, lattice: CheckerLattice
    ) -> float | None:
        """Spread between the shifts different anchor columns imply.

        A reference-free internal consistency check: the columns of one rigid
        card must agree about where it moved.  Only columns lying wholly
        inside the ROI on this frame vote -- a column the border clips
        tracks the shift at about half rate, and letting it vote refuses an
        in-range move as an inconsistency.  ``None`` when fewer than two
        columns qualify, or when there is no prior to refine.
        """
        from ._checker_detect import refine_rigid

        if self.lattice_prior is None:
            return None
        width = lab.shape[1]
        inside = [
            index for index, column in enumerate(lattice.columns)
            if column.x0 >= 0 and column.x1 <= width
        ]
        if len(inside) < 2:
            return None
        prior = self.lattice_prior[roi_index]
        estimates = [
            refine_rigid(lab, prior, anchor_col=index).dx for index in inside
        ]
        return float(max(estimates) - min(estimates))
```

- [ ] **Step 3: Carry the illuminant into the ROI view**. In `_roi_views`:

```python
        wrapped = _Image(
                arr=np.ascontiguousarray(sub),
                gamma=image.gamma,
                illuminant=image.illuminant,
        )
```

- [ ] **Step 4: Run the whole review file and the existing operation tests**

Run: `... tests/unit/correction/test_calibrate_color_rpcc_review.py tests/unit/correction/test_calibrate_color_rpcc.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py
git commit -m "fix(color): ignore clipped columns in anchor disagreement; keep ROI illuminant"
```

---

### Task 7: Gate — affected surface, lint, types, review

**Files:** none new. Fixes only if a check fails.

- [ ] **Step 1: Run the affected surface once.** That is every test that imports the changed modules: the correction directory, plus the two smoke files the branch already edits.

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction tests/smoke/test_operation.py tests/smoke/test_serialization.py -p no:cacheprovider -q -o addopts="" -n 4`
Expected: all PASS. If a failure appears outside the review file, run that test alone before attributing it to this change.

- [ ] **Step 2: Lint and type-check the changed paths**

```bash
uv run ruff check src/phenotypic/correction/_color_correction/ tests/unit/correction/test_calibrate_color_rpcc_review.py tests/unit/correction/test_checker_qc.py tests/unit/correction/test_checker_identity.py tests/unit/correction/test_checker_roi.py tests/unit/correction/test_checker_detect.py
uv run mypy src/phenotypic/correction/_color_correction/
```

Expected: no new errors relative to the start of the plan. Record the starting mypy count in Step 0 of Task 1 if you want an exact comparison.

- [ ] **Step 3: Check line endings** for every edited CRLF file: `for f in <files>; do echo $f $(grep -c $'\r' $f) $(wc -l < $f); done`. The two numbers must match on every line.

- [ ] **Step 4: Mutation spot-check.** Revert each of these one-liners in turn, run its test, confirm it fails, then restore the line:
  - `rot=lattice.rot` in `_measure_roi` (Task 5) → `test_a_rotated_card_is_measured_on_rotated_boxes`
  - `& finite` in `assign_placement` (Task 4) → `test_a_missing_tile_is_left_out_rather_than_poisoning_the_score`
  - the post-rejection `require_rank` call (Task 2) → `test_rank_is_checked_after_outlier_rejection`

- [ ] **Step 5: Code review.** Dispatch `xander-local:implementation-test-reviewer` on the diff since the plan started. It writes its report to `docs/superpowers/reports/2026-09-21-in-frame-checker-color-correction/review-fixes-test-review.md`. Address any confirmed finding, then rerun Step 1.

- [ ] **Step 6: Full regression.** This is not run here. The full `tests/unit` suite belongs to the branch's own Task 7 acceptance run (Slurm, `run_unit_suite.sbatch`), after these fixes are committed.

## Out of scope, by decision

- **The `insufficient_patches` QC flag** the spec calls for (§Missing-tile warnings), which would make a card below `min_patches` participate in `on_qc_fail`. Today it only warns. Adding the flag changes which frames get refused, so it needs its own decision.
- **"Two ROIs both identified a patch"** still raises outside the policy. It means the ROI rectangles overlap, which is a configuration error rather than a property of the frame.
- **Bootstrap `fit_lattice` on a low-contrast synthetic column.** This was noticed while building the fixture. It is not a review finding.
