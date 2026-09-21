# Review-fixes test review — `CalibrateColorRpcc` (`df868123d..HEAD`, 8 commits)

**Reviewer:** implementation-test-reviewer, 2026-09-21. Worktree
`.worktrees/calibrate-color-rpcc`, branch `feat/calibrate-color-rpcc`.

**Scope:** the eight fix commits against the findings in `code-review.md`, the
plan `review-fixes.md`, and the spec README. All `file:line` references are to
HEAD. Line numbers are counted after stripping CR, because the files use CRLF.

**What I ran:**
- The focused suites (`test_calibrate_color_rpcc_review.py`,
  `test_calibrate_color_rpcc.py`, `test_checker_{identity,qc,roi,detect}.py`):
  111 passed.
- Ad-hoc probes through `uv run python -` against the review-test fixtures
  (`render_frame`, `frozen_op`, `band_prior`). These probes wrote no files.
- Mutation checks done by monkeypatching in-process. No source was edited.

**Verdict:** the eleven original findings are fixed as the plan describes. The
maths of the ECC re-pivot is correct, including when the prior already carries a
non-zero `rot`. Two problems remain that matter:

1. The anchor-disagreement fix turns the check off on the production geometry.
   The test that guards it passes *because* the check is off.
2. A gate-flagged ROI can still escape `on_qc_fail="skip"` through the
   duplicate-patch check.

The census `accepted` list also still disagrees with the spec after outlier
rejection.

Severity counts: **high 0, medium 3, low 7**. Of the ten, 9 are confirmed (by running, or by reading plus a test-suite grep) and 1 is a suspicion.

---

## Confirmed findings

### M1 (medium): the "wholly inside" filter disables anchor disagreement on 2-column cards, and its guard test passes vacuously

`_calibrate_color_rpcc.py:446-452` (`_anchor_disagreement`) and
`tests/unit/correction/test_calibrate_color_rpcc_review.py:226-246`.

**Mechanism.** The rig's cards are 6x2 half-cards, which gives two lattice
columns. The filter keeps only columns whose *refined* `x0 >= 0 and
x1 <= width`, and returns `None` when fewer than two qualify. On a 2-column card,
clipping either column switches the check off entirely.

The filter is also evaluated on the refined lattice, which is the output of the
refinement being checked. A refinement that shifts the lattice toward an edge
therefore pushes a column out of the vote and silences its own consistency
check.

**Confirmed by running:**
- **The review test is vacuous.** For
  `test_a_clipped_column_does_not_refuse_an_in_range_shift`, `qc[0].signals`
  shows `anchor_disagreement_px = None`, `shift_px = 25.18`, and no flags. The
  test asserts only that there is no "rigid card cannot" flag, so an
  implementation that always returns `None` passes it. The earlier mutation
  claim ("reverting the clipped-column filter makes its guard fail") is true,
  but it does not show that the fix is correct.
- **A real 2-column inconsistency goes undetected.** Prior
  `col_x=(0, 100)`, ROI width 180. The frame is drawn with the columns moved by
  different amounts:

  | Column moves (true) | Per-column `refine_rigid` dx | `anchor_disagreement_px` | Flags |
  |---|---|---|---|
  | (−5, −20) | (−2.5, −20.0) | `None` | none |
  | (−2, −30) | (−2.0, −28.0) | `None` | none |

  In the second row the columns disagree by 26 px against a 12 px limit, and
  nothing is flagged. Before the fix both columns voted and the frame would
  have been refused.

**Why it matters.** The spec's reference-free gate ("spread between refinement
methods > 12 px → refuse") is silently gone whenever a card moves toward the
frame edge. That is also the direction the spec calls the likely one ("the
frame-clipped outer column").

**Fix options:**
- For a column that touches the ROI edge, estimate its shift from its
  *unclipped* edge (the inner `x1` for a left-clipped column), which tracks the
  shift at full rate, instead of dropping it.
- At minimum, record `anchor_columns_voting` in `signals`, so that an
  unavailable check is visible rather than looking like a pass.

**Test fix:**
- Assert `signals["anchor_disagreement_px"] is not None` together with the
  no-flag assertion.
- Add a negative control: a 2-column frame whose columns move 26 px apart must
  be refused.

### M2 (medium): a gate-flagged ROI can still escape `on_qc_fail` through the duplicate-patch raise

`_calibrate_color_rpcc.py:352-363` (the `if name in measured: raise ValueError`
inside the loop, which runs before the policy at `:377-394`).

**Mechanism.** The plan kept "Two ROIs both identified a patch" outside the
policy, on the grounds that it is a configuration error (overlapping
rectangles). That reasoning fails when the colliding ROI has already been
flagged `patch identity undetermined`. Its placement is then by definition a
guess, and a guess can land on ROI 0's rows. The raise happens inside the loop,
so `skip` never gets to apply.

**Confirmed by running.** `frozen_op(on_qc_fail="skip")`, with ROI 1's first
row painted in ROI 0's colours and the other rows grey (0.3 or 0.5). ROI 1's
margin is 0.065–0.068, below 0.20, so the gate would flag it. The winning
placement is still `rows 1-2 … transposed`, which is ROI 0's placement. Result:
`RuntimeError`, root `ValueError: Two ROIs both identified a patch as 'dark
skin'`, and `qc == []`. The same happens with 2–4 copied rows, where the margin
is ≥ 0.24 and the gate passes, but those cases really are ambiguous input.

A fully blank ROI 1 (15 grey/seed combinations) never collided, so the
realistic trigger is a partly occluded card whose surviving tiles resemble
another ROI's.

**Fix:**
- Only add names to `measured`, and only run the collision check, for ROIs
  whose record is `ok` (or whose margin is determined).
- Or turn a collision into a flag on both ROIs, so the policy decides.

**Test:** the fixture above under `skip` should return the image unchanged with
ROI 1 flagged.

### M3 (medium, spec mismatch): `patch_census.accepted` and `patch_census()` counts include outlier-rejected patches

`_calibrate_color_rpcc.py:396-399` and `:417-420`;
`_checker_qc.py:273` (docstring); `_calibrate_color_rpcc.py:519-522`.

**Mechanism.** `accepted = list(measured.keys())` is taken *before*
`fit_from_patch_colors` performs outlier rejection. It feeds three things: the
census warnings, `diagnostics["patch_census"]["accepted"]`, and the
`patch_census()` counts.

Three sources say the opposite:
- The spec (§Missing-tile warnings): "Emitted per frame after identity
  assignment and outlier rejection".
- The `warn_on_patch_census` docstring: "Patch names that survived detection
  and outlier rejection".
- The coordinator's question: "does `accepted` match what was fitted?"

**Confirmed by running.** Degree 3, three neutrals painted green. The profile
rejects `neutral 8`, `neutral 6.5` and `neutral 5`, but the diagnostics'
`accepted` has 24 entries and contains all three. The census warnings are `[]`,
and `CalibrateColorRpcc.patch_census(...)["per_image"] == {"f": 24}`, although
21 patches were fitted.

A user who chooses the batch degree from `patch_census()` sees 24 and does not
see that three patches never reached the fit. (Before this range, the list was
`[t["patch"] for t in tiles]`, so the mismatch predates it. It was asked about
explicitly, and the census count is what `patch_census` exists to report.)

**Fix:** after the fit, set
`accepted = [n for n in measured if n not in fitted["rejected_patches"]]`, and
issue the census warnings on that list. `require_rank` then only needs to run
once, on the post-rejection count.

### L1 (low): under `"warn"`, a frame whose ROIs are all refused dies with a misleading rank error

`_calibrate_color_rpcc.py:396-399` and `_checker_qc.py:291`.

**Confirmed by running:**
- A blank frame with no prior and `on_qc_fail="warn"`.
- Both ROIs refused by an `expect_tiles` mismatch.

In both cases the result is `RuntimeError`, root `ValueError: A degree-3
root-polynomial fit needs at least 13 patches but only 0 were accepted … Re-shoot
the card, or configure a lower degree`.

Raising is defensible, because the spec makes rank a hard error. The advice is
wrong, though: no degree fits 0 patches, and the actual cause (every ROI was
refused) only appears in a separate `UserWarning`. `patch_census` depends on
this raise to record 0, so the behaviour is load-bearing.

**Fix:** when `measured` is empty, raise a message that names the refusals, for
example `"every ROI was refused: <summary>"`.

### L2 (low): `assign_placement` silently corrupts features for a non-float reference

`_checker_identity.py:268-269`.

**Mechanism.** `ref_features = np.full_like(reference, np.nan)` inherits the
reference dtype. With an integer `reference_linear`, the NaN fill becomes
`INT_MIN`, and the float features are truncated to 0 or 1 when assigned. The old
code went through `identity_features`, which casts to float64. The claim that
the change is "identical when every tile is finite" is therefore true only for
float references.

**Confirmed by running.** Integer (×255) reference against the same values as
float: margin 0.182 against 0.749, score 0.909 against 0.000. On integer input
the gate would refuse a clean card. `numpy` emits only a
`RuntimeWarning: invalid value encountered in cast`.

**Impact.** Not reachable from the operation, because `_load_reference_data` is
float. It is reachable by any direct caller.

**Fix:** use `np.full(reference.shape, np.nan)`, or
`np.vstack(...).astype(np.float64)`.

### L3 (low, test gap): the wiring from the operation's `empty_tiles` into the gate is untested

`_calibrate_color_rpcc.py:345`; `test_calibrate_color_rpcc_review.py:144-159`.

**Confirmed by mutation.** I monkeypatched `evaluate_roi` to force
`empty_tiles=0` and ran `test_an_empty_tile_never_turns_the_fit_into_nan`. It
still **passes**. The test runs under `"warn"` and never inspects `qc[0].flags`.
The `test_checker_qc.py` parametrisation covers `evaluate_roi` alone.

On real code, `qc[0].flags` correctly holds `2 tile box(es) fall outside the
ROI…` with `signals["empty_tiles"] == 2`.

**Fix:** assert `operation.qc[0].signals["empty_tiles"] == 2` and that the flag
is present.

### L4 (low, test gap): the "every tile box falls outside the ROI" refusal has no test

`_calibrate_color_rpcc.py:321-326`.

**Confirmed.** A grep found no test for it. I ran the branch myself: a
`frozen_op` with ROI 0's prior at `col_x=(500, 560)` under `skip` produces the
flag `['every tile box falls outside the ROI']` and the image is skipped, so the
branch works. Without the branch, `assign_placement` raises `No tiles are
allowed to vote.`, a `ValueError` that bypasses the policy. That is the exact
class of bug this range fixes, and nothing guards it. The
`len(candidates) < 2` refusal (`:301-306`) is also untested; `code-review.md`
already notes that no chart shape reaches it.

### L5 (low, test strength): `test_rank_is_checked_after_outlier_rejection` does not pin which rank check fired

`test_calibrate_color_rpcc_review.py:176-180`.

**Is the try/except sound?** Yes, for the bug it was written against. Without
the post-rejection check nothing raises, the else-path computes `kept = 21`, and
the test fails. That is confirmed by the coordinator's mutation and consistent
with my run.

**The weakness.** The except-path matches only the prefix `"degree-4
root-polynomial fit needs at least 22"`, which the *pre-fit* check
(`warn_on_patch_census`) emits too, and it never asserts that rejection
happened on that path. If a fixture or refactor drift ever drops tiles before
the fit (an empty or refused tile), the test goes green through the pre-fit
check without exercising post-rejection at all.

The current message is `…but only 21 remain after outlier rejection`
(confirmed by running).

**Fix:** replace the try/except with
`pytest.raises(RuntimeError, match="only 21 remain after outlier rejection")`,
and assert `operation.fitted_profile is None`.

### L6 (low): refusals desynchronise `diagnostics["lattices"]` from ROI indices

`_calibrate_color_rpcc.py:289-292`.

**Confirmed by reading.** A `lattice not found` refusal `continue`s before
`lattices.append`, while `expect_tiles` and all-empty refusals append. Under
`warn` or `skip`, `lattices[i]` then no longer describes ROI *i*. There are no
consumers in `src/` today (grep).

**Fix:** append `None` for an ROI with no lattice, or store `roi_index` in each
entry.

### L7 (low, maintainability): `test_checker_detect.py` was re-normalised CRLF→LF in the repository

**Confirmed.** The stored blob at `df868123d` had 300 CR lines; at HEAD it has
0. The working tree is still CRLF (`core.autocrlf=true`). The commit's `623 +/-`
diff is therefore whole-file churn: the real change is 23 lines
(`git diff -w --ignore-cr-at-eol`), and `git blame` is reset for the file. This
contradicts the plan's "preserve CRLF" constraint in spirit, though not in the
working tree. The other files' blobs were already LF.

---

## Suspicions (reasoned, not reproduced)

### S1 (low): a column clipped in the *reference* frame still votes when the card moves inward

`_calibrate_color_rpcc.py:446-452`.

**Reasoning.** The filter looks only at the refined geometry. Take a prior column
fitted flush at `x0 = 0` on a card that physically extends past the frame. When
the card moves right, its `x0 >= 0`, so it votes, but its visible plateau centre
tracks the move at about half rate. For a column clipped by about 30 px, the
estimate drifts about 15 px against a 12 px limit. A roughly 30 px inward move
could then be refused while `shift_px` sits at the 30 px limit.

**Status.** Not reproduced; plausible only near the shift limit.

---

## Verified sound (the coordinator's questions)

- **Per-run state reset** (`:258-260`). `fitted_profile`, `qc` and
  `_diagnostics` reset on entry. After a raise, `fitted_profile is None` and
  `diagnostics == {}` (confirmed on the rank-failure run). `qc` is set before
  the policy raise, so a gate failure is inspectable. A mid-loop raise leaves
  `qc == []`. That is safe, but the reset of `qc`/`_diagnostics` has no test
  (on the skip path both are overwritten anyway).
- **Other per-ROI failures.** `refine()` on the prior path does not raise on
  blank or empty input: `refine_rigid` falls back to `dx = 0` and `dy = 0`, and
  `frozen` returns the prior. `extract_patch` returns empty arrays; `measure_tile`
  returns `n_pixels = 0` with NaN values. Every NaN reaching `evaluate_roi` goes
  through `nanmean`/`nanmax` guards. The only escapes I found are M2 and the
  out-of-scope `roi.anchor_col` out of range, which gives an `IndexError` (a
  configuration error).
- **`assign_placement` with all tiles finite** is identical to the old code for
  float input (see L2 for the exception). With exactly 1 finite tile:
  `n_tiles = 1`, Hungarian agreement 1, margin 0.0. The margin gate and the
  `empty_tiles` flag both refuse it. With `voting & finite` empty the function
  raises `No tiles are allowed to vote.`, which is reachable from the operation
  only if L4's branch is removed.
- **Hungarian and `n_tiles` counts.** From the operation (`voting=None`) both
  count finite tiles, so `hungarian_disagreement` is always ≥ 0. With a `voting`
  mask, `n_tiles` counts voting∩finite while Hungarian counts all finite tiles,
  and I measured a disagreement of **−10** (row-0 voting). This predates the
  range: the old Hungarian also ran over all tiles. It is unreachable from the
  operation, but worth a line in the docstring or a clamp.
- **ECC re-pivot maths** (`_checker_detect.py:622-634`). Confirmed by running
  with a prior that already carries `rot = −3°` (registered on a band rotated
  3°), against a target rotated 7°. ECC recovers −4.000° at correlation 0.9997,
  the composed `lattice.rot = −7.000°`, and every core box's median colour
  matches its tile exactly (worst error 0.0 Lab). The derivation holds: 2-D
  rotations compose additively, and the centroid is invariant under rotation
  about itself. `translated()` rounds `x0`/`x1` to int, so horizontal placement
  and the pivot are off by at most 0.5 px. The effect on rotation is
  |(I−R)e| ≈ rot·0.5 px, which is negligible. Banker's rounding can change a
  column's width by 1 px when `frac(dx) == 0.5` exactly; this is cosmetic.
- **The rotation sense flip.** The only production caller of `boxes(rot=)` is
  `_measure_roi`. `refine_rigid` sets `rot = 0.0` on its result but
  `translated()` keeps `prior.rot`. `CheckerLattice` is public
  (`phenotypic.correction.__all__`), so any lattice serialised with a non-zero
  `rot` before `5e243c959` would now rotate the other way. The branch is
  unreleased, so I rate this informational.
- **ECC test tolerances.** `test_ecc_refinement_places_boxes_on_a_rotated_band`
  uses a 2.0 Lab median tolerance against tile colour steps of tens of Lab on a
  noise-free band, so it is discriminating. Its `hypot < 3.0` bound comes close
  to the mechanism: `scipy.ndimage.rotate` pivots on the array centre, which
  sits about 25.5 px from the lattice centroid, so the true centroid
  displacement at 4° is about 1.8 px. The headroom is about 1.2 px, and the
  comment "rotated about its own centre" is slightly inaccurate. The test calls
  cv2 without `importorskip`, which is correct: `opencv-python` is a core
  dependency, and a missing module fails rather than skips.
- **The direction test** (`test_checker_roi.py`) checks only the y component,
  because both tiles share a row. A mutation of the `nx` sign alone would pass
  it, but the ECC placement test catches that mutation.
- **The illuminant fix and the length validator** are fine as written.

---

## Not checked

- **The full `patch_census` exception taxonomy.** A `pydantic.ValidationError`
  (a `ValueError` subclass) raised *inside* `apply` would be counted as a
  failed frame rather than propagating as a bug. Remaining: a test in which a
  non-`ValueError` root propagates out of the census.
- **S1** was not reproduced (it needs a physically clipped reference frame).
- **Mypy, ruff and the 660-test surface** were not re-run; I relied on the
  coordinator's results. I ran only the 111 focused tests.
- **Whether the `empty_tiles` refusal fits the spec.** The spec says a missing
  patch *warns*; an empty tile is now a *refusal* under the default `"raise"`.
  This follows the plan's decision and I did not escalate it, but a spec note
  should record the decision.
- **Noticed outside the diff, not investigated further.** ROI 1 with rows 0–3
  uniformly grey (8 of 12 tiles occluded by a flat object) **passes the gate**:
  margin 0.256, impurity 0, one Hungarian warning only. The fit then runs with
  8 mislabelled tiles, and the mean ΔE00 after correction is 9.07. The spec's
  fault-injection claim covers only a *fully* blanked card. This belongs in its
  own finding against the gate design, not this fix range.

---

## Disposition (coordinator, 2026-09-21)

Everything was applied test-first in `c955440b6..3c7d5f134`. Afterwards,
`tests/unit/correction` plus the two smoke files gave 665 passed. The only 3
failures are the `FilFinderDetector` smoke tests (optional topology
`ImportError`). mypy is unchanged at 5 errors, all pre-existing.

| ID | Outcome |
|---|---|
| M1 | **Partly fixed.** `anchor_columns_voting` is now a signal, and a warning fires when fewer than two columns vote, so a check that could not run no longer looks like a pass. The guard test asserts both. **Still open, needs a user decision:** restoring the check on clipped 2-column cards means estimating a clipped column's shift from its unclipped inner edge, which is a detection-algorithm change. The reviewer's 26 px negative control is not added, because it would fail until that decision is made. |
| M2 | Fixed. A collision is now a flag on the later ROI, whose tiles are not used, and `on_qc_fail` decides. Tested under `skip` with the reviewer's fixture. |
| M3 | Fixed. `accepted` excludes patches rejected as outliers; the census warns about them as missing; `patch_census()` counts what was fitted. |
| L1 | Fixed. When every ROI is refused, the error reads "No ROI produced usable tiles…" and names the refusals. |
| L2 | Fixed (`np.full(..., np.nan)` in float64), with a test comparing an integer reference to a float one. The implementer's matching concern about the observed side does not apply: `observed` is already converted to float64 at `_checker_identity.py:224`. |
| L3 | Test strengthened. It now asserts `signals["empty_tiles"] == 2` and the flag, and fails when the wiring is removed. |
| L4 | Test added for the all-boxes-miss refusal under `skip`. |
| L5 | Test tightened to `match="only 21 remain after outlier rejection"`, plus `fitted_profile is None`. |
| L6 | Fixed. `diagnostics["lattices"]` has one entry per ROI, `None` where no lattice was found. |
| L7 | No action. The LF blob matches the other 37 stored files in the module; the working tree is CRLF. |
| S1 | Not reproduced. It is part of the M1 decision. |
| Partial occlusion passing the gate (8 of 12 tiles, 9.07 ΔE00) | Out of scope for this fix series. It should be raised as its own finding against the gate design. |
