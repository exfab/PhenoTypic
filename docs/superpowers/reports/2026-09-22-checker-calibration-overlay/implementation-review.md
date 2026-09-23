# Implementation review — checker calibration overlay

Reviewed: `git diff 639c5cfec HEAD` on branch `feat/checker-calibration-overlay`
(worktree `.worktrees/calibrate-color-rpcc`). Reviewer: implementation-test-reviewer.
Status: COMPLETE.

## Summary

No High-severity defects. The record matches the spec's verdict and status tables on
every path I drove. A collided ROI's tiles are `excluded` and carry no foreign ΔE,
refusals keep a record, and a mid-loop exception leaves `None`. Two Medium
implementation gaps: a failed `ColorCorrector.apply()` leaves a "corrected" record
(F1), and a ROI refused after its lattice was found draws no boxes (F2). The
font-pinning fix holds at saved dpi 40–600, under PDF metrics, and with no theme
font installed. It holds by about 0.16 in of margin, not exactly, and no test pins
it (T4). The main weakness is the tests. Six renderer mutations that make the figure
visibly wrong, and both record mutations aimed at the collided/ΔE rules, leave all 21
tests green (T1, T2).

Counts: High 0 · Medium 4 (F1, F2, T1, T2) · Low 7 (F3, F4, F5, F6, T3, T4, plus the
spec-table precedence note under "Verified correct") · Suspicions 3 (S1–S3).


## Confirmed findings


### F1 — Medium — A failed `ColorCorrector.apply()` leaves a "corrected" record behind
`_calibrate_color_rpcc.py:542-550`. `keep_record("corrected…")` runs before
`ColorCorrector(...).apply(image, inplace=True)`. If that apply raises, `apply()`
surfaces a `RuntimeError`, yet `calibration_record.verdict` stays
`corrected_with_warnings` / `corrected` with `n_fitted=21` and after-ΔE values for a
correction that was never applied. `show_tiles()` then draws a figure titled
"corrected" for an image whose pixels were not corrected (in-place: possibly
half-written). **Confirmed by running**: monkeypatched `ColorCorrector.apply` to raise
on the planted-fault frame -> `RuntimeError`, record verdict
`corrected_with_warnings`, `n_fitted 21`. (`fitted_profile` is also left set; that is
pre-existing behaviour, but the record adds a user-facing verdict that is now false.)
The spec's lifecycle ("a successful run never leaves a stale one") does not cover this
exit, and no test does.
Fix: build the record first, run the correction, then assign:
```python
record_kwargs = dict(fitted=..., rejected=rejected, n_fitted=len(accepted))
try:
    out = ColorCorrector(...).apply(image, inplace=True)
except Exception as exc:
    keep_record("refused", refusal=f"correction failed: {exc}")
    raise
keep_record(verdict, **record_kwargs)
return out
```
(or assign `refused` in an `except` around the call). Add a test that monkeypatches
`ColorCorrector.apply` to raise and asserts the verdict is not `corrected*`.

### F2 — Medium — A ROI refused *after* its lattice was found records no tiles, so the figure shows no boxes
`_calibrate_color_rpcc.py:391-405` (expect-tiles / placement refusals) and `:414-419`
("every tile box falls outside the ROI") `continue` before `draft.tiles` is assigned at
`:425`. `draft.lattice` is already set at `:389`, so the record says
`lattice_found=True, n_tile_columns=2` with `tiles=[]`, and `_draw_roi` draws the crop
with **no boxes**. These are exactly the frames the spec says the overlay exists for —
"declared to hold 7 tiles but 12 were detected" is a misplaced/mis-sized lattice, and
seeing the 12 detected boxes is the only way to diagnose it. The all-empty case is
also the only producer of the spec's `empty` status for a whole ROI, and it is dropped.
**Confirmed by running**: `rois[1].expect_tiles=7`, `degree=1`, `on_qc_fail="warn"` ->
record `rois[1] = (lattice_found=True, n_tile_columns=2, len(tiles)=0, flags=['declared
to hold 7 tiles but 12 were detected'])`, verdict `corrected_with_warnings`.
Spec gap as much as code gap: `TileOverlay.patch` is required, and no identity exists
before `assign_placement`. Fix options: (a) record the lattice's full/core boxes for a
refused ROI as tiles with `patch=""` and status `excluded` (or `empty` for
`n_pixels==0`), renderer omitting the name line; or (b) add an `RoiOverlay.boxes`
field drawn as dotted/dashed grey outlines when `tiles` is empty. Either needs a spec
amendment and a test (`expect_tiles` mismatch -> boxes drawn).

### F3 — Low — Post-rejection rank failure discards which tiles were rejected
`_calibrate_color_rpcc.py:529-533`. The fit ran and `rejected` is known, but
`keep_record("refused", refusal=...)` passes neither `fitted` nor `rejected`, so every
tile is `excluded`. This matches the spec's table (refused -> excluded, ΔE `None`) and
is therefore not a defect against the spec, but the user who hits "only N remain after
outlier rejection" cannot see *which* tiles were rejected — the one fact that explains
the refusal. **Confirmed by running**: degree 4, three green neutrals -> verdict
`refused`, statuses `{'excluded'}` over all 24 tiles, `n_fitted None`. Recommend a spec
amendment: on this exit pass `rejected=rejected` and have `_tile_status` return
`rejected` when `patch in rejected` even without `fitted` (ΔE still `None`).

### F4 — Low — Under warnings-as-errors the skip path leaves no record
`_calibrate_color_rpcc.py:490-495`. `warnings.warn(...)` for the gate failure precedes
`keep_record("skipped")`; with `-W error::UserWarning` (or `simplefilter("error")`) the
warning raises first and the record stays `None`, so `show_tiles()` says "call apply()
first" after an apply that did run. Same for per-ROI `warnings.warn` at `:469`. The
refused frame the user most wants to see has no record. **Confirmed by running**
(`simplefilter("error", UserWarning)`, `on_qc_fail="skip"`, gain 1.6 -> record `None`).
Fix: call `keep_record("skipped")` before the `warnings.warn`, or build the record
before the gate block and patch the verdict.

### F5 — Low — The `show_tiles()` docstring's `finally` pattern hides the real error when no record exists
`_calibrate_color_rpcc.py:245-249` recommends
`try: op.apply(plate) finally: op.show_tiles().savefig(...)`. The record is `None`
whenever `apply()` raises before the ROI loop finishes (mid-loop exception, F4's
warnings-as-errors case), and then `show_tiles()` raises
`RuntimeError("... call apply() first")` inside `finally`, which becomes the
exception the user sees; the real cause is only in `__context__`. **Confirmed**: record is
`None` after a mid-loop raise and under F4 (both run); the masking follows from Python's
`finally` semantics. Fix: document `except RuntimeError: if op.calibration_record is not
None: op.show_tiles()...; raise`.

### F6 — Low — The "frozen" record has a writeable crop
`_calibration_overlay.py:69-73`. `frozen=True` stops field reassignment only;
`record.rois[0].crop[...] = 0` silently rewrites the record's pixels. **Confirmed by
running** (`crop.flags.writeable is True`). Since the record is meant to be handed to
another session to persist, set `crop.flags.writeable = False` in `build_overlay_record`
(it already owns the buffer, so this costs nothing).

### T1 — Medium — Nothing pins that ΔE never lands on another ROI's tile, or the status precedence
Mutation tests, run against the 21 non-monkeypatch tests of
`tests/unit/correction/test_calibration_overlay.py` (all 21 pass after each):
- **m1**: drop the `status in ("used", "partly_covered", "rejected")` guard at
  `_calibration_overlay.py:182-187`. Collided ROI 1's excluded tile "dark skin" then
  carries ROI 0's `delta_e_after = 1.86`. That is precisely the "ΔE attached to a tile
  whose patch belongs to another ROI" defect. **Survives.**
  `test_a_collided_roi_marks_its_tiles_excluded` (`:152-158`) checks statuses only.
- **m2**: swap precedence so `rejected` beats `excluded`. **Survives.** No fixture has a
  losing ROI whose name the winner had rejected.
Fix: in the collided test add
`assert all(t.delta_e_after is None and t.delta_e_before is None for t in record.rois[1].tiles)`
and `assert record.rois[1].tiles` (non-empty). Add a collided frame where ROI 0's
colliding patch is also a planted outlier, and assert ROI 1's tile is `excluded`.

### T2 — Medium — `assert_no_overlap_or_clipping` and the structure tests pass on visibly wrong figures
`test_calibration_overlay.py:186-201, 273-282`. The helper checks only text-vs-text
intersection and text-vs-figure-edge. It never relates a label to its tile, a swatch
to its label, or text to the image axes. The structure test compares *sorted
multisets* of edge colours, so it never checks which box has which colour or where
the box is. Every mutation below leaves all 21 tests green (**confirmed by running**,
and the effect of each was checked separately, e.g. m5 moved core x from 19.5 to 45.0):
- m4: each core box drawn in the colour of the mirror-image tile's status;
- m5: every core box shifted 25 px right of its tile;
- m6: side labels and swatches drawn at vertically mirrored rows, so every label names
  the wrong tile;
- m7: swatches moved 60 px below their labels;
- m8: column-0 labels drawn over the image;
- m10: `top_pad` reservation removed. It survives because no fixture puts a tile near
  the crop edge (`card_record` keeps a 12 px margin).
Only two things kill anything: the halved-*width* mutation (the existing
`test_halved_label_widths_are_caught`), and a halved-*height* mutation that I ran
(5 of the 21 fail). So the width test is **not** the only proof that the overlap
check can fail, but no committed test pins the height direction.
Fix: (a) in the structure test, pair each core `Rectangle` with its tile by position
(`p.get_xy() == (cx0-0.5, cy0-0.5)`, width/height) and assert its colour and linestyle
against that tile's status; (b) for side labels, assert each name `Text`'s
`get_position()[1]` equals its tile's centre minus half the split, in the shared-y data
coordinates, and that its swatch rectangles share that centre; (c) assert no label's
window extent intersects the image axes' window extent; (d) add a fixture with a tile
flush with the crop's top edge; (e) commit the halved-height mutation beside the
halved-width one.

### T3 — Low — Spec tests 1, 3 and the `empty`/hatch paths are only partly implemented
- Spec test 1 requires "Boxes equal `CheckerLattice.boxes(...)`". No test compares
  `full_box`/`core_box` to the lattice. A record built with the wrong `core_trim` or
  without `rot=` would pass. (By reading. m5 shows positions are unchecked downstream too.)
- `empty` status, the ` · empty` / ` · excluded` suffixes rendered as side labels, and
  the hatched swatch for `measured_srgb=None` are never drawn by any test. The planted
  frame has **0** empty tiles (**confirmed by running**), and `card_record` uses only
  `rejected`.
- `{...} <= {"excluded", "empty"}` at `:103` and `:158` is vacuously true for a ROI
  with no tiles, which is exactly F2's failure mode. Assert non-emptiness.
- `test_a_post_rejection_rank_failure_keeps_a_record` (`:125-130`) asserts only the
  verdict. Spec test 3 also requires excluded tiles, `ΔE None` and the refusal text.
- `test_record_values_match_the_run` never covers an excluded tile, so it cannot see m1.

### T4 — Low — Font pinning holds across dpi, backend and a missing font, but no test pins it
Mechanism, `_calibration_overlay.py:509-512`: the family is resolved to a concrete
name under the theme and pinned with `rc_context`. `Text` resolves family, size and
colour into its own `FontProperties` at creation, so drawing outside the theme reuses
the measured font. **Confirmed by running** (mpl 3.10.7):
- **Save dpi**: rendered at 160, then `fig.set_dpi(d)` (what `savefig(dpi=d)` does) for
  d in {40, 50, 72, 100, 160, 300, 600} on the two-band, 6x2 and 4x6 records. No
  overlap or clipping anywhere. Rendering *at* dpi 50, 72 and 300 also passes.
- **Backend**: text extents re-measured with the PDF renderer (unhinted metrics, at 72
  dpi). No overlap or clipping on any of the three records.
- **No installed theme font** (theme `font.sans-serif` replaced by a nonexistent
  family, as on bare Linux CI): `findfont` falls back to DejaVu Sans, which is then
  pinned, and all three records pass. Every text carries `('DejaVu Sans',)`.
Verdict: the guarantee is **not exact by construction** across dpi. The labels are
measured at one dpi, and hinting changes glyph advances slightly at another. It holds
**by margin**: the horizontal slack to the group/figure edge is about 0.155 in at
50 dpi and 0.162 in at 600 dpi (`_EDGE_IN` plus padding), which is far more than the
hinting drift. **No test pins it**: every overlap test draws once, at 160 dpi, with
Agg and whatever font the machine has. Add a parametrised test that re-draws the 6x2
and 4x6 figures at `fig.set_dpi(d)` for d in (50, 300). It is cheap, and it would
catch a future regression to a generic family.

### Scope — clean
No `@figure`, `PlotImage`, `savefig` call (apart from the docstring example), zarr
persistence, CLI or `deliverables` change. No `pyplot` import. The record and renderer
are not exported from `phenotypic.correction.__init__`. The only doc change is a
paragraph in `correction/CLAUDE.md`, which says saving/publication is not implemented
yet. The deferred-imports allow-list entry matches the function-local imports.
Checked by grepping the diff.

### Verified correct (record)
- **Collided ROI -> `excluded`, no foreign ΔE**: ROI 1 in `collided_frame()` (degree 2)
  has 12 tiles, all `excluded`, none with ΔE; ROI 0 all `used`. `scored` is only looked
  up for `used/partly_covered/rejected`, which require `claimed`, and a claimed ROI's
  names are disjoint from every earlier ROI's, so ΔE cannot attach to another ROI's
  patch. Confirmed by running.
- **Precedence** `_calibration_overlay.py:117-125` is empty > excluded > rejected >
  partly_covered > used. The spec table lists `rejected` above `excluded`; the code's
  order is the correct one (a losing ROI whose name the winner had rejected is
  `excluded`, not `rejected`). Recommend the spec table state precedence explicitly.
- **Mid-loop exception** resets the record to `None` (reset at `:320`). Confirmed.
- **No pinning**: `measured_srgb` / boxes are validated into plain Python tuples of
  `float`; `RoiDraft`, `CheckerLattice`, `TileMeasurement` live only in `_operate`'s
  locals and the `keep_record` closure, neither of which outlives the call. Crops are
  `np.array(..., copy=True)` with `base is None`. Confirmed by running.

## Suspicions (read-only, not confirmed by running)

- **S1 — Low — `ZeroDivisionError` in key mode for a degenerate core box.**
  `_calibration_overlay.py:399-403`: `smallest_px` is the minimum core-box side, and
  it divides `marker * _MARKER_ROOM`. A lattice with more than two columns whose core
  box has zero height or width (a large `core_trim` on a tiny tile, if validation
  allows it) would raise from `show_tiles()` instead of drawing. Not run. Fix:
  `max(smallest_px, 1.0)`.
- **S2 — Low — Pickling or deep-copying the operation after `apply()` carries the crops.**
  `_calibration_record` is a `PrivateAttr`, and pydantic includes private attributes in
  `__getstate__` and in `model_copy(deep=True)`. An op pickled to joblib/loky workers
  after an in-process `apply()` would ship about 4.4 MB of crops per instance.
  `to_json()` is unaffected (already verified). Not run.
- **S3 — Low — Pinning one family narrows glyph fallback.** With the generic
  `sans-serif` family, matplotlib ≥3.6 renders through the whole
  `font.sans-serif` list. After the pin, the chain is the single resolved font plus
  matplotlib's DejaVu default. A glyph missing from the first installed theme font
  (Δ, ·) now falls straight to DejaVu, or to tofu if that path is off. On macOS this
  resolves to Helvetica Neue, which has both. Not checked on a host where IBM Plex
  Sans is the resolved font.

## Not checked

- The full test suite, and the affected-surface run (already verified by the caller).
- Rendering on a Linux host with IBM Plex Sans installed (S3), and the Cairo and SVG
  backends. PDF metrics and a simulated missing font were checked instead (T4).
- Records for non-`uint8` inputs (a float `image.rgb` crop through `imshow`).
- `GridImage` inputs, and pipelines where the op runs more than once per instance
  (only one sequential re-apply is covered: `test_each_apply_replaces_the_record`).
- The plan's recorded deviations were not diffed line by line against the plan's
  renderer code. I reviewed the shipped renderer against the spec instead.

---

## Disposition (coordinator, 2026-09-22)

The fixes were written test-first and committed as `4a85563d6`, `f4dc95d4f`,
`167751ef2` and `1149cb428`. Afterwards, the correction, ci and tune-coverage
suites plus the two smoke files gave 960 passed. The only failures are the 3
pre-existing `FilFinderDetector` smoke cases. mypy is unchanged at 9 errors.

| ID | Outcome |
|---|---|
| F1 | Fixed. The record is assigned after the correction succeeds; a failed correction leaves `refused`, with a refusal beginning "correction failed:". |
| F2 | Fixed, with a spec amendment. The new `RoiOverlay.unidentified_boxes` field holds a refused ROI's lattice boxes, drawn in grey without labels. |
| F3 | Fixed, with a spec amendment. A post-rejection rank refusal marks the rejected tiles `rejected`. The status table is now numbered, first match wins. |
| F4 | Fixed. The record is kept before the gate's warnings, so warnings-as-errors still leaves one. |
| F5 | Fixed. The docstring example uses `except RuntimeError`, draws only when a record exists, then re-raises. |
| F6 | Fixed. Record crops are read-only. |
| T1–T4 | Fixed. The overlay tests went from 22 to 45. Mutations m1, m2, m4–m8 and m10 are each now caught by at least one test. |
| S1 | Hardened. The key-mode scale is floored at 1 px. |
| S2, S3 | Left open. S2: pickling a used operation carries its crops. S3: pinning one font narrows glyph fallback. Neither was confirmed. |
