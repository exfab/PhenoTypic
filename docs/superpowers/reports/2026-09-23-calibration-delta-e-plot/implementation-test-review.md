# Implementation + test review: calibration ΔE00 bar plot

Scope: the uncommitted diff in `.worktrees/calibration-delta-e-plot` (7 files). The
review was stopped early at the coordinator's request, so some checks are marked
**UNVERIFIED** below.

## Summary

**Nothing blocking found.** The bar data is right: tiles are ordered column-major per
ROI, as the overlay key orders them. Only tiles with both ΔE values get bars, which
means used, partly_covered and rejected tiles. Rejected tiles are hatched and left
out of the title mean. `N` in "over N fitted patches" equals `record.n_fitted` by
construction. Refused, skipped and no-lattice records draw a text-only page, so
`inspect()` returns both pages in every case where it runs. No other caller in
`src/` or `tests/` assumes `CalibrateColorRpcc.inspect()` returns a single Figure.
I grepped `.inspect(` across `src/` (including `_gui`, the coordinator and
`_store_figures`) and found none. The GUI renders through `report()`/`show()`, and
this op declares no `@figure`. Every finding below is Medium or Low.

Test run: the focused unit run (`test_calibration_overlay.py`,
`test_calibration_plot_image.py`, `test_deferred_imports.py`) passed, 175 tests
in 50 s. The integration file `tests/integration/cli/test_calibration_figure_in_store.py`
was not run.

## Critical issues (high confidence)

None.

## Likely issues (medium confidence)

### M1. Stores written before the upgrade keep the old flat `default` page, which can mix layouts in one run
- Where: `_store_figures.py` `_SameRunKeeper.keep` / `_kept_binding` (about lines
  296-395); `_store_copyout.py:_publish_binding` (lines 172-195);
  `_remove_leftovers` (lines 359-376).
- Scenario: a full run is made with the pre-change code. Each store holds
  `figures/<run>/cal/default.png`. The run is then resumed, or re-run with
  `--mode measure` under the same pipeline, with this change installed. The
  pipeline JSON is unchanged, so the run id (date + pipeline sha) is the same.
  - Images that §3a keeps (staged Stage 3, or measure with the same pipeline)
    republish the old single `default` page in the **flat** layout,
    `plots/cal/ds/<image>.png`.
  - Freshly drawn images publish the **manifest-dir** layout, `plots/cal/ds/<image>/`.
  - The layout flips between images: the exact invariant this change sets out to
    protect, reached across versions instead of across verdicts.
  - An image redrawn under the new code also leaves its old flat `<image>.png`
    beside the new directory. `_remove_leftovers` only cleans stems inside the
    directory being written.
- Impact: small in practice. The figures-in-store feature merged just before this
  change (#239), so few such stores exist. Still, nothing fences the change: no
  revision bump and no digest input.
- Fix options:
  - (a) Accept it and add a sentence to `correction/CLAUDE.md`.
  - (b) Bump `PROCESS_LAYER_SEMANTICS_REVISION` for `--mode process`. That does
    not cover full or measure runs.
  - (c) Have `_kept_binding` refuse a stored entry whose page keys differ from the
    current producer's (a binding-level failure rather than a silent keep). That
    needs the producer to declare its page keys.
  - I recommend (a) plus a test pinning whichever behaviour is chosen.

### M2. The chart's colours do not follow DESIGN.md's fixed series order
- Where: `_calibration_overlay.py`, new constants `_BEFORE_COLOUR = "#E69F00"` and
  `_AFTER_COLOUR = "#0072B2"`.
- What DESIGN.md says (lines 114-115 and 332): "NEVER reorder Okabe-Ito series.
  Series order is fixed: navy, orange, sky, green, blue, purple". For two series
  that means `#003660` and then `#E69F00`. Orange and blue are series 2 and 5.
- Orange is also the semantic "warning" colour (DESIGN.md line 322). It is the
  overlay's own `partly_covered` status colour (`STATUS_COLOURS`) too, so an
  orange "before" bar reads as "flagged" next to the overlay page.
- Fix: `_BEFORE_COLOUR = "#003660"`, `_AFTER_COLOUR = "#E69F00"`. Alternatively,
  record a deliberate deviation in the constant's comment.
- Confidence: medium. It depends on whether "fixed order" means "take the first n".

### M3. A failure in the bar chart now costs the tile-overlay page too
- Where: `_calibrate_color_rpcc.py`, `inspect()` return (the `PlotOutput(pages=(...))`
  block).
- `show_tiles()` and `show_delta_bar_plot()` both run eagerly inside the
  `PlotOutput` constructor. If `render_delta_e_bars` raises, `build_image_figures`
  records a **binding-level** failure and stores neither page. Before this change
  the overlay stood alone. The tiles Figure that was already built is not closed.
  It is not managed by pyplot, so garbage collection frees it and there is no
  registry leak.
- Judgement: acceptable, arguably correct. A binding that fails whole publishes
  nothing, so the layout cannot flip. A half-built output (tiles only) would
  publish flat for that image and flip the layout. The bar renderer's
  construction-time failure surface is also small: `_theme_font_family` is shared
  with the overlay, and the bar and annotate calls are plain matplotlib.
- Suggestion: keep the behaviour and add a one-line comment in `inspect()` saying
  it is deliberate. Optionally, close `tiles` in an `except` before re-raising.

## Maintainability concerns (low)

- **L1. The title mean differs from the profile's own mean.**
  `ColorCheckerProfile.diagnostics["mean_deltaE00_before/after"]`
  (`_color_checker_profile.py:915-916`) averages over *all measured* patches,
  rejected ones included. The bar title averages over non-rejected tiles only.
  The title says "over N fitted patches", so it is honest. A user comparing it
  with `op.fitted_profile.diagnostics` will still see different numbers whenever
  a patch was rejected. Mention it in the `render_delta_e_bars` docstring.
- **L2. `test_a_refused_frame_still_yields_both_pages` (test_calibration_plot_image.py)
  runs `on_qc_fail="skip"`, so its verdict is `skipped`, not `refused`.** That is
  the only no-fit path that reaches `inspect()` in a pipeline: "raise" and
  "warn"-all-refused both raise out of `apply()`. The name should say skipped. It
  also asserts only the page keys. Add `ax.containers == []` on the `delta_e`
  page, or assert that the PNG exists.
- **L3. The `policy == "raise"` case in `test_a_frame_with_no_fit_draws_no_bars_and_says_why`
  never checks that apply raised.** It passes whether or not `apply()` raises.
  Use `pytest.raises(RuntimeError)` for that case.
- **L4. The ordering test re-implements `_scored_tiles`' sort** (`key=(col, row)`).
  It pins column-major order, but it does not tie that order to the overlay key's
  numbering, which the docstring cites. The planted fixture has 2 columns per ROI,
  so it uses the side-label layout and has no numbered key at all. A mutation of
  `_draw_key`'s sort would not be caught here. It could be derived from the
  overlay's key texts on a fixture with more than 2 columns.
- **L5. Page labels become the deliverable file names**: `Tile-overlay.png` and
  `Delta-E00-before-and-after.png` (verified with `unique_page_stems`). They are
  stable, but they will not match the store's `tiles.png` / `delta_e.png`. Only
  the integration test's `PAGES` table pins them. That is fine; noted so a label
  edit is recognised as a rename of a user-facing file.
- **L6. Normalising also changes the 8-bit rendering path.** `imshow(uint8)` became
  `imshow(float64 in [0,1])`. matplotlib quantises float RGB back to bytes, possibly
  by truncation, so some 8-bit pixels could render one level off in the PNG.
  **UNVERIFIED.** It is visually irrelevant. It matters only if a golden overlay
  PNG exists, and I found none. The measured swatches use `rgb.normed()`, the same
  normalisation, so the crop and the swatches are consistent.
- **L7.** `normalize_rgb_bitdepth` raises `ValueError` for a float crop whose max
  is above 65535. Before, `imshow` clipped with a warning. That path is
  unreachable in practice because `Image.rgb` is uint8 or uint16.
- **L8. Width is unbounded.** It is `0.34 in × n_tiles` with no cap, at 160 dpi.
  The default checker has 24 patches, about 9.4 in. A larger chart, if ever
  configured, gives a very wide PNG. The height is fixed at 3.4 in, so long
  rotated labels plus a wrapped two-line suptitle could squeeze the axes under
  constrained layout. The overlap test only proves that text stays in frame, not
  that the axes keep a usable height. **UNVERIFIED: I did not view a rendered PNG.**

## Test coverage gaps

1. No test covers a `corrected` record with **zero** rejected tiles, which is the
   path with no rejected legend handle and no hatching. The 16-bit twin test
   (`calibrated(render_frame())`) goes through it but asserts only heights. Add
   `len(legend handles) == 2` and "no hatch" there.
2. No test covers a record where a ROI **lost a collision** (tiles `excluded` with
   `claimed=False`). The expected result is no bars for those tiles and an unchanged
   `N`.
3. The integration test shows both pages are published in a manifest dir for one
   image. It does not show two images with *different verdicts* (one corrected,
   one skipped) publishing the same layout, which is the invariant motivating
   "always both pages". A two-image `--on_qc_fail skip` integration case would
   pin it.
4. The deferred-import registry was updated for `_theme_font_family`, `Patch`,
   `render_delta_e_bars`. It passes in the focused run.

## Recommended changes (priority order)

1. Decide M1: at minimum document it; ideally pin it with a test.
2. M2: use `#003660` / `#E69F00`, or justify the choice in a comment.
3. L2 / L3: tighten the two no-fit tests (rename, assert no containers, and use
   `pytest.raises` for the raise policy).
4. Coverage gaps 1 and 3.
5. M3: add a comment in `inspect()` that whole-binding failure is intended.
