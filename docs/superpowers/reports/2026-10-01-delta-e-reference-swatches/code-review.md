# Code review: ΔE00 bar plot reference-colour swatches (commit 13c8b09)

Reviewer: independent (did not write the change). Scope: `git show 13c8b09` only.
Branch `feat/delta-e-reference-swatches`.

**Verdict: approve with nits.**

The swatches land exactly under their bar pairs at every size and dpi I tried,
including `savefig` at a dpi the figure was not built at and PDF/SVG export. The
colour encoding is right, and constrained layout reserves room for the
swatches. The real gap is in the test. It pins the order, colour and hatch of
the swatch *artists* but nothing about *where* they are drawn. Four plausible
placement bugs pass the whole calibration suite (65/65).

Severity counts: 0 blocking, 2 should-fix, 4 nits.

---

## Findings

### 1. should-fix: the test does not pin swatch placement, and no other test does

`tests/unit/correction/test_calibration_overlay.py:798-802`

The updated test selects the swatches with
`[p for p in ax.patches if not p.get_clip_on()]` and checks label order,
facecolour and hatch. It never checks that swatch *k* sits under bar pair *k*,
that it sits below the axis, or that it has a fixed size in inches. Those are
the properties this commit adds.

**Failure scenario:** a refactor anchors the swatches in reverse order or one
slot off, or places them inside the plot area on top of the bars. The artists
keep their labels and colours in bar order, so the test still passes, and every
`delta_e.png` page in the store gets swatches that are mislabelled or hidden.

**Evidence (observed).** I mutated `_calibration_overlay.py:729-732`, ran the
test, and restored the file each time with `git checkout -- <file>`. `git status`
was clean afterwards.

| Mutant | Edit | Target test | Both calibration files |
|---|---|---|---|
| M1 reversed positions | `scaled_translation(len(tiles) - 1 - i, 0, at_tick)` | **1 passed** | **65 passed** |
| M2 measured instead of reference colour | `facecolor=tile.measured_srgb` | 1 failed | not run |
| M3 no hatch | `hatch=None` | 1 failed | not run |
| M4 swatch inside the axes | `(-side / 2, gap)` | **1 passed** | **65 passed** |
| M5 no inch scaling | `transform=anchor` | **1 passed** | not run |
| M6 off by half a pair | `scaled_translation(i + 0.5, 0, at_tick)` | **1 passed** | not run |

Commands:
`QT_QPA_PLATFORM=offscreen uv run pytest -q -p no:cacheprovider -o addopts="" tests/unit/correction/test_calibration_overlay.py::test_bars_are_the_records_delta_e_one_pair_per_scored_tile`
for the target test, and
`... tests/unit/correction/test_calibration_overlay.py tests/unit/correction/test_calibration_plot_image.py`
for both files. The unmutated baseline for both files was 65 passed in 220 s.

The test therefore catches colour and hatch bugs (M2, M3) and misses every
placement bug (M1, M4, M5, M6).

**Suggested fix:** after `fig.canvas.draw()`, assert three things for each *k*:
- the centre of `swatch.get_window_extent()` equals the centre of the before/after
  bar pair, to within about 1 px;
- `swatch_extent.y1 < ax.get_window_extent().y0`;
- `swatch_extent.width / fig.dpi` equals `_TICK_SWATCH_IN`.

My scratch script measures exactly these, so they are cheap to add.

**Selector robustness (inferred, checked against the current code).**
`not p.get_clip_on()` is sound today. The bars are clipped
(`get_clip_on()` is True by default), the legend handles live in the legend and
not in `ax.patches`, and the threshold lines are `Line2D`. The rule is still
implicit, though. Any future unclipped patch would change what the test selects,
for example a marker drawn just beyond the axis. The `Returns:` docstring now
makes "unclipped `Rectangle` patches" a documented contract (`:769-770`), so
that contract at least is stated. A gid (`set_gid("reference-swatch")`) would be
a sturdier handle, but this is optional.

### 2. should-fix: the overlap/clip test still describes rotated tick labels and does not cover the swatches

`tests/unit/correction/test_calibration_overlay.py:840-866`

`test_the_bar_chart_labels_do_not_overlap_or_clip` still says "The rotated tick
labels are excluded from the overlap half only". It also still filters with
`if t.get_rotation() == 0:` (`:859`). Neither exists any more: the tick labels
are now empty strings, and the check skips them anyway. The x-axis labels are
now patches, and this test inspects only `Text`. Nothing checks that swatches
stay inside the figure or keep apart from each other.

**Failure scenario:** someone raises `_TICK_SWATCH_IN` above the per-tile pitch,
or drops the swatches from layout (`set_in_layout(False)`). Swatches then merge
or fall off the bottom edge, and this test still passes. It is the one test
whose name promises this guard.

**Evidence:** I read the code. Mutant M4 above, which moves the swatches onto the
bars, passed the whole file.

**Suggested fix:**
- update the docstring;
- add the swatch extents to the in-figure check (`box.y0 >= frame.y0`);
- assert a positive horizontal gap between neighbouring swatches at the default
  figsize.

### 3. nit: swatches merge or overlap when a caller passes a narrow `figsize`

`src/phenotypic/correction/_color_correction/_calibration_overlay.py:701,726`

The swatch side is a fixed 0.2 in. The comment `# ...; < _BAR_PITCH_IN`
compares it with the *default* figure pitch per tile. The real pitch is the
axes width divided by the number of tiles plus 0.2, and it shrinks when
`show_delta_bar_plot(figsize=...)` passes a narrower figure.

**Observed** (24 tiles, gap between neighbouring swatches, from
`scratchpad/review/analyze.py`):

| figsize | gap between swatches |
|---|---|
| default (9.36 × 3.4) | +0.123 in |
| (10, 5) | +0.149 in |
| (6, 3) | **−0.016 in** (adjacent swatches overlap and form one strip; see `fig6x3.png`) |
| (4, 2) | **−0.162 in** (matplotlib also warns "constrained_layout not applied") |

**Context, which is why this is only a nit:** the old text labels at (6, 3) were
much worse. I rendered the parent commit's module (`old_fig6x3.png`). Its
rotated names took most of the figure height and squashed the axes to a sliver.
This is not a regression. When it happens, the result is still poor: neighbouring
neutrals (`neutral 8 / 6.5 / 5`) merge into one grey bar, and the swatch is the
only patch identifier left.

**Possible fix:** clamp the side, for example
`side = min(_TICK_SWATCH_IN, 0.7 * pitch_in)`, with `pitch_in` taken from the
figure width over `len(tiles) + 0.2`. Or simply correct the comment so it states
the real invariant.

### 4. nit: `DEFERRED_SITES` not updated for the two new local imports

`tests/unit/ci/test_deferred_imports.py:101` (no entry for `ScaledTranslation`)

`render_delta_e_bars` now imports `Rectangle` and `ScaledTranslation` locally
(`_calibration_overlay.py:776-777`). The table lists `"Rectangle":
("render_calibration_overlay",)` only, and has no `ScaledTranslation` key. The
previous change on this function did register its local `Patch` import in this
table. The table exists so that "a moved import cannot go missing from a
function no other test calls" (its module docstring).

**Failure scenario:** a later edit hoists `from matplotlib.transforms import
ScaledTranslation` to module level. This per-site guard would not flag it.

**Evidence:** I read the code. `test_deferred_imports.py` passes as is (115
passed, together with the target test).

**Fix:** set `"Rectangle": ("render_calibration_overlay", "render_delta_e_bars")`
and add `"ScaledTranslation": ("render_delta_e_bars",)`.

### 5. nit: one reflowed docstring line is 104 characters

`src/phenotypic/correction/_color_correction/_calibration_overlay.py:751`

The edit rewrapped the opening of the docstring but merged the rest into
"after-value is held out rather than fitted. The title gives the mean before ->
after over the fitted", which is 104 characters. Ruff passes, so this is not
enforced in docstrings. Neighbouring lines in this docstring were already over
length before the change, so this is cosmetic only.

Evidence: `git show 13c8b09 | awk 'length > 101 && /^\+/'` printed that line.
`uv run ruff check <3 changed files>` printed "All checks passed!".

### 6. nit: the helper takes injected classes and has no type hints on `fig`/`ax`

`_calibration_overlay.py:718`

`_draw_reference_swatches(fig, ax, tiles, rectangle, scaled_translation)`
receives `Rectangle` and `ScaledTranslation` as parameters so that matplotlib
stays a deferred import. This matches the existing `_swatches(..., rectangle)`
pattern in the same module, so it is consistent. The alternative is a local
import inside the helper, which `DEFERRED_SITES` would then register.
Informational only.

---

## Checked and found no problem

- **Placement at any figsize or dpi (observed).** I measured the centre of each
  swatch's window extent against the centre of its before/after bar pair after
  `canvas.draw()`. The maximum difference was **0.000 px** in all of these cases:
  default; dpi 72; dpi 300; `fig.set_dpi(300)` after the figure was built;
  figsize (10, 5), (6, 3) and (4, 2); 3 tiles (minimum width); 1 tile; 144 tiles
  (`rois * 6`, 50 in wide). The swatch side measured exactly 0.200 × 0.200 in
  every time, and the axis-to-swatch gap was 0.050 in.
- **`savefig` at a different dpi (observed).** `savefig(dpi=72)` and
  `savefig(dpi=320)` from a figure built at 160 dpi, `export_figure` to
  PDF/SVG/PNG at dpi 300, and `savefig(bbox_inches="tight", dpi=110)` all put the
  swatches under their pairs. I looked at `default_dpi72.png`, `exportpdf-1.png`
  (pdftoppm) and `tight.png`. The reason this holds: `ScaledTranslation` registers
  the x-axis transform as its child, so it is invalidated with the axes, and
  `fig.dpi_scale_trans` follows the dpi used at save time.
- **Constrained layout reserves room (observed).** The swatch bottom sits 0.042 in
  above the figure edge, which is the default 3 pt layout pad, at every size,
  and nothing is clipped at the edges. `Patch` defaults to `in_layout=True`, and
  `clip_on=False` keeps the swatches in `Axes.get_tightbbox`.
- **No effect on data limits (inferred, consistent with the observations).** The
  transform does not contain `transData` as a branch, so `_update_patch_limits`
  does not move dataLim. xlim and ylim are set explicitly before the swatches are
  added in any case.
- **Overlap at default sizing (observed).** The gap between swatches stays
  positive at the default figsize for 1, 3, 24 and 144 tiles. The smallest gap
  was 0.123 in, at 24 tiles.
- **`reference_srgb` encoding and range.**
  `_calibrate_color_rpcc.py:467-470` builds it as
  `clip(cctf_encoding(clip(ref_linear, 0, 1), "sRGB"), 0, 1)`.
  `_load_reference_data` (`_color_checker_profile.py:175ff`) returns linear
  **sRGB** (`RGB_COLOURSPACES["sRGB"]`). So the values are display-encoded sRGB
  in [0, 1], a valid matplotlib RGB tuple, and the same colour the tile overlay's
  key already shows as its reference swatch (`_swatches`, `:493`). Observed
  range on the planted frame: 0.0 to 0.948. `TileOverlay.reference_srgb` is
  typed `Rgb` and is not optional, so every scored tile has one.
- **Refused or skipped frames.** These return before the swatches are drawn
  (`:801-806`). `test_a_frame_with_no_fit_draws_no_bars_and_says_why` passes.
- **Nothing downstream depended on the old tick text.** I grepped `src/`, `tests/`
  and `docs/` for `get_xticklabels`, `(rejected)`, `show_delta_bar_plot`,
  `render_delta_e_bars` and `delta_e.png`. The only reader of the tick text was
  the test this commit updated. `FEATURES.md`, `WORKFLOWS.md`, `DESIGN.md`, the
  GUI code and `docs/source` do not describe the x-axis labels.
  `src/phenotypic/correction/CLAUDE.md:46-51` ("paired bars per patch") is still
  accurate. Older plans and reports under `docs/superpowers/` mention the
  function but not its tick labels, and they are historical records, so they are
  not stale.
- **Store and `inspect()` pages.** The `delta_e` page key and file name are
  unchanged, and `test_calibration_plot_image.py` passes as part of the 65-test
  baseline.
- **Conventions.** matplotlib is imported inside `render_delta_e_bars`, there is
  no pyplot, Google-style docstrings are kept, and the inline comments are about
  as dense as the surrounding code. Ruff is clean on the three changed files.
- **Legend.** It still uses explicit handles, so the rejected-hatch legend entry
  is unaffected by the hatched swatches.

## Observations (not findings: the user chose this)

- With names removed, colour is now the only patch identifier on this chart.
  The six neutrals are hard to tell apart, the more so where hatching covers the
  three rejected ones, and readers with colour-vision deficiency get no text
  fallback. The patch names are still in the tile overlay's key (`show_tiles()`)
  and on each swatch artist's `label`. The user asked for this trade-off
  explicitly.
- Outside the diff and not caused by it: when exported to PDF at 72 pt/in, the
  right-margin "good <= 2" annotation is slightly cut at the edge
  (`exportpdf-1.png`). The parent commit's PDF (`oldpdf-1.png`) shows the same
  clipping.

## Artifacts

Scratch scripts and renders are in
`/tmp/claude-0/-home-user-PhenoTypic/179627e8-2eff-54b0-a7e4-c4a11b675acc/scratchpad/review/`:
`analyze.py` (the geometry measurements), `export.py`, `old_export.py`
(renders the parent commit), and the PNG, PDF and SVG files named above.
