# Calibration tile overlay figure — spec

**Status:** proposed, pending user review
**Module:** `phenotypic.correction` (`CalibrateColorRpcc`)
**Builds on:** PR #235, in-frame checker colour correction (merged); spec
`docs/superpowers/specs/2026-09-21-in-frame-checker-color-correction/README.md`
**Mockups:** `docs/superpowers/artifacts/2026-09-22-checker-calibration-overlay/`
(`panel-b-matplotlib.png` is the approved look; `figure-mockup.html` is the wider
exploration it was cut down from)

---

## Objective

Let a user look at one frame and see whether `CalibrateColorRpcc` read its
colour checker correctly: **where** it measured each tile, **which chart patch**
it matched each tile to, and **how far** each tile was from its reference colour
before and after correction.

That view is the only direct check that the correction is trustworthy. The QC
gate's numbers (placement margin, impurity, ΔE statistics) summarise the
detection. The overlay shows it, and a misplaced box or a wrong label is obvious
at a glance in a way no summary number is.

## Non-goals

- **Saving or publishing the figure.** This covers writing the record into the
  OME-Zarr store, `@figure` / `PlotImage` wiring, and CLI publication under
  `deliverables/plots/`. Another session owns those. This spec produces the two
  things that work needs: a plain-data record, and a pure render function that
  takes it (§Interfaces).
- **The other mockup panels.** The summary card (A), the chart-layout grid (C)
  and the QC table (D) are not built. The user chose panel B alone for
  simplicity.
- **Plotly.** matplotlib was chosen over plotly. matplotlib can measure its own
  text, so label columns are sized from real widths and overlap can be tested.
  It also writes a static PNG with no browser engine and no 5 MB embedded JS
  bundle.
- **Configurable ΔE bands.** They are fixed (§Colour).

## Background

After `apply()`, the operation already holds everything the figure needs except
the pixels:

| Needed | Source today |
|---|---|
| tile boxes (full and core) | `diagnostics["lattices"][i]` → `CheckerLattice.boxes(rot=lattice.rot)` and `boxes(core=core_trim, rot=lattice.rot)` |
| patch name, measured colour, impurity, pixel count | `diagnostics["tiles"]` |
| ΔE00 before / after, rejected outliers | `fitted_profile.diagnostics["patches"]`, `["rejected_patches"]` |
| reference colour | `_load_reference_data(checker_type, target_illuminant)` |
| per-ROI flags and warnings | `diagnostics["qc"]` / `self.qc` |
| the ROI pixels, **as shot** | **not kept.** `apply()` corrects the image in place, so by the time anything draws, the input pixels are gone. |

The pixels are the gap. The figure must draw on the as-shot pixels the medoid
was taken from. On corrected pixels, every swatch comparison would be circular.

**What is lost today when a frame is refused.** `_operate` raises before
`_build_diagnostics` runs on the refusal paths
(`_calibrate_color_rpcc.py:392–430`: gate failure under `"raise"`, "No ROI
produced usable tiles", and both `require_rank` calls). A refused frame is the
one a user most needs to look at, so the record must be built **before** those
raises.

## Design

### 1. `CalibrationOverlayRecord` — plain data, built during `apply()`

A frozen pydantic model in a new private module
`correction/_color_correction/_calibration_overlay.py`
(`arbitrary_types_allowed=True`; not exported from `phenotypic.correction`).

```text
CalibrationOverlayRecord
  image_name: str | None
  verdict: Literal["corrected", "corrected_with_warnings", "skipped", "refused"]
  degree: int
  n_fitted: int | None            # None when no fit happened
  n_expected: int
  refusal: str | None             # the gate / rank / no-tiles message, when refused
  rois: list[RoiOverlay]

RoiOverlay
  roi_index: int
  label: str | None
  crop: np.ndarray                # as-shot ROI pixels, (h, w, 3), OWNS its buffer
  lattice_found: bool
  n_tile_columns: int             # 0 when no lattice
  flags: list[str]
  warnings: list[str]
  tiles: list[TileOverlay]
  unidentified_boxes: list[tuple[Box, Box]]   # (full, core); default empty

TileOverlay
  row: int; col: int
  patch: str
  status: Literal["used", "partly_covered", "rejected", "excluded", "empty"]
  full_box: Box                   # Box = (y0, y1, x0, x1), floats, ROI-local
  core_box: Box
  measured_srgb: tuple[float, float, float] | None   # the medoid pixel's own sRGB; None for an empty tile
  reference_srgb: tuple[float, float, float]
  impurity: float | None
  delta_e_before: float | None
  delta_e_after: float | None
```

**`unidentified_boxes`** (amended 2026-09-22, review F2). A ROI can be refused
*after* its lattice was found: its detected tile count disagrees with
`expect_tiles`, its tile block fits the chart in fewer than two placements, or
every tile box falls outside the ROI. No tile was identified, so `tiles` is
empty, but the detected boxes are what explains the refusal. The record keeps
them as `(boxes(rot=lattice.rot), boxes(core=core_trim, rot=lattice.rot))`
pairs, whenever the lattice is set and `tiles` is empty. It is empty in every
other case, including a ROI with no lattice.

**Verdict**, decided once when the record is built:

| verdict | rule |
|---|---|
| `refused` | `_operate` raised after the ROI loop: gate failure under `on_qc_fail="raise"`, no usable tiles, or a rank failure. `refusal` holds the message. |
| `skipped` | the gate failed under `on_qc_fail="skip"`; the image is returned uncorrected |
| `corrected_with_warnings` | the fit ran and any ROI has flags (only possible under `"warn"`) or warnings |
| `corrected` | the fit ran, and no ROI has flags or warnings |

**Tile status**, decided once when the record is built, never by the renderer.
The rules are tried **top to bottom, and the first that matches wins**
(amended 2026-09-22, review F3 and the precedence note). `excluded` appears
twice because its two causes sit on either side of `rejected`:

| # | status | rule |
|---|---|---|
| 1 | `empty` | `n_pixels == 0`, so the box fell outside the ROI |
| 2 | `excluded` | this ROI lost a patch collision: it did not claim the name in `claimed_by`. A losing ROI's tile is `excluded` even when the winner's patch of that name was rejected, because that rejection is not its own. |
| 3 | `rejected` | the patch is in `fitted_profile.diagnostics["rejected_patches"]`. This includes a post-rejection rank refusal, where the fit ran but was not accepted, because which tiles were rejected is what explains that refusal. |
| 4 | `excluded` | no accepted fit: the frame was skipped or refused |
| 5 | `partly_covered` | would be `used`, but `impurity > qc_limits.max_tile_impurity` |
| 6 | `used` | its colour was in the fit |

`delta_e_before` and `delta_e_after` are `None` unless the fit ran and was
accepted, and are attached only to `used`, `partly_covered` and `rejected`
tiles. A `rejected` tile on a refused frame therefore has no ΔE. The
renderer then prints "not fitted" in their place. `reference_srgb` always comes
from `_load_reference_data`, so a refused frame still shows its reference
swatches.

**Lifecycle on the operation:**
- The record is a private attribute exposed read-only as
  `op.calibration_record: CalibrationOverlayRecord | None`.
- It is reset to `None` at the top of `_operate`, next to `fitted_profile`,
  `qc` and `_diagnostics` (line 261).
- It is assigned on **every** path that returns or raises after the ROI loop:
  the skip return, the success return, and immediately before each refusal
  raise. A refused frame therefore always has a record, and a successful run
  never leaves a stale one.
- It is assigned **before** the gate's `warnings.warn`, so warnings-as-errors
  still leave one: `skipped` under `"skip"`, and under `"warn"` a provisional
  `refused` record carrying the gate message, which every later exit replaces.
- The `corrected*` record is assigned only after `ColorCorrector.apply()`
  succeeds. If the correction raises, the record is `refused` with
  `refusal="correction failed: ..."`, and the error propagates.
- It is never serialised (`PrivateAttr`), like `_diagnostics`.

**Crops own their memory, and are read-only.** Each `crop` is
`np.array(sub, copy=True)`, and `build_overlay_record` sets its
`flags.writeable = False` to match the frozen record. It is
never a view into `image.rgb`, because a view would pin the whole-image buffer
(`abc_/CLAUDE.md`: cached crops must own their buffers). It stores the as-shot
`image.rgb` ROI slice, taken inside the ROI loop **before** correction runs, in
the image's own dtype. Cost: two rig bands of about 2150 × 340 × 3 `uint8` come
to about 4.4 MB per operation instance, freed at the next `apply()`.

### 2. `render_calibration_overlay(record, *, figsize=None) -> Figure`

A pure function in the same module. It reads nothing but the record: no
operation, image or pipeline. That is what lets the other session render a
record it loaded from a store.

**Layout:**
- One group per ROI, laid left to right. Each group is three `GridSpec`
  columns: left label column, image, right label column.
- The label axes use `sharey` with the image axes, so every label sits at its
  tile's centre row by construction.
- Tile column 0 is labelled on the left and tile column 1 on the right.
- A label block is:
  - a measured | reference swatch pair, drawn next to the image;
  - the patch name, plus a status suffix (`· 24% covered`, `· rejected`,
    `· excluded`, `· empty`);
  - a second line, `ΔE00 {before:.1f} -> {after:.1f}`, or `ΔE00 not fitted`.
- Boxes on the image:
  - every `full_box` as a thin dotted white outline;
  - every `core_box` as a 1.8 pt outline in its status colour;
  - dashed outlines for `rejected`, `excluded` and `empty`;
  - every `unidentified_boxes` pair drawn the same way as an `excluded`
    tile: the thin dotted white full box, and a dashed `#BBBBBB` core box.
    These boxes get **no labels** and no swatches, because there is no
    identity to show.
- Below each group, spanning its three columns, the ROI's flags and warnings
  are printed, wrapped by measured width. Nothing is printed when both are
  empty.
- A ROI with `lattice_found=False` shows its crop, no boxes, and its flags.
- Figure title: `{image_name} · {verdict} · degree {degree} · {n_fitted}/{n_expected} patches fitted`.
  The count reads `not fitted` when `n_fitted` is `None`.

**No overlapping text, by construction and by test:**
- **Width.** Each label column is exactly as wide as its widest label block,
  plus swatch and padding. Label widths are measured with the Agg renderer at
  the output DPI (`Text.get_window_extent`) before the figure is laid out, not
  guessed.
- **Height.** The figure is tall enough that, in every label column, the
  smallest vertical gap between neighbouring tile centres is at least one label
  block's height. The image height follows from the crop's aspect ratio; when
  that is too short, the height grows.
- **More than two tile columns** (for example a full 4 × 6 card in one ROI):
  middle tiles have no free side. Every tile then gets a small number drawn at
  its core-box centre, and the label blocks move to a key below the group,
  flowed into as many sub-columns as fit. Swatches and ΔE text stay the same.
- **`figsize`** overrides the computed size. The same measurements then check
  it: when the given size is too small for the labels, `render_calibration_overlay` raises
  `ValueError` naming the size it needs. Silently drawing overlapped text is
  what this function exists to prevent.

**Conventions:**
- An explicit `matplotlib.figure.Figure`, never `pyplot`.
- Built inside `phenotypic_mpl_context()`.
- `matplotlib` and `colour` are imported inside the function, so
  `tests/unit/ci/test_deferred_imports.py` and `test_startup_imports.py` stay
  green, with the new module added to the deferred-imports allow-list.
- **Default text styling (user decision, 2026-09-22).** No text sets a font
  size, family or colour; every text takes the `phenotypic_mpl_context()`
  defaults. The layout adapts to whatever those defaults measure, and only
  the layout changes to avoid overlap. The only colours are the two meaning-
  carrying systems in §Colour. A glyph the default font lacks is replaced in
  the text, not by changing the font: the first prototype's fallback font had
  no "→", so the ΔE line uses "->".

### 3. `CalibrateColorRpcc.show_tiles(*, figsize=None) -> Figure`

The notebook entry point:

```python
out = op.apply(image)
fig = op.show_tiles()
fig.savefig("calibration.png", dpi=160)
```

It returns `render_calibration_overlay(self.calibration_record, figsize=figsize)`.
It raises `RuntimeError("show_tiles() draws the last apply(); call apply() first")`
when the record is `None`. On a refused frame, `apply()` raises (under
`on_qc_fail="raise"`) but still leaves a record behind, so `show_tiles()` works
after catching the error. The docstring shows that pattern.

The name `show_tiles` rather than `show` is deliberate: the repo's `show()`
methods return `(Figure, Axes)` for a single axes. This figure has three axes
per ROI, and the entry point returns the `Figure` alone.

### Colour

- **Tile status**, Okabe-Ito semantic assignments from `DESIGN.md` §01:
  `used` `#009E73`, `partly_covered` `#E69F00`, `rejected` `#D55E00`,
  `excluded` and `empty` `#BBBBBB`.
- **ΔE00 after-values** use fixed bands, drawn in the darkened text variants
  `DESIGN.md` requires for data colours on a light ground:
  `≤ 2` green `#007a5a`, `2–5` orange `#a86f00`, `> 5` vermilion `#b04a00`.
  The band edges are module constants, `DELTA_E_GOOD = 2.0` and
  `DELTA_E_FAIR = 5.0`.

## Interfaces handed to the saving / publishing session

- `CalibrationOverlayRecord` is plain data: arrays, numbers and strings. That
  session decides how to persist it, for example as an attribute plus a small
  array inside the image's store. It rebuilds the record and calls
  `render_calibration_overlay(record)` from any CLI path, including re-measure
  and staged-GPU Stage 3, where `apply()` did not run in the drawing process.
- Nothing in this spec changes what `apply()` returns, the store, the CLI or
  `deliverables/`.

## Testing

All tests use the rig-shaped synthetic frame and planted faults from
`tests/unit/correction/test_calibrate_color_rpcc_review.py`: three neutrals
painted green give rejected outliers, and a grey occluder over orange gives a
partly covered tile. The frame helpers move into a shared test module rather
than being copied.

1. **Record fidelity.** Each tile's status matches the rule table. ΔE values
   equal `fitted_profile.diagnostics["patches"]`. `measured_srgb` equals the
   `diagnostics["tiles"]` medoid sRGB. Boxes equal `CheckerLattice.boxes(...)`.
2. **The as-shot crop is as shot.** `record.rois[i].crop` equals the input
   image's ROI slice, not the corrected output's. It owns its buffer
   (`crop.base is None`).
3. **Refused frames keep a record.** This covers the `"raise"` gate failure,
   "No ROI produced usable tiles", and post-rejection rank failure. After
   catching the `RuntimeError`, the record exists, its verdict is `refused`,
   its tiles are `excluded`, and ΔE is `None`. The exception is the
   post-rejection rank failure, whose rejected tiles are `rejected` (still
   with no ΔE). A skipped frame's verdict is `skipped`. A ROI refused after
   its lattice was found keeps `unidentified_boxes`, and the figure draws
   them.
4. **Per-run reset.** Apply a clean frame, then a refused one: the second
   record describes only the second frame. Calling `show_tiles()` before any
   `apply()` raises `RuntimeError`.
5. **No overlapping text, no clipping.** Render, draw with Agg, and collect
   every `Text` artist's window extent. No two extents intersect, and every
   extent lies inside the figure. Run this on the two-band frame, on a frame
   with the longest real patch names ("neutral 6.5 (.44 D) · rejected"), and on
   a single ROI holding a full 4 × 6 card, which exercises the numbered-key
   path.
6. **The overlap test can fail.** Force the label column width to half its
   measured value, and test 5 must fail. Pass a `figsize` too small for the
   labels, and `render_calibration_overlay` must raise `ValueError`.
7. **Structure.** One image axes per ROI; one core-box patch per tile; box
   edge colours equal the status colours; a ROI with no lattice has an image
   and no box patches.
8. **Lazy imports.** `tests/unit/ci/test_deferred_imports.py` and
   `test_startup_imports.py` pass with the new module registered.

## Out of scope, by decision

- Saving the record into the store, `@figure` / `PlotImage` capability, CLI
  publication: owned by another session (§Interfaces).
- Panels A, C, D of the mockup.
- Plotly rendering.
- Configurable ΔE bands.
