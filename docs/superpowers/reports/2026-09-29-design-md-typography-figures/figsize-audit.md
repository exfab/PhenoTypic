# figsize audit (plan Task 2.5)

> **Update 2026-09-30:** `figure_size_mm` no longer caps the height, and
> `MAX_FIGURE_HEIGHT_MM` is gone. Tall multi-panel figures are allowed; the Figures
> section now asks only that nothing overlaps. Findings below that depend on the
> cap are historical.

Date: 2026-09-29. Scope: every `figsize=` under `src/phenotypic` (`*.py`). No call
site was changed. Sizes below are in inches as written in the code; mm = in x 25.4.

## Method and counts

`grep -rn "figsize=" src/phenotypic --include=*.py` returns **64** lines, not the
62 the plan quotes. The two extra lines are the new theme's own docstring
(`src/phenotypic/sdk_/viz/figures/_mpl_theme.py:156`, `:214`), which did not exist
when the plan counted. Of the 64:

| Group | Lines |
|---|---|
| Docstring or doctest only | 12 |
| Error-message text, not a call (`_calibration_overlay.py:565`) | 1 |
| Real sites (sinks, forwarding calls, signature defaults) | 51 |

Many real lines are `figsize=figsize` forwards. Each is listed, but its effective
size is the default defined at the sink or signature named in the row.

Printed-size check: "7 pt at full" is the size 7 pt text ends up at when the figure
is scaled to the 159.2 mm full width (`7 x 159.2 / width_mm`). Text drawn outside
`phenotypic_mpl_context()` uses matplotlib's 10 pt default, so both are given where
it matters.

Facts I did not read in this repo, and so mark **inferred**: matplotlib's default
`figure.figsize` is (6.4, 4.8) in and default `font.size` is 10 pt (library
defaults; `grep "figure.figsize" src/` finds no override); ColorChecker charts
have 24 patches.

Finding while reading: no method in `src/` declares `@figure(backend="mpl")`
(grep for `backend="mpl"` finds only the refusal text at
`src/phenotypic/abc_/plotting/_pht_plot.py:511`). Every `@figure` site in this audit
is `backend="plotly"`. The only matplotlib figures that reach a pipeline store are
`CalibrateColorRpcc.inspect()` pages (not `@figure`, see
`src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py:333`).

## Docstring and doctest only (12, excluded)

`src/phenotypic/sdk_/viz/figures/_mpl_theme.py:156`, `:214`;
`src/phenotypic/_core/_image_parts/accessors/_objmap_accessor.py:385` (10, 8);
`src/phenotypic/_core/_image_parts/accessors/_objmask_accessor.py:301` (8, 8);
`src/phenotypic/_core/_image_parts/accessors/_grid_accessor.py:774` (12, 14),
`:777` (16, 10), `:936` (12, 14), `:939` (16, 10);
`src/phenotypic/analysis/edge/_edge_correction.py:162` (15, 10), `:171` (12, 8);
`src/phenotypic/analysis/filter/_mad_outlier.py:274` (12, 5);
`src/phenotypic/analysis/filter/_tukey_outlier.py:248` (12, 5).

These teach users sizes of 305 to 406 mm wide. If the call sites move to presets,
these examples should move with them, or the docs will keep teaching the old sizes.

## Class 1: publication candidates (5)

| Site | Function | figsize | Backend path | Audience |
|---|---|---|---|---|
| `src/phenotypic/analysis/abc_/_model_fitter.py:433` | `ModelFitter.show` signature | default (6, 4) | analyzer `show()`, mpl | notebook user; manuscript export |
| `src/phenotypic/analysis/abc_/_model_fitter.py:486` | `ModelFitter.show` | `plt.subplots(figsize=figsize)` | analyzer `show()`, mpl sink | same |
| `src/phenotypic/analysis/abc_/_model_fitter.py:626` | `ModelFitter._build_plotly_figure` signature | default (6, 4) | Plotly; only `figsize[1] * 100` is used, as height px (`:816`) | GUI analysis panel via `report()`; saved Plotly figure |
| `src/phenotypic/analysis/abc_/_model_fitter.py:892` | `ModelFitter.inspect` signature | default (6, 4) | Plotly, forwards | same |
| `src/phenotypic/analysis/abc_/_model_fitter.py:923` | `ModelFitter.inspect` | forward | Plotly | same |

Growth curves (mean +/- SE per group) are the clearest paper figure in the package.

Suggested preset for the mpl sink (`:433`/`:486`), aspect h/w = 4/6 = 0.6667:

- `half`: 77.1 x 4/6 = **51.4 mm** -> `figure_size_mm("half", 51.4)`. Recommended:
  one growth panel is a single-column figure in most journals.
- `full`: 159.2 x 4/6 = **106.1 mm** -> `figure_size_mm("full", 106.1)`.

Current size is 152.4 mm wide, so at full width it prints at 1.045x (7 pt -> 7.3 pt,
fine). Placed in a half-width slot it is scaled 0.506x: 7 pt -> 3.5 pt, and the
untheme'd 10 pt default -> 5.1 pt. That is the practical risk today.

The three Plotly lines cannot take `figure_size_mm` directly: width is autosized and
the tuple becomes pixels at 100 px/in. They need a Plotly mm-to-px mapping first
(out of scope for Task 2.5).

## Class 2: dynamic (16)

| Site | Function | Size expression | Depends on | Backend / audience |
|---|---|---|---|---|
| `src/phenotypic/analysis/edge/_edge_correction.py:245` | `EdgeCorrector._show_collapsed` | `(10, max(6, 0.5*n_groups + 2))` at `:243` | groups shown (max 20) | analyzer `show()`, mpl; notebook, GUI analysis panel PNG fallback (`src/phenotypic/_gui/analysis/_render.py:87`, themed, dpi 110) |
| `src/phenotypic/analysis/edge/_edge_correction.py:452` | `EdgeCorrector._show_individual` | `(5*n_cols, 4*n_rows)`, `n_cols = min(3, n)` | groups shown | same |
| `src/phenotypic/analysis/filter/_mad_outlier.py:359` | `MADOutlierRemover._show_individual` | `(5*n_cols, 4*n_rows)` | groups | same |
| `src/phenotypic/analysis/filter/_mad_outlier.py:512` | `MADOutlierRemover._show_collapsed` | `(10, max(6, 0.5*n + 2))` | groups | same |
| `src/phenotypic/analysis/filter/_tukey_outlier.py:337` | `TukeyOutlierRemover._show_individual` | `(5*n_cols, 4*n_rows)` (`:334`) | groups | same |
| `src/phenotypic/analysis/filter/_tukey_outlier.py:487` | `TukeyOutlierRemover._show_collapsed` | `(10, max(6, 0.5*n + 2))` (`:485`) | groups | same |
| `src/phenotypic/correction/_color_correction/_calibration_overlay.py:569` | `render_calibration_overlay` | `(need_w, need_h)` from measured text, `_IMAGE_W_IN = 1.6` per ROI | ROI count, label widths | mpl `Figure`, themed, dpi 160; pipeline store PNG page `tiles`; notebook |
| `src/phenotypic/correction/_color_correction/_calibration_overlay.py:768` | `render_delta_e_bars` | `(max(4.5, 0.34*n_tiles + 1.2), 3.4)` at `:767` | scored tiles | mpl, themed; store PNG page `delta_e` |
| `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py:286` | `CalibrateColorRpcc.show_tiles` | forward (None) | as `:569` | as `:569` |
| `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py:317` | `CalibrateColorRpcc.show_delta_bar_plot` | forward (None) | as `:768` | as `:768` |
| `src/phenotypic/_core/_image_parts/plot_accessor/_diagnostics_plotter.py:1438` | `DiagnosticsPlotter.diagnostics` | forward (None) | as `:1544` | notebook |
| `src/phenotypic/_core/_image_parts/plot_accessor/_diagnostics_plotter.py:1544` | `DiagnosticsPlotter._diagnostics_matplotlib` | `(14.0, 4 + per section 2.5 [+1.0] [+1.5])` at `:1514` | sections, flags | mpl report figure; notebook |
| `src/phenotypic/measure/_measure_symzones.py:618` | `MeasureSymZones._build_plate_overview` | `(900//100, int(900*h/w)//100)` | plate aspect | Plotly `@figure` (`:475`); store figure, notebook |
| `src/phenotypic/measure/_orientation_zones/_figures.py:224` | `_OrientationZonesFigures.inspect` | same pattern | plate aspect | Plotly `@figure` (`:155`) |
| `src/phenotypic/measure/_orientation_zones/_figures.py:312` | `.cumulative_rotation_overlay` | same pattern | plate aspect | Plotly `@figure` (`:248`) |
| `src/phenotypic/measure/_orientation_zones/_figures.py:442` | `.matched_cumulative_rotation_overlay` | same pattern | plate aspect | Plotly `@figure` (`:352`) |

Does a preset make sense?

- **Six analyzer `show()` helpers (Edge, MAD, Tukey): yes, width only.** Fix the
  width at `full` and derive height from the data, keeping the current aspect.
  Collapsed: h/w = max(6, 0.5n + 2)/10, so height = 159.2 x that.
  n <= 8: 159.2 x 0.6 = **95.5 mm**; n = 20 (the `max_groups` cap):
  159.2 x 12/10 = **191.0 mm**, under the 246.2 mm limit.
  Individual: panel aspect 4/5; with 3 columns, height = 159.2 x 4r/15 =
  42.45 mm per row. 7 rows (n = 20) needs **297.2 mm**, which exceeds
  `MAX_FIGURE_HEIGHT_MM` (246.2), so `figure_size_mm` would raise; the grid
  must cap at 5 rows (212.3 mm) per figure or paginate. These plots are
  plausible supplementary figures (edge effect, outlier QC), which is why they
  lead the priority list despite being dynamic.
- **Calibration overlay (`:569`): no.** It already draws in the theme and sizes
  itself in inches from measured 7/8 pt text, so it prints at true size when not
  scaled. It refuses a smaller `figsize` by design (`:563`). Only risk: many ROIs
  exceed 159.2 mm (inferred; each ROI is at least 1.6 in = 40.6 mm of image
  plus labels, so roughly 3 ROIs fill a page).
- **Delta E bars (`:768`): partly.** Height 3.4 in = 86.4 mm is fine. Width
  passes full (159.2 mm = 6.27 in) once 0.34n + 1.2 > 6.27, that is n >= 15
  scored tiles. For a 24-patch chart (inferred count): 9.36 in = 237.7 mm,
  scaled 0.670x at full width, so 7 pt -> **4.7 pt**. A cap at `full` with a
  tighter pitch (or two rows) would keep 7 pt.
- **Diagnostics report (`:1544`): no.** It is a multi-panel dashboard. All four
  sections plus descriptions and recommendations give (14, 19.5) in =
  355.6 x 495.3 mm; at full width 0.448x, so 7 pt -> 3.1 pt and 10 pt -> 4.5 pt.
  Treat as screen-only.
- **Four Plotly plate overlays: no.** The tuple is converted to pixels at 100 px/in
  (`src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_dash_handler.py:133`),
  so width is always 900 px. `display_h // 100` floors, so height can be up to
  99 px short of the plate aspect (read; the visual effect is inferred). Not an mm
  preset target.

## Class 3: diagnostic / interactive (30)

All are image-inspection views of one plate. They answer "is this segmentation or
grid right", and users who need a paper panel crop the image instead. Default
`None` falls to `plt.subplots(figsize=None)` at
`src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_mpl_handler.py:277`,
that is matplotlib's (6.4, 4.8) in (inferred default).

Image accessor `.show()` (mpl), audience notebook user:

- `src/phenotypic/_core/_image_parts/_image_handler.py:744`, `:749` (`ImageHandler.show`, forward)
- `src/phenotypic/_core/_image_parts/accessor_abstracts/_multichannel_accessor.py:176` (overlay), `:192`, `:195` (`MultiChannelAccessor.show`)
- `src/phenotypic/_core/_image_parts/accessor_abstracts/_single_channel_accessor.py:85` (overlay), `:100` (`SingleChannelAccessor.show`)
- `src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_mpl_handler.py:277` (`_mpl_plot`, the sink), `:421` (`_plot_overlay`)
- `src/phenotypic/_core/_image_parts/accessors/_objmap_accessor.py:343` (default None), `:396` (`ObjectMap.show`)
- `src/phenotypic/_core/_image_parts/accessors/_objmask_accessor.py:308` (`ObjectMask.show`)

Image accessor `.dash()` (Plotly; tuple -> px at 100 px/in, None autosizes), audience notebook / GUI:

- `src/phenotypic/_core/_image_parts/_image_handler.py:781`, `:786` (`ImageHandler.dash`)
- `src/phenotypic/_core/_image_parts/accessor_abstracts/_multichannel_accessor.py:270`, `:285`, `:289` (`MultiChannelAccessor.dash`)
- `src/phenotypic/_core/_image_parts/accessor_abstracts/_single_channel_accessor.py:164`, `:178` (`SingleChannelAccessor.dash`)
- `src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_dash_handler.py:359` (`_plotly_overlay`)

Fixed-size inspection figures (mpl, notebook):

| Site | Function | figsize | Width mm | 7 pt at full | 10 pt at full |
|---|---|---|---|---|---|
| `src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_mpl_handler.py:198`, `:202` | `histogram`, 2-D case | (10, 5) at `:143` | 254.0 | 4.4 pt | 6.3 pt |
| `src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_mpl_handler.py:215`, `:221` | `histogram`, 3-D case | (10, 5) | 254.0 | 4.4 pt | 6.3 pt |
| `src/phenotypic/_core/_image_parts/color_space_accessors/_hsv_accessor.py:148` | `HsvAccessor.histogram` | (10, 5) at `:103` | 254.0 | 4.4 pt | 6.3 pt |
| `src/phenotypic/_core/_image_parts/color_space_accessors/_hsv_accessor.py:248` | `HsvAccessor.show` | (10, 8) at `:218` | 254.0 | 4.4 pt | 6.3 pt |
| `src/phenotypic/_core/_image_parts/color_space_accessors/_hsv_accessor.py:312` | `HsvAccessor.show_objects` | (10, 8) at `:279` | 254.0 | 4.4 pt | 6.3 pt |
| `src/phenotypic/_core/_image_parts/accessors/_grid_accessor.py:784` | `GridAccessor.show_column_overlay` | (9, 10) at `:737` | 228.6 | 4.9 pt | 7.0 pt |
| `src/phenotypic/_core/_image_parts/accessors/_grid_accessor.py:948` | `GridAccessor.show_row_overlay` | (9, 10) at `:900` | 228.6 | 4.9 pt | 7.0 pt |
| `src/phenotypic/measure/_measure_color_composition.py:638` | `MeasureColorComposition.visualize_masks` | (15, 10) at `:574` | 381.0 | 2.9 pt | 4.2 pt |

`visualize_masks` is documented "for debugging purposes" (`:576`). The grid
overlays and HSV views could appear in a methods figure, but their subject is a
plate image whose useful size is the image, so they keep their own size.

## Sites whose printed text would be wrong

Scaled down (text too small) if printed to fit A4 text width:

- `src/phenotypic/measure/_measure_color_composition.py:638` (15, 10): 381 mm, 7 pt -> 2.9 pt.
- `src/phenotypic/_core/_image_parts/plot_accessor/_diagnostics_plotter.py:1544` default (14, 19.5): 356 x 495 mm, 7 pt -> 3.1 pt.
- Analyzer individual grids at 3 columns, `src/phenotypic/analysis/edge/_edge_correction.py:452`, `src/phenotypic/analysis/filter/_mad_outlier.py:359`, `src/phenotypic/analysis/filter/_tukey_outlier.py:337` (15 in wide): 381 mm, 7 pt -> 2.9 pt; at 7 rows the height (711 mm, 297 mm after scaling) does not fit a page at all.
- Analyzer collapsed views `src/phenotypic/analysis/edge/_edge_correction.py:245`, `src/phenotypic/analysis/filter/_mad_outlier.py:512`, `src/phenotypic/analysis/filter/_tukey_outlier.py:487` (10 in wide): 7 pt -> 4.4 pt; their `legend_fontsize` 9 pt -> 5.6 pt.
- Histograms and HSV views (10 in): 7 pt -> 4.4 pt. Grid overlays (9 in): 4.9 pt.
- `src/phenotypic/correction/_color_correction/_calibration_overlay.py:768` with >= 15 scored tiles; 24 tiles gives 4.7 pt.

Scaled up (text too large) if printed at a fixed slot width:

- `ModelFitter.show` placed at `full` is fine (1.045x). An analyzer individual grid
  with one group (5, 4) in = 127 mm, printed full width: 1.254x, 7 pt -> 8.8 pt
  and 10 pt -> 12.5 pt.
- Delta E bars at the 4.5 in minimum (114.3 mm) stretched to full: 1.393x,
  7 pt -> 9.7 pt. Printed at its own size it is correct.

## Docstring drift noticed

`_mpl_plot` says a `None` figsize is "automatically calculated as integer
dimensions in [6, 30] that best match the image aspect ratio"
(`src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_mpl_handler.py:265`
), and the accessor `show()` docstrings say "auto-calculated from array
aspect ratio". The code at `:277` passes `None` straight to `plt.subplots`, so no
aspect-based sizing happens. Out of scope here; worth a separate fix.

## Summary

| Class | Real sites | Notes |
|---|---|---|
| Publication candidate | 5 | 2 mpl (`ModelFitter.show`), 3 Plotly (`ModelFitter` inspect path) |
| Dynamic | 16 | 6 analyzer `show()` helpers, 4 RPCC calibration, 2 diagnostics report, 4 Plotly plate overlays |
| Diagnostic / interactive | 30 | 12 accessor `.show()` mpl, 10 accessor `.dash()` Plotly, 8 fixed-size inspection figures |
| Total | 51 | plus 12 docstring-only and 1 error-message line = 64 |

Migrate first, in order:

1. `src/phenotypic/analysis/abc_/_model_fitter.py:433` / `:486`: default to
   `figure_size_mm("half", 51.4)`. Growth curves are the most-published output and
   the change is one default.
2. The six analyzer helpers (`src/phenotypic/analysis/edge/_edge_correction.py:245`, `:452`;
   `src/phenotypic/analysis/filter/_mad_outlier.py:359`, `:512`;
   `src/phenotypic/analysis/filter/_tukey_outlier.py:337`, `:487`): full width,
   data-driven height, capped at 246.2 mm (paginate the individual grid beyond 5
   rows). One shared helper would cover all six, since the formulas are identical.
3. `src/phenotypic/correction/_color_correction/_calibration_overlay.py:768`: cap
   the default width at `full` so a 24-patch chart keeps 7 pt. It already stores
   in every pipeline run that uses `CalibrateColorRpcc`.
4. The Plotly `ModelFitter` path (`src/phenotypic/analysis/abc_/_model_fitter.py:626`, `:892`)
   once a Plotly mm-to-px mapping exists.
5. Update the 12 docstring examples to match whatever lands in 1 to 3.

Leave alone: the 30 diagnostic sites, the calibration tile overlay (already
text-sized in the theme), the diagnostics report, and the four Plotly plate overlays.
