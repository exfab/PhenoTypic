# Code review — Panel→Plotly/Dash plotting migration

- **Branch:** `worktree-migrate-and-refactor-plotting`
- **HEAD:** `c15c038e959b44e3f57e835c369fb74cc43410c1`
- **Base:** `main` (merge-base diff; branch is ahead 4 / behind 27)
- **Diff:** 57 files, +5036 / −3096 (25 added, 11 deleted, 21 modified)
- **Repo:** `exfab/PhenoTypic` — no open PR; this is a file review.

## Change summary

Removes **Panel** (and HoloViz deps `param`/`bokeh`/`jupyter-bokeh`) and replaces every
Panel dashboard with a shared **Plotly** figure layer plus thin shells (ipywidgets for
notebooks, Dash for the GUI). A new `FigureProvider` mixin + `@figure` decorator (in
`abc_/`) turn figure-builder methods into `.inspect()`/`.dash()`/dashboards; control-free
providers compose a single subplot `go.Figure`, providers with `Control`s get a shell. The
four Panel dashboards (diagnostics, detect-modes, grid-finder, color-correction) are
re-expressed as `FigureProvider` helpers/reports, `MeasureSymmetricZones.inspect` is
converted as the pydantic proof, and a shared Plotly theme lands in `viz/`.

## How this review was produced

Seven independent reviewers (one per coherent diff slice, P1–P7) audited the **code**
against the design contract in `design.md`/`DEFERRED.md` (locked decisions D1–D13) — the
design docs themselves were treated as the contract, not review targets. Each finding below
was then re-verified against source at HEAD and assigned a confidence score (0–100):

> **0** false positive / pre-existing · **25** plausible, unverified · **50** verified but
> minor/rare · **75** verified, important, likely hit in practice or named in CLAUDE.md ·
> **100** certain, frequent, evidence-confirmed.

Per request, **all** findings are surfaced with scores; nothing is filtered out. Builds,
typecheck, lint, and the test suite were **not** run (CI covers those).

---

## Findings

### High confidence (≥ 80)

**[85] `gui/_figure_dashboard.py` added under `gui/` without updating `FEATURES.md` — will block merge.** _(lens: CLAUDE.md)_
Root CLAUDE.md: *"The `gui-checks` workflow's `features-md-gate` job rejects any PR that touches `src/phenotypic/gui/` without modifying `FEATURES.md`."* The diff adds a new file under `src/phenotypic/gui/` but `git diff --name-only main...HEAD` shows no `FEATURES.md` (or `WORKFLOWS.md`) change. The CI `features-md-gate` will reject this regardless of the file being a thin/deferred shell. Add a `FEATURES.md` row (and, if it introduces a user flow, the `WORKFLOWS.md` round-trip).
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/gui/_figure_dashboard.py#L1-L15

**[80] `GridFitReport.dash()` composition drops every panel's annotations — reference-line labels and empty-state placeholders vanish.** _(lens: bug + behavior-regression)_
The compose loop re-adds each sub-figure's `layout.shapes` onto the subplot but never carries `layout.annotations`. Plotly's `add_vline(annotation_text=...)` creates a **shape *and* a separate annotation**, so in the composed dashboard the pitch/cell-area reference lines appear *unlabeled* (e.g. "Expected cell area" at L304; "{label} fit (…)"/"1x ip (…)" at L411/L416). `_empty_figure()` is annotation-only (no trace, no shape), so an empty panel renders as a blank row with no "No objects detected" text — the old matplotlib code drew that text into the axes. Carry `sub.layout.annotations` (remapped to each subplot's axes) alongside the shapes.
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/grid/_grid_fit_report.py#L151-L175

### Medium confidence (50–79)

**[60] `ColorCorrectionReport` declares no `Control`s, contradicting the locked design (ROI selector + shell).** _(lens: design contract)_
`design.md` §9 specifies color-correction as *"ROI selector (select) | yes"* (an ipywidgets/Dash shell). The implementation declares zero `Control`s, so `FigureProvider.dash()` returns a composed `go.Figure` instead of an interactive ROI-selectable shell. `rois` is accepted by `__init__` but never bound to a `Control`. This may be an intentional scope trim, but it is not recorded in `DEFERRED.md`, so the report silently delivers less than the contract.
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/correction/_color_correction/_color_correction_report.py#L126-L140

**[60] `GUI_AVAILABLE` becomes permanently `False` after this branch removes `panel`/`param`.** _(lens: behavior-regression / migration completeness)_
`gui/__init__.py` (not modified by this branch) computes `GUI_AVAILABLE` from `find_spec("panel") and find_spec("param")` — exactly the deps this branch deletes from the `gui` extra. After the migration the GUI is Dash-based, yet the public `GUI_AVAILABLE` flag (exported in `__all__`) will report it unavailable. Internally inert today (only consumer is a type-only assertion in `tests/unit/gui/test_optional_deps.py`), but the check now tests the wrong (removed) libraries. Update `_check_gui_deps()` to test `dash` (or retire the flag).
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/gui/__init__.py#L23-L31

**[55] Plotly theme `paper_bgcolor` is white, but DESIGN.md sets the figure background to `#f5f7fa`.** _(lens: DESIGN.md)_
DESIGN.md's rcParams block sets `"figure.facecolor": "#f5f7fa"` (and `"axes.facecolor": "#ffffff"`). The Plotly equivalents are `paper_bgcolor` ↔ figure facecolor and `plot_bgcolor` ↔ axes facecolor, but the theme sets **both** to `WHITE`. This makes interactive Plotly figures use a white backdrop that doesn't match the `#f5f7fa` backdrop of the static matplotlib renderer (the D6 dual path) or the documented token. Set `paper_bgcolor=BG`.
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/viz/figures/_theme.py#L134-L138

**[50] `_theme.FONT_FAMILY` hardcodes a stripped `"Roboto, sans-serif"` instead of the DESIGN.md font stack.** _(lens: DESIGN.md)_
DESIGN.md §02: *"import the matching FONT_SIZE_\* and FONT_FAMILY_\* constants from `gui/_design.py` — never hardcode … a font-family string."* `_theme.py` hardcodes `"Roboto, sans-serif"`, dropping the cross-platform fallbacks in `_design.FONT_FAMILY_BODY` (`-apple-system`, `Segoe UI`, …), so it degrades to generic sans-serif when Roboto is absent and silently drifts from the GUI font. Note the layering tension — `viz/` deliberately avoids importing `gui/` to stay import-light — so the fix is likely a shared constant, not a `gui` import.
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/viz/figures/_theme.py#L107-L111

**[50] `GridFitReport.fig_successive_diffs` drops the 2× / 3× image-pitch reference lines the old plot had.** _(lens: behavior-regression)_
The old `AutoGridFinder._plot_successive_diffs` drew grey 2× and 3× image-pitch markers (for "spotting sparse-coverage peaks") plus an explicit x-range to keep the 3× marker visible. The new method renders only the green fit-pitch and vermilion 1× lines with Plotly auto-range. The new docstring claims it "Ports" the old method but the port is partial. Low functional impact, but a documented diagnostic was lost.
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/grid/_grid_fit_report.py#L407-L418

### Low confidence (< 50)

**[45] `GridFitReport.fig_axis_occupancy` drops the image-pitch-counts overlay shown on disagreement.** _(lens: behavior-regression)_
The old `_plot_axis_occupancy` overlaid `ip_counts` as grey markers and reported both occupancies when fit and image-pitch disagreed. The new bar chart shows only `fit_counts`. The data still surfaces in the summary `go.Table`, so this is a lost *visual* cross-check, not lost data; docstring documents the reduced behavior.
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/grid/_grid_fit_report.py#L447-L473

**[40] `FigureProvider.inspect()` raises a cryptic `TypeError` if the subject param name is passed as an override.** _(lens: bug, latent)_
`valid` includes the figure method's subject param, so `inspect(image=img)` (or `overrides={"image": …}`) for a `wants_subject` method passes the unknown-override check, then calls `method(subject_positional, image=img)` → `TypeError: got multiple values for argument 'image'` instead of a clean `ValueError`. Currently latent: the only operation proof (`MeasureSymmetricZones`) defines its own `inspect`, so the mixin dispatcher isn't hit by it — but it's a trap for future operations relying on the mixin and called by keyword. Exclude the resolved subject param from `valid`/overrides.
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/abc_/_figure_provider.py#L383-L399

**[35] `_color_correction_report.py` module docstring says the patches strip uses `go.Heatmap`, but the code uses `go.Image`.** _(lens: code-comment)_
The docstring lists *"patches … (`go.Heatmap`)"`* while the implementation builds `go.Figure(go.Image(...))` (and the test asserts `go.Image`). Stale docstring; harmless at runtime.
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/correction/_color_correction/_color_correction_report.py#L17-L20

**[30] Diagnostics `scales` control listed in design.md §9 is omitted with no `DEFERRED.md` reconciliation.** _(lens: design contract / doc sync)_
`design.md` §9 lists `scales` among the diagnostics controls; the code intentionally omits it (an in-code comment explains a list doesn't fit the float/select/bool/text `Control` kinds), but the design table and `DEFERRED.md` aren't updated to record the gap. Doc-sync nit; the code decision is sound and documented inline.
https://github.com/exfab/PhenoTypic/blob/c15c038e959b44e3f57e835c369fb74cc43410c1/src/phenotypic/_core/_image_parts/plot_accessor/_diagnostics_plotter.py#L50-L56

---

## Per-slice coverage

- **P1 — `FigureProvider` contract** (`abc_/_figure_provider.py`, `abc_/__init__.py`, test): decorator/`Control`/`FigureSpec` validation, `iter_figures()` order, `id()`-dedup, `wants_subject`, pydantic-safety all check out. One latent dispatcher edge case (F1).
- **P2 — shared viz layer** (`viz/figures/_theme.py`, `viz/notebook/_adapter.py`, tests): theme registration, notebook adapter render loop + closure binding, and import-boundary test verified sound. Two DESIGN.md token deviations (F2, F3).
- **P3 — diagnostics / detect-modes plotters** (`_diagnostics_plotter.py` +717, `_detect_modes_plotter.py`, `util/image_metrics.py`, tests): dual matplotlib+Plotly path preserved, metric reuse intact, faceting correct. One doc-sync nit (F4).
- **P4 — plot accessors & MRO relink** (`_dash_plot_accessor.py`, `_plot_accessor.py`, `_image_color_handler.py`, deleted panel handler, tests): **clean.** `image.plot.dash.<name>()` dispatch, MRO relink to `ImageColorSpace`, lazy caching, and `image.panel` removal all verified correct.
- **P5 — grid fit report** (`_grid_fit_report.py` +583, `_auto_grid_finder.py` −536, test): no grid-**fitting** algorithm logic was lost (only visualization relocated). Findings are in the relocated diagnostics rendering (F5, F6, F7).
- **P6 — color report + pydantic proof + Dash shell** (`_color_correction_report.py`, `_color_checker_profile.py`, `_measure_symmetric_zones.py`, `gui/_figure_dashboard.py`, tests): pydantic proof (`PrivateAttr` cache, `for_save` flatten) is sound; profile stays lean. Findings: GUI ledger gate (F8), design-control gap (F10), stale docstring (F9).
- **P7 — Panel removal, deps, docs, dangling-ref sweep** (deleted `panel_.py`/`_dashboard_registry.py`/`panel_accessor/`, `pyproject.toml`, `uv.lock`, docs): **clean.** No live dangling references to removed Panel modules/symbols; deps removed from the `gui` extra and absent from the lockfile; public `__init__` exports cleaned. The one residual (F11, `GUI_AVAILABLE`) lives in an unmodified file but is triggered by this branch's dep removal.

## Out of scope

- `design.md` / `DEFERRED.md` prose (treated as the contract, not reviewed).
- Builds, typecheck, lint, and the test suite (CI runs these separately).
- Creating a PR or posting to GitHub.
