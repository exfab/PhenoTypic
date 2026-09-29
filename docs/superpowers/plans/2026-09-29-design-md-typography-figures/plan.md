# Nunito Sans chrome, figure defaults and DESIGN.md format: implementation plan

**Goal:** Carry out three settled specs in dependency order: the GUI chrome moves
from Comfortaa to Nunito Sans, the matplotlib theme adopts the static-figure defaults,
and `DESIGN.md` is restructured into the Google DESIGN.md format with a new Figures
section.

**Specs** (read the relevant one before each phase):

- `docs/superpowers/specs/2026-09-29-chrome-font-nunito-sans/design.md` (Phase 1)
- `docs/superpowers/specs/2026-09-29-figure-style-defaults/design.md` (Phase 2, and
  the Figures section text in Phase 3)
- `docs/superpowers/specs/2026-09-29-design-md-format/design.md` (Phase 3)

**Tooling:** `uv` for everything Python; `npx @google/design.md@0.4.0` for the
DESIGN.md linter (Node 22 is available in the cloud container). Pass explicit paths to
`ruff check --fix`. Use the `run-phenotypic-test` skill before any wide pytest run;
GUI tests need `QT_QPA_PLATFORM=offscreen`.

**Order.** Phase 1 lands first because the front matter in Phase 3 records its tokens.
Phase 2 is independent of Phase 1 and may run in parallel with it, but must land
before Phase 3, whose Figures section describes the theme Phase 2 ships. Each phase
leaves the tree green and could merge on its own.

**Out of scope, by the specs' non-goals:** tests that compare `DESIGN.md` with code, a
CI gate on the DESIGN.md linter, per-journal presets, Plotly chart styling, and
migrating the existing `figsize=` call sites.

---

## Phase 1: Nunito Sans chrome

> **Status (2026-09-29):** done. The first screenshot capture ran while the cloud
> environment blocked `cdn.jsdelivr.net` (Dash Bootstrap's stylesheet) and was
> discarded; the regeneration after network access opened is committed, and the
> headings-versus-UI-titles check was made on those screenshots.
> Beyond the plan: the 600 italic is loaded too (headings are 600), five CSS blocks
> in `DESIGN.md` split one token per line by commit `92e3dab0` were restored from
> its parent, and the CLI dashboard and processing report were moved off the
> retired DM font trio onto the same tokens (`GOOGLE_FONTS_URL`,
> `type_tokens_css()`).

### Task 1.1: Role fonts

**Files:** `src/phenotypic/_gui/_design.py`, `tests/unit/gui/test_config_and_design.py`

- [x] In `_design.py`, set `_DISPLAY_PRIMARY`, `_BODY_PRIMARY` and `_SPECIES_PRIMARY`
  (`:168` to `:171`) to `"Nunito Sans"`, and point `_FALLBACK_SPECIES` at
  `_FALLBACK_SANS`.
- [x] Replace the Comfortaa entry of `_GOOGLE_FONTS_URL` (`:181`) with
  `family=Nunito+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400`. Keep the IBM Plex
  Serif, IBM Plex Sans and JetBrains Mono entries; the chart subsystem still loads the
  first two.
- [x] Rewrite the role comment block (`:6`, `:148` to `:193`) for the new split:
  Nunito Sans for display, body and species, JetBrains Mono for all data and table
  values, IBM Plex loaded only for charts.
- [x] Update `test_font_family_constants_carry_role_fonts` (`:431` to `:443`):
  display, body and species start with `'Nunito Sans'`; species now ends with the sans
  stack rather than `serif`.
- [x] Run `uv run pytest tests/unit/gui/test_config_and_design.py -q`.

### Task 1.2: Size ladder and data alias

**Files:** `src/phenotypic/_gui/_design.py`, `tests/unit/gui/test_config_and_design.py`

- [x] Add the primitive `TEXT_DATA = "0.9375rem"` between `TEXT_SM` and `TEXT_BASE`.
  Set `TEXT_BASE = "1rem"` and `TEXT_MD = "1.125rem"`, and update the pixel comments
  beside them.
- [x] Add the alias `FONT_SIZE_DATA = TEXT_DATA`, export it in `__all__`, emit
  `--text-data` and `--font-size-data` in the CSS token block, and point `.text-data`
  and `.text-data--muted` (`:712`, `:713`) at `var(--font-size-data)`.
- [x] Extend `test_type_scale_is_monotonic`,
  `test_semantic_font_size_aliases_resolve_to_primitives` and
  `test_semantic_font_size_aliases_cover_full_scale` with `TEXT_DATA` and
  `FONT_SIZE_DATA`, and update the docstrings that say "eight".
- [x] Run the same test file.

### Task 1.3: Display weight 600

**Files:** `src/phenotypic/_gui/_design.py`

- [x] Change `font-weight: 400` to `600` in `.text-display`, `.text-title`,
  `.text-header`, `.text-h2` and `.text-h3` (`:696` to `:700`). `DESIGN.md` mentions
  `TEXT_STYLE_*` Python constants, but `_design.py` defines none, so the CSS classes
  are the only place to change; Task 1.5 corrects that mention.
- [x] Grep the tests for a pinned heading weight of 400 and update any hit.

### Task 1.4: Table values in mono

**Files:** `src/phenotypic/_gui/browse/_assets/browse.css`,
`src/phenotypic/_gui/run_console/_assets/run_console.css`,
`src/phenotypic/_gui/analysis/_assets/analysis.css`

- [x] `.browse-csv-metadata-table td` (`browse.css:300`): add
  `font-family: var(--font-mono)` and `font-size: var(--font-size-data)`.
- [x] `.run-console-recents-cell-mode` and `.run-console-recents-cell-dash`
  (`run_console.css:289`, `:294`): add `font-family: var(--font-mono)`.
- [x] `.analysis-post-preview-table td` (`analysis.css:68`): add
  `font-family: var(--font-mono)`.
- [x] Give the header cells of the three tables the Label style (mono, caption size,
  uppercase, `--tracking-wide`, muted), matching the Typography section's assignment
  of table headers.
- [x] Grep `tests/` for the three class names and run whatever tests reference them.

### Task 1.5: DESIGN.md text for Phase 1

**Files:** `DESIGN.md`

- [x] Replace the 33 Comfortaa mentions (header, Overview, Key Characteristics, 02.1,
  02.4, 02.7 and the component recipes in 05, 09, 13 and 14) with Nunito Sans,
  removing every sentence that justified IBM Plex Serif by Comfortaa's missing
  italic.
- [x] Update 02.2 (body 16 px, Body Large 18 px, new Data rung at 15 px) and 02.4
  (display styles at 600; Data Value styles on `--font-size-data`).
- [x] Remove the call-site discipline's reference to `TEXT_STYLE_*` constants (02
  intro), which do not exist in `_design.py`; the `.text-*` classes are the text
  styles.
- [x] State the role rule in 02.1: Nunito Sans for all general text and formatting,
  JetBrains Mono for every data value and table value.
- [x] Restore the 02.3 CSS block from `_design.py:573` onward, one declaration per
  line, and delete the stray editing note in the Absolute Constraints (line 131).

### Task 1.6: Ledgers, comments and screenshots

**Files:** `src/phenotypic/_gui/FEATURES.md`, `src/phenotypic/_gui/shell/_assets/shell.css`,
`tests/unit/viz/test_theme.py`, `docs/source/tutorials/gui/`

- [x] Update the "Semantic typography tokens" row in `FEATURES.md` (`:591`); the
  `features-md-gate` job requires a `FEATURES.md` change for any `_gui/` diff.
- [x] Update the role-font comment in `shell.css` (`:14`).
- [x] In `tests/unit/viz/test_theme.py`, change
  `test_chart_body_font_intentionally_differs_from_gui_chrome` to expect
  `'Nunito Sans'` for the chrome, and correct the stale `#f5f7fa` in the docstring at
  `:99` to `#FBFEF8`.
- [x] Regenerate the tutorial screenshots with
  `uv run python scripts/capture_gui_tutorial_screenshots.py` per the
  `gui-tutorial-capture` skill, and commit the full set.

### Phase 1 gate

- [x] Derive the affected surface from importers:
  `grep -rl "_gui._design\|_gui import _design\|from phenotypic._gui._design" tests/`
  plus `tests/unit/viz/test_theme.py` and `tests/unit/ci/test_startup_imports.py`.
  Run it once with `QT_QPA_PLATFORM=offscreen`.
- [x] Open the hub (`uv run phenotypic-gui --root <images>`) and look at the builder,
  results viewer, run console, browse and analysis pages for truncated or re-wrapped
  labels, and confirm that headings and UI titles, now both at 600, still read as
  two levels. If they do not, drop UI Title to 500 as the spec allows.

---

## Phase 2: Figure theme

> **Status (2026-09-29):** done. The figsize audit is in
> `docs/superpowers/reports/2026-09-29-design-md-typography-figures/figsize-audit.md`.

### Task 2.1: Published Okabe-Ito order

**Files:** `src/phenotypic/sdk_/_palette.py`

- [x] Add `OKABE_ITO_PUBLISHED`: black, orange, sky blue, bluish green, blue,
  vermilion, reddish purple, with yellow (`#F0E442`) available as a separate constant
  for large fills. Leave `OKABE_ITO` unchanged; Plotly and the GUI read it.

### Task 2.2: Theme defaults

**Files:** `src/phenotypic/sdk_/viz/figures/_mpl_theme.py`, `tests/unit/viz/`

- [x] Rewrite `phenotypic_rc()` to the Figures defaults: `font.family` sans-serif
  with `font.sans-serif` `["DejaVu Sans"]`, `mathtext.fontset` `"dejavusans"`; sizes
  7 pt for ticks, legend and annotations, 8 pt for axis and colorbar labels; axes and
  tick width 0.6, `lines.linewidth` 1.25, `lines.markersize` 3, tick length 3; white
  `figure.facecolor`, `axes.facecolor` and `savefig.facecolor`; `axes.grid` False;
  top and right spines off; `legend.frameon` False; `axes.prop_cycle` from
  `OKABE_ITO_PUBLISHED`; `pdf.fonttype` 42, `ps.fonttype` 42, `svg.fonttype`
  `"none"` and a fixed `svg.hashsalt`.
- [x] Update the module docstring, which currently cites the screen rcParams block.
- [x] Add `tests/unit/viz/test_mpl_theme.py` pinning those values, and a test that no
  font requested by the theme falls back (`font_manager.findfont` with
  `fallback_to_default=False` resolves DejaVu Sans).
- [x] `tests/unit/abc_/plotting/test_figure_backend.py:118` compares against
  `phenotypic_rc()` itself and should pass unchanged; run it.

### Task 2.3: Size presets and export helper

**Files:** `src/phenotypic/sdk_/viz/figures/_mpl_theme.py` (or a new private module
beside it), `src/phenotypic/sdk_/viz/figures/__init__.py`,
`tests/unit/ci/test_deferred_imports.py`

- [x] Add the width presets (`full` 159.2 mm, `half` 77.1 mm) and
  `figure_size_mm(width, height_mm)` returning a `figsize` tuple in inches; reject an
  unknown preset name and a height above 246.2 mm with `ValueError`.
- [x] Add `export_figure(fig, path)`, which saves inside `phenotypic_mpl_context()`,
  infers the format from the suffix, passes `metadata={"CreationDate": None}` for PDF
  and `{"Date": None}` for SVG, and uses 300 dpi for PNG.
- [x] Export both from `phenotypic.sdk_.viz.figures`, keeping matplotlib imports inside
  the functions so the startup and deferred-import guards stay green; add the new
  names to `test_deferred_imports.py` beside `phenotypic_mpl_context`.
- [x] Tests: preset arithmetic; `export_figure` writes a PDF whose font dictionary is
  `/TrueType` rather than `/Type3`; two exports of the same figure are byte-identical
  for PDF and SVG.

### Task 2.4: Existing callers

**Files:** `src/phenotypic/correction/_color_correction/_calibration_overlay.py`,
`src/phenotypic/_gui/analysis/_render.py`

- [x] `_theme_font_family()` (`_calibration_overlay.py:502`) resolves whatever the
  theme names, so it needs no change; run
  `tests/unit/correction/test_calibration_overlay.py` to confirm, since its label
  clipping guard now measures DejaVu Sans at 7 and 8 pt.
- [x] Run the GUI analysis tests that render through `_render.py:86`.

### Task 2.5: figsize audit report

- [x] Classify the 62 `figsize=` sites in `src/` as publication figures (candidates
  for a preset) or diagnostics (keep their own size), and write the result to
  `docs/superpowers/reports/2026-09-29-design-md-typography-figures/figsize-audit.md`.
  No call site changes in this plan.

### Phase 2 gate

- [x] Run the affected surface once: importers of `phenotypic.sdk_.viz.figures`,
  `_mpl_theme` and `_palette` (derive with grep), plus
  `tests/unit/ci/test_startup_imports.py` and `tests/unit/ci/test_deferred_imports.py`.
- [x] Render one pipeline figure declared `backend="mpl"` and one analyzer figure
  and look at them. No `@figure(backend="mpl")` site exists in `src/` (the earlier
  count of one was a docstring in `_pht_plot.py`), so the check used
  `render_delta_e_bars`, a themed matplotlib figure: white ground, no grid, DejaVu
  Sans at 7 / 8 pt. Its bar colors are set by the chart itself, not the cycle.

---

## Phase 3: DESIGN.md in the DESIGN.md format

> **Status (2026-09-29):** done; `npx @google/design.md@0.4.0 lint DESIGN.md` reports
> 0 errors and 0 warnings. Departures from the design (domain sections kept separate,
> components modeled for every color, the WCAG AA color fixes the lint surfaced) are
> recorded under "As built" in the format spec.

### Task 3.1: Front matter

**Files:** `DESIGN.md`

- [x] Add the YAML front matter the format spec describes: `version: alpha`, `name`,
  `description`, and the `colors`, `typography`, `rounded`, `spacing` and
  `components` groups as tabled in the format spec, with every value copied from
  `_gui/_design.py` after Phase 1.
- [x] Record each badge and alert variant as a component with its text and background
  colors, so the contrast rule checks them.

### Task 3.2: Headings and section order

- [x] Apply the section map in the format spec: drop the numbers, use the canonical
  names for recognized sections, merge 03 with 13, 05 with 14 and 06 with 11 and 12,
  split 04 into Elevation & Depth and Shapes, and place the domain sections between
  Components and Do's and Don'ts, with Logo and Branding after the Overview.
- [x] Keep the Absolute Constraints as a subsection of the Overview; move 08's Do and
  Don't lists into the final `## Do's and Don'ts`.

### Task 3.3: Figures section and scoping

- [x] Insert the Figures section verbatim from the figure-defaults spec, after Data
  Visualization.
- [x] Remove the matplotlib `rcParams` block from Code Integration.
- [x] Scope the three Absolute Constraints and the matching Do and Don't lines (mono
  numbers, navy-first order, navy-to-sky ramp) to "GUI chrome and on-screen charts".

### Task 3.4: Cross-references

- [x] Inside `DESIGN.md`, replace the 42 numbered references
  (`grep -nE "section [0-9]{2}|§ ?[0-9]{2}" DESIGN.md`) with heading names.
- [x] Outside it, replace the 41 lines in 20 files that cite a numbered section
  (`grep -rnE "DESIGN\.md" src tests` filtered for two-digit section numbers; 14 of
  them in `_gui/_design.py`) with heading names. Comments and docstrings only; no
  behavior changes.

### Task 3.5: Documentation of the linter

**Files:** `CLAUDE.md`, `src/phenotypic/_gui/CLAUDE.md`

- [x] Under "Linting & Type Checking" in `CLAUDE.md`, add
  `npx @google/design.md@0.4.0 lint DESIGN.md`, described as a manual check.
- [x] Update the Module Guides entry for `DESIGN.md` to mention figures, and the
  palette-rules paragraph in `_gui/CLAUDE.md` (`:528`) if it cites a section number.

### Phase 3 gate

- [x] `npx @google/design.md@0.4.0 lint DESIGN.md` reports no errors and no warnings.
- [x] Both cross-reference greps return nothing.
- [x] Run `tests/unit/viz/test_theme.py`, `tests/unit/gui/test_config_and_design.py`
  and `tests/unit/abc_/plotting/test_figure_backend.py`, whose docstrings cite
  `DESIGN.md`.

---

## End of implementation

- [ ] Run the full sharded regression once, on the HPCC, per the `run-phenotypic-test`
  and `slurm-job` skills. The cloud container cannot run it at the stated scale.
- [ ] Run each failing test in isolation before attributing it to this change.
