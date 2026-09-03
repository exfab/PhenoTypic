# Scatter tab — style, sizing and page-size controls

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> or `superpowers:executing-plans` to work this task-by-task. Steps use checkbox
> (`- [ ]`) syntax.

**Goal:** Build the one row of spec §9 that shipped unimplemented — *Sizing:
section/facet/axis/tick/legend type sizes, marker size, opacity, facet height* — plus
spec §11's page-size control, and reorganize the popover so ~18 controls stay legible.

**Spec:** `docs/superpowers/specs/2026-09-01-results-scatter-tab/design.md` §9 (table row
"Sizing"), §11 ("Page size is a control, default 16x12 in").

**Branch:** `feat/results-scatter-gui`.

## Why this is not a greenfield feature

**Every value these controls would set already exists and is already consumed.** The work
is a UI surface and a wire, not a rendering change:

| `FigureSpec` field | Default | Consumed at |
|---|---|---|
| `sizes["section"]` | 14 | `_pdf.py:115` (page title) |
| `sizes["facet"]` | 9 | `_figure.py:475` (subplot annotations) |
| `sizes["axis"]` | 8 | `_figure.py:470` (`layout.font`) |
| `sizes["tick"]` | 7 | `_figure.py:476-477` (both axes) |
| `sizes["legend"]` | 8 | `_figure.py:462` |
| `marker_size` | 6 | `_figure.py:282` |
| `marker_opacity` | 0.5 | `_figure.py:283` |
| `share_axes` | `True` | `_figure.py:380-381, 454` |

`_layout.py:52-53` records the deferral as *"Nothing reads them, and chrome nothing reads
is chrome nobody can trust."* **That justification is now false** and the docstring must
be corrected in Task 1 whether or not the rest lands — the table above is the counter-
evidence. Two fields have no home yet: **facet height** (§9) and **page size** (§11);
`export_sections_pdf(..., width_in=16, height_in=12)` (`_pdf.py:44-46`) exposes the
latter as kwargs that `_callbacks.py:872` never passes.

## Decisions (locked 2026-09-02, with the user)

| # | Decision | Consequence |
|---|---|---|
| D1 | Collapsible **Data / Style / Legend / Export** sections in the **same** popover | No second tab-bar button; Style ships collapsed so the roles stay what you see first |
| D2 | Facet height drives **screen and PDF** | Replaces the `dcc.Graph` fixed `72vh`; a tall grid scrolls instead of squashing. This is the plan's one behaviour change to an existing surface |
| D3 | Page size = **presets + custom** (16×12 default, Letter landscape, A4 landscape, Custom→two inch steppers) | `width_in`/`height_in` widen from `int` to `float` for A4 |
| D4 | **No persistence** — session only | Matches every existing control on the tab. Nothing new on disk |

## Architecture

**Two stores, not one, and not eighteen Inputs.** The figure callback already takes 15
`Input`s (`_callbacks.py:918-938`) and the export callback restates 8 of them as `State`s
(`_callbacks.py:1057-1069`). Eight more steppers as individual Inputs would take those to
23 and 16. `STORE_SCATTER_LEGEND` already set the precedent for the alternative, and
`_ids.py` states the reason for it: *"One payload rather than two Inputs so the figure
callback reads legend state the same way whether it came from the corner control or the
collapse switch."*

- **`STORE_SCATTER_STYLE`** — everything that changes the *figure*: the five type sizes,
  marker size, opacity, facet height. An `Input` to the figure callback, a `State` on the
  export callback.
- **`STORE_SCATTER_PAGE`** — page width/height in inches. Export-only, so it is a `State`
  on the export callback and **not** an Input anywhere. Splitting it is what stops a
  page-size change from re-rendering the on-screen figure, which it cannot affect.

**One pattern-matching callback, not eight.** Every stepper gets an id of the shape
`{"type": SCATTER_STYLE_STEP, "field": <field>, "dir": -1 | 1}`; one `ALL`-keyed callback
reads `dash.callback_context` and writes the single store. Adding a ninth field later is
a row in a table, not a new callback.

> Two known traps here, both already documented in the repo. `gui/CLAUDE.md` warns that a
> wildcard output returning a single component must be wrapped in a 1-tuple. Separately,
> pattern-matching ids **break the browser-automation accessibility tree** (`querySelector`
> rejects the JSON id as a selector) — the existing viewer already has this, and it is why
> the E2E test in Task 8 must drive the steppers by CSS/DOM rather than by a11y role.

**`_figure_spec()` stays the one constructor** (`_callbacks.py:427-467`). It gains one
parameter, and the property its docstring claims — *"One constructor for the screen and
the export, so a role cannot be carried on one path and dropped on the other"* — extends
to sizing for free. Do not build a second spec for the export path.

## Global constraints

- **`uv` is the sole runner.** Never bare `python`/`pip`.
- **Lint with explicit paths:** `uv run ruff check --fix <paths you changed>`. A bare run
  rewrites the whole repo.
- **`_design.py` / `_config.py` own every constant.** No inline hex, font family, or
  duplicated literal. Stepper bounds go in `_config.py` beside `TILE_DIM_MIN`/`_MAX`/
  `_STEP` and `step_dim_alpha`, which is the pattern to copy.
- **A number is only valid where it was measured.** This branch's single most common
  defect (five instances — see the parent plan's table). Applies sharply to Task 6: do
  not assume the px→pt conversion survives a non-integer inch value.
- **GUI chrome is CI-gated.** `FEATURES.md`, `WORKFLOWS.md` and the tutorial screenshots
  must move together — use the **`gui-tutorial-capture`** skill. `features-md-gate`
  rejects any PR touching `gui/` without a `FEATURES.md` change.
- **Escalate** only public-interface changes (per the parent plan): `__init__.py`
  exports, `phenotypic.schema`, URL routes/query params, CLI flags, the dependency set,
  or what an existing surface renders. **D2 is such a change** — it alters how the Scatter
  figure sizes on screen — and is pre-approved by this plan; nothing else here is.

---

## Task 1 — Correct the two stale docstrings (ships alone, first)

- [ ] `_layout.py:5` says *"The eight controls in the popover"*. There are ten: section,
      facet rows, facet columns, X, Y, colour, marker shape, legend corner, collapse
      switch, show-removed switch. Fix the count, or better, stop counting in prose.
- [ ] `_layout.py:52-53` — replace the "Nothing reads them" deferral with what is
      actually true: the values are read (cite the table above), and what is missing is
      the control. If Tasks 2-7 land, delete the paragraph instead.
- [ ] `_figure.py` module docstring: check whether it claims the sizes are fixed.

**Verify:** `uv run ruff check --fix <paths>`; no test change expected.

## Task 2 — `_config.py`: bounds and the step helper

- [ ] Add `SCATTER_STYLE_FIELDS`: an ordered mapping of field → `(label, min, max, step,
      default)`. Defaults **must** be read from `FigureSpec`'s own defaults rather than
      re-spelled, or the popover and the dataclass can disagree.
- [ ] Bounds, and the honest basis for each — state it in the comment:
      - type sizes `6..24`, step 1. DESIGN.md §06 anchors the *defaults* (axis labels
        7–8 px, chart title 13 px); the range is chosen to bracket them generously.
      - marker size `2..20`, step 1 — chosen.
      - opacity `0.05..1.0`, step `0.05` — step mirrors `TILE_DIM_STEP`.
      - facet height `120..600` px, step 20 — chosen; see Task 5 for the one that is
        measurable.
- [ ] Add `step_style_value(current, field, direction)` generalizing `step_dim_alpha`:
      same clamp-and-round shape, field-keyed bounds. Round floats to 2 dp for the same
      reason `step_dim_alpha` does — repeated clicks otherwise accumulate binary drift.
- [ ] Add `SCATTER_PAGE_PRESETS`: `("16 x 12 in", 16.0, 12.0)`, `("Letter landscape",
      11.0, 8.5)`, `("A4 landscape", 11.69, 8.27)`, plus a `custom` sentinel.

**Verify:** unit tests in `tests/unit/gui/test_config_and_design.py` — every field's
default lands inside its own bounds (a table-driven loop, so a future field cannot be
added outside its range); stepping past a bound clamps; stepping opacity ten times from
the default lands on an exact 2 dp value.

## Task 3 — `FigureSpec` gains `facet_height`; `_ids.py` gains the ids

- [ ] `_spec.py`: add `facet_height: int = 220` with a docstring line matching the
      existing `Args:` style. Leave `sizes`/`marker_*` alone.
- [ ] `_ids.py`: add `SCATTER_STYLE_STEP` (the pattern-matching `type`),
      `SCATTER_STYLE_READOUT`, `SCATTER_PAGE_PRESET`, `SCATTER_PAGE_WIDTH`,
      `SCATTER_PAGE_HEIGHT`, `STORE_SCATTER_STYLE`, `STORE_SCATTER_PAGE`. Follow the
      file's convention: every id `scatter-`/`store-scatter-` prefixed, each with a `#:`
      comment saying what reads it, and each added to `__all__`.

**Verify:** `test_scatter_spec.py` — a `FigureSpec` built with no arguments still carries
every default the figure builder reads.

## Task 4 — The stepper widget and the restructured popover (D1)

- [ ] Build `_build_style_stepper(field)` in `_layout.py`, modelled on
      `colony_view/_layout.py:115` `_build_dim_stepper` — `[ − ] label 8 [ + ]`, mono
      readout seeded from the default so it reads correctly before the first store echo.
      Ids are the pattern-matching dicts from Task 3.
- [ ] Restructure the popover body into a `dbc.Accordion` with four items: **Data**
      (the seven role dropdowns, open), **Style** (eight steppers, closed), **Legend**
      (corner + collapse, closed), **Export** (page size, closed). Move `show removed`
      into Data — it selects rows, not styling.
- [ ] Keep the `maxHeight: 70vh, overflowY: auto` on the body.

**Verify:** `test_scatter_layout.py` — the accordion mounts every id exactly once; each
stepper's readout is seeded from the matching `FigureSpec` default; no id is duplicated
(a Dash duplicate-id error names the surface, which is why they are all `scatter-`
prefixed). A build-time assertion, not a rendering one.

## Task 5 — Wire the style store into the figure (D2 — the risky seam)

This is the one task that changes an existing surface's behaviour. Do it alone, and
review it alone.

- [ ] One `ALL`-keyed callback over `SCATTER_STYLE_STEP` → `STORE_SCATTER_STYLE`,
      plus a second echoing the store into the readouts (mirroring the shared tile-dim
      readout callback).
- [ ] `_figure_spec()` takes the style payload and threads it into `FigureSpec`. An
      absent or malformed payload must fall back to the dataclass defaults, not raise —
      the store is empty on first render.
- [ ] Figure callback gains `Input(STORE_SCATTER_STYLE, "data")` and a second `Output`:
      `SCATTER_GRAPH.style`, set to `{"height": f"{facet_height * n_rows}px"}`.
      `n_rows` comes from the already-computed facet plan; do not recompute it.
- [ ] Remove the fixed `"height": "72vh"` at `_layout.py:611` and confirm the tab body
      scrolls rather than clipping. **Check the one-facet-row case** — at 220 px a
      single-row figure is far shorter than today's 72vh, which is a visible regression
      if the floor is wrong. Decide the floor by looking at it, and record what you saw.
- [ ] `_figure.py` must **not** set `height` for the screen path (the Graph div owns it)
      and must keep setting it for export (`_pdf.py` owns the page).

**Verify:** `test_scatter_figure.py` — a spec with non-default sizes produces a figure
whose `layout.font.size`, `legend.font.size`, tick fonts and annotation fonts all match,
and whose marker size/opacity match. Prove each assertion can fail by reverting one
field at a time; a single "the sizes are applied" test that passes when four of five are
dropped is the failure mode to avoid here.

## Task 6 — Page size (D3), and the px→pt claim

- [ ] Preset dropdown + two inch steppers revealed only for Custom; both write
      `STORE_SCATTER_PAGE`.
- [ ] Export callback passes `width_in`/`height_in` from that store.
- [ ] Widen `_pdf.export_sections_pdf`'s `width_in: int` / `height_in: int` to `float`.
      Internal (`_`-prefixed module) — no escalation.
- [ ] **Measure, do not assume.** `_pdf.py:34` pins `_PIXELS_PER_INCH = 96` with a
      comment recording that both earlier values were wrong *"from the same habit of
      carrying a number out of the context that measured it"*, and that only a rendered
      MediaBox settled it. A4 landscape is 11.69 × 8.27 in → 1122.24 × 793.92 px, which
      is the first non-integer this path has seen. Render it and read the MediaBox back
      before claiming it round-trips; if it does not, the fix is in this task, not a
      follow-up.

**Verify:** extend `test_scatter_pdf.py`'s existing
`test_the_rendered_page_measures_the_requested_inches` to the A4 preset. Assert the
MediaBox in points with a tolerance derived from the conversion (a half-point rounding at
72 pt/in), **not** a guessed epsilon.

## Task 7 — Ledgers and screenshots (CI-gated)

- [ ] Invoke the **`gui-tutorial-capture`** skill and follow it. `FEATURES.md` gains a
      row per new affordance; `WORKFLOWS.md` and
      `scripts/capture_gui_tutorial_screenshots.py` change together if the Scatter
      walkthrough gains a step.
- [ ] Re-run the capture script and commit refreshed PNGs. On this machine that needs the
      Slurm batch script at
      `docs/superpowers/plans/2026-09-01-results-scatter-tab/run_gui_capture.sbatch`.

## Task 8 — The E2E test that does not exist

Found during the 2026-09-02 browser session: the tab has **nine unit test modules and no
Playwright E2E**, so the click → crop → inspector chain against a real store had no
automated coverage. Adding controls is the right moment to close that.

- [ ] `tests/e2e/gui/test_scatter_tab.py`, following `test_qc_tab.py` /
      `test_heatmap_tab.py` for fixture shape.
- [ ] Cover, at minimum: the tab renders; a stepper click changes the rendered
      `layout.font.size`; the pager advances; a point click opens the inspector with a
      resolvable title. Read Plotly state via `page.evaluate` on `_fullLayout` — the
      traces are `Scattergl` and have **no DOM points to query**.
- [ ] Consult the **`ci-flaky-quarantine`** skill before marking anything `ci_flaky`.

---

## What this plan deliberately does not do

- **`share_axes`** has a `FigureSpec` field and no control, exactly like the sizes did.
  It is *not* in §9's control table, so adding one is a spec change, not an
  implementation gap. Raise it separately.
- **Persistence** (D4). If it is wanted later, `STORE_SCATTER_STYLE` is already the one
  place a `storage_type` would go.
- **The default-Y observation.** The tab opens on the first numeric measurement, which on
  the verification run is `GridSpatial_LeftNeighborObjLabel` — null for 375 of 723
  plottable rows, so the tab opens reporting *"386 rows excluded"* and 16 of 22 strains.
  `_default_y_col` behaves as written; whether "first numeric measurement" is the right
  rule is a spec question, and changing it changes what an existing surface offers.

## Verification fixture

`/Volumes/T9/exfab/UCR_029_E_D-Maresca/2026-08-11-migration-test` — the spec's own
fixture. Numbers re-derived from `deliverables/measurements.parquet` on 2026-09-02 and
matching the spec: 844 rows, 723 plottable, 81 excluded for a null
`Metadata_ImageDatetime`, 22 strain sections, first section `A3*`. A section with six
facets (`C14#*`, `Grid_ColNum` 0/2/4/6/8/10) is the useful one for Task 5 — it is where
the 72vh squash is worst and where facet height has to earn itself.
