# Plotting-migration review fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all 11 findings from [`REVIEW.md`](./REVIEW.md) on branch `worktree-migrate-and-refactor-plotting`, applying the option agreed during brainstorming for each finding.

**Architecture:** Small, surgical fixes across five areas (the `FigureProvider` contract, the shared Plotly theme, the grid-fit report, the GUI availability flag, and design docs). One real runtime bug (F5, dropped annotations in the grid dashboard); the rest are correctness hardening, DESIGN.md adherence, dead-code removal, and scope-doc reconciliation. No public API changes.

**Tech Stack:** Python 3.12, pydantic v2, Plotly, Dash, pytest, `uv`.

**Conventions:**
- Run everything via `uv run` (never bare `python`/`pip`).
- After each code task: `uv run ruff check --fix` then `uv run mypy src/phenotypic`.
- Commit on this feature branch (not main). End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- The doc-only tasks have no tests; code tasks are test-first (TDD).

---

## Coverage map (finding → task)

| Finding | Score | Resolution (agreed option) | Task |
|---|---|---|---|
| F9 stale `go.Heatmap` docstring | 35 | Fix docstring → `go.Image` | T1 |
| F4 `scales` control omitted | 30 | Record in DEFERRED.md | T1 |
| F10 color report control-free vs design | 60 | Document deferral (DEFERRED.md + design §9 + cross-ref) | T1 |
| F7 axis-occupancy overlay dropped | 45 | Accept + fix "Ports" docstring + DEFERRED note | T1 (doc) / T4 (docstring) |
| F1 `inspect()` override collision | 40 | Validate overrides against `spec.controls` | T2 |
| F2 `paper_bgcolor` white vs DESIGN.md | 55 | Set `paper_bgcolor=BG` | T3 |
| F3 hardcoded `FONT_FAMILY` | 50 | Full stack literal + drift test (Option C) | T3 |
| F5 grid `dash()` drops annotations | 80 | Carry annotations in compose loop | T4 |
| F6 successive-diffs 2×/3× lines dropped | 50 | Restore lines + x-range | T4 |
| F8 dead Dash stub trips gui gate | 85 | Delete stub; notes → DEFERRED.md | T5 |
| F11 `GUI_AVAILABLE` stale after dep removal | 60 | Repoint to `dash` + FEATURES.md row | T6 |

**Gate note:** the branch's *only* `src/phenotypic/gui/` change is the F8 stub. Deleting it (T5) clears the `features-md-gate`. T6 re-touches `gui/__init__.py`, which re-arms the gate, so T6 includes the required `FEATURES.md` modification (a real, test-backed row). If you would rather keep this PR's `FEATURES.md` strictly UI-affordance-only, split T6 into its own follow-up PR instead; everything else is independent.

---

## Task 1: Reconcile design docs with shipped scope (F9, F4, F10, F7-doc)

Doc-only. No tests; verified by the doc gate scripts in T7.

**Files:**
- Modify: `src/phenotypic/correction/_color_correction/_color_correction_report.py` (module docstring, ~L17-20 and the class docstring)
- Modify: `docs/design_outlines/migrate_plotting_panel_to_plotly_dash/design.md` (§9 color-correction row, ~L294)
- Modify: `docs/design_outlines/migrate_plotting_panel_to_plotly_dash/DEFERRED.md` (append a section)

- [ ] **Step 1: Fix the stale trace-type in the color report module docstring (F9)**

In `_color_correction_report.py`, the module docstring bullet for the patches strip says `go.Heatmap` but the code builds `go.Image`. Change the line:

```
* ``patches`` — matched reference/measured/corrected swatch strip (``go.Heatmap``).
```
to:
```
* ``patches`` — matched reference/measured/corrected swatch strip (``go.Image``).
```

- [ ] **Step 2: Add a control-free cross-reference to the color report class docstring (F10)**

In the `ColorCorrectionReport` class docstring (the block that already notes "With no controls declared, `.dash()` composes the figures into a single stacked `go.Figure`"), append one sentence:

```
The interactive ROI-selector control specified in design.md §9 is deferred (see
DEFERRED.md, "Scope reductions recorded post-review"); this report currently
ships as a control-free composed figure.
```

- [ ] **Step 3: Amend design.md §9 color-correction row (F10)**

Replace the color-correction row in the §9 table:

```
| color-correction | `ColorCorrectionReport` (was `ColorCorrectionDashboard`) | ROI selector (select) | yes | Drops `param.Parameterized`; `show_*` toggles → collapsible cards (D12). |
```
with:
```
| color-correction | `ColorCorrectionReport` (was `ColorCorrectionDashboard`) | none — ROI selector deferred (see DEFERRED.md) | no — composed `go.Figure` | Drops `param.Parameterized`; `show_*` toggles → collapsible cards (D12). |
```

- [ ] **Step 4: Append the scope-reductions section to DEFERRED.md (F4, F7, F10, and the F8 restore-notes anchor)**

Append to `DEFERRED.md`:

```markdown
---

## D. Scope reductions recorded post-review

These were shipped intentionally but were not originally logged here; the
review (REVIEW.md) flagged the design/code divergence. Each is additive to
converge later, none is a regression in numeric output.

- **Diagnostics `scales` control (design.md §9).** Listed as a diagnostics
  control, but `ridge_scales` is list-valued and does not map to any
  `Control` kind (float/select/bool/text). Shipped without it; the figure
  uses a fixed scale set. Converge if a multi-select `Control` kind is added.
- **Color-correction ROI selector (design.md §9, D11).** `ColorCorrectionReport`
  ships control-free (`.dash()` → composed `go.Figure`) rather than with an
  ROI `select` Control + shell. The ipywidgets path exists
  (`FigureProvider.dash` → `build_notebook_dashboard`) so this is a pure
  add-a-Control convergence when ROI-by-ROI recompute is wanted.
- **Grid axis-occupancy image-pitch overlay (grid `_grid_fit_report.py`).**
  `fig_axis_occupancy` shows fitted per-cell counts only; the old
  `_plot_axis_occupancy` overlaid image-pitch counts when fit and image-pitch
  disagreed. The disagreement is still surfaced numerically in the summary
  `go.Table` (`fit_occupied` vs `ip_occupied`); restoring the visual overlay
  needs `ip_counts`/`agree` plumbed into the per-axis stats dict.
- **Dash web-GUI figure adapter (was `gui/_figure_dashboard.py`).** Removed as
  a deferred stub. To restore: build a controls panel + `dbc.Accordion` of
  per-figure `dcc.Graph`s from `provider.iter_figures()` / `figures(subject)`,
  one Dash callback per control-bearing figure (controls deduped by identity,
  figures seeded on load). Mirror `gui/analysis/_render.py` (renders
  `node.dash() -> go.Figure` into `dcc.Graph`). Protocol:
  `phenotypic.abc_._figure_provider`; design: design.md §7.
```

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/correction/_color_correction/_color_correction_report.py \
        docs/design_outlines/migrate_plotting_panel_to_plotly_dash/design.md \
        docs/design_outlines/migrate_plotting_panel_to_plotly_dash/DEFERRED.md
git commit -m "docs(plotting): reconcile design spec with shipped scope (F4/F9/F10)"
```

---

## Task 2: Harden `FigureProvider.inspect()` override validation (F1)

`inspect()` validates `**overrides` against the figure method's *entire* signature, so passing a non-control kwarg that happens to be a real parameter (notably the subject, e.g. `inspect(image=img)`) slips past the check and then raises a cryptic `TypeError: got multiple values for argument 'image'`. Overrides are meant to be *control* values only, so validate against `spec.controls`.

**Files:**
- Modify: `src/phenotypic/abc_/_figure_provider.py` (the `inspect()` method, ~L383-399)
- Test: `tests/unit/abc_/test_figure_provider.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/abc_/test_figure_provider.py`:

```python
def test_inspect_rejects_non_control_overrides():
    """inspect() overrides must be declared controls; a stray kwarg (incl. the
    subject param name) raises ValueError, not a cryptic TypeError."""
    import plotly.graph_objects as go
    import pytest
    from phenotypic.abc_._figure_provider import FigureProvider, figure

    class _Prov(FigureProvider):
        @figure(title="Main", primary=True)
        def fig_main(self, image=None) -> go.Figure:  # image == subject param
            return go.Figure()

    prov = _Prov()
    with pytest.raises(ValueError):
        prov.inspect(not_a_control=1)
    with pytest.raises(ValueError):
        prov.inspect(image="passing the subject by keyword is not an override")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/abc_/test_figure_provider.py::test_inspect_rejects_non_control_overrides -v`
Expected: FAIL — the second `inspect(image=...)` raises `TypeError` (not `ValueError`), so `pytest.raises(ValueError)` does not catch it.

- [ ] **Step 3: Implement the fix**

In `_figure_provider.py` `inspect()`, replace:

```python
        valid = set(inspect.signature(method).parameters)
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(
                f"inspect(): unknown override(s) {sorted(unknown)} for figure "
                f"{spec.name!r}"
            )
        kwargs: dict[str, Any] = {kw: c.default for kw, c in spec.controls.items()}
        kwargs.update(overrides)
        if "for_save" in valid:
            kwargs["for_save"] = for_save
```
with:
```python
        valid_params = set(inspect.signature(method).parameters)
        unknown = set(overrides) - set(spec.controls)
        if unknown:
            raise ValueError(
                f"inspect(): unknown override(s) {sorted(unknown)} for figure "
                f"{spec.name!r}; valid controls: {sorted(spec.controls)}"
            )
        kwargs: dict[str, Any] = {kw: c.default for kw, c in spec.controls.items()}
        kwargs.update(overrides)
        if "for_save" in valid_params:
            kwargs["for_save"] = for_save
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/abc_/test_figure_provider.py -v`
Expected: PASS (new test green, existing tests still green).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/abc_/_figure_provider.py tests/unit/abc_/test_figure_provider.py
git commit -m "fix(abc_): validate inspect() overrides against controls, not signature (F1)"
```

---

## Task 3: Theme background + font stack (F2, F3)

F2: `paper_bgcolor` is white but DESIGN.md sets the figure background to `BG` (`#f5f7fa`). F3: `FONT_FAMILY` is a stripped `"Roboto, sans-serif"` that drops the cross-platform fallback stack and silently drifts from `gui/_design.FONT_FAMILY_BODY`. `viz/` must not import `gui/` (layering / import-rules test), so use the full literal plus a drift-guard test (the test layer is exempt from the import rule).

**Files:**
- Modify: `src/phenotypic/viz/figures/_theme.py` (`FONT_FAMILY` ~L110; `paper_bgcolor` ~L136)
- Test: `tests/unit/viz/test_theme.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/viz/test_theme.py`:

```python
def test_paper_bgcolor_matches_design_bg():
    import plotly.io as pio
    from phenotypic.viz.figures._theme import BG, PHENOTYPIC_TEMPLATE_NAME

    tmpl = pio.templates[PHENOTYPIC_TEMPLATE_NAME]
    assert tmpl.layout.paper_bgcolor == BG  # #f5f7fa, not white


def test_font_family_does_not_drift_from_gui_design():
    from phenotypic.gui._design import FONT_FAMILY_BODY
    from phenotypic.viz.figures._theme import FONT_FAMILY

    assert FONT_FAMILY == FONT_FAMILY_BODY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/viz/test_theme.py -v -k "paper_bgcolor or font_family"`
Expected: FAIL — `paper_bgcolor` is `#ffffff` and `FONT_FAMILY` is `"Roboto, sans-serif"`.

- [ ] **Step 3: Implement the fixes**

In `_theme.py`, change `FONT_FAMILY` (~L110):
```python
FONT_FAMILY: str = "Roboto, sans-serif"
```
to:
```python
FONT_FAMILY: str = "'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
```
and in `register_phenotypic_template()` change (~L136):
```python
            paper_bgcolor=WHITE,
```
to:
```python
            paper_bgcolor=BG,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/viz/test_theme.py -v`
Expected: PASS. If `test_font_family_does_not_drift_from_gui_design` still fails, copy the exact value of `phenotypic.gui._design.FONT_FAMILY_BODY` into `FONT_FAMILY` (the test is the source of truth for the literal).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/viz/figures/_theme.py tests/unit/viz/test_theme.py
git commit -m "fix(viz): align theme paper bg + font stack with DESIGN.md (F2/F3)"
```

---

## Task 4: Grid report — carry annotations + restore pitch markers (F5, F6, F7-docstring)

F5 (the one real runtime bug): `GridFitReport.dash()` re-adds each panel's `layout.shapes` but not `layout.annotations`, so `add_vline(annotation_text=...)` labels and the `_empty_figure` placeholder vanish in the composed dashboard. F6: `fig_successive_diffs` dropped the 2×/3× image-pitch reference lines. F7: fix the overstated "Ports" docstring on `fig_axis_occupancy`.

**Files:**
- Modify: `src/phenotypic/grid/_grid_fit_report.py` (`dash()` ~L151-181; `fig_successive_diffs` ~L366-430; `fig_axis_occupancy` docstring ~L434-439)
- Test: `tests/unit/grid/test_grid_fit_report.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/grid/test_grid_fit_report.py`. Construct the fitted report exactly the way the existing tests in this file do (reuse the same fixture/builder that yields a `GridFitReport` or calls `finder.dashboard(image)`); then assert:

```python
def test_dash_carries_reference_line_labels():
    """Composed dashboard keeps add_vline annotation labels (F5)."""
    import plotly.graph_objects as go

    fig = _build_fitted_report().dash()   # reuse this file's existing construction
    assert isinstance(fig, go.Figure)
    labels = [a.text for a in fig.layout.annotations if a.text]
    assert any("fit (" in t for t in labels), "pitch-fit label lost in composition"


def test_successive_diffs_has_2x_3x_pitch_markers():
    """fig_successive_diffs draws 2x and 3x image-pitch reference lines (F6)."""
    report = _build_fitted_report()
    fig = report.fig_successive_diffs()
    labels = [a.text for a in fig.layout.annotations if a.text]
    assert any("2x ip" in t for t in labels)
    assert any("3x ip" in t for t in labels)
```

(If the file has no shared `_build_fitted_report()` helper, add one that mirrors the existing tests' setup — a single fitted `AutoGridFinder` over `load_synth_yeast_plate()` whose report the other tests already build.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/grid/test_grid_fit_report.py -v -k "reference_line_labels or 2x_3x"`
Expected: FAIL — no annotations survive composition (F5); no 2x/3x labels exist (F6).

- [ ] **Step 3: Fix F5 — carry annotations in the compose loop**

In `dash()`, immediately after the existing shapes loop:
```python
            for shape in sub.layout.shapes:
                composed.add_shape(shape.to_plotly_json(), row=row, col=1)
```
add:
```python
            # Carry annotations too (add_vline labels are annotations, not
            # shapes; the empty-state placeholder is a paper-anchored
            # annotation). Data-anchored annotations remap with row/col;
            # paper-anchored ones (empty-state) are re-anchored to this
            # subplot's domain so they stay centered.
            axis_suffix = "" if row == 1 else str(row)
            for ann in sub.layout.annotations:
                payload = ann.to_plotly_json()
                if payload.get("xref") == "paper" or payload.get("yref") == "paper":
                    payload["xref"] = f"x{axis_suffix} domain"
                    payload["yref"] = f"y{axis_suffix} domain"
                    payload["x"] = 0.5
                    payload["y"] = 0.5
                    composed.add_annotation(payload)
                else:
                    composed.add_annotation(payload, row=row, col=1)
```

- [ ] **Step 4: Fix F6 — restore 2×/3× pitch lines and x-range in `fig_successive_diffs`**

Before the `for stats, color in (...)` loop, initialize a pitch tracker:
```python
        fig = go.Figure()
        any_data = False
        max_pitch = 0.0
```
Inside the loop, after the existing 1× `add_vline` block (the `annotation_text=f"{label} 1x ip ..."` call), add:
```python
            max_pitch = max(max_pitch, image_pitch)
            fig.add_vline(
                x=2 * image_pitch,
                line=dict(color=_GREY, width=1.0, dash="dot"),
                annotation_text=f"{label} 2x ip ({2 * image_pitch:.0f})",
            )
            fig.add_vline(
                x=3 * image_pitch,
                line=dict(color=_GREY, width=1.0, dash="dot"),
                annotation_text=f"{label} 3x ip ({3 * image_pitch:.0f})",
            )
```
After the loop, before/with the final `fig.update_layout(...)`, clamp the x-range so the 3× marker stays visible:
```python
        if max_pitch > 0:
            fig.update_xaxes(range=[0, 3.5 * max_pitch])
```
Update the `fig_successive_diffs` docstring line that says "a green fitted-pitch reference line and a vermilion 1x image-pitch line" to also mention "plus grey 2x/3x image-pitch markers for spotting sparse-coverage peaks."

- [ ] **Step 5: Fix F7 — correct the `fig_axis_occupancy` docstring**

Change its docstring opener from:
```
Ports :meth:`AutoGridFinder._plot_axis_occupancy`: the fitted
per-index detection counts as grouped bars (row vs. col); cells
with zero detections are colored vermilion to flag gaps.
```
to:
```
Adapted from :meth:`AutoGridFinder._plot_axis_occupancy`: fitted
per-index detection counts as grouped bars (row vs. col); zero-count
cells are vermilion. The image-pitch-count overlay shown on
fit/image-pitch disagreement is intentionally dropped here (the
fit-vs-image-pitch occupancy is reported in the summary table); see
DEFERRED.md.
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/grid/test_grid_fit_report.py -v`
Expected: PASS (new tests green; existing report tests still green).

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/grid/_grid_fit_report.py tests/unit/grid/test_grid_fit_report.py
git commit -m "fix(grid): carry annotations in dashboard compose + restore pitch markers (F5/F6/F7)"
```

---

## Task 5: Remove the dead Dash stub (F8)

`gui/_figure_dashboard.py` is a deferred stub whose two functions both raise `NotImplementedError` and which nothing imports. It is the branch's only `src/phenotypic/gui/` change, so deleting it clears the `features-md-gate`. Restore-notes already captured in DEFERRED.md (T1, Step 4).

**Files:**
- Delete: `src/phenotypic/gui/_figure_dashboard.py`

- [ ] **Step 1: Confirm it is unused**

Run: `git grep -n "_figure_dashboard\|build_figure_dashboard\|register_figure_dashboard_callbacks" -- src/ tests/`
Expected: only matches inside `src/phenotypic/gui/_figure_dashboard.py` itself (no external importers). If anything else references it, stop and reassess.

- [ ] **Step 2: Delete the file**

```bash
git rm src/phenotypic/gui/_figure_dashboard.py
```

- [ ] **Step 3: Verify nothing broke**

Run: `uv run pytest tests/unit/abc_/test_figure_provider.py tests/unit/viz tests/unit/core -q`
Expected: PASS (the FigureProvider/notebook path never referenced the Dash stub).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(gui): remove deferred FigureProvider Dash stub; notes -> DEFERRED.md (F8)"
```

---

## Task 6: Repoint `GUI_AVAILABLE` to the Dash stack (F11)

The migration removes `panel`/`param`, but `gui/__init__._check_gui_deps()` still gates `GUI_AVAILABLE` on them, so the flag goes permanently False even though the (Dash) GUI is available. Repoint it to `dash`. This touches `gui/__init__.py`, which re-arms the `features-md-gate`, so add the required (honest, test-backed) FEATURES.md row.

**Files:**
- Modify: `src/phenotypic/gui/__init__.py` (`_check_gui_deps`, ~L23-30)
- Modify: `tests/unit/gui/test_optional_deps.py` (`test_gui_available_flag`)
- Modify: `src/phenotypic/gui/FEATURES.md` (add a row — gate requirement)

- [ ] **Step 1: Update the test to assert the new semantics**

In `tests/unit/gui/test_optional_deps.py`, replace `test_gui_available_flag`:

```python
    def test_gui_available_flag(self):
        """GUI_AVAILABLE reflects whether the Dash GUI stack is importable."""
        import importlib.util

        from phenotypic.gui import GUI_AVAILABLE

        assert isinstance(GUI_AVAILABLE, bool)
        assert GUI_AVAILABLE == (importlib.util.find_spec("dash") is not None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/gui/test_optional_deps.py::TestOptionalDependencies::test_gui_available_flag -v`
Expected: FAIL — `_check_gui_deps()` still checks `panel`/`param`, so `GUI_AVAILABLE` is False while `dash` is importable (False == True is False).

- [ ] **Step 3: Implement — repoint `_check_gui_deps`**

In `gui/__init__.py`, replace:
```python
def _check_gui_deps() -> bool:
    """Check if Panel GUI dependencies are available."""
    import importlib.util

    return all(
        importlib.util.find_spec(pkg) is not None for pkg in ["panel", "param"]
    )
```
with:
```python
def _check_gui_deps() -> bool:
    """Check if the Dash GUI stack is importable (Panel was removed)."""
    import importlib.util

    return importlib.util.find_spec("dash") is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/gui/test_optional_deps.py -v`
Expected: PASS.

- [ ] **Step 5: Add the FEATURES.md row (gate requirement)**

In `src/phenotypic/gui/FEATURES.md`, add a new section after the status legend block (before "## Top bar (shell chrome)"):

```markdown
## GUI availability

| Feature               | Element                       | Expected behaviour                                                                          | Status     | Test layer | Test ref                                                                                  |
|-----------------------|-------------------------------|--------------------------------------------------------------------------------------------|------------|------------|-------------------------------------------------------------------------------------------|
| GUI availability flag | `phenotypic.gui.GUI_AVAILABLE`| True when the Dash GUI stack (`dash`) is importable; no longer tied to the removed Panel deps | ✅ shipping | unit       | tests/unit/gui/test_optional_deps.py::TestOptionalDependencies::test_gui_available_flag    |
```

- [ ] **Step 6: Validate the ledger + commit**

Run: `uv run python scripts/check_features_md.py`
Expected: PASS (syntax valid; the ✅ shipping row's Test ref resolves to the updated test).

```bash
git add src/phenotypic/gui/__init__.py tests/unit/gui/test_optional_deps.py src/phenotypic/gui/FEATURES.md
git commit -m "fix(gui): GUI_AVAILABLE checks dash, not removed Panel deps (F11)"
```

---

## Task 7: Final verification

- [ ] **Step 1: Lint + type-check**

Run: `uv run ruff check --fix && uv run mypy src/phenotypic`
Expected: no errors.

- [ ] **Step 2: Run all touched unit suites**

Run:
```bash
uv run pytest tests/unit/abc_/test_figure_provider.py tests/unit/viz \
  tests/unit/grid/test_grid_fit_report.py tests/unit/gui/test_optional_deps.py -v
```
Expected: PASS.

- [ ] **Step 3: Doc gates**

Run: `uv run python scripts/check_features_md.py`
Expected: PASS.

- [ ] **Step 4: Doctests on touched modules with doctest examples**

Run: `uv run pytest --doctest-modules src/phenotypic/viz/figures/_theme.py`
Expected: PASS.

- [ ] **Step 5: Broader regression sweep**

Run: `uv run pytest tests/unit -q`
Expected: PASS (or no new failures vs the branch baseline).

- [ ] **Step 6: Manual render check for F5 (cannot be fully asserted headless)**

In a notebook or `uv run python`:
```python
from phenotypic.data import load_synth_yeast_plate
from phenotypic.grid import AutoGridFinder
img = load_synth_yeast_plate()
f = AutoGridFinder()
f.measure(img)                 # mirror however the existing grid tests fit
fig = f.dashboard(img)
print(sum(1 for a in fig.layout.annotations if a.text))  # > 0: labels present
fig.show()                     # eyeball: pitch lines labeled; empty panels show text
```

- [ ] **Step 7 (optional): annotate REVIEW.md**

Add a "Resolution" note to each finding in `REVIEW.md` pointing at the commit/task that fixed it, so the review doc reflects the closed state. Commit:
```bash
git add docs/design_outlines/migrate_plotting_panel_to_plotly_dash/REVIEW.md
git commit -m "docs(plotting): mark review findings resolved"
```

---

## Self-review (author checklist, completed)

- **Spec coverage:** all 11 findings (F1–F11) map to a task in the coverage table; none unaddressed.
- **Placeholder scan:** production-code steps carry exact code. The only deliberate "mirror existing fixture" notes are in Task 4's *test setup* (the grid-report construction helper already exists in that test file); all assertions are concrete.
- **Type/name consistency:** `valid_params`/`spec.controls` (T2), `BG`/`FONT_FAMILY`/`PHENOTYPIC_TEMPLATE_NAME` (T3), `max_pitch`/`axis_suffix` (T4), `_check_gui_deps`/`GUI_AVAILABLE` and the exact test node id `TestOptionalDependencies::test_gui_available_flag` reused in the FEATURES.md Test ref (T6) all match across tasks.

## Risks / watch-items

- **T4 empty-state annotation re-anchoring** uses Plotly subplot domain refs (`x{n} domain`); verify the empty-panel placeholder renders centered (Step 6). If a future Plotly version changes domain-ref handling, fall back to giving `_empty_figure` an invisible centered scatter trace instead of a paper annotation.
- **T6 gate coupling:** if the team prefers FEATURES.md to stay strictly UI-affordance rows, split T6 to a standalone PR so this migration PR's only gui/ delta is the F8 deletion (which needs no ledger row).
