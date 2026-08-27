# Phase 5 — Heatmap / Error / QC: unmount

**Spec:** §3, §4, §5, §6. **Depends on:** phase 1 (shares `_layout.py` and `_callbacks.py`).
**Blocks:** phase 6.

**Deliverable:** the results viewer's `dbc.Tabs` holds **two** tabs — Plate and Colony.
`_heatmap_tab/`, `_error_tab/` and `_qc_tab/` stay on disk with their unit tests passing;
their e2e tests are skip-marked. **`colony_view/` is not touched**, so the curation radial
survives.

> **`_error_tab/` is a CLI dependency, not only a GUI tab.**
> `_cli/_cli_error_outputs.py:84` imports `capture_error_source_fingerprints`,
> `compute_all_category_analysis` and `publish_error_analysis` from
> `results_viewer/_error_tab/_publication.py`, **on every finalize**. The spec says
> `_error_tab/` "stays on disk" but never says it *must*. An executor reading "unmount
> Error" as licence to delete the package breaks CLI finalization, and **no GUI test catches
> it** — the failure surfaces in a CLI run, in a different subsystem from the one being
> edited. The same caution applies less sharply to `_qc_tab/` and `_heatmap_tab/`; grep
> before removing anything:
>
> ```bash
> uv run grep -rn "_error_tab\|_qc_tab\|_heatmap_tab" src/phenotypic/_cli/ src/phenotypic/analysis/
> ```

> **The correction this phase encodes (spec §5).** An earlier draft had the viewer go
> read-only on the premise that unmounting QC and Error takes the curation radial with
> them. That premise is false: the radial is mounted on **Colony** as well —
> `colony_view/_grid.py:47, :462` builds `build_radial_trigger` on every tile and
> `colony_view/_callbacks.py:43` builds the popover body. Colony survives, so the radial
> survives unless deliberately torn out. Read-only was chosen as the *cheaper* option and
> is in fact the more expensive one. **Consequence:** `_shared/_radial.py` and
> `_shared/_triage_callbacks.py` are **not** unmounted — they drop from two consumer
> surfaces to one.

---

### Task 5.1: Unmount the three tabs

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_layout.py:65, :66, :72` (imports), `:609, :610, :615` (bodies), `:630-647` (three `dbc.Tab` entries)
- Modify: `src/phenotypic/gui/results_viewer/_callbacks.py:70, :76-78, :79` (imports), `:113, :114, :115` (register calls)
- Modify: `tests/unit/gui/results_viewer/test_layout_tab_shape.py`

**Interfaces:**
- Consumes: `test_results_tabs_expose_exactly_the_mounted_surfaces` from phase 1 task 1.1.
- Produces: the final two-tab shape phase 6 asserts against a live app.

- [ ] **Step 1: Drive the change from the tab-shape test**

Edit the expected list in `test_layout_tab_shape.py` down to its final value:

```python
    assert _tab_ids(built_results_layout) == [
        ids.TAB_PLATE_ID,
        ids.TAB_COLONY_ID,
    ]
```

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_layout_tab_shape.py -v
```
Expected: FAIL, showing five actual tabs against two expected.

- [ ] **Step 3: Drop the three layout imports**

Remove from `_layout.py`:
```python
from phenotypic.gui.results_viewer._error_tab import build_error_tab_body      # :65
from phenotypic.gui.results_viewer._heatmap_tab import build_heatmap_tab_body  # :66
from phenotypic.gui.results_viewer._qc_tab import build_qc_tab_body            # :72
```

- [ ] **Step 4: Drop the three body constructions**

Remove `heatmap_tab_body = ...` (`:609`), the `error_tab_body = build_error_tab_body(...)`
call (`:610-614`), and the `qc_tab_body = build_qc_tab_body(...)` call (`:615-618`).

Then check whether `_resolve_measurement_schema` and `_resolve_qc_recipe` still have
callers:

```bash
uv run grep -n "_resolve_measurement_schema\|_resolve_qc_recipe" \
  src/phenotypic/gui/results_viewer/_layout.py
```
If a helper's only caller was a removed body, **leave the helper defined** — it belongs to
the retained packages' surface area and removing it is deletion, not unmounting. Remove
only the now-dead local call, and delete the `schema = _resolve_measurement_schema(...)`
assignment plus its explanatory comment at `:605-608` if nothing else uses `schema`.

- [ ] **Step 5: Drop the three `dbc.Tab` entries**

Remove the QC, Heatmap and Error entries so `dbc.Tabs` reads:

```python
    tabs = dbc.Tabs(
        [
            dbc.Tab(
                cards_column,
                label="Plate",
                tab_id=ids.TAB_PLATE_ID,
            ),
            dbc.Tab(
                colony_tab_body,
                label="Colony",
                tab_id=ids.TAB_COLONY_ID,
            ),
        ],
        id=ids.TABS_ID,
        active_tab=ids.TAB_PLATE_ID,
    )
```

`active_tab=ids.TAB_PLATE_ID` needs no change — Plate is still first (spec §3).

- [ ] **Step 6: Drop the three imports and the three register calls**

In `_callbacks.py` remove the imports at `:70`, `:76-78`, `:79`, and the three dispatch
lines at `:113-115`. The post-change dispatch list — verify yours matches exactly:

```python
    _layout.register_callbacks(app, output_root)
    _filter_panel.register_callbacks(app, output_root, filtered_state)
    _filter_offcanvas.register_filter_offcanvas_callbacks(app)
    _viewer_card.register_callbacks(app, output_root)
    _colony_callbacks.register_callbacks(app, output_root, filtered_state)
    _register_plot_refresh_callback(app, output_root, filtered_state)
    _register_clientside_callbacks(app)
```

> Spec §4 lists this without `_register_plot_refresh_callback`. That line exists at
> `_callbacks.py:117` today, is unrelated to the three unmounted tabs, and **stays**. The
> spec's list is an illustration of the three removals, not a complete replacement body.
> Recorded here so an executor does not delete a live registration to match the spec.

**Removing `register_qc_callbacks` matters beyond dead code:** it opens
`deliverables/qc/qc.duckdb` state at registration time, so leaving it wired has a side
effect, not merely an inert binding.

- [ ] **Step 7: Run the tab-shape test, then the suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_layout_tab_shape.py -v
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/results_viewer -n 4 -q
```
Expected: PASS, PASS. The retained `_heatmap_tab/`, `_error_tab/`, `_qc_tab/` unit tests
must still pass — they test the packages, not the mount.

- [ ] **Step 8: Prove the curation radial is untouched**

```bash
git diff --stat src/phenotypic/gui/results_viewer/colony_view/ \
                 src/phenotypic/gui/results_viewer/_shared/_radial.py \
                 src/phenotypic/gui/results_viewer/_shared/_triage_callbacks.py \
                 tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py
# The curation rows live in the LEDGER too, and only the code half was watched.
uv run grep -c "Colony radial lazy-populate\|colony-bulk-mark-dropdown" \
  src/phenotypic/gui/FEATURES.md    # must be 2, unchanged from baseline
```
Expected: **empty** (adjust the `_shared` paths if they resolve elsewhere —
`uv run grep -rl "build_radial_trigger" src/`). A non-empty diff on the colony test file
is a spec §5 violation: stop and escalate.

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py -q
```
Expected: PASS, unmodified.

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/_layout.py \
                        src/phenotypic/gui/results_viewer/_callbacks.py \
                        tests/unit/gui/results_viewer/test_layout_tab_shape.py
git add -A src/phenotypic/gui/results_viewer tests/unit/gui/results_viewer
git commit -m "refactor(gui): unmount the QC, Heatmap and Error tabs"
```

---

### Task 5.2: Skip-mark the three e2e suites

**Files:**
- Modify: e2e modules driving the QC, Heatmap and Error tabs

- [ ] **Step 1: Find them**

```bash
uv run grep -rln "heatmap\|qc-tab\|TAB_QC\|error-tab\|TAB_ERROR\|TAB_HEATMAP" tests/e2e/gui/
```
Cross-check each hit actually drives the tab rather than merely mentioning it; a Colony
test may reference an error **category** without touching the Error tab.

- [ ] **Step 2: Add the module-level skip**

Same shape as phase 4 task 4.2 step 2, with the reason naming spec §3:

```python
import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "QC/Heatmap/Error are unmounted by "
        "docs/superpowers/specs/2026-08-26-gui-simplification-removals "
        "(spec section 3). These tests are the acceptance suite for the "
        "overhauled tabs; delete this marker when the surface returns."
    )
)
```

- [ ] **Step 3: Confirm skip, not fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -q
```
Expected: the marked modules skip; **every Colony and Browse-Single e2e test still passes**.
Curation e2e coverage that runs through Colony must remain green — that is spec §5 checked
end to end.

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/gui
git commit -m "test(gui): skip-mark the QC, Heatmap and Error e2e suites"
```

---

### Task 5.3: Mark the three tabs unmounted in the ledgers; retire their tutorials

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md` — sections `## QC tab` (`:594`), `## QC Review sub-view` (`:624`), `## Heatmap tab` (`:665`), `## Error analysis tab` (`:684`). **Anchor on the headings**; the spec's `587/617/658/677` are each ~7 lines early and land *inside the preceding section*. **Edit**, not remove.
- Modify: `src/phenotypic/gui/WORKFLOWS.md:46, :47, :51, :52` — remove
- Modify: `scripts/capture_gui_tutorial_screenshots.py:1750, :1810, :1900, :1947` + call sites
- Delete: `docs/source/tutorials/gui/10_qc_curation_loop.md`, `11_heatmap_exploration.md`, `15_qc_review.md`, `17_error_analysis.md` + their image directories
- Modify: `docs/source/tutorials/gui/index.md`

- [ ] **Step 1: Edit the four FEATURES.md rows to unmounted**

Follow phase 4 task 4.3 step 1's pattern for each of the four **headings** above, pointing
at spec §3, and use the same status: **`⏸ unmounted`**.

That value was settled during plan refinement (ledger ORCH-5) and needs no change to
`scripts/check_features_md.py` — the row loop skips any status that is neither
`✅ shipping` nor `🚧 in progress`, so `⏸ unmounted` stops ref resolution and passes
`--strict`. Phase 4 task 4.3 step 1 carries the full reasoning and the table of why none of
the three existing values fits.

**If phase 4 has not run yet** (phases landing as separate PRs), this phase introduces the
value; nothing about it depends on phase 4 having gone first.

- [ ] **Step 2: Remove the four WORKFLOWS.md rows and four capture functions**

Rows `:46` `qc_curation_loop`, `:47` `heatmap_exploration`, `:51` `qc_review`,
`:52` `error_analysis`. Then remove `_capture_qc_curation_loop` (`:1750`),
`_capture_qc_review` (`:1810`), `_capture_heatmap_exploration` (`:1900`),
`_capture_error_analysis` (`:1947`) and every dispatch site:

```bash
uv run grep -n "_capture_qc_curation_loop\|_capture_qc_review\|_capture_heatmap_exploration\|_capture_error_analysis" \
  scripts/capture_gui_tutorial_screenshots.py
```

- [ ] **Step 3: Delete the four tutorial pages and images; update the toctree**

```bash
git rm docs/source/tutorials/gui/10_qc_curation_loop.md \
       docs/source/tutorials/gui/11_heatmap_exploration.md \
       docs/source/tutorials/gui/15_qc_review.md \
       docs/source/tutorials/gui/17_error_analysis.md
git rm -r docs/source/_static/gui_images/qc_curation_loop \
          docs/source/_static/gui_images/heatmap_exploration \
          docs/source/_static/gui_images/qc_review \
          docs/source/_static/gui_images/error_analysis
```
Then remove all four from `docs/source/tutorials/gui/index.md`.

**Do not renumber the surviving pages.** Renumbering rewrites every cross-reference and
every image path for no benefit; gaps in the sequence are fine and are the cheaper state.

- [ ] **Step 4: Run the three gates**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --skip-cli
```
Expected: all exit 0.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
           scripts/capture_gui_tutorial_screenshots.py docs/source
git commit -m "docs(gui): mark QC/Heatmap/Error unmounted and retire their tutorials"
```
