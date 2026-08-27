# Phase 1 — Results Timeline tab: delete

**Spec:** §1.1, §1.2, §1.3, §6 of
[`2026-08-26-gui-simplification-removals`](../../specs/2026-08-26-gui-simplification-removals/design.md)

**Deliverable:** the results viewer builds a `dbc.Tabs` with **five** tabs (Plate, Colony,
QC, Heatmap, Error). `results_viewer/timeline_view/` and its assets are gone. Nothing under
`src/phenotypic/gui/results_viewer/` imports `timeline_view`. `_shared/timeline/` is still
present — Browse still needs it until phase 2.

---

### Task 1.1: Prove the tab count before touching anything

**Files:**
- Test: `tests/unit/gui/results_viewer/test_layout_tab_shape.py` (create)

**Interfaces:**
- Consumes: `phenotypic.gui.results_viewer._layout.build_layout`, `..._ids`
- Produces: `test_results_tabs_expose_exactly_the_mounted_surfaces` — phase 5 edits this
  same test to drop to two tabs. It is the executable record of tab shape across the plan.

- [ ] **Step 1: Read how an existing layout test builds its `output_root`**

The layout builder needs a real `OutputRoot`. Reuse the fixture pattern already used by
the results-viewer unit tests rather than inventing one:

```bash
uv run grep -rn "OutputRoot" tests/unit/gui/results_viewer/ | head -20
```

Note the fixture name and module it comes from; use it verbatim in step 2.

- [ ] **Step 2: Write the test at its *current* expected value**

Write it green-on-arrival so it pins today's shape. It will be edited — deliberately —
in step 6 of task 1.4 and again in phase 5.

```python
"""Pin the set of tabs the results viewer actually mounts.

This test is edited deliberately as surfaces are removed. Each edit is the
executable statement that a tab came off; a surprise failure here means a tab
moved without a spec change behind it.
"""

from phenotypic.gui.results_viewer import _ids as ids


def _tab_ids(layout) -> list[str]:
    """Collect ``tab_id`` from the single ``dbc.Tabs`` in a built layout."""
    found: list[str] = []

    def walk(node) -> None:
        children = getattr(node, "children", None)
        if getattr(node, "_type", type(node).__name__) == "Tabs" or type(
            node
        ).__name__ == "Tabs":
            for tab in children or []:
                found.append(tab.tab_id)
            return
        if isinstance(children, (list, tuple)):
            for child in children:
                walk(child)
        elif children is not None:
            walk(children)

    walk(layout)
    return found


def test_results_tabs_expose_exactly_the_mounted_surfaces(built_results_layout):
    assert _tab_ids(built_results_layout) == [
        ids.TAB_PLATE_ID,
        ids.TAB_COLONY_ID,
        ids.TAB_QC_ID,
        ids.TAB_HEATMAP_ID,
        ids.TAB_ERROR_ID,
        ids.TAB_TIMELINE_ID,
    ]
```

Add a `built_results_layout` fixture to the nearest `conftest.py` that calls
`_layout.build_layout(...)` with the same arguments the existing results-viewer tests use.

- [ ] **Step 3: Run it and confirm it passes**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_layout_tab_shape.py -v
```
Expected: PASS, 1 test. If it fails, the fixture is wrong — fix the fixture, not the
expected list.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/gui/results_viewer/test_layout_tab_shape.py \
        tests/unit/gui/results_viewer/conftest.py
git commit -m "test(gui): pin the results viewer's mounted tab set"
```

---

### Task 1.2: Delete the timeline_view package and its assets

**Files:**
- Delete: `src/phenotypic/gui/results_viewer/timeline_view/` (6 modules)
- Delete: `src/phenotypic/gui/results_viewer/_assets/timeline.js`
- Delete: `src/phenotypic/gui/results_viewer/_assets/timeline.css`
- Delete: `tests/gui/results_viewer/timeline_view/` (7 files)
- Delete: `tests/e2e/gui/test_results_timeline.py`
- Delete: `tests/integration/gui/test_timeline_thumb_url.py`

**Interfaces:**
- Consumes: nothing.
- Produces: absence. Task 1.3 repairs the importers this breaks.

- [ ] **Step 1: Record what currently imports the package**

```bash
uv run grep -rn "timeline_view" src/ tests/ scripts/ docs/ --include='*.py' --include='*.md'
```
Paste the output into the commit body. It is the checklist for task 1.3 and phase 6's
dangling-reference test, and it is cheaper to capture now than to reconstruct after
deletion.

- [ ] **Step 2: Confirm `_assets/timeline.js` is results-viewer-local**

`browse/_assets/timeline.js` is a **separate vendored copy** guarded by a CI
byte-equality check between the two. Deleting only the results-viewer copy here is
correct and intentional; phase 2 deletes the browse copy, and phase 3 removes the guard.

```bash
uv run grep -rn "timeline.js" .github/ scripts/ tests/ | head
```
Expected: a byte-equality guard naming both copies. Note where it lives; phase 3 removes it.

- [ ] **Step 3: Delete**

```bash
git rm -r src/phenotypic/gui/results_viewer/timeline_view
git rm src/phenotypic/gui/results_viewer/_assets/timeline.js
git rm src/phenotypic/gui/results_viewer/_assets/timeline.css
git rm -r tests/gui/results_viewer/timeline_view
git rm tests/e2e/gui/test_results_timeline.py
git rm tests/integration/gui/test_timeline_thumb_url.py
```

- [ ] **Step 4: Confirm the import is now broken — this is the expected state**

```bash
QT_QPA_PLATFORM=offscreen uv run python -c \
  "from phenotypic.gui.results_viewer import _layout"
```
Expected: `ModuleNotFoundError: No module named
'phenotypic.gui.results_viewer.timeline_view'`. If it *succeeds*, an import you thought
existed does not, and step 1's list is wrong — re-derive it before continuing.

Do **not** commit here; the tree does not import. Task 1.3 makes it whole.

---

### Task 1.3: Repair the three importers

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_layout.py:74` (import), `:619` (body), `:648-653` (tab)
- Modify: `src/phenotypic/gui/results_viewer/_callbacks.py:83-85` (import), `:116` (register)
- Modify: `src/phenotypic/gui/results_viewer/_app.py:82` (thumb-route registration)
- Modify: `src/phenotypic/gui/results_viewer/_ids.py:521` (`TAB_TIMELINE_ID`)
- Modify: `tests/unit/gui/results_viewer/test_layout_tab_shape.py`

**Interfaces:**
- Consumes: the importer list from task 1.2 step 1.
- Produces: a results viewer that imports clean with five tabs.

- [ ] **Step 1: Drop the layout import**

Remove the whole `from phenotypic.gui.results_viewer.timeline_view import (...)` statement
beginning at `_layout.py:74`.

- [ ] **Step 2: Drop the body construction and the tab**

Remove the `timeline_tab_body = _timeline_layout.layout(output_root)` line (`:619`), and
remove this entry from the `dbc.Tabs` list:

```python
            dbc.Tab(
                timeline_tab_body,
                label="Timeline",
                tab_id=ids.TAB_TIMELINE_ID,
            ),
```

Leave `active_tab=ids.TAB_PLATE_ID` alone — Plate is still first.

- [ ] **Step 3: Drop the callback import and its register call**

In `_callbacks.py`, remove the import block at `:83-85`:

```python
from phenotypic.gui.results_viewer.timeline_view import (
    _callbacks as _timeline_callbacks,
)
```

and the dispatch line at `:116`:

```python
    _timeline_callbacks.register_callbacks(app, output_root)
```

**This is not optional.** `suppress_callback_exceptions=True` is set at
`results_viewer/_app.py:144`, so a register call against an absent layout would never
error and never fire — a silent dead registration, which is precisely the residue this
spec exists to avoid (spec §4).

- [ ] **Step 4: Drop the thumb-route registration in `_app.py`**

Remove the `timeline_view` import at `:82` and the registration call that uses it. Grep
first so you remove the call, not just the import:

```bash
uv run grep -n "timeline" src/phenotypic/gui/results_viewer/_app.py
```

- [ ] **Step 5: Drop `TAB_TIMELINE_ID`**

Remove the constant at `_ids.py:521` **and** its entry in that module's `__all__`.

```bash
uv run grep -rn "TAB_TIMELINE_ID" src/ tests/
```
Expected after the edit: only `tests/unit/gui/results_viewer/test_layout_tab_shape.py`,
which step 6 fixes.

- [ ] **Step 6: Update the tab-shape test to five tabs**

Remove `ids.TAB_TIMELINE_ID` from the expected list in
`test_results_tabs_expose_exactly_the_mounted_surfaces`.

- [ ] **Step 7: Run the results-viewer unit suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/results_viewer -n 4 -q
```
Expected: PASS. A failure naming `timeline` means an importer was missed — return to
task 1.2 step 1's list.

- [ ] **Step 8: Lint the files you changed**

```bash
uv run ruff check --fix \
  src/phenotypic/gui/results_viewer/_layout.py \
  src/phenotypic/gui/results_viewer/_callbacks.py \
  src/phenotypic/gui/results_viewer/_app.py \
  src/phenotypic/gui/results_viewer/_ids.py \
  tests/unit/gui/results_viewer/test_layout_tab_shape.py
```

- [ ] **Step 9: Commit deletion and repair together**

They are one reviewable unit — the tree does not import between them.

```bash
git add -A src/phenotypic/gui/results_viewer tests/unit/gui/results_viewer tests/gui tests/e2e tests/integration
git commit -m "refactor(gui): delete the Results Timeline tab"
```

---

### Task 1.4: Retire the Results Timeline ledger rows, tutorial and capture function

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md:372-394` (Results Timeline rows — remove)
- Modify: `src/phenotypic/gui/WORKFLOWS.md:55` (`results_timeline` row — remove)
- Modify: `scripts/capture_gui_tutorial_screenshots.py:1246` (`_capture_results_timeline`) and its call sites
- Delete: `docs/source/tutorials/gui/20_results_timeline.md` and `docs/source/_static/gui_images/results_timeline/`
- Modify: `docs/source/tutorials/gui/index.md`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: green `check_features_md.py --strict` and `check_workflows_md.py -v`.

- [ ] **Step 1: Remove the FEATURES.md rows**

**Anchor on the heading, never the line number.** Delete the rows under
`### Results Timeline tab` — the heading is at `:379`, and the section runs to `:400`.

> **The spec's `372-394` is wrong and dangerous.** `:372-377` are **Colony curation rows**
> — `Colony radial lazy-populate`, `Custom folder + ＋ Add custom`,
> `Bulk "Mark N as ▾" (colony)`, `Pixel layer toggle`. Deleting that range removes four
> curation rows, violating spec §5, and leaves `:395-400` as orphaned Timeline rows whose
> Test refs point at files this phase deletes — which fails `features-md-gate`. Verify
> before and after:
>
> ```bash
> uv run grep -n "^### Results Timeline tab" src/phenotypic/gui/FEATURES.md
> uv run grep -c "Colony radial lazy-populate" src/phenotypic/gui/FEATURES.md   # must stay 1
> ```

These are **deleted** surfaces, so the rows go entirely — unlike phases 4 and 5, where rows
are edited to say *unmounted*.

Leave the two shared-engine rows **alone** (they are at `:543-544`, not the spec's
`:536-537` — `:536-537` are run-console rows). The engine still has a live consumer — Browse — until phase 2 task 2.5
deletes it, and `check_features_md.py` resolves refs for `✅ shipping` rows. Phase 2 task 2.6
retires them.

- [ ] **Step 2: Remove the WORKFLOWS.md row**

Delete line `:55`, the `results_timeline` row.

- [ ] **Step 3: Remove the capture function and every call site**

```bash
uv run grep -n "_capture_results_timeline" scripts/capture_gui_tutorial_screenshots.py
```
Remove the `def _capture_results_timeline(...)` body at `:1246` and each call site the
grep reports (the spec records call sites near `:671, :680-683, :2417, :2454`, plus a
results-timeline harness block at `:2381-2464` — grep is authoritative, the line numbers
are a hint).

`check_workflows_md.py` requires each `_capture_<id>` to be both **defined** and
**dispatched** from `capture_workflow_screenshots` or
`capture_standalone_viewer_screenshots`. Removing the row without removing the dispatch,
or vice versa, fails the gate.

- [ ] **Step 4: Delete the tutorial page and its images**

```bash
git rm docs/source/tutorials/gui/20_results_timeline.md
git rm -r docs/source/_static/gui_images/results_timeline
```
Then remove `20_results_timeline` from the toctree in
`docs/source/tutorials/gui/index.md`.

- [ ] **Step 5: Run both ledger gates**

```bash
uv run python scripts/check_features_md.py
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
```
Expected: all three exit 0. A `check_workflows_md.py` failure naming a **missing image
directory** means step 4 removed images for a row that still exists — recheck step 2.

- [ ] **Step 6: Smoke the capture script**

```bash
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --skip-cli
```
Expected: exit 0. `--skip-cli` skips the synthetic-dataset build and CLI run, which
`gui-checks.yml:253` does NOT — it runs the script bare. Bare is a full dataset build plus a
Playwright capture of every tutorial page; running that once per phase is minutes each time.
Run it bare **once, at the end**, and `--skip-cli` in between. The script accepts only
`--force`, `--headed` and `--skip-cli` (`:2913-2926`); `--smoke` does not exist and argparse
exits 2.

- [ ] **Step 7: Commit**

```bash
git add -A src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
           scripts/capture_gui_tutorial_screenshots.py docs/source
git commit -m "docs(gui): retire the Results Timeline ledger rows and tutorial"
```
