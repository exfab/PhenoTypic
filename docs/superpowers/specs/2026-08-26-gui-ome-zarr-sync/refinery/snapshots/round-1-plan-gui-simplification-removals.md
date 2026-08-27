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
# Phase 2 — Browse Timeline mode: delete

**Spec:** §1.1, §1.2, §1.3, §6. **Depends on:** nothing (phase 1 touched a different
sub-app). **Blocks:** phase 3.

**Deliverable:** the Browse tab has no view-mode toggle and no timeline body, **and the
shared timeline engine is gone** — this phase kills the second and last of its two
consumers, so it takes the engine with it (tasks 2.5-2.6).

The Browse tab has no view-mode toggle and no timeline body. Single mode
is the whole tab, with **behaviour unchanged**. The four timeline-only browse helpers and
the browse copy of `timeline.js` / `timeline.css` are gone. `_shared/timeline/` still
exists — phase 3 removes it.

> **The asymmetry to hold on to:** phase 1 removed a tab from a tab bar. This phase removes
> a *mode* from a two-mode tab, which means the surviving mode stops being conditional.
> `BROWSE_SINGLE_BODY` currently renders with a `display` style driven by the toggle
> callback (`browse/_callbacks.py:1353-1360`); after this phase it renders unconditionally.
> Getting that wrong leaves Single permanently hidden and every Browse e2e test failing at
> once.

---

### Task 2.1: Pin Single mode's behaviour before touching it

**Files:**
- Test: `tests/gui/browse/test_single_mode_survives_removal.py` (create)

**Interfaces:**
- Produces: `test_browse_layout_has_no_view_mode_toggle`,
  `test_browse_single_body_is_unconditional` — the executable statement of this phase's
  "Single is unchanged" contract.

- [ ] **Step 1: Enumerate what Single mode owns today**

```bash
uv run grep -n "BROWSE_SINGLE\|BROWSE_VIEW_MODE\|BROWSE_TIMELINE_BODY" \
  src/phenotypic/gui/browse/_layout.py src/phenotypic/gui/browse/_callbacks.py
```
Record the result. `BROWSE_VIEW_MODE_TOGGLE` and `BROWSE_TIMELINE_BODY` disappear;
`BROWSE_SINGLE_BODY` **survives** and must stop being style-driven.

- [ ] **Step 2: Write the failing test**

```python
"""Browse keeps Single mode and loses the mode toggle.

The point of this file is the *second* test: a removal that hides the surviving
mode instead of unhiding it is the failure mode with the widest blast radius,
and it is invisible in a unit test that only checks for absence.
"""

from phenotypic.gui.browse import _ids as ids
from phenotypic.gui.browse._layout import build_browse_layout


def _ids_in(component) -> set[str]:
    found: set[str] = set()

    def walk(node) -> None:
        node_id = getattr(node, "id", None)
        if isinstance(node_id, str):
            found.add(node_id)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                walk(child)
        elif children is not None:
            walk(children)

    walk(component)
    return found


def test_browse_layout_has_no_view_mode_toggle():
    present = _ids_in(build_browse_layout())
    assert ids.BROWSE_SINGLE_BODY in present
    assert not any(name.startswith("browse-tl-") for name in present)
    assert "browse-view-mode-toggle" not in present
    assert "browse-timeline-body" not in present


def test_browse_single_body_is_unconditional():
    """Single must not be hidden by a leftover ``display: none``."""

    def find(node, target):
        if getattr(node, "id", None) == target:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                hit = find(child, target)
                if hit is not None:
                    return hit
        elif children is not None:
            return find(children, target)
        return None

    body = find(build_browse_layout(), ids.BROWSE_SINGLE_BODY)
    assert body is not None
    assert (getattr(body, "style", None) or {}).get("display") != "none"
```

Note the string literals `"browse-view-mode-toggle"` / `"browse-timeline-body"` — the
constants are being deleted, so the test cannot import them.

- [ ] **Step 3: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/gui/browse/test_single_mode_survives_removal.py -v
```
Expected: `test_browse_layout_has_no_view_mode_toggle` FAILS (the toggle is present).
`test_browse_single_body_is_unconditional` may pass already — that is fine; it is a guard
against what task 2.3 could break, not a statement about today.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/gui/browse/test_single_mode_survives_removal.py
git commit -m "test(gui): pin Browse Single mode against the timeline removal"
```

---

### Task 2.2: Delete the four browse helpers, the assets and their tests

**Files:**
- Delete: `src/phenotypic/gui/browse/_thumb_routes.py`, `_timeline_records.py`,
  `_capture_time.py`, `_plate_pattern.py`
- Delete: `src/phenotypic/gui/browse/_assets/timeline.js`, `_assets/timeline.css`
- Delete: `tests/gui/browse/test_timeline_callbacks_helpers.py`,
  `test_timeline_records.py`, `test_capture_time.py`, `test_plate_pattern.py`
- Delete: `tests/e2e/gui/test_browse_timeline.py`, `tests/e2e/gui/test_browse_compare_strip.py`

- [ ] **Step 1: Re-verify each helper is timeline-only**

The spec (§1.1) records call-site evidence for all four. Re-run it — this is the single
highest-risk claim in the phase, because three of the four have names that sound like
general Browse infrastructure:

```bash
for m in _thumb_routes _timeline_records _capture_time _plate_pattern; do
  echo "--- $m"
  uv run grep -rn "$m" src/ tests/ --include='*.py' | grep -v "browse/$m.py"
done
```

Expected, per spec:
- `_thumb_routes` → `browse/_app.py:33, :84` only.
- `_timeline_records` → `browse/_callbacks.py:50` only.
- `_capture_time` → `browse/_callbacks.py:44` only, feeding `_capture_time_of` at
  `:1531` inside `build_browse_records`. **It is not the EXIF chip** on the front metadata
  row — that resolves through `browse/_metadata.py`. Confirm with
  `uv run grep -n "capture\|exif" src/phenotypic/gui/browse/_metadata.py`.
- `_plate_pattern` → `browse/_callbacks.py:46` and `_timeline_records.py:17` only.

**If any grep shows a consumer outside those, stop.** A helper with a Single-mode consumer
is not deletable and the spec needs an amendment, not a workaround.

- [ ] **Step 2: Delete**

```bash
git rm src/phenotypic/gui/browse/_thumb_routes.py \
       src/phenotypic/gui/browse/_timeline_records.py \
       src/phenotypic/gui/browse/_capture_time.py \
       src/phenotypic/gui/browse/_plate_pattern.py \
       src/phenotypic/gui/browse/_assets/timeline.js \
       src/phenotypic/gui/browse/_assets/timeline.css
git rm tests/gui/browse/test_timeline_callbacks_helpers.py \
       tests/gui/browse/test_timeline_records.py \
       tests/gui/browse/test_capture_time.py \
       tests/gui/browse/test_plate_pattern.py \
       tests/e2e/gui/test_browse_timeline.py \
       tests/e2e/gui/test_browse_compare_strip.py
```

`test_browse_compare_strip.py` goes despite its name: its own module docstring records
that it drives the Browse **Timeline** surface (`#browse-tl-compare-btn`,
`window.__phenotypicTimeline`). Confirm before deleting:

```bash
git show HEAD:tests/e2e/gui/test_browse_compare_strip.py | head -20
```

Do not commit yet — the tree does not import.

---

### Task 2.3: Strip the timeline surface out of Browse

**Files:**
- Modify: `src/phenotypic/gui/browse/_ids.py` — 64 `BROWSE_TL_*` names + `:47-49`
- Modify: `src/phenotypic/gui/browse/_layout.py` — toggle `:274`, single body `:296`, `build_timeline_body` `:320-585`
- Modify: `src/phenotypic/gui/browse/_callbacks.py` — imports `:30, :39, :44, :46, :50`; TL callbacks
- Modify: `src/phenotypic/gui/browse/_cache.py` — timeline-thumb cache entries
- Modify: `src/phenotypic/gui/browse/_app.py:33, :84` — `_thumb_routes` registration

- [ ] **Step 1: Remove the ids**

Delete all 64 `BROWSE_TL_*` constants, plus `BROWSE_VIEW_MODE_TOGGLE` (`:47`) and
`BROWSE_TIMELINE_BODY` (`:49`), and each of their `__all__` entries (`:130-132` and the
`BROWSE_TL_*` block). **Keep `BROWSE_SINGLE_BODY` (`:48`).**

```bash
uv run grep -c "BROWSE_TL" src/phenotypic/gui/browse/_ids.py
```
Expected after the edit: `0`. Before: `64`.

- [ ] **Step 2: Remove the layout surface**

- Delete `build_timeline_body()` entirely (`_layout.py:320` to the end of that function,
  which closes with `id=ids.BROWSE_TIMELINE_BODY` at `:584`).
- Delete the view-mode toggle component at `:274`.
- At `:296`, make `BROWSE_SINGLE_BODY` unconditional: remove any `style` that the toggle
  callback drove, and remove the wrapper that switched between the two bodies so Single is
  a direct child.
- Remove `build_timeline_body` from `__all__` if present.

- [ ] **Step 3: Remove the callbacks and imports**

In `_callbacks.py` remove:
- `:30` `stepped_timeline_tile_size_from_trigger` (from `_shared.timeline`)
- `:39` `from phenotypic.gui._shared.timeline import build_matrix, build_timeline_grid`
- `:44` `from phenotypic.gui.browse._capture_time import read_capture_time`
- `:46` the `_plate_pattern` import block
- `:50` the `_timeline_records` import block
- the timeline helpers and callbacks: `timeline_thumb_url` (`:505`),
  `timeline_revision_token` (`:510`), `render_timeline_grid` (`:527`),
  `_reset_timeline_for_source` (`:931`), the `_capture_time_of` helper at `:1531`, and the
  mode-switch callback at `:1353-1360` **including its clientside counterpart at `:1370`**.

```bash
uv run grep -n "timeline\|BROWSE_TL\|view_mode" src/phenotypic/gui/browse/_callbacks.py
```
Expected after the edit: no hits. Grep is the completion check — the callback bodies are
long and interleaved, so read the whole file once after editing.

- [ ] **Step 4: Remove the cache entries and the thumb route**

```bash
uv run grep -n "timeline\|thumb" src/phenotypic/gui/browse/_cache.py \
                                  src/phenotypic/gui/browse/_app.py
```
Remove the timeline-thumb cache entries in `_cache.py`, and in `_app.py` the
`_thumb_routes` import (`:33`) and its `register` call (`:84`).

**Do not remove the Browse DZI/`BrowseCache` path.** Browse keeps libvips → DZI → OSD as
its only pixel path (viv-rebuild spec §9); only the *timeline thumbnail* route goes.

- [ ] **Step 4b: Repair the five test files this phase breaks**

Neither the spec's delete list nor an earlier draft of this plan mentioned these. Verified
2026-08-26 — reference counts are to `BROWSE_TL_*` / `_thumb_routes` / `BROWSE_VIEW_MODE_TOGGLE`
/ `BROWSE_TIMELINE_BODY`:

| File | Refs | Action |
|---|---|---|
| `tests/gui/browse/test_thumb_routes.py` | 2 | **delete** — `from ... import _thumb_routes` at `:10` is a collection `ImportError` |
| `tests/gui/browse/test_ids.py` | 33 | **edit** — drop the timeline half, keep the Single-mode id assertions |
| `tests/gui/browse/test_layout.py` | 22 | **edit** — same |
| `tests/gui/browse/test_compare_layout.py` | 7 | **edit** — same |
| `tests/integration/gui/test_browse_refresh.py::test_timeline_reset_consumes_shared_refresh_revision` | — | **delete that test** |

**Two `✅ shipping` FEATURES.md rows point at tests in that list, and both sit outside any
section this phase edits** — so breaking them fails `features-md-gate` from rows nobody
touched:

```text
FEATURES.md:64  Refresh button              -> test_browse_refresh.py::test_timeline_reset_consumes_shared_refresh_revision
FEATURES.md:87  Image dropdown + ‹/› stepper -> test_layout.py::test_prev_next_buttons_use_wider_stable_hit_target
```

Row `:87`'s target survives (it is a Single-mode test — keep it when editing
`test_layout.py`). Row `:64`'s target is being deleted: **repoint it** at a surviving
refresh test, or restate the row without the Timeline half. Do this in task 2.4's ledger
commit, not here.

- [ ] **Step 5: Run the tests written in task 2.1**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/gui/browse/test_single_mode_survives_removal.py -v
```
Expected: both PASS.

- [ ] **Step 6: Run the whole browse unit suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/browse -n 4 -q
```
Expected: PASS. **Any failure in a non-timeline browse test is a regression in Single
mode** and must be fixed here, not deferred.

- [ ] **Step 7: Run the Browse e2e suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k browse -q
```
Expected: PASS for the Single-mode tests. This is the check that catches a Single body
left hidden — the unit test in 2.1 checks the built layout, this checks the running app.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/browse/
git add -A src/phenotypic/gui/browse tests/gui/browse tests/gui/browse tests/e2e/gui
git commit -m "refactor(gui): delete Browse Timeline mode, keep Single unchanged"
```

---

### Task 2.4: Retire the Browse Timeline ledger rows, tutorial and capture function

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md:104-126` (Browse Timeline block — remove)
- Modify: `src/phenotypic/gui/WORKFLOWS.md:54` (`browse_timeline` row — remove)
- Modify: `scripts/capture_gui_tutorial_screenshots.py:1156` and its call sites
- Delete: `docs/source/tutorials/gui/19_browse_timeline.md` and its image directory
- Modify: `docs/source/tutorials/gui/index.md`

- [ ] **Step 1: Remove the ledger rows and the tutorial**

Follow phase 1 task 1.4 steps 1-4 verbatim, substituting: `FEATURES.md:104-126`,
`WORKFLOWS.md:54`, `_capture_browse_timeline` (`:1156`), page
`19_browse_timeline.md`, images `docs/source/_static/gui_images/browse_timeline/`.

**Do not touch the `18_browse.md` page** — that is Single mode's tutorial and it survives.

- [ ] **Step 2: Run the three gates**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --skip-cli
```
Expected: all exit 0.

- [ ] **Step 3: Commit**

```bash
git add -A src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
           scripts/capture_gui_tutorial_screenshots.py docs/source
git commit -m "docs(gui): retire the Browse Timeline ledger rows and tutorial"
```

---

> **Tasks 2.5-2.6 were phase 3.** Split out, they bought nothing: phase 1 kills the
> results consumer and tasks 2.1-2.4 kill the browse consumer, so the engine's consumer
> count reaches zero here. The split cost a conditional in *both* directions — phase 1 task
> 1.4 had to say "remove `FEATURES.md:536-537` only if phase 3 is in the same PR", and the
> old phase 3 carried its mirror. A branch that exists only because of a phase boundary is
> that boundary billing itself to the executor.

### Task 2.5: Delete the engine and its tests

**Files:**
- Delete: `src/phenotypic/gui/_shared/timeline/` (5 modules)
- Delete: `tests/gui/_shared/timeline/` (8 files)
- Modify: whichever CI file carries the `timeline.js` byte-equality guard (located in
  task 1.2 step 2)

**Interfaces:**
- Consumes: the absence established by phases 1 and 2.
- Produces: absence. Phase 6's dangling-reference test makes it permanent.

- [ ] **Step 1: Confirm both consumers are actually gone**

This is the precondition, and it is cheap to check:

```bash
uv run grep -rn "_shared.timeline\|_shared import timeline" src/ tests/ --include='*.py'
```

Expected: **no hits in `src/`**. Hits are allowed only in `tests/gui/_shared/timeline/`,
which this task deletes.

**If `src/` has any hit, stop.** Phase 1 or 2 is incomplete; finish it first. Deleting
here would leave an unimportable tree with no phase left to repair it.

- [ ] **Step 2: Confirm the timeline stylesheet/script guard's remaining subject**

```bash
uv run grep -rn "timeline.js\|timeline.css" .github/ scripts/ tests/ src/
```

Expected: only the byte-equality guard, now referring to two paths that no longer exist.
Note the exact file and line.

- [ ] **Step 3: Delete the engine, its tests, and the guard**

```bash
git rm -r src/phenotypic/gui/_shared/timeline
git rm -r tests/gui/_shared/timeline
```
Then remove the byte-equality guard identified in step 2. A guard comparing two absent
files either fails the build or silently passes on a vacuous truth; neither is a state to
leave behind.

- [ ] **Step 4: Prove both apps still import and build**

```bash
QT_QPA_PLATFORM=offscreen uv run python -c "
from phenotypic.gui.shell._app import create_app as shell_app
from phenotypic.gui.results_viewer._app import create_app as rv_app
from phenotypic.gui.browse._layout import build_browse_layout
build_browse_layout()
print('imports clean')
"
```
Expected: `imports clean`. This is the check phase 6 formalizes as a test; running it by
hand here catches the failure at the moment it is introduced.

- [ ] **Step 5: Run the GUI unit suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui -n 4 -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/gui tests/gui .github
git commit -m "refactor(gui): delete the shared timeline engine with its last consumer"
```

---

### Task 2.6: Retire the shared-engine ledger rows

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md:536-537`

- [ ] **Step 1: Remove the two rows**

Delete the **Timeline shared engine** and **Compare-strip cap logic** rows at
`FEATURES.md:536-537`. If phase 1 task 1.4 step 1 already removed them (same-PR
execution), confirm and skip:

```bash
uv run grep -n "shared engine\|Compare-strip" src/phenotypic/gui/FEATURES.md
```

- [ ] **Step 2: Run the ledger gates**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
```
Expected: both exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/phenotypic/gui/FEATURES.md
git commit -m "docs(gui): retire the timeline shared-engine ledger rows"
```
# Phase 4 — Tune: unmount

**Spec:** §2, §6. **Depends on:** nothing. **Blocks:** nothing.

**Deliverable:** `/tune/` is unreachable — no dispatcher entry, no nav leaf, no chrome
wrap. `src/phenotypic/gui/tune/` stays on disk, imports cleanly, and keeps its unit tests
passing. Its e2e tests are **skip-marked with a reason naming this spec**, not deleted:
they are the acceptance suite for the eventual re-mount.

> **`MOUNT_TUNE` and `TITLE_TUNE` stay** in `_config.py` (`:235`, `:844`). They are
> declarations, not registrations; the retained `gui/tune/` code still references them,
> and removing them is churn against a sub-app we intend to bring back (spec §2).

---

### Task 4.1: Remove the mount and the nav leaf

**Files:**
- Modify: `src/phenotypic/gui/shell/_app.py:56` (import), `:285`, `:603-613`, `:655`, `:667`
- Modify: `src/phenotypic/gui/shell/_layout.py:37, :72, :130, :140, :172`
- Test: `tests/unit/gui/shell/test_tune_is_unmounted.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `test_nav_model_has_no_tune_leaf`, `test_dispatcher_has_no_tune_mount` —
  phase 6's dangling-reference test asserts the same `NAV_MODEL` property; these are the
  narrower, faster versions.

- [ ] **Step 1: Write the failing test**

```python
"""Tune is unmounted: unreachable from the UI, still importable on disk.

Both halves matter. The first two tests are the unmount; the third is the
'still importable' half of the contract, which is what distinguishes this
phase from a deletion.
"""

import importlib

from phenotypic.gui._config import MOUNT_TUNE
from phenotypic.gui.shell._layout import NAV_MODEL


def _leaf_ids(model) -> set[str]:
    found: set[str] = set()
    for group in model:
        for leaf in (group[1] if isinstance(group, tuple) else group):
            found.add(leaf)
    return found


def test_nav_model_has_no_tune_leaf():
    assert "shell-tab-tune" not in {str(x) for x in _leaf_ids(NAV_MODEL)}


def test_dispatcher_has_no_tune_mount(built_hub_dispatcher_mounts):
    assert MOUNT_TUNE.rstrip("/") not in built_hub_dispatcher_mounts


def test_tune_package_is_still_importable():
    assert importlib.import_module("phenotypic.gui.tune") is not None
```

Add a `built_hub_dispatcher_mounts` fixture returning the `DispatcherMiddleware` mount
keys from `shell/_app.py`'s `create_app`. Derive it by reading how `:655` builds that dict.
`_leaf_ids` must match `NAV_MODEL`'s real shape — read `shell/_layout.py:163-175` and
adapt; do not guess.

- [ ] **Step 2: Run it and watch two of three fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/shell/test_tune_is_unmounted.py -v
```
Expected: `test_tune_package_is_still_importable` PASSES; the other two FAIL.

- [ ] **Step 3: Remove the mount from `shell/_app.py`**

Remove:
- the `tune` import at `:285` and the `MOUNT_TUNE` / `SHELL_TAB_TUNE` imports at `:56, :70`
  **only if** nothing else in the file uses them (grep after editing);
- `_tick("tune")` at `:603`;
- the `tune_app = tune.create_app(...)` construction at `:604-610`;
- the chrome wrap at `:612-613`;
- the `MOUNT_TUNE.rstrip("/"): tune_app.server` dispatcher entry at `:655`;
- the `MOUNT_TUNE` reference at `:667`.

```bash
uv run grep -n "tune\|MOUNT_TUNE\|SHELL_TAB_TUNE" src/phenotypic/gui/shell/_app.py
```
Expected after the edit: no hits (the comment at `:600` may be reworded or removed).

- [ ] **Step 4: Remove the nav leaf from `shell/_layout.py`**

Remove `SHELL_TAB_TUNE` from the import (`:72`), from `_TAB_HREFS` (`:130`), from the
label map (`:140`), and from the `NAV_MODEL` Pipeline group tuple (`:172`) — so
`(SHELL_TAB_BUILDER, SHELL_TAB_TUNE, SHELL_TAB_RUN)` becomes
`(SHELL_TAB_BUILDER, SHELL_TAB_RUN)`. Remove the `MOUNT_TUNE` import at `:37` if it has no
other user in the file.

Update the prose comment at `:163` — it names "tune in ..." as part of the Pipeline group.

- [ ] **Step 5: Run the test**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/shell/test_tune_is_unmounted.py -v
```
Expected: all three PASS.

- [ ] **Step 6: Confirm `_config.py` was NOT touched**

```bash
git diff --stat src/phenotypic/gui/_config.py
```
Expected: **empty**. `MOUNT_TUNE` and `TITLE_TUNE` stay (spec §2). A diff here is a spec
violation.

- [ ] **Step 7: Run the shell suite and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/shell -n 4 -q
uv run ruff check --fix src/phenotypic/gui/shell/ tests/unit/gui/shell/
git add -A src/phenotypic/gui/shell tests/unit/gui/shell
git commit -m "refactor(gui): unmount the Tune sub-app, retaining its package"
```

---

### Task 4.2: Skip-mark the Tune e2e suite

**Files:**
- Modify: every test module under `tests/e2e/gui/` that drives `/tune/`

- [ ] **Step 1: Find them**

```bash
uv run grep -rln "tune" tests/e2e/gui/
```

- [ ] **Step 2: Add a module-level skip whose reason names this spec**

At the top of each module found, after the docstring:

```python
import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "Tune is unmounted by "
        "docs/superpowers/specs/2026-08-26-gui-simplification-removals "
        "(spec section 2). These tests are the acceptance suite for the "
        "re-mount; delete this marker when /tune/ is mounted again."
    )
)
```

The reason string must contain the spec path so the marks are greppable when the surface
returns. That greppability is the whole mitigation for spec §10's "skip-marks rot" risk —
a bare `reason="unmounted"` does not discharge it.

- [ ] **Step 3: Confirm they skip rather than fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k tune -q
```
Expected: all skipped, none failed, none errored. A **collection error** means the module
imports something from the mount path at import time — move the `pytestmark` above the
offending import, or guard it.

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/gui
git commit -m "test(gui): skip-mark the Tune e2e suite pending re-mount"
```

---

### Task 4.3: Mark Tune unmounted in the ledgers and retire its tutorial

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md`, section `` ## Tune co-pilot (`/tune/`) `` (**edit**, not remove) — heading at `:426`, section runs to `:493`. The spec's `419-486` is wrong; `:419-425` are Run-console rows.
- Modify: `src/phenotypic/gui/WORKFLOWS.md:56` (`tune_copilot` row — remove)
- Modify: `scripts/capture_gui_tutorial_screenshots.py:2813` + call sites + harness block `:2381-2464`, `:2661`
- Delete: `docs/source/tutorials/gui/16_tune_copilot.md` + its image directory
- Modify: `docs/source/tutorials/gui/index.md`

- [ ] **Step 1: Edit — do not delete — the FEATURES.md section**

The rows under `` ## Tune co-pilot (`/tune/`) `` (`:426-493`) describe an **unmounted** surface. Per spec §6, mark
them unmounted with a pointer to this spec and change their status **off** `✅ shipping`
so `check_features_md.py` stops resolving their refs. Add a section note:

```markdown
> **Unmounted.** The `/tune/` sub-app is retained on disk but is not reachable from the
> UI. See `docs/superpowers/specs/2026-08-26-gui-simplification-removals` §2. The rows
> below describe code that exists and is unit-tested, not a surface a user can reach.
```

**Use the status `⏸ unmounted`.** This was settled during plan refinement (ledger ORCH-5);
do not re-derive it and do not escalate it.

`scripts/check_features_md.py` defines three constants — `✅ shipping` (`:33`),
`🚧 in progress` (`:34`), `🔭 planned` (`:35`) — but its row loop (`:148-176`) **never
validates status against an allowed set**. It branches: `== STATUS_IN_PROGRESS` → collect,
which `--strict` then fails (`:178-184`); `!= STATUS_SHIPPING` → `continue`, silently
skipped; otherwise resolve refs. So `⏸ unmounted` requires **no script change**: it stops
ref resolution and passes `--strict`.

None of the three existing values is right, which is why a fourth is warranted:

| Value | Why not |
|---|---|
| `✅ shipping` | keeps resolving refs *and* asserts a user can reach it — the false claim this spec exists to prevent |
| `🚧 in progress` | **fails `--strict`**, i.e. blocks the merge gate |
| `🔭 planned` | passes, but describes a fully built, parked sub-app as unbuilt |

Spec §6 asks the ledger to carry "exists but unreachable" as a state. `⏸ unmounted` is that
state named honestly.

- [ ] **Step 2: Remove the WORKFLOWS.md row, capture function and tutorial**

WORKFLOWS.md rows are workflow round-trips, not capability records — an unreachable
workflow cannot be captured, so row `:56` is **removed** rather than annotated. Then follow
phase 1 task 1.4 steps 3-4 for `_capture_tune_copilot` (`:2813`), its call sites, the
harness blocks at `:2381-2464` and `:2661`, page `16_tune_copilot.md`, and its images.

- [ ] **Step 3: Run the three gates**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --skip-cli
```
Expected: all exit 0.

- [ ] **Step 4: Commit**

```bash
git add -A src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
           scripts/capture_gui_tutorial_screenshots.py docs/source
git commit -m "docs(gui): mark Tune unmounted and retire its workflow row"
```
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
# Phase 6 — Verification & docs

**Spec:** §7, §6 (last paragraph). **Depends on:** phases 1-5.

**Deliverable:** the positive checks spec §7 requires, plus `gui/CLAUDE.md` brought in line
with the five remaining mounts.

> **Why positive checks.** Removal is verified by absence, which is the weakest kind of
> test — an absent module cannot fail a test that no longer exists. These carry the weight
> instead: they assert what must still *work*.
>
> Spec §7 names three. **Two survive review**: check 1 (the apps build) reduced to the half
> that is not already covered, and check 3 (curation), re-pointed at the tests that actually
> prove it. Check 2 (no dangling references) is cut — see task 6.2 for why.

---

### Task 6.1: Both apps import and build a layout

**Files:**
- Test: `tests/unit/gui/test_apps_build_after_simplification.py` (create)

**Interfaces:**
- Consumes: nothing; `test_layout_tab_shape.py` (phase 1, edited in phase 5) owns tab shape.
- Produces: the check that catches a missed import of a deleted module — specifically in
  the shell, which no other test builds. Spec §7 check 1.

- [ ] **Step 1: Write the test**

```python
"""The hub app still constructs after the simplification.

Spec section 7, check 1 -- reduced to its load-bearing half. A deleted module
that some module still imports fails here at ``create_app`` time, which is the
only place it *can* fail now that the tests for the deleted surfaces are gone
too.

The results viewer's own tab shape is NOT re-asserted here: phase 1 created
``test_layout_tab_shape.py`` for exactly that and phase 5 edited it to the
final two-tab list. Restating it would ship a second ``dbc.Tabs`` walker beside
that file's, two walkers over one structure. What is genuinely uncovered is the
SHELL's ``create_app`` -- no phase 1-5 test builds it, and it is what catches a
missed import after the Tune unmount.
"""


def test_hub_app_constructs(tmp_output_root):
    from phenotypic.gui.shell._app import create_app

    assert create_app(root=tmp_output_root) is not None
```

`tmp_output_root` must be a real, discoverable output root. **Reuse the existing fixture**
rather than writing one — the results-viewer e2e suite already builds sandboxes with
`write_master` / `write_measurements_mirror` from `tests._output_layout`:

```bash
uv run grep -rn "write_master\|OutputRoot.discover\|tmp_output_root" tests/ | head -20
```
Match `create_app`'s real signature in both apps before writing the calls above.

- [ ] **Step 2: Run it**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/test_apps_build_after_simplification.py -v
```
Expected: PASS. A `ModuleNotFoundError` here names the exact missed import — fix it in the
phase that owned that module, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/gui/test_apps_build_after_simplification.py
git commit -m "test(gui): assert both apps build after the simplification"
```

---

### Task 6.2: ~~No dangling references~~ — **cut**

> **Dropped in round-1 review.** An earlier draft added
> `tests/unit/gui/test_no_dangling_removed_references.py`, an `rglob` + substring scan
> asserting no `gui/` module mentions a removed name. Four reasons it does not earn its
> keep:
>
> 1. **The real failure mode fails loudly at `create_app`**, which task 6.1 already covers.
>    The residual case it claims to catch — a lazy in-function import or an `importlib`
>    string — is hypothetical: spec §1.1 enumerates every call site of all six deleted
>    modules and every one is a top-level import.
> 2. **It is a substring scan**, so `_capture_time` matches `_capture_timeout`,
>    `read_capture_time`, and any prose mention. The draft conceded this and instructed that
>    docstring hits "should be reworded" — a test dictating prose.
> 3. Its `test_nav_model_carries_no_tune_leaf` **duplicates phase 4's** version and does it
>    worse: `"tune" not in repr(NAV_MODEL).lower()` trips on any word containing "tune" and
>    passes if the leaf is spelled differently.
> 4. Ruff's `F401` plus task 6.1's import test already cover what matters.
>
> Nothing replaces it. If a deleted module is still referenced, `create_app` raises
> `ModuleNotFoundError` and task 6.1 fails — which is the loud, specific failure this scan
> would have restated less precisely.

---

### Task 6.3: Colony curation still works — unmodified

**Files:**
- Verify only: `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py`

**Interfaces:**
- Consumes: everything. This is spec §7 check 3 and the executable statement of §5.

> **`test_colony_callbacks_helpers.py` is the constraint, not the proof.** All 15 of its
> tests drive pure `_triage_callbacks` helpers against **hand-built `ctx.triggered` dicts**
> in `tmp_path`. Nothing asserts that a radial exists on a tile, and nothing asserts that
> anything reaches disk — so it would pass **unmodified** while the radial ceased to be
> rendered at all. Keep the unmodified rule (it is the executable statement that §5 was
> respected), but the chain is proved by three other files, all collected by
> `pyproject.toml:218`'s `testpaths` and **none of them run by any command in this plan
> before this fix**.

- [ ] **Step 1: Prove the file is byte-unchanged across the whole plan**

```bash
git diff --stat <baseline-sha> -- \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py
```
where `<baseline-sha>` is the commit this plan started from. Expected: **empty output**.

Per spec §7: "If a test in `test_colony_callbacks_helpers.py` needs editing, §5 has been
violated." A non-empty diff is a **stop-and-escalate**, not something to reconcile.

- [ ] **Step 2: Run the three tests that actually prove the chain**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/gui/results_viewer/colony_view/test_grid.py::test_build_grid_tiles_carry_radial_trigger_not_old_remove_button \
  tests/integration/gui/test_triage_callbacks.py::test_colony_wedge_mark_writes_category_parquet_and_drops_mirror \
  tests/unit/cli/test_cli_error_outputs.py \
  -v
```

| Test | What it is the proof of |
|---|---|
| `test_grid.py:454` | the **only** assertion that a cell carries `colony-radial-trigger` — exactly what a deck.gl rewrite endangers |
| `test_triage_callbacks.py:227` | `deliverables/errors/debris.parquet` gains the object and the mirror drops it — the executable statement of spec §5 |
| `test_cli_error_outputs.py` | the `reemit_error_deliverables` end of the round-trip |

Expected: PASS. Note the first lives under `tests/gui/`, not `tests/unit/gui/` — a
`tests/unit/gui` invocation never touches it.

- [ ] **Step 3: Run the rest of the colony suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py \
  tests/unit/gui/results_viewer/colony_view \
  tests/gui/results_viewer/colony_view -n 4 -q
```
Expected: PASS.

For the record, the chain verified during review and unchanged by this plan: radial →
`apply_wedge_mark` / `bulk_mark` → `mutate_and_payload` → `_save_locked` →
`_publish_if_current` (`_curation_labels.py:773`) → `_write_curated_mirror` +
`_write_category_parquets` + `_write_labels_parquet`. `colony_view/_callbacks.py:46-51`
imports the same helpers QC does, so **no hop needs QC or Error mounted** — which is spec
§5's claim, confirmed.

- [ ] **Step 4: No commit** — this task is verification only.

---

### Task 6.4: Bring `gui/CLAUDE.md` in line

**Files:**
- Modify: `src/phenotypic/gui/CLAUDE.md`

- [ ] **Step 1: Cut the sub-app table to five mounts**

```bash
uv run grep -n "tune\|Timeline\|Heatmap\|Error\|QC\|mount" src/phenotypic/gui/CLAUDE.md
```
Remove Tune from the mount table so it lists five, and mark the Error-analysis-tab section
unmounted with a pointer to the spec (spec §6, last paragraph).

- [ ] **Step 2: Check the root `CLAUDE.md` GUI section too**

```bash
uv run grep -n "tune\|Timeline" CLAUDE.md
```
The root file's "GUI hub" section lists sub-apps; if it names Tune as reachable, correct it.
`AGENTS.md` is a symlink to `CLAUDE.md`, so it follows automatically — do not edit it
separately.

- [ ] **Step 3: Final full gate run**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui -n 4 -q
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --skip-cli
```
Expected: all green. Then run the full unit suite as a Slurm job per the
**`run-phenotypic-test`** skill and report it as "green except the known baseline failure".

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/gui/CLAUDE.md CLAUDE.md
git commit -m "docs(gui): record the five remaining mounts and the unmounted tabs"
```
# GUI simplification — removals: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a GUI whose results viewer has exactly two tabs (Plate, Colony), whose
Browse tab has no view-mode toggle, and whose Tune sub-app is unreachable — deleting the
Timeline engine outright and unmounting Tune / Heatmap / Error / QC without deleting them.

**Architecture:** Two mechanisms, kept strictly distinct. **Delete** removes modules,
tests, assets, ledger rows, tutorial pages and capture functions from the tree. **Unmount**
removes the mount, the nav leaf, the `dbc.Tab`, *and the callback registration*, while the
package stays on disk with its unit tests passing and its e2e tests skip-marked. Work is
sequenced consumer-first: both Timeline consumers die before the shared engine they share,
so no phase ever leaves an import dangling.

**Tech Stack:** Python 3.11+, Dash / dash-bootstrap-components, Flask blueprints, pytest,
Playwright (e2e), `uv` as the sole runner.

**Spec:** [`docs/superpowers/specs/2026-08-26-gui-simplification-removals/design.md`](../../specs/2026-08-26-gui-simplification-removals/design.md)

**Baseline:** branch `feat/gui-ome-zarr-sync`, restacked onto
`worktree-ome-zarr-image-store` head `bf0d01a1`. Every `file:line` in the spec was
re-verified against this tree on 2026-08-26 and holds (see §Verified baseline).

---

## Global Constraints

- **`uv` is the sole runner.** Never bare `python`/`pip`. `uv run <cmd>`.
- **`QT_QPA_PLATFORM=offscreen` is mandatory** for any pytest invocation. Without it the
  interpreter aborts at ~79% with no summary.
- **Never `pytest -n auto`.** `nproc` reports the node's cores, not the allocation's.
  Pass an explicit `-n 4` or omit `-n`. Use the **`run-phenotypic-test`** skill for any
  non-trivial run; the full `tests/unit` suite is ~65 minutes and is a Slurm job
  (`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`).
- **`uv run ruff check --fix <explicit paths you changed>`.** Never bare `ruff check --fix`.
- **Known-failing baseline test, not caused by this work:**
  `tests/unit/cli/test_cli_terminal_failures.py::test_concurrent_process_appends_do_not_lose_records`
  fails on a 4-core allocation because it spawns 8 processes with a 20 s join timeout.
  Report the suite as "green except this one" and re-check it is still *this* test failing
  for *this* reason.
- **Three CI gates in `.github/workflows/gui-checks.yml` bind every phase that touches
  `src/phenotypic/gui/`:**
  - `features-md-gate` — a PR touching `gui/` **must** modify
    `src/phenotypic/gui/FEATURES.md`; then `scripts/check_features_md.py` and
    `--strict` must pass.
  - `workflows-md-gate` — `scripts/check_workflows_md.py -v` enforces the
    WORKFLOWS.md ↔ capture-function ↔ tutorial-page round trip.
  - `smoke-capture` — runs `scripts/capture_gui_tutorial_screenshots.py`.
  Ledger, capture-script and tutorial edits therefore live **inside** the phase that
  removes the surface, never in a follow-up phase.
- **The ledgers are at `src/phenotypic/gui/FEATURES.md` and
  `src/phenotypic/gui/WORKFLOWS.md`**, not the repo root. The spec cites them by bare
  filename; the paths above are the real ones.
- **NEVER edit FEATURES.md by line number. Anchor on its `##` / `###` headings.**
  **Every line range the spec §6 gives for FEATURES.md is wrong**, by a consistent ≈ +7
  offset past line ~400 — verified 2026-08-26. The worst case is not cosmetic: spec §6 and
  an earlier draft of phase 1 said "delete the Results Timeline rows at `FEATURES.md:372-394`",
  but `:372-377` are **Colony curation rows** (`Colony radial lazy-populate`,
  `Custom folder + ＋ Add custom`, `Bulk "Mark N as ▾" (colony)`, `Pixel layer toggle`) and
  the `### Results Timeline tab` heading is at `:379`. Following the spec literally
  **deletes four curation rows** — a direct spec §5 violation, and phase 5's `git diff`
  guard would not catch it because that guard watches `colony_view/` and the radial modules,
  not the ledger.

  | Section | Heading to anchor on | Spec said | Actually |
  |---|---|---|---|
  | Browse Timeline | `## Browse tab (source image viewer)` block | 104-126 | 102-126 |
  | Results Timeline | `### Results Timeline tab` | 372-394 | **379**-400 |
  | Tune co-pilot | `` ## Tune co-pilot (`/tune/`) `` | 419-486 | **426**-493 |
  | Timeline shared engine | rows naming `gui/_shared/timeline/` | 536-537 | **543**-544 |
  | QC tab | `## QC tab` | 587 | **594** |
  | QC Review | `## QC Review sub-view` | 617 | **624** |
  | Heatmap tab | `## Heatmap tab` | 658 | **665** |
  | Error analysis | `## Error analysis tab` | 677 | **684** |

  **WORKFLOWS.md's row numbers `:46/47/51/52/54/55/56` ARE correct** — verified individually.
  The defect is specific to FEATURES.md.
- **Add FEATURES.md to phase 5's spec-§5 diff guard.** The curation rows live in the ledger
  as well as in code, and only the code half was being watched.
- **Unmounted ≠ deleted in the ledger.** An unmounted surface's FEATURES.md row is
  **edited to say unmounted, with a pointer to this spec** — not removed. A deleted
  surface's row is removed. `check_features_md.py` only resolves refs for `✅ shipping`
  rows, so an unmounted row must not carry that status.
- **`colony_view/` is not touched by this plan.** See spec §5. If a test under
  `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` needs editing, the
  plan has been violated — stop and escalate.
- **Browse Single mode behaviour is unchanged.** Every J/K nav, filmstrip, keep-position
  and preparation affordance stays. Single stops being *one of two modes* and becomes the
  whole tab; that is a layout change, not a behaviour change.

---

## Verified baseline

Re-verified in this worktree on 2026-08-26. These are facts the phases depend on:

| Claim | Verified |
|---|---|
| All 10 delete-target paths exist | yes |
| All 6 delete-target test paths exist | yes |
| `results_viewer/_layout.py` imports at `:65` `_error_tab`, `:66` `_heatmap_tab`, `:72` `_qc_tab`, `:74` `timeline_view` | yes |
| `results_viewer/_layout.py` bodies at `:609` heatmap, `:610` error, `:615` qc; `dbc.Tabs` holds **6** tabs at `:622-655`; `active_tab=ids.TAB_PLATE_ID` at `:656` | yes |
| `results_viewer/_callbacks.py` register calls at `:113` heatmap, `:114` qc, `:115` error, `:116` timeline | yes |
| `browse/_ids.py` carries **64** `BROWSE_TL_*` names plus `BROWSE_VIEW_MODE_TOGGLE` (`:47`), `BROWSE_SINGLE_BODY` (`:48`), `BROWSE_TIMELINE_BODY` (`:49`) | yes |
| `browse/_callbacks.py` imports the four doomed modules at `:39, :44, :46, :50` | yes |
| `browse/_app.py` registers `_thumb_routes` at `:33` (import) and `:84` (call) | yes |
| `browse/_layout.py:320` `build_timeline_body`, toggle at `:274`, single body at `:296` | yes |
| Capture fns at `:1156, :1246, :1750, :1810, :1900, :1947, :2813` | yes |
| WORKFLOWS.md rows at `:46, :47, :51, :52, :54, :55, :56` | yes |
| Tutorial pages `10, 11, 15, 16, 17, 19, 20` exist; highest is `20_results_timeline.md` | yes |

---

## Phases

Phase 1 before phase 2; phases 4 and 5 are independent of both and of each other.

| # | Phase | Deliverable | Doc |
|---|---|---|---|
| 1 | Results Timeline tab — delete | Results viewer has 5 tabs; `timeline_view/` gone | [phase-1](phase-1-results-timeline.md) |
| 2 | Browse Timeline mode **+ the shared engine** — delete | Browse has no view-mode toggle; `_shared/timeline/` gone | [phase-2](phase-2-browse-timeline.md) |
| 4 | Tune — unmount | `/tune/` 404s; `gui/tune/` still imports and unit-tests | [phase-4](phase-4-tune-unmount.md) |
| 5 | Heatmap / Error / QC — unmount | Results viewer has 2 tabs; 3 packages retained | [phase-5](phase-5-analysis-tabs-unmount.md) |
| 6 | Verification & docs | Shell-build test, curation gate, `gui/CLAUDE.md` | [phase-6](phase-6-verification.md) |

**Phase 3 was folded into phase 2** (tasks 2.5-2.6). The shared engine's consumer count
reaches zero at the end of phase 2's first four tasks, so a separate phase bought nothing
and cost a same-PR conditional in both directions. Numbering is left with a gap rather than
renumbered, so the phase names in the ledger and commits keep resolving.

## Definition of done

1. `uv run pytest tests/unit/gui -n 4` green (minus the known baseline failure).
2. `uv run python scripts/check_features_md.py --strict` exits 0.
3. `uv run python scripts/check_workflows_md.py -v` exits 0.
4. `uv run python scripts/capture_gui_tutorial_screenshots.py --skip-cli` exits 0.
5. The two new checks from phase 6 pass, and the three curation-chain tests it names are
   green (one of them lives under `tests/gui/`, which a `tests/unit/gui` run never reaches).
6. `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` passes **unmodified** —
   `git diff --stat` shows zero lines changed in that file across the whole plan.
