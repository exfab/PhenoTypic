# Phase 6 — Verification & docs

**Spec:** §7, §6 (last paragraph). **Depends on:** phases 1-5.

**Deliverable:** the three positive checks spec §7 requires, plus `gui/CLAUDE.md` brought
in line with the five remaining mounts.

> **Why positive checks.** Removal is verified by absence, which is the weakest kind of
> test — an absent module cannot fail a test that no longer exists. These three carry the
> weight instead: they assert what must still *work*.

---

### Task 6.1: Both apps import and build a layout

**Files:**
- Test: `tests/unit/gui/test_apps_build_after_simplification.py` (create)

**Interfaces:**
- Consumes: the two-tab shape from phase 5.
- Produces: the check that catches a missed import of a deleted module. This is spec §7
  check 1.

- [ ] **Step 1: Write the test**

```python
"""Both apps still construct after the simplification.

Spec section 7, check 1. A deleted module that some module still imports fails
here at ``create_app`` time, which is the only place it *can* fail now that the
tests for the deleted surfaces are gone too.
"""

import dash_bootstrap_components as dbc

from phenotypic.gui.results_viewer import _ids as ids


def _only_tabs(layout) -> dbc.Tabs:
    found = []

    def walk(node) -> None:
        if isinstance(node, dbc.Tabs):
            found.append(node)
            return
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                walk(child)
        elif children is not None:
            walk(children)

    walk(layout)
    assert len(found) == 1, f"expected exactly one dbc.Tabs, found {len(found)}"
    return found[0]


def test_hub_app_constructs(tmp_output_root):
    from phenotypic.gui.shell._app import create_app

    assert create_app(root=tmp_output_root) is not None


def test_standalone_results_viewer_constructs_with_two_tabs(tmp_output_root):
    from phenotypic.gui.results_viewer._app import create_app

    app = create_app(output_root=tmp_output_root)
    tabs = _only_tabs(app.layout)
    assert [t.tab_id for t in tabs.children] == [
        ids.TAB_PLATE_ID,
        ids.TAB_COLONY_ID,
    ]
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

### Task 6.2: No dangling references

**Files:**
- Test: `tests/unit/gui/test_no_dangling_removed_references.py` (create)

**Interfaces:**
- Produces: spec §7 check 2. This is what stops a deleted module name from creeping back
  in via a future edit.

- [ ] **Step 1: Write the test**

```python
"""No module under ``src/phenotypic/gui/`` references a removed surface.

Spec section 7, check 2. A source scan rather than an import scan: an import
inside a function body or a string used by ``importlib`` would both slip past
a module-import check.
"""

from pathlib import Path

import phenotypic.gui as gui_pkg
from phenotypic.gui.shell._layout import NAV_MODEL

REMOVED_MODULE_NAMES = (
    "_shared.timeline",
    "_shared import timeline",
    "timeline_view",
    "_thumb_routes",
    "_timeline_records",
    "_capture_time",
    "_plate_pattern",
)


def test_no_gui_module_references_a_removed_module():
    root = Path(gui_pkg.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in REMOVED_MODULE_NAMES:
            if name in text:
                offenders.append(f"{path.relative_to(root)}: {name}")
    assert not offenders, "removed modules still referenced:\n" + "\n".join(offenders)


def test_nav_model_carries_no_tune_leaf():
    assert "tune" not in repr(NAV_MODEL).lower()
```

- [ ] **Step 2: Run it**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/test_no_dangling_removed_references.py -v
```
Expected: PASS. A hit on `_capture_time` inside a *docstring* still counts and should be
reworded — a name that survives only in prose invites its reintroduction.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/gui/test_no_dangling_removed_references.py
git commit -m "test(gui): forbid references to the removed timeline modules"
```

---

### Task 6.3: Colony curation still works — unmodified

**Files:**
- Verify only: `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py`

**Interfaces:**
- Consumes: everything. This is spec §7 check 3 and the executable statement of §5.

- [ ] **Step 1: Prove the file is byte-unchanged across the whole plan**

```bash
git diff --stat <baseline-sha> -- \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py
```
where `<baseline-sha>` is the commit this plan started from. Expected: **empty output**.

Per spec §7: "If a test in `test_colony_callbacks_helpers.py` needs editing, §5 has been
violated." A non-empty diff is a **stop-and-escalate**, not something to reconcile.

- [ ] **Step 2: Run the colony suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py \
  tests/unit/gui/results_viewer/colony_view -n 4 -q
```
Expected: PASS.

- [ ] **Step 3: Confirm the radial's write path still has a producer**

The radial writes `deliverables/errors/<category>.parquet`, and the CLI's
`reemit_error_deliverables` round-trip is its counterpart. Confirm both ends survive:

```bash
uv run grep -rn "reemit_error_deliverables" src/phenotypic/ | head
uv run grep -rn "build_radial_trigger\|CurationLabels" src/phenotypic/gui/ | head
```
Expected: both present, the radial's hits now confined to `colony_view/` and `_shared/`.

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
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --smoke
```
Expected: all green. Then run the full unit suite as a Slurm job per the
**`run-phenotypic-test`** skill and report it as "green except the known baseline failure".

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/gui/CLAUDE.md CLAUDE.md
git commit -m "docs(gui): record the five remaining mounts and the unmounted tabs"
```
