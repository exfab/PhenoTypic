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
- Modify: `src/phenotypic/gui/FEATURES.md:419-486` (Tune co-pilot section — **edit**, not remove)
- Modify: `src/phenotypic/gui/WORKFLOWS.md:56` (`tune_copilot` row — remove)
- Modify: `scripts/capture_gui_tutorial_screenshots.py:2813` + call sites + harness block `:2381-2464`, `:2661`
- Delete: `docs/source/tutorials/gui/16_tune_copilot.md` + its image directory
- Modify: `docs/source/tutorials/gui/index.md`

- [ ] **Step 1: Edit — do not delete — the FEATURES.md section**

The Tune co-pilot rows at `:419-486` describe an **unmounted** surface. Per spec §6, mark
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
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --smoke
```
Expected: all exit 0.

- [ ] **Step 4: Commit**

```bash
git add -A src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
           scripts/capture_gui_tutorial_screenshots.py docs/source
git commit -m "docs(gui): mark Tune unmounted and retire its workflow row"
```
