# Phase 3 — Shared timeline engine: delete

**Spec:** §1.1, §1.2, §6. **Depends on:** phases 1 **and** 2. **Blocks:** nothing.

**Deliverable:** `src/phenotypic/gui/_shared/timeline/` (5 modules) and
`tests/gui/_shared/timeline/` (8 files) are gone, along with the CI byte-equality guard
between the two now-deleted `timeline.js` copies. Nothing under `src/phenotypic/gui/`
imports `_shared.timeline`.

> **Why this is its own phase.** The engine is surface-agnostic by construction: its
> controller finds siblings by CSS class scoped to `.timeline-body`, never by
> `browse-tl-*` id, and a CI guard enforces that the two vendored `timeline.js` copies stay
> byte-equal. That design is what makes it deletable in one move *after* both consumers
> are gone — and undeletable before. Attempting it earlier produces a broken import in
> whichever consumer is still standing.

---

### Task 3.1: Delete the engine and its tests

**Files:**
- Delete: `src/phenotypic/gui/_shared/timeline/` (5 modules)
- Delete: `tests/gui/_shared/timeline/` (8 files)
- Modify: whichever CI file carries the `timeline.js` byte-equality guard (located in
  phase 1, task 1.2 step 2)

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

### Task 3.2: Retire the shared-engine ledger rows

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
