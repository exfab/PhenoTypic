# Phase 4 — Verification, ledgers and docs

**Spec:** §6. **Depends on:** phases 1-3.

**Deliverable:** all six of spec §6's checks green, the three `gui-checks` gates passing,
and the ledgers, tutorial and `gui/CLAUDE.md` describing the Viv-backed preview.

---

### Task 4.1: Close out spec §6's six checks

- [ ] **Step 1: Walk the checklist**

| Spec §6 check | Where it lives |
|---|---|
| Range on the preview route — `206`, not `200` | phase 1 task 1.1 |
| Session isolation — against a **real** second sandbox, not a crafted path | phase 1 task 1.1 |
| Traversal — per-segment guard rejects `..` in any position | phase 1 task 1.1 |
| Freshness survives the swap — via a **nested chunk** rewrite | phase 1 task 1.2 |
| Point picker unaffected — its tests pass **unmodified** | phase 2 task 2.2 step 6 |
| Scratch cap — oldest-first, focused scope never evicted | phase 3 task 3.2 |

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_preview_zarr_routes.py \
  tests/unit/gui/builder/test_preview_retention.py \
  tests/unit/gui/builder/test_shared_viv_asset.py \
  -n 4 -v
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder -k point_picker -q
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k builder -q
```
Expected: all PASS.

- [ ] **Step 2: Re-prove the point picker diff is empty across the whole plan**

```bash
git diff --stat <plan-baseline-sha> -- src/phenotypic/gui/builder/_point_picker.py
```
Expected: **empty**. Spec §4 makes this the executable statement that the picker stays on
DZI. A non-empty diff is a stop-and-escalate.

- [ ] **Step 3: Confirm both measurements were actually taken**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-builder-preview-viv/preview_scratch_budget.py
```
Expected: exit 0 with the derived cap printed. A non-zero exit means phase 3's constants
were never filled in and the cap in the code is an invented number.

---

### Task 4.2: Confirm `_dzi_tiler`'s real consumer set

**Files:**
- Test: `tests/unit/gui/test_dzi_tiler_consumers.py` (create)

**Interfaces:**
- Produces: a guard against a future "cleanup" deleting a module four surfaces still need.

> Three separate specs describe `_dzi_tiler` as being "removed" from *a* path. Read
> together they invite the conclusion that the module is dead. It is not: Browse keeps
> libvips → DZI → `BrowseCache` → OSD as its **only** pixel path, and the point picker has
> no store to read.

- [ ] **Step 1: Write the test**

```python
"""``_dzi_tiler`` keeps four consumers after the Viv migrations.

Recorded as a test because three specs each say the tiler is 'removed from
this path', and read together they suggest a module that can be deleted. It
cannot: Browse has no store behind its source images, and the point picker
picks points before any pipeline node has run.
"""

from pathlib import Path

import phenotypic.gui as gui_pkg

EXPECTED_CONSUMERS = {
    "browse/_app.py",
    "browse/_preparation.py",
    "browse/_preparation_routes.py",
    "builder/_point_picker.py",
}


def test_dzi_tiler_consumer_set_is_exactly_what_the_specs_expect():
    root = Path(gui_pkg.__file__).parent
    found = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*.py")
        if path.name != "_dzi_tiler.py" and "_dzi_tiler" in path.read_text("utf-8")
    }
    assert found == EXPECTED_CONSUMERS, (
        "the _dzi_tiler consumer set changed; update this test AND the specs "
        f"that enumerate it.\n  unexpected: {sorted(found - EXPECTED_CONSUMERS)}"
        f"\n  missing:    {sorted(EXPECTED_CONSUMERS - found)}"
    )
```

- [ ] **Step 2: Run it**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/test_dzi_tiler_consumers.py -v
```
Expected: PASS. A failure naming `results_viewer/_tile_routes.py` means viv-rebuild phase 3
step 4 was not completed; naming `builder/_preview_tiles.py` means phase 2 step 5 was not.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/gui/test_dzi_tiler_consumers.py
git commit -m "test(gui): pin the _dzi_tiler consumer set against stray cleanup"
```

---

### Task 4.3: Ledgers, tutorial and CLAUDE.md

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md`, `src/phenotypic/gui/WORKFLOWS.md`
- Modify: `scripts/capture_gui_tutorial_screenshots.py`
- Modify: `docs/source/tutorials/gui/03_build_pipeline.md`
- Modify: `src/phenotypic/gui/CLAUDE.md`

- [ ] **Step 1: Update the preview rows**

```bash
uv run grep -n "preview\|node preview\|DZI" src/phenotypic/gui/FEATURES.md | head -20
```
Update the node-preview rows to describe a Viv-backed pane over `/preview-zarr/...`, and
add a row for the retention policy — a cache that silently evicts is a user-visible
behaviour and belongs in the ledger.

- [ ] **Step 2: Refresh the builder tutorial and its capture**

`03_build_pipeline.md` shows the preview pane. Update prose and re-capture per the
**`gui-tutorial-capture`** skill, keeping the WORKFLOWS.md ↔ capture-function ↔
tutorial-page round trip closed:

```bash
uv run grep -n "build_pipeline\|_capture_build" \
  src/phenotypic/gui/WORKFLOWS.md scripts/capture_gui_tutorial_screenshots.py
```

- [ ] **Step 3: Update `gui/CLAUDE.md`**

Record: the builder preview reads its scratch `.ome.zarr` through `/preview-zarr/...` and
renders with the shared Viv façade; the point picker stays on `_dzi_tiler`; scratch scopes
are capped and swept at startup; the Viv artifact is served once, from the results-viewer
package.

- [ ] **Step 4: Run the three gates**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --smoke
```
Expected: all exit 0.

- [ ] **Step 5: Full suite as a Slurm job**

```bash
sbatch docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch
```
Report as "green except the known baseline failure", re-confirming it is still that test
failing for that reason.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
           src/phenotypic/gui/CLAUDE.md scripts/capture_gui_tutorial_screenshots.py \
           docs/source
git commit -m "docs(gui): record the Viv-backed builder preview and its cache policy"
```

---

### Task 4.4: Resolve or re-file the spec's open questions

**Files:**
- Modify: `docs/superpowers/specs/2026-08-26-builder-preview-viv/design.md`

- [ ] **Step 1: Close OQ1 with the mechanism actually used**

Spec §7 OQ1 asks whether `DispatcherMiddleware` can serve one `_assets/viv/` to both
sub-apps. Phase 2 task 2.1 step 1 determined it. Record the answer and the evidence.

- [ ] **Step 2: Close OQ2 with the measured cap**

Record `PREVIEW_SCOPE_RETENTION` and the per-revision measurement behind it.

- [ ] **Step 3: Leave OQ3 open, deliberately**

Whether preview stores should pyramid at all stays open — they are single-level today,
which is right for a preview pane, and it only changes if the pane grows a deep-zoom
gesture. Recorded so the decision is not made by accident later.

- [ ] **Step 4: Present the spec edits to the user; do not self-approve**

Closing an open question is a spec change. Draft it, report it, and wait.
