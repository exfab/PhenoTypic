# Phase 4 — Colony on deck.gl views (D1)

**Spec:** §6.2. **Depends on:** phases 0-3. **Blocks:** nothing.

**Deliverable:** the Colony grid renders as one `OrthographicView` per colony, each centred
on its centroid, with the Viv layer stack rendering into all of them and a single shared
`viewState`. Curation is retained throughout.

> **This phase is optional and is the first thing to cut.** Spec §6.2 stages the work as
> "ship first (D3), then D1", where D3 is "keep today's `build_tile_grid` chrome and change
> only the crop route — from overlay-PNG slicing to a level-0 chunk read." **D3 is already
> landed** on the store branch ([DRIFT.md](DRIFT.md) D-2): `_shared/tiles.py:665`
> `crop_colony` already prefers `crop_store_rgb`. So the staging already paid off — the
> data path is done and only the rendering layer remains.

---

### Task 4.1: Establish the virtualization cap before building on it

**Files:**
- Create: `docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/colony_view_budget.py`
- Modify: `docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/README.md`

**Interfaces:**
- Produces: the measured cell-count cap phase 4 task 4.3 enforces.

> Spec §6.2: "deck.gl re-renders every view each frame, so D1 needs a virtualization cap on
> cell count. The number is not set here — it is measured during D1, and until then D3 has
> no such limit." So the cap is measured **first**, not chosen and then justified.

- [ ] **Step 1: Write the budget script**

Per root `CLAUDE.md`: stdlib + numpy/scipy only (this one needs only the stdlib), never
imports `phenotypic`, exits non-zero on failure.

```python
"""Re-derive the per-frame cost of one OrthographicView per colony.

Claim under test (viewer-viv-rebuild spec section 6.2): deck.gl re-renders
every view each frame, so an uncapped colony grid degrades linearly in cell
count. This script derives the draw-call and texture budget for a plate's
worth of cells so the cap is chosen against a number.

Exits non-zero until the prototype measurement in task 4.1 step 3 has been
recorded, so no later task can proceed on an unmeasured cap.
"""

import sys

#: Layers rendered into EACH view: base image + label overlay.
LAYERS_PER_VIEW = 2
#: A common plate: 32 x 48 = 1536 colonies (backend section 2.3's example).
PLATE_CELLS = 1536
#: Measured cap, filled in from the prototype in step 3. None until measured.
RECORDED_CAP: int | None = None
#: Frame time, in ms, observed at RECORDED_CAP. Recorded beside the number so
#: the cap can be re-judged later without re-running the prototype blind.
RECORDED_FRAME_MS: float | None = None


def draw_calls(cells: int, layers: int = LAYERS_PER_VIEW) -> int:
    """Draw calls per frame: every view renders every layer."""
    return cells * layers


def crop_texture_bytes(cells: int, crop_px: int = 64, channels: int = 3) -> int:
    """Resident texture bytes for a grid of RGB crops."""
    return cells * crop_px * crop_px * channels


def main() -> int:
    for cells in (64, 256, 1024, PLATE_CELLS):
        print(
            f"{cells:5d} cells: {draw_calls(cells):6d} draw calls/frame, "
            f"{crop_texture_bytes(cells) / 1e6:7.2f} MB textures"
        )
    if RECORDED_CAP is None or RECORDED_FRAME_MS is None:
        print("NO MEASUREMENT: run the prototype in task 4.1 step 3")
        return 1
    print(
        f"cap {RECORDED_CAP} cells "
        f"({draw_calls(RECORDED_CAP)} draw calls, "
        f"{crop_texture_bytes(RECORDED_CAP) / 1e6:.1f} MB, "
        f"{RECORDED_FRAME_MS:.1f} ms/frame measured)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> **This script fails closed on a missing measurement and asserts nothing else.** An earlier
> draft additionally refuted the cap against a `draw_calls(cap) > 4096` ceiling; that 4096
> was invented here and appears in no spec. **Dropped** (user ruling, 2026-08-26) — a
> measured cap validated against a guessed budget inverts the point. The script's whole job
> is to stop phase 4 proceeding on an unmeasured number, and to keep the measurement beside
> the number it justifies.
>
> This is the **only** surviving logic-validation script of the three originally proposed,
> kept because its number lands in shipped code as a behavioural cap.

- [ ] **Step 2: Run it and confirm it fails on the missing cap**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/colony_view_budget.py
```
Expected: prints the table, then `NO MEASUREMENT`, exit **1**. The script failing at this
point is the intended state — it is what stops phase 4 proceeding on an invented number.

- [ ] **Step 3: Measure, then fill in both constants**

Render a prototype grid at 64, 256, 1024 and 1536 cells and record observed frame time at
each. Choose the cap at the largest count that holds an interactive frame budget, write it
into `RECORDED_CAP` **and its measured frame time into `RECORDED_FRAME_MS`**, and record the
full table in `spike/README.md`.

The interactive budget is a judgement, not a spec number — spec §9.1 makes interactivity a
**target**, not a gate. Record what you chose and why; a later reader can re-judge the cap
from `RECORDED_FRAME_MS` without re-running the prototype.

- [ ] **Step 4: Re-run and commit**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/colony_view_budget.py
git add docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/ \
        docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/
git commit -m "docs(viv): measure the colony-view virtualization cap"
```

---

### Task 4.2: Shared camera as a value, not a sync protocol

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_assets/viv_viewer.js` (add `setGridViews`)
- Test: `tests/e2e/gui/test_colony_shared_camera.py` (create)

**Interfaces:**
- Consumes: `window.phenotypicViv` from phase 2.
- Produces: `setGridViews(containerId, cells, sharedViewState)` where `cells` is
  `[{id, centroidRr, centroidCc, size}, ...]`.

> **The napari implementation this ports.** `gui/_smart_grid/` patches `viewer.grid` so
> only **visible** layers get cells, then `create_overlay_clones` duplicates every
> Labels/Points/Shapes visual into **each** viewbox — every cell shows a different base
> image under the same annotation, sharing one camera.
>
> **The deck.gl translation:** zoom edits **one shared `viewState`** applied to every view.
> The shared camera is a **value, not a sync protocol** — there is no per-view listener
> reconciling positions, which is what makes it correct by construction rather than
> eventually consistent. `create_overlay_clones`' GPU-resource cleanup dance
> (`cleanup_clones`) has no deck.gl equivalent and is **not ported**.

- [ ] **Step 1: Read the napari original before porting**

```bash
uv run grep -rn "create_overlay_clones\|cleanup_clones\|viewer.grid" src/phenotypic/gui/_smart_grid/
```
Per the **`porting-a-reference-algorithm`** skill: cite `file:line` for each claim about
what it does, and diff line-by-line rather than inspecting and summarising. Record any
deviation in a drift-register row, however small.

- [ ] **Step 2: Write the failing e2e test**

```python
"""One shared viewState drives every colony view.

The assertion is that all views report the SAME zoom after one is changed --
not that they converge. A sync protocol would pass a convergence test and
still show tearing mid-gesture; a shared value cannot.
"""

import pytest


@pytest.mark.e2e
def test_zooming_one_cell_moves_every_cell(page, live_viewer_url):
    page.goto(live_viewer_url)
    page.wait_for_function("() => window.phenotypicViv !== undefined")
    page.click("[data-testid='tab-colony']")
    page.wait_for_selector(".colony-grid-view")

    page.evaluate(
        """() => window.phenotypicViv.setViewState(
               'colony-grid', {zoom: 3, target: [0, 0, 0]})"""
    )
    zooms = page.evaluate(
        """() => window.phenotypicViv.__debugViewStates('colony-grid')
                   .map(v => v.zoom)"""
    )
    assert len(zooms) > 1
    assert len(set(zooms)) == 1, f"views drifted apart: {sorted(set(zooms))}"
```

- [ ] **Step 3: Implement `setGridViews` with a single `viewState`**

One `OrthographicView` per cell, each with its own `x`/`y`/`width`/`height` and a
`target` at the colony centroid, but **one** `viewState` object shared by all of them.

- [ ] **Step 4: Make the shared-camera lock a visible affordance**

Spec §6.2: "The 'Shared camera' lock is a visible affordance, not hidden behaviour, so the
eventual unlock-one-cell mode has somewhere to live." Add the toggle to the Colony chrome
now, even though it only has one state today — retrofitting an affordance for a mode that
already shipped as invisible behaviour is the expensive order.

- [ ] **Step 5: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui/test_colony_shared_camera.py -v
git add src/phenotypic/gui/results_viewer/_assets/viv_viewer.js \
        tests/e2e/gui/test_colony_shared_camera.py
git commit -m "feat(gui): render colony cells as deck.gl views on one shared viewState"
```

---

### Task 4.3: Enforce the cap, and keep curation working

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/colony_view/_grid.py` — **rendering only**
- Test: `tests/unit/gui/results_viewer/test_colony_view_cap.py` (create)

> **The constraint that governs this task.** `colony_view/` is where the curation radial
> lives: `_grid.py:47, :462` build `build_radial_trigger` on every tile and
> `colony_view/_callbacks.py:43` builds the popover body. Curation is **retained** — the
> radial's six wedges are the real `ERROR_CATEGORY_COLORS` map (`oversegmented`,
> `undersegmented`, `merged`, `background_noise`, `debris`, `other`, each in its fixed
> Okabe-Ito slot), with the restore centre node and the custom-category strip, matching
> `_shared/_radial.py`'s anatomy. Bulk-mark still writes
> `deliverables/errors/<category>.parquet`.
>
> **`tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` must pass
> unmodified.** If it needs editing, this task has overreached from rendering into
> curation — stop and escalate.

- [ ] **Step 1: Write the cap test**

```python
"""Above the cap, cells virtualize rather than all mounting.

Spec section 6.2 records the cap as measured during D1, so this test reads it
from the single source rather than restating a literal -- a cap that appears
in two places drifts.
"""

from phenotypic.gui.results_viewer.colony_view._grid import (
    COLONY_VIEW_CELL_CAP,
    plan_visible_cells,
)


def test_cells_beyond_the_cap_are_not_mounted():
    cells = [{"id": i} for i in range(COLONY_VIEW_CELL_CAP * 2)]
    visible = plan_visible_cells(cells, focus_index=0)
    assert len(visible) <= COLONY_VIEW_CELL_CAP


def test_the_focused_cell_is_always_visible():
    cells = [{"id": i} for i in range(COLONY_VIEW_CELL_CAP * 2)]
    focus = COLONY_VIEW_CELL_CAP + 5
    visible = plan_visible_cells(cells, focus_index=focus)
    assert any(c["id"] == focus for c in visible)
```

- [ ] **Step 2: Implement, taking `COLONY_VIEW_CELL_CAP` from task 4.1's measurement**

Add both to `_grid.py`. Keep `build_radial_trigger` on every **mounted** cell — a
virtualized-out cell has no radial because it has no tile, which is correct; a mounted cell
missing its radial is a curation regression.

- [ ] **Step 3: Prove curation is untouched**

```bash
git diff --stat src/phenotypic/gui/results_viewer/colony_view/_callbacks.py \
                 tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py
```
Expected: **empty**.

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py \
  tests/unit/gui/results_viewer/test_colony_view_cap.py -v
```
Expected: PASS, PASS.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/colony_view/_grid.py \
                        tests/unit/gui/results_viewer/test_colony_view_cap.py
git add src/phenotypic/gui/results_viewer/colony_view/_grid.py \
        tests/unit/gui/results_viewer/test_colony_view_cap.py
git commit -m "feat(gui): cap colony views at the measured budget, curation retained"
```
