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

import math
import sys

#: Layers rendered into EACH view. NOT a constant to assert: Plate section
#: 6.1's Layers panel exposes rgb, gray, detect_mat and objmap, all
#: independently toggleable, so the worst case is 4. Take it from the stack
#: the caller actually builds.
VISIBLE_LAYERS = 4
#: A common plate: 32 x 48 = 1536 colonies (backend section 2.3's example).
PLATE_CELLS = 1536
#: The reference plate, and the chunk grid it implies.
PLATE_HW = (4000, 3000)
CHUNK_PX = 1024
#: Measured cap, filled in from the prototype in step 3. None until measured.
RECORDED_CAP: int | None = None
#: Frame time, in ms, observed at RECORDED_CAP. Recorded beside the number so
#: the cap can be re-judged later without re-running the prototype blind.
RECORDED_FRAME_MS: float | None = None


def tiles_touched(centroids, level, cell_src_px, chunk_px=CHUNK_PX) -> set:
    """UNION of chunk indices the cells touch.

    The load-bearing correction. An earlier draft modelled per-cell 64x64
    crops as if each cell owned private pixels -- that is D3, the
    overlay-PNG-slicing world. In D1 every cell is a VIEWPORT ONTO THE SAME
    STORE, so tiles are SHARED between cells and the resident set is the
    union, bounded above by the whole level. That makes the bound
    closed-form and small rather than linear in cell count.
    """
    touched = set()
    half = cell_src_px / 2
    for rr, cc in centroids:
        for ty in range(int((rr - half) / 2**level) // chunk_px,
                        int((rr + half) / 2**level) // chunk_px + 1):
            for tx in range(int((cc - half) / 2**level) // chunk_px,
                            int((cc + half) / 2**level) // chunk_px + 1):
                touched.add((ty, tx))
    return touched


def resident_bytes(n_tiles, n_series, chunk_px=CHUNK_PX, channels=3, itemsize=1):
    """Bytes the tile cache holds for one store."""
    return n_tiles * chunk_px**2 * channels * itemsize * n_series


def level_tiles(h, w, level, chunk_px=CHUNK_PX) -> int:
    """Every chunk at a level -- the ceiling the union cannot exceed."""
    return (math.ceil(h / 2**level / chunk_px)
            * math.ceil(w / 2**level / chunk_px))


def main() -> int:
    h, w = PLATE_HW
    ceiling = level_tiles(h, w, 0)
    print(
        f"level-0 ceiling: {ceiling} tiles = "
        f"{resident_bytes(ceiling, 1) / 1e6:.0f} MB for the WHOLE level"
    )
    for cells in (64, 256, 1024, PLATE_CELLS):
        print(f"{cells:5d} cells: {cells * VISIBLE_LAYERS:6d} draw calls/frame")
    if RECORDED_CAP is None or RECORDED_FRAME_MS is None:
        print("NO MEASUREMENT: run the prototype in task 4.1 step 3")
        return 1
    print(
        f"cap {RECORDED_CAP} cells, {RECORDED_FRAME_MS:.1f} ms/frame measured; "
        f"set maxCacheByteSize >= {resident_bytes(ceiling, VISIBLE_LAYERS) / 1e6:.0f} MB "
        f"per distinct store in the grid"
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

### Task 4.2: One dynamic viewState, per-view target overrides

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_assets/viv_viewer.js` (add `setGridViews`)
- Test: `tests/e2e/gui/test_colony_shared_camera.py` (create)

**Interfaces:**
- Consumes: `window.phenotypicViv` from phase 2.
- Produces: `setGridViews(containerId, cells, sharedViewState)` where `cells` is
  `[{id, centroidRr, centroidCc, size}, ...]`.

> **The napari implementation this ports** (`gui/_smart_grid/`, 378 lines, read in full
> during plan refinement — every claim below carries its `file:line`):
>
> - **Visible-layers-only grid.** `install_smart_grid` (`_install.py:14-15`) shadows exactly
>   two methods on the grid *instance* (`:90-91`). The real predicate (`:64-70`) is
>   `layer.visible and not (_overlay_enabled and is_overlay_layer(layer))` — visible **AND
>   non-overlay**. The overlay exclusion is not a detail: it is what frees the
>   Labels/Points/Shapes layers from owning a cell so they can be cloned into every cell.
>   The two are **one coupled mechanism**, not a sequence. `_overlay_enabled` and
>   `_labels_enabled` are live user toggles (`_grid_popup.py:40-51`), so the reference has
>   two runtime modes.
>   **Not ported, declared:** the colony grid has no visibility mapping at all — napari's
>   cells are per-*layer*, colony cells are per-*region of one image*, so the concept has no
>   analogue. The two user toggles go with it.
> - **Detach + draw-order.** `_install.py:130-136` sets the original overlay visuals'
>   `node.parent = None`, because `patched_position` returns `(-1,-1)` and napari never
>   re-parents them — left attached they draw canvas-wide at `order=100`, on top of
>   everything. Clones then take `order = len(viewer.layers) + 10`
>   (`_overlay_visuals.py:33, :39`). The deck.gl analogue is layer order within each view's
>   stack; state it explicitly when building the stack rather than relying on insertion
>   order.
> - **Overlay cloning.** `create_overlay_clones` (`_overlay_visuals.py:20-42`) iterates
>   `canvas.grid_views × overlay_layers`, calls `create_vispy_layer(layer)` and parents each
>   clone to `viewbox.scene`. `is_overlay_layer` (`:16-18`) is exactly
>   `Labels | Points | Shapes`. Note it clones only **visible** overlay layers (`:26-28`) —
>   "every Labels/Points/Shapes visual" overstates it.
>
> **Correction — napari IS a sync protocol, and the deck.gl design is a declared
> deviation, not a faithful port.** An earlier draft of this plan (and spec §6.2's "sharing
> one camera") claimed the reference shares a camera *value* with "no per-view listener
> reconciling positions". **That is false**, verified in napari's source:
>
> ```text
> napari/_vispy/canvas.py:1121-1123   camera = VispyCamera(view, self.viewer.camera, self.viewer.dims)
>                                     self.grid_views.append(view); self.grid_cameras.append(camera)
> napari/_vispy/camera.py:50-56       self._camera.events.center.connect(self._on_center_change)
>                                     ... zoom / angles / perspective
> napari/_vispy/canvas.py:646-648     # sync all cameras
>                                     for camera in (self.camera, *self.grid_cameras):
>                                         camera.on_draw(event)
> ```
>
> One camera **model**, *N* `VispyCamera` objects, event-connected and re-reconciled on
> every draw — precisely the per-view listener protocol the draft said did not exist.
> (`_smart_grid/` itself contains no camera code; it patches grid geometry only. The
> behaviour is napari's, inherited.)
>
> **All four deviations from the reference ride ONE user gate**, not just the camera:
> the visibility mapping is not ported, the two `_grid_popup.py` toggles go with it,
> `cleanup_clones` is not ported, and the camera fan-out collapses to a single value. The
> `porting-a-reference-algorithm` skill treats the drift register as a whole.
>
> **The deck.gl translation collapses that fan-out to a single `viewState`.** It is still
> the better design — a shared value cannot tear mid-gesture the way an event-reconciled
> set can — but it is a **deliberate simplification of the reference**, and per the
> `porting-a-reference-algorithm` skill it is recorded as a **declared deviation** requiring
> a user gate, not presented as fidelity. Do not argue "correct by construction" against the
> strawman the draft invented.
>
> **Why `cleanup_clones` is not ported — narrow claim only.** It
> (`_overlay_visuals.py:45-58`) calls `clone.close()` per clone then
> `canvas._scene_canvas.context.finish()` (a glFinish, so deletions land before
> reallocation), reclaiming GPU resources for *N overlay-layers × M viewboxes* separately
> instantiated vispy visuals. It is **not** a nicety in the reference: it runs at the top of
> every scenegraph rebuild (`_install.py:125`) against wholesale recreation at `:137`, and
> rebuilds fire on every visibility toggle, name change and layer insert/remove
> (`:157-183`). Without it those visuals leak per event.
>
> The narrow claim holds: deck.gl reconciles layers by id and finalizes them when dropped,
> and multi-view draws **one** layer instance per view rather than N instances, so there is
> no clone lifecycle to manage. **The broad claim does not.** The leak *class* survives —
> Viv's `TileLayer` texture cache grows with the union of all *N* views' visible tiles,
> bounded only by `maxCacheSize` / `maxCacheByteSize`, and **nothing in this phase sets,
> bounds, or tests it.** Task 4.1's cap bounds *cell count*, not *cached tile bytes*.
>
> **But the bound is far tighter than "leak" suggests, and task 4.1 now derives it.** Cells
> are viewports onto ONE store, so tiles are shared: the resident set is the *union*, and
> the union cannot exceed the level. A 4000×3000 plate at level 0 is
> `ceil(4000/1024) × ceil(3000/1024) = 4 × 3 = 12` chunks ≈ **36 MB for the entire level**,
> coarser levels less. So set `maxCacheByteSize` from
> `level_tiles × per_tile_bytes × visible_layers` — derived, not guessed.
>
> **The parameter that actually drives it is the number of distinct STORES in the grid, not
> the cell count.** `colony_view/_grid.py` builds tiles carrying their own
> `Metadata_Dataset` / `Metadata_ImageFile` (`:284, :347, :398`), so a grid can span images:
> 1536 cells over one plate is ~36 MB; 1536 cells over 1536 *different* images is 1536× that.
> Cell count is only a proxy for the quantity worth capping.

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
    states = page.evaluate(
        """() => window.phenotypicViv.__debugViewStates('colony-grid')"""
    )
    # page.evaluate marshals JS objects into Python DICTS -- attribute
    # access raises AttributeError and the test ERRORS rather than failing
    # informatively, which is the worst outcome for a test whose job is to
    # fail informatively.
    zooms = [s["zoom"] for s in states]
    targets = [tuple(s["target"][:2]) for s in states]

    assert len(zooms) > 1
    assert len(set(zooms)) == 1, f"zoom drifted apart: {sorted(set(zooms))}"

    # The complementary half, and the one that matters. Asserting only
    # "all zooms equal" is satisfied PERFECTLY by the bug this test exists
    # to catch: a single shared viewState gives every cell the same target,
    # so the grid renders one colony N times -- with identical zooms.
    assert len(set(targets)) == len(targets), (
        f"every cell is showing the same region: {targets[:4]}"
    )
```

- [ ] **Step 3: Implement `setGridViews` — `target` is per-view, `zoom` is shared**

> **Read this before writing the call.** An earlier draft said "one `OrthographicView` per
> cell, each with its own `x`/`y`/`width`/`height` and a `target` at the colony centroid,
> but **one** `viewState` object shared by all of them." **`target` does not live on a
> `View`.** A deck.gl `View` carries `id`/`x`/`y`/`width`/`height`; `target` and `zoom` live
> in the **viewState**. Built literally as written, every cell inherits the same `target`
> and the grid renders **the same colony N times**.

There are **two** deck.gl forms that give per-view targets, and they are not equivalent —
one of them is the very sync protocol this design exists to avoid.

**Rejected — the keyed viewState map.** The developer guide (`docs/developer-guide/views.md`)
says: *"To specify the view state for a particular view, you can add a key to the
`viewState` object. This key should correspond to the `id` of the view."* That works, but
`onViewStateChange` then fires with `{viewId, viewState}` for the **one** view the gesture
touched, and keeping `zoom` common requires the handler to fan the new zoom back across
every entry — per-view reconciliation, reintroduced one layer down. The first real user
gesture drifts the zooms apart, and step 2's test cannot catch it because it drives
`setViewState(...)` programmatically rather than gesturing.

**Chosen — `View.viewState` merge.** The API reference (`docs/api-reference/core/view.md`)
records that a `View`'s `viewState` property, when it carries an `id`, *"merges with a
dynamic view state… useful for sharing view states… or overriding parts of a dynamic
state"*. So there is **one** dynamic viewState holding `zoom`, and each `View` overrides
only its own `target`:

```javascript
// ONE dynamic viewState. There is only one zoom, so it cannot drift --
// a value, not a protocol. This is the form that matches the principle.
const viewState = {"colony-shared": {zoom: shared.zoom}};

// Each View overrides ONLY its target, merging over the shared zoom.
const views = cells.map((c, i) => new OrthographicView({
  id: `cell-${c.id}`,
  x: layout[i].x, y: layout[i].y,
  width: layout[i].w, height: layout[i].h,
  viewState: {id: "colony-shared", target: [c.centroidCc, c.centroidRr, 0]},
}));
```

**State the controller decision explicitly.** If the views carry no `controller`, zoom is
programmatic-only — and then step 4's "Shared camera" lock has nothing to constrain, which
must be said rather than left implied. If they do carry one, `setViewState(containerId, …)`
updates the single `colony-shared` entry, and that is what the façade method means in grid
mode.

A "shared viewState object" and "a shared zoom over per-cell target overrides" are
different things, and only the second renders a grid **and** stays shared under gesture.

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
