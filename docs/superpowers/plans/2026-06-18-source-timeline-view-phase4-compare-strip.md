# Source Timeline View — Phase 4: Synced Compare Strip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, viewport-**synced "Compare" strip** to **both** timeline surfaces (Browse + Results) — a deep-zoom side-by-side view of a small selected set of cells, where pan/zoom in any viewer propagates to all the others. Two triggers (spec §7, D5): (a) clicking an **axis row-header** compares that row's full time-course; (b) **multi-selecting cells** (shift/ctrl-click on tiles) compares an arbitrary set. Hard-capped at `TIMELINE_COMPARE_CAP` (≈12) live OpenSeadragon viewers under the ~16-WebGL-context browser ceiling, with a visible **"Showing first 12 of N — narrow the selection"** notice on over-selection (never a silent truncation).

**Architecture:** A **surface-agnostic shared controller** in the timeline shared JS assets — `openCompareStrip(refs, { dziUrlBuilder, titleFor })` — builds a modal, mounts ≤cap OSD viewers (each opening `dziUrlBuilder(ref)`), binds a **shared viewport** with a feedback-loop guard (mirroring the proven `results_viewer.js` lock-views pattern), renders the over-cap notice, and tears down **all** viewers on close (freeing WebGL contexts). The **selection mechanism** is added to the Phase 2 focus-navigate controller (`timeline.js`, `window.__phenotypicTimeline`): a selected-set, shift/ctrl-click toggling on tiles, a **"Compare selected"** button, and **row-header → select-whole-row + open**. Each surface supplies its own `dziUrlBuilder` (Browse: `/tiles/<token>.dzi` from the cell's `data-ref`; Results: `/tiles/<dataset>/<stem>.dzi` from the cell's `data-ref`). The shared controller is **vendored-OSD only** (no CDN) — each surface already vendors OpenSeadragon and `timeline.js` runs under the surface's own asset folder.

**Tech Stack:** Vanilla JS (controller + selection), Dash + dash-bootstrap-components (`dbc.Modal`), OpenSeadragon (vendored, already present on both surfaces), pytest (pure cap-logic unit) + Playwright (`tests/e2e/gui`).

## Global Constraints

- **`uv` is the sole runner.** Every command is `uv run …`; never bare `python`/`pip`.
- **Phases 1–2 must be merged/available.** This plan consumes the Phase 1 shared engine (`build_timeline_grid` emits `data-ref`, `data-row`, `data-col`, `data-row-index`, `data-col-index`, `data-key="row::time"` on every cell; axis labels carry `timeline-axis-label timeline-axis-label--y` / `--x`), the Phase 1 `_config` constants, and the Phase 2 focus-navigate controller `window.__phenotypicTimeline` with `ns.attach(containerId)` plus the `BROWSE_TL_*` ids and `build_timeline_body()` layout. **Phase 3 (Results) need not be merged** to land the Browse wiring + the shared controller + the constant; the Results wiring is described as "mirror the Browse trigger wiring" against TODO-from-Phase-3 ids.
- **Hard cap with a visible notice (spec §7, §12):** never silently truncate. `> TIMELINE_COMPARE_CAP` selected → mount the first `cap` (by grid order) **and** render `"Showing first {cap} of {N} — narrow the selection"`. The cap-vs-notice decision is a **pure function** so it is unit-testable without a browser.
- **Shared viewport with a feedback guard (spec §7):** pan/zoom in any viewer propagates to the rest, guarded by a single re-entrancy flag while applying a propagated viewport (mirror `results_viewer.js`'s `_broadcasting` guard — verified at `results_viewer/_assets/results_viewer.js:165-181`). Use the **exact OSD API the repo already uses**: `viewer.viewport.getCenter(true)` / `getZoom(true)` to read, `viewer.viewport.zoomTo(zoom, null, true)` / `panTo(center, true)` to write, `viewer.addHandler("animation", handler)` to subscribe.
- **Tear down every viewer on close (spec §7, §12 WebGL ceiling):** call `viewer.destroy()` on each mounted OSD viewer when the modal closes, so WebGL contexts are released and a later open starts clean.
- **§15.10 accepted v1 DZI spike:** opening on ≤cap distinct cells fires ≤cap concurrent DZI pyramid builds (the `_dzi_tiler._get_lock` per-image lock only serialises *duplicate* requests, not distinct ones). This is an **accepted, documented** v1 CPU spike — subsequent opens are cached + instant. The optional fast-follow (warm selected cells' DZI on selection, before the strip opens) is **noted but NOT implemented** in this phase.
- **Per-container selection state (cross-surface singleton caveat, spec §15.9):** `window.__phenotypicTimeline` is currently a single namespace object. Browse and Results live on **separate Flask servers / separate page loads**, so a single page never hosts both surfaces at once — but the selection-set and triggers must still be keyed by the **triggering container id** so the controller reads the right surface's selection (Phase 3's job is to ensure the Results container is distinct). The compare controller **accepts the triggering container element** and reads its selection; it does not assume a global.
- **Single-source constants** in `_config.py` (`TIMELINE_COMPARE_CAP`); new Browse ids in `browse/_ids.py`. Don't re-spell literals; don't import `dash` from `_config.py`.
- **FEATURES.md gate:** any `src/phenotypic/gui/` change must modify `FEATURES.md`; `✅ shipping` rows need a resolvable `path::test`; never leave a row `🚧 in progress` (merge gate rejects it).
- **Verify Dash/JS wiring in a live browser (project rule):** the selection + compare-strip flow is JS-heavy and only trustworthy when driven live — it carries Playwright e2e tests (on the Browse surface concretely; the Results e2e mirrors once Phase 3 lands), per spec §16.9.

---

### Task 1: `TIMELINE_COMPARE_CAP` constant

**Files:**
- Modify: `src/phenotypic/gui/_config.py` (append to the "Timeline view" block + extend `__all__`)
- Test: `tests/gui/_shared/timeline/test_constants.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `TIMELINE_COMPARE_CAP: int` (≈12) — the hard cap on live OSD viewers in the Compare strip (spec §7, §9 "Compare-strip cap"; under the ~16-WebGL-context ceiling, §12).

> **Verified (do this first):** `TIMELINE_COMPARE_CAP` is **not** present in `_config.py` today and is **not** added by the Phase 1 plan (Phase 1 adds `THUMB_SIZE_BUCKETS`, `TIMELINE_TILE_SIZE_*`, `TIMELINE_GRID_GAP_PX`, `TIMELINE_FOCUS_MARGIN`, `TIMELINE_MOUNT_CAP`, `TIMELINE_WARM_CONCURRENCY`, `BROWSE_THUMB_URL_SEGMENT`, `VIEWER_THUMB_URL_SEGMENT`, `snap_thumb_bucket` — but **no** compare cap). Confirm with `grep -n "TIMELINE_COMPARE_CAP" src/phenotypic/gui/_config.py` (expect no hits) before adding. If a concurrent Phase-1 edit has since added it, skip the implementation and keep only the test.

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/_shared/timeline/test_constants.py`:

```python
def test_compare_cap_is_a_small_positive_int_under_webgl_ceiling() -> None:
    # Hard cap on live OSD viewers in the synced Compare strip (spec §7/§9);
    # must stay below the ~16-WebGL-context browser ceiling (spec §12).
    from phenotypic.gui._config import TIMELINE_COMPARE_CAP

    assert isinstance(TIMELINE_COMPARE_CAP, int)
    assert 1 <= TIMELINE_COMPARE_CAP <= 16
    assert TIMELINE_COMPARE_CAP == 12  # spec §7: "~12 live viewers"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_constants.py::test_compare_cap_is_a_small_positive_int_under_webgl_ceiling -v`
Expected: FAIL with `ImportError: cannot import name 'TIMELINE_COMPARE_CAP'`.

- [ ] **Step 3: Write minimal implementation**

Append to the "Timeline view" block in `src/phenotypic/gui/_config.py` (after `TIMELINE_WARM_CONCURRENCY`):

```python
#: Hard cap on live OpenSeadragon viewers in the synced Compare strip
#: (spec §7/§9). Each OSD viewer holds its own canvas/WebGL context and
#: browsers cap live WebGL contexts (~16 in Chrome, §12); 12 leaves headroom.
#: An over-cap selection renders the first TIMELINE_COMPARE_CAP viewers PLUS a
#: visible "Showing first N of M" notice — never a silent truncation.
TIMELINE_COMPARE_CAP: int = 12
```

Then add `"TIMELINE_COMPARE_CAP"` to the module's `__all__` (in the Timeline-view block alongside the other `TIMELINE_*` names).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_constants.py -v`
Expected: PASS (all prior constant tests + the new compare-cap test).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_config.py tests/gui/_shared/timeline/test_constants.py
git commit -m "feat(gui-timeline): TIMELINE_COMPARE_CAP constant for the synced Compare strip"
```

---

### Task 2: Pure cap/over-selection helper (`compare_selection_plan`)

**Files:**
- Modify: `src/phenotypic/gui/_shared/timeline/_grid.py` (append a pure helper) — OR create a new `src/phenotypic/gui/_shared/timeline/_compare.py` if you prefer a dedicated module; **pick one** and export it from `__init__` in Task 6.
- Test: `tests/gui/_shared/timeline/test_compare.py`

**Interfaces:**
- Consumes: `TIMELINE_COMPARE_CAP` (Task 1).
- Produces: `compare_selection_plan(refs: Sequence[object], *, cap: int = TIMELINE_COMPARE_CAP) -> ComparePlan` where `ComparePlan(shown: tuple[object, ...], total: int, over_cap: bool, notice: str | None)`. When `len(refs) <= cap` → `shown == tuple(refs)`, `over_cap is False`, `notice is None`. When `len(refs) > cap` → `shown == tuple(refs[:cap])`, `over_cap is True`, `notice == "Showing first {cap} of {total} — narrow the selection"`. This is the **single source of truth** for the cap/notice text; the JS controller renders the same string, and the unit test pins it so the JS and Python copies can't drift (the test docstring names the JS mirror site).

> **Why a pure Python helper for JS-rendered text?** The notice string is the load-bearing user-facing contract (spec §7 verbatim). Pinning it in a Python unit makes the cap logic testable without a browser and gives the JS controller a canonical wording to copy. The JS does the actual DOM render (it owns the viewers); this helper is the spec-text guard + the documented mirror.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/_shared/timeline/test_compare.py`:

```python
"""Pure cap/over-selection planning for the synced Compare strip (spec §7)."""
from __future__ import annotations

from phenotypic.gui._config import TIMELINE_COMPARE_CAP
from phenotypic.gui._shared.timeline import compare_selection_plan


def test_under_cap_shows_all_no_notice() -> None:
    plan = compare_selection_plan(["a", "b", "c"], cap=12)
    assert plan.shown == ("a", "b", "c")
    assert plan.total == 3
    assert plan.over_cap is False
    assert plan.notice is None


def test_exactly_cap_shows_all_no_notice() -> None:
    refs = [str(i) for i in range(12)]
    plan = compare_selection_plan(refs, cap=12)
    assert plan.shown == tuple(refs)
    assert plan.over_cap is False
    assert plan.notice is None


def test_over_cap_truncates_to_cap_and_emits_notice() -> None:
    refs = [str(i) for i in range(20)]
    plan = compare_selection_plan(refs, cap=12)
    assert plan.shown == tuple(refs[:12])  # first cap, by selection order
    assert plan.total == 20
    assert plan.over_cap is True
    # EXACT spec §7 wording. The JS controller in browse/_assets/timeline.js
    # (renderOverCapNotice) MUST render this identical string — keep coupled.
    assert plan.notice == "Showing first 12 of 20 — narrow the selection"


def test_default_cap_is_the_config_constant() -> None:
    refs = [str(i) for i in range(TIMELINE_COMPARE_CAP + 1)]
    plan = compare_selection_plan(refs)  # no cap kwarg → uses TIMELINE_COMPARE_CAP
    assert len(plan.shown) == TIMELINE_COMPARE_CAP
    assert plan.over_cap is True


def test_empty_selection_is_a_clean_empty_plan() -> None:
    plan = compare_selection_plan([])
    assert plan.shown == ()
    assert plan.total == 0
    assert plan.over_cap is False
    assert plan.notice is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_compare.py -v`
Expected: FAIL with `ImportError: cannot import name 'compare_selection_plan'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/phenotypic/gui/_shared/timeline/_grid.py` (or the new `_compare.py`):

```python
from dataclasses import dataclass
from collections.abc import Sequence

from phenotypic.gui._config import TIMELINE_COMPARE_CAP


@dataclass(frozen=True)
class ComparePlan:
    """The bounded plan for opening a synced Compare strip (spec §7).

    Attributes:
        shown: The refs that will be mounted (≤ ``cap``, in selection order).
        total: The full selection size (``len(refs)``).
        over_cap: ``True`` when ``total`` exceeded ``cap`` and ``shown`` was
            truncated to the first ``cap``.
        notice: The verbatim over-cap notice to display, or ``None``.
    """

    shown: tuple[object, ...]
    total: int
    over_cap: bool
    notice: str | None


def compare_selection_plan(
    refs: Sequence[object], *, cap: int = TIMELINE_COMPARE_CAP
) -> ComparePlan:
    """Bound a Compare-strip selection to ``cap`` viewers, never truncate silently.

    The synced Compare strip mounts one OSD viewer per ref, each holding a
    WebGL context; the browser ceiling (~16) forces a hard cap (spec §7/§12).
    When the selection exceeds ``cap`` this returns the first ``cap`` refs (by
    selection order) AND a visible notice so the user knows the rest were
    held back — never a silent drop.

    Args:
        refs: The selected cell refs, in selection order.
        cap: Maximum live viewers (defaults to :data:`TIMELINE_COMPARE_CAP`).

    Returns:
        A :class:`ComparePlan`. ``notice`` is non-``None`` only when over cap.
    """
    total = len(refs)
    if total <= cap:
        return ComparePlan(shown=tuple(refs), total=total, over_cap=False, notice=None)
    return ComparePlan(
        shown=tuple(refs[:cap]),
        total=total,
        over_cap=True,
        notice=f"Showing first {cap} of {total} — narrow the selection",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_compare.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/timeline/_grid.py tests/gui/_shared/timeline/test_compare.py
git commit -m "feat(gui-timeline): compare_selection_plan (cap + over-cap notice, pure)"
```

---

### Task 3: Export `compare_selection_plan`/`ComparePlan` from the engine package

**Files:**
- Modify: `src/phenotypic/gui/_shared/timeline/__init__.py`
- Test: `tests/gui/_shared/timeline/test_public_api.py` (append)

**Interfaces:**
- Consumes: Task 2 symbols.
- Produces: `from phenotypic.gui._shared.timeline import compare_selection_plan, ComparePlan`.

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/_shared/timeline/test_public_api.py`:

```python
def test_compare_helpers_are_exported() -> None:
    import phenotypic.gui._shared.timeline as timeline

    for name in ("compare_selection_plan", "ComparePlan"):
        assert name in timeline.__all__
        assert hasattr(timeline, name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_public_api.py::test_compare_helpers_are_exported -v`
Expected: FAIL with `AssertionError` / `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

In `src/phenotypic/gui/_shared/timeline/__init__.py`, import `compare_selection_plan` + `ComparePlan` from wherever Task 2 placed them (`._grid` or `._compare`) and add both to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_public_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/timeline/__init__.py tests/gui/_shared/timeline/test_public_api.py
git commit -m "feat(gui-timeline): export compare_selection_plan + ComparePlan"
```

---

### Task 4: Shared compare-strip controller (`openCompareStrip`) in the timeline JS

**Files:**
- Modify: `src/phenotypic/gui/browse/_assets/timeline.js` (the Phase 2 `window.__phenotypicTimeline` controller — add the compare-strip section)
- Modify: `src/phenotypic/gui/browse/_assets/browse.css` (compare-modal + selection-highlight styles, using `_design` CSS variables — never raw hex)
- Test: covered live by the Browse e2e in Task 7 (no standalone unit harness for OSD — OSD needs a real canvas/WebGL context; the pure cap logic is already unit-tested in Task 2).

**Interfaces:**
- Consumes: vendored OpenSeadragon (loaded the same way `browse.js`/`results_viewer.js` load it — `window.OpenSeadragon`, `prefixUrl: appPrefix + "assets/openseadragon/images/"`); `window.__phenotypicAppPrefix`.
- Produces: `window.__phenotypicTimeline.openCompareStrip(refs, opts)` where `opts = { dziUrlBuilder: (ref) => url, titleFor: (ref) => string, cap: number }`. It (1) bounds `refs` to `cap` (the cap is **supplied by the caller** — Task 6 reads it off the triggering container's `data-compare-cap`; the controller trusts `opts.cap` and does not re-fabricate a numeric fallback) with an over-cap notice, (2) builds the modal DOM (a flex strip of ≤cap viewer cells), (3) mounts one OSD viewer per shown ref via `dziUrlBuilder(ref)`, (4) binds the shared viewport with a feedback guard, (5) renders the over-cap notice when applicable, and (6) tears down **all** viewers on close.

> **Controller state is a process-global SINGLETON (honesty note — M3):** `_compareViewers`, `_broadcasting`, and the modal are deliberately **module-global** — at most **one** Compare strip is open at a time, on **one** surface. Browse and Results live on separate Flask servers / page loads and never co-mount, so a single global strip is correct, not a limitation. The thing that is **per-container** is the *selection set* (Task 6, now the `.timeline-cell--selected` DOM class), so the triggers read the right surface's selection. Do **not** generalize the controller state to per-container — that would over-engineer a singleton that is intentionally global.

> **VERIFIED OSD facts (mirror them exactly):**
> - **Mount options** — both surfaces construct `window.OpenSeadragon({ element, prefixUrl: appPrefix + "assets/openseadragon/images/", tileSources: dziUrl, showNavigator: false, showRotationControl: false, constrainDuringPan: true, visibilityRatio: 0.5, immediateRender: false, … })` (verified `browse.js:90-102`, `results_viewer.js:124-138`). Reuse this shape.
> - **Viewport-sync API** — read with `viewer.viewport.getCenter(true)` + `getZoom(true)`; write with `peer.viewport.zoomTo(zoom, null, true)` + `peer.viewport.panTo(center, true)`; subscribe with `viewer.addHandler("animation", handler)` (verified `results_viewer.js:167-190`). The `_broadcasting` re-entrancy guard there is the canonical feedback-loop guard — copy it.
> - **Teardown** — `viewer.destroy()` (verified `browse.js:81,87`, `results_viewer.js:120,151`).
> - **OSD load** — `ns.osdReady` promise that injects `appPrefix + "assets/openseadragon/images/"`'s sibling `openseadragon.min.js` if `window.OpenSeadragon` is absent (verified `browse.js:63-74`). Reuse the surface's existing `ns.osdReady`; do NOT add a CDN load.

- [ ] **Step 1: (No standalone unit test — see Interfaces)**

The controller's pure decision (cap/notice) is already pinned by Task 2's unit test; OSD mounting/teardown/sync needs a live canvas and is exercised by the Task 7 Playwright e2e. Write the implementation, then prove it under Task 7.

- [ ] **Step 2: Write the implementation**

Add a new IIFE-scoped section to `src/phenotypic/gui/browse/_assets/timeline.js` (it shares the `window.__phenotypicTimeline` namespace `ns` from Phase 2). Sketch (match the surrounding file's style — `"use strict"`, `appPrefix`, the `ns.osdReady` pattern Phase 2 already established, JSDoc):

```javascript
/* ============================================================
 * Synced "Compare" strip (spec §7, §15.10). Surface-agnostic: the
 * caller supplies dziUrlBuilder(ref) + titleFor(ref); this owns the modal,
 * the ≤cap OSD viewers, the shared viewport (feedback-guarded), the over-cap
 * notice, and full teardown on close. Mirrors results_viewer.js lock-views.
 * ============================================================ */
(function () {
    "use strict";
    const ns = (window.__phenotypicTimeline = window.__phenotypicTimeline || {});
    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix : "/";

    // Reuse the surface's OSD loader if Phase 2 exposed one; else lazy-inject the
    // vendored copy (NO CDN). Both browse.js and results_viewer.js vendor OSD.
    ns.osdReady = ns.osdReady || (function () {
        return new Promise(function (resolve, reject) {
            if (window.OpenSeadragon) { resolve(); return; }
            const tag = document.createElement("script");
            tag.src = appPrefix + "assets/openseadragon/openseadragon.min.js";
            tag.async = true;
            tag.onload = function () { resolve(); };
            tag.onerror = function () { reject(new Error("OSD vendored load failed")); };
            document.head.appendChild(tag);
        });
    })();

    // CONTROLLER STATE — deliberately process-global (M3): at most one strip is
    // open at a time, on one surface (Browse/Results never co-mount).
    let _compareViewers = [];      // OSD viewers currently mounted in the strip
    let _broadcasting = false;     // feedback-loop guard while applying a peer viewport
    // COMMITTED TEST SEAM (M1): the mounted viewer list is mirrored onto the
    // namespace so the Playwright e2e (Task 7) can drive vs[0]/vs[1] and assert
    // viewport sync. Kept in lock-step with _compareViewers in openCompareStrip
    // + teardownCompare. This is an intentional public seam, not incidental.
    ns.__compareViewers = ns.__compareViewers || [];

    // Mirror the focused source viewer's viewport onto every peer. The guard
    // makes the peer-applied "animation" events no-ops, so a pan/zoom in one
    // viewer does not bounce back through the others (results_viewer.js:167).
    function broadcastViewport(src) {
        if (_broadcasting) { return; }
        _broadcasting = true;
        try {
            const center = src.viewport.getCenter(true);
            const zoom = src.viewport.getZoom(true);
            _compareViewers.forEach(function (v) {
                if (v === src) { return; }
                v.viewport.zoomTo(zoom, null, true);
                v.viewport.panTo(center, true);
            });
        } finally {
            _broadcasting = false;
        }
    }

    function teardownCompare() {
        // viewer.destroy() removes the OSD canvas AND detaches the handlers we
        // added via addHandler (incl. the "animation" sync handler), so there is
        // no handler leak — destroy() is the complete teardown (C3).
        _compareViewers.forEach(function (v) {
            try { v.destroy(); } catch (e) { /* already torn down */ }
        });
        _compareViewers = [];
        ns.__compareViewers = [];                // keep the seam in lock-step (M1)
        const modal = document.getElementById("timeline-compare-modal");
        if (modal) { modal.remove(); }  // release the strip DOM + canvases
    }
    ns.closeCompareStrip = teardownCompare;

    // refs: array (selection order). opts: { dziUrlBuilder, titleFor, cap }.
    // The cap is the caller's responsibility (Task 6 reads data-compare-cap);
    // openCompareStrip trusts opts.cap and does NOT re-fabricate a fallback
    // number — the cap flows from the Python TIMELINE_COMPARE_CAP → DOM → here.
    ns.openCompareStrip = async function (refs, opts) {
        await ns.osdReady;
        teardownCompare();                       // never stack strips
        const cap = opts.cap;
        const list = Array.isArray(refs) ? refs : [];
        const shown = list.slice(0, cap);
        const overCap = list.length > cap;

        // Build the modal DOM (a single dynamically-created overlay; the Dash
        // layout only needs the Compare button — see Task 5). Close button +
        // backdrop click both call teardownCompare. See the DOM CONTRACT below.
        const modal = buildCompareModal();        // local helper; sets id=timeline-compare-modal
        if (overCap) {
            // EXACT mirror of compare_selection_plan(...).notice (Task 2),
            // including the em-dash "—". The unit test pins the Python copy.
            renderOverCapNotice(modal,
                "Showing first " + cap + " of " + list.length + " — narrow the selection");
        }
        const strip = modal.querySelector(".timeline-compare-strip");
        shown.forEach(function (ref, i) {
            const cell = document.createElement("div");
            cell.className = "timeline-compare-cell";
            const title = document.createElement("div");
            title.className = "timeline-compare-cell-title";
            title.textContent = (opts && opts.titleFor) ? opts.titleFor(ref) : String(ref);
            const host = document.createElement("div");
            host.className = "timeline-compare-osd osd-canvas";
            host.id = "timeline-compare-osd-" + i;
            cell.appendChild(title); cell.appendChild(host); strip.appendChild(cell);

            const viewer = window.OpenSeadragon({
                element: host,
                prefixUrl: appPrefix + "assets/openseadragon/images/",
                tileSources: opts.dziUrlBuilder(ref),
                showNavigator: false,
                showRotationControl: false,
                constrainDuringPan: true,
                visibilityRatio: 0.5,
                immediateRender: false,
            });
            viewer.addHandler("animation", function () { broadcastViewport(viewer); });
            _compareViewers.push(viewer);
        });
        ns.__compareViewers = _compareViewers;   // refresh the seam each open (M1)
        document.body.appendChild(modal);
        return _compareViewers.length;
    };

    // --- Modal DOM contract (M2) — both this controller AND the Task 7 e2e
    //     depend on these EXACT ids/classes/nesting; keep them in sync:
    //
    //   <div id="timeline-compare-modal" class="timeline-compare-modal">
    //     <button id="timeline-compare-close" class="timeline-compare-close">×</button>
    //     <div class="timeline-compare-notice"></div>      <!-- over-cap slot; empty otherwise -->
    //     <div class="timeline-compare-strip">             <!-- horizontal flex row -->
    //       <div class="timeline-compare-cell">            <!-- one per shown ref -->
    //         <div class="timeline-compare-cell-title">…</div>
    //         <div class="timeline-compare-osd osd-canvas" id="timeline-compare-osd-<i>"></div>
    //       </div> …
    //     </div>
    //   </div>
    //
    function buildCompareModal() {
        const modal = document.createElement("div");
        modal.id = "timeline-compare-modal";
        modal.className = "timeline-compare-modal";
        // Backdrop click (on the overlay itself, not its children) closes.
        modal.addEventListener("click", function (ev) {
            if (ev.target === modal) { teardownCompare(); }
        });
        const close = document.createElement("button");
        close.id = "timeline-compare-close";
        close.className = "timeline-compare-close";
        close.type = "button";
        close.textContent = "×";
        close.addEventListener("click", teardownCompare);
        const notice = document.createElement("div");
        notice.className = "timeline-compare-notice";   // empty unless over cap
        const strip = document.createElement("div");
        strip.className = "timeline-compare-strip";
        modal.appendChild(close);
        modal.appendChild(notice);
        modal.appendChild(strip);
        return modal;
    }

    function renderOverCapNotice(modal, text) {
        const notice = modal.querySelector(".timeline-compare-notice");
        if (notice) { notice.textContent = text; }
    }
})();
```

`buildCompareModal()` / `renderOverCapNotice()` are written above (not left as sketches) so the DOM contract is explicit and the Task 7 e2e selectors are guaranteed to match. Add the matching CSS to `browse.css` (`.timeline-compare-modal` fixed full-viewport overlay + backdrop, `.timeline-compare-strip` horizontal flex with per-cell min-width, `.timeline-compare-cell` column, `.timeline-compare-osd` sized box, `.timeline-compare-notice` warning text, `.timeline-compare-close` corner button, `.timeline-cell--selected` selection ring) — use `var(--color-*)` / `var(--sp-*)` / `var(--radius)` tokens, never raw hex (GUI module guide).

> **Note (DOM vs `dbc.Modal`):** the controller builds the modal in **plain DOM** (it owns the dynamic viewer set) rather than toggling a Dash `dbc.Modal`, because the viewers are created/destroyed entirely client-side and a server round-trip per open is unnecessary. The Dash layout (Task 5) provides only the "Compare selected" button. This matches how `results_viewer.js` owns its viewers outside Dash's vdom. (Open Question OQ-1 below weighs a `dbc.Modal` alternative — default is plain DOM.)

- [ ] **Step 3: Verify it loads (smoke), incl. the committed test seam**

There is no unit harness for OSD; the controller is exercised in Task 7's Playwright e2e. As a cheap pre-check, confirm the file parses, the namespace export is present, and the **committed `ns.__compareViewers` seam** that Task 7 reads is assigned (M1):
`uv run python -c "import pathlib; s=pathlib.Path('src/phenotypic/gui/browse/_assets/timeline.js').read_text(); assert 'openCompareStrip' in s and 'broadcastViewport' in s and 'destroy()' in s and 'ns.__compareViewers = _compareViewers' in s and 'timeline-compare-strip' in s"`

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/gui/browse/_assets/timeline.js src/phenotypic/gui/browse/_assets/browse.css
git commit -m "feat(gui-timeline): shared openCompareStrip controller (≤cap synced OSD viewers)"
```

---

### Task 5: Browse compare id + layout (Compare button + `data-compare-cap`)

**Files:**
- Modify: `src/phenotypic/gui/browse/_ids.py` (append the compare button id + `__all__`)
- Modify: `src/phenotypic/gui/browse/_layout.py` (`build_timeline_body()` — add the Compare button + the grid container's `data-compare-cap`)
- Test: `tests/gui/browse/test_ids.py` (append) + `tests/gui/browse/test_layout.py` (append)

**Interfaces:**
- Consumes: Phase 2 Browse ids + `build_timeline_body()`; `TIMELINE_COMPARE_CAP` (Task 1).
- Produces:
  - One new id: `BROWSE_TL_COMPARE_BTN` ("Compare selected" button — a `timeline.js` DOM target, **no Dash callback**). (C5: no speculative `*_HOST`/`*_INPUT` ids — the modal is created in plain DOM appended to `document.body` and the v1 triggers are pure-JS, so neither a Dash host div nor a JS→Dash bridge input has a concrete consumer. Add them only if a future server callback needs them.)
  - `build_timeline_body()` (modified): the `BROWSE_TL_GRID` container additionally carries a static `data-compare-cap = str(TIMELINE_COMPARE_CAP)` data-attr (so `timeline.js` reads the cap off the DOM, like it already reads `data-focus-margin`/`data-mount-cap`/`data-warm-concurrency`); a "Compare selected" button (`BROWSE_TL_COMPARE_BTN`) is added next to the tile-size stepper.

> **Verified:** the Phase 2 `BROWSE_TL_GRID` container already exposes static data-attrs read by the controller (`data-focus-margin`, `data-mount-cap`, `data-warm-concurrency`) — adding `data-compare-cap` follows that exact established pattern (Phase 2 plan, "Global Constraints" + Task 6). `build_timeline_body()` lives in `browse/_layout.py` and is wired into `build_browse_layout()` (Phase 2 Task 6). The four `BROWSE_TL_NAV_*` ids and `BROWSE_TL_POSITION` are precedent for "DOM target, no Dash callback" ids (Phase 2 Task 1).

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/browse/test_ids.py`:

```python
def test_compare_button_id_is_a_nonempty_str() -> None:
    from phenotypic.gui.browse import _ids

    assert isinstance(_ids.BROWSE_TL_COMPARE_BTN, str)
    assert _ids.BROWSE_TL_COMPARE_BTN
```

Append to `tests/gui/browse/test_layout.py` (reuse the `_walk_ids` helper added in Phase 2 Task 6):

```python
def test_timeline_body_has_compare_button() -> None:
    from phenotypic.gui.browse._layout import build_browse_layout
    from phenotypic.gui.browse import _ids

    ids = _walk_ids(build_browse_layout())
    assert _ids.BROWSE_TL_COMPARE_BTN in ids


def test_grid_container_exposes_compare_cap_dataattr() -> None:
    # timeline.js reads the cap off the DOM (like data-focus-margin), so the
    # static data-compare-cap must equal TIMELINE_COMPARE_CAP.
    from phenotypic.gui._config import TIMELINE_COMPARE_CAP
    from phenotypic.gui.browse._layout import build_browse_layout
    from phenotypic.gui.browse import _ids

    def _find(node, target_id):
        if getattr(node, "id", None) == target_id:
            return node
        children = getattr(node, "children", None)
        seq = children if isinstance(children, (list, tuple)) else (
            [children] if children is not None else []
        )
        for child in seq:
            hit = _find(child, target_id)
            if hit is not None:
                return hit
        return None

    grid = _find(build_browse_layout(), _ids.BROWSE_TL_GRID)
    assert grid is not None
    props = grid.to_plotly_json().get("props", {})
    assert props.get("data-compare-cap") == str(TIMELINE_COMPARE_CAP)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_ids.py::test_compare_button_id_is_a_nonempty_str tests/gui/browse/test_layout.py::test_timeline_body_has_compare_button tests/gui/browse/test_layout.py::test_grid_container_exposes_compare_cap_dataattr -v`
Expected: FAIL (id/attr absent).

- [ ] **Step 3: Write minimal implementation**

Append to `src/phenotypic/gui/browse/_ids.py` (before `__all__`, then add it to `__all__`):

```python
# --- Compare strip (Phase 4) ---------------------------------------------
BROWSE_TL_COMPARE_BTN = "browse-tl-compare-btn"      # "Compare selected" (timeline.js target)
```

In `src/phenotypic/gui/browse/_layout.py` `build_timeline_body()`:
- import `TIMELINE_COMPARE_CAP` from `_config`;
- add `**{"data-compare-cap": str(TIMELINE_COMPARE_CAP)}` to the `BROWSE_TL_GRID` container's props (alongside the Phase 2 `data-focus-margin`/`data-mount-cap`/`data-warm-concurrency`);
- add an `html.Button("Compare selected", id=ids.BROWSE_TL_COMPARE_BTN, n_clicks=0, type="button", disabled=False)` next to the tile-size stepper.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/browse/test_ids.py tests/gui/browse/test_layout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_ids.py src/phenotypic/gui/browse/_layout.py tests/gui/browse/test_ids.py tests/gui/browse/test_layout.py
git commit -m "feat(gui-timeline): Browse Compare button id + data-compare-cap"
```

---

### Task 6: Browse selection + triggers in `timeline.js` (multi-select, row-header, Compare button)

**Files:**
- Modify: `src/phenotypic/gui/browse/_assets/timeline.js` (extend the Phase 2 `ns.attach` controller with selection + triggers)
- Test: covered live in Task 7's Browse e2e.

**Interfaces:**
- Consumes: the Phase 2 controller (`ns.attach(containerId)`, `ns._focus`, `cellAt`, the delegated container click listener), the cells' `data-ref`/`data-row`/`data-col`/`data-key`/`data-row-index`/`data-col-index`, the axis row-headers (`.timeline-axis-label--y` carrying **`data-row`/`data-row-index`** — see the verified note below), the `BROWSE_TL_COMPARE_BTN` button, and the container's `data-compare-cap`. Browse builds `dziUrlBuilder(ref)` = `appPrefix + "tiles/" + encodeURIComponent(ref) + ".dzi"` (the cell's `data-ref` is the Browse **token**, per Phase 2 `ref_builder=lambda r: _source_render.encode_token(str(r))`).
- Produces: shift/ctrl/cmd-click on a tile toggles the `.timeline-cell--selected` class; the "Compare selected" button reads the selected refs (in grid order, **derived from the DOM class**) and calls `ns.openCompareStrip(refs, { dziUrlBuilder, titleFor, cap })`; clicking a **row-header** selects that whole row's populated cells and opens the strip immediately.

> **Phase 1 row-header verified (OQ-4 RESOLVED upstream):** the Phase 1 plan has been amended so axis labels carry data attributes — `--y` row headers get **`data-row`/`data-row-index`** and `--x` time headers get `data-col`/`data-col-index`. So a row-header click matches its cells by **attribute**: read the clicked header's `data-row` and collect `container.querySelectorAll('.timeline-cell[data-src][data-row="<value>"]')`. **Do NOT match by `textContent`** — the attribute match is robust against whitespace / duplicate-looking labels and is now available with no Phase 1 follow-up.

> **DOM class is the SINGLE source of truth for selection (M4):** the selection is tracked **only** by the `.timeline-cell--selected` class on cells — there is **no** parallel `Set`. The Phase 2 render callback replaces cell children on each (Y/time/pattern/source) change, which clears the class; a separate `Set` would persist stale refs and silently desync from the rendered grid. By deriving the selection from the live DOM (`selectionRefsInGridOrder()` queries `.timeline-cell--selected`), a re-render naturally resets the selection — re-render-safe and simpler. Because the class lives on the surface's own cells, two distinct containers (Browse vs a future Results) never share a selection; this is per-surface by construction, with no per-container bookkeeping object. (The Compare *controller* state — `_compareViewers`, the modal — is a deliberate global singleton, M3; only the selection is per-surface, and it is just the DOM.)

- [ ] **Step 1: (Covered by Task 7 e2e — no standalone unit harness)**

- [ ] **Step 2: Write the implementation**

Extend `ns.attach(containerId)` in `timeline.js` (the Phase 2 controller). Within the existing delegated container click listener (the one Phase 2 added for `.timeline-cell-popout`), branch on the event:

```javascript
// Inside ns.attach, alongside the Phase 2 listeners. SELECTION IS THE DOM:
// the .timeline-cell--selected class is the single source of truth (M4) — no
// parallel Set, so a Phase-2 grid re-render (which clears the class) cannot
// desync. Selection is per-surface by construction (the class lives on this
// surface's cells); a future Results container never shares it.

function selectionRefsInGridOrder() {
    // Stable order = grid order: sort selected cells by (rowIndex, colIndex),
    // reading the live DOM each time the button fires.
    return Array.from(container.querySelectorAll(".timeline-cell.timeline-cell--selected[data-ref]"))
        .sort(function (a, b) {
            const ra = parseInt(a.getAttribute("data-row-index"), 10) || 0;
            const rb = parseInt(b.getAttribute("data-row-index"), 10) || 0;
            if (ra !== rb) { return ra - rb; }
            return (parseInt(a.getAttribute("data-col-index"), 10) || 0)
                 - (parseInt(b.getAttribute("data-col-index"), 10) || 0);
        })
        .map(function (el) { return el.getAttribute("data-ref"); });
}

function browseDziUrl(ref) {
    return appPrefix + "tiles/" + encodeURIComponent(ref) + ".dzi";
}
// SINGLE place the cap literal lives in JS (C1): read data-compare-cap off the
// container (written from the Python TIMELINE_COMPARE_CAP in Task 5), with one
// fallback. openCompareStrip trusts the cap and never re-fabricates it.
function compareCap() { return num(container, "data-compare-cap", 12); }

// (a) shift/ctrl/cmd-click a tile toggles selection (does NOT open the pop-out).
//     The class IS the state — no Set to keep in sync.
container.addEventListener("click", function (ev) {
    if (!(ev.shiftKey || ev.ctrlKey || ev.metaKey)) { return; }
    const cell = ev.target && ev.target.closest
        ? ev.target.closest(".timeline-cell[data-ref]") : null;
    if (!cell) { return; }
    ev.preventDefault();
    cell.classList.toggle("timeline-cell--selected");
});

// (b) Compare-selected button opens the strip for the current selection.
const compareBtn = document.getElementById("browse-tl-compare-btn");
if (compareBtn && !compareBtn.dataset.tlCompareBound) {
    compareBtn.dataset.tlCompareBound = "1";
    compareBtn.addEventListener("click", function () {
        const refs = selectionRefsInGridOrder();
        if (!refs.length) { return; }
        ns.openCompareStrip(refs, {
            dziUrlBuilder: browseDziUrl,
            titleFor: function (r) { return r; },
            cap: compareCap(),
        });
    });
}

// (c) row-header click → select that whole row's populated cells + open.
//     Match cells by the header's data-row ATTRIBUTE (OQ-4 resolved: Phase 1
//     now emits data-row on the --y label), NOT textContent.
if (!container.__tlHeaderBound) {
    container.__tlHeaderBound = true;
    container.addEventListener("click", function (ev) {
        const header = ev.target && ev.target.closest
            ? ev.target.closest(".timeline-axis-label--y") : null;
        if (!header) { return; }
        const rowValue = header.getAttribute("data-row");
        if (rowValue === null) { return; }
        const cells = Array.from(container.querySelectorAll(".timeline-cell[data-src][data-row]"))
            .filter(function (c) { return c.getAttribute("data-row") === rowValue; });
        const refs = cells.map(function (c) { return c.getAttribute("data-ref"); });
        if (!refs.length) { return; }
        ns.openCompareStrip(refs, {
            dziUrlBuilder: browseDziUrl,
            titleFor: function (r) { return r; },
            cap: compareCap(),
        });
    });
}
```

> **Click-handler ordering (important):** the Phase 2 pop-out handler fires on a *plain* tile click (no modifier). Guard the **selection** branch on a modifier key (`shiftKey || ctrlKey || metaKey`) and the **pop-out** branch on its absence, so a plain click still pops out and a shift/ctrl-click only toggles selection (does not also pop out). Confirm the Phase 2 pop-out delegated handler does not fire on a modified click — if it currently fires unconditionally on `.timeline-cell-popout`, the modifier branch above (which targets the whole `.timeline-cell`, not the ⤢ button) won't conflict, but assert this in the e2e (a shift-click should not open the pop-out modal).

- [ ] **Step 3: Smoke-parse**

`uv run python -c "import pathlib; s=pathlib.Path('src/phenotypic/gui/browse/_assets/timeline.js').read_text(); assert 'timeline-cell--selected' in s and 'timeline-axis-label--y' in s and 'openCompareStrip' in s and 'new Set(' not in s and 'getAttribute(\"data-row\")' in s"`

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/gui/browse/_assets/timeline.js
git commit -m "feat(gui-timeline): Browse selection + row-header/Compare-button triggers"
```

---

### Task 7: Browse Compare-strip e2e (Playwright)

**Files:**
- Create/extend: `tests/e2e/gui/test_browse_compare_strip.py` (new file; reuse the Phase 2 `live_browse_timeline` fixture from `tests/e2e/gui/conftest.py` / the Phase 2 test module)
- Test: itself (this IS the test task)

**Interfaces:**
- Consumes: the Phase 2 `live_browse_timeline` fixture (server up, Browse open, Timeline mode on, a seeded ≥3×3 matrix under `plate1` via the **sidebar-tree-click** source seeding — verified idiom `tests/e2e/gui/test_shared_source_root.py::_select_plate1_source`, NOT localStorage injection); the Task 4/6 controller; the Task 5 layout (`#browse-tl-compare-btn`).
- Produces: e2e coverage of the four behaviors (multi-select → Compare → N viewers; viewport sync; row-header → strip; over-cap notice).

> **Fixture reuse (verified):** Phase 2 Task 7 defines `live_browse_timeline` in `tests/e2e/gui/conftest.py` seeding `t0/t1/t2 × plateA/plateB/plateC` PNGs under `fake_sandbox/plate1` and selecting `plate1` via the sidebar tree (`page.click('button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]')`), then switching to Timeline mode and waiting for `.timeline-cell[data-src]`. **Reuse it as-is.** A 3×3 matrix yields ≤9 cells (under the cap of 12), so the over-cap test needs a *larger* seed — add a dedicated `live_browse_timeline_large` fixture seeding **exactly 14** populated cells (`t0..t6 × plateA/plateB` = 7×2 = 14) so the over-cap notice asserts the exact `"Showing first 12 of 14 — narrow the selection"` string (Step 3). **Mark the module `ci_flaky`** per `tests/CLAUDE.md` (the OSD-mount + tile-fetch budget on GHA shared runners is the exact wall-clock-poll signature `ci_flaky` exists for); pair the marker with a one-line comment naming the budget.

- [ ] **Step 1: Write the test (it fails until Tasks 4–6 land)**

Create `tests/e2e/gui/test_browse_compare_strip.py`:

```python
"""Browse synced Compare strip e2e (spec §7). Mirror once Phase 3 lands Results."""
from __future__ import annotations

import pytest

# Module-level: OSD mount + per-viewer tile fetch on GHA shared runners
# stochastically exceeds the wait_for_selector budget (tests/CLAUDE.md).
pytestmark = pytest.mark.ci_flaky


def test_multiselect_then_compare_mounts_exactly_n_viewers(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell[data-src]")
    cells = page.query_selector_all(".timeline-cell[data-src]")
    assert len(cells) >= 3
    # Shift-click 3 distinct populated tiles → 3 selected.
    for cell in cells[:3]:
        cell.click(modifiers=["Shift"])
    assert len(page.query_selector_all(".timeline-cell--selected")) == 3
    page.click("#browse-tl-compare-btn")
    # Exactly 3 OSD viewers mount, each with a canvas.
    page.wait_for_selector("#timeline-compare-modal .timeline-compare-osd canvas", timeout=15_000)
    osd_cells = page.query_selector_all("#timeline-compare-modal .timeline-compare-cell")
    assert len(osd_cells) == 3
    canvases = page.query_selector_all("#timeline-compare-modal .timeline-compare-osd canvas")
    assert len(canvases) >= 3  # OSD draws ≥1 canvas per viewer


def test_pan_zoom_one_viewer_propagates_to_peers(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell[data-src]")
    cells = page.query_selector_all(".timeline-cell[data-src]")
    for cell in cells[:2]:
        cell.click(modifiers=["Shift"])
    page.click("#browse-tl-compare-btn")
    page.wait_for_selector("#timeline-compare-modal .timeline-compare-osd canvas", timeout=15_000)
    # Drive viewer[0]'s viewport via the OSD API and poll viewer[1]'s zoom to
    # confirm the shared viewport propagated. window.__phenotypicTimeline
    # .__compareViewers is the COMMITTED test seam assigned in Task 4 (M1).
    page.evaluate(
        "() => { const vs = window.__phenotypicTimeline.__compareViewers; "
        "vs[0].viewport.zoomTo(2.0); }"
    )
    page.wait_for_function(
        "() => { const vs = window.__phenotypicTimeline.__compareViewers; "
        "return Math.abs(vs[1].viewport.getZoom(true) - vs[0].viewport.getZoom(true)) < 0.05; }",
        timeout=10_000,
    )


def test_row_header_click_opens_strip_for_that_row(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector(".timeline-axis-label--y")
    page.click(".timeline-axis-label--y")  # first row header
    page.wait_for_selector("#timeline-compare-modal .timeline-compare-osd canvas", timeout=15_000)
    osd_cells = page.query_selector_all("#timeline-compare-modal .timeline-compare-cell")
    # The seeded matrix has 3 time columns per row → 3 viewers for that row.
    assert len(osd_cells) == 3


def test_over_cap_selection_shows_notice(live_browse_timeline_large) -> None:
    page = live_browse_timeline_large  # exactly 14 populated cells (see fixture)
    page.wait_for_selector(".timeline-cell[data-src]")
    # Select via JS by toggling the class on EVERY populated cell directly,
    # NOT by physical clicks (M5/C2): in the no-scroll centered window,
    # off-window cells are positioned via CSS transform and are not reliably
    # hit-testable, so 14 physical shift-clicks would be flaky. The selection
    # source of truth is the .timeline-cell--selected class (M4), so setting it
    # is equivalent to clicking. Assert the seeded count is exactly 14 first.
    total = page.evaluate(
        "() => { const cs = document.querySelectorAll('.timeline-cell[data-src]'); "
        "cs.forEach(c => c.classList.add('timeline-cell--selected')); return cs.length; }"
    )
    assert total == 14
    page.click("#browse-tl-compare-btn")
    page.wait_for_selector("#timeline-compare-modal .timeline-compare-notice", timeout=15_000)
    notice = page.text_content("#timeline-compare-modal .timeline-compare-notice")
    # EXACT full string — guards the em-dash "—" coupling between the JS mirror
    # and the Python compare_selection_plan(...).notice (Task 2). 14 selected,
    # cap 12 → "Showing first 12 of 14 — narrow the selection".
    assert notice == "Showing first 12 of 14 — narrow the selection"
    # Cap honored: exactly 12 viewer cells despite 14 selected.
    assert len(page.query_selector_all("#timeline-compare-modal .timeline-compare-cell")) == 12


def test_shift_click_does_not_open_popout(live_browse_timeline) -> None:
    # A modified click toggles selection only; the deep-zoom pop-out (Phase 2)
    # must NOT open on shift-click.
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell[data-src]")
    page.query_selector(".timeline-cell[data-src]").click(modifiers=["Shift"])
    assert page.query_selector("#timeline-compare-modal") is None
```

> **Test seam for the viewer list (`__compareViewers`):** the sync test reads the mounted viewers via `window.__phenotypicTimeline.__compareViewers`. This is **already a committed seam in Task 4** (M1) — `openCompareStrip` assigns `ns.__compareViewers = _compareViewers` on each open and `teardownCompare` resets it to `[]`. No extra work in Task 7; the seam exists before this test runs (Task 4 commits first).

- [ ] **Step 2: Run the e2e**

Run: `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_compare_strip.py -v`
Expected: with Tasks 1–6 landed, PASS. If a timing budget flakes on CI only, the `ci_flaky` module marker keeps the CI lane green (`-m "not ci_flaky"`).

- [ ] **Step 3: Add the `live_browse_timeline_large` fixture**

Add a function-scoped `live_browse_timeline_large` fixture (alongside `live_browse_timeline` in `conftest.py` or the Phase 2 test module) seeding **exactly 14** populated cells under `plate1` (`t0..t6 × plateA, plateB` = 7×2 = 14), reusing the same sidebar-tree-click seeding path. The exact count (14) is load-bearing — the over-cap test asserts the full notice string `"Showing first 12 of 14 — narrow the selection"`; if you change the seed, update that assertion to match.

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/gui/test_browse_compare_strip.py tests/e2e/gui/conftest.py
git commit -m "test(gui-timeline): Browse compare-strip e2e (multi-select, sync, row-header, over-cap)"
```

---

### Task 8: Results-surface wiring (mirror Browse — TODO-from-Phase-3 ids)

**Files:**
- Modify (when Phase 3 lands): `src/phenotypic/gui/results_viewer/timeline_view/_assets/timeline.js` (or wherever Phase 3 mounts the Results timeline controller) + the Results timeline `_layout.py`/`_ids.py`.
- Test (when Phase 3 lands): `tests/e2e/gui/test_results_compare_strip.py` (mirror Task 7).

**This task does NOT hard-depend on Phase 3 internals.** It documents the mirror so the Results surface reuses the same shared controller. **Do not block Phase 4 merge on it** — land it as a follow-up once Phase 3's Results Timeline tab + ids exist.

**Interfaces (mirror of Browse):**
- The Results timeline grid cells carry `data-ref = "<dataset>/<stem>"` (Phase 3's `ref_builder`), so `dziUrlBuilder(ref)` = `appPrefix + "tiles/" + ref.split("/").map(encodeURIComponent).join("/") + ".dzi"` — i.e. `/tiles/<dataset>/<stem>.dzi` (verified Results DZI route `results_viewer/_tile_routes.py`, prefix `VIEWER_TILES_PREFIX`). **Do NOT `encodeURIComponent` the whole `dataset/stem`** — that would percent-encode the `/` separator; encode each segment, join with `/`, exactly as `results_viewer.js:297-298` does.
- The same `ns.openCompareStrip(refs, opts)` controller is reused unchanged — Phase 3 must load the **same `timeline.js`** under the Results asset folder (each surface vendors its own OSD + copies the shared `timeline.js`, mirroring how both surfaces vendor `openseadragon.min.js`). The selection lives in the `.timeline-cell--selected` DOM class on the surface's own cells (Task 6, M4), so a distinct Results container naturally has its own selection — no shared state.
- Mirror the Browse layout additions: a "Compare selected" button id `TODO-from-Phase-3` (mirror `BROWSE_TL_COMPARE_BTN`), the Results grid container's `data-compare-cap = str(TIMELINE_COMPARE_CAP)` (mirror Task 5), and the row-header `.timeline-axis-label--y` triggers (Phase 1 emits the same class — and the `data-row` attribute — on both surfaces).

> **Singleton caveat — be precise (M3):** the Compare *controller state* (`_compareViewers`, `_broadcasting`, the modal) is a deliberate **process-global singleton** — exactly one strip is open at a time, on one surface, and Browse/Results never co-mount (separate Flask servers / page loads). **Do not** re-architect it to be per-container — that would over-engineer a correctly-singleton component. What *is* per-surface is the **selection**, and only because it is the `.timeline-cell--selected` class on the surface's own cells (no bookkeeping object to key). Phase 3's only obligation is to give the Results timeline container a distinct id and call `ns.attach(<results-container-id>)`; the shared controller needs no per-surface generalization.

- [ ] **Step 1:** (Deferred to the Phase-3 follow-up.) When Phase 3 lands, copy `timeline.js` (incl. the Task 4/6 compare sections) into the Results asset folder, add the Results "Compare selected" button + `data-compare-cap` to the Results timeline layout, and create `tests/e2e/gui/test_results_compare_strip.py` mirroring Task 7 against the real Results reference data (per spec §16.9: Results over `…/data/results/2026-06-16/`, X=`Metadata_ImageNumber`, Y=`Metadata_PlateNum`).
- [ ] **Step 2:** Add the Results FEATURES.md rows (Task 9 already drafts both surfaces; ensure the Results rows resolve to the Results e2e once it exists).
- [ ] **Step 3: Commit** (with the Phase-3 follow-up PR, not this phase).

---

### Task 9: FEATURES.md rows + WORKFLOWS.md decision + lint/typecheck

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md` (Browse + Results compare-strip rows)
- Modify (decision): `src/phenotypic/gui/WORKFLOWS.md` — see the decision note below.
- Test: the CI `features-md-gate` (and `workflows-md-gate` if a row is added) enforce these; locally run the check scripts.

**Interfaces:**
- Consumes: every affordance from Tasks 4–7.
- Produces: FEATURES rows with resolvable `path::test` refs.

> **Verified FEATURES.md structure:** there is a `## Browse tab (source image viewer)` section (header `| Feature | Element | Expected behaviour | Status | Test layer | Test ref |`) and a `## Results Viewer integration` section. `check_features_md.py` enforces cell-count == header-count; match the 6-column header. Use `✅ shipping` only with a resolvable `path::test`; never `🚧 in progress` (merge gate rejects it). For the **Results** rows (Task 8 is a follow-up), set status `🔭 planned` (no test yet) — or omit them until Phase 3 + the Results e2e land, and add them in the Task 8 follow-up PR. **Default: add the Browse rows `✅ shipping` now (Task 7 e2e exists); add Results rows in the Task 8 follow-up.**

- [ ] **Step 1: Add the Browse FEATURES rows**

Under the `## Browse tab (source image viewer)` table in `src/phenotypic/gui/FEATURES.md`, append (one row per affordance; `Test ref` points at the Task 7 e2e):

```markdown
| Compare strip (Browse) | `#browse-tl-compare-btn` + `window.__phenotypicTimeline.openCompareStrip` | Opens a modal of ≤`TIMELINE_COMPARE_CAP` deep-zoom OSD viewers for the selected cells with a shared viewport (pan/zoom in one propagates to all); tears down all viewers on close. | ✅ shipping | e2e | tests/e2e/gui/test_browse_compare_strip.py::test_multiselect_then_compare_mounts_exactly_n_viewers |
| Compare viewport sync (Browse) | Shared-viewport feedback guard in `timeline.js` | Pan/zoom in any Compare viewer mirrors to every peer, guarded against feedback loops. | ✅ shipping | e2e | tests/e2e/gui/test_browse_compare_strip.py::test_pan_zoom_one_viewer_propagates_to_peers |
| Multi-select cells (Browse) | `.timeline-cell--selected` + shift/ctrl-click | Shift/ctrl-click toggles a tile's membership in the Compare selection (a plain click still opens the pop-out). | ✅ shipping | e2e | tests/e2e/gui/test_browse_compare_strip.py::test_shift_click_does_not_open_popout |
| Row-header compare (Browse) | `.timeline-axis-label--y` click | Clicking an axis row-header opens the Compare strip for that row's full time-course. | ✅ shipping | e2e | tests/e2e/gui/test_browse_compare_strip.py::test_row_header_click_opens_strip_for_that_row |
| Compare over-cap notice (Browse) | `.timeline-compare-notice` | Selecting more than `TIMELINE_COMPARE_CAP` cells mounts the first cap and shows "Showing first N of M — narrow the selection" — never a silent truncation. | ✅ shipping | e2e | tests/e2e/gui/test_browse_compare_strip.py::test_over_cap_selection_shows_notice |
```

Also add a cap-logic infra row under `## Cross-cutting infrastructure` (the pure helper is the spec-text guard):

```markdown
| Compare-strip cap logic | `compare_selection_plan` (`gui/_shared/timeline/`) | Pure cap/over-cap planner: ≤cap → all shown, >cap → first cap + the verbatim notice string the JS controller mirrors. | 🧪 internal | unit | tests/gui/_shared/timeline/test_compare.py::test_over_cap_truncates_to_cap_and_emits_notice |
```

- [ ] **Step 2: WORKFLOWS.md decision (no new row — fold into existing tutorials)**

**Decision (recommended default):** do **NOT** add a new WORKFLOWS.md row for the Compare strip. Rationale: §7 is a *feature within* the two existing planned timeline workflows (Browse "find the ideal starting time" and Results "trait emergence over time", spec §11), not a standalone end-to-end flow worth its own tutorial page + `_capture_<id>`. Adding a WORKFLOWS row would force a new `_capture_compare_strip` screenshot function + a `docs/source/tutorials/gui/` page (the `workflows-md-gate` round-trip), which is disproportionate for an in-flow affordance. Instead, the Compare strip is demonstrated **inside** the Browse/Results timeline tutorial pages those workflows already own. **Note this explicitly** in the Phase 6 (docs) plan so the timeline tutorial capture includes a Compare-strip screenshot. (If a reviewer wants a dedicated flow, that is OQ-5 below.)

Because this task touches `src/phenotypic/gui/` (FEATURES.md), no WORKFLOWS.md edit is required by the gate (only FEATURES.md is the mandatory touch).

- [ ] **Step 3: Lint, typecheck, run the gates**

Run:
```bash
uv run ruff check src/phenotypic/gui/_config.py src/phenotypic/gui/_shared/timeline src/phenotypic/gui/browse/_ids.py src/phenotypic/gui/browse/_layout.py
uv run mypy src/phenotypic/gui/_shared/timeline
uv run python scripts/check_features_md.py        # FEATURES.md gate
uv run pytest tests/gui/_shared/timeline tests/gui/browse/test_ids.py tests/gui/browse/test_layout.py -q
```
Expected: clean lint/types; FEATURES gate green; unit suites green.

- [ ] **Step 4: Run the full Browse e2e (local, incl. ci_flaky)**

Run: `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_compare_strip.py -v`
Expected: PASS locally (the `ci_flaky` marker absorbs CI shared-runner timing).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/FEATURES.md
git commit -m "docs(gui-timeline): FEATURES rows for the synced Compare strip (Browse)"
```

---

## Phase 4 deliverable

A working, viewport-**synced Compare strip** on the Browse timeline surface: shift/ctrl-click multi-select + a "Compare selected" button + row-header-to-compare, opening a modal of ≤`TIMELINE_COMPARE_CAP` deep-zoom OSD viewers with a shared, feedback-guarded viewport, an over-cap notice, and full teardown on close. The controller (`window.__phenotypicTimeline.openCompareStrip`) is **surface-agnostic** (per-surface `dziUrlBuilder`/`titleFor`), so the Results surface (Task 8, a Phase-3 follow-up) reuses it unchanged. The cap/over-cap text is pinned by a pure unit (`compare_selection_plan`) and the live flow by Browse Playwright e2e.

## Subsequent work (separate plans / follow-ups)

- **Task 8 (Results follow-up):** copy `timeline.js` into the Results asset folder, mirror the Compare button + `data-compare-cap` + row-header triggers on the Results Timeline tab (TODO-from-Phase-3 ids), add `tests/e2e/gui/test_results_compare_strip.py`, and add the Results FEATURES rows. Verify live against the real reference Results data per spec §16.9.
- **Optional fast-follow (§15.10):** warm the selected cells' DZI pyramids on selection (before the strip opens) to mask the accepted ≤cap concurrent-DZI-build CPU spike. Not in v1.
- **Phase 6 (docs):** include a Compare-strip screenshot in the Browse + Results timeline tutorial captures (per the Task 9 WORKFLOWS decision — folded into the existing flows, no new WORKFLOWS row).

---

## Open Questions

> **Status (2026-06-18 plan review):** OQ-1/2/3/5 were **accepted with the recommended defaults** by the reviewer (already reflected in the tasks above). OQ-4 was **resolved upstream** (the Phase 1 plan now emits `data-row`/`data-row-index` on `--y` headers and `data-col`/`data-col-index` on `--x` headers), so Task 6 matches row-headers by attribute, not `textContent`. These are retained as the decision record; no further human input is needed to start implementation.

1. **(ACCEPTED — plain DOM)** Modal (plain DOM) vs Dash `dbc.Modal`. The controller owns the dynamic viewer set client-side (mirrors `results_viewer.js` owning its viewers outside Dash's vdom), so a `dbc.Modal` + server round-trip per open adds no value and complicates teardown/WebGL-context release. A `dbc.Modal` would give Dash-native focus-trapping/aria for free; revisit only if accessibility becomes a hard requirement.
2. **(ACCEPTED — all three modifiers toggle)** Shift/ctrl/cmd-click each toggle selection (forgiving across OSes; matches the colony view's `data-key` shift-click precedent, spec §15.9). No contiguous range-select — the matrix is 2-D and a "range" is ambiguous; a rectangle-select (anchor + corner) is a possible future add, not v1.
3. **(ACCEPTED — open immediately)** Row-header click opens the strip immediately. Spec §7/D5 says a row-header "compares that row's full time-course," implying open-on-click; the select-then-compare alternative adds a click for no gain here.
4. **(RESOLVED upstream — match by `data-row` attribute)** The Phase 1 plan was amended to put `data-row`/`data-row-index` on the `--y` axis label, so Task 6 matches a row-header to its cells by the `data-row` attribute (robust against whitespace / duplicate-looking labels). The earlier `textContent` fallback and the "file a Phase 1 follow-up" note are withdrawn — it is done.
5. **(ACCEPTED — no new WORKFLOWS row)** Fold the Compare strip into the two existing timeline tutorials (Task 9 Step 2 rationale) — a Compare-strip screenshot is captured inside the existing Browse/Results timeline tutorial captures in Phase 6 — rather than minting a standalone tutorial page + `_capture_compare_strip` (the `workflows-md-gate` round-trip).
