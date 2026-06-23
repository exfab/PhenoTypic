/*
 * timeline.js — focus-and-navigate controller (spec §16) for the timeline
 * matrix. The matrix is NOT scrollable: a no-scroll viewport renders a
 * CENTERED window around exactly one FOCUSED cell, the inner grid shifted by a
 * CSS transform. ←/→/↑/↓ and the four on-edge ◀▶▲▼ buttons move focus
 * (clamped to matrix bounds, no wrap). The focused neighborhood + a margin ring
 * (data-focus-margin) mounts <img>; cells beyond offload (img.remove) to bound
 * decoded-image memory, with data-mount-cap as the absolute LRU ceiling.
 * Enter/Space opens the pop-out for the focused cell. A generation-guarded warm
 * loop pre-fetches thumbnails NEIGHBORHOOD-FIRST (expanding rings from focus).
 * Cells are addressed by [data-row-index][data-col-index]; every grid
 * coordinate (empty or populated) is addressable (spec §16.8).
 *
 * SURFACE-AGNOSTIC: this file is vendored BYTE-FOR-BYTE into both the Browse
 * (`browse/_assets/`) and Results (`results_viewer/_assets/`) Dash apps — which
 * are SEPARATE apps mounted via DispatcherMiddleware (shell/_app.py::compose_hub),
 * each with its own assets_folder, so window.__phenotypicTimeline never collides
 * across surfaces. The controller locates its sibling controls — the four nav
 * buttons, the position readout, the hidden pop-out bridge input — by STABLE
 * CLASS scoped to the enclosing `.timeline-body`, NEVER by a surface-specific
 * id. The ONLY surface-specific input is the container id, passed as the
 * attach(containerId) parameter by each surface's clientside callback. A CI
 * byte-equality guard enforces the two vendored copies never drift.
 */
(function () {
    "use strict";
    const ns = (window.__phenotypicTimeline = window.__phenotypicTimeline || {});
    // ``window.__phenotypicAppPrefix`` is injected by the Dash factory so
    // fetch/OSD URLs survive a path-stripping proxy; default to "/".
    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix : "/";
    ns._generation = ns._generation || 0;
    ns._mounted = ns._mounted || [];          // LRU order of mounted cells
    ns._focus = ns._focus || { rowIndex: 0, colIndex: 0 };
    // Monotonic nonce appended to every bridge write so re-opening the pop-out
    // on the SAME cell still CHANGES the controlled dcc.Input value (otherwise
    // Dash's onChange never fires and the modal — left is_open=true on the
    // server after a close — won't reopen). Deterministic counter, not
    // Date.now/Math.random, so the value is reproducible. The server callback
    // strips the `#<nonce>` suffix before decoding the token (POP-OUT M5).
    ns._popoutNonce = ns._popoutNonce || 0;

    function num(el, attr, dflt) {
        const v = parseFloat(el.getAttribute(attr));
        return Number.isFinite(v) ? v : dflt;
    }

    // --- Surface-agnostic sibling-control lookup ---------------------------
    // The controls live as siblings inside the enclosing `.timeline-body`; find
    // them by stable class scoped to that body so this file is portable across
    // Browse and Results (never a surface-specific id). `body(container)` walks
    // up to the enclosing timeline body (falls back to document if absent).
    function body(container) {
        return (container && container.closest)
            ? (container.closest(".timeline-body") || document) : document;
    }
    function ctrl(container, selector) {
        return body(container).querySelector(selector);
    }
    // Resolve the JS→Dash pop-out bridge to the actual <input> element.
    // dcc.Input puts the `className` (".timeline-popout-bridge") on a WRAPPER
    // div (class "dash-input-container dash-input timeline-popout-bridge") and
    // the real <input id=…> is its child — so a bare class lookup returns the
    // wrapper, on which setting `.value` + dispatching 'input' does nothing.
    // Walk to the contained <input> (or use the node itself if it already is
    // one, e.g. a plain <input class="timeline-popout-bridge">).
    function bridgeInput(container) {
        const el = ctrl(container, ".timeline-popout-bridge");
        if (!el) { return null; }
        if (el.tagName && el.tagName.toUpperCase() === "INPUT") { return el; }
        return el.querySelector("input") || el;
    }

    // --- Grid geometry -----------------------------------------------------
    // Cells (populated AND empty) carry data-row-index/data-col-index. The
    // inner grid carries `.timeline-grid-container`; the no-scroll viewport is
    // its enclosing `.timeline-viewport`. Cells are addressed by coordinate,
    // not DOM order.
    function cellAt(container, r, c) {
        return container.querySelector(
            '.timeline-cell[data-row-index="' + r + '"][data-col-index="' + c + '"]'
        );
    }
    function bounds(container) {
        let maxRow = 0, maxCol = 0;
        container.querySelectorAll("[data-row-index]").forEach(function (el) {
            maxRow = Math.max(maxRow, parseInt(el.getAttribute("data-row-index"), 10) || 0);
            maxCol = Math.max(maxCol, parseInt(el.getAttribute("data-col-index"), 10) || 0);
        });
        return { maxRow: maxRow, maxCol: maxCol };
    }
    function firstPopulatedCell(container) {
        // Smallest row-index, then col-index, among populated (data-src) cells.
        const cells = Array.from(container.querySelectorAll(".timeline-cell[data-src]"));
        let best = null;
        cells.forEach(function (el) {
            const r = parseInt(el.getAttribute("data-row-index"), 10) || 0;
            const c = parseInt(el.getAttribute("data-col-index"), 10) || 0;
            if (!best || r < best.rowIndex || (r === best.rowIndex && c < best.colIndex)) {
                best = { rowIndex: r, colIndex: c };
            }
        });
        return best || { rowIndex: 0, colIndex: 0 };
    }

    // visibleHalfCols/Rows: how many cells fit each side of centre at the
    // current rendered tile size. Read the focused cell's box (incl. CSS gap)
    // and the viewport box; fall back to a small default if unmeasurable.
    function visibleHalf(container, viewport) {
        const sample = container.querySelector(".timeline-cell");
        const vp = viewport.getBoundingClientRect();
        if (!sample) { return { halfCols: 2, halfRows: 2 }; }
        const box = sample.getBoundingClientRect();
        const w = box.width || 1, h = box.height || 1;
        return {
            halfCols: Math.max(1, Math.floor(vp.width / w / 2)),
            halfRows: Math.max(1, Math.floor(vp.height / h / 2)),
        };
    }

    // --- Mount / offload ---------------------------------------------------
    function mount(cell) {
        if (!cell || cell.querySelector("img")) { return; }
        const src = cell.getAttribute("data-src");
        if (!src) { return; }            // empty placeholder — nothing to mount
        const img = document.createElement("img");
        img.src = src;
        img.className = "timeline-cell-img";
        img.loading = "lazy";
        // Fill the (definitively-sized, position:relative) cell box as an
        // absolutely-positioned BACKGROUND layer so layout geometry — which the
        // focus window + off-screen-ring measurements depend on — is stable
        // even before the thumbnail bytes arrive (or if the fetch fails). It is
        // pointer-events:none so it never intercepts clicks on the
        // hover-revealed ⤢ pop-out button stacked above it.
        img.style.position = "absolute";
        img.style.top = "0";
        img.style.left = "0";
        img.style.width = "100%";
        img.style.height = "100%";
        img.style.objectFit = "cover";
        img.style.display = "block";
        img.style.pointerEvents = "none";
        cell.insertBefore(img, cell.firstChild);
        ns._mounted.push(cell);
    }
    function offload(cell) {
        const img = cell && cell.querySelector("img");
        if (img) { img.remove(); }
    }

    // Mount the focus window + margin ring; offload everything outside it.
    function syncWindow(container, viewport, focusMargin, cap) {
        const { halfCols, halfRows } = visibleHalf(container, viewport);
        const colReach = halfCols + focusMargin;
        const rowReach = halfRows + focusMargin;
        const f = ns._focus;
        container.querySelectorAll(".timeline-cell[data-src]").forEach(function (cell) {
            const r = parseInt(cell.getAttribute("data-row-index"), 10) || 0;
            const c = parseInt(cell.getAttribute("data-col-index"), 10) || 0;
            const inWindow = Math.abs(r - f.rowIndex) <= rowReach
                && Math.abs(c - f.colIndex) <= colReach;
            if (inWindow) { mount(cell); } else { offload(cell); }
        });
        // LRU ceiling: never exceed data-mount-cap, even if the window does.
        while (ns._mounted.length > cap) {
            const old = ns._mounted.shift();
            offload(old);
        }
    }

    // Position the inner grid via a CSS transform (no scrollbar —
    // overflow:hidden on the viewport). CLAMP-TRANSLATE (spec §16.1, OQ-1
    // default): translate to centre the focused cell, BUT clamp so the grid
    // never pulls past its own edges — keeps the window full (no empty gutters)
    // including at the default focus (0,0) and at matrix corners; the focused
    // cell then sits off-centre near edges, which the .timeline-cell--focused
    // highlight keeps unambiguous.
    function recenter(container, viewport) {
        const focused = cellAt(container, ns._focus.rowIndex, ns._focus.colIndex);
        if (!focused) { return; }
        const vp = viewport.getBoundingClientRect();
        const gr = container.getBoundingClientRect();
        const fb = focused.getBoundingClientRect();
        // Focused-cell centre relative to the grid's own box. fb/gr both carry
        // the current transform, so (fb - gr) is transform-invariant.
        const cellCenterX = (fb.left + fb.width / 2) - gr.left;
        const cellCenterY = (fb.top + fb.height / 2) - gr.top;
        // Ideal "centre the cell" translation.
        let dx = vp.width / 2 - cellCenterX;
        let dy = vp.height / 2 - cellCenterY;
        // Clamp so the grid edge never crosses into the viewport interior.
        // Allowed dx range: [vp.width - gridWidth, 0] (right edge ≥ viewport
        // right, left edge ≤ viewport left). When the grid is smaller than the
        // viewport, the range collapses and we pin to 0 (top-left aligned).
        const gridWidth = gr.width, gridHeight = gr.height;
        const minDx = Math.min(0, vp.width - gridWidth);
        const minDy = Math.min(0, vp.height - gridHeight);
        dx = Math.max(minDx, Math.min(0, dx));
        dy = Math.max(minDy, Math.min(0, dy));
        container.style.transform = "translate(" + dx + "px, " + dy + "px)";
    }

    function highlight(container) {
        container.querySelectorAll(".timeline-cell--focused").forEach(function (el) {
            el.classList.remove("timeline-cell--focused");
        });
        const focused = cellAt(container, ns._focus.rowIndex, ns._focus.colIndex);
        if (focused) { focused.classList.add("timeline-cell--focused"); }
    }

    function updateReadout(container) {
        const readout = ctrl(container, ".timeline-position");
        if (!readout) { return; }
        const b = bounds(container);
        const focused = cellAt(container, ns._focus.rowIndex, ns._focus.colIndex);
        const rowVal = focused ? (focused.getAttribute("data-row") || "") : "";
        const colVal = focused ? (focused.getAttribute("data-col") || "") : "";
        readout.textContent =
            "row " + (ns._focus.rowIndex + 1) + "/" + (b.maxRow + 1)
            + " · time " + (ns._focus.colIndex + 1) + "/" + (b.maxCol + 1)
            + (rowVal || colVal ? "  (" + rowVal + " · " + colVal + ")" : "");
    }

    function toggleEdgeButtons(container) {
        const b = bounds(container);
        const f = ns._focus;
        const set = function (cls, disabled) {
            const el = ctrl(container, cls);   // by class, scoped to the body
            if (el) { el.disabled = !!disabled; }
        };
        set(".timeline-nav-up", f.rowIndex <= 0);
        set(".timeline-nav-down", f.rowIndex >= b.maxRow);
        set(".timeline-nav-left", f.colIndex <= 0);
        set(".timeline-nav-right", f.colIndex >= b.maxCol);
    }

    // One place that re-renders everything after a focus change.
    function applyFocus(container, viewport, focusMargin, cap) {
        syncWindow(container, viewport, focusMargin, cap);
        recenter(container, viewport);
        highlight(container);
        updateReadout(container);
        toggleEdgeButtons(container);
    }

    function moveFocus(container, viewport, focusMargin, cap, dRow, dCol) {
        const b = bounds(container);
        const f = ns._focus;
        f.rowIndex = Math.min(b.maxRow, Math.max(0, f.rowIndex + dRow));   // clamp
        f.colIndex = Math.min(b.maxCol, Math.max(0, f.colIndex + dCol));   // no wrap
        applyFocus(container, viewport, focusMargin, cap);
    }

    // Enter/Space → open the pop-out for the focused cell, using the SAME hidden
    // bridge input the ⤢ click uses (Task 9): set the bridge `.value` to the
    // focused cell's data-ref + dispatch an 'input' event. The bridge is found
    // by class (`.timeline-popout-bridge`, scoped to the body), so this path is
    // identical on Browse + Results.
    function openFocusedPopout(container) {
        const focused = cellAt(container, ns._focus.rowIndex, ns._focus.colIndex);
        if (!focused) { return; }
        const ref = focused.getAttribute("data-ref");
        const bridge = bridgeInput(container);
        if (!ref || !bridge) { return; }
        setBridge(bridge, ref);
    }

    // Set the JS→Dash bridge input's value and fire the React-aware event so
    // Dash's controlled <input> registers the change. React overrides the
    // value setter on the input element prototype; calling the *native*
    // descriptor setter before dispatching 'input' is the standard way to make
    // a programmatic value change trip React's onChange (Dash's dcc.Input).
    //
    // Appends `#<monotonic-nonce>` so RE-OPENING on the same cell still changes
    // the value (Dash onChange only fires on a value change). `#` is outside
    // the base64url token alphabet (A-Za-z0-9-_), so the server callback
    // splits it off cleanly before decoding (POP-OUT M5). Surface-agnostic:
    // both Browse and Results decode the same `<token>#<nonce>` shape.
    function setBridge(bridge, value) {
        ns._popoutNonce += 1;
        const stamped = value + "#" + ns._popoutNonce;
        try {
            const proto = window.HTMLInputElement && window.HTMLInputElement.prototype;
            const desc = proto && Object.getOwnPropertyDescriptor(proto, "value");
            if (desc && desc.set) {
                desc.set.call(bridge, stamped);
            } else {
                bridge.value = stamped;
            }
        } catch (e) {
            bridge.value = stamped;
        }
        bridge.dispatchEvent(new Event("input", { bubbles: true }));
    }

    // --- Background warm — neighborhood-first (expanding rings from focus) ---
    function warm(container, viewport, generation) {
        const concurrency = num(container, "data-warm-concurrency", 2);
        const b = bounds(container);
        const f = ns._focus;
        // Order populated cells by Chebyshev distance from the focus, so the
        // cells the user is most likely to reach next warm first (spec §16.3).
        const cells = Array.from(container.querySelectorAll(".timeline-cell[data-src]"));
        cells.sort(function (a, e) {
            const da = Math.max(
                Math.abs((parseInt(a.getAttribute("data-row-index"), 10) || 0) - f.rowIndex),
                Math.abs((parseInt(a.getAttribute("data-col-index"), 10) || 0) - f.colIndex)
            );
            const de = Math.max(
                Math.abs((parseInt(e.getAttribute("data-row-index"), 10) || 0) - f.rowIndex),
                Math.abs((parseInt(e.getAttribute("data-col-index"), 10) || 0) - f.colIndex)
            );
            return da - de;
        });
        void b;
        let i = 0;
        function pump() {
            if (generation !== ns._generation) { return; }   // matrix rebuilt → abort
            while (i < cells.length) {
                const src = cells[i++].getAttribute("data-src");
                if (!src) { continue; }
                fetch(src, { credentials: "same-origin" })
                    .catch(function () {})
                    .then(function () { if (generation === ns._generation) pump(); });
                return;
            }
        }
        for (let k = 0; k < concurrency; k++) { pump(); }
    }

    // --- Attach ------------------------------------------------------------
    ns.attach = function (containerId) {
        const container = document.getElementById(containerId);
        if (!container) { return; }
        // Surface-agnostic: resolve the no-scroll viewport by stable class.
        const viewport = container.closest(".timeline-viewport") || container.parentNode;
        // First-paint resilience (OQ-2/W2): the timeline body starts
        // display:none, so on the first attach getBoundingClientRect() can read
        // 0 (window mis-sizes, transform clamps to nothing). Re-schedule via
        // requestAnimationFrame until the viewport has a non-zero width, so the
        // controller self-corrects regardless of whether the toggle callback
        // fired attach before or after the body was shown. Cap the retries so a
        // permanently-hidden body (single-view mode) never spins an unbounded
        // rAF loop; an explicit attach (toggle/render clientside callback) once
        // the body is shown restarts the count.
        if (viewport.getBoundingClientRect().width === 0) {
            const tries = (container._tlWaitTries || 0) + 1;
            container._tlWaitTries = tries;
            if (tries <= 240) {   // ~4s at 60fps — plenty for the body to show
                window.requestAnimationFrame(function () { ns.attach(containerId); });
            }
            return;
        }
        container._tlWaitTries = 0;
        // Mark this node as attached so the <body> re-attach observer below
        // only re-fires for a FRESH node Dash inserts (a re-render), never for
        // our own transform/highlight mutations on the SAME node — otherwise
        // recenter()'s style write would feed back into the observer and spin.
        container.dataset.tlAttached = "1";
        ns._generation += 1;
        const generation = ns._generation;
        const cap = num(container, "data-mount-cap", 400);
        const focusMargin = num(container, "data-focus-margin", 2);
        ns._mounted = [];
        ns._focus = firstPopulatedCell(container);   // start at first populated cell

        applyFocus(container, viewport, focusMargin, cap);

        // Keyboard: scoped to the viewport, but IGNORED when a text input /
        // select / textarea holds focus (so typing a pattern never navigates).
        if (!viewport.dataset.tlKeysBound) {
            viewport.dataset.tlKeysBound = "1";
            viewport.addEventListener("keydown", function (ev) {
                const ae = document.activeElement;
                const tag = ae && ae.tagName ? ae.tagName.toUpperCase() : "";
                if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") { return; }
                const cur = document.getElementById(containerId);
                if (!cur) { return; }
                if (ev.key === "ArrowLeft") { moveFocus(cur, viewport, focusMargin, cap, 0, -1); ev.preventDefault(); }
                else if (ev.key === "ArrowRight") { moveFocus(cur, viewport, focusMargin, cap, 0, 1); ev.preventDefault(); }
                else if (ev.key === "ArrowUp") { moveFocus(cur, viewport, focusMargin, cap, -1, 0); ev.preventDefault(); }
                else if (ev.key === "ArrowDown") { moveFocus(cur, viewport, focusMargin, cap, 1, 0); ev.preventDefault(); }
                else if (ev.key === "Enter" || ev.key === " ") { openFocusedPopout(cur); ev.preventDefault(); }
            });
        }

        // Edge buttons — found by stable class scoped to the body (idempotent
        // bind via a dataset flag on each button), so the binding is portable
        // across surfaces.
        const bindNav = function (cls, dRow, dCol) {
            const btn = ctrl(container, cls);
            if (!btn || btn.dataset.tlNavBound === "1") { return; }
            btn.dataset.tlNavBound = "1";
            btn.addEventListener("click", function () {
                const cur = document.getElementById(containerId);
                if (cur) { moveFocus(cur, viewport, focusMargin, cap, dRow, dCol); }
            });
        };
        bindNav(".timeline-nav-up", -1, 0);
        bindNav(".timeline-nav-down", 1, 0);
        bindNav(".timeline-nav-left", 0, -1);
        bindNav(".timeline-nav-right", 0, 1);

        // Hover-revealed ⤢ click bridge (shares the same hidden input as
        // Enter/Space — see Task 9). Delegated, idempotent.
        if (!container.dataset.tlPopoutBound) {
            container.dataset.tlPopoutBound = "1";
            container.addEventListener("click", function (ev) {
                const btn = ev.target && ev.target.closest
                    ? ev.target.closest(".timeline-cell-popout") : null;
                if (!btn) { return; }
                const cell = btn.closest(".timeline-cell");
                const ref = cell && cell.getAttribute("data-ref");
                const bridge = bridgeInput(container);
                if (ref && bridge) {
                    setBridge(bridge, ref);
                }
            });
        }

        bindCompareTriggers(container);

        warm(container, viewport, generation);
    };

    // ====================================================================
    // Compare-strip SELECTION + TRIGGERS (spec §7, §15.10). Bound inside
    // attach(); surface-agnostic.
    //
    // SELECTION IS THE DOM (M4): the `.timeline-cell--selected` class is the
    // single source of truth — there is NO parallel JS Set. The Phase 2 render
    // callback replaces cell children on each (Y/time/pattern/source) change,
    // which clears the class, so a separate Set would persist stale refs and
    // silently desync from the rendered grid. By deriving the selection from
    // the live DOM, a re-render naturally resets it (re-render-safe + simpler).
    // Because the class lives on THIS surface's own cells, two distinct
    // containers (Browse vs a future Results) never share a selection — it is
    // per-surface by construction, with no per-container bookkeeping object.
    // ====================================================================

    // Build the surface's DZI URL from a cell ref. Surface-agnostic: a Browse
    // ref is an opaque base64url token (no "/") → encode the whole token; a
    // Results ref is "<dataset>/<stem>" → encode EACH segment and join with "/"
    // (NEVER encodeURIComponent the whole thing — that would percent-encode the
    // "/" separator). Mirrors browse.js (`/tiles/<token>.dzi`) and
    // results_viewer.js:297-298 (`/tiles/<dataset>/<stem>.dzi`).
    function timelineDziUrl(ref) {
        const r = String(ref);
        const path = r.indexOf("/") === -1
            ? encodeURIComponent(r)
            : r.split("/").map(encodeURIComponent).join("/");
        return appPrefix + "tiles/" + path + ".dzi";
    }

    // SINGLE place the cap literal lives in JS (C1): read data-compare-cap off
    // the container (written from the Python TIMELINE_COMPARE_CAP in the
    // layout), with one fallback. openCompareStrip trusts this cap and never
    // re-fabricates it.
    function compareCap(container) { return num(container, "data-compare-cap", 12); }

    // Selected refs in GRID order (stable): sort selected cells by
    // (rowIndex, colIndex), reading the LIVE DOM each time (M4 — no Set).
    function selectionRefsInGridOrder(container) {
        return Array.from(
            container.querySelectorAll(".timeline-cell.timeline-cell--selected[data-ref]")
        ).sort(function (a, b) {
            const ra = parseInt(a.getAttribute("data-row-index"), 10) || 0;
            const rb = parseInt(b.getAttribute("data-row-index"), 10) || 0;
            if (ra !== rb) { return ra - rb; }
            return (parseInt(a.getAttribute("data-col-index"), 10) || 0)
                 - (parseInt(b.getAttribute("data-col-index"), 10) || 0);
        }).map(function (el) { return el.getAttribute("data-ref"); });
    }

    function openCompareFor(container, refs) {
        if (!refs.length || !ns.openCompareStrip) { return; }
        ns.openCompareStrip(refs, {
            dziUrlBuilder: timelineDziUrl,
            titleFor: function (r) { return String(r); },
            cap: compareCap(container),
        });
    }

    function bindCompareTriggers(container) {
        // (a) shift/ctrl/cmd-click a tile toggles selection (does NOT open the
        //     pop-out: the Phase 2 pop-out handler fires only on a plain ⤢
        //     click, and this branch only acts on a modified click). The class
        //     IS the state — no Set to keep in sync.
        if (!container.dataset.tlSelectBound) {
            container.dataset.tlSelectBound = "1";
            container.addEventListener("click", function (ev) {
                if (!(ev.shiftKey || ev.ctrlKey || ev.metaKey)) { return; }
                const cell = ev.target && ev.target.closest
                    ? ev.target.closest(".timeline-cell[data-ref]") : null;
                if (!cell) { return; }
                ev.preventDefault();
                cell.classList.toggle("timeline-cell--selected");
            });
        }

        // (b) row-header click → select that whole row's populated cells + open
        //     the strip immediately. Match cells by the header's data-row
        //     ATTRIBUTE (Phase 1 emits data-row on the --y label), NOT
        //     textContent — robust against whitespace / duplicate-looking
        //     labels. Delegated on the container, idempotent.
        if (!container.dataset.tlHeaderBound) {
            container.dataset.tlHeaderBound = "1";
            container.addEventListener("click", function (ev) {
                const header = ev.target && ev.target.closest
                    ? ev.target.closest(".timeline-axis-label--y") : null;
                if (!header) { return; }
                const rowValue = header.getAttribute("data-row");
                if (rowValue === null) { return; }
                const refs = Array.from(
                    container.querySelectorAll(".timeline-cell[data-src][data-row][data-ref]")
                ).filter(function (c) {
                    return c.getAttribute("data-row") === rowValue;
                }).map(function (c) { return c.getAttribute("data-ref"); });
                openCompareFor(container, refs);
            });
        }

        // (c) "Compare selected" button → open the strip for the current
        //     selection. Found by the SURFACE-AGNOSTIC class
        //     `.timeline-compare-btn` (scoped to the body), never a
        //     surface-specific id, so the vendored controller wires Browse +
        //     Results identically. Idempotent via a dataset flag.
        const compareBtn = ctrl(container, ".timeline-compare-btn");
        if (compareBtn && compareBtn.dataset.tlCompareBound !== "1") {
            compareBtn.dataset.tlCompareBound = "1";
            compareBtn.addEventListener("click", function () {
                const cur = document.getElementById(container.id);
                if (!cur) { return; }
                openCompareFor(cur, selectionRefsInGridOrder(cur));
            });
        }
    }

    // Cancel any in-flight background warm (W4): the view-mode toggle calls this
    // when switching AWAY from Timeline so an in-flight neighborhood-first warm
    // stops asking the server to render thumbnails the user no longer sees.
    // Bumping the generation makes the running pump() loops bail (they guard on
    // `generation !== ns._generation`). A later attach() bumps again and
    // restarts warm, so this is safe to call repeatedly.
    ns.cancelWarm = function () {
        ns._generation += 1;
    };

    // Re-attach when Dash replaces the container (tab/body re-render), mirroring
    // results_viewer.js: poll-until-present + a <body> MutationObserver, both
    // idempotent (the dataset flags above make a re-attach cheap). spec §15.7.
    // Surface-agnostic: discover the grid container by the stable class
    // `.timeline-grid-container` and re-attach using its OWN id — so the
    // byte-identical vendored copy finds Browse's `browse-tl-grid` and Results'
    // `timeline-grid` (or whatever id that surface assigns) without hardcoding.
    function startReattachObserver() {
        if (!document.body || ns._reattachBound) { return; }
        ns._reattachBound = true;
        const obs = new MutationObserver(function () {
            const grid = document.querySelector(".timeline-grid-container");
            // Only (re)attach to a grid node that has NOT yet been attached —
            // a fresh node Dash inserted on a re-render. Our own
            // transform/highlight mutations keep dataset.tlAttached set on the
            // SAME node, so they never re-trigger attach (no feedback loop).
            if (grid && grid.id && grid.dataset.tlAttached !== "1") {
                ns.attach(grid.id);
            }
        });
        obs.observe(document.body, { childList: true, subtree: true });
    }
    startReattachObserver();
})();

/* ============================================================
 * Synced "Compare" strip (spec §7, §15.10). Surface-agnostic: the caller
 * supplies dziUrlBuilder(ref) + titleFor(ref); this owns the modal, the ≤cap
 * OSD viewers, the shared viewport (feedback-guarded), the over-cap notice, and
 * full teardown on close. Mirrors results_viewer.js lock-views (lines 165-216).
 *
 * CONTROLLER STATE is a process-global SINGLETON: at most ONE strip is open at
 * a time, on ONE surface. Browse and Results live on SEPARATE Flask servers /
 * page loads (DispatcherMiddleware mounts) and never co-mount, so a single
 * global strip is correct, not a limitation. (The thing that is PER-SURFACE is
 * the SELECTION set — the `.timeline-cell--selected` class on the surface's own
 * cells, bound in attach() above; the controller state below is deliberately
 * global and must NOT be generalized to per-container.)
 * ============================================================ */
(function () {
    "use strict";
    const ns = (window.__phenotypicTimeline = window.__phenotypicTimeline || {});
    // ``window.__phenotypicAppPrefix`` is injected by the Dash factory so
    // fetch/OSD URLs survive a path-stripping proxy; default to "/".
    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix : "/";

    // Reuse the surface's OSD loader if one already exists; else lazy-inject the
    // VENDORED copy (NO CDN). Both browse.js and results_viewer.js vendor OSD
    // under `<appPrefix>assets/openseadragon/`.
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

    // CONTROLLER STATE — deliberately process-global (see header): one strip,
    // one surface.
    let _compareViewers = [];      // OSD viewers currently mounted in the strip
    let _broadcasting = false;     // feedback-loop guard while applying a peer viewport
    // COMMITTED TEST SEAM: the mounted viewer list is mirrored onto the
    // namespace so the Playwright e2e can drive vs[0]/vs[1] and assert viewport
    // sync. Kept in lock-step with _compareViewers in openCompareStrip +
    // teardownCompare. This is an intentional public seam, not incidental.
    ns.__compareViewers = ns.__compareViewers || [];

    // Mirror the focused source viewer's viewport onto every peer. The guard
    // makes the peer-applied "animation" events no-ops, so a pan/zoom in one
    // viewer does not bounce back through the others (results_viewer.js:167-181).
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
        // added via addHandler (incl. the "animation" sync handler), so there
        // is no handler leak — destroy() is the complete teardown.
        _compareViewers.forEach(function (v) {
            try { v.destroy(); } catch (e) { /* already torn down */ }
        });
        _compareViewers = [];
        ns.__compareViewers = [];                // keep the seam in lock-step
        const modal = document.getElementById("timeline-compare-modal");
        if (modal) { modal.remove(); }           // release the strip DOM + canvases
    }
    ns.closeCompareStrip = teardownCompare;

    // --- Modal DOM contract — both this controller AND the e2e depend on these
    //     EXACT ids/classes/nesting; keep them in sync:
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

    // refs: array (selection order). opts: { dziUrlBuilder, titleFor, cap }.
    // The cap is the caller's responsibility (the trigger reads
    // data-compare-cap); openCompareStrip trusts opts.cap and does NOT
    // re-fabricate a fallback number — the cap flows from the Python
    // TIMELINE_COMPARE_CAP → DOM → here.
    ns.openCompareStrip = async function (refs, opts) {
        await ns.osdReady;
        teardownCompare();                       // never stack strips
        const cap = opts.cap;
        const list = Array.isArray(refs) ? refs : [];
        const shown = list.slice(0, cap);
        const overCap = list.length > cap;

        const modal = buildCompareModal();        // sets id=timeline-compare-modal
        if (overCap) {
            // EXACT mirror of compare_selection_plan(...).notice, including the
            // em-dash "—". The Python unit test pins this wording.
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
        ns.__compareViewers = _compareViewers;   // refresh the seam each open
        document.body.appendChild(modal);
        return _compareViewers.length;
    };
})();
