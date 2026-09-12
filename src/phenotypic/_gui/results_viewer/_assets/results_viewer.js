/*
 * results_viewer.js
 * -------------------------------------------------------------------------
 * Client-side lifecycle layer for the PhenoTypic Results Viewer.
 *
 * Responsibilities:
 *   1. Maintain a registry of mounted Plate stages keyed by their host divId.
 *   2. Drive `window.phenotypicViv` -- the facade over the vendored Viv +
 *      deck.gl bundle -- from the per-card source spec and display state
 *      Dash hands over.
 *   3. Implement an opt-in "lock views" mode that mirrors one stage's
 *      viewState onto its peers.
 *   4. Watch the cards container for DOM removals and destroy orphaned
 *      stages automatically (cards come and go via Dash callbacks).
 *   5. Bridge shift-click tile selection into a Dash store (section G).
 *   6. Drive data-attribute-declared drag-splitters (section H).
 *
 * OpenSeadragon is gone from this file. The Plate surface reads OME-Zarr
 * chunks directly in the browser over the `/zarr/...` byte route, so there
 * is no server-rendered DZI pyramid to point a tile viewer at. Browse and
 * the builder's point picker keep OSD and their own `_dzi_tiler` path.
 *
 * The Dash app loads this file because `assets_folder="_assets"` is set in
 * `_app.py`, so it is served at `/assets/results_viewer.js`.
 *
 * Public surface (under `window.__phenotypicResultsViewer`):
 *   - stages                     : Map<divId, stage record>.
 *   - mountStage(record)         : (re)source a stage from one card record.
 *   - disposeStage(divId)        : tear down and forget a stage.
 *   - setLockViews(active)       : toggle mirrored pan/zoom.
 *   - applyPlateSources(records) : helper invoked from clientside callbacks.
 *   - onElementsMounted(sel, fn) : attach to matching elements now and as
 *                                  Dash mounts more of them (section F).
 *   - clampSidebarWidth(px, min, max)
 *                              : the splitter's width clamp; `min`/`max`
 *                                default to the module's own bounds
 *                                (section H).
 * -------------------------------------------------------------------------
 */

/* ============================================================
 * (A) Plate stage registry, driven through the Viv facade.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer = window.__phenotypicResultsViewer || {};
    ns.stages = ns.stages || new Map();   // divId -> stage record
    ns.stageLoadEpochs = ns.stageLoadEpochs || new Map();
    ns.stageLoadQueues = ns.stageLoadQueues || new Map();

    /** Facade layer ids; mirrored in `_viewer_card.py`. */
    const IMAGE_LAYER = "image";
    const LABEL_LAYER = "labels";

    /** Separators for the readout, kept out of source as escapes. */
    const DOT = "·";
    const TIMES = "×";
    const SQUARED = "²";

    function facade() {
        return window.phenotypicViv || null;
    }

    /**
     * Identity of one (store generation, displayed series, label) triple.
     *
     * `setSource` re-opens the store and rebuilds every layer, so it must
     * fire only when one of those actually changed. Opacity and label
     * visibility change far more often -- a slider drag is a stream of
     * them -- and are applied without re-sourcing.
     */
    function sourceSignature(spec, display) {
        if (!spec) return null;
        const series = (display && display.seriesPath) || spec.seriesPath;
        return [spec.storeUrl, series, spec.labelPath || ""].join(" ");
    }

    function writeText(elementId, text) {
        const el = document.getElementById(elementId);
        if (el) el.textContent = text;
    }

    /** Render the served-level readout from what the facade reported. */
    function renderLevel(elementId, info) {
        if (!info) {
            writeText(elementId, "no image");
            return;
        }
        const chunk = info.tileSize
            ? " " + DOT + " " + info.tileSize + SQUARED + " chunks"
            : "";
        writeText(
            elementId,
            "pyramid level " + info.level + " of " + info.levels + " " +
            DOT + " " + info.width + TIMES + info.height + chunk
        );
    }

    function renderZoom(elementId, viewState) {
        if (!viewState || typeof viewState.zoom !== "number") {
            writeText(elementId, "");
            return;
        }
        writeText(elementId, Math.round(Math.pow(2, viewState.zoom) * 100) + "%");
    }

    /**
     * (Re)source one card's stage from its record.
     *
     * @param {{id: string, levelReadoutId: string, zoomReadoutId: string,
     *          spec: object|null, display: object|null}} record
     */
    async function mountStageNow(record, epoch) {
        const viv = facade();
        if (!viv) {
            console.error("[results_viewer] window.phenotypicViv is missing");
            return null;
        }
        const divId = record.id;
        const isCurrent = function () {
            return ns.stageLoadEpochs.get(divId) === epoch;
        };
        const el = document.getElementById(divId);
        if (!el) {
            console.warn("[results_viewer] mountStage: no element", divId);
            return null;
        }
        let entry = ns.stages.get(divId);
        if (!entry) {
            await viv.mount(divId, {
                onLevelChange: function (info) {
                    renderLevel(record.levelReadoutId, info);
                },
                onViewStateChange: function (viewState) {
                    renderZoom(record.zoomReadoutId, viewState);
                    if (ns.lockViewsActive) {
                        ns._broadcastViewState(divId, viewState);
                    }
                },
                // Re-fetching a spec after a re-promote needs a server
                // round-trip Dash owns, so recovery is a Refresh rather
                // than a silent re-source. The read still throws
                // `StaleGenerationError`, which is visible in the console.
                refetchSource: null
            });
            if (!isCurrent()) return null;
            entry = {signature: null, opacity: {}, labelVisible: true};
            ns.stages.set(divId, entry);
        }
        if (!isCurrent()) return null;

        const display = record.display || {};
        const signature = sourceSignature(record.spec, display);
        if (signature !== entry.signature) {
            const spec = Object.assign({}, record.spec);
            if (display.seriesPath) spec.seriesPath = display.seriesPath;
            renderLevel(record.levelReadoutId, null);
            try {
                const loaded = await viv.setSource(divId, spec);
                if (loaded === undefined || !isCurrent()) return null;
            } catch (err) {
                if (!isCurrent()) return null;
                console.error("[results_viewer] setSource failed", divId, err);
                writeText(record.levelReadoutId, String(err.message || err));
                return null;
            }
            entry.signature = signature;
            // A fresh source rebuilds every layer, so the display state has
            // to be re-applied rather than assumed to have survived.
            entry.opacity = {};
            entry.labelVisible = true;
        }
        if (!isCurrent()) return null;
        entry.record = record;

        const opacity = display.opacity || {};
        [IMAGE_LAYER, LABEL_LAYER].forEach(function (layer) {
            const value = opacity[layer];
            if (typeof value === "number" && entry.opacity[layer] !== value) {
                entry.opacity[layer] = value;
                viv.setLayerOpacity(divId, layer, value);
            }
        });
        const labelVisible = display.labelVisible !== false;
        if (labelVisible !== entry.labelVisible) {
            entry.labelVisible = labelVisible;
            viv.setLayerVisibility(divId, LABEL_LAYER, labelVisible);
        }
        return entry;
    }

    ns.mountStage = function (record) {
        const divId = record.id;
        const epoch = (ns.stageLoadEpochs.get(divId) || 0) + 1;
        ns.stageLoadEpochs.set(divId, epoch);
        return mountStageNow(record, epoch);
    };

    /** Destroy the stage (if any) registered under the given div id. */
    ns.disposeStage = function (divId) {
        if (!ns.stages.has(divId)) return;
        const viv = facade();
        try { if (viv) viv.destroy(divId); }
        catch (e) { console.error(e); }
        ns.stages.delete(divId);
        ns.stageLoadEpochs.set(divId, (ns.stageLoadEpochs.get(divId) || 0) + 1);
    };
})();

/* ============================================================
 * (B) Lock-views: mirrored pan/zoom across all mounted stages.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer;
    ns.lockViewsActive = false;
    let _broadcasting = false;       // re-entrancy guard

    /**
     * Mirror one stage's viewState onto every peer.
     *
     * deck.gl re-enters `onViewStateChange` for each viewer pushed into, so
     * the guard is load-bearing rather than defensive: without it the first
     * pan recurses until the stack gives out.
     */
    ns._broadcastViewState = function (srcDivId, viewState) {
        if (_broadcasting) return;
        const viv = window.phenotypicViv;
        if (!viv) return;
        _broadcasting = true;
        try {
            ns.stages.forEach(function (_entry, divId) {
                if (divId === srcDivId) return;
                viv.setViewState(divId, viewState);
            });
        } finally {
            _broadcasting = false;
        }
    };

    /**
     * Toggle lock-views mode. The Python clientside callback in
     * `_callbacks.py` invokes this on STORE_LOCK_VIEWS changes.
     */
    ns.setLockViews = function (active) {
        ns.lockViewsActive = !!active;
        return ns.lockViewsActive;
    };
})();

/* ============================================================
 * (C) MutationObserver: destroy stages when their card is removed.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer;

    function findStages(node) {
        if (!node) return [];
        const out = [];
        const isStage = function (el) {
            return el.classList && el.classList.contains("plate-stage__canvas");
        };
        if (node.nodeType === 1 && isStage(node)) out.push(node);
        if (node.querySelectorAll) {
            node.querySelectorAll(".plate-stage__canvas").forEach(function (el) {
                out.push(el);
            });
        }
        return out;
    }

    function startObserver() {
        const container = document.getElementById("cards-container");
        if (!container) return false;
        const obs = new MutationObserver(function (mutations) {
            mutations.forEach(function (m) {
                m.removedNodes.forEach(function (n) {
                    findStages(n).forEach(function (canvas) {
                        if (canvas.id) ns.disposeStage(canvas.id);
                    });
                });
            });
        });
        obs.observe(container, { childList: true, subtree: true });
        ns._cardsObserver = obs;
        console.info("[results_viewer] cards-container observer attached");
        return true;
    }

    // Dash mounts the layout asynchronously; poll until the container
    // exists, then attach the observer.
    if (!startObserver()) {
        const interval = setInterval(function () {
            if (startObserver()) clearInterval(interval);
        }, 100);
    }
})();

/* ============================================================
 * (D) Helper invoked by the Python clientside callbacks.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer;

    /**
     * Apply a batch of source specs to mounted Plate stages.
     *
     * Each entry describes one card:
     *   { id, levelReadoutId, zoomReadoutId, spec, display }
     * A null `spec` disposes that card's stage -- the card is cleared, its
     * image has no store, or the store could not be read.
     *
     * The spec arrives exactly as `build_source_spec` produced it and is
     * handed to the facade unmodified apart from the displayed series the
     * Layers panel selected.
     */
    ns.applyPlateSources = async function (records) {
        if (!Array.isArray(records)) return null;
        await Promise.all(records.map(async function (record) {
            if (!record || !record.id) return;
            if (!record.spec) {
                ns.disposeStage(record.id);
                return;
            }
            await ns.mountStage(record);
        }));
        return null;
    };
})();

/* ============================================================
 * (E) Colony Viv mount + browser-driven virtualization focus.
 * ============================================================ */
(function () {
    "use strict";
    const STAGE_ID = "colony-viv-grid-stage";
    const FOCUS_STORE_ID = "store-colony-grid-focus";
    let mountedElement = null;
    let mountTimer = null;
    let focusTimer = null;
    let mountEpoch = 0;
    let lastFocus = null;
    let mountedSignature = "";
    let geometryFrame = null;
    let resizeObserver = null;
    let cameraState = null;

    const CAMERA_COMMANDS = {
        "colony-camera-pan-up": {action: "pan", dx: 0, dy: -1},
        "colony-camera-pan-down": {action: "pan", dx: 0, dy: 1},
        "colony-camera-pan-left": {action: "pan", dx: -1, dy: 0},
        "colony-camera-pan-right": {action: "pan", dx: 1, dy: 0},
        "colony-camera-center": {action: "center"},
        "colony-camera-fit": {action: "fit"},
        "colony-camera-zoom-out": {action: "zoom", delta: -0.5},
        "colony-camera-zoom-in": {action: "zoom", delta: 0.5},
        "colony-camera-one-to-one": {action: "oneToOne"}
    };

    function cropSize(stage) {
        const value = Number(stage && stage.dataset.colonyCropSize);
        return value > 0 ? value : 64;
    }

    function cellSignature(stage) {
        if (!stage) return "";
        const cells = Array.from(document.querySelectorAll("[data-colony-viv-cell]"));
        const sources = cells.map(function (cell) {
            return cell.dataset.colonyVivCell || "";
        }).sort();
        return [cropSize(stage)].concat(sources).join("|");
    }

    function cellRecords(stage) {
        const stageRect = stage.getBoundingClientRect();
        const records = [];
        document.querySelectorAll("[data-colony-viv-cell]").forEach(function (cell) {
            let record;
            try { record = JSON.parse(cell.dataset.colonyVivCell); }
            catch (_err) { return; }
            const frame = cell.querySelector(".colony-cell-frame") || cell;
            const rect = frame.getBoundingClientRect();
            record.x = rect.left - stageRect.left;
            record.y = rect.top - stageRect.top;
            record.width = rect.width;
            record.height = rect.height;
            records.push(record);
        });
        return records;
    }

    function syncStageClip(stage) {
        const container = document.getElementById("colony-grid-container");
        const root = stage.closest(".colony-view-root");
        if (!container || !root) return;
        const containerRect = container.getBoundingClientRect();
        const rootRect = root.getBoundingClientRect();
        // A fixed canvas does not inherit the Colony root's overflow clip.
        // Recreate that clip explicitly by intersecting the scrolling grid
        // with the visible tab panel. Otherwise a grid whose top has scrolled
        // above the viewport can paint OME-Zarr tiles over the header/tabs.
        const rect = {
            top: Math.max(containerRect.top, rootRect.top),
            right: Math.min(containerRect.right, rootRect.right),
            bottom: Math.min(containerRect.bottom, rootRect.bottom),
            left: Math.max(containerRect.left, rootRect.left)
        };
        const width = stage.clientWidth;
        const height = stage.clientHeight;
        const top = Math.max(0, Math.min(height, rect.top));
        const left = Math.max(0, Math.min(width, rect.left));
        const right = Math.max(0, width - Math.max(0, Math.min(width, rect.right)));
        const bottom = Math.max(
            0, height - Math.max(0, Math.min(height, rect.bottom))
        );
        const clip = `inset(${top}px ${right}px ${bottom}px ${left}px)`;
        if (stage.style.clipPath !== clip) stage.style.clipPath = clip;
    }

    function focusFirstPopulatedColumn() {
        const container = document.getElementById("colony-grid-container");
        if (!container || container.dataset.colonyInitialFocus === "1") return;
        container.dataset.colonyInitialFocus = "1";
        if (container.scrollLeft !== 0) return;
        const first = container.querySelector("[data-colony-viv-cell]");
        if (!first) return;
        const frame = first.querySelector(".colony-cell-frame") || first;
        const containerRect = container.getBoundingClientRect();
        const frameRect = frame.getBoundingClientRect();
        const target = container.scrollLeft + frameRect.left - containerRect.left -
            Math.max(0, (container.clientWidth - frameRect.width) / 2);
        container.scrollLeft = Math.max(0, target);
    }

    async function mountGrid() {
        const epoch = ++mountEpoch;
        const stage = document.getElementById(STAGE_ID);
        const viv = window.phenotypicViv;
        if (!stage || !viv) return;
        syncStageClip(stage);
        if (mountedElement !== stage) {
            if (mountedElement) {
                try { viv.destroy(STAGE_ID); } catch (_err) { /* already gone */ }
            }
            mountedElement = stage;
            await viv.mount(STAGE_ID, {});
        }
        const cells = cellRecords(stage);
        if (!cells.length || epoch !== mountEpoch) return;
        try {
            const count = await viv.setGridSources(
                STAGE_ID,
                cells,
                cameraState ? {
                    zoomOffset: cameraState.zoomOffset,
                    offsetX: cameraState.offsetX,
                    offsetY: cameraState.offsetY
                } : {zoomOffset: 0, offsetX: 0, offsetY: 0},
                {cropSize: cropSize(stage)}
            );
            if (epoch === mountEpoch && count > 0) {
                mountedSignature = cellSignature(stage);
                const surface = stage.closest(".colony-grid-viv-surface");
                if (surface) surface.classList.add("colony-grid-viv-active");
                observeGridGeometry(stage);
                updateCameraToolbar(await viv.getGridCameraState(STAGE_ID));
                focusFirstPopulatedColumn();
            }
        } catch (err) {
            console.error("[results_viewer] Colony Viv mount failed", err);
        }
    }

    function observeGridGeometry(stage) {
        if (resizeObserver) resizeObserver.disconnect();
        if (!window.ResizeObserver) return;
        resizeObserver = new ResizeObserver(scheduleGeometrySync);
        resizeObserver.observe(stage);
        document.querySelectorAll("[data-colony-viv-cell]").forEach(function (cell) {
            resizeObserver.observe(cell.querySelector(".colony-cell-frame") || cell);
        });
    }

    async function syncColonyViewports() {
        geometryFrame = null;
        const stage = document.getElementById(STAGE_ID);
        const viv = window.phenotypicViv;
        if (!stage || stage !== mountedElement || !viv) return;
        syncStageClip(stage);
        const cells = cellRecords(stage);
        if (!cells.length) return;
        try {
            await viv.setGridViews(
                STAGE_ID,
                cells,
                null,
                {cropSize: cropSize(stage)}
            );
            updateCameraToolbar(await viv.getGridCameraState(STAGE_ID));
        } catch (err) {
            console.error("[results_viewer] Colony viewport sync failed", err);
        }
    }

    function scheduleGeometrySync() {
        if (geometryFrame !== null) return;
        geometryFrame = window.requestAnimationFrame(syncColonyViewports);
    }

    function updateCameraToolbar(state) {
        if (!state) return;
        cameraState = state;
        const readout = document.getElementById("colony-camera-zoom-readout");
        if (readout) readout.textContent = `${state.zoomPercent}%`;
        const setDisabled = function (id, disabled) {
            const button = document.getElementById(id);
            if (button) button.disabled = Boolean(disabled);
        };
        ["colony-camera-pan-up", "colony-camera-pan-down",
         "colony-camera-pan-left", "colony-camera-pan-right"].forEach(
            function (id) { setDisabled(id, !state.canPan); }
        );
        setDisabled("colony-camera-zoom-out", !state.canZoomOut);
        setDisabled("colony-camera-zoom-in", !state.canZoomIn);
    }

    async function applyCameraCommand(command) {
        const viv = window.phenotypicViv;
        if (!viv || !mountedElement) return;
        try {
            updateCameraToolbar(await viv.setGridCamera(STAGE_ID, command));
        } catch (err) {
            console.error("[results_viewer] Colony camera command failed", err);
        }
    }

    function scheduleMount() {
        if (mountTimer !== null) window.clearTimeout(mountTimer);
        mountTimer = window.setTimeout(function () {
            mountTimer = null;
            mountGrid();
        }, 50);
    }

    function reconcileGrid() {
        const stage = document.getElementById(STAGE_ID);
        const signature = cellSignature(stage);
        if (stage !== mountedElement || signature !== mountedSignature) {
            scheduleMount();
        } else {
            scheduleGeometrySync();
        }
    }

    function mutationTouchesGrid(mutations) {
        const selector = "#colony-viv-grid-stage, [data-colony-viv-cell]";
        return mutations.some(function (mutation) {
            return Array.from(mutation.addedNodes)
                .concat(Array.from(mutation.removedNodes))
                .some(function (node) {
                    if (node.nodeType !== Node.ELEMENT_NODE) return false;
                    return node.matches(selector) || Boolean(node.querySelector(selector));
                });
        });
    }

    function mutationChangesGridGeometry(mutations) {
        return mutations.some(function (mutation) {
            const target = mutation.target;
            return target.nodeType === Node.ELEMENT_NODE && target.closest(
                "#colony-grid-container"
            );
        });
    }

    function updateFocusFromViewport() {
        focusTimer = null;
        const nodes = document.querySelectorAll("[data-colony-grid-index]");
        if (!nodes.length || !window.dash_clientside) return;
        const cx = window.innerWidth / 2;
        const cy = window.innerHeight / 2;
        let best = null;
        let bestDistance = Infinity;
        nodes.forEach(function (node) {
            const rect = node.getBoundingClientRect();
            if (rect.bottom < 0 || rect.top > window.innerHeight ||
                rect.right < 0 || rect.left > window.innerWidth) return;
            const dx = (rect.left + rect.right) / 2 - cx;
            const dy = (rect.top + rect.bottom) / 2 - cy;
            const distance = dx * dx + dy * dy;
            if (distance < bestDistance) {
                bestDistance = distance;
                best = Number(node.dataset.colonyGridIndex);
            }
        });
        if (Number.isFinite(best) && best !== lastFocus) {
            lastFocus = best;
            window.dash_clientside.set_props(FOCUS_STORE_ID, {data: best});
        }
    }

    function scheduleFocus() {
        if (focusTimer !== null) window.clearTimeout(focusTimer);
        focusTimer = window.setTimeout(updateFocusFromViewport, 120);
    }

    document.addEventListener("click", function (event) {
        const button = event.target.closest && event.target.closest("button");
        const command = button && CAMERA_COMMANDS[button.id];
        if (command) applyCameraCommand(command);
    });
    document.addEventListener("keydown", function (event) {
        const target = event.target;
        const controls = target && target.closest && target.closest(
            "#colony-camera-toolbar, #colony-viv-grid-stage"
        );
        if (!controls) return;
        const keyCommands = {
            ArrowUp: CAMERA_COMMANDS["colony-camera-pan-up"],
            ArrowDown: CAMERA_COMMANDS["colony-camera-pan-down"],
            ArrowLeft: CAMERA_COMMANDS["colony-camera-pan-left"],
            ArrowRight: CAMERA_COMMANDS["colony-camera-pan-right"],
            "+": CAMERA_COMMANDS["colony-camera-zoom-in"],
            "=": CAMERA_COMMANDS["colony-camera-zoom-in"],
            "-": CAMERA_COMMANDS["colony-camera-zoom-out"],
            "_": CAMERA_COMMANDS["colony-camera-zoom-out"],
            "0": CAMERA_COMMANDS["colony-camera-center"],
            "1": CAMERA_COMMANDS["colony-camera-one-to-one"],
            f: CAMERA_COMMANDS["colony-camera-fit"],
            F: CAMERA_COMMANDS["colony-camera-fit"]
        };
        const command = keyCommands[event.key];
        if (!command) return;
        event.preventDefault();
        applyCameraCommand(command);
    });

    const observer = new MutationObserver(function (mutations) {
        if (mutationTouchesGrid(mutations)) {
            reconcileGrid();
        } else if (mutationChangesGridGeometry(mutations)) {
            scheduleGeometrySync();
        }
        scheduleFocus();
    });
    observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["style", "data-colony-viv-cell"],
        childList: true,
        subtree: true
    });
    window.addEventListener("scroll", function () {
        scheduleFocus();
        // The shared Viv stage is viewport-sized rather than grid-sized. Any
        // page or nested-container scroll changes every tile's coordinates in
        // that fixed stage even though no element resized.
        scheduleGeometrySync();
    }, true);
    window.addEventListener("resize", function () {
        scheduleFocus();
        scheduleGeometrySync();
    });
    scheduleMount();
    scheduleFocus();
})();

/* ============================================================
 * (F) Shared mount observer.
 *
 * Dash mounts tabs lazily and replaces whole subtrees on re-render, so
 * every delegated listener below has to attach to nodes that do not
 * exist yet AND re-attach to nodes that are swapped out later.
 *
 * The shape that used to serve that was `setInterval(tryAttach, 100)`
 * beside a <body> MutationObserver. It only ever cleared once EVERY
 * target existed, so a surface that is never mounted -- the QC gallery
 * and the QC worklist splitter both are -- left a 100 ms timer running
 * for the life of the session while the observer was already doing the
 * same work. Attachment is observer-only now, so it terminates by
 * construction rather than by a poll that happens to succeed.
 *
 * `ns.onElementsMounted(selector, attach)` calls `attach(element)` for
 * every current match and for every match added afterwards. Two
 * requirements on callers:
 *
 *   - `attach` MUST be idempotent (all callers guard on a dataset flag).
 *     A re-render inside an already-attached subtree re-reports nodes
 *     that are already wired.
 *   - Read per-element configuration inside the handler, not at attach
 *     time, so Dash rewriting an attribute on a node it keeps is seen.
 *
 * Only ADDED nodes are inspected, never the whole document per
 * mutation: this page runs Viv rendering and a virtualized colony grid,
 * both of which mutate the DOM continuously.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer =
        window.__phenotypicResultsViewer || {};

    function attachWithin(node, selector, attach) {
        if (!node || node.nodeType !== 1) return;   // 1 === ELEMENT_NODE
        if (typeof node.matches === "function" && node.matches(selector)) {
            attach(node);
        }
        if (typeof node.querySelectorAll === "function") {
            node.querySelectorAll(selector).forEach(attach);
        }
    }

    ns.onElementsMounted = function (selector, attach) {
        function start() {
            attachWithin(document.body, selector, attach);
            const obs = new MutationObserver(function (mutations) {
                mutations.forEach(function (m) {
                    m.addedNodes.forEach(function (node) {
                        attachWithin(node, selector, attach);
                    });
                });
            });
            obs.observe(document.body, { childList: true, subtree: true });
        }
        if (document.body) {
            start();
        } else {
            // One-shot, not a poll: the only reason body can be missing is
            // that this file was loaded from <head>.
            document.addEventListener("DOMContentLoaded", start, { once: true });
        }
    };
})();

/* ============================================================
 * (G) Tile multi-select shift-click bridge (colony grid + QC gallery).
 *
 * Bridges native checkbox click events on a tile container into a
 * Dash dcc.Store so the Python side can apply selection logic (single
 * toggle, shift-range, etc.) without round-tripping every image's
 * <input> through the server.
 *
 * The SAME bridge serves two surfaces (M1: QC-review selection parity):
 *   - #colony-grid-container -> store-colony-selection-delta
 *   - #qc-review-gallery     -> store-qc-gallery-selection-delta
 * QC tiles carry the same `.colony-cell-checkbox` + `data-key` chrome as
 * colony tiles, so one parameterized listener covers both; each surface
 * emits to its own delta store, and a per-surface Python consumer folds
 * the delta into the SHARED store-colony-selection.
 *
 * Each click emits a payload of the form:
 *   { key: [image_file, label], shift: <bool>, ts: <ms> }
 * via window.dash_clientside.set_props.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer;

    // (containerId, deltaStoreId, datasetFlag) per surface. The dataset
    // flag is the single source of truth for "this container already has
    // our listener", so a container reported twice by the mount observer
    // is wired once.
    const BRIDGES = [
        {
            containerId: "colony-grid-container",
            deltaStoreId: "store-colony-selection-delta",
            flag: "_colonyShiftBridge",
        },
        {
            containerId: "qc-review-gallery",
            deltaStoreId: "store-qc-gallery-selection-delta",
            flag: "_qcShiftBridge",
        },
    ];

    function attachListener(container, deltaStoreId, flag) {
        if (container.dataset[flag] === "1") return;
        container.dataset[flag] = "1";
        container.addEventListener("click", function (event) {
            const tgt = event.target;
            if (!tgt || !tgt.classList ||
                !tgt.classList.contains("colony-cell-checkbox")) {
                return;
            }
            const raw = tgt.dataset ? tgt.dataset.key : null;
            if (!raw || typeof raw !== "string") {
                console.warn("[results_viewer] tile checkbox missing data-key");
                return;
            }
            const parts = raw.split("::");
            if (parts.length !== 2) {
                console.warn("[results_viewer] tile data-key malformed:", raw);
                return;
            }
            const imageFile = parts[0];
            const label = parseInt(parts[1], 10);
            if (Number.isNaN(label)) {
                console.warn("[results_viewer] tile label not an int:", raw);
                return;
            }
            // Suppress the native toggle: Python owns the checked state.
            event.preventDefault();
            const payload = {
                key: [imageFile, label],
                shift: !!event.shiftKey,
                ts: Date.now(),
            };
            const dc = window.dash_clientside;
            if (!dc || typeof dc.set_props !== "function") {
                console.warn("[results_viewer] dash_clientside.set_props unavailable");
                return;
            }
            dc.set_props(deltaStoreId, { data: payload });
        });
        console.info(
            "[results_viewer] shift-click bridge attached:", container.id
        );
    }

    const BRIDGE_BY_ID = new Map(
        BRIDGES.map(function (b) { return [b.containerId, b]; })
    );
    const CONTAINER_SELECTOR = BRIDGES.map(function (b) {
        return "#" + b.containerId;
    }).join(", ");

    // Both surfaces mount lazily and either can be re-rendered later, so
    // section (F) drives attachment for whichever container appears. It
    // used to be a poll that waited for BOTH: the QC gallery is unmounted,
    // so that poll never terminated.
    ns.onElementsMounted(CONTAINER_SELECTOR, function (container) {
        const bridge = BRIDGE_BY_ID.get(container.id);
        if (bridge) {
            attachListener(container, bridge.deltaStoreId, bridge.flag);
        }
    });
})();

/* ============================================================
 * (H) Generic drag-splitter.
 *
 * A thin handle between a sizeable side pane and the rest of a surface.
 * Dragging it sets the pane's width live, clamped to the handle's bounds
 * (defaulting to [MIN_W, MAX_W] px); on mouse-up the final width is
 * persisted to a Dash store via window.dash_clientside.set_props, so a
 * Python callback can re-apply it across re-renders and collapse.
 *
 * Every identifier is DATA-DRIVEN, because more than one surface needs
 * this. The handle declares its own wiring:
 *
 *   data-splitter-target : id of the pane whose width the drag sets
 *   data-splitter-store  : id of the dcc.Store the final width goes to
 *   data-splitter-edge   : "left" if the handle sits on the pane's LEFT
 *                          edge; anything else (incl. absent) means the
 *                          right edge
 *   data-splitter-min    : lower clamp in px, default MIN_W
 *   data-splitter-max    : upper clamp in px, default MAX_W
 *
 * The edge is what decides the SIGN of the drag, and it cannot be
 * inferred. A left pane's handle follows it, so the pane grows as the
 * cursor moves right (+dx). A right-docked pane's handle is on its
 * leading edge, so the pane grows as the cursor moves LEFT (-dx). Assume
 * the first and the second pane's edge runs away from the cursor: drag
 * left to widen and it narrows instead.
 *
 * Carrying `data-splitter-target` is what MAKES an element a handle --
 * this module names no surface and no id of its own. Python owns both
 * ids, which is where they belong: they are the same ids its callbacks
 * already bind.
 *
 * Attachment goes through section (F)'s mount observer, so a handle that
 * mounts with a lazily-rendered tab is picked up, a handle Dash replaces
 * on re-render is re-wired, and a surface that never mounts costs
 * nothing. It used to be a poll that never terminated.
 *
 * `clampSidebarWidth` is exposed on the namespace so a test can drive the
 * exact clamp the drag uses without a real pointer drag.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer =
        window.__phenotypicResultsViewer || {};

    const MIN_W = 140;
    const MAX_W = 380;
    const DEFAULT_W = 180;

    //: An element carrying this attribute IS a splitter handle.
    const HANDLE_SELECTOR = "[data-splitter-target]";
    //: dataset key for the idempotence flag -> `data-splitter-attached`.
    const ATTACHED_FLAG = "splitterAttached";

    // A bound that is absent, blank or unparseable falls back to the
    // module's own. `Number()` alone cannot decide this: it maps "" and
    // null to 0, both of which are finite, so `data-splitter-min=""` --
    // what an empty attribute reads as -- would be honoured as a zero
    // floor and let the pane be dragged shut. Emptiness is checked before
    // the numeric coercion, not after it.
    function boundOr(value, fallback) {
        if (value === null || value === undefined) return fallback;
        if (String(value).trim() === "") return fallback;
        const n = Number(value);
        return Number.isFinite(n) ? n : fallback;
    }

    // `min`/`max` are optional: omit them and the module's own bounds
    // apply, which is what every caller predating per-pane bounds did.
    ns.clampSidebarWidth = function (px, min, max) {
        const lo = boundOr(min, MIN_W);
        const hi = boundOr(max, MAX_W);
        const n = Math.round(Number(px));
        // The garbage-input fallback is clamped too: DEFAULT_W is the
        // worklist's default and can sit outside a pane's own bounds.
        if (!Number.isFinite(n)) return Math.max(lo, Math.min(hi, DEFAULT_W));
        return Math.max(lo, Math.min(hi, n));
    };

    function persistWidth(storeId, px) {
        // A handle with no store is legal: the drag still resizes, the
        // width just does not survive a re-render.
        if (!storeId) return;
        const dc = window.dash_clientside;
        if (!dc || typeof dc.set_props !== "function") {
            console.warn("[results_viewer] dash_clientside.set_props unavailable");
            return;
        }
        dc.set_props(storeId, { data: px });
    }

    function attachSplitter(handle) {
        if (handle.dataset[ATTACHED_FLAG] === "1") return;
        handle.dataset[ATTACHED_FLAG] = "1";
        handle.addEventListener("mousedown", function (downEvt) {
            // Resolved per drag, not per attach: Dash can rewrite a
            // node's attributes without replacing the node.
            const targetId = handle.dataset.splitterTarget;
            const pane = targetId ? document.getElementById(targetId) : null;
            if (!pane) return;
            downEvt.preventDefault();  // don't text-select while dragging
            const startX = downEvt.clientX;
            const startW = pane.getBoundingClientRect().width;
            // Resolved per drag for the same reason the ids are.
            const sign = handle.dataset.splitterEdge === "left" ? -1 : 1;
            const minW = handle.dataset.splitterMin;
            const maxW = handle.dataset.splitterMax;
            // Visual feedback during the drag.
            document.body.style.userSelect = "none";
            document.body.style.cursor = "col-resize";

            function onMove(moveEvt) {
                const next = ns.clampSidebarWidth(
                    startW + sign * (moveEvt.clientX - startX), minW, maxW
                );
                pane.style.width = next + "px";
            }
            function onUp() {
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup", onUp);
                document.body.style.userSelect = "";
                document.body.style.cursor = "";
                const finalW = ns.clampSidebarWidth(
                    pane.getBoundingClientRect().width, minW, maxW
                );
                pane.style.width = finalW + "px";
                // survives re-renders + collapse
                persistWidth(handle.dataset.splitterStore, finalW);
            }
            document.addEventListener("mousemove", onMove);
            document.addEventListener("mouseup", onUp);
        });
        console.info("[results_viewer] splitter attached:", handle.id);
    }

    ns.onElementsMounted(HANDLE_SELECTOR, attachSplitter);
})();
