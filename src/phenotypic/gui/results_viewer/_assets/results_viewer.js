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
 *   5. Bridge shift-click tile selection into a Dash store (section E).
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
    async function mountStageNow(record) {
        const viv = facade();
        if (!viv) {
            console.error("[results_viewer] window.phenotypicViv is missing");
            return null;
        }
        const divId = record.id;
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
            entry = {signature: null, opacity: {}, labelVisible: true};
            ns.stages.set(divId, entry);
        }
        entry.record = record;

        const display = record.display || {};
        const signature = sourceSignature(record.spec, display);
        if (signature !== entry.signature) {
            const spec = Object.assign({}, record.spec);
            if (display.seriesPath) spec.seriesPath = display.seriesPath;
            renderLevel(record.levelReadoutId, null);
            try {
                await viv.setSource(divId, spec);
            } catch (err) {
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
        const previous = ns.stageLoadQueues.get(divId) || Promise.resolve();
        const queued = previous.catch(function () { return null; }).then(function () {
            if (ns.stageLoadEpochs.get(divId) !== epoch) return null;
            return mountStageNow(record);
        });
        ns.stageLoadQueues.set(divId, queued);
        return queued.finally(function () {
            if (ns.stageLoadQueues.get(divId) === queued) {
                ns.stageLoadQueues.delete(divId);
            }
        });
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
 * (F) Tile multi-select shift-click bridge (colony grid + QC gallery).
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

    // (containerId, deltaStoreId, datasetFlag) per surface. The dataset
    // flag is the single source of truth for "this container already has
    // our listener", so the polling path and the body MutationObserver
    // agree even if they race on the same fresh container.
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

    function tryAttach() {
        // Returns true only once EVERY surface's container is present, so
        // the initial poll keeps running until both have mounted (QC mounts
        // later than the colony grid).
        let allPresent = true;
        BRIDGES.forEach(function (b) {
            const container = document.getElementById(b.containerId);
            if (container) {
                attachListener(container, b.deltaStoreId, b.flag);
            } else {
                allPresent = false;
            }
        });
        return allPresent;
    }

    // Dash mounts the tabs lazily; poll until every container exists, then
    // attach the delegated listeners and stop polling.
    if (!tryAttach()) {
        const interval = setInterval(function () {
            if (tryAttach()) clearInterval(interval);
        }, 100);
    }

    // If Dash later replaces a container (tab switch / re-render), the
    // dataset flag goes away with the old node. Watch <body> for re-mounts
    // and re-attach to whichever fresh container appears.
    function startReattachObserver() {
        if (!document.body) return false;
        const obs = new MutationObserver(function () {
            BRIDGES.forEach(function (b) {
                const container = document.getElementById(b.containerId);
                if (!container) return;
                // attachListener is idempotent — its dataset-flag guard
                // makes a no-op cheap, so we can fire on every mutation.
                attachListener(container, b.deltaStoreId, b.flag);
            });
        });
        obs.observe(document.body, { childList: true, subtree: true });
        return true;
    }

    if (!startReattachObserver()) {
        const bodyInterval = setInterval(function () {
            if (startReattachObserver()) clearInterval(bodyInterval);
        }, 100);
    }
})();

/* ============================================================
 * (F) QC Review worklist drag-splitter.
 *
 * A thin handle (#qc-review-splitter) between the worklist sidebar and
 * the detail/gallery pane. Dragging it widens/narrows the worklist
 * (#qc-review-worklist) live, clamped to [MIN, MAX] px; on mouse-up the
 * final width is persisted to the Dash store `store-qc-sidebar-width`
 * via window.dash_clientside.set_props, so a Python callback can re-apply
 * it across re-renders + collapse. Mirrors the colony shift-click bridge:
 * poll-to-attach + a <body> MutationObserver re-attach, both idempotent
 * via a dataset flag.
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

    ns.clampSidebarWidth = function (px) {
        const n = Math.round(Number(px));
        if (!Number.isFinite(n)) return 180;  // default
        return Math.max(MIN_W, Math.min(MAX_W, n));
    };

    function persistWidth(px) {
        const dc = window.dash_clientside;
        if (!dc || typeof dc.set_props !== "function") {
            console.warn("[results_viewer] dash_clientside.set_props unavailable");
            return;
        }
        dc.set_props("store-qc-sidebar-width", { data: px });
    }

    function attachSplitter(handle) {
        if (handle.dataset._qcSplitter === "1") return;
        handle.dataset._qcSplitter = "1";
        handle.addEventListener("mousedown", function (downEvt) {
            const worklist = document.getElementById("qc-review-worklist");
            if (!worklist) return;
            downEvt.preventDefault();  // don't text-select while dragging
            const startX = downEvt.clientX;
            const startW = worklist.getBoundingClientRect().width;
            // Visual feedback during the drag.
            document.body.style.userSelect = "none";
            document.body.style.cursor = "col-resize";

            function onMove(moveEvt) {
                const next = ns.clampSidebarWidth(startW + (moveEvt.clientX - startX));
                worklist.style.width = next + "px";
            }
            function onUp() {
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup", onUp);
                document.body.style.userSelect = "";
                document.body.style.cursor = "";
                const finalW = ns.clampSidebarWidth(
                    worklist.getBoundingClientRect().width
                );
                worklist.style.width = finalW + "px";
                persistWidth(finalW);  // survives re-renders + collapse
            }
            document.addEventListener("mousemove", onMove);
            document.addEventListener("mouseup", onUp);
        });
        console.info("[results_viewer] QC review splitter attached");
    }

    function tryAttach() {
        const handle = document.getElementById("qc-review-splitter");
        if (!handle) return false;
        attachSplitter(handle);
        return true;
    }

    if (!tryAttach()) {
        const interval = setInterval(function () {
            if (tryAttach()) clearInterval(interval);
        }, 100);
    }

    // Re-attach if Dash re-mounts the Review subtree (tab/sub-view switch).
    function startReattachObserver() {
        if (!document.body) return false;
        const obs = new MutationObserver(function () {
            const handle = document.getElementById("qc-review-splitter");
            if (handle) attachSplitter(handle);  // idempotent
        });
        obs.observe(document.body, { childList: true, subtree: true });
        return true;
    }
    if (!startReattachObserver()) {
        const bodyInterval = setInterval(function () {
            if (startReattachObserver()) clearInterval(bodyInterval);
        }, 100);
    }
})();
