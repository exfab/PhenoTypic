/**
 * Node-preview stage glue. Exposes `window.__phenotypicNodePreview`.
 *
 * The pane reads OME-Zarr chunks straight out of the node's preview store,
 * in the browser, through the SHARED Viv facade -- the same
 * `window.phenotypicViv` the results viewer's Plate drives. There is no
 * server-rendered pyramid on this path any more: no PNG staging, no DZI, no
 * OpenSeadragon.
 *
 * Nothing here touches the vendored bundle's own global. `viv_viewer.js` is
 * the only file that does, which is what lets the artifact be replaced
 * without editing any surface -- and is why this file names it nowhere.
 *
 * The `spec` this receives is `build_source_spec`'s dict, narrowed by
 * `build_channel_spec` to the layer the radio selected. It carries a
 * GENERATION TOKEN inside `storeUrl`, and Python rebuilds it on every layer
 * switch and every scope recompute -- so a spec whose `storeUrl` differs is a
 * different publish, not merely a different view, and must be re-sourced.
 *
 * The point picker deliberately does NOT come through here: it picks points
 * on a source image before any node has run, so there is no store to read and
 * it stays on DZI + OpenSeadragon (`point_picker.js`).
 */
(function () {
    "use strict";
    const ns = window.__phenotypicNodePreview =
        window.__phenotypicNodePreview || {};

    /** Facade layer ids; mirrored in `_preview_callbacks.py`. */
    const IMAGE_LAYER = "image";
    const LABEL_LAYER = "labels";

    // The mounted container id, and the spec signature it currently shows.
    // `setSource` re-opens the store and rebuilds every layer, so it fires
    // only when the store generation or the displayed series actually moved.
    let mountedId = null;
    let signature = null;
    let loadEpoch = 0;

    function facade() {
        return window.phenotypicViv || null;
    }

    function sourceSignature(spec) {
        if (!spec) return null;
        return [
            spec.storeUrl,
            spec.seriesPath,
            spec.labelPath || "",
            spec.imageVisible === false ? "label-only" : "image"
        ].join(" ");
    }

    /**
     * Mount (once) and point the stage at one node store generation.
     *
     * @param {string} divId Id of the host element.
     * @param {object} spec `build_channel_spec`'s dict.
     */
    ns.mountViewer = async function (divId, spec) {
        const viv = facade();
        if (!viv) {
            console.error("[node_preview] window.phenotypicViv is missing");
            return null;
        }
        const el = document.getElementById(divId);
        if (!el || !spec) { return null; }
        if (mountedId && mountedId !== divId) { ns.disposeViewer(); }
        const epoch = ++loadEpoch;
        const isCurrent = function () {
            return loadEpoch === epoch && mountedId === divId;
        };
        if (!mountedId) {
            // `refetchSource: null` -- recovery after a re-promote needs a
            // server round-trip Dash owns (the token is minted Python-side),
            // so a stale read throws `StaleGenerationError` into the console
            // and the next layer switch or reopen re-resolves it.
            await viv.mount(divId, { refetchSource: null });
            if (loadEpoch !== epoch) return null;
            mountedId = divId;
            signature = null;
        }
        const next = sourceSignature(spec);
        if (next === signature) { return null; }
        try {
            const loaded = await viv.setSource(divId, spec);
            if (loaded === undefined || !isCurrent()) return null;
        } catch (err) {
            if (!isCurrent()) return null;
            console.error("[node_preview] setSource failed", divId, err);
            signature = null;
            return null;
        }
        signature = next;
        // Applied AFTER the source: `setSource` rebuilds every layer, and the
        // facade's viewer keeps its hidden-id set across a re-source, so the
        // objmap-only channel's hidden image layer must be re-asserted in
        // BOTH directions rather than assumed to have survived.
        await viv.setLayerVisibility(
            divId, IMAGE_LAYER, spec.imageVisible !== false
        );
        if (!isCurrent()) return null;
        if (spec.labelPath) {
            await viv.setLayerVisibility(divId, LABEL_LAYER, true);
            if (!isCurrent()) return null;
        }
        return divId;
    };

    /** Tear the stage down, freeing its WebGL context. */
    ns.disposeViewer = function () {
        loadEpoch += 1;
        if (!mountedId) { return; }
        const viv = facade();
        try { if (viv) viv.destroy(mountedId); }
        catch (e) { console.error(e); }
        mountedId = null;
        signature = null;
    };
})();
