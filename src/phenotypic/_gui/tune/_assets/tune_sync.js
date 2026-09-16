/* Curate linked-zoom mirror (Task B4).
 *
 * Mirrors the x/y axis range between the two side-by-side overlay graphs
 * (graph A and graph B) so a pan/zoom on one re-frames the other, keeping the
 * same colonies in register while a user compares two candidates'
 * segmentations.
 *
 * Critical guard: a naive A->B->A relayout chain relays infinitely (each
 * applied range fires the partner's ``relayoutData`` again). We propagate ONLY
 * from the graph the user actually interacted with -- read
 * ``dash_clientside.callback_context.triggered[0].prop_id`` and forward only
 * the range keys. The partner figure we return carries the same range, so its
 * own subsequent relayout is a no-op range and the cycle terminates.
 *
 * Tokens only: this file references no colors / fonts -- the overlay figures
 * own their styling via the injected design tokens. It only reshapes axis
 * ranges, which are data coordinates, not design values.
 */
window.dash_clientside = window.dash_clientside || {};

(function () {
    "use strict";

    /* Extract the explicit axis range from a relayoutData payload as
     * ``{x: [lo, hi], y: [lo, hi]}``; ``null`` when the payload carries no
     * explicit range (autorange reset, hover, or a non-zoom relayout) so the
     * caller skips the mirror rather than clobbering the partner's autorange. */
    function extractRange(relayout) {
        if (!relayout) {
            return null;
        }
        var out = {};
        var hasX =
            Object.prototype.hasOwnProperty.call(relayout, "xaxis.range[0]") &&
            Object.prototype.hasOwnProperty.call(relayout, "xaxis.range[1]");
        var hasY =
            Object.prototype.hasOwnProperty.call(relayout, "yaxis.range[0]") &&
            Object.prototype.hasOwnProperty.call(relayout, "yaxis.range[1]");
        if (hasX) {
            out.x = [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]];
        }
        if (hasY) {
            out.y = [relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]];
        }
        return hasX || hasY ? out : null;
    }

    /* Deep-ish clone of a figure with the partner's axis ranges applied. The
     * input figure is treated as immutable (Dash diffs on reference); we build
     * a fresh object so the partner graph re-renders. */
    function withRange(figure, range) {
        var fig = figure ? JSON.parse(JSON.stringify(figure)) : { layout: {} };
        fig.layout = fig.layout || {};
        fig.layout.xaxis = fig.layout.xaxis || {};
        fig.layout.yaxis = fig.layout.yaxis || {};
        if (range.x) {
            fig.layout.xaxis.range = range.x;
            fig.layout.xaxis.autorange = false;
        }
        if (range.y) {
            fig.layout.yaxis.range = range.y;
            fig.layout.yaxis.autorange = false;
        }
        return fig;
    }

    window.dash_clientside.tune_sync = {
        /* Mirror the user-driven graph's range onto the partner figure.
         *
         * Registered twice (A->B and B->A); each registration passes its OWN
         * graph's relayout prop-id as ``selfPropId`` so the callback no-ops
         * unless its graph is the trigger -- the triggered-prop guard that
         * stops the A->B->A infinite relayout. */
        mirrorRange: function (relayoutSelf, partnerFigure, selfPropId) {
            var ctx = window.dash_clientside.callback_context;
            var triggered = (ctx && ctx.triggered) || [];
            if (!triggered.length) {
                return window.dash_clientside.no_update;
            }
            var firedProp = triggered[0].prop_id || "";
            if (firedProp !== selfPropId) {
                return window.dash_clientside.no_update;
            }
            var range = extractRange(relayoutSelf);
            if (range === null) {
                return window.dash_clientside.no_update;
            }
            return withRange(partnerFigure, range);
        },
    };
})();
