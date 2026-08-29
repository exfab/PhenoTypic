/**
 * Entry point for the vendored PhenoTypic Viv bundle.
 *
 * Everything the results-viewer facade needs is hung off ONE global,
 * `window.__vivBundle`. The facade (`_assets/viv_viewer.js`) is the only
 * thing that touches it; Dash clientside callbacks talk to the facade.
 *
 * ORDERING RULE (spec decision C1): the zstd wasm codec is registered with
 * zarrita's codec registry at MODULE EVALUATION TIME -- i.e. before this file
 * finishes executing, and therefore before any consumer can reach
 * `__vivBundle` at all, let alone open a store. Registering late does not
 * degrade; every read fails.
 *
 * zarrita 0.5.4 already ships a *lazy* zstd entry (`() => import(...)`).
 * We overwrite it with an eagerly-imported constructor for two reasons:
 * the dynamic import is a code-split point an IIFE cannot honour, and an
 * eager entry makes the ordering rule observable rather than incidental.
 *
 * React is deliberately absent. `@vivjs/viewers` is a React component set;
 * this bundle uses `@vivjs/loaders` + `@vivjs/layers` and drives deck.gl's
 * imperative `Deck` directly, which is why the artifact has no framework in
 * it.
 */
import * as zarr from "zarrita";
import Zstd from "numcodecs/zstd";
import Blosc from "numcodecs/blosc";
import Gzip from "numcodecs/gzip";

import {
  loadOmeZarr,
  loadOmeZarrFromStore,
  ZarrPixelSource,
  getChannelStats,
  getImageSize,
  isInterleaved,
} from "@vivjs/loaders";
import {
  ImageLayer,
  MultiscaleImageLayer,
  XRLayer,
  XR3DLayer,
} from "@vivjs/layers";
import {
  AdditiveColormapExtension,
  ColorPaletteExtension,
  LensExtension,
} from "@vivjs/extensions";
import { Deck, OrthographicView, COORDINATE_SYSTEM } from "@deck.gl/core";

// --- C1: register before anything can open a store -------------------------
zarr.registry.set("zstd", () => Zstd);
zarr.registry.set("blosc", () => Blosc);
zarr.registry.set("gzip", () => Gzip);

const VERSION = __PHENOTYPIC_VIV_BUNDLE_VERSION__;

/**
 * Minimal imperative viewer over deck.gl.
 *
 * Deliberately thin: layer construction, view state and layer visibility.
 * Anything policy-shaped (which series is primary, where the objmap lives,
 * generation tokens) belongs in the facade or in Python, never here -- this
 * file is replaced wholesale when Viv is upgraded.
 */
function createViewer(el, opts = {}) {
  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  el.appendChild(canvas);

  let viewState = opts.initialViewState || { target: [0, 0, 0], zoom: 0 };
  let layers = [];
  const hidden = new Set();

  const deck = new Deck({
    canvas,
    views: opts.views || [new OrthographicView({ id: "ortho", controller: true })],
    viewState,
    controller: true,
    onViewStateChange: ({ viewState: next }) => {
      viewState = next;
      deck.setProps({ viewState });
      if (opts.onViewStateChange) opts.onViewStateChange(next);
    },
    layers: [],
  });

  function render() {
    deck.setProps({
      layers: layers.filter((layer) => !hidden.has(layer.id)),
    });
  }

  return {
    deck,
    get viewState() {
      return viewState;
    },
    setLayers(next) {
      layers = next;
      render();
    },
    setViewState(next) {
      viewState = next;
      deck.setProps({ viewState: next });
    },
    setViews(views) {
      deck.setProps({ views });
    },
    setLayerVisibility(name, visible) {
      if (visible) hidden.delete(name);
      else hidden.add(name);
      render();
    },
    finalize() {
      deck.finalize();
      if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
    },
  };
}

const bundle = {
  VERSION,
  zarr,
  numcodecs: { Zstd, Blosc, Gzip },
  viv: {
    loadOmeZarr,
    loadOmeZarrFromStore,
    ZarrPixelSource,
    getChannelStats,
    getImageSize,
    isInterleaved,
  },
  layers: { ImageLayer, MultiscaleImageLayer, XRLayer, XR3DLayer },
  extensions: {
    AdditiveColormapExtension,
    ColorPaletteExtension,
    LensExtension,
  },
  deck: { Deck, OrthographicView, COORDINATE_SYSTEM },
  createViewer,
};

if (typeof window !== "undefined") {
  window.__vivBundle = bundle;
}

export default bundle;
