/**
 * Imperative facade over the vendored Viv + deck.gl bundle.
 *
 * Dash clientside callbacks talk to `window.phenotypicViv` and never to Viv
 * directly, so the vendored artifact can be replaced without touching Python.
 * Everything policy-shaped -- which series is primary, where the objmap
 * lives, which generation token is current -- is resolved SERVER-SIDE and
 * arrives here as a `spec`. This file guesses nothing about the store.
 *
 * ORDERING RULE (spec decision C1): the wasm zstd codec must be registered
 * with zarrita's registry BEFORE any store is opened. Registering late does
 * not degrade -- every read fails. `ready()` is what every entry point
 * awaits, which is how the ordering is enforced rather than merely
 * documented.
 *
 * THE SOURCE SPEC, which phases 3 and 6 both produce:
 *
 *     {
 *       storeUrl:   "<prefix>/zarr/<ds>/<stem>.ome.zarr/<token>",
 *       seriesPath: "rgb",                     // resolved, never guessed
 *       labelPath:  "rgb/labels/objmap"|null,  // OPTIONAL -- may be absent
 *     }
 *
 * `storeUrl` carries the generation token as a path segment, so every key
 * resolved against it belongs to one publish by construction.
 */
(function () {
  "use strict";

  /** @type {Map<string, object>} containerId -> instance record. */
  const instances = new Map();

  /** Layer ids `setLayerVisibility` addresses. Stable across phases. */
  const IMAGE_LAYER_ID = "image";
  const LABEL_LAYER_ID = "labels";

  // `ready()` is a FUNCTION, not a promise, and that is load-bearing.
  //
  // Dash walks `_assets/` with
  // `for current, _, files in sorted(os.walk(walk_dir))`, which appends every
  // ROOT-level asset before any SUBDIRECTORY asset -- so this file is loaded
  // BEFORE `viv/viv-bundle.min.js`:
  //
  //     /assets/results_viewer.js
  //     /assets/viv_viewer.js            <- this file, FIRST
  //     /assets/openseadragon/openseadragon.min.js
  //     /assets/viv/viv-bundle.min.js    <- the bundle, LAST
  //
  // Measured against a real Dash index in the phase-0 spike. So
  // `window.__vivBundle` is `undefined` while this file executes, and any
  // read of it at MODULE SCOPE captures that -- every method then fails on a
  // property access rather than on anything diagnosable.
  //
  // The spike recommended `const ready = (async () => { ... })()` as the fix.
  // IT IS NOT ONE. An async function body runs SYNCHRONOUSLY up to its first
  // `await`, and there is no `await` before the global is read, so that form
  // captures `undefined` at module-evaluation time exactly like the eager
  // one -- and worse, it turns the failure into a rejected promise created
  // before anything could handle it. Measured: it fails
  // `test_the_facade_survives_loading_before_the_bundle` with
  // "bundle asset did not load".
  //
  // Deferring creation of the promise to the first CALL is what actually
  // moves the read past the bundle's execution, because every entry point
  // calls `ready()` from inside a Dash callback, long after page load.
  let bundlePromise = null;

  /** Resolve the vendored bundle, registering the zstd codec exactly once. */
  function ready() {
    if (bundlePromise) return bundlePromise;
    bundlePromise = (async () => {
      const bundle = window.__vivBundle;
      if (!bundle) {
        throw new Error(
          "viv: bundle asset did not load (expected window.__vivBundle " +
            "from _assets/viv/viv-bundle.min.js)"
        );
      }
      // Idempotent: `entry.mjs` already registers this at module-evaluation
      // time. Repeated here so the ordering rule holds for ANY bundle the
      // facade is pointed at, not only one that happens to self-register.
      bundle.zarr.registry.set("zstd", () => bundle.numcodecs.Zstd);
      return bundle;
    })();
    return bundlePromise;
  }

  // ---- the byte-route store ------------------------------------------
  // A hand-written zarrita store rather than `bundle.zarr.FetchStore`,
  // because `FetchStore` maps every non-2xx that is not a 404 onto one
  // opaque `Unexpected response status` Error -- which erases the 409/422
  // distinction the byte route exists to make. Mirrors
  // `@zarrita/storage@0.1.4`'s `fetch.ts` otherwise, including its
  // HEAD-then-range suffix read; the upstream file is vendored beside the
  // spec so the two can be diffed.

  /** Thrown when the byte route reports the instance's token is stale. */
  class StaleGenerationError extends Error {
    constructor(url) {
      super(`viv: store was re-promoted; generation token is stale (${url})`);
      this.name = "StaleGenerationError";
      this.url = url;
    }
  }

  /** Thrown when this build cannot decode the store (route answers 422). */
  class StoreUnreadableError extends Error {
    constructor(message, url) {
      super(message);
      this.name = "StoreUnreadableError";
      this.url = url;
    }
  }

  function joinUrl(base, tail) {
    if (!tail) return base;
    return `${base.replace(/\/+$/, "")}/${String(tail).replace(/^\/+/, "")}`;
  }

  /**
   * Apply the byte route's status contract to one response.
   *
   * The contract is shared with `results_viewer/_zarr_routes.py`:
   *
   *   200/206 -> bytes
   *   404     -> ABSENT. Zarr's data model needs this: a sparse store omits
   *              any chunk equal to `fill_value`, and a v2 metadata probe
   *              (`.zattrs`/`.zgroup`/`.zarray`) is answered 404 by design.
   *   409     -> stale token. Raised as its own error and NOT mapped to
   *              `undefined`: an absent chunk is filled with `fill_value`,
   *              so a swallowed 409 renders BLACK TILES after every promote
   *              -- which reads as empty data rather than as an error. That
   *              is the plausible-wrong-pixels failure the token exists to
   *              prevent, moved to the client and made harder to see.
   *   422     -> this build cannot decode the store; surfaced verbatim.
   */
  async function handleResponse(response, url) {
    if (response.status === 404) return undefined;
    if (response.status === 200 || response.status === 206) {
      return new Uint8Array(await response.arrayBuffer());
    }
    if (response.status === 409) throw new StaleGenerationError(url);
    if (response.status === 422) {
      let detail = "";
      try {
        detail = await response.text();
      } catch (err) {
        detail = "";
      }
      throw new StoreUnreadableError(
        detail || `viv: store is unreadable by this build (${url})`,
        url
      );
    }
    throw new Error(
      `viv: unexpected response status ${response.status} ` +
        `${response.statusText} for ${url}`
    );
  }

  /**
   * A read-only zarrita store rooted at `baseUrl`.
   *
   * @param {string} baseUrl Absolute or app-relative URL of a zarr GROUP or
   *   ARRAY root -- NOT the store root, because `loadOmeZarrFromStore` calls
   *   `zarr.root(store)` and resolves keys from there. Callers pass the
   *   resolved series or label URL.
   * @param {{onStale?: function}} hooks Notified (once, best-effort) when a
   *   read reports a stale generation token.
   */
  function createByteRouteStore(baseUrl, hooks) {
    const root = baseUrl.replace(/\/+$/, "");
    const notifyStale = (err) => {
      if (err instanceof StaleGenerationError && hooks && hooks.onStale) {
        try {
          hooks.onStale(err);
        } catch (ignored) {
          /* a hook must not mask the read error */
        }
      }
      throw err;
    };
    return {
      url: root,
      async get(key) {
        const url = joinUrl(root, key);
        const response = await fetch(url);
        return handleResponse(response, url).catch(notifyStale);
      },
      async getRange(key, range) {
        const url = joinUrl(root, key);
        let response;
        if ("suffixLength" in range) {
          // Mirrors upstream `fetch_suffix` with `useSuffixRequest: false`:
          // HEAD for the length, then an offset range. Upstream passes the
          // FULL length as the range length rather than the suffix length,
          // so the request overshoots EOF and the server clamps. Replicated
          // deliberately -- diverging here would make the vendored
          // `fetch.ts` stop being evidence for what the client does.
          const head = await fetch(url, { method: "HEAD" });
          if (!head.ok) {
            return handleResponse(head, url).catch(notifyStale);
          }
          const length = Number(head.headers.get("Content-Length"));
          const offset = length - range.suffixLength;
          response = await fetch(url, {
            headers: { Range: `bytes=${offset}-${offset + length - 1}` },
          });
        } else {
          response = await fetch(url, {
            headers: {
              Range: `bytes=${range.offset}-${range.offset + range.length - 1}`,
            },
          });
        }
        return handleResponse(response, url).catch(notifyStale);
      },
    };
  }

  // ---- layers ---------------------------------------------------------

  /** Full representable range of a Viv dtype, as a contrast limit. */
  function dtypeDomain(dtype) {
    if (dtype === "Uint8" || dtype === "Int8") return [0, 255];
    if (dtype === "Float32" || dtype === "Float64") return [0, 1];
    return [0, 65535];
  }

  /**
   * Default layers for one loaded source pair.
   *
   * Deliberately plain: full-range contrast from the dtype, one primary
   * colour per channel, and the label layer at half opacity above the
   * image. Phase 3 owns real contrast and colour policy and replaces this
   * wholesale by passing `opts.buildLayers`.
   */
  function defaultLayers(bundle, loaded) {
    const { image, label } = loaded;
    const layers = [];
    const source = image.data[0];
    const channelAxis = source.labels.indexOf("c");
    const nChannels = channelAxis === -1 ? 1 : source.shape[channelAxis];
    const palette = [
      [255, 0, 0],
      [0, 255, 0],
      [0, 0, 255],
      [255, 255, 255],
    ];
    const domain = dtypeDomain(source.dtype);
    // `MultiscaleImageLayer.loader` is the LEVEL ARRAY; `ImageLayer.loader`
    // is a SINGLE pixel source (`loader.getRaster(...)`, @vivjs/layers dist:776).
    // Passing the array to `ImageLayer` fails at deck.gl layer
    // initialization with `e.getRaster is not a function` and paints
    // nothing. A store below the pyramid's `stop_px` has exactly one level,
    // so this is the ordinary small-image path, not an edge case.
    const multiscale = image.data.length > 1;
    const ImageClass = multiscale
      ? bundle.layers.MultiscaleImageLayer
      : bundle.layers.ImageLayer;
    layers.push(
      new ImageClass({
        id: IMAGE_LAYER_ID,
        loader: multiscale ? image.data : image.data[0],
        selections: Array.from({ length: nChannels }, (_, c) =>
          channelAxis === -1 ? {} : { c }
        ),
        contrastLimits: Array.from({ length: nChannels }, () => domain),
        colors: Array.from(
          { length: nChannels },
          (_, c) => palette[c % palette.length]
        ),
        channelsVisible: Array.from({ length: nChannels }, () => true),
      })
    );
    if (label) {
      const labelSource = label.data[0];
      const labelMultiscale = label.data.length > 1;
      const LabelClass = labelMultiscale
        ? bundle.layers.MultiscaleImageLayer
        : bundle.layers.ImageLayer;
      layers.push(
        new LabelClass({
          id: LABEL_LAYER_ID,
          loader: labelMultiscale ? label.data : label.data[0],
          selections: [{}],
          contrastLimits: [dtypeDomain(labelSource.dtype)],
          colors: [[255, 255, 0]],
          channelsVisible: [true],
          opacity: 0.5,
        })
      );
    }
    return layers;
  }

  // ---- public surface -------------------------------------------------

  /** Rebuild and install this instance's layers from its loaded sources. */
  function rebuildLayers(record, bundle) {
    const build = record.options.buildLayers || defaultLayers;
    record.viewer.setLayers(build(bundle, record.loaded, record.spec));
  }

  function requireInstance(containerId) {
    const record = instances.get(containerId);
    if (!record) throw new Error(`viv: #${containerId} not mounted`);
    return record;
  }

  /**
   * Create a deck.gl viewer inside `#containerId`.
   *
   * @param {string} containerId Id of an existing DOM element.
   * @param {object} [opts] Passed through to the bundle's `createViewer`
   *   (`initialViewState`, `views`, `onViewStateChange`), plus two facade
   *   options: `buildLayers({bundle, loaded, spec})` replacing the default
   *   layer policy, and `refetchSource(containerId)` returning a fresh spec
   *   after a re-promote.
   */
  async function mount(containerId, opts) {
    const bundle = await ready();
    const el = document.getElementById(containerId);
    if (!el) throw new Error(`viv: no element #${containerId}`);
    const options = opts || {};
    const viewer = bundle.createViewer(el, options);
    instances.set(containerId, {
      viewer,
      options,
      spec: null,
      loaded: null,
      resourcing: null,
    });
    return viewer;
  }

  /**
   * Point a mounted viewer at one generation of one store.
   *
   * `spec.labelPath` is RESOLVED SERVER-SIDE from `phenotypic.labels.objmap`
   * and MAY BE ABSENT -- `build_phenotypic_attributes` omits the `labels`
   * key entirely for a store with no label image. Never derive it as
   * `${seriesPath}/labels/objmap` here: backend section 1.1 forbids
   * hard-coding it, and a `gray`-primary store has no `rgb` group at all.
   */
  async function setSource(containerId, spec) {
    const bundle = await ready();
    const record = requireInstance(containerId);
    if (!spec || !spec.storeUrl || !spec.seriesPath) {
      throw new Error(
        "viv: source spec needs storeUrl and seriesPath (both resolved " +
          "server-side)"
      );
    }
    const hooks = { onStale: () => resourceAfterPromote(containerId) };
    const image = await bundle.viv.loadOmeZarrFromStore(
      createByteRouteStore(joinUrl(spec.storeUrl, spec.seriesPath), hooks)
    );
    let label = null;
    if (spec.labelPath) {
      label = await bundle.viv.loadOmeZarrFromStore(
        createByteRouteStore(joinUrl(spec.storeUrl, spec.labelPath), hooks)
      );
    }
    const loaded = { image, label };
    record.spec = spec;
    record.loaded = loaded;
    // `setLayers` does not reset visibility: the bundle's viewer keeps its
    // own hidden-id set across a re-source, so a layer hidden before a
    // re-promote stays hidden after one.
    rebuildLayers(record, bundle);
    return loaded;
  }

  /**
   * Re-fetch the spec and re-source after a re-promote (a 409).
   *
   * Coalesced: a pan issues many concurrent chunk reads, and every one of
   * them reports the same stale token. Without the in-flight guard a single
   * promote would trigger dozens of duplicate re-sources.
   */
  function resourceAfterPromote(containerId) {
    const record = instances.get(containerId);
    if (!record) return null;
    if (record.resourcing) return record.resourcing;
    const refetch = record.options.refetchSource;
    if (!refetch) {
      // No recovery path was wired. The read still throws
      // `StaleGenerationError` -- it is simply not repaired here.
      return null;
    }
    record.resourcing = Promise.resolve(refetch(containerId))
      .then((fresh) => setSource(containerId, fresh))
      .finally(() => {
        record.resourcing = null;
      });
    return record.resourcing;
  }

  function setViewState(containerId, viewState) {
    const record = instances.get(containerId);
    if (record) record.viewer.setViewState(viewState);
  }

  /**
   * Show or hide one layer by id (`IMAGE_LAYER_ID` / `LABEL_LAYER_ID`).
   *
   * Async, and rebuilding on the way back, for a measured reason. The
   * bundle's viewer hides a layer by FILTERING it out of the array it hands
   * deck.gl, and deck.gl finalizes a layer the moment it leaves that array
   * -- a finalized instance handed back paints nothing. So re-showing has
   * to give deck.gl a fresh descriptor. Same id and class, so deck.gl
   * diffs it as an update rather than a replacement.
   *
   * The obvious alternative -- `visible: false` on the layer instead of
   * removing it -- DOES NOT WORK for `MultiscaleImageLayer`. It builds its
   * background sublayer as
   * `new ImageLayer(this.props, {..., visible: !viewportId || ...})`
   * (`@vivjs/layers` dist:1009-1013, vendored beside the spec), which
   * overrides the inherited `visible` unconditionally. The composite and
   * its tiled sublayer go invisible; the background one keeps painting the
   * whole extent at low resolution, so the canvas never clears. Measured.
   */
  async function setLayerVisibility(containerId, name, visible) {
    const record = instances.get(containerId);
    if (!record) return;
    record.viewer.setLayerVisibility(name, visible);
    if (visible && record.loaded) rebuildLayers(record, await ready());
  }

  function destroy(containerId) {
    const record = instances.get(containerId);
    if (!record) return;
    record.viewer.finalize();
    instances.delete(containerId);
  }

  /** The vendored bundle's version string. Awaits the bundle, so async. */
  async function version() {
    return (await ready()).VERSION;
  }

  /**
   * TEST SEAM -- not API. Read a corner of one store array through the same
   * byte-route store production reads use, and return it flat.
   *
   * Exists so the zstd ordering rule can be asserted on DECODED PIXEL
   * VALUES rather than on the registry's contents, without a WebGL context.
   *
   * @param {string} storeUrl One store generation's base URL.
   * @param {string} arrayPath Path of a zarr ARRAY, e.g. `"rgb/0"`.
   * @param {number} [size] Corner edge length along the last two axes.
   */
  async function __debugReadChunk(storeUrl, arrayPath, size) {
    const bundle = await ready();
    const edge = size || 4;
    const store = createByteRouteStore(joinUrl(storeUrl, arrayPath), null);
    const arr = await bundle.zarr.open(bundle.zarr.root(store), {
      kind: "array",
    });
    const ndim = arr.shape.length;
    const selection = arr.shape.map((_, axis) =>
      axis < ndim - 2 ? 0 : bundle.zarr.slice(0, edge)
    );
    const chunk = await bundle.zarr.get(arr, selection);
    return Array.from(chunk.data).map(Number);
  }

  window.phenotypicViv = {
    ready,
    version,
    mount,
    setSource,
    setViewState,
    setLayerVisibility,
    destroy,
    errors: { StaleGenerationError, StoreUnreadableError },
    IMAGE_LAYER_ID,
    LABEL_LAYER_ID,
    __debugReadChunk,
  };
})();
