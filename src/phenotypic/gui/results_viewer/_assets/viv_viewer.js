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
 * `build_source_spec` returns those three plus `series`, `pyramid`, `token`
 * and `measured`, which the surface's own chrome reads and this file
 * ignores. Extra keys are deliberate: it means the dict crosses the Dash
 * boundary once, unmodified, rather than being split and re-joined.
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

  // ---- colony grid mode ------------------------------------------------
  //
  // ONE dynamic viewState, carrying `zoom`, that every per-cell `View`
  // merges its own `target` over. There is only one zoom, so it cannot
  // drift -- a value, not a protocol.
  //
  // The REJECTED alternative is the keyed viewState map deck.gl's
  // developer guide describes ("add a key to the `viewState` object
  // corresponding to the `id` of the view"). It renders identically and it
  // is itself a sync protocol: `onViewStateChange` fires with
  // `{viewId, viewState}` for the ONE view a gesture touched, so keeping
  // `zoom` common means a handler fanning the new zoom back across every
  // entry. The first real user gesture drifts the zooms apart, and a test
  // that drives `setViewState` programmatically cannot catch it.
  //
  // The merge form is deck.gl's own (`View.filterViewState`, core
  // `src/views/view.ts`): a `View` whose `props.viewState` carries an `id`
  // is `deepMergeViewState(shared, own)`, and `deepMergeViewState` merges
  // position arrays component-wise with the view's finite values winning.
  // So `target` is per view and `zoom` is shared, from one stored object.

  /** Id of the single shared dynamic viewState every grid view merges. */
  const GRID_VIEW_STATE_ID = "colony-shared";

  /** Prefix of the per-cell view ids. `cell-<id>`. */
  const GRID_CELL_VIEW_PREFIX = "cell-";

  /** Gap, in CSS px, between grid cells when the caller names none. */
  const GRID_GAP_PX = 8;

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
  function defaultLayers(bundle, loaded, spec, overrides) {
    const { image, label } = loaded;
    const opacityFor = (id, fallback) => {
      const value = overrides && overrides[id];
      return typeof value === "number" ? value : fallback;
    };
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
        opacity: opacityFor(IMAGE_LAYER_ID, 1),
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
          opacity: opacityFor(LABEL_LAYER_ID, 0.5),
        })
      );
    }
    return layers;
  }

  // ---- public surface -------------------------------------------------

  /**
   * Bytes one store's whole pyramid occupies in a `TileLayer` cache.
   *
   * Derived, not guessed. `level_tiles x per_tile_bytes x visible_layers`
   * over every level, which is the CEILING the resident set cannot exceed:
   * in grid mode every cell is a viewport onto the SAME store, so tiles are
   * SHARED between cells and the resident set is their UNION, not the sum.
   * A 4000x3000 plate at level 0 is `ceil(4000/1024) x ceil(3000/1024)` =
   * 12 chunks, ~38 MB; the coarser levels add about a third again.
   * `logic_validation_scripts/2026-08-26-viewer-viv-rebuild/
   * colony_view_budget.py` re-derives it.
   *
   * The quantity that drives this is the number of distinct STORES in the
   * grid, NOT the cell count: 1536 cells over one plate is ~50 MB; 1536
   * cells over 1536 different images is 1536x that. One instance holds one
   * store, so this is the per-instance bound.
   *
   * It is not only a leak guard. `Tileset2D._resizeCache` defaults to
   * `5 x selectedTiles.length` entries, and `TileLayer.renderLayers` draws
   * `tileset.tiles` -- the CACHE, not the selection. With N views over one
   * layer instance the tileset re-selects for whichever viewport updated
   * last, so cells whose tiles have been evicted paint nothing. Sizing the
   * cache to hold the union is what makes multi-view render at all.
   */
  function gridTileCacheBytes(loaded, layerCount) {
    let bytes = 0;
    for (const source of [loaded.image, loaded.label]) {
      if (!source) continue;
      for (const level of source.data) {
        const tile = level.tileSize || 1024;
        const shape = level.shape;
        const height = shape[shape.length - 2];
        const width = shape[shape.length - 1];
        const channelAxis = level.labels.indexOf("c");
        const channels = channelAxis === -1 ? 1 : shape[channelAxis];
        const itemsize = /(8)$/.test(level.dtype)
          ? 1
          : /(16)$/.test(level.dtype)
            ? 2
            : 4;
        bytes +=
          Math.ceil(height / tile) *
          Math.ceil(width / tile) *
          tile *
          tile *
          channels *
          itemsize;
      }
    }
    return bytes * Math.max(1, layerCount);
  }

  /** Rebuild and install this instance's layers from its loaded sources. */
  function rebuildLayers(record, bundle) {
    const build = record.options.buildLayers || defaultLayers;
    let layers = build(bundle, record.loaded, record.spec, record.opacity);
    if (record.grid && record.loaded) {
      // Cloned rather than passed into the builder, so a caller-supplied
      // `buildLayers` (the Plate's) gets the bound too without knowing
      // about grid mode.
      const budget = gridTileCacheBytes(record.loaded, layers.length);
      layers = layers.map((layer) =>
        layer.clone({ maxCacheByteSize: budget })
      );
    }
    record.viewer.setLayers(layers);
  }

  // ---- the colony grid -------------------------------------------------

  /**
   * Pack `cells` into a uniform grid of `size`-px viewports.
   *
   * Uniform on purpose: the colony grid crops every colony to one
   * `max_size` square so the tiles share a canvas, and a per-cell size
   * would make the packing depend on iteration order.
   */
  function gridLayout(cells, el, options) {
    const first = cells.length ? cells[0] : null;
    const size = Math.max(
      1,
      Math.round(options.cellSize || (first && first.size) || 64)
    );
    const gap = options.gap === undefined ? GRID_GAP_PX : options.gap;
    const available =
      (el && el.clientWidth) || cells.length * (size + gap) || size;
    const columns = Math.max(
      1,
      options.columns || Math.floor((available + gap) / (size + gap))
    );
    return cells.map((cell, index) => ({
      x: (index % columns) * (size + gap),
      y: Math.floor(index / columns) * (size + gap),
      w: size,
      h: size,
    }));
  }

  /** Push this instance's single shared viewState at deck.gl. */
  function applyGridViewState(record) {
    record.viewer.setViewState({
      [GRID_VIEW_STATE_ID]: { ...record.grid.shared },
    });
  }

  /**
   * Render one `OrthographicView` per colony over ONE shared `viewState`.
   *
   * @param {string} containerId Id of a mounted instance.
   * @param {Array<{id: string|number, centroidRr: number,
   *   centroidCc: number, size?: number}>} cells One entry per colony, in
   *   the grid's own reading order. `centroidRr`/`centroidCc` are STORE
   *   pixel coordinates; the view's `target` is `[cc, rr, 0]` because
   *   deck.gl's target is `[x, y, z]`.
   * @param {{zoom?: number}} [sharedViewState] The one dynamic view state.
   *   Only `zoom` is meaningful -- every cell's `target` overrides.
   * @param {{cellSize?: number, gap?: number, columns?: number}} [opts]
   *   Packing. `columns` defaults to what fits the container's width.
   *
   * CONTROLLER DECISION, stated rather than implied: every cell view
   * carries `controller: true`. Two reasons. (1) The vendored bundle passes
   * `controller: true` at the DECK level, and deck.gl's backward-compat
   * path force-assigns that onto `views[0]` alone
   * (`core/src/lib/deck.ts:1240-1242`) -- so leaving the views without one
   * makes exactly one cell behave differently from the other N-1, which is
   * worse than either uniform choice. (2) Without a controller the "Shared
   * camera" lock would have nothing to constrain: zoom would be
   * programmatic-only and no gesture could ever move it.
   *
   * What the gesture is allowed to change is `zoom` and ONLY `zoom` -- the
   * facade's `onViewStateChange` projects the controller's output down to
   * that one number. Pan is discarded by construction, because each view's
   * `target` override wins on every render, so a cell cannot be dragged off
   * its colony.
   */
  async function setGridViews(containerId, cells, sharedViewState, opts) {
    const bundle = await ready();
    const record = requireInstance(containerId);
    const list = Array.from(cells || []);
    const options = opts || {};
    if (!list.length) {
      record.grid = null;
      record.viewer.setViews([
        new bundle.deck.OrthographicView({ id: "ortho", controller: true }),
      ]);
      if (record.loaded) rebuildLayers(record, bundle);
      return 0;
    }
    const layout = gridLayout(list, document.getElementById(containerId), options);
    const views = list.map(
      (cell, index) =>
        new bundle.deck.OrthographicView({
          id: `${GRID_CELL_VIEW_PREFIX}${cell.id}`,
          x: layout[index].x,
          y: layout[index].y,
          width: layout[index].w,
          height: layout[index].h,
          controller: true,
          // Overrides ONLY `target`, merging over the shared zoom. A `View`
          // carries no target of its own -- `target` and `zoom` both live
          // in the viewState -- so building this literally as "one View per
          // cell plus one shared viewState" without the merge renders the
          // SAME colony N times.
          viewState: {
            id: GRID_VIEW_STATE_ID,
            target: [Number(cell.centroidCc), Number(cell.centroidRr), 0],
          },
        })
    );
    record.grid = {
      cells: list,
      views,
      shared: { zoom: 0, ...(sharedViewState || {}) },
    };
    record.viewer.setViews(views);
    applyGridViewState(record);
    if (record.loaded) rebuildLayers(record, bundle);
    return views.length;
  }

  /**
   * TEST SEAM -- not API. The view state deck.gl ACTUALLY rendered each
   * cell with, read off the live `Viewport`s rather than recomputed.
   *
   * Returns `[{id, zoom, target}, ...]` in the grid's own cell order.
   * `page.evaluate` marshals these into Python DICTS, so read them as
   * `s["zoom"]` and `s["target"]`.
   */
  function __debugViewStates(containerId) {
    const record = requireInstance(containerId);
    if (!record.grid) return [];
    const byId = new Map(
      record.viewer.deck.getViewports().map((viewport) => [viewport.id, viewport])
    );
    return record.grid.views
      .map((view) => byId.get(view.id))
      .filter(Boolean)
      .map((viewport) => ({
        id: viewport.id,
        zoom: viewport.zoom,
        target: Array.from(viewport.target),
      }));
  }

  /**
   * Report the pyramid level deck.gl is ACTUALLY serving.
   *
   * Not computed. `MultiscaleImageLayer` resolves a tile as
   * `loader[Math.round(-z)].getTile(config)` (`@vivjs/layers` index.mjs:951,
   * vendored beside the spec), so the only honest observation of "the level
   * being served" is which element of the loader array is asked for a tile.
   * This wraps each one.
   *
   * A server-side number -- `select_pyramid_level` over the same target
   * pixel size -- would name a level nobody rendered, and a readout labelled
   * "the level actually being served" is trusted precisely when diagnosing
   * the bug it would be misreporting.
   */
  function instrumentServedLevel(record, loaded) {
    const notify = record.options.onLevelChange;
    if (!notify || !loaded.image) return;
    const sources = loaded.image.data;
    sources.forEach((source, level) => {
      if (source.__phenotypicLevelProbe) return;
      const original = source.getTile.bind(source);
      // An OWN property shadowing `ZarrPixelSource.prototype.getTile`; the
      // prototype is shared across every source this page opens, so patching
      // it there would cross-report between cards.
      source.getTile = (config) => {
        if (record.level !== level) {
          record.level = level;
          const shape = source.shape;
          try {
            notify({
              level,
              levels: sources.length,
              height: shape[shape.length - 2],
              width: shape[shape.length - 1],
              tileSize: source.tileSize,
            });
          } catch (ignored) {
            /* a readout must not break a tile fetch */
          }
        }
        return original(config);
      };
      source.__phenotypicLevelProbe = true;
    });
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
   *   (`initialViewState`, `views`, `onViewStateChange`), plus three facade
   *   options: `buildLayers(bundle, loaded, spec, opacity)` replacing the
   *   default layer policy, `refetchSource(containerId)` returning a fresh
   *   spec after a re-promote, and `onLevelChange({level, levels, height,
   *   width, tileSize})` fired when the pyramid level deck.gl is serving
   *   changes.
   */
  async function mount(containerId, opts) {
    const bundle = await ready();
    const el = document.getElementById(containerId);
    if (!el) throw new Error(`viv: no element #${containerId}`);
    const options = opts || {};
    // The vendored viewer's own `onViewStateChange` runs FIRST and does
    // `deck.setProps({viewState: next})` with the raw per-view state the
    // controller produced -- which in grid mode replaces the keyed
    // `{colony-shared: {...}}` object with an unkeyed one, and every view
    // then falls back to reading the whole object as its state. Restoring
    // the shared entry here happens synchronously inside the same handler
    // call, before deck.gl's next animation frame, so no frame is drawn
    // from the clobbered state.
    //
    // Only `zoom` is taken. That is the whole camera design in one line:
    // one number, projected out of whatever the controller produced.
    const viewerOptions = {
      ...options,
      onViewStateChange: (next) => {
        const current = instances.get(containerId);
        if (current && current.grid && next) {
          current.grid.shared = { ...current.grid.shared, zoom: next.zoom };
          applyGridViewState(current);
        }
        if (options.onViewStateChange) options.onViewStateChange(next);
      },
    };
    const viewer = bundle.createViewer(el, viewerOptions);
    instances.set(containerId, {
      viewer,
      options,
      spec: null,
      loaded: null,
      resourcing: null,
      // Per-layer opacity, addressed by the same ids `setLayerVisibility`
      // uses. Kept on the RECORD rather than on the layer descriptors so it
      // survives the wholesale rebuild every `setSource` performs.
      opacity: {},
      level: null,
      // Grid mode's record, or null for a single-view surface (the Plate).
      grid: null,
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
    record.level = null;
    instrumentServedLevel(record, loaded);
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

  /**
   * Set the view state.
   *
   * In GRID mode this updates the single `colony-shared` entry -- there is
   * exactly one, so nothing is fanned across views and nothing can drift.
   * A `target` passed here is stored but has no visible effect: every
   * cell's own `target` override wins the merge.
   */
  function setViewState(containerId, viewState) {
    const record = instances.get(containerId);
    if (!record) return;
    if (record.grid) {
      record.grid.shared = { ...record.grid.shared, ...(viewState || {}) };
      delete record.grid.shared.id;
      applyGridViewState(record);
      return;
    }
    record.viewer.setViewState(viewState);
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

  /**
   * Set one layer's opacity (`IMAGE_LAYER_ID` / `LABEL_LAYER_ID`).
   *
   * Rebuilds rather than mutating the live layer: deck.gl layer props are
   * immutable once handed over, and the composite `MultiscaleImageLayer`
   * additionally forwards `opacity` into its `refinementStrategy` choice
   * (`@vivjs/layers` index.mjs:997), so an in-place poke would leave the
   * tiling strategy disagreeing with the opacity it was chosen for.
   */
  async function setLayerOpacity(containerId, name, opacity) {
    const record = instances.get(containerId);
    if (!record || !record.loaded) return;
    record.opacity[name] = Math.max(0, Math.min(1, Number(opacity)));
    rebuildLayers(record, await ready());
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
    setGridViews,
    setLayerVisibility,
    setLayerOpacity,
    destroy,
    errors: { StaleGenerationError, StoreUnreadableError },
    IMAGE_LAYER_ID,
    LABEL_LAYER_ID,
    GRID_VIEW_STATE_ID,
    __debugReadChunk,
    __debugViewStates,
  };
})();
