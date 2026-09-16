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

  /** Half-stop microscope zoom step: two clicks double display scale. */
  const GRID_ZOOM_STEP = 0.5;

  /** One D-pad press moves one tenth of the currently visible source span. */
  const GRID_PAN_FRACTION = 0.1;

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
   * Image series use one primary colour per channel. The objmap is not an
   * intensity image: it uses a saturated cyclic hue map with label 0 fully
   * transparent, nearest-neighbour sampling, and the configured opacity over
   * the image. Treating label ids as one tinted channel makes the objmap look
   * like a second grayscale image and visually merges adjacent objects.
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
      const labelMultiscale = label.data.length > 1;
      const requestedDomain = spec && spec.labelColorDomain;
      const labelDomain =
        Array.isArray(requestedDomain) &&
        requestedDomain.length === 2 &&
        Number.isFinite(Number(requestedDomain[0])) &&
        Number.isFinite(Number(requestedDomain[1])) &&
        Number(requestedDomain[1]) > Number(requestedDomain[0])
          ? requestedDomain.map(Number)
          : [0, 255];
      const LabelClass = labelMultiscale
        ? bundle.layers.MultiscaleImageLayer
        : bundle.layers.ImageLayer;
      layers.push(
        new LabelClass({
          id: LABEL_LAYER_ID,
          loader: labelMultiscale ? label.data : label.data[0],
          selections: [{}],
          contrastLimits: [labelDomain],
          channelsVisible: [true],
          colormap: "hsv",
          extensions: [new bundle.extensions.AdditiveColormapExtension()],
          useTransparentColor: true,
          interpolation: "nearest",
          opacity: opacityFor(LABEL_LAYER_ID, 0.5),
        })
      );
    }
    return layers;
  }

  // ---- public surface -------------------------------------------------

  /** Spatial chunk keys intersecting fixed object-centred ROIs at one level. */
  function roiChunkKeys(level, baseLevel, cells, cropSize) {
    const keys = new Set();
    const shape = level.shape;
    const height = Number(shape[shape.length - 2]);
    const width = Number(shape[shape.length - 1]);
    const baseShape = baseLevel.shape;
    const baseHeight = Number(baseShape[baseShape.length - 2]);
    const baseWidth = Number(baseShape[baseShape.length - 1]);
    const tileValue = level.tileSize || 1024;
    const tileHeight = Array.isArray(tileValue)
      ? Number(tileValue[tileValue.length - 2])
      : Number(tileValue);
    const tileWidth = Array.isArray(tileValue)
      ? Number(tileValue[tileValue.length - 1])
      : Number(tileValue);
    const scaleY = baseHeight / height;
    const scaleX = baseWidth / width;
    const halfHeight = Number(cropSize) / (2 * scaleY);
    const halfWidth = Number(cropSize) / (2 * scaleX);
    if (
      !(height > 0 && width > 0 && tileHeight > 0 && tileWidth > 0) ||
      !(halfHeight > 0 && halfWidth > 0)
    ) return keys;

    cells.forEach((cell) => {
      const cx = Number(cell.centroidCc) / scaleX;
      const cy = Number(cell.centroidRr) / scaleY;
      if (!Number.isFinite(cx) || !Number.isFinite(cy)) return;
      const left = Math.max(0, cx - halfWidth);
      const right = Math.min(width, cx + halfWidth);
      const top = Math.max(0, cy - halfHeight);
      const bottom = Math.min(height, cy + halfHeight);
      if (!(right > left && bottom > top)) return;
      const x0 = Math.floor(left / tileWidth);
      const x1 = Math.floor((right - Number.EPSILON) / tileWidth);
      const y0 = Math.floor(top / tileHeight);
      const y1 = Math.floor((bottom - Number.EPSILON) / tileHeight);
      for (let yy = y0; yy <= y1; yy += 1) {
        for (let xx = x0; xx <= x1; xx += 1) {
          keys.add(`${yy}:${xx}`);
        }
      }
    });
    return keys;
  }

  /**
   * Deduplicated mounted-ROI working set at the active pyramid level.
   *
   * Crop size determines the common comparison window; it does NOT by
   * itself determine I/O. A crop crossing a chunk corner needs four chunks,
   * while overlapping crops from the same store share entries. Count the
   * union for this source group, separately for image and label layers.
   */
  function gridTileCacheEntries(loaded, cells, grid) {
    const entries = { image: 1, label: 1 };
    for (const [name, source] of [
      ["image", loaded.image],
      ["label", loaded.label],
    ]) {
      if (!source || !source.data.length) continue;
      const levelIndex = Math.max(
        0,
        Math.min(
          source.data.length - 1,
          Math.round(-Number(grid.shared.zoom || 0))
        )
      );
      entries[name] = Math.max(1, roiChunkKeys(
        source.data[levelIndex],
        source.data[0],
        cells,
        grid.cropSize
      ).size);
    }
    return entries;
  }

  /** Rebuild and install this instance's layers from its loaded sources. */
  function rebuildLayers(record, bundle) {
    const build = record.options.buildLayers || defaultLayers;
    let layers = build(bundle, record.loaded, record.spec, record.opacity);
    if (record.grid && record.loaded) {
      // Cloned rather than passed into the builder, so a caller-supplied
      // `buildLayers` (the Plate's) gets the bound too without knowing
      // about grid mode.
      const budgets = gridTileCacheEntries(
        record.loaded,
        record.grid.cells,
        record.grid
      );
      layers = layers.map((layer) =>
        layer.clone({
          maxCacheSize:
            layer.id === LABEL_LAYER_ID ? budgets.label : budgets.image,
        })
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
    if (
      cells.length &&
      cells.every((cell) =>
        Number.isFinite(cell.x) && Number.isFinite(cell.y) &&
        Number.isFinite(cell.width) && Number.isFinite(cell.height)
      )
    ) {
      return cells.map((cell) => ({
        x: Number(cell.x),
        y: Number(cell.y),
        w: Number(cell.width),
        h: Number(cell.height),
      }));
    }
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

  /** Clamp the shared camera to its fixed object-centred crop region. */
  function clampGridCamera(grid) {
    const shared = grid.shared;
    const fitZoom = Number(grid.fitZoom);
    const zoomOffset = Math.max(
      0,
      Math.min(-fitZoom, Number(shared.zoomOffset || 0))
    );
    shared.zoomOffset = zoomOffset;
    shared.zoom = fitZoom + zoomOffset;
    const scale = 2 ** shared.zoom;
    const viewportWidth = Math.min(...grid.layout.map((box) => box.w));
    const viewportHeight = Math.min(...grid.layout.map((box) => box.h));
    const visibleWidth = viewportWidth / scale;
    const visibleHeight = viewportHeight / scale;
    const maxX = Math.max(0, (grid.cropSize - visibleWidth) / 2);
    const maxY = Math.max(0, (grid.cropSize - visibleHeight) / 2);
    shared.offsetX = Math.max(
      -maxX,
      Math.min(maxX, Number(shared.offsetX || 0))
    );
    shared.offsetY = Math.max(
      -maxY,
      Math.min(maxY, Number(shared.offsetY || 0))
    );
    grid.bounds = { maxX, maxY, visibleWidth, visibleHeight };
  }

  /** Push only the dynamic zoom into the one shared deck.gl viewState. */
  function applyGridViewState(record) {
    record.viewer.setViewState({
      [GRID_VIEW_STATE_ID]: { zoom: record.grid.shared.zoom },
    });
  }

  /** Rebuild passive cell views from DOM geometry and shared camera offset. */
  function installGridViews(bundle, record) {
    const grid = record.grid;
    clampGridCamera(grid);
    // The bundle creates Deck with a top-level controller for ordinary
    // single views. deck.gl otherwise assigns it to the first grid view,
    // overriding that view's passive controller setting.
    record.viewer.deck.setProps({ controller: false });
    grid.views = grid.cells.map(
      (cell, index) =>
        new bundle.deck.OrthographicView({
          id: `${GRID_CELL_VIEW_PREFIX}${cell.id}`,
          x: grid.layout[index].x,
          y: grid.layout[index].y,
          width: grid.layout[index].w,
          height: grid.layout[index].h,
          // Comparison tiles are passive. The microscope-stage toolbar owns
          // one bounded camera, so hundreds of competing drag controllers do
          // not run only to have their pan targets discarded.
          controller: false,
          viewState: {
            id: GRID_VIEW_STATE_ID,
            target: [
              Number(cell.centroidCc) + grid.shared.offsetX,
              Number(cell.centroidRr) + grid.shared.offsetY,
              0,
            ],
          },
        })
    );
    record.viewer.setViews(grid.views);
    applyGridViewState(record);
  }

  /** Install source-specific grid layers using the mounted ROI cache union. */
  function installGridSourceLayers(bundle, record) {
    if (!record.grid || !record.grid.sources) return;
    const sourceForView = new Map();
    record.grid.cells.forEach((cell) => {
      const viewId = `${GRID_CELL_VIEW_PREFIX}${cell.id}`;
      sourceForView.set(viewId, viewId);
    });
    const sourceForLayer = new Map();
    const layers = [];
    let groupIndex = 0;
    record.grid.sources.forEach((group) => {
      const built = defaultLayers(bundle, group.loaded, group.spec, null);
      const budgets = gridTileCacheEntries(
        group.loaded,
        group.cells,
        record.grid
      );
      group.cells.forEach((cell, cellIndex) => {
        const viewId = `${GRID_CELL_VIEW_PREFIX}${cell.id}`;
        built.forEach((layer) => {
          const clone = layer.clone({
            id: `colony-source-${groupIndex}-cell-${cellIndex}-${layer.id}`,
            viewportId: viewId,
            maxCacheSize:
              layer.id === LABEL_LAYER_ID ? budgets.label : budgets.image,
          });
          sourceForLayer.set(clone.id, viewId);
          layers.push(clone);
        });
      });
      groupIndex += 1;
    });
    const sourceKeyForLayer = (layer) => {
      const direct = sourceForLayer.get(layer.id);
      if (direct !== undefined) return direct;
      // deck.gl applies layerFilter to generated sublayers too. Viv prefixes
      // each tiled/background sublayer id with its parent composite id; an
      // exact-only lookup rejects the pixels after accepting the descriptor.
      for (const [parentId, key] of sourceForLayer.entries()) {
        if (layer.id.startsWith(`${parentId}-`)) return key;
      }
      return null;
    };
    record.viewer.deck.setProps({
      layerFilter: ({ layer, viewport }) =>
        sourceKeyForLayer(layer) === sourceForView.get(viewport.id),
    });
    record.viewer.setLayers(layers);
  }

  /** Public state used by the toolbar to render limits and scale. */
  function getGridCameraState(containerId) {
    const record = requireInstance(containerId);
    if (!record.grid) return null;
    const grid = record.grid;
    clampGridCamera(grid);
    const epsilon = 1e-9;
    return {
      zoom: grid.shared.zoom,
      fitZoom: grid.fitZoom,
      zoomOffset: grid.shared.zoomOffset,
      zoomPercent: Math.round(100 * (2 ** grid.shared.zoom)),
      offsetX: grid.shared.offsetX,
      offsetY: grid.shared.offsetY,
      canPan: grid.bounds.maxX > epsilon || grid.bounds.maxY > epsilon,
      canZoomOut: grid.shared.zoomOffset > epsilon,
      canZoomIn: grid.shared.zoom < -epsilon,
      cropSize: grid.cropSize,
      activeLevel: grid.sources
        ? Math.max(0, Math.round(-grid.shared.zoom))
        : 0,
    };
  }

  /** Apply one linked, bounded microscope-stage command to every tile. */
  async function setGridCamera(containerId, command) {
    const bundle = await ready();
    const record = requireInstance(containerId);
    if (!record.grid) return null;
    const grid = record.grid;
    const previousLevel = Math.max(0, Math.round(-grid.shared.zoom));
    const action = command || {};
    const kind = String(action.action || "");

    if (kind === "fit") {
      grid.shared.zoomOffset = 0;
      grid.shared.offsetX = 0;
      grid.shared.offsetY = 0;
    } else if (kind === "center") {
      grid.shared.offsetX = 0;
      grid.shared.offsetY = 0;
    } else if (kind === "oneToOne") {
      grid.shared.zoomOffset = -grid.fitZoom;
      grid.shared.offsetX = 0;
      grid.shared.offsetY = 0;
    } else if (kind === "zoom") {
      grid.shared.zoomOffset += Number(action.delta || 0);
    } else if (kind === "pan") {
      clampGridCamera(grid);
      const stepX = grid.bounds.visibleWidth * GRID_PAN_FRACTION;
      const stepY = grid.bounds.visibleHeight * GRID_PAN_FRACTION;
      grid.shared.offsetX += Number(action.dx || 0) * stepX;
      grid.shared.offsetY += Number(action.dy || 0) * stepY;
    }

    installGridViews(bundle, record);
    const nextLevel = Math.max(0, Math.round(-grid.shared.zoom));
    if (grid.sources && nextLevel !== previousLevel) {
      installGridSourceLayers(bundle, record);
    }
    return getGridCameraState(containerId);
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
   * @param {{zoomOffset?: number, offsetX?: number, offsetY?: number}}
   *   [sharedViewState] Optional shared camera override.
   * @param {{cellSize?: number, gap?: number, columns?: number,
   *   cropSize?: number}} [opts] DOM packing fallback plus the uniform
   *   source-pixel crop side.
   */
  async function setGridViews(containerId, cells, sharedViewState, opts) {
    const bundle = await ready();
    const record = requireInstance(containerId);
    const list = Array.from(cells || []);
    const options = opts || {};
    if (!list.length) {
      record.grid = null;
      record.viewer.deck.setProps({ controller: true });
      record.viewer.setViews([
        new bundle.deck.OrthographicView({ id: "ortho", controller: true }),
      ]);
      if (record.loaded) rebuildLayers(record, bundle);
      return 0;
    }
    const layout = gridLayout(
      list,
      document.getElementById(containerId),
      options
    );
    const previous = record.grid;
    const previousLevel = previous
      ? Math.max(0, Math.round(-Number(previous.shared.zoom || 0)))
      : null;
    const cropCandidate = Number(
      options.cropSize || (previous && previous.cropSize) || 64
    );
    const cropSize = cropCandidate > 0 ? cropCandidate : 64;
    const viewportSide = Math.min(
      ...layout.map((box) => Math.min(box.w, box.h))
    );
    const fitZoom = Math.min(0, Math.log2(viewportSide / cropSize));
    const preserve = previous && sharedViewState == null;
    const shared = preserve
      ? { ...previous.shared }
      : {
          zoomOffset: 0,
          offsetX: 0,
          offsetY: 0,
          ...(sharedViewState || {}),
        };
    if (
      Number.isFinite(Number(shared.zoom)) &&
      !Number.isFinite(Number(shared.zoomOffset))
    ) {
      shared.zoomOffset = Number(shared.zoom) - fitZoom;
    }
    record.grid = {
      cells: list,
      layout,
      views: [],
      shared,
      fitZoom,
      cropSize,
      sources: previous && previous.sources,
    };
    installGridViews(bundle, record);
    const nextLevel = Math.max(
      0,
      Math.round(-Number(record.grid.shared.zoom || 0))
    );
    if (record.grid.sources && nextLevel !== previousLevel) {
      installGridSourceLayers(bundle, record);
    } else if (record.loaded) {
      rebuildLayers(record, bundle);
    }
    return record.grid.views.length;
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
      resourcingEpoch: null,
      // Every setSource claims a new epoch before its first await. Deferred
      // reads from an older source may still finish, but cannot commit into
      // this record after a newer source or destroy has invalidated them.
      sourceEpoch: 0,
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
    const record = requireInstance(containerId);
    const epoch = record.sourceEpoch + 1;
    record.sourceEpoch = epoch;
    const isCurrent = () => (
      instances.get(containerId) === record && record.sourceEpoch === epoch
    );
    if (!spec || !spec.storeUrl || !spec.seriesPath) {
      throw new Error(
        "viv: source spec needs storeUrl and seriesPath (both resolved " +
          "server-side)"
      );
    }
    const bundle = await ready();
    if (!isCurrent()) return undefined;
    record.viewer.deck.setProps({ layerFilter: null });
    const hooks = {
      onStale: () => {
        if (isCurrent()) resourceAfterPromote(containerId, record, epoch);
      },
    };
    const image = await bundle.viv.loadOmeZarrFromStore(
      createByteRouteStore(joinUrl(spec.storeUrl, spec.seriesPath), hooks)
    );
    if (!isCurrent()) return undefined;
    let label = null;
    if (spec.labelPath) {
      label = await bundle.viv.loadOmeZarrFromStore(
        createByteRouteStore(joinUrl(spec.storeUrl, spec.labelPath), hooks)
      );
      if (!isCurrent()) return undefined;
    }
    const loaded = { image, label };
    if (!isCurrent()) return undefined;
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
   * Load and render a Colony grid whose cells may come from different stores.
   *
   * Sources are loaded once per unique store. Each cell receives a lightweight
   * layer descriptor bound through Viv's `viewportId`, because one TileLayer
   * tileset cannot be updated from several deck.gl viewports. The descriptors
   * share the loaded sources and use the store's mounted-ROI union as their
   * cache bound; `layerFilter` routes each descriptor to exactly one view.
   */
  async function setGridSources(containerId, cells, sharedViewState, opts) {
    const bundle = await ready();
    const record = requireInstance(containerId);
    const epoch = (record.gridSourceEpoch || 0) + 1;
    record.gridSourceEpoch = epoch;
    record.loaded = null;
    record.spec = null;

    const list = Array.from(cells || []).filter(
      (cell) => cell && cell.spec && cell.spec.storeUrl && cell.spec.seriesPath
    );
    const groups = new Map();
    list.forEach((cell) => {
      const spec = cell.spec;
      const key = [spec.storeUrl, spec.seriesPath, spec.labelPath || ""].join(" ");
      cell.sourceKey = key;
      if (!groups.has(key)) groups.set(key, { spec, cells: [], loaded: null });
      groups.get(key).cells.push(cell);
    });

    await Promise.all(Array.from(groups.values()).map(async (group) => {
      const hooks = { onStale: () => null };
      const image = await bundle.viv.loadOmeZarrFromStore(
        createByteRouteStore(
          joinUrl(group.spec.storeUrl, group.spec.seriesPath), hooks
        )
      );
      let label = null;
      if (group.spec.labelPath) {
        label = await bundle.viv.loadOmeZarrFromStore(
          createByteRouteStore(
            joinUrl(group.spec.storeUrl, group.spec.labelPath), hooks
          )
        );
      }
      group.loaded = { image, label };
    }));

    if (instances.get(containerId) !== record || record.gridSourceEpoch !== epoch) {
      return 0;
    }
    if (record.grid) record.grid.sources = null;
    await setGridViews(containerId, list, sharedViewState, opts);

    record.grid.sources = groups;
    installGridSourceLayers(bundle, record);
    return list.length;
  }

  /**
   * Re-fetch the spec and re-source after a re-promote (a 409).
   *
   * Coalesced: a pan issues many concurrent chunk reads, and every one of
   * them reports the same stale token. Without the in-flight guard a single
   * promote would trigger dozens of duplicate re-sources.
   */
  function resourceAfterPromote(containerId, record, sourceEpoch) {
    if (
      instances.get(containerId) !== record ||
      record.sourceEpoch !== sourceEpoch
    ) return null;
    if (
      record.resourcing && record.resourcingEpoch === sourceEpoch
    ) return record.resourcing;
    const refetch = record.options.refetchSource;
    if (!refetch) {
      // No recovery path was wired. The read still throws
      // `StaleGenerationError` -- it is simply not repaired here.
      return null;
    }
    const recovery = Promise.resolve(refetch(containerId))
      .then((fresh) => {
        if (
          instances.get(containerId) !== record ||
          record.sourceEpoch !== sourceEpoch
        ) return undefined;
        return setSource(containerId, fresh);
      })
      .finally(() => {
        if (record.resourcing === recovery) {
          record.resourcing = null;
          record.resourcingEpoch = null;
        }
      });
    record.resourcing = recovery;
    record.resourcingEpoch = sourceEpoch;
    return recovery;
  }

  /**
   * Set the view state.
   *
   * In GRID mode this updates the single `colony-shared` entry -- there is
   * exactly one, so nothing is fanned across views and nothing can drift.
   * The Colony toolbar should use `setGridCamera`, which owns the bounded
   * offset model as well as zoom. This compatibility entry point retains
   * direct zoom updates for existing callers.
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
    record.sourceEpoch += 1;
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
    setGridSources,
    setGridCamera,
    getGridCameraState,
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
