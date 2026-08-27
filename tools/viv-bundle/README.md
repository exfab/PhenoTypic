# Viv bundle build recipe

Builds the vendored browser bundle the results viewer loads:

    src/phenotypic/gui/results_viewer/_assets/viv/viv-bundle.min.js

Built **outside** this repo — there is no npm in CI, by design (viewer-viv-rebuild
spec §3, plan Global Constraints). Run this by hand when upgrading Viv, then commit
the artifact.

## Recipe

    cd tools/viv-bundle
    npm ci             # lockfile is pinned; never `npm install`
    node build.mjs     # writes ../../src/phenotypic/gui/results_viewer/_assets/viv/viv-bundle.min.js

Then bump `VERSION` to match `package.json`'s pinned versions and commit both the
artifact and `VERSION`. The GUI logs `VERSION` at startup; a mismatch between it and
the string embedded in the bundle is the only signal that the artifact is stale.
Nothing *fails* on drift — see spec §10, open question 3.

`npm install` is used exactly once in this recipe's life: to *create*
`package-lock.json`. Every build after that is `npm ci`, which installs the locked
tree and fails rather than silently resolving something newer. If you must add a
dependency, edit `package.json`, run `npm install --package-lock-only`, and commit
the lockfile change as its own reviewable diff.

## Recorded build environment

Last built 2026-08-27 on UCR HPCC (Linux x86_64, `r32`):

| | |
|---|---|
| node | v24.16.0 |
| npm | 11.13.0 |
| command | `npm ci && node build.mjs` |
| artifact size | 2,616,449 B (2.50 MiB) |

Record the node version here whenever the artifact is rebuilt. With no npm in CI,
this table plus the lockfile is the whole provenance of a committed binary.

## What is in the bundle

| Package | Version | Why |
|---|---|---|
| `@vivjs/loaders` | 0.22.1 | `loadOmeZarr`, `ZarrPixelSource` |
| `@vivjs/layers` | 0.22.1 | `MultiscaleImageLayer`, `ImageLayer`, `XRLayer` |
| `@vivjs/extensions` | 0.22.1 | `ColorPaletteExtension`, `LensExtension` |
| `@deck.gl/*` | 9.3.10 | `Deck`, `OrthographicView`, tiling |
| `@luma.gl/*` | 9.3.6 | GPU abstraction under deck.gl |
| `zarrita` | 0.5.4 | Zarr v3 client, sharding codec, codec registry |
| `numcodecs` | 0.3.2 | wasm zstd / blosc / gzip |
| `esbuild` | 0.28.2 | bundler (dev only) |

`zarrita` is pinned to **0.5.4 on purpose**: `@vivjs/loaders@0.22.1` depends on
`zarrita@^0.5.4`, and the codec registry is *module state*. If npm resolved two
copies, `registry.set("zstd", ...)` in `entry.mjs` would write to a registry the
loaders never read, and every chunk read would fail with `Unknown codec: zstd`
while the registration looked fine. Verify after any bump:

    find node_modules -name package.json -path '*zarrita*' -not -path '*/dist/*'

must list exactly one `zarrita`.

**React is deliberately absent.** `@vivjs/viewers` is a React component set;
this bundle uses the loaders and layers and drives deck.gl's imperative `Deck`
directly. `@deck.gl/react` and `react` are not dependencies.

## What the bundle exposes

One global, `window.__vivBundle`:

| Key | Contents |
|---|---|
| `VERSION` | the contents of `VERSION`, stamped in at build time |
| `zarr` | the zarrita namespace — `registry`, `root`, `open`, `get`, `slice`, `FetchStore` |
| `numcodecs` | `{ Zstd, Blosc, Gzip }` |
| `viv` | `loadOmeZarr`, `loadOmeZarrFromStore`, `ZarrPixelSource`, `getImageSize`, … |
| `layers` | `MultiscaleImageLayer`, `ImageLayer`, `XRLayer`, `XR3DLayer` |
| `extensions` | `ColorPaletteExtension`, `LensExtension` |
| `deck` | `Deck`, `OrthographicView`, `COORDINATE_SYSTEM` |
| `createViewer(el, opts)` | thin imperative deck.gl wrapper the façade builds on |

The façade (`_assets/viv_viewer.js`) is the only thing that touches this global.

## The ordering rule (decision C1)

`entry.mjs` calls `zarr.registry.set("zstd", () => Zstd)` at module-evaluation
time — before `window.__vivBundle` is assigned, so before any consumer can open a
store. zarrita 0.5.4 already ships a *lazy* zstd entry (`() => import("numcodecs/zstd")`);
the eager one replaces it because a dynamic import is a code-split point an IIFE
cannot honour, and because eager registration makes the ordering observable rather
than incidental.

Verified 2026-08-27: deleting `zstd` from the registry and then reading a chunk
throws `Unknown codec: zstd` — the read fails rather than returning `fill_value`
zeros. See the spike findings.

## Range requests are not optional

The PhenoTypic store is **sharded** (`[3,4096,4096]` shards of `[1,1024,1024]`
chunks). zarrita's sharding codec asserts `store.getRange` and issues two ranged
GETs per cold tile: a suffix read of the shard index, then the inner chunk. A
server without `Range` returns the whole shard for both. On the 4000×3000
reference plate that is 36,045,031 B twice instead of 772 B + 1,048,609 B.

`python -m http.server` has no `Range` support. Never serve a store with it.
