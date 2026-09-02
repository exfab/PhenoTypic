# Spike gate findings — phase 0

Run 2026-08-27 against the landed CLI writer. Environment: node v24.16.0, npm 11.13.0,
Playwright chromium present, npm registry reachable, `zarr` 3.1.5.

## Task 0.1 — a store written by the real writer

Two stores, because the first turned out to be an invalid subject:

| Store | Source | Extent | Levels |
|---|---|---|---|
| `plate.ome.zarr` | `load_synth_yeast_plate()` | 800×600 | 2 |
| `big.ome.zarr` | synthetic `GridImage(nrows=32, ncols=48)` | **4000×3000** | **4** |

**The synth yeast plate cannot serve as the spike subject.** At 800×600 the chunk clamps to
the whole image (`chunk_shape [3,600,800]`), so it exercises neither the 1024² inner chunk
nor sharding. Spec §1.4's reference plate is 4000×3000; that is what the numbers below use.

### Backend §1.4 — every claim confirmed on the reference plate

```text
shape          [3, 3000, 4000]
shard shape    [3, 4096, 4096]      <- §1.4
inner chunk    [1, 1024, 1024]      <- §1.4
inner codecs   bytes, zstd          <- decision C1
key encoding   {"name": "default", "configuration": {"separator": "."}}
pyramid        levels 4, stop_px 512, downsample {image: mean, label: nearest}
```

`levels 4` at 4000 px matches `ceil(log2(4000/512)) + 1`, the ladder whose `floor`/`ceil`
boundary backend §1.3 records as having failed once.

### Q3 — does the `"."` separator round-trip? **ANSWERED: yes.**

```text
rgb/0/c.0.0.0     3-D  (channel, y, x)
gray/0/c.0.0      2-D  (y, x)
```

Confirms the plan's round-2 corrected literals (FLOW-13). One path segment per chunk key, as
backend §1.4 requires for Windows `MAX_PATH`.

---

## FINDING — spec §1.4's "verified" file-count table is wrong

§1.4 states as **verified** that 4 levels (auto) yields **40** files: 16 data + 24 metadata.
Measured on the reference plate:

```text
total 36  =  12 data  +  23 zarr.json  +  1 METADATA.ome.xml
```

**The gap is the objmap.** Zarr does not write a chunk whose contents equal `fill_value`, and
a Stage-1 objmap is all zeros — so the label levels contain **zero** chunk files, not four:

```text
rgb            L0=1 L1=1 L2=1 L3=1
gray           L0=1 L1=1 L2=1 L3=1
detect_mat     L0=1 L1=1 L2=1 L3=1
labels/objmap  L0=0 L1=0 L2=0 L3=0      <- sparse; the table assumed 4
```

Not a defect — sparse storage is correct and strictly better. But the table is presented as
verified and is off by 4 for **every image between Stage 1 and Stage 3**, which backend §3.3
guarantees carries a zeros objmap. The inode budget built on it (~40/image, 400k at 10k
images) is conservative by ~10%.

**Recorded for a backend-spec amendment. Not blocking, and it makes the format cheaper, not
dearer.**

---

## Task 0.2 — Range. The measurement that existed in no document.

One 16-byte ranged request for `big.ome.zarr/rgb/0/c.0.0.0`:

| Server | Code | Downloaded |
|---|---|---|
| `python -m http.server` | `200` | **36,045,031 B** (34.4 MiB) |
| Flask `send_file(conditional=True)` | `206` | **16 B** |

`SimpleHTTPRequestHandler` has **no Range support at all** — it ignores the header and sends
the whole file. This is exactly why round 2 split this step: an earlier draft served the
spike store with `http.server` and expected `206`, which is unreachable, and drove Q1/Q2/Q4
against that same server — measuring the no-Range regime throughout without recording a
single byte count.

### Shard amplification, measured

At 4000×3000 the level-0 `rgb` shard is a **single 34.4 MiB file**. A deck.gl tile fetch
wants one 1024² inner chunk ≈ 3.15 MB (`1024×1024×3×1`):

```text
34.4 MiB / 3.0 MiB  ≈  11.5x  amplification per tile, without Range
```

So `conditional=True` is not a nicety — it is the difference between a 3 MB and a 34 MB tile
fetch. That is the justification phase 1 needs, and the number §5.2's "accepted risk" was
missing.

Below the brief's 96 MB worst case only because the shard clamps to the array; a full
4096×4096 shard over a larger plate would reach it.

---

## Q1, Q2, Q4 — ANSWERED against the real bundle

Run 2026-08-27. Subject: `/tmp/spike/big.ome.zarr` (the 4000×3000 reference plate),
served by `range_server.py` (Flask `send_file(conditional=True)`) on `:8100`.
Client: the **committed** artifact
`src/phenotypic/gui/results_viewer/_assets/viv/viv-bundle.min.js` —
`viv 0.22.1 / deck.gl 9.3.10 / zarrita 0.5.4 / bundle 1`, 2,616,449 B — loaded from the
same origin into Playwright chromium. Not a CDN stand-in.

Bundle preflight, read out of the live page:

```text
typeof window.__vivBundle          "object"
__vivBundle.VERSION                "viv 0.22.1 / deck.gl 9.3.10 / zarrita 0.5.4 / bundle 1"
zarr.registry.has("zstd")          true      <- before any store was opened
registry keys                      blosc, lz4, zstd, gzip, zlib, transpose, bytes,
                                   crc32c, vlen-utf8, json2, bitround
```

### Q1 — does unmodified Viv resolve our `bioformats2raw.layout` series list? **NO.**

`loadOmeZarr(<root>, {type: "multiscales"})` against the store root:

```text
{ ok: false, error: "Node not found: v2 array" }
```

Pointed at a **resolved series** it works with nothing patched:

```text
loadOmeZarr(<root>/rgb)  ->  { ok: true, levels: 4, shape: [3, 3000, 4000],
                               labels: ["c","y","x"], dtype: "Uint8", tileSize: 1024 }
```

**Why.** `@vivjs/loaders`' `loadMultiscales(store, path = "")` opens the group at `path`
and reads `attributes.ome.multiscales`; our root group carries only
`{version, "bioformats2raw.layout": 3}`. With no `multiscales` key it falls back to
`paths = ["0"]` and tries to open `<root>/0` as an array. Our root's children are
`rgb`, `gray`, `detect_mat`, `OME` — hence the error. **Nothing in `@vivjs/loaders`
reads `OME/zarr.json` at all**; the only bioformats2raw path is
`DEPRECATED_loadBioformatsZarr`, which wants numeric series directories and OME-XML,
not our named series. The series list is there and readable — it is simply nobody's
job in Viv:

```text
GET /big.ome.zarr/OME/zarr.json  ->  200  attributes.ome.series = ["rgb","gray","detect_mat"]
```

**Shape of the adaptation: small, and outside Viv.** No Viv patch. Resolve the series
list from `attributes.phenotypic.series` (or `OME/zarr.json`) server-side, pick the
primary (`rgb` when present, else `gray`), and hand `loadOmeZarr` the per-series URL.
That is the resolution the plan's Global Constraints already require us to own, so it
adds a resolver, not a fork.

### Q2 — does `labels/objmap` attach to the primary series automatically? **NO.**

`loadOmeZarr(<root>/rgb)` returns `{data, metadata}` and nothing else:

```text
Object.keys(source)              ["data", "metadata"]
Object.keys(source.metadata)     ["version", "multiscales", "omero"]
"labels" in source.metadata      false
ZarrPixelSource prototype        shape, dtype, getRaster, getTile, onTileError, ...
```

Our store *does* record the child both ways — `rgb/labels/zarr.json` carries
`ome.labels: ["objmap"]`, and the root carries
`attributes.phenotypic.labels = {objmap: "rgb/labels/objmap"}` — but **Viv reads
neither**. There is no label-layer machinery in `@vivjs/loaders`; in vizarr that lives
in the app, not the library.

Handed the **resolved** path from `phenotypic.labels.objmap`, the label group loads as
its own multiscale source with no special-casing:

```text
loadOmeZarr(<root>/rgb/labels/objmap)  ->  { levels: 4, shape: [3000, 4000],
                                             dtype: "Uint16" }
```

**Shape of the adaptation: exactly what the plan already mandates.** The façade resolves
`phenotypic.labels.objmap` (with `.get` — the key is optional) and constructs the label
source explicitly. Since Viv resolves *nothing* by convention here, there is no
`rgb/labels/objmap` hard-coding to fight: a `gray`-primary store works for free.

### Q4 — does the wasm zstd codec decode a CLI-written chunk? **YES — pixel-exact.**

Not "the codec registered". The browser decoded level-0 chunk `c.0.0.0` of `rgb` through
`ZarrPixelSource.getTile({x: 0, y: 0, selection: {c: 0}})` (1024×1024, `Uint8Array`) and
through raw `zarrita.get` with Viv out of the path. Both agree with Python:

```text
python  zarr.open_array('/tmp/spike/big.ome.zarr/rgb/0')[0, :4, :4]
        [[ 94, 216,  14, 214],
         [ 50, 148, 172, 247],
         [ 29, 135, 160, 234],
         [154, 128,  91,  23]]

browser Viv  getTile  rows[0..3][0..3]   -> identical
browser raw  zarrita.get(slice(0,4), slice(0,4)) -> identical (flat, same 16 values)
MATCH: True
```

**And the negative control passes.** Delete the codec, then read:

```text
zarr.registry.delete("zstd"); await zarr.get(rgb/1, ...)  ->  "threw: Unknown codec: zstd"
```

The read **fails** rather than returning `fill_value` zeros — which is the failure mode
that would have made a broken bundle look like an empty plate. `registry.delete` exists,
so plan phase 2 task 2.3 step 4's test is runnable and must not be weakened.

### Bonus: it renders, and sharding forces Range

A `MultiscaleImageLayer` over the same source painted the plate in headless chromium
(SwiftShader). 900×700 viewport, `zoom: -3`; 188,000 non-background pixels in a 4:3
rectangle (= the 4000×3000 extent), 5,000+ distinct colours (the store is uniform
noise, so a flat grey field of *varying* pixels is exactly right):

![level-3 render of big.ome.zarr](render-big-ome-zarr-level3.png)

The network trace is the important part. zarrita's sharding codec asserts
`store.getRange` and issues **two ranged GETs per cold tile** — a suffix read of the
shard index, then the inner chunk:

```text
GET /rgb/0/c.0.0.0   Range: bytes=36044259-72089289   ->      772 B   (shard index: 3x4x4 x 16 B + 4 B crc)
GET /rgb/0/c.0.0.0   Range: bytes=0-1048608           -> 1,048,609 B  (one inner chunk, compressed)
```

Against `python -m http.server` **both** of those return the whole 36,045,031 B shard.

```text
cold tile, with Range   :        772 +  1,048,609  =  1,049,381 B
cold tile, without Range: 36,045,031 + 36,045,031  = 72,090,062 B   ->  68.7x
steady state (shard index cached), without Range   : 36,045,031 B   ->  34.4x
```

That supersedes the 11.5× figure recorded above, which compared a *decompressed* 3.0 MB
tile against the shard. The measured compressed inner chunk is 1.0 MB (the store is
random noise, so this is a near-worst case for zstd), and the shard-index read doubles
the cost of a cold tile without Range. `conditional=True` is not a nicety.

### FINDING — Playwright's default browser has no WebGL; Viv e2e tests must ask for the full Chromium

Probed on this node, six flag combinations including `--enable-unsafe-swiftshader`,
`--use-gl=angle --use-angle=swiftshader`, `--no-sandbox`:

```text
pw.chromium.launch(...)                          -> canvas.getContext('webgl2') === null
pw.chromium.launch(channel="chromium", ...)      -> webgl2 ANGLE (SwiftShader Device (Subzero))
```

Playwright's default `chromium` launch uses `chromium_headless_shell`, which ships
**no** GL stack; the full `chrome-linux64` build beside it does (`libEGL.so`,
`libGLESv2.so`, `libvk_swiftshader.so`, `vk_swiftshader_icd.json`). The rendering run
above additionally needed an X display — under bare headless the GPU process still died
with `BindToCurrentSequence failed`, and succeeded under `xvfb-run -a`.

**Every Viv rendering test must launch `channel="chromium"` under `xvfb-run`.** Decode
tests (Q4's shape) need neither. Without this, a deck.gl test on a headless runner
reports `Failed to create WebGL context` and zero painted pixels — a red that looks like
a rendering bug and is not one. This directly constrains phases 3-5's e2e suite and the
three `gui-checks` gates.

### Console noise worth knowing about before phase 1

zarrita probes the Zarr **v2** metadata names before giving up, so a normal open emits
404s for `.zattrs`, `.zgroup` and `.zarray` beside every `zarr.json`. 17 of them on a
first open of `rgb`; 3 on a warm one. Harmless, but the byte route must answer them
cheaply and the browser console will show them. Not a defect to chase.

### FINDING — Dash loads `viv_viewer.js` BEFORE `viv/viv-bundle.min.js`; task 2.3's façade must not snapshot the global

`_assets/` is walked with `for current, _, files in sorted(os.walk(walk_dir))`, so every
**root-level** asset is appended before any **subdirectory** asset. Measured by rendering
a real Dash index against the committed layout:

```text
/assets/results_viewer.js
/assets/viv_viewer.js                      <- the façade, FIRST
/assets/openseadragon/openseadragon.min.js
/assets/viv/viv-bundle.min.js              <- the bundle, LAST
```

Plan phase 2 task 2.3's façade snippet opens with `const bundle = window.__vivBundle;`
at IIFE top level. At that moment the bundle has not executed, so `bundle` is
`undefined`, and every method fails on a property access rather than on anything
diagnosable. Nothing in the plan catches this — the ordering is a property of Dash's
asset walk, not of either file.

**Recommended fix, one line, keeps the committed artifact path:** resolve the global
lazily inside `ready` (and therefore inside every method, which already awaits it)
instead of snapshotting it at module scope:

```javascript
const ready = (async () => {
  const bundle = window.__vivBundle;   // resolved at await time, not at load time
  if (!bundle) throw new Error("viv: bundle asset did not load");
  bundle.zarr.registry.set("zstd", () => bundle.numcodecs.Zstd);
  return bundle;
})();
```

Each method then does `const bundle = await ready;`. The alternatives — renaming the
artifact so it sorts first at root level, or `assets_ignore` plus an explicit
`external_scripts` order — both fight the plan's pinned path for no gain.

Note the same walk order means the **2.5 MiB bundle loads on every results-viewer page**,
including ones with no Viv surface, exactly as `openseadragon.min.js` already does. Phase 3
should decide whether that is accepted (localhost / SSH tunnel, per the plan's recorded
cost) or whether both get deferred loading.

## §5.2 chunk-size decision — pending

Needs a cold-pan measurement over a real SSH tunnel at 1024² and 512². The governance is
explicit: the backend spec may be amended from 1024² to 512² **only** gated on a
measurement. Not yet taken.

---

# Phase 4 task 4.1 — the colony-view virtualization cap, MEASURED

Run 2026-08-27 on compute node `r32`. Subject: the same fixture store the Viv
e2e suite uses (1200×900 uint8 noise, 3 pyramid levels, written by `save2zarr`),
served over the phase-1 byte route by a live `phenotypic-gui`. Client: the
**committed** bundle plus the shipped façade — the prototype drives
`window.phenotypicViv.setGridViews(...)`, not a hand-rolled deck.gl page, so the
numbers are the code's and not a lookalike's.

```text
renderer  ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)), SwiftShader driver)
canvas    1200 x 800 CSS px, cells sized to FILL it at each count
method    120 -> 70 rAF frames per count, first 10 discarded; the ONE shared
          zoom is nudged every frame so deck.gl must redraw (an idle grid is
          not redrawn at all, and would time nothing)
```

Cells are sized so **every** cell is on screen at each count. An offscreen view
is culled cheaply, so a fixed cell size would have measured 1536 cells by
drawing 200 of them.

| cells | cell px | median ms | p95 ms | fps | ×frame(1) | draw calls |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 977 | 91.2 | 102.0 | 11.0 | 1.00 | 2 |
| 4 | 487 | 79.4 | 89.0 | 12.6 | 0.87 | 8 |
| 16 | 242 | 80.3 | 90.3 | 12.5 | 0.88 | 32 |
| 64 | 120 | 127.8 | 144.2 | 7.8 | 1.40 | 128 |
| **128** | **84** | **163.6** | **180.1** | **6.1** | **1.79** | **256** |
| 256 | 59 | 304.5 | 401.4 | 3.3 | 3.34 | 512 |
| 512 | 41 | 451.3 | 580.4 | 2.2 | 4.95 | 1024 |
| 1024 | 28 | 809.2 | 942.6 | 1.2 | 8.87 | 2048 |
| 1536 | 23 | 1198.2 | 1335.3 | 0.8 | 13.14 | 3072 |

**`RECORDED_CAP = 128`, `RECORDED_FRAME_MS = 163.6`.**

## Why 128, and why the criterion is a ratio

The interactive budget is a judgement — spec §9.1 makes interactivity a
**target**, not a gate — so the reasoning matters more than the number.

**An absolute millisecond budget was not available to choose against.** This node
has no GPU; Chromium falls back to SwiftShader, and a **single** view already
costs 91 ms. No cell count reaches 60 fps, or 30, or even 15 — including a count
of one. A cap picked as "the largest N under 33 ms" would have been `0`, and one
picked as "under 100 ms" would have encoded this node's software rasterizer
rather than anything about the design.

So the cap is chosen on the **ratio** `frame(N) / frame(1)`, which is invariant
under a uniform hardware speedup in a way an absolute threshold is not. The rule:

> **the grid must not cost more than the canvas it draws into** —
> `frame(N) ≤ 2 × frame(1)`.

128 cells sits at 1.79×; 256 jumps to 3.34×. Past 128 the per-view work, not the
canvas, is what the frame is spent on. A later reader with a GPU can re-judge
from `RECORDED_FRAME_MS` without re-running the prototype.

Two supporting observations:

- **The floor is flat to ~16 cells** (79–91 ms; 4 and 16 cells are *faster* than
  1, because at 1 cell the single view fills the whole canvas and every fragment
  is shaded). Below ~16 cells the grid is free.
- **The marginal cost is linear and stable**: ≈ 0.65–0.75 ms per view across
  64 → 1536, exactly the linear degradation spec §6.2 predicted. It is the slope
  the cap is bounding, and the slope is the part a faster GPU shrinks.

## What the measurement ALSO settled — the camera, at every count

The prototype read `__debugViewStates` back at each count. At every one of the
nine, off the live `Viewport`s deck.gl rendered with:

```text
distinct zooms   = 1        at 1, 4, 16, 64, 128, 256, 512, 1024, 1536 cells
distinct targets = N        (1, 4, 16, 64, 128, 256, 512, 1024, 1536)
```

One zoom and N distinct targets, up to 1536 views. That is the whole task-4.2
claim — one shared value, per-view `target` overrides — measured rather than
argued, and it is the pair the e2e test asserts.

## Two things the byte model got wrong, and what they cost

`colony_view_budget.py` is the executable form of the resident-set bound. Two
corrections landed in it while writing it:

1. **An unclamped cell window escapes the level it is inside.** The union of
   chunks a full plate's cells touch came out as **16** against a **12**-chunk
   level-0 ceiling — an impossible resident set. Cause: the first column of an
   arrayed plate sits 31 px from the edge on the reference geometry, so a 64-px
   cell window reaches `cc = -0.75` and contributes chunk index `-1`. The window
   is clamped to the level's extent; the union is then 12, and the bound holds.
2. **The union is the bound, not the sum.** Already the plan's round-2
   correction, and it is what makes the number small: 1536 cells over ONE store
   touch **12** level-0 chunks, not 1536 windows' worth. The quantity that
   actually drives the cache is the number of distinct **stores** in the grid.

Derived cache bound, over the whole pyramid and all four toggleable layers:
**≥ 151 MB per distinct store**. The façade computes it per instance
(`gridTileCacheBytes` in `_assets/viv_viewer.js`) rather than hard-coding it.

## FINDING — the cache bound is a CORRECTNESS requirement, not just a leak guard

The plan frames `maxCacheByteSize` as bounding the leak class `cleanup_clones`
covers in the napari reference. It is that, but reading deck.gl's own source
while deriving it turned up something stronger:

```text
@deck.gl/geo-layers/src/tile-layer/tile-layer.ts   renderLayers()
    return this.state.tileset.tiles.map(...)          <- the CACHE, not the selection
@deck.gl/geo-layers/src/tileset-2d/tileset-2d.ts   update(viewport, ...)
    if (!this._viewport || !viewport.equals(this._viewport)) { ... }
    _resizeCache(): maxCacheSize ?? DEFAULT_CACHE_SCALE * this.selectedTiles.length
@vivjs/layers dist/index.mjs:900-908               MultiscaleImageLayerBase._updateTileset()
    // with no `viewportId` prop, super._updateTileset() runs for EVERY viewport
```

One `MultiscaleImageLayer` instance draws into all N views. Its `Tileset2D`
re-selects tiles for whichever viewport updated last, but `renderLayers` draws
the whole **cache** — so a cell whose tiles were evicted paints nothing. The
default cache is `5 × selectedTiles.length`, sized for one viewport. Sizing it to
hold the union is what makes multi-view render **at all**, not merely render
without leaking.
