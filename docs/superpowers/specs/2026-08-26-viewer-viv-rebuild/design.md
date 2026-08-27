# Results viewer rebuild on Viv

**Date:** 2026-08-26
**Status:** Draft — blocked on the OME-Zarr backend
**Scope:** The Plate and Colony surfaces of the results viewer, their pixel path, and the
JS packaging that path requires.

## Summary

The results viewer stops **producing** image pyramids and starts **consuming** them.

Today three of the GUI's four pixel paths materialize a full-resolution raster and then
build a pyramid that the OME-Zarr store will already contain. With
[Viv](https://github.com/hms-dbmi/viv) the browser reads zarr chunks directly over HTTP
range requests, so the server-side tiling step is **deleted**, not optimized.

| Surface | Today | After |
|---|---|---|
| Plate | `_load_hdf_layer_rgb` reads the whole layer → PNG → `_dzi_tiler` writes DZI into `.viewer_cache/` → OpenSeadragon | Viv/deck.gl reads pyramid levels from the store; server serves bytes |
| Colony tiles | decode the entire `deliverables/overlays/<ds>/<stem>.png`, LRU it, slice 64² crops | level-0 chunk read per crop |
| Builder preview | HDF layer → PNG → `_dzi_tiler` → DZI | cycle 3; see §7 |
| **Browse** | libvips → DZI → `BrowseCache` → OSD | **unchanged** |

Browse is the deliberate survivor: it reads arbitrary source images with no run behind
them, so it has no store to read and keeps the legacy TIFF path as its only path.

## Locked decisions

From the decision walkthrough
(`docs/superpowers/artifacts/2026-08-26-gui-ome-zarr-sync/gui-design-walkthrough.html`,
rev 2). Each is recorded with what it costs, so a later reader is not re-deriving it.

| # | Decision | Chosen | Note |
|---|---|---|---|
| A | Viv into Dash | **Vendored pre-built bundle + imperative façade** | §3 |
| B | Bytes to the browser | **Range-capable static route** | §4 |
| C1 | Codec | **Keep zstd, ship the wasm codec** | §5 |
| C2 | Chunk size | **Keep 1024², unmeasured** | §5.2 — accepted risk, not a verified choice |
| D | Colony grid | **D3 then D1** — re-source the DOM grid, then deck.gl views | §6 |
| E | Builder preview | **Ephemeral store** | §7, cycle 3 |
| — | Curation | **Retained on Colony** | superseded read-only; see removals spec §5 |
| — | Scope | **Three spec→plan cycles** | this is cycle 2 |
| — | Governance | **May amend the backend spec, on evidence** | §5.2 |

## 1. Dependencies on the backend contract

This spec is written against
[2026-08-18-ome-zarr-image-store](../2026-08-18-ome-zarr-image-store/design.md), which at
time of writing is **specification only** — there is no zarr code in `src/`. The clauses
this design actually leans on:

| Clause | What this spec needs from it |
|---|---|
| §1 layout | A named-series collection: `rgb` / `gray` / `detect_mat` siblings plus `labels/objmap` under the primary series |
| §1.1 | `phenotypic.labels.objmap` records the resolved label path — readers **must not** hard-code `rgb/labels/objmap` |
| §1.3 | Levels halve to ≤512 px; the resolved count and downsample methods persist in `phenotypic.pyramid` so the client never infers them |
| §1.4 | `(1,1024,1024)` chunks, `(C,4096,4096)` shards, `zstd`, `"."` chunk-key separator |
| §4.2 | The four mtime/fingerprint traps — a store's `st_mtime_ns` does not change when a nested chunk is rewritten; staleness must key on the root `zarr.json` |

**Nothing in this spec may start before that lands.** The removals spec (cycle 1) is
sequenced first precisely because it has no such dependency.

## 2. Feasibility — verified and unverified

Stated separately so the unverified claims cannot quietly become facts.

| Claim | Status | Evidence |
|---|---|---|
| zarrita.js reads Zarr v3 **with ZEP2 sharding**, in-browser, zero deps | **verified** | zarrita.dev; `manzt/zarrita.js` |
| Viv and vizarr are **MIT**, compatible with Apache-2.0 | **verified** | `hms-dbmi/viv`, `BioNGFF/vizarr` |
| **zstd** needs the `numcodecs.js` wasm codec registered via `zarr.registry.set()` before first read — not in zarrita's built-in registry | **verified, conditional** | numcodecs.js; see §5.1 |
| vizarr resolves *our* `bioformats2raw.layout` series list and label child without patching | **UNVERIFIED** | §2.1 |
| 1024² chunks pan acceptably over an SSH tunnel | **UNVERIFIED, accepted** | §5.2 |

### 2.1 Required spike, before implementation

Build one store with the real writer and open it with an unmodified vizarr/Viv:

1. Does the series list resolve, or does it need patching?
2. Does `labels/objmap` attach to the primary series as a label layer?
3. Does the `"."` chunk-key separator round-trip?
4. Does the wasm zstd codec decode a CLI-written chunk?

Failure on (1) or (2) is not fatal — it moves work from "configure Viv" to "adapt Viv",
which decision A already permits — but it changes the estimate, so it is answered before
the plan is written, not during it.

## 3. Decision A — packaging (`tools/viv-bundle/`)

**There is no `package.json` anywhere in this repo.** Every line of GUI JS is either
hand-written vanilla (`builder.js`, `browse.js`) or a vendored pre-built bundle
(`openseadragon.min.js`, `cytoscape-dagre.min.js`) dropped into a Dash `_assets/` folder.
Viv is React + deck.gl; vizarr is Preact + Vite. Neither drops in as a file.

**Chosen:** build Viv + deck.gl into one IIFE **outside** the repo, commit the artifact
beside `openseadragon.min.js`, and drive it from a hand-written façade.

- The bundle lands at `results_viewer/_assets/viv/viv-bundle.min.js`.
- `results_viewer/_assets/viv_viewer.js` is the façade: `mount`, `setSource`,
  `setViewState`, `setLayerVisibility`, `destroy`. Dash clientside callbacks talk only to
  the façade, never to Viv directly.
- **The build recipe is committed** at `tools/viv-bundle/` — a pinned lockfile, a build
  script, and a recorded version string the GUI logs at startup. With no npm in CI,
  nothing else will tell you the bundle is stale; this is the only thing standing between
  a vendored artifact and rot.
- `NOTICE` gains Viv and vizarr entries; `licenses/viv-MIT.txt` and
  `licenses/vizarr-MIT.txt` are added, matching the existing SAM2 / micro-sam pattern.
- Per the **`porting-a-reference-algorithm`** skill, the upstream sources we adapt are
  vendored **read-only** under `docs/superpowers/specs/2026-08-26-viewer-viv-rebuild/refs/`
  so every `file:line` citation resolves. They are never linted, formatted, or fixed.

**Costs accepted:** bundle provenance lives outside the repo; upgrading Viv is a manual
ceremony; the bundle is ~1 MB-class, which is acceptable because the only deployment is
localhost or an SSH tunnel.

## 4. Decision B — the byte route

A `/zarr/<dataset>/<stem>.ome.zarr/<path...>` Flask route on the results viewer's
blueprint, using `send_file(..., conditional=True)`.

- `conditional=True` is what provides HTTP Range, which **sharding requires**: a sharded
  read is a shard-index fetch followed by a byte-range fetch.
- The server does no decode. Per-request memory is a sendfile buffer.
- The path guard is the existing `is_safe_path_component` from `gui/_shared/tiles.py`,
  applied **per path segment**, not once over the whole URL — the traversal surface is
  wider than today's route because the tail is arbitrary depth inside a store.
- `_dzi_tiler` is removed from this path entirely. No `.viewer_cache/` directory, no
  tiling pass, no full raster ever resident.

### 4.1 The staleness traps

Backend §4.2 records that a store directory's `st_mtime_ns` does **not** change when a
nested chunk is rewritten. Every staleness check moves to the root `zarr.json`, which the
promote protocol writes last:

| Site | Fix |
|---|---|
| `_tile_routes.py:471` | `file_fingerprint()` raises `IsADirectoryError` on a store — use `paths_fingerprint()` |
| `_tile_routes.py:469, :477` | `stat().st_mtime_ns` compare + `os.utime` retarget to the root `zarr.json` |
| `_shared/tiles.py:518` | mtime-keyed crop path |
| `builder/_preview_tiles.py:76` | same compare (cycle 3) |

The production tile route keys on a **content fingerprint**; only the crop path uses
mtime. Both need fixing, for different reasons.

## 5. Decision C — codec and chunk shape

### 5.1 zstd stays; the wasm codec ships

`numcodecs.js`'s zstd is registered in the bundle's startup path **before any store is
opened**. This is a hard ordering rule — register late and every read fails — so it is
pinned by a test that opens a **CLI-written** store in a browser, not merely one that
asserts the codec registered.

The backend spec is untouched by this decision, and zstd remains the better stored format.

### 5.2 1024² chunks stay, unmeasured — an accepted risk

A deck.gl tile fetch pulls a whole inner chunk: at 1024²×3×u8 that is roughly **3 MB per
tile**. Whether that pans acceptably over an SSH tunnel **has not been measured**.

This is recorded as an accepted risk rather than a verified choice. The fallback is named
so it is cheap if the risk lands: moving to 512² is a **pure §1.4 amendment** to the
backend spec with no GUI rework — the client reads whatever `phenotypic.pyramid` describes.
It would shift the spec's verified file-count table (16 / 40 / 132) and quadruple chunk
count.

Governance for that amendment is granted: this work may file a recorded amendment against
`2026-08-18-ome-zarr-image-store`, **gated on a measurement**. An amendment backed by a
number is how the format stays right; one backed by convenience is how it drifts.

## 6. Decision D — Plate and Colony

Mockups: `docs/superpowers/artifacts/2026-08-26-gui-ome-zarr-sync/mockup/`
(`Main.dc.html` = Plate, `ColonyGrid.dc.html` = Colony), published as a canvas at
`https://claude.ai/code/artifact/7a8c50b6-042f-4948-9452-d6b6e557239f`.

All chrome derives from the existing system — `gui/_design.py` and
`results_viewer/_assets/results_viewer.css` — not from new invention: navy `#003660`
header with the gold `#febc11` JetBrains-Mono pipeline chip, `#0e1620` image stage at
`--radius` 6px, colony cells at `rgba(27,117,188,0.13)` with the navy 2px selected
outline, Comfortaa display/body, JetBrains Mono for all numerics.

### 6.1 Plate

A **full-canvas** deep-zoom surface with floating controls, in the vizarr / avivator
posture rather than the current card-plus-sidebar one.

- Controls float **over** the canvas, so the image gets the full frame instead of losing
  ~300 px to a sidebar.
- The Layers panel lists the store's real series — `rgb`, `gray`, `detect_mat` — plus
  `objmap` tagged as a **label image**, each with visibility, opacity, and a swatch.
  It reads the series list from the store, never a hard-coded set (backend §1.1).
- A navigator inset carries over OpenSeadragon's one genuinely missed affordance.
- A **pyramid readout** names the level actually being served
  (`level 1 of 4 · 2048×1536 · zstd · 1024² chunks`). This is instrumentation, not
  decoration: it is the pyramid the old DZI path rebuilt from scratch every time, and it
  is the fastest way to diagnose a level-selection bug.
- Image stepping (`‹ dataset / stem ›`) and the object/grid summary sit top-left.

### 6.2 Colony — a port of `gui/_smart_grid/`

The napari implementation patches `viewer.grid` so only **visible** layers get cells, then
`create_overlay_clones` duplicates every Labels/Points/Shapes visual into **each**
viewbox — every cell shows a different base image under the same annotation, sharing one
camera.

**Target (D1):** one `OrthographicView` per colony, each centred on its centroid, with the
Viv layer stack rendering into all of them. Zoom edits **one shared `viewState`** applied
to every view — the shared camera is a value, not a sync protocol. The "Shared camera"
lock is a visible affordance, not hidden behaviour, so the eventual unlock-one-cell mode
has somewhere to live. `create_overlay_clones`' GPU-resource cleanup dance
(`cleanup_clones`) has no deck.gl equivalent and is not ported.

**Ship first (D3):** keep today's `build_tile_grid` chrome and change only the crop
route — from overlay-PNG slicing to a level-0 chunk read. This is where the staging pays:
D3 and D1 differ in the *rendering* layer, not the data path, and both need the same zarr
crop reader underneath, so nothing is wasted and the risky half is de-risked by the safe
half.

Under either, **curation is retained**: the radial's six wedges are the real
`ERROR_CATEGORY_COLORS` map (`oversegmented` `undersegmented` `merged` `background_noise`
`debris` `other`, each in its fixed Okabe-Ito slot), with the restore centre node and the
custom-category strip, matching `_shared/_radial.py`'s anatomy. Bulk-mark still writes
`deliverables/errors/<category>.parquet`.

**Cap:** deck.gl re-renders every view each frame, so D1 needs a virtualization cap on
cell count. The number is not set here — it is measured during D1, and until then D3 has
no such limit.

## 7. Decision E — builder preview (cycle 3)

The builder previews a pipeline against an **input image**: no run, no `.ome.zarr`, no
pyramid. "Optimize preview with zarr" therefore needs a source of zarr first.

**Chosen:** each preview node writes its layer to a scratch `.ome.zarr` under the existing
builder-tiles sandbox, reusing the CLI writer, so the viewer reads it exactly like a real
run — one reader, one code path, and the writer gets exercised constantly during
development. Costs: a write the preview does not have today, a scratch dir to
garbage-collect, and pyramiding that is mostly wasted at preview resolutions.

This is **cycle 3** and out of scope for this spec's plan.

## 8. Testing

- **Spike gate (§2.1)** before implementation; its findings amend this spec.
- **Codec ordering**: open a CLI-written store in a real browser. Not "the codec
  registered" — the actual read.
- **Level selection**: assert the level chosen for a target pixel size matches
  `phenotypic.pyramid`'s recorded ladder, including the `ceil` boundary that backend §1.3
  records as having failed once already.
- **Staleness**: a rewritten nested chunk must invalidate; this is the §4.1 trap and the
  test must fail if the check is keyed on the store directory.
- **Curation regression**: the existing colony curation tests pass **unmodified**.
- **Label path**: a store whose primary series is `gray` (no `rgb`) must resolve its
  objmap through `phenotypic.labels.objmap`, proving nothing hard-codes `rgb/labels/objmap`.

Per **`run-phenotypic-test`**: the full `tests/unit` suite is a ~65-minute Slurm job with
`QT_QPA_PLATFORM=offscreen` mandatory.

## 9. Non-goals

- Browse stays on libvips → DZI → `BrowseCache` → OSD. It is not migrated.
- The unmounted Heatmap / Error / QC tabs are not revived here.
- No new measurement, analysis, or export capability.
- `--mode process --layer {...}` output formats are unchanged (backend §4.3).

## 10. Open questions

1. **Series resolution** (§2.1) — the spike answers it.
2. **D1's virtualization cap** — measured during D1.
3. **Bundle staleness signalling** — the recipe is committed and the version logged, but
   nothing *fails* when the bundle drifts from the lockfile. A CI check that rebuilds and
   compares hashes would need npm in CI, which decision A exists to avoid. Currently
   unresolved; the version string is a mitigation, not an answer.
