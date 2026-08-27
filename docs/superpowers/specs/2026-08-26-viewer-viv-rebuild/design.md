# Results viewer rebuild on Viv

**Date:** 2026-08-26
**Status:** Draft — refined through two reviewer rounds; the OME-Zarr backend has **landed**
(see §1), so this is no longer blocked on it
**Scope:** The Plate and Colony surfaces of the results viewer, the builder node preview
(§7, folded in from a retired cycle 3), their pixel path, and the JS packaging that path
requires.

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
| Builder preview | per-node `.ome.zarr` → PNG → `_dzi_tiler` → DZI | Viv over `/preview-zarr/…`; see §7 |
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
| E | Builder preview | **Ephemeral store — landed; render swap in scope** | §7 |
| — | Curation | **Retained on Colony** | superseded read-only; see removals spec §5 |
| — | Scope | **Two spec→plan cycles** | cycle 3 folded in as phase 6, 2026-08-26 |
| — | Governance | **May amend the backend spec, on evidence** | §5.2 |

## 1. Dependencies on the backend contract

> **Amended 2026-08-26.** This section originally read: "…which at time of writing is
> **specification only** — there is no zarr code in `src/`." **That is no longer true.**
> The store branch landed (248 files, including `sdk_/ngff_.py` at 1,506 lines), and this
> spec's branch is stacked on it. Roughly a third of §4.1 and §6.2 below is **already
> implemented**: the four staleness traps, the pyramid-aware server reads, and §6.2's "ship
> first (D3)" crop re-sourcing. §7's decision E is implemented too. The plan's
> [`DRIFT.md`](../../plans/2026-08-26-viewer-viv-rebuild/DRIFT.md) enumerates each with its
> evidence and is required reading before this spec.

This spec is written against
[2026-08-18-ome-zarr-image-store](../2026-08-18-ome-zarr-image-store/design.md). The clauses
this design actually leans on:

| Clause | What this spec needs from it |
|---|---|
| §1 layout | A named-series collection: `rgb` / `gray` / `detect_mat` siblings plus `labels/objmap` under the primary series |
| §1.1 | `phenotypic.labels.objmap` records the resolved label path — readers **must not** hard-code `rgb/labels/objmap` |
| §1.3 | Levels halve to ≤512 px; the resolved count and downsample methods persist in `phenotypic.pyramid` so the client never infers them |
| §1.4 | `(1,1024,1024)` chunks, `(C,4096,4096)` shards, `zstd`, `"."` chunk-key separator |
| §4.2 | The four mtime/fingerprint traps — a store's `st_mtime_ns` does not change when a nested chunk is rewritten; staleness must key on the root `zarr.json` |

**That backend has landed.** The removals spec (cycle 1) remains sequenced first, but for
scheduling rather than dependency reasons — it was always the only one that could execute
against an unlanded backend.

### 1.1 Stage 2 is read-only — the mid-run objmap benefit does not exist

> **Amended 2026-08-26.** Backend §3.4 states that Stage 2 "opens the promoted store and
> overwrites `labels/objmap` in place", and §6.2 below inherits that to claim "the GUI can
> render a real objmap mid-run".
>
> **The landed engine inverted this.** Stage 2 reads the store **read-only** and never
> writes into it; its result is a Stage-2 signal under `.phenotypic/progress/` — the
> retained raw detector output `stage2_raw/<ds>/<stem>.npy` plus a consumable token
> `stage2_done/<ds>/<stem>.json`. Stage 3 replays the raw array, measures, re-promotes the
> store, and consumes the token.
>
> **Consequence:** between Stage 1 and Stage 3 the in-store `labels/objmap` is **zeros**.
> The mid-run-objmap benefit is not available, and the Layers panel must not present a
> zeros objmap as a fault — an empty segmentation is the correct rendering of a correct
> store. A store mid-run additionally has no `tables/` group, and the absence of
> `attributes.phenotypic.tables.measurements` is the reliable discriminator between "not
> yet measured" and "measured, found nothing" (`_image_io_handler.py:1143-1167`).
>
> Backend spec `2026-08-18` §3.4 carries the same stale claim. It belongs to the parent
> branch (`worktree-ome-zarr-image-store`) and is **not** edited from here; this note is the
> record that it needs the same correction.

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
  tiling pass, no full raster ever resident. **The module itself survives** — it retains
  four consumers in Browse and the builder point picker (§9).

### 4.0 The tail is not unrestricted

> **Amended 2026-08-26.** This section originally sketched the route as serving an
> unrestricted path tail. Since it was written, per-object measurements moved **inside**
> each store at `tables/measurements/table.parquet`, and forward runs write no external
> per-image parquet — a group that appears in no layout diagram in either spec. An
> unrestricted tail therefore serves the measurement table to any browser that asks.

The route serves only the groups a pixel client needs. Two properties are normative:

1. **The restriction is enforced on the *resolved* path, not the URL segments.** Checking
   the unresolved first segment leaves a symlink inside a readable root
   (`<store>/rgb/x -> ../tables/measurements/table.parquet`) passing both the head check and
   containment, and the file is served. Resolve first, then test
   `resolved.relative_to(store).parts[0]`.
2. **The readable set is derived per store, not hard-coded.** `attributes.phenotypic.series`
   legitimately contains `original` when the image carries one
   (`_image_io_handler.py:1012-1014`), so a fixed `{rgb, gray, detect_mat}` set makes the
   Layers panel list a series the route 404s. This is the same hard-coding §1's label-path
   rule forbids, one layer down: read the readable set from `series` + `labels`.

**What remains exposed, stated so the decision is made on the facts:** the root `zarr.json`
is mandatory (the client bootstraps from it) and carries
`attributes.phenotypic.metadata` — the `protected`, `public` and `imported` sections plus
`work_id` (`sdk_/ngff_.py:559-583`). `OME/METADATA.ome.xml` carries the same `Metadata_*`
sections. The narrowing keeps the per-object measurement table off the wire; it does not
make the route metadata-free.

### 4.1 The staleness traps

Backend §4.2 records that a store directory's `st_mtime_ns` does **not** change when a
nested chunk is rewritten. Every staleness check moves to the root `zarr.json`, which the
promote protocol writes last:

| Site | Fix |
|---|---|
| `_tile_routes.py:471` | `file_fingerprint()` raises `IsADirectoryError` on a store — use `paths_fingerprint()` |
| `_tile_routes.py:469, :477` | `stat().st_mtime_ns` compare + `os.utime` retarget to the root `zarr.json` |
| `_shared/tiles.py:518` | mtime-keyed crop path |
| `builder/_preview_tiles.py:78-87` | already fixed on the store branch |

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

## 7. Decision E — builder preview (in scope; folded in 2026-08-26)

> **Amended 2026-08-26.** This section originally deferred the builder preview to a separate
> "cycle 3" with its own spec and plan. That split was withdrawn: it bought no parallelism
> (the work is blocked on §§3-4 and reuses all of them), it forced §6.1's
> `build_source_spec` to be written with a signature the deferred plan already knew was
> wrong and refactored two phases later, it parked this spec's own open question about
> asset mounting inside the deferred document, and it ran the whole FEATURES / WORKFLOWS /
> `gui/CLAUDE.md` / capture-script gate pass twice. The builder preview is now **phase 6 of
> this spec's plan**.

**Decision E as originally chosen has already landed.** Each preview node writes its layer
to a scratch `.ome.zarr` under the builder-tiles sandbox, reusing the CLI writer:
`_preview_cache.py:48` (`BASE_STORE_NAME = "base_00.ome.zarr"`), `:255`
(`f"{i:02d}_{op_key}.ome.zarr"`), read back at `_preview_tiles.py:52-65` via
`Image.load_layer_zarr`, with freshness keyed on the root `zarr.json` (`:78-87`).

**What remains is the client-side render swap** — the same DZI → Viv change §6.1 makes for
Plate, applied to the preview pane:

- **One bundle, two mounts.** The builder serves the *same* vendored artifact as the results
  viewer. Committing a second ~1 MB copy would put two artifacts under one build recipe,
  drifting independently with no npm in CI to catch it.
- **A second byte route,** `/preview-zarr/<session_id>/<scope_hash>/<block_id>/<path...>`.
  It cannot reuse §4's route, which resolves through `OutputRoot.store_path`; preview stores
  live under the builder sandbox and have no `OutputRoot`. **The resolver is shared; the
  routes are not** — one `resolve_within_root(root, tail, *, allowed_roots)` in
  `gui/_shared/`, with each route keeping its own resolution and guard regime.
- **Session scoping is a capability URL, not an authentication check.** `_validate`
  (`_preview_tiles.py:107-116`) validates the *shape* of `session_id`, `scope_hash` and
  `block_id`; nothing binds the request to a session. Isolation rests on `session_id` being
  `uuid.uuid4().hex` — 122 bits, unguessable — carried **in the URL path**, where it reaches
  access logs, the OOD reverse proxy's logs, browser history and `Referer`. **This is an
  accepted risk, recorded rather than mitigated** (user ruling, 2026-08-26): the entropy is
  adequate and the exposure matches the existing `/preview-tiles/` route, so changing it
  would diverge the two routes for no behaviour change. The id is a secret; treat it as one.
- **No scratch garbage-collection work.** `init_cache()` already calls `wipe_cache()` on the
  whole cache root at startup and registers an `atexit` wipe (`_preview_cache.py:61-68`),
  and `wipe_scope` reclaims on fingerprint change. That **is** the stated policy. An earlier
  draft specified a measured retention cap with oldest-first eviction and a startup sweep;
  withdrawn 2026-08-26 as machinery built over wipes that already run.
- **The point picker stays on DZI.** `builder/_point_picker.py:417` picks points on a
  *source image*, before any pipeline node has run, so there is no store to read. Migrating
  it would mean writing a store just to render a picker.

**Not in scope:** what the preview computes (`_run_operations`, the scope/fingerprint model,
the manifest shape), and preview pyramiding — preview stores stay single-level, which is
right for a preview pane.

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
- **No authentication is added, and none is assumed to exist.** The GUI has none today: the
  default bind is loopback (`gui/_config.py:221`), but the documented Open OnDemand recipe
  is `--host 0.0.0.0` on a compute node (`docs/source/how_to/pages/gui_hub.md:116, :124`).
  On that path both new byte routes are reachable by anything that can reach the node's
  port, on a multi-user cluster with shared `/rhome` and `/bigdata`. This is **pre-existing**
  — the DZI tile and crop routes already serve pixels the same way — and is recorded as an
  assumption rather than introduced by this work: *loopback, or a proxy that authenticates,
  is assumed.*
- **`_dzi_tiler` is not deleted, and neither is `_tile_routes.py` as a module.** The tiler
  keeps four consumers (`browse/_app.py:40`, `browse/_preparation.py:711`,
  `browse/_preparation_routes.py:95`, `builder/_point_picker.py:417`), and
  `builder/_preview_tiles.py:31` imports `_TILE_NAME_RE` and `_json_error` from
  `_tile_routes` — so deleting either module breaks a different sub-app from the one being
  edited. Only the Plate DZI *routes and calls* come off.

### 9.1 Non-functional requirements

**Correctness outranks performance here, and the ordering is deliberate.** This rebuild is
motivated by performance — it exists to stop materializing a full-resolution raster to
build a pyramid the store already contains — but that motivation does **not** license
trading away correctness to reach a speed target.

- **Requirement (binding): correctness.** The pixels rendered, the pyramid level served,
  the label path resolved, and the curation writes must all be right. A faster surface that
  renders the wrong level, resolves a hard-coded label path, or serves a stale chunk is a
  failure, not a trade.
- **Target (non-binding): interactive over an SSH tunnel.** Pan and zoom on Plate should
  stay interactive over a tunnel from a workstation, and Colony should hold an interactive
  frame rate up to its measured cap (§6.2). This is a **target**, not a gate: missing it is
  a recorded finding and an input to the §5.2 chunk-size decision, not a reason to ship
  something incorrect or to add complexity the spec does not otherwise justify.

For the reviewer panel's precedence table: this target sits **above** "performance without
a spec-stated requirement" and **below** correctness, data integrity, and faithfulness to
published references. A simplicity argument that costs measured interactivity is a real
conflict to adjudicate; one that costs only unmeasured speculation is not.

*Settled by the user, 2026-08-26: "Interactive over ssh would be nice but correctness is
most important."*

## 10. Open questions

1. **Series resolution** (§2.1) — the spike answers it.
2. **D1's virtualization cap** — measured during D1. It is the one number in this design
   that lands in shipped code as a behavioural cap, and it is the only surviving
   logic-validation script (user ruling, 2026-08-26).
3. **Bundle staleness signalling** — the recipe is committed and the version logged, but
   nothing *fails* when the bundle drifts from the lockfile. A CI check that rebuilds and
   compares hashes would need npm in CI, which decision A exists to avoid. Currently
   unresolved; the version string is a mitigation, not an answer.
4. **Serving one `_assets/viv/` to both sub-apps** (§7) — folded in from the retired cycle-3
   spec. Whether Dash's `assets_folder` / `assets_url_path` can point the builder at the
   results viewer's copy, or whether a small Flask route is needed, is unverified. It is a
   packaging question, not a design one: either answer satisfies decision A, and it is
   answered while the artifact is being built rather than a phase later.
5. **The promote generation window** — `promote_store` (`sdk_/ngff_.py:1235-1300`) publishes
   by renaming the whole store directory, and the byte routes resolve fresh per request
   holding no handle. A client can therefore combine metadata from promote *N* with chunks
   from *N+1*. Benign for a run-store re-promote (extent unchanged, `labels/objmap` goes
   zeros → real); **not** benign for the builder preview, where re-running a node
   legitimately changes extent. The old path was coherent because of `_store_content_token`
   (`_tile_routes.py:505-527`), which this design deletes. Carrying a generation token in
   the URL is the intended answer — see the plan's phase 1 — but the choice of token and its
   invalidation cost are not settled here.
