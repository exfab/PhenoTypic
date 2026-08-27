# Drift register — spec vs. the landed OME-Zarr store branch

The Viv rebuild spec (dated 2026-08-26) says of the backend, at §1:

> This spec is written against [2026-08-18-ome-zarr-image-store], which at time of writing
> is **specification only** — there is no zarr code in `src/`.

**That is no longer true.** The store branch (`worktree-ome-zarr-image-store`, head
`bf0d01a1`, 248 files) landed, and this plan's branch is stacked on it. A substantial part
of the Viv spec's §4.1 and §6.2 scope is already implemented.

Each row below was verified in this worktree on 2026-08-26 by reading the named file. Rows
D-1 through D-3 **remove work from this plan**. Row D-4 **falsifies a spec claim**. Rows
D-5 and D-6 are constraints the spec does not mention.

---

## D-1 — §4.1's four staleness traps are already fixed

**Spec says:** four sites need repair because a store directory's `st_mtime_ns` does not
change when a nested chunk is rewritten — `_tile_routes.py:471, :469, :477`,
`_shared/tiles.py:518`, `builder/_preview_tiles.py:76`.

**Reality:** all of them already key on the root `zarr.json`.

| Evidence | What it shows |
|---|---|
| `results_viewer/_tile_routes.py:476-479` | A docstring stating the trap verbatim and that staleness is keyed on the root `zarr.json` |
| `_tile_routes.py:496, :503` | Compares and `os.utime`s against `root_stat`, not the store dir |
| `_tile_routes.py:527-528` | `_store_content_token` uses `paths_fingerprint([root_json])` — `file_fingerprint` is explicitly noted as unusable |
| `builder/_preview_tiles.py:78-87` | Same fix, same reasoning, keyed on `ngff_.STORE_ROOT_JSON` |
| `tests/unit/gui/results_viewer/test_tile_cache_invalidation.py` (297 lines) | Landed test coverage |

**Effect on this plan:** phase 1 does **not** re-plan the traps. It inherits them and adds
one regression test (spec §8's "staleness" check) proving the byte route respects the same
key.

---

## D-2 — §6.2's "ship first (D3)" is already landed

**Spec says:** ship D3 first — "keep today's `build_tile_grid` chrome and change only the
crop route — from overlay-PNG slicing to a level-0 chunk read."

**Reality:** `_shared/tiles.py:665` `crop_colony` already prefers the store
(`crop_store_rgb`, `:545`) and falls back to the baked overlay only for a standalone
deliverables bundle that ships overlays but no `results/` stores. `select_pyramid_level`
(`:378`), `_read_store_level` (`:477`) and `_crop_store_layer_window` (`:599`) are all
present, with `tests/unit/gui/shared/test_tiles_zarr.py` (265 lines) covering them.

Note the deliberate design already in place: `StoreUnreadable` is **not** caught in
`crop_colony`, because falling back to the overlay would show plausible pixels while hiding
a run-wide actionable condition. The caller turns it into a `422`.

**Effect on this plan:** phase 4 is **only** the D1 half — deck.gl `OrthographicView`s
sharing one `viewState`. It is genuinely optional, which is why the README marks it the
first thing to cut.

---

## D-3 — server-side pyramid-aware zarr reads already exist

`_load_zarr_layer_rgb` / `_load_zarr_level_rgb` (`_shared/tiles.py:423, :454`) already
select the smallest pyramid level covering a target pixel size, and `_tile_routes.py:31`
already imports and uses them. Spec §4.2's `_load_hdf_layer_rgb` → `_load_zarr_layer_rgb`
rename is done.

**Effect:** what remains is not "teach the server to read zarr" but "stop the server
building a DZI pyramid from what it read, and hand the raw chunks to the browser instead."
That is a smaller, sharper change than the spec's framing implies, and it is phases 1-3.

---

## D-4 — the spec's mid-run objmap claim is FALSE against the landed engine

**Spec §5 of the backend, quoted by Viv spec §6.2's rationale:** Stage 2 "opens the
promoted store and overwrites `labels/objmap` in place", which "buys ... the GUI can render
a real objmap mid-run."

**Reality — the landed engine inverted this.** From the worktree's own `CLAUDE.md`:

> Stage 2 reads that store **read-only** and never writes into it; its result is a
> **Stage-2 signal** under `.phenotypic/progress/`: the retained **raw** detector output
> `stage2_raw/<ds>/<stem>.npy` plus a consumable **token** `stage2_done/<ds>/<stem>.json`.
> Stage 3 replays the raw array, measures, re-promotes the store, and consumes the token.

So between Stage 1 and Stage 3 the store's `labels/objmap` holds **zeros**, not detector
output. The mid-run-objmap benefit does not exist.

**Effect on this plan:** the Plate Layers panel (phase 3) must not promise a live objmap
during a staged run. An `objmap` that is all-zeros mid-run is the **correct** rendering of
a correct store, and the pyramid readout / layer list must not present it as an error.
Phase 3 task 3.4 covers this explicitly.

**This warrants a spec amendment**, and it is one of the two spec-change items the refinery
should gate to the user (the other is D-6).

---

## D-5 — `_dzi_tiler` cannot be deleted, only unhooked from the Plate path

Spec §4 says "`_dzi_tiler` is removed from this path entirely." Correct as written — but an
executor reading it as "delete the module" breaks five live consumers:

```text
browse/_app.py:40                  DZI_BACKEND_INFO
browse/_preparation.py:711         tile()
browse/_preparation_routes.py:95   DZI_BACKEND_INFO
builder/_point_picker.py:417       tile()
builder/_preview_tiles.py:144      tile()
```

Browse keeps libvips → DZI → `BrowseCache` → OSD as its **only** path (spec §9), so the
module stays. Only `_tile_routes.py:34, :458, :551` come off.

---

## D-6 — measurements now live inside the store

The landed layout puts authoritative per-object measurements at
`tables/measurements/table.parquet` **inside each `.ome.zarr`**, and forward runs no longer
write external per-image parquets. Neither the Viv spec nor the backend spec's §1 layout
diagram shows a `tables/` group.

**Effect on this plan:** the byte route (phase 1) serves an arbitrary-depth tail inside the
store, so it would serve `tables/measurements/table.parquet` to any browser that asks.
Phase 1 task 1.3 restricts the route to the pixel groups the client legitimately needs
rather than exposing the whole store. **This is a security-relevant divergence from the
spec's route sketch and needs the user's sign-off**, since it constrains a route the spec
describes as unrestricted.

---

## Summary of plan-scope changes

| Spec section | Status | This plan |
|---|---|---|
| §4.1 staleness traps | **done** | inherit; add one regression test |
| §4.2 `_load_zarr_layer_rgb` | **done** | inherit |
| §6.2 Colony D3 | **done** | phase 4 is D1 only, and is optional |
| §4 byte route | to do | phase 1 |
| §3 bundle + façade | to do | phase 2 |
| §6.1 Plate surface | to do | phase 3 |
| §6.2 Colony D1 | to do | phase 4 |
| §5.2 chunk measurement | to do | phase 0 |
| §2.1 spike | to do | phase 0 |
